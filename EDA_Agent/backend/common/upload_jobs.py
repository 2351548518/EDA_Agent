"""
文档上传任务队列模块 - PostgreSQL 持久化的异步任务队列

主要功能：
1. 管理文档上传后的异步处理任务
2. 后台线程池处理任务
3. 任务状态持久化到 PostgreSQL
4. 服务重启后自动恢复未完成的任务

任务状态流转：
1. queued: 任务已排队，等待处理
2. processing: 正在处理中
3. completed: 处理完成
4. failed: 处理失败

数据库表：upload_jobs
- task_id: 任务唯一标识（主键）
- filename: 文件名
- file_path: 文件路径
- status: 任务状态
- message: 状态消息
- chunks_processed: 已处理的分块数
- error: 错误信息
- created_at/updated_at/started_at/finished_at: 时间戳
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from backend.db.postgres import postgres_client


# 任务处理函数的类型定义
JobProcessor = Callable[[str, str, str], None]


class UploadJobQueue:
    """
    文档上传任务队列。

    使用 PostgreSQL 持久化任务状态，
    后台线程池异步执行任务处理。
    """

    def __init__(self):
        # 条件变量，用于线程间通信
        self._condition = threading.Condition()
        # 任务处理函数
        self._processor: Optional[JobProcessor] = None
        # 后台工作线程
        self._worker_thread: Optional[threading.Thread] = None

        # 确保数据库表存在
        self._ensure_schema()
        # 重置服务重启前的"处理中"任务为"排队"状态
        self._reset_stale_jobs()

    def set_processor(self, processor: JobProcessor) -> None:
        """
        设置任务处理器。

        处理器是一个函数，签名为：
            (task_id: str, file_path: str, filename: str) -> None

        Args:
            processor: 任务处理函数
        """
        with self._condition:
            self._processor = processor
            # 如果工作线程不存在或已停止，启动新的工作线程
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()
            self._condition.notify_all()

    def submit_job(self, filename: str, file_path: str, task_id: Optional[str] = None) -> str:
        """
        提交一个新任务。

        Args:
            filename: 文件名
            file_path: 文件路径
            task_id: 可选的任务 ID，不提供则自动生成

        Returns:
            任务 ID
        """
        task_id = task_id or uuid4().hex
        now = self._now()

        # 保存任务到数据库
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO upload_jobs (
                        task_id, filename, file_path, status, message,
                        chunks_processed, error, created_at, updated_at, started_at, finished_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO NOTHING
                    """,
                    (
                        task_id,
                        filename,
                        file_path,
                        "queued",
                        "任务已排队，正在等待后台处理...",
                        0,
                        None,
                        now,
                        now,
                        None,
                        None,
                    ),
                )
                if cur.rowcount == 0:
                    raise ValueError(f"任务已存在: {task_id}")

        # 通知工作线程有新任务
        with self._condition:
            self._condition.notify_all()

        return task_id

    def get_job(self, task_id: str) -> Optional[dict[str, Any]]:
        """
        获取任务信息。

        Args:
            task_id: 任务 ID

        Returns:
            任务信息字典，或 None（如果不存在）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT task_id, filename, file_path, status, message,
                           chunks_processed, error, created_at, updated_at,
                           started_at, finished_at
                    FROM upload_jobs
                    WHERE task_id = %s
                    """,
                    (task_id,),
                )
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_jobs(self) -> list[dict[str, Any]]:
        """
        获取所有任务列表。

        Returns:
            任务信息字典列表
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT task_id, filename, file_path, status, message,
                           chunks_processed, error, created_at, updated_at,
                           started_at, finished_at
                    FROM upload_jobs
                    ORDER BY created_at DESC
                    """
                )
                rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_job(self, task_id: str, **updates: Any) -> None:
        """
        更新任务状态。

        Args:
            task_id: 任务 ID
            **updates: 要更新的字段
        """
        if not updates:
            return

        # 只允许更新特定字段，防止意外修改
        allowed_fields = {
            "status",
            "message",
            "chunks_processed",
            "error",
            "started_at",
            "finished_at",
            "file_path",
            "filename",
        }
        filtered_updates = {key: value for key, value in updates.items() if key in allowed_fields}
        if not filtered_updates:
            return

        filtered_updates["updated_at"] = self._now()
        columns = ", ".join(f"{key} = %s" for key in filtered_updates)
        values = list(filtered_updates.values()) + [task_id]

        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE upload_jobs SET {columns} WHERE task_id = %s", values)

    def _worker_loop(self) -> None:
        """
        后台工作线程的主循环。

        持续从队列中获取任务并执行。
        使用 FOR UPDATE SKIP LOCKED 避免多线程竞争。
        """
        while True:
            # 获取下一个待处理任务
            job = self._claim_next_job()
            if not job:
                # 没有任务，等待通知或超时
                with self._condition:
                    self._condition.wait(timeout=1.0)
                continue

            processor = self._processor
            if processor is None:
                # 处理器未设置，任务等待
                self.update_job(
                    job["task_id"],
                    status="queued",
                    message="后台处理器尚未初始化，任务稍后重试",
                )
                with self._condition:
                    self._condition.wait(timeout=1.0)
                continue

            try:
                # 更新状态为处理中
                self.update_job(
                    job["task_id"],
                    status="processing",
                    message="正在切片和向量化...",
                    started_at=job.get("started_at") or self._now(),
                )
                # 执行任务处理
                processor(job["task_id"], job["file_path"], job["filename"])
            except Exception as exc:
                # 处理失败，更新状态
                self.update_job(
                    job["task_id"],
                    status="failed",
                    message=f"文档上传失败: {exc}",
                    error=str(exc),
                    finished_at=self._now(),
                )

    def _claim_next_job(self) -> Optional[dict[str, Any]]:
        """
        认领下一个待处理任务。

        使用 FOR UPDATE SKIP LOCKED 锁定并返回任务，
        避免多个工作线程竞争同一个任务。

        Returns:
            任务信息字典，或 None（如果没有待处理任务）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH next_job AS (
                        SELECT task_id
                        FROM upload_jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE upload_jobs u
                    SET status = 'processing',
                        message = '正在切片和向量化...',
                        started_at = COALESCE(u.started_at, %s),
                        updated_at = %s
                    FROM next_job
                    WHERE u.task_id = next_job.task_id
                    RETURNING u.task_id, u.filename, u.file_path, u.status, u.message,
                              u.chunks_processed, u.error, u.created_at, u.updated_at,
                              u.started_at, u.finished_at
                    """,
                    (self._now(), self._now()),
                )
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def _reset_stale_jobs(self) -> None:
        """
        重置服务重启前处于"处理中"状态的任务。

        这些任务可能是服务崩溃前正在处理的，
        重启后需要重新排队处理。
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE upload_jobs
                    SET status = 'queued',
                        message = '任务在服务重启后自动重新排队',
                        updated_at = %s
                    WHERE status = 'processing'
                    """,
                    (self._now(),),
                )

    def _ensure_schema(self) -> None:
        """
        确保数据库表存在。
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upload_jobs (
                        task_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        status TEXT NOT NULL,
                        message TEXT NOT NULL,
                        chunks_processed INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        started_at TIMESTAMPTZ,
                        finished_at TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_upload_jobs_status_created_at
                    ON upload_jobs(status, created_at)
                    """
                )

    @staticmethod
    def _now() -> str:
        """获取当前 UTC 时间（ISO 格式字符串）"""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _dt_to_iso(value: Any) -> Optional[str]:
        """将日期时间转换为 ISO 格式字符串"""
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        """将数据库行转换为字典"""
        return {
            "task_id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "status": row[3],
            "message": row[4],
            "chunks_processed": int(row[5] or 0),
            "error": row[6],
            "created_at": self._dt_to_iso(row[7]),
            "updated_at": self._dt_to_iso(row[8]),
            "started_at": self._dt_to_iso(row[9]),
            "finished_at": self._dt_to_iso(row[10]),
        }


# 全局单例
upload_job_registry = UploadJobQueue()