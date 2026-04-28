"""
图片资产存储模块 - PostgreSQL 持久化的图片元数据与占位符映射

主要功能：
1. 保存图片与 chunk 的映射关系
2. 根据 chunk_id 查询图片
3. 根据 filename 查询所有图片
4. 按 task_id 清理图片记录

数据库表：chunk_images
- id: BIGSERIAL 主键
- task_id: 上传任务 ID
- filename: 来源文件名
- chunk_id: 所属 chunk ID
- page_number: 图片所在页码
- image_token: 占位符 token（8位十六进制）
- placeholder: chunk 中的占位符字符串
- image_path: 图片文件路径或对象存储 key
- mime_type: 图片 MIME 类型
- width: 图片宽度
- height: 图片高度
- sha256: 图片哈希（用于去重）
- record_type: 固定为 'image'
- metadata: 扩展字段（JSONB）
- created_at: 创建时间

约束与索引：
- UNIQUE(filename, chunk_id, image_token)
- idx_chunk_images_filename
- idx_chunk_images_chunk_id
- idx_chunk_images_task_id
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2.extras import Json

from backend.db.postgres import postgres_client


class ChunkImageStorage:
    """
    图片资产存储服务。

    负责：
    - 图片元数据的持久化
    - 占位符与真实图片的映射管理
    - 图片记录的查询和删除
    """

    def __init__(self):
        self._ensure_schema()

    def save_many(self, records: list[dict]) -> int:
        """
        批量保存图片记录。

        Args:
            records: 图片记录列表，每条记录包含：
                - task_id: 任务 ID
                - filename: 文件名
                - chunk_id: 所属 chunk ID
                - page_number: 页码
                - image_token: 占位符 token
                - placeholder: 占位符字符串
                - image_path: 图片路径
                - mime_type: MIME 类型
                - width: 宽度（可选）
                - height: 高度（可选）
                - sha256: 图片哈希（可选）
                - metadata: 扩展数据（可选）
                - image_bytes: 图片字节数据（用于计算 sha256）

        Returns:
            成功插入的记录数
        """
        if not records:
            return 0

        now = self._now()
        saved = 0

        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                for record in records:
                    # 计算 sha256（如果提供了 image_bytes）
                    sha256 = record.get("sha256")
                    if not sha256 and record.get("image_bytes"):
                        sha256 = hashlib.sha256(record["image_bytes"]).hexdigest()

                    metadata = record.get("metadata") or {}
                    image_path = record.get("image_path") or ""

                    cur.execute(
                        """
                        INSERT INTO chunk_images (
                            task_id, filename, chunk_id, page_number,
                            image_token, placeholder, image_path, mime_type,
                            width, height, sha256, record_type, metadata, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (filename, chunk_id, image_token) DO UPDATE SET
                            image_path = EXCLUDED.image_path,
                            mime_type = EXCLUDED.mime_type,
                            width = EXCLUDED.width,
                            height = EXCLUDED.height,
                            sha256 = EXCLUDED.sha256,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            record.get("task_id"),
                            record.get("filename"),
                            record.get("chunk_id"),
                            record.get("page_number", 0),
                            record.get("image_token"),
                            record.get("placeholder"),
                            image_path,
                            record.get("mime_type", "image/png"),
                            record.get("width"),
                            record.get("height"),
                            sha256,
                            "image",
                            Json(metadata),
                            now,
                        ),
                    )
                    saved += cur.rowcount

        return saved

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """
        根据 chunk_id 列表查询图片记录。

        Args:
            chunk_ids: chunk ID 列表

        Returns:
            图片记录列表
        """
        if not chunk_ids:
            return []

        placeholders = ",".join(["%s"] * len(chunk_ids))
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, task_id, filename, chunk_id, page_number,
                           image_token, placeholder, image_path, mime_type,
                           width, height, sha256, record_type, metadata, created_at
                    FROM chunk_images
                    WHERE chunk_id IN ({placeholders})
                    ORDER BY page_number, image_token
                    """,
                    chunk_ids,
                )
                rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_by_filename(self, filename: str) -> list[dict[str, Any]]:
        """
        根据文件名查询所有图片记录。

        Args:
            filename: 文件名

        Returns:
            图片记录列表
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, task_id, filename, chunk_id, page_number,
                           image_token, placeholder, image_path, mime_type,
                           width, height, sha256, record_type, metadata, created_at
                    FROM chunk_images
                    WHERE filename = %s
                    ORDER BY page_number, image_token
                    """,
                    (filename,),
                )
                rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_by_filename_and_token(self, filename: str, image_token: str) -> Optional[dict[str, Any]]:
        """
        根据文件名和图片 token 查询单条图片记录。

        Args:
            filename: 文件名
            image_token: 图片 token

        Returns:
            图片记录字典，或 None
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, task_id, filename, chunk_id, page_number,
                           image_token, placeholder, image_path, mime_type,
                           width, height, sha256, record_type, metadata, created_at
                    FROM chunk_images
                    WHERE filename = %s AND image_token = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (filename, image_token),
                )
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_task_id(self, task_id: str) -> list[dict[str, Any]]:
        """
        根据任务 ID 查询所有图片记录。

        Args:
            task_id: 任务 ID

        Returns:
            图片记录列表
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, task_id, filename, chunk_id, page_number,
                           image_token, placeholder, image_path, mime_type,
                           width, height, sha256, record_type, metadata, created_at
                    FROM chunk_images
                    WHERE task_id = %s
                    ORDER BY page_number, image_token
                    """,
                    (task_id,),
                )
                rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete_by_filename(self, filename: str) -> int:
        """
        删除指定文件名的所有图片记录。

        Args:
            filename: 文件名

        Returns:
            删除的记录数
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chunk_images
                    WHERE filename = %s
                    """,
                    (filename,),
                )
                return cur.rowcount

    def delete_by_task_id(self, task_id: str) -> int:
        """
        删除指定任务的所有图片记录。

        Args:
            task_id: 任务 ID

        Returns:
            删除的记录数
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chunk_images
                    WHERE task_id = %s
                    """,
                    (task_id,),
                )
                return cur.rowcount

    def _ensure_schema(self) -> None:
        """
        确保数据库表存在，如不存在则创建。
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                # 创建图片资产表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_images (
                        id BIGSERIAL PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        page_number INTEGER NOT NULL DEFAULT 0,
                        image_token TEXT NOT NULL,
                        placeholder TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        mime_type TEXT NOT NULL DEFAULT 'image/png',
                        width INTEGER,
                        height INTEGER,
                        sha256 TEXT,
                        record_type TEXT NOT NULL DEFAULT 'image',
                        metadata JSONB,
                        created_at TIMESTAMPTZ NOT NULL,
                        UNIQUE(filename, chunk_id, image_token)
                    )
                    """
                )
                # 创建索引
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_images_filename
                    ON chunk_images(filename)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_images_chunk_id
                    ON chunk_images(chunk_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_images_task_id
                    ON chunk_images(task_id)
                    """
                )

    @staticmethod
    def _now() -> str:
        """获取当前 UTC 时间（ISO 格式字符串）"""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        """将数据库行转换为字典"""
        return {
            "id": row[0],
            "task_id": row[1],
            "filename": row[2],
            "chunk_id": row[3],
            "page_number": row[4],
            "image_token": row[5],
            "placeholder": row[6],
            "image_path": row[7],
            "mime_type": row[8],
            "width": row[9],
            "height": row[10],
            "sha256": row[11],
            "record_type": row[12],
            "metadata": row[13],
            "created_at": row[14].isoformat() if row[14] else None,
        }


# 全局单例
chunk_image_storage = ChunkImageStorage()