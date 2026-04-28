"""
父级分块存储模块 - PostgreSQL 存储用于 Auto-merging Retriever

Auto-merging 策略说明：
- L3（叶子分块）存储在 Milvus 用于向量检索
- L1/L2（父级分块）存储在 PostgreSQL
- 当多个 L3 指向同一个父分块时，可以合并为父分块
- 合并后的上下文更大，更有利于生成答案

本模块负责：
1. L1/L2 父级分块的写入和读取
2. 按文件名删除分块
3. 按 chunk_id 批量查询父分块
"""

from __future__ import annotations

from typing import Dict, List

from psycopg2.extras import Json

from backend.db.postgres import postgres_client


class ParentChunkStore:
    """
    基于 PostgreSQL 的父级分块存储。

    用于 Auto-merging Retriever 场景，
    存储 L1 和 L2 分块，提供快速的父子关系查询。
    """

    def __init__(self):
        # 确保表结构存在
        self._ensure_schema()

    @staticmethod
    def _sanitize_text(value: str) -> str:
        """
        清理 PostgreSQL text/jsonb 不接受的空字节。

        PostgreSQL 不允许在 text/json 字段中包含 "\\x00"，
        PDF 解析文本中偶发该字符时会导致写入失败。
        """
        if not isinstance(value, str):
            return value
        return value.replace("\x00", "")

    def _load(self) -> Dict[str, dict]:
        """
        加载所有父级分块（调试用）。

        Returns:
            chunk_id -> payload 的映射字典
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, payload FROM parent_chunks")
                rows = cur.fetchall()
        return {row[0]: row[1] for row in rows if isinstance(row[1], dict)}

    def upsert_documents(self, docs: List[dict]) -> int:
        """
        写入或更新父级分块。

        使用 UPSERT 语义：
        - 如果 chunk_id 不存在，插入新记录
        - 如果 chunk_id 已存在，更新记录

        Args:
            docs: 分块列表

        Returns:
            写入的记录数量
        """
        if not docs:
            return 0

        upserted = 0
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                for doc in docs:
                    chunk_id = (doc.get("chunk_id") or "").strip()
                    if not chunk_id:
                        continue

                    # 构建 payload（包含分块的完整信息）
                    payload = {
                        "text": self._sanitize_text(doc.get("text", "")),
                        "filename": self._sanitize_text(doc.get("filename", "")),
                        "file_type": self._sanitize_text(doc.get("file_type", "")),
                        "file_path": self._sanitize_text(doc.get("file_path", "")),
                        "page_number": doc.get("page_number", 0),
                        "chunk_id": self._sanitize_text(chunk_id),
                        "parent_chunk_id": self._sanitize_text(doc.get("parent_chunk_id", "")),
                        "root_chunk_id": self._sanitize_text(doc.get("root_chunk_id", "")),
                        "chunk_level": int(doc.get("chunk_level", 0) or 0),
                        "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    }

                    # UPSERT 操作
                    cur.execute(
                        """
                        INSERT INTO parent_chunks (chunk_id, filename, payload)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (chunk_id)
                        DO UPDATE SET
                            filename = EXCLUDED.filename,
                            payload = EXCLUDED.payload
                        """,
                        (chunk_id, payload["filename"], Json(payload)),
                    )
                    upserted += 1

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """
        根据 chunk_id 批量查询分块。

        用于 Auto-merging 场景：根据子分块的 parent_chunk_id 查询父分块。

        Args:
            chunk_ids: 分块 ID 列表

        Returns:
            分块 payload 列表
        """
        if not chunk_ids:
            return []

        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                # 使用 ANY 查询批量获取
                cur.execute(
                    """
                    SELECT payload
                    FROM parent_chunks
                    WHERE chunk_id = ANY(%s)
                    """,
                    (chunk_ids,),
                )
                rows = cur.fetchall()
        return [row[0] for row in rows if isinstance(row[0], dict)]

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除父级分块。

        Args:
            filename: 要删除的文件名

        Returns:
            删除的记录数量
        """
        if not filename:
            return 0

        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM parent_chunks WHERE filename = %s", (filename,))
                return cur.rowcount

    def _ensure_schema(self) -> None:
        """
        确保数据库表存在。

        表结构：
        - chunk_id TEXT PRIMARY KEY：分块唯一标识
        - filename TEXT NOT NULL：来源文件名
        - payload JSONB NOT NULL：分块完整信息（JSONB 格式便于存储复杂结构）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                # 创建分块表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS parent_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        payload JSONB NOT NULL
                    )
                    """
                )
                # 创建文件名索引，加速按文件名删除/查询
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_parent_chunks_filename
                    ON parent_chunks(filename)
                    """
                )