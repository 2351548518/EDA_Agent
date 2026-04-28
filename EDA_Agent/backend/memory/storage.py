"""
对话历史存储模块 - 使用 PostgreSQL 持久化用户会话消息

主要功能：
1. 保存和加载用户对话历史
2. 按用户和会话组织数据
3. 支持 RAG trace 信息的附加存储

数据库表结构：
- conversation_sessions: 存储用户会话数据
    - user_id: 用户 ID（主键组成部分）
    - session_id: 会话 ID（主键组成部分）
    - messages: 消息列表（JSONB 格式）
    - metadata: 元数据（JSONB 格式）
    - updated_at: 最后更新时间

架构说明：
- 使用 psycopg2 连接池管理数据库连接
- JSONB 格式存储消息，支持复杂结构
- 每次保存使用 UPSERT 语义，自动创建或更新
"""

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from psycopg2.extras import Json

from backend.db.postgres import postgres_client


class ConversationStorage:
    """
    基于 PostgreSQL 的对话历史存储。

    负责：
    - 用户的会话消息持久化
    - 会话列表管理
    - 消息的序列化和反序列化
    """

    def __init__(self):
        # 确保数据库表存在
        self._ensure_schema()

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict | None = None,
        extra_message_data: list | None = None,
    ):
        """
        保存用户会话消息。

        使用 UPSERT 语义：
        - 如果会话不存在，创建新记录
        - 如果会话已存在，更新消息和元数据

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            messages: LangChain 消息列表
            metadata: 额外的元数据
            extra_message_data: 每条消息的额外数据（如 rag_trace）
        """
        serialized = []
        now = self._now()

        # 序列化消息
        for idx, msg in enumerate(messages):
            record = {
                "type": msg.type,  # human / ai / system
                "content": msg.content,
                "timestamp": now,
            }
            # 附加每条消息的额外数据（如 rag_trace）
            if extra_message_data and idx < len(extra_message_data):
                extra = extra_message_data[idx] or {}
                if "rag_trace" in extra:
                    record["rag_trace"] = extra["rag_trace"]
            serialized.append(record)

        # 保存到数据库
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_sessions (user_id, session_id, messages, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET
                        messages = EXCLUDED.messages,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        user_id,
                        session_id,
                        Json(serialized),
                        Json(metadata or {}),
                        now,
                    ),
                )

    def load(self, user_id: str, session_id: str) -> list:
        """
        加载用户会话消息。

        Args:
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            LangChain 消息列表（HumanMessage / AIMessage / SystemMessage）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT messages
                    FROM conversation_sessions
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (user_id, session_id),
                )
                row = cur.fetchone()

        # 如果没有记录或消息为空，返回空列表
        if not row or not isinstance(row[0], list):
            return []

        # 反序列化消息
        messages = []
        for msg_data in row[0]:
            if not isinstance(msg_data, dict):
                continue
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")

            # 根据消息类型创建相应的 LangChain 消息对象
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))

        return messages

    def list_sessions(self, user_id: str) -> list:
        """
        获取用户的所有会话 ID 列表。

        按最后更新时间倒序排列。

        Args:
            user_id: 用户 ID

        Returns:
            session_id 列表
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id
                    FROM conversation_sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        return [row[0] for row in rows]

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        删除指定的用户会话。

        Args:
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            是否成功删除（True/False）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM conversation_sessions
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (user_id, session_id),
                )
                return cur.rowcount > 0

    def _load(self) -> dict:
        """
        加载所有会话数据（用于管理界面）。

        Returns:
            dict: 按 user_id -> session_id -> session_data 组织的数据
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, session_id, messages, metadata, updated_at
                    FROM conversation_sessions
                    ORDER BY updated_at DESC
                    """
                )
                rows = cur.fetchall()

        data: dict[str, dict[str, Any]] = {}
        for user_id, session_id, messages, metadata, updated_at in rows:
            # 按用户分组
            user_bucket = data.setdefault(user_id, {})
            # 按会话分组
            user_bucket[session_id] = {
                "messages": messages if isinstance(messages, list) else [],
                "metadata": metadata if isinstance(metadata, dict) else {},
                "updated_at": self._to_iso(updated_at) or "",
            }
        return data

    def _ensure_schema(self) -> None:
        """
        确保数据库表存在，如不存在则创建。

        表结构：
        - user_id TEXT NOT NULL（主键的一部分）
        - session_id TEXT NOT NULL（主键的一部分）
        - messages JSONB NOT NULL（消息列表）
        - metadata JSONB NOT NULL（元数据）
        - updated_at TIMESTAMPTZ NOT NULL（更新时间）
        """
        with postgres_client.connection() as conn:
            with conn.cursor() as cur:
                # 创建会话表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        messages JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    )
                    """
                )
                # 创建索引，按用户和更新时间排序
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_updated
                    ON conversation_sessions(user_id, updated_at DESC)
                    """
                )

    @staticmethod
    def _now() -> str:
        """获取当前 UTC 时间（ISO 格式字符串）"""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_iso(value: Any) -> str | None:
        """
        将值转换为 ISO 格式字符串。

        Args:
            value: 要转换的值

        Returns:
            ISO 格式字符串，或 None
        """
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)