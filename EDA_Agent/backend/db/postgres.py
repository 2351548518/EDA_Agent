"""
PostgreSQL 数据库连接模块 - 提供数据库连接池管理

主要功能：
1. 管理 psycopg2 连接池
2. 提供上下文管理器用于自动连接获取/释放
3. 支持事务自动提交/回滚

数据库配置（从环境变量读取）：
- PGHOST: 数据库主机
- PGPORT: 数据库端口
- PGDATABASE: 数据库名
- PGUSER: 用户名
- PGPASSWORD: 密码
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

# 检查 psycopg2 是否可用
try:
    from psycopg2 import pool
except ImportError as exc:
    raise RuntimeError(
        "psycopg2 is required for PostgreSQL storage. Install it with: pip install psycopg2-binary"
    ) from exc


class PostgresClient:
    """
    PostgreSQL 连接池管理器。

    使用 ThreadedConnectionPool 支持多线程并发访问。
    """

    def __init__(self):
        # 从环境变量读取配置
        host = os.getenv("PGHOST", "127.0.0.1")
        port = int(os.getenv("PGPORT", "5432"))
        dbname = os.getenv("PGDATABASE", "eda_agent")
        user = os.getenv("PGUSER", "eda_user")
        password = os.getenv("PGPASSWORD", "eda_password")

        # 创建连接池
        self._pool = pool.ThreadedConnectionPool(
            minconn=1,       # 最小连接数
            maxconn=8,       # 最大连接数
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )

    @contextmanager
    def connection(self) -> Iterator:
        """
        获取数据库连接的上下文管理器。

        使用方式：
            with postgres_client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")
                    rows = cur.fetchall()

        自动处理：
        - 正常退出：自动提交事务
        - 异常退出：自动回滚事务
        - 无论如何：最终都会释放连接回连接池
        """
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()  # 正常退出时提交事务
        except Exception:
            conn.rollback()  # 异常时回滚事务
            raise
        finally:
            self._pool.putconn(conn)  # 释放连接回连接池


# 全局单例
postgres_client = PostgresClient()