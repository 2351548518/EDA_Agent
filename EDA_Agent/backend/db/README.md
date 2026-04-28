# Database Modules

## postgres.py
PostgreSQL 连接管理（`psycopg2.ThreadedConnectionPool`），被以下模块使用：
- `memory/storage.py` — 会话持久化
- `common/upload_jobs.py` — 上传任务队列
- `rag/vector_store/parent_chunk_store.py` — 父级分块存储
