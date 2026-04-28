"""
API 路由模块 - 定义所有后端 API 端点

本模块定义了项目所有的 HTTP API 接口，包括：
- 会话管理（获取消息列表、列出会话、删除会话）
- 聊天功能（同步/流式）
- 文档管理（上传、列表、删除）

整体架构：
- 使用 FastAPI 的 APIRouter 来组织路由
- 每个路由对应一个处理函数
- 请求/响应数据使用 Pydantic 模型定义在 schemas.py 中
"""

import re
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

# 导入请求/响应模型
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    SessionListResponse,
    SessionInfo,
    SessionMessagesResponse,
    MessageInfo,
    SessionDeleteResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentUploadResponse,
    DocumentUploadTaskInfo,
    DocumentDeleteResponse,
)

# 导入 Agent 服务和 RAG 相关组件
from backend.agent.service import chat_with_agent, chat_with_agent_stream, storage
from backend.rag.vector_store.document_loader import DocumentLoader
from backend.rag.vector_store.parent_chunk_store import ParentChunkStore
from backend.rag.vector_store.milvus_writer import MilvusWriter
from backend.rag.vector_store.milvus_client import MilvusManager
from backend.rag.vector_store.embedding import EmbeddingService
from backend.common.paths import DATA_DIR
from backend.common.upload_jobs import upload_job_registry

# 文档上传存储目录
UPLOAD_DIR = DATA_DIR / "documents"

# 全局单例组件（避免重复初始化，提升性能）
# DocumentLoader: 负责加载和分片文档
loader = DocumentLoader()

# ParentChunkStore: PostgreSQL 存储，用于 Auto-merging 检索时存储父级分块
parent_chunk_store = ParentChunkStore()

# MilvusManager: Milvus 向量数据库客户端，负责向量检索
milvus_manager = MilvusManager()

# EmbeddingService: 文本向量化服务，同时支持密集向量和稀疏向量（BM25）
embedding_service = EmbeddingService()

# MilvusWriter: 文档写入 Milvus 的组件，依赖 EmbeddingService 和 MilvusManager
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

# 创建 APIRouter 实例
router = APIRouter()


def _process_document_upload(task_id: str, file_path: str, filename: str) -> None:
    """
    处理文档上传任务的回调函数。

    该函数在后台线程中执行，负责：
    1. 加载文档并进行三层分块（L1/L2/L3）
    2. 将 L1 和 L2 父级分块写入 PostgreSQL
    3. 将 L3 叶子分块写入 Milvus 向量数据库
    4. 更新任务状态

    Args:
        task_id: 上传任务的唯一标识
        file_path: 文件在服务器上的存储路径
        filename: 原始文件名
    """
    # 更新任务状态为"处理中"
    upload_job_registry.update_job(task_id, status="processing", message="正在切片和向量化...")

    try:
        # 1. 加载文档并分片（返回三层分块）
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise RuntimeError("文档处理失败，未能提取内容")

        # 2. 分离父级分块（L1、L2）和叶子分块（L3）
        # L1/L2 存储在 PostgreSQL（ParentChunkStore），用于 Auto-merging
        # L3 是最小单元，存储在 Milvus 用于向量检索
        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]

        if not leaf_docs:
            raise RuntimeError("文档处理失败，未生成可检索叶子分块")

        # 3. 写入存储
        parent_chunk_store.upsert_documents(parent_docs)  # 父级分块 -> PostgreSQL
        milvus_writer.write_documents(leaf_docs)          # 叶子分块 -> Milvus

        # 4. 更新任务状态为"已完成"
        upload_job_registry.update_job(
            task_id,
            status="completed",
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（已完成异步处理）"
            ),
            chunks_processed=len(leaf_docs),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        # 处理失败，更新任务状态为"失败"
        upload_job_registry.update_job(
            task_id,
            status="failed",
            message=f"文档上传失败: {exc}",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


# 将处理器注册到上传任务队列
upload_job_registry.set_processor(_process_document_upload)


# =============================================================================
# 会话管理 API
# =============================================================================

@router.get("/sessions/{user_id}/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(user_id: str, session_id: str):
    """
    获取指定用户会话的所有消息。

    Args:
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        SessionMessagesResponse: 包含消息列表的响应
    """
    try:
        data = storage._load()
        if user_id not in data or session_id not in data[user_id]:
            return SessionMessagesResponse(messages=[])

        session_data = data[user_id][session_id]
        messages = []

        # 将存储的消息数据转换为 MessageInfo 对象
        for msg_data in session_data.get("messages", []):
            messages.append(MessageInfo(
                type=msg_data["type"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                rag_trace=msg_data.get("rag_trace")  # 包含 RAG 检索的追踪信息
            ))

        return SessionMessagesResponse(messages=messages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{user_id}", response_model=SessionListResponse)
async def list_sessions(user_id: str):
    """
    获取指定用户的所有会话列表。

    Args:
        user_id: 用户 ID

    Returns:
        SessionListResponse: 包含会话列表的响应，按更新时间倒序排列
    """
    try:
        data = storage._load()
        if user_id not in data:
            return SessionListResponse(sessions=[])

        sessions = []
        for session_id, session_data in data[user_id].items():
            sessions.append(SessionInfo(
                session_id=session_id,
                updated_at=session_data.get("updated_at", ""),
                message_count=len(session_data.get("messages", []))
            ))

        # 按更新时间倒序排列，最新的会话排在前面
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{user_id}/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(user_id: str, session_id: str):
    """
    删除指定的用户会话。

    Args:
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        SessionDeleteResponse: 删除操作的结果
    """
    try:
        deleted = storage.delete_session(user_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 聊天 API
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    同步聊天接口 - 与 Agent 对话（非流式返回）。

    Agent 会根据用户消息判断是否需要调用知识库检索工具（search_knowledge_base）。
    如果调用，Agent 会获得 RAG 检索结果，然后基于这些结果生成回答。

    请求体 (ChatRequest):
        - message: 用户消息
        - user_id: 用户 ID（可选，默认 "default_user"）
        - session_id: 会话 ID（可选，默认 "default_session"）

    响应体 (ChatResponse):
        - response: AI 回复内容
        - rag_trace: RAG 检索的追踪信息（可选）

    错误处理：
        - 429: 上游模型服务限流/额度限制
        - 401/403: 认证/授权错误
        - 500: 其他内部错误
    """
    try:
        # 调用 Agent 服务处理消息
        resp = chat_with_agent(request.message, request.user_id, request.session_id)

        # 将响应转换为 ChatResponse 格式
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)

    except Exception as e:
        message = str(e)
        # 从错误信息中提取状态码
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式聊天接口 - 与 Agent 对话（Server-Sent Events 流式返回）。

    与同步接口不同，流式接口会将 AI 生成的每个内容片段实时推送给前端，
    让用户看到逐字生成的效果。同时，RAG 检索的中间步骤也会实时推送。

    请求体 (ChatRequest):
        - message: 用户消息
        - user_id: 用户 ID
        - session_id: 会话 ID

    SSE 事件格式：
        - {"type": "content", "content": "..."}: 内容片段
        - {"type": "rag_step", "step": {...}}: RAG 检索步骤更新
        - {"type": "trace", "rag_trace": {...}}: RAG 追踪信息
        - {"type": "error", "content": "..."}: 错误信息
        - [DONE]: 结束信号

    错误处理：
        - 如果客户端断开连接（AbortController），会取消后台任务
        - 异常信息会通过 SSE 的 error 事件推送
    """
    async def event_generator():
        try:
            # chat_with_agent_stream 已经生成了 SSE 格式的字符串 (data: {...}\n\n)
            # 每次迭代返回一个新的 chunk
            async for chunk in chat_with_agent_stream(
                request.message,
                request.user_id,
                request.session_id
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            # SSE 格式：data: {...}\n\n
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，确保实时推送
        },
    )


# =============================================================================
# 文档管理 API
# =============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    获取已上传的文档列表。

    通过查询 Milvus 向量数据库，统计每个文件名下的分块数量。

    响应体 (DocumentListResponse):
        - documents: 文档信息列表，每个文档包含：
            - filename: 文件名
            - file_type: 文件类型（PDF/Word/Excel）
            - chunk_count: 分块数量
            - uploaded_at: 上传时间（可选）
    """
    try:
        milvus_manager.init_collection()

        # 查询所有文档的分块信息
        results = milvus_manager.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        # 按文件名分组统计
        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并异步进行切片与向量化处理。

    支持的文件类型：PDF、Word（.doc/.docx）、Excel（.xls/.xlsx）

    处理流程：
    1. 验证文件类型
    2. 创建上传目录（如不存在）
    3. 初始化 Milvus 集合（如不存在）
    4. 删除同名文件的旧数据（避免重复）
    5. 保存文件到上传目录
    6. 提交异步处理任务

    请求：
        - file: 上传的文件（multipart/form-data）

    响应 (DocumentUploadResponse):
        - filename: 原始文件名
        - task_id: 任务 ID，可用于查询处理状态
        - status: 任务状态（"queued"）
        - chunks_processed: 已处理的分块数
        - message: 状态信息
    """
    try:
        filename = file.filename
        file_lower = filename.lower()

        # 验证文件类型
        if not (file_lower.endswith(".pdf") or
                file_lower.endswith((".docx", ".doc")) or
                file_lower.endswith((".xlsx", ".xls"))):
            raise HTTPException(
                status_code=400,
                detail="仅支持 PDF、Word 和 Excel 文档"
            )

        # 确保上传目录存在
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 确保 Milvus 集合已初始化
        milvus_manager.init_collection()

        # 生成唯一任务 ID
        task_id = uuid4().hex

        # 存储的文件名添加 task_id 前缀，避免同名文件覆盖
        stored_filename = f"{task_id}_{Path(filename).name}"

        # 删除同名文件的旧数据
        delete_expr = f'filename == "{filename}"'
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        # 保存上传的文件
        file_path = UPLOAD_DIR / stored_filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 提交异步处理任务
        upload_job_registry.submit_job(filename, str(file_path), task_id=task_id)

        return DocumentUploadResponse(
            filename=filename,
            task_id=task_id,
            status="queued",
            chunks_processed=0,
            message=f"{filename} 已上传，后台正在异步切片和向量化处理",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.get("/documents/upload/tasks/{task_id}", response_model=DocumentUploadTaskInfo)
async def get_upload_task(task_id: str):
    """
    查询文档上传任务的处理状态。

    Args:
        task_id: 任务 ID（在上传响应中获得）

    Returns:
        DocumentUploadTaskInfo: 任务的详细信息，包括：
            - task_id: 任务 ID
            - filename: 文件名
            - status: 状态（queued/processing/completed/failed）
            - message: 状态消息
            - chunks_processed: 已处理的分块数
            - error: 错误信息（如有）
            - created_at/updated_at/started_at/finished_at: 时间戳
    """
    task = upload_job_registry.get_job(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return DocumentUploadTaskInfo(**task)


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str):
    """
    删除文档的向量数据（保留本地文件）。

    删除操作会：
    1. 从 Milvus 中删除该文件名的所有向量数据
    2. 从 PostgreSQL 的 parent_chunks 表中删除相关记录

    注意：本地上传的文件不会被删除。

    Args:
        filename: 要删除的文件名

    Returns:
        DocumentDeleteResponse: 删除结果，包含：
            - filename: 文件名
            - chunks_deleted: 删除的分块数量
            - message: 操作结果描述
    """
    try:
        milvus_manager.init_collection()

        # 从 Milvus 删除向量数据
        delete_expr = f'filename == "{filename}"'
        result = milvus_manager.delete(delete_expr)

        # 从 PostgreSQL 删除父级分块数据
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")