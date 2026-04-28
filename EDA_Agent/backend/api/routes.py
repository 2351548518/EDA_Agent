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
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from urllib.parse import quote

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
from backend.memory.chunk_image_storage import chunk_image_storage
from backend.common.paths import DATA_DIR
from backend.common.upload_jobs import upload_job_registry

# 文档上传存储目录
UPLOAD_DIR = DATA_DIR / "documents"
IMAGE_ASSET_DIR = DATA_DIR / "images"

# 启动时确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_ASSET_DIR, exist_ok=True)


def _ext_from_mime(mime_type: str) -> str:
    """根据 MIME 类型推断图片后缀。"""
    mime = (mime_type or "").lower()
    if "jpeg" in mime or "jpg" in mime:
        return ".jpg"
    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    if "gif" in mime:
        return ".gif"
    if "bmp" in mime:
        return ".bmp"
    return ".bin"

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
    4. 保存图片占位符映射到 PostgreSQL
    5. 更新任务状态

    Args:
        task_id: 上传任务的唯一标识
        file_path: 文件在服务器上的存储路径
        filename: 原始文件名
    """
    # 更新任务状态为"处理中"
    upload_job_registry.update_job(task_id, status="processing", message="正在切片和向量化...")

    try:
        # 1. 加载文档并分片（返回三层分块和图片资产）
        chunks, image_assets = loader.load_document(file_path, filename)
        if not chunks:
            raise RuntimeError("文档处理失败，未能提取内容")

        # 2. 分离父级分块（L1、L2）和叶子分块（L3）
        # L1/L2 存储在 PostgreSQL（ParentChunkStore），用于 Auto-merging
        # L3 是最小单元，存储在 Milvus 用于向量检索
        parent_docs = [doc for doc in chunks if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in chunks if int(doc.get("chunk_level", 0) or 0) == 3]

        if not leaf_docs:
            raise RuntimeError("文档处理失败，未生成可检索叶子分块")

        # 3. 构建图片记录（用于 Milvus 和 PostgreSQL）
        milvus_image_records = []
        if image_assets:
            os.makedirs(IMAGE_ASSET_DIR, exist_ok=True)

            # 建立 chunk_id -> root_chunk_id 的映射
            chunk_to_root = {}
            for chunk in chunks:
                cid = chunk.get("chunk_id", "")
                root = chunk.get("root_chunk_id", "")
                if cid and root:
                    chunk_to_root[cid] = root

            for asset in image_assets:
                chunk_id = asset.get("chunk_id", "")
                token = asset.get("image_token", "")
                mime_type = asset.get("mime_type", "image/png")
                image_bytes = asset.get("image_bytes")

                # 落盘原图，供后续通过 API 读取
                ext = _ext_from_mime(mime_type)
                asset_filename = f"{task_id}_{token}{ext}"
                local_image_path = IMAGE_ASSET_DIR / asset_filename
                if image_bytes:
                    with open(local_image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                # 对外可访问路径（方案1：通过后端接口读取）
                image_url_path = f"/images/{quote(filename, safe='')}/{token}"

                # 构建图片记录
                image_id = f"img_{task_id}_{token}"
                image_record = {
                    "image_id": image_id,
                    "image_token": token,
                    "placeholder": asset.get("placeholder", ""),
                    "chunk_id": chunk_id,
                    "root_chunk_id": chunk_to_root.get(chunk_id, ""),
                    "page_number": asset.get("page_number", 0),
                    "filename": filename,
                    "file_type": "image",
                    "image_path": image_url_path,
                    "image_bytes": image_bytes,
                    "mime_type": mime_type,
                    "width": asset.get("width"),
                    "height": asset.get("height"),
                }
                milvus_image_records.append(image_record)

                # 保存到 PostgreSQL（映射表）
                asset["task_id"] = task_id
                asset["filename"] = filename
                asset["image_id"] = image_id
                asset["root_chunk_id"] = image_record["root_chunk_id"]
                asset["image_path"] = image_url_path
                asset["metadata"] = {
                    **(asset.get("metadata") or {}),
                    "local_path": str(local_image_path),
                }

        # 4. 写入存储
        parent_chunk_store.upsert_documents(parent_docs)  # 父级分块 -> PostgreSQL
        # 叶子分块和图片记录 -> Milvus
        milvus_writer.write_documents_and_images(leaf_docs, milvus_image_records)

        # 5. 保存图片映射到 PostgreSQL
        if image_assets:
            chunk_image_storage.save_many(image_assets)

        # 6. 更新任务状态为"已完成"
        upload_job_registry.update_job(
            task_id,
            status="completed",
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个，图片 {len(milvus_image_records)} 张（已完成异步处理）"
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


@router.get("/images/{filename}/{image_token}")
async def get_image_asset(filename: str, image_token: str):
    """
    根据 filename + image_token 返回图片原始二进制内容。

    这是占位符映射的图片读取入口：
    placeholder -> token -> PostgreSQL(chunk_images) -> local file -> bytes
    """
    record = chunk_image_storage.get_by_filename_and_token(filename, image_token)
    if not record:
        raise HTTPException(status_code=404, detail="图片记录不存在")

    metadata = record.get("metadata") or {}
    local_path = metadata.get("local_path")
    if not local_path:
        raise HTTPException(status_code=404, detail="图片本地路径不存在")

    image_file = Path(local_path)
    if not image_file.exists() or not image_file.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在")

    return FileResponse(
        path=str(image_file),
        media_type=record.get("mime_type") or "application/octet-stream",
        filename=image_file.name,
    )


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
        user_sessions = data.get(user_id, {})
        if session_id not in user_sessions:
            return SessionMessagesResponse(messages=[])

        session_data = user_sessions[session_id]
        messages = []

        # 将存储的消息数据转换为 MessageInfo 对象
        for msg_data in session_data.get("messages", []):
            if not isinstance(msg_data, dict):
                continue
            content = msg_data.get("content", "")
            # content 可能是复杂结构（如多模态），需要转为字符串
            if isinstance(content, (list, dict)):
                import json
                content = json.dumps(content)
            messages.append(MessageInfo(
                type=msg_data.get("type", "human"),
                content=str(content),
                timestamp=msg_data.get("timestamp", ""),
                rag_trace=msg_data.get("rag_trace")
            ))

        return SessionMessagesResponse(messages=messages)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load session messages: {str(e)}")


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
        user_sessions = data.get(user_id, {})
        if not user_sessions:
            return SessionListResponse(sessions=[])

        sessions = []
        for session_id, session_data in user_sessions.items():
            sessions.append(SessionInfo(
                session_id=session_id,
                updated_at=session_data.get("updated_at", ""),
                message_count=len(session_data.get("messages", []))
            ))

        # 按更新时间倒序排列，最新的会话排在前面
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


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
        resp = chat_with_agent(
            request.message,
            request.user_id,
            request.session_id,
            request.image,
        )

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
        - image: 图片 base64 编码字符串（可选）

    SSE 事件格式：
        - {"type": "content", "content": "..."}: 内容片段
        - {"type": "rag_step", "step": {...}}: RAG 检索步骤更新
        - {"type": "trace", "rag_trace": {...}}: RAG 追踪信息
        - {"type": "image_hit", "images": [...]}: 图片检索结果
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
                request.session_id,
                request.image,
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
# 图片检索 API
# =============================================================================

class ImageSearchRequest(BaseModel):
    """图片搜索请求模型"""
    image: str  # base64 编码的图片数据
    query: Optional[str] = ""  # 可选的文本查询
    top_k: Optional[int] = 5


class ImageHit(BaseModel):
    """单张图片检索结果"""
    image_id: str
    image_token: str
    placeholder: str
    chunk_id: str
    root_chunk_id: str
    filename: str
    page_number: int
    mime_type: str
    width: Optional[int]
    height: Optional[int]
    score: float
    image_url: str  # 可访问的图片 URL


class ImageSearchResponse(BaseModel):
    """图片搜索响应"""
    image_hits: List[ImageHit]
    text_contexts: List[Dict[str, Any]]
    meta: Dict[str, Any]


@router.post("/images/search", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest):
    """
    根据图片进行向量检索（以图搜图）。

    请求体：
        - image: 图片的 base64 编码字符串
        - query: 可选的文本查询
        - top_k: 返回结果数量

    响应体：
        - image_hits: 图片检索命中结果列表
        - text_contexts: 关联的文本上下文
        - meta: 检索元信息
    """
    from backend.rag.vector_store.retrieval_service import retrieve_images

    # 将 base64 转换为字节
    import base64
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # 执行图片检索
    result = retrieve_images(image_bytes, top_k=request.top_k)

    image_hits = result.get("image_hits", [])
    text_contexts = result.get("text_contexts", [])

    # 构建图片可访问 URL
    enriched_hits = []
    for hit in image_hits:
        filename = hit.get("filename", "")
        token = hit.get("image_token", "")
        image_url = f"/images/{quote(filename, safe='')}/{token}"
        hit["image_url"] = image_url
        enriched_hits.append(ImageHit(**hit))

    return ImageSearchResponse(
        image_hits=enriched_hits,
        text_contexts=text_contexts,
        meta=result.get("meta", {}),
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

        # 删除图片映射记录
        chunk_image_storage.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")