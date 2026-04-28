"""
API 请求和响应的数据模型定义。

使用 Pydantic 定义了所有 API 的输入输出格式，确保：
1. 类型安全
2. 数据验证
3. 自动文档生成（OpenAPI/Swagger）

数据模型分类：
- 请求模型：ChatRequest（聊天请求）
- 响应模型：各类业务响应模型
- 嵌套模型：RetrievedChunk、RagTrace 等用于组合
"""

from pydantic import BaseModel
from typing import Optional, List


# =============================================================================
# 请求模型
# =============================================================================

class ChatRequest(BaseModel):
    """
    聊天接口的请求模型。

    Attributes:
        message: 用户发送的消息内容
        user_id: 用户唯一标识（用于区分不同用户的数据）
        session_id: 会话唯一标识（用于区分同一用户的不同对话）
    """
    message: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = "default_session"


# =============================================================================
# 响应模型
# =============================================================================

class RetrievedChunk(BaseModel):
    """
    检索到的文档分块信息。

    Attributes:
        filename: 来源文件名
        page_number: 页码
        text: 分块文本内容
        score: 检索相关性得分（可选）
        rrf_rank: RRF 融合后的排名（可选）
        rerank_score: 重排序后的得分（可选）
    """
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RagTrace(BaseModel):
    """
    RAG（检索增强生成）流程的追踪信息。

    记录了整个 RAG 流程的关键节点和参数，便于调试和性能分析。

    Attributes:
        tool_used: 是否使用了检索工具
        tool_name: 使用的工具名称
        query: 原始查询
        expanded_query: 扩展后的查询
        step_back_question: 退步问题（step-back 策略用）
        step_back_answer: 退步问题的回答
        expansion_type: 扩展策略类型（step_back/hyde/complex）
        hypothetical_doc: HyDE 生成的假设性文档
        retrieval_stage: 检索阶段（initial/expanded）
        grade_score: 文档相关性评分结果
        grade_route: 评分后的路由决策
        rewrite_needed: 是否需要重写查询
        rewrite_strategy: 重写策略
        rewrite_query: 重写后的查询
        rerank_enabled: 是否启用了重排序
        rerank_applied: 是否实际应用了重排序
        rerank_model: 重排序使用的模型
        rerank_endpoint: 重排序服务地址
        rerank_error: 重排序错误信息
        retrieval_mode: 检索模式（hybrid/dense_fallback/failed）
        candidate_k: 候选文档数量
        leaf_retrieve_level: 叶子分块层级
        auto_merge_enabled: 是否启用自动合并
        auto_merge_applied: 是否实际应用了自动合并
        auto_merge_threshold: 自动合并阈值
        auto_merge_replaced_chunks: 被替换的分块数量
        auto_merge_steps: 合并步骤数
        retrieved_chunks: 检索到的分块列表
        initial_retrieved_chunks: 初始检索结果
        expanded_retrieved_chunks: 扩展检索后的结果
    """
    tool_used: bool
    tool_name: str
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None


class ChatResponse(BaseModel):
    """
    聊天接口的响应模型。

    Attributes:
        response: AI 生成的回答内容
        rag_trace: RAG 流程的追踪信息（可选）
    """
    response: str
    rag_trace: Optional[RagTrace] = None


class MessageInfo(BaseModel):
    """
    单条消息的信息。

    Attributes:
        type: 消息类型（human/ai/system）
        content: 消息内容
        timestamp: 消息时间戳（ISO 格式）
        rag_trace: 该消息关联的 RAG 追踪信息（可选）
    """
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(BaseModel):
    """
    获取会话消息列表的响应模型。

    Attributes:
        messages: 消息列表
    """
    messages: List[MessageInfo]


class SessionInfo(BaseModel):
    """
    会话的基本信息。

    Attributes:
        session_id: 会话 ID
        updated_at: 最后更新时间（ISO 格式）
        message_count: 消息数量
    """
    session_id: str
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    """
    获取会话列表的响应模型。

    Attributes:
        sessions: 会话信息列表
    """
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    """
    删除会话的响应模型。

    Attributes:
        session_id: 被删除的会话 ID
        message: 操作结果描述
    """
    session_id: str
    message: str


class DocumentInfo(BaseModel):
    """
    文档的基本信息。

    Attributes:
        filename: 文件名
        file_type: 文件类型（PDF/Word/Excel）
        chunk_count: 分块数量
        uploaded_at: 上传时间（可选）
    """
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """
    获取文档列表的响应模型。

    Attributes:
        documents: 文档信息列表
    """
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    """
    文档上传的响应模型。

    Attributes:
        filename: 原始文件名
        task_id: 任务 ID，可用于查询处理状态
        status: 任务状态
        chunks_processed: 已处理的分块数量
        message: 状态描述信息
    """
    filename: str
    task_id: Optional[str] = None
    status: Optional[str] = None
    chunks_processed: Optional[int] = 0
    message: str


class DocumentUploadTaskInfo(BaseModel):
    """
    文档上传任务的详细信息。

    Attributes:
        task_id: 任务 ID
        filename: 文件名
        status: 任务状态（queued/processing/completed/failed）
        message: 状态消息
        chunks_processed: 已处理的分块数
        error: 错误信息（如有）
        created_at: 创建时间
        updated_at: 最后更新时间
        started_at: 开始处理时间
        finished_at: 完成时间
    """
    task_id: str
    filename: str
    status: str
    message: str
    chunks_processed: int = 0
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class DocumentDeleteResponse(BaseModel):
    """
    文档删除的响应模型。

    Attributes:
        filename: 被删除的文件名
        chunks_deleted: 删除的分块数量
        message: 操作结果描述
    """
    filename: str
    chunks_deleted: int
    message: str