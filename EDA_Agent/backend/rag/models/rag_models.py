"""
RAG 数据模型模块 - 定义 RAG 工作流中使用的 Pydantic 模型

包含：
1. GradeDocuments: 文档相关性评估结果
2. RewriteStrategy: 查询扩展策略选择结果
3. RAGState: LangGraph 工作流状态定义
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """
    文档相关性评估结果。

    由 Grader Model 输出，判断文档是否与问题相关。
    """
    score: str = Field(
        description="相关性评分：'yes' 表示相关，'no' 表示不相关",
    )


class RewriteStrategy(BaseModel):
    """
    查询扩展策略选择结果。

    由 Router Model 输出，选择最适合当前查询的扩展策略。
    """
    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    """
    RAG 工作流的状态定义。

    使用 TypedDict 定义状态的类型注解，
    LangGraph 使用这些定义来追踪状态。

    状态字段说明：
    - question: 原始用户问题
    - query: 当前使用的查询文本
    - context: 格式化后的上下文（用于 LLM）
    - docs: 检索到的文档列表
    - route: 路由决策（决定下一步）
    - expansion_type: 扩展策略类型
    - expanded_query: 扩展后的查询
    - step_back_question: 退步问题
    - step_back_answer: 退步问题答案
    - hypothetical_doc: HyDE 假设性文档
    - rag_trace: 追踪信息
    """
    question: str
    query: str
    context: str
    docs: list[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]