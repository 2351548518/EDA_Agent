"""
RAG 工作流入口模块 - 提供统一的 RAG 执行接口

本模块是 RAG 流程的入口点，
封装了 LangGraph 工作图的构建和执行细节，
对外提供简单的 run_rag_graph 函数。

使用方式：
    result = run_rag_graph("如何设计一个时钟树？")
    docs = result["docs"]
    rag_trace = result["rag_trace"]
"""

from backend.rag.workflows.graph_builder import build_rag_graph

# 全局单例：构建一次，多次使用
rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    """
    执行 RAG 工作流。

    初始化状态并调用 LangGraph 工作图进行处理。

    初始状态包含：
    - question: 原始问题
    - query: 查询文本（初始等于 question）
    - context: 格式化后的上下文（初始为空）
    - docs: 检索到的文档列表（初始为空）
    - route: 路由决策（初始为 None）
    - expansion_type: 扩展策略类型
    - expanded_query: 扩展后的查询
    - step_back_question: 退步问题
    - step_back_answer: 退步问题答案
    - hypothetical_doc: HyDE 假设性文档
    - rag_trace: RAG 追踪信息

    Args:
        question: 用户的问题

    Returns:
        dict: 包含 docs（检索结果）和 rag_trace（追踪信息）
    """
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
    })