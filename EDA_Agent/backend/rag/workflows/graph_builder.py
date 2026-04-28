"""
LangGraph 工作图构建模块 - 定义 RAG 流程的图结构

本模块使用 LangGraph 构建 RAG 检索工作流。

工作图结构：

    ┌─────────────────┐
    │ retrieve_initial │  初始检索
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  grade_documents │  评估相关性
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────┐   ┌─────────────┐
│  END     │   │rewrite_question│
│(生成答案) │   └──────┬──────┘
└──────────┘          │
                      ▼
             ┌──────────────────┐
             │ retrieve_expanded │  扩展检索
             └────────┬─────────┘
                      │
                      ▼
                    END

流程说明：
1. retrieve_initial: 使用原始查询检索文档
2. grade_documents: 评估检索到的文档与问题的相关性
3. 如果相关 -> 直接生成答案
4. 如果不相关 -> rewrite_question -> retrieve_expanded -> 合并结果

LangGraph 特性：
- StateGraph: 基于状态的图
- add_node: 添加节点
- add_edge: 添加边（确定流程）
- add_conditional_edges: 添加条件边（根据状态路由）
"""

from langgraph.graph import END, StateGraph

from backend.rag.models.rag_models import RAGState
from backend.rag.workflows.nodes import (
    grade_documents_node,
    retrieve_expanded,
    retrieve_initial,
    rewrite_question_node,
)


def build_rag_graph():
    """
    构建 RAG 工作流图。

    Returns:
        编译后的 LangGraph 工作图
    """
    # 创建状态图
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("retrieve_initial", retrieve_initial)      # 初始检索
    graph.add_node("grade_documents", grade_documents_node)    # 相关性评估
    graph.add_node("rewrite_question", rewrite_question_node)  # 查询重写
    graph.add_node("retrieve_expanded", retrieve_expanded)     # 扩展检索

    # 设置入口点
    graph.set_entry_point("retrieve_initial")

    # 添加边
    graph.add_edge("retrieve_initial", "grade_documents")

    # 条件边：根据评估结果决定下一步
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),  # 根据 route 字段决定路由
        {
            "generate_answer": END,           # 相关 -> 结束（生成答案）
            "rewrite_question": "rewrite_question",  # 不相关 -> 重写查询
        },
    )

    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)

    # 编译并返回
    return graph.compile()