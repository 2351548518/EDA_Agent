"""
模型提供者模块 - 集中管理 RAG 流程中使用的 LLM 模型

本模块负责初始化和管理 RAG 流程中的两个 LLM：
1. Grader Model: 用于评估文档相关性
2. Router Model: 用于选择查询扩展策略

使用全局单例模式，避免重复初始化。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# 环境变量配置
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL")

# 全局单例
_grader_model = None
_router_model = None


def get_grader_model():
    """
    获取文档相关性评估模型。

    如果未配置 GRADE_MODEL 或 API_KEY，返回 None。

    Returns:
        初始化好的 chat model，或 None
    """
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,  # 评估使用确定性的回答
            stream_usage=False,
        )
    return _grader_model


def get_router_model():
    """
    获取查询扩展路由模型。

    如果未配置 MODEL 或 API_KEY，返回 None。

    Returns:
        初始化好的 chat model，或 None
    """
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,  # 路由使用确定性的回答
            stream_usage=False,
        )
    return _router_model