"""
Agent 工具模块 - 定义 Agent 可使用的工具函数

本模块定义了 LangChain Agent 可调用的工具：

1. get_current_weather: 天气查询工具（调用高德地图 API）
2. search_knowledge_base: 知识库检索工具（调用 RAG 流程）

工具调用控制：
- search_knowledge_base 每轮对话只能调用一次
- 通过全局变量 _KNOWLEDGE_TOOL_CALLS_THIS_TURN 控制
- 调用 reset_tool_call_guards() 重置计数

RAG 步骤实时推送：
- 通过 emit_rag_step 函数将 RAG 检索步骤实时推送给前端
- 使用跨线程安全的方式调度到主事件循环
"""

from typing import Optional
import os
import requests
from dotenv import load_dotenv

# 尝试导入 LangChain 的 tool 装饰器
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

# 高德天气 API 配置
AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# ============================================================================
# 全局状态管理
# ============================================================================

# 最近一次 RAG 检索的上下文（包含 rag_trace 等信息）
_LAST_RAG_CONTEXT = None

# 本轮对话中知识库检索工具的调用次数
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0

# RAG 步骤队列，用于实时推送检索步骤给前端
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming

# RAG 步骤队列所在的事件循环，用于跨线程调度
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    """
    设置最近一次 RAG 检索的上下文。

    Args:
        context: 包含 rag_trace 等信息的字典
    """
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """
    获取最近一次 RAG 检索上下文，并可选地清除。

    Args:
        clear: 是否在读取后清除上下文，默认 True

    Returns:
        RAG 上下文字典，或 None
    """
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """
    重置工具调用计数。

    每轮对话开始时调用，确保工具调用计数从头开始。
    这防止了上一轮对话的调用残留影响本轮对话。
    """
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """
    设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。

    这个函数在流式聊天开始前被调用，用于设置队列接收 RAG 步骤信息。
    捕获的事件循环用于后续的跨线程安全调度。

    Args:
        queue: asyncio.Queue 实例，用于接收 RAG 步骤
    """
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            # 获取当前正在运行的事件循环
            _RAG_STEP_LOOP = asyncio.get_running_loop()
            print(f"[set_rag_step_queue] captured running loop: {_RAG_STEP_LOOP}")
        except RuntimeError:
            # 如果没有运行中的循环，获取或创建一个
            _RAG_STEP_LOOP = asyncio.get_event_loop()
            print(f"[set_rag_step_queue] got event loop: {_RAG_STEP_LOOP}")
    else:
        _RAG_STEP_LOOP = None
        print(f"[set_rag_step_queue] queue cleared")


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """
    向队列发送一个 RAG 检索步骤。

    该函数是跨线程安全的，可以从工具执行的线程中调用，
    将步骤信息调度到主事件循环中，最终推送给前端。

    关键机制：
    - 使用 call_soon_threadsafe 将操作调度到主事件循环
    - 这确保了即使工具在独立线程中运行，也能安全地将数据放入队列

    Args:
        icon: 步骤图标（如 "🔍"）
        label: 步骤标签/标题
        detail: 步骤详细信息
    """
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is None or _RAG_STEP_LOOP is None:
        print(f"[emit_rag_step] QUEUE={_RAG_STEP_QUEUE}, LOOP={_RAG_STEP_LOOP}")
        return

    step = {"icon": icon, "label": label, "detail": detail}
    try:
        if not _RAG_STEP_LOOP.is_closed():
            # 跨线程安全地调度到主事件循环
            _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        else:
            print(f"[emit_rag_step] loop is closed")
    except Exception as e:
        print(f"[emit_rag_step] error: {e}")


# ============================================================================
# 工具定义
# ============================================================================

@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """
    获取指定城市的天气信息。

    Args:
        location: 城市名称（如 "北京"）
        extensions: 天气类型，"base" 返回实时天气，"all" 返回天气预报

    Returns:
        格式化的天气信息字符串
    """
    # 参数验证
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    # 检查 API 配置
    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    # 构造请求参数
    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        # 调用高德天气 API
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 检查 API 返回状态
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        # 解析实时天气数据
        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        # 解析天气预报数据
        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """
    在知识库中搜索相关信息（使用混合检索：密集向量 + 稀疏向量）。

    这是 RAG（检索增强生成）的核心工具。
    Agent 根据用户问题调用此工具，获取相关文档片段，
    然后基于这些片段生成回答。

    调用限制：
    - 每轮对话只能调用一次
    - 超过限制时返回提示信息，要求 Agent 直接基于已有结果回答

    Args:
        query: 用户的问题/查询

    Returns:
        格式化的检索结果字符串，包含文档片段的编号、来源和内容
    """
    # 全局变量，用于控制每轮对话只能调用一次
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN

    # 检查调用次数，超过限制则拒绝执行
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    # 导入 RAG pipeline 并执行检索
    from backend.rag.workflows.pipeline import run_rag_graph

    # 执行 RAG 流程
    rag_result = run_rag_graph(query)

    # 提取检索结果
    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    # 提取 RAG trace 信息，用于前端展示和调试
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}

    # 将 RAG trace 保存到全局上下文，供 Agent Service 后续获取
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    # 没有检索到相关文档
    if not docs:
        return "No relevant documents found in the knowledge base."

    # 格式化检索结果
    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)