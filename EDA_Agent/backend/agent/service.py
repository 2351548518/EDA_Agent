"""
Agent 服务模块 - 负责与 AI Agent 交互和对话历史管理

主要功能：
1. 创建和管理 LangChain Agent（集成 RAG 工具）
2. 处理同步/流式聊天请求
3. 管理对话历史的加载和保存
4. 长对话摘要压缩

架构说明：
- Agent 使用 LangChain 的 create_agent 函数创建
- Agent 配备两个工具：get_current_weather（天气查询）和 search_knowledge_base（RAG 检索）
- 对话历史存储在 PostgreSQL 数据库中
- 当对话长度超过 50 条时，会自动将早期消息压缩成摘要
"""

import json

from dotenv import load_dotenv
import os
import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage

# 导入 Agent 工具
from backend.tools.agent_tools import (
    get_current_weather,
    search_knowledge_base,
    get_last_rag_context,
    reset_tool_call_guards,
    set_rag_step_queue,
)

# 导入对话存储
from backend.memory.storage import ConversationStorage

# 导入提示词模板
from backend.common.prompts import AGENT_SYSTEM_PROMPT, SUMMARY_PROMPT_TEMPLATE

load_dotenv()

# 从环境变量读取配置
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


def create_agent_instance():
    """
    创建 LangChain Agent 实例。

    使用 ARK API 初始化 chat model，然后创建配备工具的 agent。
    工具包括：
    - get_current_weather: 获取天气信息
    - search_knowledge_base: RAG 知识库检索

    Returns:
        agent: LangChain Agent 实例
        model: Chat model 实例（用于摘要生成）
    """
    # 初始化 chat model
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",  # 使用 OpenAI 兼容接口
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,  # 较低的随机性，保持回答一致性
        stream_usage=True,  # 启用流式使用的追踪
    )

    # 创建 Agent，传入 model、工具列表和系统提示词
    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        # 将 RAG 检索当成一个工具，agent 可以根据需要调用它来获取信息
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    return agent, model


# 全局单例：Agent 和 Model
agent, model = create_agent_instance()

# 对话存储：负责从 PostgreSQL 加载/保存对话历史
storage = ConversationStorage()


def summarize_old_messages(model, messages: list) -> str:
    """
    将旧消息总结为摘要。

    当对话长度超过一定阈值时，会将早期消息压缩成摘要，
    以减少 Token 消耗同时保留关键信息。

    Args:
        model: 用于生成摘要的 chat model
        messages: 要总结的消息列表

    Returns:
        生成的摘要文本
    """
    # 构建旧对话文本
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 使用提示词模板生成摘要
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(old_conversation=old_conversation)

    summary = model.invoke(summary_prompt).content
    return summary


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """
    使用 Agent 处理用户消息并返回响应（同步版本）。

    处理流程：
    1. 加载对话历史
    2. 清理可能残留的 RAG 上下文
    3. 如果对话过长，压缩早期消息
    4. 调用 Agent 处理消息
    5. 提取响应内容和 RAG trace
    6. 保存对话历史

    Args:
        user_text: 用户消息
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        dict: 包含 response 和 rag_trace 的字典
    """
    # 加载用户的对话历史
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 如果对话超过 50 条，将前 40 条压缩成摘要
    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        # 在最新消息前插入摘要作为系统消息
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    # 添加用户新消息
    messages.append(HumanMessage(content=user_text))

    # 调用 Agent 处理
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},  # 限制递归深度，防止无限循环
    )

    # 提取 AI 的回复内容
    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    # 添加 AI 回复到对话历史
    messages.append(AIMessage(content=response_content))

    # 获取本轮的 RAG trace（如有）
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 保存对话历史，extra_message_data 用于存储每条消息的额外信息（如 rag_trace）
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session"
):
    """
    使用 Agent 处理用户消息并流式返回响应（异步 SSE 版本）。

    与同步版本不同，流式版本会实时推送：
    - AI 生成的内容片段
    - RAG 检索的中间步骤（通过 set_rag_step_queue）
    - 最终的 RAG trace 信息

    架构说明：
    - 使用统一输出队列 + 后台任务模式
    - RAG 检索步骤在工具执行期间实时推送，而不是等待工具完成
    - 客户端断开连接时会自动取消后台任务

    Yields:
        SSE 格式的数据字符串，类型包括：
        - content 事件：AI 生成的内容片段
        - rag_step 事件：RAG 检索步骤更新
        - trace 事件：RAG 追踪信息
        - error 事件：错误信息
        - [DONE]：结束信号
    """
    # 加载对话历史
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)

    # 每轮对话开始时重置工具调用计数
    # 一次对话轮次内对同一工具的调用超过限制时，工具会拒绝执行
    # 这有助于防止 agent 过度依赖某个工具导致的循环调用
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RAGStepProxy:
        """
        RAG 步骤代理对象。

        将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。
        这样可以将 RAG 步骤信息以统一格式推送给前端。
        """
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    # 设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度
    set_rag_step_queue(_RAGStepProxy())

    # 如果对话过长，进行摘要压缩
    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    # 添加用户消息
    messages.append(HumanMessage(content=user_text))

    full_response = ""  # 完整回复，用于最后保存

    async def _agent_worker():
        """
        后台任务：运行 Agent 并将内容 chunk 推入输出队列。

        使用 agent.astream 流式获取 AI 生成的消息片段，
        每次有新内容生成时都立即放入 output_queue。
        结束时推送 None 作为"哨兵"信号。
        """
        nonlocal full_response  # 在外层函数中定义的变量

        try:
            # 使用流式模式运行 Agent
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",  # 每次返回一条消息片段
                config={"recursion_limit": 8},  # 限制递归深度
            ):
                # 只处理 AIMessageChunk 类型，跳过其他类型
                if not isinstance(msg, AIMessageChunk):
                    continue

                # 跳过工具调用块
                if getattr(msg, "tool_call_chunks", None):
                    continue

                # 提取消息内容（支持多种格式）
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    # 放入队列，等待主循环推送
                    await output_queue.put({"type": "content", "content": content})

        except Exception as e:
            # 推送错误信息
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 Agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    """
    主循环部分不断从 output_queue 取出事件，并通过 yield 以 SSE 格式推送给前端。
    这样，前端可以实时收到内容片段和 RAG 步骤进度。
    """
    try:
        while True:
            event = await output_queue.get()
            if event is None:  # 收到哨兵，结束
                break
            yield f"data: {json.dumps(event)}\n\n"

    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务，避免资源泄漏
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭

    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话历史
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)