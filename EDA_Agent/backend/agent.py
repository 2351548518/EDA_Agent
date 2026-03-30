from dotenv import load_dotenv
import os
import json
import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class ConversationStorage:
    """
    对话存储
    {
    "user_fg329z9vq": {
        "session_1774333245520": {
        "messages": [
            {
            "type": "human",
            "content": "你是",
            "timestamp": "2026-03-24T14:21:05.989543"
            },
            {
            "type": "ai",
            "content": "我是一个专业的集成电路（IC）领域问答助手，专注于 EDA 工具、芯片设计、验证流程、工艺制程等相关问题。当回答时，我可以使用工具来辅助。当用户询问文档或知识性问题时，我会使用 search_knowledge_base 工具检索相关知识库。不要在一轮中重复调用同一个工具。每轮对话最多只能调用一次知识库检索工具。一旦我调用了 search_knowledge_base 并收到结果，我必须立即基于该结果生成最终答案。收到 search_knowledge_base 结果后，我不得再次调用任何工具（包括 get_current_weather 或 search_knowledge_base）。如果检索到的上下文不足以回答问题，请诚实地说明我不知道，而不是编造事实。如果工具结果包含 Step-back Question/Answer，请利用该通用原则进行推理和回答，但不要显式展示推理过程。如果我不知道答案，请诚实地承认。",
            "timestamp": "2026-03-24T14:21:05.989543",
            "rag_trace": null
            }
        ],
        "metadata": {},
        "updated_at": "2026-03-24T14:21:05.989543"
        },
    """

    def __init__(self, storage_file: str = None):
        if storage_file:
            storage_path = os.path.abspath(storage_file)
        else:
            package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(package_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "customer_service_history.json")

        self.storage_file = storage_path

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        data = self._load()

        if user_id not in data:
            data[user_id] = {}

        serialized = []
        for idx, msg in enumerate(messages):
            record = {
                "type": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            if extra_message_data and idx < len(extra_message_data):
                extra = extra_message_data[idx] or {}
                if "rag_trace" in extra:
                    record["rag_trace"] = extra["rag_trace"]
            serialized.append(record)

        data[user_id][session_id] = {
            "messages": serialized,
            "metadata": metadata or {},
            "updated_at": datetime.now().isoformat()
        }

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        data = self._load()

        if user_id not in data or session_id not in data[user_id]:
            return []

        messages = []
        for msg_data in data[user_id][session_id]["messages"]:
            if msg_data["type"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                messages.append(AIMessage(content=msg_data["content"]))
            elif msg_data["type"] == "system":
                messages.append(SystemMessage(content=msg_data["content"]))

        return messages

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        data = self._load()
        if user_id not in data:
            return []
        return list(data[user_id].keys())

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        data = self._load()
        if user_id not in data or session_id not in data[user_id]:
            return False

        del data[user_id][session_id]
        if not data[user_id]:
            del data[user_id]

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def _load(self) -> dict:
        """加载数据"""
        if not os.path.exists(self.storage_file):
            return {}
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}



def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base], # 这里把RAG 检索 当成一个工具，agent 可以根据需要调用它来获取信息
        system_prompt=(
            "你是一个专业的集成电路（IC）领域问答助手，专注于 EDA 工具、芯片设计、验证流程、工艺制程等相关问题。 "
            "当回答时，你可以使用工具来辅助。 "
            "当用户询问文档或知识性问题时，使用 search_knowledge_base 工具检索相关知识库。 "
            "不要在一轮中重复调用同一个工具。每轮对话最多只能调用一次知识库检索工具。 "
            "一旦你调用了 search_knowledge_base 并收到结果，你必须立即基于该结果生成最终答案。 "
            "收到 search_knowledge_base 结果后，你不得再次调用任何工具（包括 get_current_weather 或 search_knowledge_base）。 "
            "如果检索到的上下文不足以回答问题，请诚实地说明你不知道，而不是编造事实。 "
            "如果工具结果包含 Step-back Question/Answer，请利用该通用原则进行推理和回答， "
            "但不要显式展示推理过程。 "
            "如果你不知道答案，请诚实地承认。"
        ),
    )
    return agent, model


agent, model = create_agent_instance()

storage = ConversationStorage()

def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

                        {old_conversation}

                        总结（包含用户信息、重要事实、待办事项）：
                        """

    summary = model.invoke(summary_prompt).content
    return summary


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()
    
    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])

        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

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
    
    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。
    
    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True) # 获取最近一次 RAG 检索上下文，默认读取后清空。
    reset_tool_call_guards() # 每轮对话开始时重置工具调用计数。 一次对话轮次内对同一工具的调用超过限制时，工具会拒绝执行并返回提示信息，这有助于防止 agent 过度依赖某个工具导致的循环调用或性能问题。

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy()) # 设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。

    if len(messages) > 50:
        # 大于 50 条消息时，先总结前 40 条，生成摘要消息插入对话开头，减少 agent 上下文长度，同时保留最新的 10 条消息供 agent 参考。
        summary = summarize_old_messages(model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))

    full_response = ""

    async def _agent_worker():
        """
        后台任务：运行 agent 并将内容 chunk 推入输出队列。


        定义了一个后台异步任务 _agent_worker，它会调用 agent 的流式接口 agent.astream，不断获取 AI 生成的消息片段。每当有新内容生成时，都会立即放入 output_queue，并拼接到 full_response 变量中。异常时会推送错误信息，结束时会推送一个 None 作为“哨兵”信号，通知主循环任务已完成。
        """
        nonlocal full_response # 声明 full_response 这个变量不是当前函数的局部变量，而是“外层但非全局”的变量，以便在这个异步函数内修改它的值。
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages", # 指定流式输出的模式为“messages”，意味着每次返回的都是一条消息片段，而不是一次性返回全部内容
                config={"recursion_limit": 8}, # 限制递归深度，防止 agent 递归调用过多导致栈溢出
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

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
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())


    """
    主循环部分不断从 output_queue 取出事件，并通过 yield 以 SSE 格式推送给前端。这样，前端可以实时收到内容片段和 RAG 步骤进度。如果客户端断开连接，会捕获 GeneratorExit，取消后台任务并做清理。
    """

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
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


    """
    最后，函数会获取本轮的 RAG trace 信息（如果有），推送给前端，并发送结束信号 [DONE]。最后一步是把本轮 AI 回复和 trace 信息保存到存储中，保证对话历史完整。
    """

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
