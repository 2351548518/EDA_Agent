"""
提示词模板模块 - 集中管理所有后端使用的提示词模板

包含的提示词：
1. AGENT_SYSTEM_PROMPT: Agent 系统提示词
2. SUMMARY_PROMPT_TEMPLATE: 对话摘要提示词
3. RAG_GRADE_PROMPT: 文档相关性评估提示词
4. RAG_REWRITE_STRATEGY_PROMPT_TEMPLATE: 查询扩展策略选择提示词
5. STEP_BACK_QUESTION_PROMPT_TEMPLATE: 退步问题生成提示词
6. STEP_BACK_ANSWER_PROMPT_TEMPLATE: 退步问题回答提示词
7. HYDE_PROMPT_TEMPLATE: HyDE 假设性文档生成提示词
8. 各类评测提示词（用于测试和评估）
"""

"""Centralized prompt definitions for backend modules."""

# ============================================================================
# Agent 系统提示词
# ============================================================================

AGENT_SYSTEM_PROMPT = (
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
)

# ============================================================================
# 对话摘要提示词
# ============================================================================

SUMMARY_PROMPT_TEMPLATE = """请总结以下对话的关键信息：

{old_conversation}

总结（包含用户信息、重要事实、待办事项）：
"""

# ============================================================================
# RAG 相关提示词
# ============================================================================

RAG_GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n"
    "Please output your answer as a JSON object."
)

RAG_REWRITE_STRATEGY_PROMPT_TEMPLATE = (
    "请根据用户问题选择最合适的查询扩展策略，以 JSON 格式输出策略名。\n"
    "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
    "- hyde：模糊、概念性、需要解释或定义的问题。\n"
    "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
    "用户问题：{question}"
)

STEP_BACK_QUESTION_PROMPT_TEMPLATE = (
    "请将用户的具体问题抽象成更高层次、更概括的'退步问题'，"
    "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
    "用户问题：{query}"
)

STEP_BACK_ANSWER_PROMPT_TEMPLATE = (
    "请简要回答以下退步问题，提供通用原理/背景知识，"
    "控制在120字以内。只输出答案，不要列出推理过程。\n"
    "退步问题：{step_back_question}"
)

HYDE_PROMPT_TEMPLATE = (
    "请基于用户问题生成一段'假设性文档'，内容应像真实资料片段，"
    "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
    "只输出文档正文，不要标题或解释。\n"
    "用户问题：{query}"
)

# ============================================================================
# 评测相关提示词（用于测试和评估）
# ============================================================================

TEST_CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

TEST_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

TEST_GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

TEST_RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""