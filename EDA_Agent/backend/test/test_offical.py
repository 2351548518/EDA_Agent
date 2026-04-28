from dotenv import load_dotenv
load_dotenv()  # 必须在 langchain 相关模块 import 之前
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict
from langsmith import evaluate
from backend.agent.service import chat_with_agent
from backend.common.prompts import (
    TEST_CORRECTNESS_INSTRUCTIONS,
    TEST_GROUNDED_INSTRUCTIONS,
    TEST_RELEVANCE_INSTRUCTIONS,
    TEST_RETRIEVAL_RELEVANCE_INSTRUCTIONS,
)
from uuid import uuid4

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL")

# 1. Select your dataset
dataset_name = "eda_rag"

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = TEST_CORRECTNESS_INSTRUCTIONS

"""
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True, # 开启流式输出，模型会边生成边返回内容，提高响应速度和用户体验
        )
"""

# Grader LLM
grader_llm = init_chat_model(
    model=GRADE_MODEL, 
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    # Run evaluator
    grade = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]

# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

# Grade prompt
relevance_instructions = TEST_RELEVANCE_INSTRUCTIONS

# Grader LLM
# relevance_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
#     RelevanceGrade, method="json_schema", strict=True
# )
relevance_llm = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)


# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = TEST_GROUNDED_INSTRUCTIONS

# Grader LLM
grounded_llm = init_chat_model(
    model=GRADE_MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0
).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

# Grade prompt
retrieval_relevance_instructions = TEST_RETRIEVAL_RELEVANCE_INSTRUCTIONS

# Grader LLM
retrieval_relevance_llm = init_chat_model(
    model=GRADE_MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]


def target_function(inputs: dict) -> dict:
    question = inputs["question"]
    session_id = f"langsmith_eval_{uuid4().hex}"
    result = chat_with_agent(
        user_text=question,
        user_id="langsmith_eval_user",
        session_id=session_id,
    )

    response_text = ""
    documents = []
    if isinstance(result, dict):
        response_text = str(result.get("response", "") or "")
        rag_trace = result.get("rag_trace", {}) or {}
        # 从 rag_trace 中提取检索到的文档
        retrieved_chunks = rag_trace.get("retrieved_chunks", []) if rag_trace else []
        for chunk in retrieved_chunks:
            # 创建类似 Document 的对象，评估函数需要 page_content 属性
            doc_content = chunk.get("text", "")
            if doc_content:
                documents.append(type('Document', (), {'page_content': doc_content})())
    else:
        response_text = str(result)

    print(f"Debug: target_function got result: {response_text}, rag_trace: {rag_trace}")
    print(f"Debug: target_function got result: {response_text}, documents: {[doc.page_content for doc in documents]}")
    return {
        "answer": response_text,
        "documents": documents,
    }

experiment_results = evaluate(
    target_function,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
)

# Explore results locally as a dataframe if you have pandas installed
# experiment_results.to_pandas()