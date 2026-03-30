from collections import defaultdict
from typing import List, Tuple, Dict, Any
import os
import json
import requests
from dotenv import load_dotenv

from milvus_client import MilvusManager
from embedding import EmbeddingService
from parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 全局初始化检索依赖，避免反复构造
_embedding_service = EmbeddingService()
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    # 函数会遍历所有输入的分块文档，将拥有相同 parent_chunk_id 的分块归为一组
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 只有当某个父分块下的子分块数量达到或超过 threshold（默认2）时，才会考虑将这些子分块合并为父分块
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids: # 如果没有任何父分块满足合并条件，则直接返回原始文档列表和0（表示没有进行合并）
        return docs, 0

    # 根据满足合并条件的 parent_chunk_id 列表，从父分块存储中批量查询对应的父分块文档
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    # 再次遍历原始分块列表：如果某个分块没有父分块，或者父分块信息缺失，则直接保留原分块；否则，将其替换为父分块，并更新父分块的分数（取所有子分块分数的最大值），并标记该父分块是由子分块合并而来，同时记录合并的子分块数量
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    # 为了避免重复，函数会对合并后的结果去重，依据 chunk_id 或（文件名、页码、文本内容）三元组作为唯一标识。最终返回去重后的文档列表和实际合并的父分块数量。
    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    # L3 在 Milvus 中是叶子分块，L2 是父分块，L1 是祖父分块。只有当某个父分块下的子分块数量达到或超过 AUTO_MERGE_THRESHOLD 时，才会将这些子分块合并为父分块。L2 和 L1 在 ParentChunkStore 中存储，合并时会更新父分块的分数为其子分块分数的最大值，并标记该父分块是由子分块合并而来，同时记录合并的子分块数量。最终返回合并后的文档列表（根据 score 排序后取 top_k）和包含合并相关元信息的字典（是否启用自动合并、是否实际应用了自动合并、使用的合并阈值、被替换的子分块数量以及执行的合并步骤数等）。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    """
    进行 API 级精排，支持返回 rerank_score
    1. 首先将原始检索结果附加 RRF 排名信息，构造重排序服务的输入。
    2. 调用外部重排序服务，传入查询和文档列表，获取重排序结果。
    3. 根据重排序结果调整原始文档顺序，并附加重排序得分信息。
    4. 返回重排序后的文档列表和重排序相关的元信息（是否启用重排序、使用的模型、调用的 endpoint、错误信息等）。
    """

    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)] # 通过 {**doc, "rrf_rank": i} 这种写法，将原有的字典内容展开，并新增一个键值对 "rrf_rank": i，表示该文档在原始列表中的顺序排名
    
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return docs_with_rank[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return docs_with_rank[:top_k], meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    
    candidate_k = max(top_k * 3, top_k)

    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}" # 默认只检索叶子分块，减少无关信息干扰。可以通过工具参数调整这个过滤条件，比如允许检索更高层级的分块，或者特定类型的文档等。
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr, # 用于过滤检索结果的表达式，比如只检索特定层级或类型的文档
        )
        """
        retrieved = [
            {
                "id": "唯一标识符",
                "text": "文档内容片段",
                "filename": "来源文件名",
                "page_number": "页码信息",
                "chunk_id": "分块ID",
                "parent_chunk_id": "父分块ID",
                "root_chunk_id": "根分块ID",
                "chunk_level": 3,
                "chunk_idx": 0,
                "score": 0.85, # 原始检索得分
            },
            ...
        ]
        """
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        """
        reranked = [
            {
                "id": "唯一标识符",
                "text": "文档内容片段",
                "filename": "来源文件名",
                "page_number": "页码信息",
                "chunk_id": "分块ID",
                "parent_chunk_id": "父分块ID",
                "root_chunk_id": "根分块ID",
                "chunk_level": 3,
                "chunk_idx": 0,
                "score": 0.85, # 原始检索得分
                "rerank_score": 0.92, # 重排序得分
                "rrf_rank": 5, # RRF 原始排名
            },
            ...
        ]
        其中 rerank_meta 包含了重排序相关的元信息，比如是否启用了重排序、使用的模型、调用的 endpoint、错误信息（如果有的话）以及候选文档数量等。
        rerank_meta = {
                    "rerank_enabled": True, # 是否启用了重排序功能
                    "rerank_applied": True, # 是否实际应用了重排序（可能因为某些原因未能成功调用重排序服务）
                    "rerank_model": "模型名称", # 使用的重排序模型
                    "rerank_endpoint": "重排序服务的URL", # 调用的重排序服务的URL
                    "rerank_error": None, # 如果调用重排序服务失败，这里会有错误信息，否则为 None
                    "candidate_count": 60, # 参与重排序的候选文档数量
                }
        """
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)
        return {"docs": merged_docs, "meta": rerank_meta}
    except Exception:
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)
            return {"docs": merged_docs, "meta": rerank_meta}
        except Exception:
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }
