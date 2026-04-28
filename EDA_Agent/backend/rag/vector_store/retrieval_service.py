"""
检索服务模块 - 核心 RAG 检索逻辑

本模块实现了完整的检索流程，包括：

1. 混合检索（Hybrid Search）
   - 密集向量检索：语义相似度
   - 稀疏向量检索：BM25 关键词匹配
   - RRF 融合：合并两路检索结果

2. 重排序（Rerank）
   - 调用外部重排序服务
   - 优化检索结果的排序

3. Auto-merging
   - 将多个相关 L3 分块合并为更大的父分块
   - 提供更完整的上下文

4. 查询扩展（Query Expansion）
   - Step-back：生成高层概念问题
   - HyDE：生成假设性文档

检索流程：
query -> 混合检索 -> 重排序 -> Auto-merging -> 返回结果
          ↓
    如相关性不足
          ↓
    查询扩展（Step-back/HyDE）-> 扩展检索 -> 合并结果 -> 返回
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import re
import requests
from dotenv import load_dotenv

from backend.rag.vector_store.milvus_client import MilvusManager
from backend.rag.vector_store.embedding import EmbeddingService
from backend.rag.vector_store.parent_chunk_store import ParentChunkStore
from backend.memory.chunk_image_storage import chunk_image_storage
from backend.common.prompts import (
    HYDE_PROMPT_TEMPLATE,
    STEP_BACK_ANSWER_PROMPT_TEMPLATE,
    STEP_BACK_QUESTION_PROMPT_TEMPLATE,
)
from langchain.chat_models import init_chat_model

load_dotenv()

# 环境变量配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")

# Auto-merging 配置
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))  # 合并阈值
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))  # 检索层级

# 全局单例：避免重复初始化
_embedding_service = EmbeddingService()
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

# 图片占位符正则
_IMAGE_PH_RE = re.compile(r"<<IMAGE:([0-9a-f]{8})>>")

_stepback_model = None


def _get_rerank_endpoint() -> str:
    """
    获取重排序服务的完整 URL。

    Returns:
        重排序 API 的完整地址
    """
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    """
    将子分块合并为父分块（Auto-merging 的核心逻辑）。

    合并策略：
    - 统计每个父分块下的子分块数量
    - 如果子分块数量 >= threshold，将这些子分块替换为父分块
    - 父分块的 score 取所有子分块的最大值

    Example:
        L3_1 (score=0.9) -> parent=L2_1
        L3_2 (score=0.8) -> parent=L2_1
        L3_3 (score=0.7) -> parent=L2_1
        如果 threshold=2，则 L3_1/2/3 合并为 L2_1（合并后 score=0.9）

    Args:
        docs: L3 分块列表
        threshold: 合并阈值（父分块下至少需要多少子分块才合并）

    Returns:
        (合并后的文档列表, 合并的父分块数量)
    """
    # 按 parent_chunk_id 分组
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 找出满足合并条件的父分块
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    # 查询父分块数据
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0

    # 执行合并
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        # 如果没有父分块，或者父分块不在合并列表中，保留原分块
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        # 替换为父分块
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            # 父分块的 score 取子分块的最大值
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    # 去重
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
    """
    执行两级 Auto-merging。

    合并层级：L3 -> L2 -> L1

    两段合并：
    1. 第一段：将 L3 合并为 L2
    2. 第二段：将合并后的 L2 合并为 L1

    Args:
        docs: 原始 L3 分块列表
        top_k: 返回的最大数量

    Returns:
        (合并后的文档列表, 元信息字典)
    """
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 第一段：L3 -> L2
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    # 第二段：L2 -> L1
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    # 按分数排序并截取 top_k
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
    调用外部重排序服务优化检索结果。

    重排序流程：
    1. 将原始结果附加 RRF 排名
    2. 调用重排序 API
    3. 根据重排序结果调整文档顺序

    Args:
        query: 查询文本
        docs: 原始检索结果
        top_k: 返回数量

    Returns:
        (重排序后的文档列表, 元信息字典)
    """
    # 附加原始排名
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]

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

    # 构造重排序请求
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

        # 解析重排序结果
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
    """
    获取 Step-back 策略使用的 LLM。

    使用全局缓存避免重复初始化。
    """
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
    """
    生成退步问题（Step-back Question）。

    将具体问题抽象为更高层次的概念问题，
    用于理解问题的本质和通用原理。
    """
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = STEP_BACK_QUESTION_PROMPT_TEMPLATE.format(query=query)
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    """
    回答退步问题。

    提供通用原理和背景知识。
    """
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = STEP_BACK_ANSWER_PROMPT_TEMPLATE.format(step_back_question=step_back_question)
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    """
    生成假设性文档（HyDE 策略）。

    根据用户问题生成一段"假设的"答案文档，
    这个文档可能包含合理推测，但用于帮助检索真实的相关文档。
    """
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    """
    执行 Step-back 查询扩展。

    流程：
    1. 生成退步问题
    2. 回答退步问题
    3. 将原问题、退步问题、退步答案组合为扩展查询

    Returns:
        包含 step_back_question, step_back_answer, expanded_query 的字典
    """
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


def _extract_image_tokens(text: str) -> List[str]:
    """
    从文本中提取图片占位符 token。

    Args:
        text: 包含占位符的文本

    Returns:
        token 列表
    """
    matches = _IMAGE_PH_RE.findall(text)
    return matches


def _enrich_chunks_with_images(chunks: List[dict]) -> List[dict]:
    """
    为检索到的文本 chunk 补全图片信息。

    流程：
    1. 提取每个 chunk 中的图片 token
    2. 批量查询 PostgreSQL 获取图片信息
    3. 将占位符替换为真实 URL

    Args:
        chunks: 检索到的文本 chunk 列表

    Returns:
         enriched 后的 chunk 列表
    """
    if not chunks:
        return chunks

    # 收集所有 chunk_id 和 image_tokens
    chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    if not chunk_ids:
        return chunks

    # 查询图片映射
    image_records = chunk_image_storage.get_by_chunk_ids(chunk_ids)
    if not image_records:
        return chunks

    # 建立 token -> image_info 的映射
    token_to_image: Dict[str, dict] = {}
    for record in image_records:
        token = record.get("image_token", "")
        if token:
            token_to_image[token] = record

    # 为每个 chunk 补全图片
    for chunk in chunks:
        text = chunk.get("text", "")
        tokens = _extract_image_tokens(text)

        images = []
        for token in tokens:
            if token in token_to_image:
                img_info = token_to_image[token]
                images.append({
                    "image_token": token,
                    "placeholder": img_info.get("placeholder", f"<<IMAGE:{token}>>"),
                    "image_path": img_info.get("image_path", ""),
                    "mime_type": img_info.get("mime_type", "image/png"),
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                })

        chunk["images"] = images
        chunk["image_count"] = len(images)

    return chunks


def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    核心检索函数。

    完整检索流程：
    1. 生成查询向量（密集 + 稀疏）
    2. 执行混合检索（仅 text 记录）
    3. 执行重排序
    4. 执行 Auto-merging
    5. 补全图片信息
    6. 返回结果和元信息

    如果混合检索失败，降级为纯密集向量检索。

    Args:
        query: 用户查询
        top_k: 返回结果数量

    Returns:
        包含以下键的字典：
        - docs: 检索到的文档列表
        - meta: 检索过程的元信息
    """
    # 候选数量，多取一些用于后续筛选
    candidate_k = max(top_k * 3, top_k)

    # 默认只检索叶子分块（L3）+ 仅 text 记录
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    try:
        # 生成查询向量
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        # 混合检索（过滤 record_type = "text"）
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
            record_type="text",
        )

        # 重排序
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)

        # Auto-merging
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)

        # 补全图片信息
        merged_docs = _enrich_chunks_with_images(merged_docs)

        # 组装元信息
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)

        return {"docs": merged_docs, "meta": rerank_meta}

    except Exception:
        # 降级：尝试纯密集向量检索
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]

            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
                record_type="text",
            )

            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)

            # 补全图片信息
            merged_docs = _enrich_chunks_with_images(merged_docs)

            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)

            return {"docs": merged_docs, "meta": rerank_meta}

        except Exception:
            # 完全失败
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


def retrieve_images(
    image_data: bytes,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    图片检索函数。

    完整检索流程：
    1. 对查询图片生成向量
    2. 在 Milvus 中检索 image 记录
    3. 根据 chunk_id 回溯获取文本上下文
    4. 返回图片命中结果和文本上下文

    Args:
        image_data: 查询图片的字节数据
        top_k: 返回结果数量

    Returns:
        包含以下键的字典：
        - image_hits: 图片检索命中结果
        - text_contexts: 关联的文本上下文 chunk
        - meta: 检索过程的元信息
    """
    try:
        # 生成图片向量
        image_embedding = _embedding_service.embed_image(image_data)
        if image_embedding is None:
            return {
                "image_hits": [],
                "text_contexts": [],
                "meta": {"retrieval_mode": "failed", "error": "image_embedding_failed"},
            }

        # 图片向量检索
        image_hits = _milvus_manager.image_retrieve(
            image_embedding=image_embedding,
            top_k=top_k,
        )

        # 收集 chunk_id，回溯文本上下文
        chunk_ids = [hit.get("chunk_id") for hit in image_hits if hit.get("chunk_id")]
        root_chunk_ids = [hit.get("root_chunk_id") for hit in image_hits if hit.get("root_chunk_id")]

        # 查询文本上下文
        all_chunk_ids = list(set(chunk_ids + root_chunk_ids))
        text_contexts = []

        if all_chunk_ids:
            # 从 Milvus 获取文本 chunk
            text_chunks = _milvus_manager.get_chunks_by_ids(all_chunk_ids)
            # 只保留 text 类型的 chunk
            text_contexts = [c for c in text_chunks if c.get("record_type") == "text"]

            # 补全图片信息
            text_contexts = _enrich_chunks_with_images(text_contexts)

        return {
            "image_hits": image_hits,
            "text_contexts": text_contexts,
            "meta": {
                "retrieval_mode": "image",
                "image_count": len(image_hits),
                "context_count": len(text_contexts),
            },
        }

    except Exception as e:
        return {
            "image_hits": [],
            "text_contexts": [],
            "meta": {"retrieval_mode": "failed", "error": str(e)},
        }