"""
Milvus 文档写入模块 - 将分块文档和图片向量化并写入 Milvus

主要功能：
1. 批量写入文本 chunk 到 Milvus
2. 批量写入图片记录到 Milvus
3. 文本向量使用 dense_embedding + sparse_embedding
4. 图片向量使用统一的 embedding 字段
5. 支持 BM25 语料库拟合

写入规则：
- chunk 记录：保存清洗后的文本内容，record_type="text"
- image 记录：每张图片单独写一条记录，record_type="image"

写入流程：
1. fit_corpus: 用所有文档文本拟合语料库（计算 IDF）
2. write_documents: 写入文本 chunk
3. write_images: 写入图片记录
"""

import json
from typing import List, Dict, Optional

from backend.rag.vector_store.embedding import EmbeddingService
from backend.rag.vector_store.milvus_client import MilvusManager


class MilvusWriter:
    """
    文档和图片向量化并写入 Milvus - 支持密集+稀疏混合向量和图片向量

    负责将分块后的文档和图片写入向量数据库。
    """

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        """
        初始化 MilvusWriter。

        Args:
            embedding_service: 文本向量化服务（用于生成密集/稀疏向量）
            milvus_manager: Milvus 客户端（用于写入数据）
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50):
        """
        批量写入文本 chunk 到 Milvus。

        处理流程：
        1. 确保 Milvus 集合已初始化
        2. 用所有文档文本拟合语料库（用于 BM25 IDF 计算）
        3. 分批处理：
           - 生成密集向量和稀疏向量
           - 构造写入数据
           - 插入 Milvus

        Args:
            documents: 文档列表（通常是 L3 叶子分块）
            batch_size: 每批处理的文档数量
        """
        if not documents:
            return

        # 确保 Milvus 集合已初始化
        self.milvus_manager.init_collection()

        # 先拟合语料库（用于 BM25 IDF 计算）
        # 使用 clean_text 拟合（无占位符）
        all_texts = []
        for doc in documents:
            text = doc.get("clean_text") or doc.get("text", "")
            all_texts.append(text)

        self.embedding_service.fit_corpus(all_texts)

        total = len(documents)
        # 分批处理，避免内存占用过大
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]

            # 使用 clean_text 进行向量化
            texts = [doc.get("clean_text") or doc.get("text", "") for doc in batch]

            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            # 构造写入数据
            insert_data = []
            for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings):
                # 序列化 image_tokens 为 JSON 字符串
                image_tokens_str = json.dumps(doc.get("image_tokens", []), ensure_ascii=False)

                record = {
                    # 向量字段
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "embedding": dense_emb,  # 统一使用 dense embedding
                    # 文本和元数据
                    "text": doc.get("text", ""),
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    # 分块层级关系
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                    # 记录类型
                    "record_type": "text",
                    # 图片记录专用字段（text 记录填默认值，避免 schema 非空字段缺失）
                    "image_id": "",
                    "image_token": "",
                    "placeholder": "",
                    # 图片信息
                    "has_image": doc.get("has_image", False),
                    "image_count": doc.get("image_count", 0),
                    "image_tokens": image_tokens_str,
                    # 图片元数据字段（text 记录填默认值）
                    "mime_type": "",
                    "width": 0,
                    "height": 0,
                }
                insert_data.append(record)

            # 写入 Milvus
            self.milvus_manager.insert(insert_data)

    def write_images(self, image_records: list[dict], batch_size: int = 50):
        """
        批量写入图片记录到 Milvus。

        处理流程：
        1. 确保 Milvus 集合已初始化
        2. 分批处理：
           - 对图片字节数据生成向量
           - 构造写入数据
           - 插入 Milvus

        Args:
            image_records: 图片记录列表，每条记录包含：
                - image_id: 图片唯一标识
                - image_token: 占位符 token
                - placeholder: 占位符字符串
                - chunk_id: 所属 chunk ID
                - root_chunk_id: 根块 ID
                - page_number: 页码
                - filename: 文件名
                - image_bytes: 图片字节数据
                - mime_type: MIME 类型
                - width: 宽度
                - height: 高度
            batch_size: 每批处理的记录数量
        """
        if not image_records:
            return

        # 确保 Milvus 集合已初始化
        self.milvus_manager.init_collection()

        total = len(image_records)
        # 分批处理
        for i in range(0, total, batch_size):
            batch = image_records[i:i + batch_size]

            insert_data = []
            for record in batch:
                # Milvus INT64 字段不接受 nil，统一将宽高归一为 int
                try:
                    width = int(record.get("width")) if record.get("width") is not None else 0
                except (TypeError, ValueError):
                    width = 0

                try:
                    height = int(record.get("height")) if record.get("height") is not None else 0
                except (TypeError, ValueError):
                    height = 0

                # 生成图片向量
                image_bytes = record.get("image_bytes")
                if image_bytes:
                    embedding_vec = self.embedding_service.embed_image(image_bytes)
                    if embedding_vec is None:
                        embedding_vec = self.embedding_service.get_zero_vector()
                else:
                    embedding_vec = self.embedding_service.get_zero_vector()

                insert_record = {
                    # 统一向量字段（图片向量）
                    "embedding": embedding_vec,
                    # 图片记录不使用 dense 和 sparse
                    "dense_embedding": self.embedding_service.get_zero_vector(),
                    "sparse_embedding": {},
                    # 文本为空
                    "text": "",
                    # 元数据
                    "filename": record.get("filename", ""),
                    "file_type": record.get("file_type", "image"),
                    "file_path": record.get("image_path", ""),
                    "page_number": record.get("page_number", 0),
                    "chunk_idx": 0,
                    # 分块层级关系
                    "chunk_id": record.get("chunk_id", ""),
                    "parent_chunk_id": "",
                    "root_chunk_id": record.get("root_chunk_id", ""),
                    "chunk_level": 0,
                    # 记录类型
                    "record_type": "image",
                    # 图片记录专用字段
                    "image_id": record.get("image_id", ""),
                    "image_token": record.get("image_token", ""),
                    "placeholder": record.get("placeholder", ""),
                    # 图片元数据
                    "mime_type": record.get("mime_type", "image/png"),
                    "width": width,
                    "height": height,
                    # text 记录的图片信息字段（空）
                    "has_image": False,
                    "image_count": 0,
                    "image_tokens": "[]",
                }
                insert_data.append(insert_record)

            # 写入 Milvus
            self.milvus_manager.insert(insert_data)

    def write_documents_and_images(
        self,
        documents: list[dict],
        image_records: list[dict],
        batch_size: int = 50,
    ):
        """
        批量写入文本 chunk 和图片记录到 Milvus。

        先写文本 chunk，再写图片记录。

        Args:
            documents: 文本 chunk 列表
            image_records: 图片记录列表
            batch_size: 每批处理的记录数量
        """
        self.write_documents(documents, batch_size)
        self.write_images(image_records, batch_size)