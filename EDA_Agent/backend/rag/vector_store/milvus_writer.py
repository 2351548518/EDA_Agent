"""
Milvus 文档写入模块 - 将分块文档向量化并写入 Milvus

主要功能：
1. 批量写入文档到 Milvus
2. 同时生成密集向量和稀疏向量
3. 支持 BM25 语料库拟合

写入流程：
1. fit_corpus: 用所有文档文本拟合语料库（计算 IDF）
2. get_all_embeddings: 同时生成密集和稀疏向量
3. 批量插入 Milvus
"""

from backend.rag.vector_store.embedding import EmbeddingService
from backend.rag.vector_store.milvus_client import MilvusManager


class MilvusWriter:
    """
    文档向量化并写入 Milvus - 支持密集+稀疏混合向量

    负责将分块后的文档写入向量数据库，
    同时生成两种向量用于混合检索。
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
        批量写入文档到 Milvus。

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
        # 所有文档一起拟合，可以得到更准确的 IDF 值
        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.fit_corpus(all_texts)

        total = len(documents)
        # 分批处理，避免内存占用过大
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            # 构造写入数据
            insert_data = [
                {
                    # 向量字段
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    # 文本和元数据
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    # 分块层级关系
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            # 写入 Milvus
            self.milvus_manager.insert(insert_data)