"""
Milvus 向量数据库客户端 - 支持密集向量+稀疏向量混合检索

主要功能：
1. 集合管理：创建、删除 Milvus 集合
2. 数据操作：插入、查询、删除向量数据
3. 混合检索：同时使用密集向量和稀疏向量进行检索
4. RRF 融合：使用 Reciprocal Rank Fusion 合并多路检索结果

数据库 Schema：
- id: 主键（自增）
- dense_embedding: 密集向量（float32）
- sparse_embedding: 稀疏向量（sparse float）
- text: 原始文本内容
- filename/file_type/file_path: 文件元数据
- page_number/chunk_idx: 分块元数据
- chunk_id/parent_chunk_id/root_chunk_id: 分块层级关系（用于 Auto-merging）
- chunk_level: 分块层级（1=L1父分块, 2=L2子父分块, 3=L3叶子分块）

索引类型：
- 密集向量：HNSW 索引（适合高维向量近似搜索）
- 稀疏向量：SPARSE_INVERTED_INDEX（适合 BM25 等稀疏检索）
"""

import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

load_dotenv()


class MilvusManager:
    """
    Milvus 连接和集合管理 - 支持密集+稀疏混合检索

    负责：
    - 连接 Milvus 服务器
    - 管理集合的创建和删除
    - 执行向量插入和检索操作
    """

    def __init__(self):
        # 从环境变量读取配置
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")

        # 创建 Milvus 客户端
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

    def init_collection(self, dense_dim: int = 4096):
        """
        初始化 Milvus 集合 - 支持密集向量和稀疏向量混合存储。

        如果集合已存在，则直接返回。
        首次创建时会建立完整的 schema 和索引。

        Schema 字段说明：
        - id: INT64 主键，自增
        - dense_embedding: 密集向量，维度由参数指定（Qwen3-VL-Embedding-8B 为 4096）
        - sparse_embedding: 稀疏向量（Milvus 原生支持）
        - text: 文本内容，最大 2000 字符
        - filename/file_type/file_path: 文件元数据
        - page_number/chunk_idx: 分块位置信息
        - chunk_id/parent_chunk_id/root_chunk_id: 分块层级关系
        - chunk_level: 分块层级（1/2/3）

        索引说明：
        - 密集向量使用 HNSW 索引（M=16, efConstruction=256）
        - 稀疏向量使用 SPARSE_INVERTED_INDEX

        Args:
            dense_dim: 密集向量维度，默认 4096（Qwen3-VL-Embedding-8B）
        """
        if not self.client.has_collection(self.collection_name):
            # 创建 Schema
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

            # 添加主键
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

            # 添加密集向量字段
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)

            # 添加稀疏向量字段
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

            # 添加文本和元数据字段
            schema.add_field("text", DataType.VARCHAR, max_length=2000)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("chunk_idx", DataType.INT64)

            # 添加 Auto-merging 所需层级字段
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_level", DataType.INT64)

            # 创建索引参数
            index_params = self.client.prepare_index_params()

            # 密集向量索引 - HNSW（更适合混合检索场景）
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",  # 内积相似度
                params={"M": 16, "efConstruction": 256}
            )

            # 稀疏向量索引
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

    def insert(self, data: list[dict]):
        """
        插入数据到 Milvus。

        Args:
            data: 数据列表，每条数据是一个字典

        Returns:
            插入结果
        """
        return self.client.insert(self.collection_name, data)

    def query(self, filter_expr: str = "", output_fields: list[str] = None, limit: int = 10000):
        """
        查询数据（非向量搜索）。

        用于获取文档列表等场景。

        Args:
            filter_expr: 过滤表达式
            output_fields: 要返回的字段列表
            limit: 返回数量限制

        Returns:
            查询结果列表
        """
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["filename", "file_type"],
            limit=limit
        )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        根据 chunk_id 批量查询分块（用于 Auto-merging 拉取父块）。

        Args:
            chunk_ids: 分块 ID 列表

        Returns:
            分块数据列表
        """
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []

        # 构造过滤表达式
        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"

        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        混合检索 - 使用 RRF 融合密集向量和稀疏向量的检索结果。

        混合检索流程：
        1. 同时执行密集向量检索和稀疏向量检索
        2. 使用 RRF（Reciprocal Rank Fusion）算法融合两路结果
        3. 返回融合排序后的 top_k 结果

        RRF 公式：
        RRF_score(d) = Σ 1 / (k + rank(d))

        Args:
            dense_embedding: 密集向量查询向量
            sparse_embedding: 稀疏向量查询向量 {index: value, ...}
            top_k: 返回结果数量
            rrf_k: RRF 算法参数，默认 60
            filter_expr: 过滤表达式

        Returns:
            检索结果列表，每条结果包含：
            - id: 主键
            - text/filename/file_type/page_number: 元数据
            - chunk_id/parent_chunk_id/root_chunk_id/chunk_level/chunk_idx: 分块信息
            - score: 检索得分
        """
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]

        # 构造密集向量搜索请求
        # 每次检索更多结果（top_k * 2），为后续融合留出空间
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        # 构造稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        # 使用 RRF 排序算法融合结果
        reranker = RRFRanker(k=rrf_k)

        # 执行混合搜索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            ranker=reranker,
            limit=top_k,
            output_fields=output_fields
        )

        # 格式化返回结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": entity.get("text", ""),
                    "filename": entity.get("filename", ""),
                    "file_type": entity.get("file_type", ""),
                    "page_number": entity.get("page_number", 0),
                    "chunk_id": entity.get("chunk_id", ""),
                    "parent_chunk_id": entity.get("parent_chunk_id", ""),
                    "root_chunk_id": entity.get("root_chunk_id", ""),
                    "chunk_level": entity.get("chunk_level", 0),
                    "chunk_idx": entity.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })

        return formatted_results

    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        仅使用密集向量检索（降级模式）。

        当稀疏向量不可用时，使用此方法进行纯语义检索。

        Args:
            dense_embedding: 密集向量查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[dense_embedding],
            anns_field="dense_embedding",
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            filter=filter_expr,
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })

        return formatted_results

    def delete(self, filter_expr: str):
        """
        删除数据。

        Args:
            filter_expr: 过滤表达式

        Returns:
            删除结果
        """
        return self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self.client.has_collection(self.collection_name)

    def drop_collection(self):
        """删除集合（用于重建 schema）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)