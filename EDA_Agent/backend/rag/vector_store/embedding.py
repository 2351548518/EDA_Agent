"""
文本向量化服务模块 - 支持密集向量和稀疏向量（BM25）

主要功能：
1. 密集向量生成：调用 ARK API 将文本转换为高维向量
2. 稀疏向量生成：实现 BM25 算法生成稀疏向量
3. 混合支持：同时生成密集和稀疏向量用于混合检索

BM25 算法说明：
- 一种基于词频的经典检索算法
- 结合词频（TF）和逆文档频率（IDF）
- 通过参数 k1 和 b 控制词频饱和和文档长度归一化

稀疏向量结构：
- {index: score, ...} 格式
- index 是词在词汇表中的位置
- score 是 BM25 计算的权重
"""

import os
import re
import math
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """
    文本向量化服务 - 同时支持密集向量和稀疏向量（BM25）

    密集向量用于语义相似度检索
    稀疏向量用于关键词精确匹配
    两者结合可以实现混合检索效果
    """

    def __init__(self):
        # ARK API 配置
        self.base_url = os.getenv("BASE_URL")
        self.embedder = os.getenv("EMBEDDER")
        self.api_key = os.getenv("ARK_API_KEY")

        # BM25 参数
        self.k1 = 1.5  # 词频饱和参数，控制词频增长的平滑程度
        self.b = 0.75  # 文档长度归一化参数，控制文档长度的影响程度

        # 词汇表（将词映射到稀疏向量索引）
        self._vocab = {}
        self._vocab_counter = 0

        # 文档频率统计（用于 IDF 计算）
        self._doc_freq = Counter()
        self._total_docs = 0
        self._avg_doc_len = 0

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        调用嵌入 API 生成密集向量。

        Args:
            texts: 待转换的文本列表（支持批量）

        Returns:
            向量列表，每个向量是 float 列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.embedder,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            # 调用 ARK Embedding API
            response = requests.post(f"{self.base_url}/embeddings", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            raise Exception(f"嵌入 API 调用失败: {str(e)}")

    def tokenize(self, text: str) -> list[str]:
        """
        简单分词器 - 支持中英文混合。

        处理规则：
        - 中文字符：每个字符作为一个 token
        - 英文字母：按单词分割
        - 标点和特殊字符：忽略

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        text = text.lower()

        tokens = []
        # 匹配中文字符（Unicode 范围 4e00-9fff）
        chinese_pattern = re.compile(r'[一-鿿]')
        # 匹配英文字母
        english_pattern = re.compile(r'[a-zA-Z]+')

        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                # 中文字符单独作为一个 token
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                # 英文单词
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                # 跳过标点和特殊字符
                i += 1

        return tokens

    def fit_corpus(self, texts: list[str]):
        """
        拟合语料库，计算 IDF 和平均文档长度。

        在文档写入前调用，用于：
        1. 统计每个词出现在多少文档中（文档频率）
        2. 建立词汇表
        3. 计算平均文档长度

        Args:
            texts: 文档列表
        """
        self._total_docs = len(texts)
        total_len = 0

        for text in texts:
            tokens = self.tokenize(text)
            total_len += len(tokens)

            # 统计文档频率（每个词在多少文档中出现）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freq[token] += 1

                # 建立词汇表
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1

        self._avg_doc_len = total_len / self._total_docs if self._total_docs > 0 else 1

    def get_sparse_embedding(self, text: str) -> dict:
        """
        生成 BM25 稀疏向量。

        BM25 公式：
        score(D, Q) = IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * |D| / avgdl))

        Args:
            text: 输入文本

        Returns:
            稀疏向量 {index: value, ...}
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)  # 词频

        sparse_vector = {}

        for token, freq in tf.items():
            # 新词加入词汇表
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1

            idx = self._vocab[token]

            # 计算 IDF（逆文档频率）
            df = self._doc_freq.get(token, 0)
            if df == 0:
                # 新词，使用平滑 IDF
                idf = math.log((self._total_docs + 1) / 1)
            else:
                # 标准 BM25 IDF 公式
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

            # 计算 BM25 分数
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
            score = idf * numerator / denominator

            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """
        批量生成 BM25 稀疏向量。

        Args:
            texts: 文本列表

        Returns:
            稀疏向量列表
        """
        return [self.get_sparse_embedding(text) for text in texts]

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """
        同时生成密集向量和稀疏向量。

        用于混合检索场景，同时提供两种向量给 Milvus。

        Args:
            texts: 文本列表

        Returns:
            (密集向量列表, 稀疏向量列表)
        """
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings