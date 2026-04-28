"""
多模态向量化服务模块 - 使用 DashScope qwen3-vl-embedding

主要功能：
1. 文本向量生成：调用 DashScope qwen3-vl-embedding 生成文本向量
2. 图片向量生成：调用 DashScope qwen3-vl-embedding 生成图片向量
3. 稀疏向量生成：实现 BM25 算法生成稀疏向量（用于混合检索）
4. 统一语义空间：文本和图片使用同一个模型，同一语义空间

qwen3-vl-embedding 说明：
- 支持 2560/2048/1536/1024/768/512/256 维度
- enable_fusion=False：文本和图片各自输出独立向量
- 文本向量和图片向量可在同一 embedding 字段检索
"""

import os
import re
import math
import logging
import time
import base64
from collections import Counter
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# DashScope imports
try:
    import dashscope
    from dashscope import MultiModalEmbedding
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning("dashscope not installed, multimodal embedding will not be available")


# qwen3-vl-embedding 模型配置（从 .env 加载）
MULTIMODAL_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-vl-embedding")
DEFAULT_EMBEDDING_DIM = int(os.getenv("DEFAULT_EMBEDDING_DIM", "1024"))


class EmbeddingService:
    """
    多模态向量化服务 - 使用 DashScope qwen3-vl-embedding

    同时支持：
    - 文本向量：用于文本检索
    - 图片向量：用于图片检索
    - 稀疏向量：BM25 用于关键词匹配
    """

    def __init__(self):
        # DashScope API 配置
        self._dashscope_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ARK_API_KEY")
        self._embedding_dim = int(os.getenv("DEFAULT_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))

        # 确保 dashscope API key 已设置（参考代码的方式）
        import dashscope
        dashscope.api_key = self._dashscope_api_key

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        # 词汇表（将词映射到稀疏向量索引）
        self._vocab = {}
        self._vocab_counter = 0

        # 文档频率统计（用于 IDF 计算）
        self._doc_freq = Counter()
        self._total_docs = 0
        self._avg_doc_len = 0

    def _call_multimodal_embedding(
        self,
        inputs: List[dict],
        dimension: int = None,
        retry: int = 3,
    ) -> List[float]:
        """
        调用 DashScope qwen3-vl-embedding 生成向量。

        Args:
            inputs: 输入列表，如 [{"text": "文本"}] 或 [{"image": "data:image/png;base64,..."}]
            dimension: 向量维度
            retry: 重试次数

        Returns:
            向量列表
        """
        if not DASHSCOPE_AVAILABLE:
            raise RuntimeError("dashscope not installed, cannot generate embeddings")

        dimension = dimension or self._embedding_dim

        for attempt in range(retry):
            try:
                resp = MultiModalEmbedding.call(
                    model=MULTIMODAL_MODEL,
                    input=inputs,
                    enable_fusion=False,  # 独立向量模式
                    dimension=dimension,
                )

                print(f"[MultimodalEmbed] Response status: {resp.status_code}")
                if resp.status_code != 200:
                    print(f"[MultimodalEmbed] Error response: {resp.message}")
                    print(f"[MultimodalEmbed] Error code: {getattr(resp, 'code', 'N/A')}")
                    raise RuntimeError(f"qwen3-vl-embedding 调用失败: {resp.message}")

                return resp.output["embeddings"][0]["embedding"]
            except Exception as e:
                if attempt < retry - 1:
                    wait = 2 ** attempt
                    logger.warning(f"[MultimodalEmbed] 调用失败，{wait}s 后重试 ({attempt+1}/{retry}): {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"[MultimodalEmbed] 最终失败: {e}")
                    raise
        return []

    def embed_text(self, text: str, dimension: int = None) -> List[float]:
        """
        使用 qwen3-vl-embedding 生成文本向量。

        Args:
            text: 文本内容
            dimension: 向量维度，默认使用配置值

        Returns:
            文本向量
        """
        if not DASHSCOPE_AVAILABLE:
            raise RuntimeError("dashscope not installed, cannot embed text")

        dimension = dimension or self._embedding_dim
        return self._call_multimodal_embedding([{"text": text}], dimension)

    def embed_texts(self, texts: List[str], dimension: int = None) -> List[List[float]]:
        """
        批量生成文本向量。

        qwen3-vl-embedding 不支持批量文本，逐条调用。

        Args:
            texts: 文本列表
            dimension: 向量维度

        Returns:
            文本向量列表
        """
        if not texts:
            return []
        dimension = dimension or self._embedding_dim
        return [self.embed_text(t, dimension) for t in texts]

    def embed_image(self, image_data: bytes, dimension: int = None) -> Optional[List[float]]:
        """
        使用 qwen3-vl-embedding 生成图片向量。

        Args:
            image_data: 图片的字节数据
            dimension: 向量维度，默认使用配置值

        Returns:
            图片向量，或 None（如果失败）
        """
        if not DASHSCOPE_AVAILABLE:
            logger.warning("dashscope not installed, skipping image embedding")
            return None

        dimension = dimension or self._embedding_dim

        try:
            # 检测图片 MIME 类型
            mime_type = self._detect_image_mime_type(image_data)
            b64_data = base64.b64encode(image_data).decode("utf-8")

            # DashScope 期望 data:image/xxx;base64,xxxxx 格式
            image_input = f"data:{mime_type};base64,{b64_data}"

            logger.info(f"[EmbedService] calling _call_multimodal_embedding, mime_type={mime_type}, total_len={len(image_input)}")

            result = self._call_multimodal_embedding([{"image": image_input}], dimension)
            logger.info(f"[EmbedService] embedding successful, vector_dim={len(result) if result else 0}")
            return result
        except Exception as e:
            logger.error(f"[EmbedService] 图片向量化最终失败: {type(e).__name__}: {e}")
            return None

    def embed_images(
        self,
        images: List[bytes],
        dimension: int = None,
    ) -> List[Optional[List[float]]]:
        """
        批量生成图片向量。

        Args:
            images: 图片字节数据列表
            dimension: 向量维度

        Returns:
            图片向量列表
        """
        return [self.embed_image(img, dimension) for img in images]

    def get_embeddings(self, texts: List[str]) -> list[list[float]]:
        """
        批量生成文本向量（兼容旧接口）。

        内部调用 embed_texts。

        Args:
            texts: 待转换的文本列表

        Returns:
            向量列表，每个向量是 float 列表
        """
        return self.embed_texts(texts)

    def tokenize(self, text: str) -> list[str]:
        """
        简单分词器 - 支持中英文混合。

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        text = text.lower()

        tokens = []
        chinese_pattern = re.compile(r'[一-鿿]')
        english_pattern = re.compile(r'[a-zA-Z]+')

        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1

        return tokens

    def fit_corpus(self, texts: list[str]):
        """
        拟合语料库，计算 IDF 和平均文档长度。

        Args:
            texts: 文档列表
        """
        self._total_docs = len(texts)
        total_len = 0

        for text in texts:
            tokens = self.tokenize(text)
            total_len += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freq[token] += 1
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1

        self._avg_doc_len = total_len / self._total_docs if self._total_docs > 0 else 1

    def get_sparse_embedding(self, text: str) -> dict:
        """
        生成 BM25 稀疏向量。

        Args:
            text: 输入文本

        Returns:
            稀疏向量 {index: value, ...}
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)

        sparse_vector = {}

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1

            idx = self._vocab[token]

            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((self._total_docs + 1) / 1)
            else:
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

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

    def get_all_embeddings(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[dict]]:
        """
        同时生成密集向量和稀疏向量。

        用于混合检索场景。

        Args:
            texts: 文本列表

        Returns:
            (密集向量列表, 稀疏向量列表)
        """
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings

    def get_zero_vector(self, dim: int = None) -> list[float]:
        """
        生成零向量。

        用于图片向量生成失败时的降级处理。

        Args:
            dim: 向量维度，默认使用配置值

        Returns:
            全零向量
        """
        dim = dim or self._embedding_dim
        return [0.0] * dim

    def _detect_image_mime_type(self, image_data: bytes) -> str:
        """
        根据图片字节数据检测 MIME 类型。

        Args:
            image_data: 图片字节数据

        Returns:
            MIME 类型字符串，如 "image/png", "image/jpeg", "image/gif", "image/webp"
        """
        if not image_data or len(image_data) < 4:
            return "image/png"  # 默认值

        # 检测常见图片格式的魔数（文件头）
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        # JPEG: FF D8 FF
        if image_data[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        # GIF: 47 49 46 38 39|37 61 (GIF89a or GIF87a)
        if image_data[:6] in (b'GIF89a', b'GIF87a'):
            return "image/gif"
        # WebP: 52 49 46 46 ?? ?? ?? ?? 57 45 42 50 (RIFF....WEBP)
        if image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            return "image/webp"
        # BMP: 42 4D (BM)
        if image_data[:2] == b'BM':
            return "image/bmp"

        return "image/png"  # 默认值
