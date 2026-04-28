"""
文档加载和分片模块 - 支持 PDF、Word、Excel 文档

主要功能：
1. 文档加载：支持 PDF、Word、Excel 格式
2. 三层分块：使用滑动窗口实现 L1/L2/L3 三层分块
3. 层级关系维护：维护 chunk_id、parent_chunk_id、root_chunk_id 关系

三层分块策略：
- L1（Level 1）：最大分块（~1200 字符），保留最多上下文
- L2（Level 2）：中等分块（~600 字符），平衡粒度和完整性
- L3（Level 3）：最小分块（~300 字符），最细粒度检索

分块关系：
- L3 的 parent_chunk_id 指向 L2
- L2 的 parent_chunk_id 指向 L1
- L1 是根节点，root_chunk_id 等于自身的 chunk_id

用途：
- L3 用于精确检索
- L2/L1 用于 Auto-merging，将多个相关 L3 合并为更大的上下文
"""

import os
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader


class DocumentLoader:
    """
    文档加载和三层分片服务。

    使用 RecursiveCharacterTextSplitter 实现重叠滑动窗口分块，
    生成 L1/L2/L3 三层分块，形成树状层级关系。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档加载器。

        Args:
            chunk_size: 基础分块大小（用于计算 L3 大小）
            chunk_overlap: 基础重叠大小（用于计算各层重叠）
        """
        # 根据基础参数计算各层分块大小
        # L1 是最大的，保留最多上下文
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)

        # L2 是中等大小
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)

        # L3 是最小的，用于精确检索
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        # 创建各层分块器
        # separators 定义分块时的分割优先级
        separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=separators,
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=separators,
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=separators,
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        """
        构建分块唯一标识符。

        格式：{filename}::p{page_number}::l{level}::{index}
        例如：paper.pdf::p3::l2::5

        Args:
            filename: 文件名
            page_number: 页码
            level: 分块层级（1/2/3）
            index: 该层级内的分块序号

        Returns:
            唯一标识符字符串
        """
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        """
        将一页文档内容分解为三层分块。

        分块过程：
        1. 使用 L1 splitter 将文本分成 L1 块
        2. 对每个 L1 块使用 L2 splitter 分成 L2 块
        3. 对每个 L2 块使用 L3 splitter 分成 L3 块
        4. 维护父子关系：L3.parent=L2, L2.parent=L1, L1.parent=空

        分块关系图示：
        L1 ─┬─ L2 ─┬─ L3
            │       └─ L3
            └─ L2 ─┬─ L3
                   └─ L3

        Args:
            text: 该页的文本内容
            base_doc: 基础文档信息（filename, file_type 等）
            page_global_chunk_idx: 全局分块索引（用于追踪）

        Returns:
            该页的三层分块列表
        """
        if not text:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        # L1 分块
        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue

            # 构建 L1 分块
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",      # L1 是根节点
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            # L2 分块
            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue

                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,  # L2 的父节点是 L1
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                # L3 分块（叶子节点）
                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue

                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1

                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,  # L3 的父节点是 L2
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并分片。

        支持的文件类型：
        - PDF: 使用 PyPDFLoader
        - Word (.doc/.docx): 使用 Docx2txtLoader
        - Excel (.xls/.xlsx): 使用 UnstructuredExcelLoader

        Args:
            file_path: 文件在服务器上的路径
            filename: 原始文件名

        Returns:
            分片后的文档列表，包含三层分块
        """
        file_lower = filename.lower()

        # 根据文件类型选择加载器
        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            # 加载文档
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0

            # 逐页处理
            for doc in raw_docs:
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": doc.metadata.get("page", 0),
                }
                # 将该页分解为三层分块
                page_chunks = self._split_page_to_three_levels(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)

            return documents

        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹加载所有支持的文档并分片。

        Args:
            folder_path: 文件夹路径

        Returns:
            所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()

            # 只处理支持的文档类型
            if not (file_lower.endswith(".pdf") or
                    file_lower.endswith((".docx", ".doc")) or
                    file_lower.endswith((".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                # 单个文件失败不影响其他文件
                continue

        return all_documents