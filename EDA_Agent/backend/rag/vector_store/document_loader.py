"""
文档加载和分片模块 - 支持 PDF、Word、Excel 文档的多模态解析

主要功能：
1. 文档加载：支持 PDF、Word、Excel 格式
2. 多模态解析：PDF/DOCX 支持文本与图片提取，图片以占位符形式嵌入
3. 三层分块：使用滑动窗口实现 L1/L2/L3 三层分块
4. 层级关系维护：维护 chunk_id、parent_chunk_id、root_chunk_id 关系

占位符格式：<<IMAGE:xxxxxxxx>>（8位十六进制 token）
- 解析阶段：嵌入到 chunk content
- 向量化前：剥离占位符生成 clean_text，避免污染文本向量

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
from typing import Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .doc_image_parser import (
    MultimodalParser,
    extract_placeholders,
    clean_content,
)


class DocumentLoader:
    """
    文档加载和三层分片服务 - 支持多模态解析。

    使用 RecursiveCharacterTextSplitter 实现重叠滑动窗口分块，
    生成 L1/L2/L3 三层分块，形成树状层级关系。
    对于 PDF/DOCX，图片以占位符形式嵌入到文本中。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档加载器。

        Args:
            chunk_size: 基础分块大小（用于计算 L3 大小）
            chunk_overlap: 基础重叠大小（用于计算各层重叠）
        """
        # 根据基础参数计算各层分块大小
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)

        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)

        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

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

    def _is_multimodal_file(self, filename: str) -> bool:
        """判断是否为支持多模态的文件类型"""
        ext = filename.lower()
        return ext.endswith(".pdf") or ext.endswith((".docx", ".doc"))

    def _build_base_doc(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        page_number: int = 0,
    ) -> Dict:
        """构建基础文档信息字典"""
        return {
            "filename": filename,
            "file_path": file_path,
            "file_type": file_type,
            "page_number": page_number,
        }

    def _process_multimodal_file(
        self,
        file_path: str,
        filename: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        处理多模态文件（PDF/DOCX），提取文本和图片。

        Returns:
            (chunks, image_assets)
            - chunks: 带占位符的三层分块
            - image_assets: 图片资产列表
        """
        file_type = "PDF" if filename.lower().endswith(".pdf") else "Word"

        # 使用 MultimodalParser 解析
        parser = MultimodalParser(file_path, filename)
        parser.parse()

        initial_chunks, image_assets = parser.build_chunks_with_placeholders(
            text_splitter=self._splitter_level_3,  # 基础分块用 L3 splitter
            chunk_size=300,
            chunk_overlap=50,
        )

        # 补齐基础元数据，供后续三层扩展与写入使用
        for chunk in initial_chunks:
            chunk.setdefault("filename", filename)
            chunk.setdefault("file_path", file_path)
            chunk.setdefault("file_type", file_type)

        chunks, linked_assets = self._extend_chunks_to_three_levels(initial_chunks, image_assets)

        return chunks, linked_assets

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> Tuple[List[Dict], int]:
        """
        将一页文档内容分解为三层分块。

        Args:
            text: 该页的文本内容（不含图片占位符）
            base_doc: 基础文档信息
            page_global_chunk_idx: 全局分块索引

        Returns:
            (chunks, new_global_idx)
        """
        if not text:
            return [], page_global_chunk_idx

        chunks = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue

            level_1_tokens = extract_placeholders(level_1_text)

            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
                "has_image": len(level_1_tokens) > 0,
                "image_count": len(level_1_tokens),
                "image_tokens": level_1_tokens,
                "clean_text": clean_content(level_1_text),
            }
            page_global_chunk_idx += 1
            chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue

                level_2_tokens = extract_placeholders(level_2_text)

                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                    "has_image": len(level_2_tokens) > 0,
                    "image_count": len(level_2_tokens),
                    "image_tokens": level_2_tokens,
                    "clean_text": clean_content(level_2_text),
                }
                page_global_chunk_idx += 1
                chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue

                    level_3_tokens = extract_placeholders(level_3_text)

                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1

                    chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                        "has_image": len(level_3_tokens) > 0,
                        "image_count": len(level_3_tokens),
                        "image_tokens": level_3_tokens,
                        "clean_text": clean_content(level_3_text),
                    })
                    page_global_chunk_idx += 1

        return chunks, page_global_chunk_idx

    def _extend_chunks_to_three_levels(
        self,
        initial_chunks: List[Dict],
        image_assets: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        将 multimodal parser 返回的初始 chunks 扩展为三层结构。

        Args:
            initial_chunks: multimodal parser 返回的 L3 chunks（带占位符）
            image_assets: 图片资产列表

        Returns:
            (all_chunks, image_assets)
        """
        if not initial_chunks:
            return [], []

        # 按页码分组处理
        page_chunks: Dict[int, List[Dict]] = {}
        for chunk in initial_chunks:
            page = chunk.get("page_number", 1)
            if page not in page_chunks:
                page_chunks[page] = []
            page_chunks[page].append(chunk)

        all_chunks = []
        all_image_assets = []
        page_global_chunk_idx = 0

        for page_num in sorted(page_chunks.keys()):
            page_items = page_chunks[page_num]

            # 合并同一页的所有文本
            combined_text = ""
            base_doc = self._build_base_doc(
                filename=page_items[0]["filename"],
                file_path=page_items[0]["file_path"],
                file_type=page_items[0]["file_type"],
                page_number=page_num,
            )

            # 收集该页的占位符和图片资产
            page_placeholders = []
            page_image_assets = []

            for chunk in page_items:
                combined_text += clean_content(chunk["text"]) + "\n\n"
                tokens = extract_placeholders(chunk["text"])
                page_placeholders.extend(tokens)

            file_type = page_items[0].get("file_type")
            if not file_type:
                file_type = "PDF" if str(page_items[0].get("filename", "")).lower().endswith(".pdf") else "Word"

            # 为该页创建三层分块
            chunks, page_global_chunk_idx = self._split_page_to_three_levels(
                text=combined_text.strip(),
                base_doc={
                    **base_doc,
                    "file_type": file_type,
                },
                page_global_chunk_idx=page_global_chunk_idx,
            )

            # 将图片占位符分配给最接近的 chunk
            # 简单策略：追加到每页最后一个 chunk
            if page_placeholders and chunks:
                last_chunk = chunks[-1]
                placeholder_str = "\n".join([f"<<IMAGE:{t}>>" for t in page_placeholders])
                last_chunk["text"] = last_chunk["text"] + "\n" + placeholder_str
                final_tokens = extract_placeholders(last_chunk["text"])
                last_chunk["has_image"] = len(final_tokens) > 0
                last_chunk["image_count"] = len(final_tokens)
                last_chunk["image_tokens"] = final_tokens
                last_chunk["clean_text"] = clean_content(last_chunk["text"])

                # 关联图片资产
                for asset in image_assets:
                    if asset["image_token"] in page_placeholders:
                        asset["chunk_id"] = last_chunk["chunk_id"]
                        asset["filename"] = base_doc["filename"]
                        asset["file_path"] = base_doc["file_path"]
                        asset["file_type"] = file_type
                        page_image_assets.append(asset)

            all_chunks.extend(chunks)
            all_image_assets.extend(page_image_assets)

        return all_chunks, all_image_assets

    def load_document(self, file_path: str, filename: str) -> Tuple[List[Dict], List[Dict]]:
        """
        加载单个文档并分片。

        支持的文件类型：
        - PDF: 使用 fitz（PyMuPDF）解析文本与图片
        - Word (.doc/.docx): 使用 DocxDocument 解析文本与图片
        - Excel (.xls/.xlsx): 使用纯文本模式

        Args:
            file_path: 文件在服务器上的路径
            filename: 原始文件名

        Returns:
            (chunks, image_assets)
            - chunks: 三层分块列表
            - image_assets: 图片资产列表（仅 PDF/DOCX 有值）
        """
        file_lower = filename.lower()

        if file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc")):
            # 多模态文件处理
            return self._process_multimodal_file(file_path, filename)
        elif file_lower.endswith((".xlsx", ".xls")):
            # Excel 保持纯文本处理
            return self._load_excel_document(file_path, filename, "Excel")
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

    def _load_excel_document(
        self,
        file_path: str,
        filename: str,
        doc_type: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """加载 Excel 文档（纯文本模式）"""
        from langchain_community.document_loaders import UnstructuredExcelLoader

        loader = UnstructuredExcelLoader(file_path)
        try:
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0

            for doc_idx, doc in enumerate(raw_docs):
                base_doc = self._build_base_doc(
                    filename=filename,
                    file_path=file_path,
                    file_type=doc_type,
                    page_number=doc_idx + 1,
                )
                text = (doc.page_content or "").strip()
                if not text:
                    continue

                chunks, page_global_chunk_idx = self._split_page_to_three_levels(
                    text=text,
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                documents.extend(chunks)

            return documents, []

        except Exception as e:
            raise Exception(f"处理 Excel 文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        从文件夹加载所有支持的文档并分片。

        Args:
            folder_path: 文件夹路径

        Returns:
            (all_chunks, all_image_assets)
        """
        all_chunks = []
        all_image_assets = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()

            if not (
                file_lower.endswith(".pdf") or
                file_lower.endswith((".docx", ".doc")) or
                file_lower.endswith((".xlsx", ".xls"))
            ):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                chunks, image_assets = self.load_document(file_path, filename)
                all_chunks.extend(chunks)
                all_image_assets.extend(image_assets)
            except Exception:
                continue

        return all_chunks, all_image_assets