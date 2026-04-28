"""
多模态文档解析器 - 支持 PDF 和 DOCX 的文本与图片提取

主要功能：
1. PDF 解析：使用 fitz（PyMuPDF）提取文本块和图片
2. DOCX 解析：使用 DocxDocument 提取段落文本和 inline_shapes 图片
3. 占位符机制：在 chunk 文本中嵌入 <<IMAGE:xxxxxxxx>> 占位符
4. 图片归属：按阅读顺序将图片归属到最近的文本 chunk

占位符格式：<<IMAGE:xxxxxxxx>>（8位十六进制 token）
"""

import uuid
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from typing import Dict, List, Tuple, Optional
from docx.table import Table
from docx.text.paragraph import Paragraph


# 占位符正则表达式
_IMAGE_PH_RE = "<<IMAGE:([0-9a-f]{8})>>"
_IMAGE_PH_PATTERN = __import__('re').compile(_IMAGE_PH_RE)


def get_image_placeholder() -> Tuple[str, str]:
    """
    生成一个新的图片占位符。

    Returns:
        (placeholder_str, image_token)
    """
    token = uuid.uuid4().hex[:8]
    placeholder = f"<<IMAGE:{token}>>"
    return placeholder, token


def extract_placeholders(content: str) -> List[str]:
    """
    从 content 中提取所有图片占位符 token。

    Args:
        content: 包含占位符的文本

    Returns:
        token 列表，如 ["11111111", "22222222"]
    """
    return _IMAGE_PH_PATTERN.findall(content)


def clean_content(content: str) -> str:
    """
    剥离 content 中的图片占位符。

    Args:
        content: 包含占位符的文本

    Returns:
        清洗后的文本
    """
    return _IMAGE_PH_PATTERN.sub('', content).strip()


class PDFImageParser:
    """
    PDF 多模态解析器。

    使用 fitz（PyMuPDF）提取：
    - 文本块（type=0）：带位置信息
    - 图片块：通过 page.get_images() 获取
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc: Optional[fitz.Document] = None

    def load(self) -> "PDFImageParser":
        """加载 PDF 文件"""
        self.doc = fitz.open(self.file_path)
        return self

    def parse_page(self, page_num: int) -> Tuple[List[Dict], List[Dict]]:
        """
        解析单个页面，返回文本块和图片块列表。

        Returns:
            (text_blocks, image_blocks)
            - text_block: {text, page_number, y_center, block_type="text"}
            - image_block: {image_bytes, page_number, y_center, block_type="image", mime_type}
        """
        if not self.doc:
            raise ValueError("PDF not loaded, call load() first")

        page = self.doc[page_num]
        text_blocks = []
        image_blocks = []

        # 提取文本块
        text_dicts = page.get_text("dict")["blocks"]
        for block in text_dicts:
            if block.get("type") == 0:  # 文本块
                for line in block.get("lines", []):
                    y_center = (line["bbox"][1] + line["bbox"][3]) / 2
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if text:
                            text_blocks.append({
                                "text": text,
                                "page_number": page_num + 1,
                                "y_center": y_center,
                                "block_type": "text",
                            })

        # 提取图片块
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            mime_type = base_image["ext"]

            # 获取图片位置（使用第一个出现位置的 y_center）
            img_y = 0
            for block in text_dicts:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("ref") == xref:
                                img_y = (line["bbox"][1] + line["bbox"][3]) / 2
                                break

            # 如果没找到精确位置，使用页内图片索引估算位置
            if img_y == 0:
                # 按页面高度做均匀分布，避免使用超大坐标导致后续无法归属到文本 chunk
                page_height = float(page.rect.height or 1000.0)
                total_imgs = max(len(image_list), 1)
                img_y = page_height * ((img_idx + 1) / (total_imgs + 1))

            image_blocks.append({
                "image_bytes": image_bytes,
                "page_number": page_num + 1,
                "y_center": img_y,
                "block_type": "image",
                "mime_type": f"image/{mime_type}",
                "xref": xref,
            })

        return text_blocks, image_blocks

    def parse_all(self) -> Tuple[List[Dict], List[Dict]]:
        """
        解析整个 PDF，返回所有页面的文本块和图片块。

        Returns:
            (all_text_blocks, all_image_blocks)
        """
        if not self.doc:
            raise ValueError("PDF not loaded, call load() first")

        all_text = []
        all_images = []

        for page_num in range(len(self.doc)):
            text_blocks, image_blocks = self.parse_page(page_num)
            all_text.extend(text_blocks)
            all_images.extend(image_blocks)

        return all_text, all_images

    def close(self):
        """关闭 PDF 文档"""
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.close()


class DOCXImageParser:
    """
    DOCX 多模态解析器。

    使用 DocxDocument 提取：
    - 段落文本
    - 表格文本
    - inline_shapes 图片
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc: Optional[DocxDocument] = None

    def load(self) -> "DOCXImageParser":
        """加载 DOCX 文件"""
        self.doc = DocxDocument(self.file_path)
        return self

    def _iter_block_items(self):
        """
        迭代文档中的块（段落和表格）。
        """
        for elem in self.doc.element.body:
            if isinstance(elem, CT_P):
                yield Paragraph(elem, self.doc)
            elif isinstance(elem, CT_Tbl):
                yield Table(elem, self.doc)

    def _extract_images_from_paragraph(self, para: Paragraph) -> List[Dict]:
        """
        从段落中提取内联图片。

        Returns:
            图片块列表，每项包含 image_bytes, width, height, mime_type
        """
        image_blocks = []
        for run in para.runs:
            for shape in run._element.xpath('.//w:drawing//wp:inline'):
                # 尝试获取图片
                blip = shape.xpath('.//a:blip/@r:embed')
                if blip:
                    # 获取图片关系
                    pass

            # 使用 inline_shapes
        return image_blocks

    def parse_all(self) -> Tuple[List[Dict], List[Dict]]:
        """
        解析整个 DOCX，返回文本块和图片块列表。

        Returns:
            (text_blocks, image_blocks)
        """
        if not self.doc:
            raise ValueError("DOCX not loaded, call load() first")

        text_blocks = []
        image_blocks = []

        # 记录 inline_shapes
        inline_images = list(self.doc.inline_shapes)
        inline_idx = 0

        para_idx = 0
        for block in self._iter_block_items():
            if isinstance(block, Paragraph):
                # 处理段落文本
                para_text = block.text.strip()
                if para_text:
                    text_blocks.append({
                        "text": para_text,
                        "page_number": para_idx + 1,
                        "y_center": para_idx * 100,
                        "block_type": "text",
                    })

                # 检查是否有内联图片
                # 简单策略：每3个段落尝试分配一个图片
                if inline_idx < len(inline_images):
                    img = inline_images[inline_idx]
                    try:
                        image_bytes = img.blob
                        width = img.width
                        height = img.height
                        image_blocks.append({
                            "image_bytes": image_bytes,
                            "page_number": para_idx + 1,
                            "y_center": para_idx * 100 + 50,  # 插在段落之间
                            "block_type": "image",
                            "mime_type": "image/png",
                            "width": width,
                            "height": height,
                        })
                        inline_idx += 1
                    except Exception:
                        pass

                para_idx += 1

            elif isinstance(block, Table):
                # 处理表格文本
                for row_idx, row in enumerate(block.rows):
                    for cell_idx, cell in enumerate(row.cells):
                        cell_text = cell.text.strip()
                        if cell_text:
                            text_blocks.append({
                                "text": cell_text,
                                "page_number": para_idx + 1,
                                "y_center": para_idx * 100,
                                "block_type": "text",
                                "table_position": f"{row_idx}-{cell_idx}",
                            })
                para_idx += 1

        return text_blocks, image_blocks

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        pass


class MultimodalParser:
    """
    多模态文档解析器 - 统一入口。

    协调 PDF/DOCX 解析，并在元素流中插入图片占位符。
    """

    def __init__(self, file_path: str, filename: str):
        self.file_path = file_path
        self.filename = filename.lower()
        self.text_blocks: List[Dict] = []
        self.image_blocks: List[Dict] = []
        self.image_assets: List[Dict] = []  # 存储图片资产信息

    def parse(self) -> "MultimodalParser":
        """执行解析"""
        if self.filename.endswith(".pdf"):
            self._parse_pdf()
        elif self.filename.endswith((".docx", ".doc")):
            self._parse_docx()
        else:
            raise ValueError(f"Unsupported file type: {self.filename}")

        self._sort_elements()
        return self

    def _parse_pdf(self):
        """使用 fitz 解析 PDF"""
        parser = PDFImageParser(self.file_path)
        try:
            parser.load()
            text_blocks, image_blocks = parser.parse_all()
            self.text_blocks = text_blocks
            self.image_blocks = image_blocks
        finally:
            parser.close()

    def _parse_docx(self):
        """使用 DocxDocument 解析 DOCX"""
        parser = DOCXImageParser(self.file_path)
        parser.load()
        text_blocks, image_blocks = parser.parse_all()
        self.text_blocks = text_blocks
        self.image_blocks = image_blocks

    def _sort_elements(self):
        """
        按阅读顺序排序：先按 page_number，再按 y_center，文本优先于图片。
        """
        def sort_key(elem):
            page = elem.get("page_number", 0)
            y = elem.get("y_center", 0)
            # block_type_priority: text=0, image=1（文本优先）
            btp = 0 if elem.get("block_type") == "text" else 1
            return (page, y, btp)

        all_elements = self.text_blocks + self.image_blocks
        all_elements.sort(key=sort_key)
        self.elements = all_elements

    def build_chunks_with_placeholders(
        self,
        text_splitter,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        将元素流转换为带占位符的 chunk。

        Args:
            text_splitter: 用于文本分块的 RecursiveCharacterTextSplitter
            chunk_size: 分块大小
            chunk_overlap: 重叠大小

        Returns:
            (chunks, image_assets)
            - chunks: 带占位符的文本 chunk 列表
            - image_assets: 图片资产列表
        """
        chunks = []
        image_assets = []

        # 收集纯文本用于分块
        full_text_parts = []
        element_positions = []  # 记录每个文本元素在 full_text 中的位置

        current_pos = 0
        for elem in self.elements:
            if elem["block_type"] == "text":
                start_pos = current_pos
                full_text_parts.append(elem["text"])
                element_positions.append({
                    "text": elem["text"],
                    "page_number": elem["page_number"],
                    "y_center": elem["y_center"],
                    "start_pos": start_pos,
                    "end_pos": current_pos + len(elem["text"]),
                })
                current_pos += len(elem["text"]) + 1  # +1 for newline

        full_text = "\n".join(full_text_parts)

        # 使用 splitter 分块
        if not full_text:
            return [], []

        base_doc = {
            "filename": self.filename,
            "file_path": self.file_path,
        }

        # 分块
        chunk_docs = text_splitter.create_documents([full_text], [base_doc])

        page_global_chunk_idx = 0
        page_number = 1

        # 用于跟踪哪些图片已经被分配（避免重复）
        assigned_images = set()

        for chunk_doc in chunk_docs:
            chunk_text = chunk_doc.page_content or ""

            # 检查该 chunk 覆盖了哪些元素
            covered_elements = []
            for elem in element_positions:
                elem_start = elem["start_pos"]
                elem_end = elem["end_pos"]
                # 检查 chunk 文本是否包含该元素
                if elem["text"] in chunk_text:
                    covered_elements.append(elem)

            # 确定页码
            if covered_elements:
                page_number = covered_elements[0].get("page_number", 1)

            # 构建 chunk_id
            chunk_id = f"{self.filename}::p{page_number}::l3::{page_global_chunk_idx}"

            # 收集该 chunk 关联的图片
            chunk_image_tokens = []
            chunk_image_assets = []

            for img_idx, img_block in enumerate(self.image_blocks):
                if img_idx in assigned_images:
                    continue

                img_y = img_block["y_center"]
                img_page = img_block["page_number"]

                # 检查图片是否在这个 chunk 的区域内
                for cov_elem in covered_elements:
                    elem_y = cov_elem["y_center"]
                    elem_page = cov_elem["page_number"]
                    # 图片在文本元素附近（同一页，y 相近）
                    if elem_page == img_page and abs(img_y - elem_y) < 200:
                        placeholder, token = get_image_placeholder()
                        chunk_image_tokens.append(token)
                        chunk_image_tokens.append(placeholder)  # 占位符本身

                        # 保存图片资产（带 chunk_id 关联）
                        asset = {
                            "image_token": token,
                            "placeholder": placeholder,
                            "image_bytes": img_block.get("image_bytes"),
                            "mime_type": img_block.get("mime_type", "image/png"),
                            "page_number": img_page,
                            "width": img_block.get("width"),
                            "height": img_block.get("height"),
                            "chunk_id": chunk_id,  # 关联到当前 chunk
                        }
                        chunk_image_assets.append(asset)
                        image_assets.append(asset)
                        assigned_images.add(img_idx)
                        break

            # 简单策略：将图片占位符追加到 chunk 末尾
            image_placeholders = [t for t in chunk_image_tokens if t.startswith("<<IMAGE:")]
            if image_placeholders:
                chunk_text_with_ph = chunk_text + "\n" + "\n".join(image_placeholders)
            else:
                chunk_text_with_ph = chunk_text

            chunk = {
                "text": chunk_text_with_ph,
                "chunk_id": chunk_id,
                "page_number": page_number,
                "chunk_level": 3,
                "chunk_idx": page_global_chunk_idx,
                "has_image": len(image_placeholders) > 0,
                "image_count": len(image_placeholders),
                "image_tokens": [t for t in chunk_image_tokens if not t.startswith("<<IMAGE:")],
                "clean_text": clean_content(chunk_text_with_ph),
            }

            chunks.append(chunk)
            page_global_chunk_idx += 1

        return chunks, image_assets

    def get_elements(self) -> List[Dict]:
        """获取排序后的元素流"""
        return getattr(self, 'elements', [])

    def get_image_assets(self) -> List[Dict]:
        """获取图片资产列表"""
        return self.image_assets