#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese SFT 数据处理脚本
从 cleaned_text_dataset.jsonl 读取数据，使用LLM判断是否保留，
然后生成SFT格式的问答对

使用方法：
    python ch_sft_data_process.py --input /path/to/input.jsonl --output /path/to/output.jsonl

后台运行：
    nohup python ch_sft_data_process.py > process.log 2>&1 &
"""

import os
import sys
import json
import re
import argparse
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from collections import Counter

from openai import OpenAI, AsyncOpenAI


# ==================== 配置参数（可通过命令行覆盖）====================

# OpenAI API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "sk-MTIzLTExNzAwNDQwMDgzLTE3NzQ1ODMyMTEwOTc=")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.scnet.cn/api/llm/v1")
MODEL = "MiniMax-M2.5"  # 或其他模型

# 处理参数
MAX_WORKERS = 5          # 并行线程数
SAVE_INTERVAL = 10      # 每多少条保存一次
MIN_TEXT_LENGTH = 50     # 最小文本长度
BATCH_SIZE = 10          # 批处理大小

# 路径配置
BASE_DATA_DIR = os.getenv("LLM_AGENT_DATA_DIR", "/data/llm_agent/data")
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")


# ==================== 数据过滤器 ====================

class DataFilter:
    """使用LLM判断数据是否应该保留"""

    FILTER_PROMPT_TEMPLATE = """请仔细评估以下文本内容，判断是否应该保留作为模型微调的问答对数据。

【评估标准】：
1. 内容质量：是否包含有意义的知识、概念或信息
2. 教育价值：是否适合用于学习或教学
3. 完整性：是否包含完整的句子或段落，不是碎片化的内容
4. 清晰度：语言表达是否清晰、连贯
5. 专业性：是否与电子电路、集成电路、半导体等相关领域有关

【需要排除的情况】：
- 纯目录、页码、版权信息等元数据
- 内容过于简短或无实质信息
- 重复、混乱或无意义的内容
- 广告、版权声明等非技术内容
- HTML标签、表格等格式混乱的内容

【文本内容】：
{text}

请输出以下格式的判断结果（必须严格按格式输出）：
{{
    "should_keep": true/false,
    "reason": "保留/排除的原因，简要说明"
    "suggested_question": "如果保留，建议生成什么问题（可选）"
}}

注意：只输出JSON格式，不要有其他说明文字。"""

    def __init__(self, client: OpenAI, model: str = MODEL):
        self.client = client
        self.model = model

    def should_keep(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        判断是否应该保留这条数据
        返回: (是否保留, 原因, 建议的问题)
        """
        try:
            # 截取前2000个字符作为评估内容
            text_sample = text[:2000] if len(text) > 2000 else text
            prompt = self.FILTER_PROMPT_TEMPLATE.format(text=text_sample)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=60
            )

            content = response.choices[0].message.content.strip()
            content = self._clean_json_response(content)

            result = json.loads(content)

            should_keep = result.get("should_keep", False)
            reason = result.get("reason", "")
            suggested_question = result.get("suggested_question", None)

            return should_keep, reason, suggested_question

        except json.JSONDecodeError as e:
            print(f"JSON decode error in filter: {e}")
            print(f"Raw content: {content[:200] if 'content' in locals() else 'N/A'}")
            return False, f"解析错误: {e}", None
        except Exception as e:
            print(f"Error in filter: {e}")
            return False, f"错误: {e}", None

    @staticmethod
    def _clean_json_response(content: str) -> str:
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content.strip()


# ==================== 语言检测器 ====================

class LanguageDetector:
    """语言检测器（使用字符特征检测）"""

    def detect(self, text: str) -> str:
        """检测文本语言"""
        counts = {
            'Chinese': 0,
            'Japanese': 0,
            'Korean': 0,
            'English': 0
        }
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                counts['Chinese'] += 1
            elif ('\u3040' <= char <= '\u309F') or ('\u30A0' <= char <= '\u30FF'):
                counts['Japanese'] += 1
            elif ('\uAC00' <= char <= '\uD7AF') or ('\u1100' <= char <= '\u11FF'):
                counts['Korean'] += 1
            elif ('A' <= char <= 'Z') or ('a' <= char <= 'z'):
                counts['English'] += 1

        max_lang = max(counts, key=counts.get)
        if counts[max_lang] == 0:
            return 'English'
        return max_lang


# ==================== Prompt模板 ====================

class PromptTemplates:
    """不同语言的Prompt模板"""

    @classmethod
    def get_qa_prompt(cls, language: str, text: str, suggested_question: Optional[str] = None) -> str:
        prompts = {
            'Chinese': cls._chinese_prompt(text, suggested_question),
            'English': cls._english_prompt(text, suggested_question),
        }
        return prompts.get(language, cls._english_prompt(text, suggested_question))

    @staticmethod
    def _chinese_prompt(text: str, suggested_question: Optional[str] = None) -> str:
        suggested = f"\n参考问题：{suggested_question}" if suggested_question else ""
        return f"""请基于以下文本内容，生成一个高质量的问答对用于模型微调。

文本内容：
{text[:3000]}
{suggested}

请按以下要求生成：
1. **question**: 基于文本内容生成一个具体、清晰的问题。问题应该能够测试对文本的理解深度。
2. **cot** (思维链): 提供详细的思考过程，说明如何逐步分析文本并得出答案。
3. **answer**: 基于文本内容提供完整、准确的回答。如果原文回答不够清晰或完整，请适当重写使其更适合作为SFT训练数据。
4. **category**: 从以下类别中选择一个最符合的："基本概念", "技术原理", "应用场景", "设计方法", "制造工艺", "性能分析", "器件物理", "电路设计", "EDA工具", "其他"

请以JSON格式输出，不要包含任何其他说明文字：
{{
    "question": "生成的问题",
    "cot": "详细的思考过程...",
    "answer": "回答内容...",
    "category": "类别"
}}"""

    @staticmethod
    def _english_prompt(text: str, suggested_question: Optional[str] = None) -> str:
        suggested = f"\nSuggested question: {suggested_question}" if suggested_question else ""
        return f"""Please generate a high-quality question-answer pair based on the following text for model fine-tuning.

Text content:
{text[:3000]}
{suggested}

Please generate according to the following requirements:
1. **question**: Generate a specific, clear question based on the text content. The question should test deep understanding of the text.
2. **cot** (Chain of Thought): Provide a detailed reasoning process, explaining how to analyze the text step by step and arrive at the answer.
3. **answer**: Provide a complete and accurate answer based on the text content. If the original answer is not clear or complete, please rewrite it appropriately to make it more suitable as SFT training data.
4. **category**: Choose the most appropriate one from the following categories: "Basic Concepts", "Technical Principles", "Application Scenarios", "Design Methods", "Manufacturing Process", "Performance Analysis", "Device Physics", "Circuit Design", "EDA Tools", "Others"

Please output in JSON format without any other explanatory text:
{{
    "question": "Generated question",
    "cot": "Detailed reasoning process...",
    "answer": "Answer content...",
    "category": "Category"
}}"""


# ==================== 数据处理器 ====================

class SFTDataProcessor:
    """SFT数据处理主类"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = MODEL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.language_detector = LanguageDetector()
        self.data_filter = DataFilter(self.client, model)

    def _clean_json_response(self, content: str) -> str:
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content.strip()

    def _get_text_hash(self, text: str, length: int = 200) -> str:
        """获取文本的标识符（用于去重）"""
        import hashlib
        sample = text[:length] if len(text) > length else text
        return hashlib.md5(sample.encode('utf-8')).hexdigest()

    def filter_and_generate_qa(
        self,
        text: str,
        source: Optional[str],
        section: Optional[str]
    ) -> Optional[Dict]:
        """
        先过滤数据，然后生成QA对
        返回: SFT格式数据 或 None（如果不保留）
        """
        try:
            # 第一步：判断是否应该保留
            should_keep, reason, suggested_question = self.data_filter.should_keep(text)

            if not should_keep:
                print(f"  跳过数据: {reason[:50]}...")
                return None

            # 第二步：检测语言
            language = self.language_detector.detect(text)

            # 第三步：生成QA对
            prompt = PromptTemplates.get_qa_prompt(language, text, suggested_question)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in generating high-quality training data for fine-tuning language models. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()
            content = self._clean_json_response(content)
            qa_data = json.loads(content)

            result = {
                "question": qa_data.get("question", ""),
                "cot": qa_data.get("cot", ""),
                "answer": qa_data.get("answer", ""),
                "language": language,
                "category": qa_data.get("category", "其他"),
                "source": source,
                "domain": section,
                "original_text_preview": text[:500] if len(text) > 500 else text,
                "filter_reason": reason  # 记录保留原因
            }

            if not result["question"] or not result["answer"]:
                print(f"  Warning: Missing question or answer")
                return None

            return result

        except json.JSONDecodeError as e:
            print(f"  JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"  Error processing item: {e}")
            return None

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        max_workers: int = MAX_WORKERS,
        max_items: Optional[int] = None,
        save_interval: int = SAVE_INTERVAL,
        min_text_length: int = MIN_TEXT_LENGTH,
        skip_existing: bool = True
    ) -> str:
        """批量处理数据集"""

        results = []
        processed_count = 0
        success_count = 0
        filter_count = 0

        # 加载输入数据
        print(f"正在加载输入数据: {input_path}")
        input_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    input_data.append(item)
                except:
                    continue

        print(f"共加载 {len(input_data)} 条输入数据")

        # 检查已处理的数据
        processed_hashes = set()
        if skip_existing and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            text_preview = item.get('original_text_preview', '')
                            if text_preview:
                                text_hash = self._get_text_hash(text_preview)
                                processed_hashes.add(text_hash)
                        except:
                            continue
                print(f"发现已有输出文件，包含 {len(processed_hashes)} 条已处理数据")
            except Exception as e:
                print(f"读取已有文件失败: {e}")

        # 确定要处理的数据范围
        total_items = len(input_data) if max_items is None else min(max_items, len(input_data))

        print(f"\n[{datetime.now()}] 开始处理 {total_items} 条数据...")
        print(f"将根据内容跳过 {len(processed_hashes)} 条已处理数据")
        print(f"输出文件: {output_path}")
        print(f"最小文本长度: {min_text_length}")
        print("=" * 60)

        # 处理数据
        items_to_process = input_data[:total_items]

        for idx, item in enumerate(tqdm(items_to_process, desc="Processing")):
            text = item.get('text', '')
            source = item.get('source', '')
            section = item.get('section', '')
            chunk_id = item.get('chunk_id', idx)

            # 检查文本长度
            if not text or len(text.strip()) < min_text_length:
                continue

            # 检查是否已处理过
            if len(processed_hashes) > 0:
                text_hash = self._get_text_hash(text)
                if text_hash in processed_hashes:
                    continue

            # 处理单条数据（过滤 + 生成QA）
            result = self.filter_and_generate_qa(text, source, section)

            if result:
                results.append(result)
                success_count += 1

                # 定期保存
                if len(results) % save_interval == 0:
                    self._append_to_jsonl(results, output_path)
                    results = []
            else:
                filter_count += 1

            processed_count += 1

        # 保存剩余结果
        if results:
            self._append_to_jsonl(results, output_path)

        # 统计信息
        print(f"\n[{datetime.now()}] 处理完成!")
        print(f"总条目: {total_items}")
        print(f"成功生成QA: {success_count}")
        print(f"被过滤: {filter_count}")
        print(f"失败/跳过: {total_items - success_count - filter_count}")
        print(f"输出文件: {output_path}")

        return output_path

    @staticmethod
    def _append_to_jsonl(data: List[Dict], filepath: str):
        """追加数据到jsonl文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Chinese SFT数据处理 - 带数据过滤')
    parser.add_argument('--input', type=str,
                        default=os.path.join(BASE_DATA_DIR, 'pdf/final_dataset/cleaned_text_dataset.jsonl'),
                        help='输入jsonl文件路径')
    parser.add_argument('--output', type=str,
                        default=os.path.join(BASE_OUTPUT_DIR, 'sft_data/ch_sft_data.jsonl'),
                        help='输出jsonl文件路径')
    parser.add_argument('--max-items', type=int, default=None, help='处理条目数限制（默认全部）')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='并行线程数')
    parser.add_argument('--save-interval', type=int, default=SAVE_INTERVAL, help='保存间隔')
    parser.add_argument('--min-text-length', type=int, default=MIN_TEXT_LENGTH, help='最小文本长度')
    parser.add_argument('--test', action='store_true', help='测试模式：只处理1条数据并打印详细信息')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的数据')
    args = parser.parse_args()

    # 禁用代理
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

    print("=" * 60)
    print("Chinese SFT 数据处理 - 带数据过滤")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print(f"API Key: {'已设置' if API_KEY != 'your-api-key-here' else '未设置'}")
    print(f"Model: {MODEL}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"Max Items: {args.max_items if args.max_items else '全部'}")
    print("=" * 60)

    # 初始化处理器
    processor = SFTDataProcessor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )

    if args.test:
        # 测试模式
        print("\n[测试模式] 处理第一条数据...")

        # 加载一条测试数据
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    test_item = json.loads(line.strip())
                    break
                except:
                    continue

        print(f"\n原始数据:")
        print(f"  Source: {test_item.get('source', 'N/A')}")
        print(f"  Section: {test_item.get('section', 'N/A')}")
        print(f"  Text length: {len(test_item.get('text', ''))}")
        print(f"  Text preview: {test_item.get('text', '')[:200]}...")

        print(f"\n开始处理...")
        test_result = processor.filter_and_generate_qa(
            text=test_item.get('text', ''),
            source=test_item.get('source'),
            section=test_item.get('section')
        )

        if test_result:
            print("\n处理结果:")
            print(json.dumps(test_result, indent=2, ensure_ascii=False))
        else:
            print("数据被过滤或处理失败")
    else:
        # 批量处理
        processor.process_dataset(
            input_path=args.input,
            output_path=args.output,
            max_workers=args.max_workers,
            max_items=args.max_items,
            save_interval=args.save_interval,
            min_text_length=args.min_text_length,
            skip_existing=not args.no_skip
        )

        # 验证输出
        if os.path.exists(args.output):
            print("\n验证输出文件...")
            output_data = []
            with open(args.output, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        output_data.append(json.loads(line))
                    except:
                        continue

            print(f"共生成 {len(output_data)} 条SFT数据")

            # 统计
            if output_data:
                languages = [item.get('language', 'Unknown') for item in output_data]
                categories = [item.get('category', 'Unknown') for item in output_data]

                print("\n语言分布:")
                for lang, count in Counter(languages).most_common():
                    print(f"  {lang}: {count}")

                print("\n类别分布:")
                for cat, count in Counter(categories).most_common():
                    print(f"  {cat}: {count}")

    print(f"\n结束时间: {datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
