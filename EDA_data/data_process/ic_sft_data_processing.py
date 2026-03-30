#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC Textbook Dataset SFT数据处理脚本
将原始数据集转换为SFT微调格式（使用LLM检测语言）

后台运行方法：
    nohup python sft_data_processing.py > process.log 2>&1 &

查看进度：
    tail -f process.log
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from collections import Counter

from datasets import load_dataset
from openai import OpenAI


# ==================== 配置参数（可通过命令行覆盖）====================

# OpenAI API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "sk-MTIzLTExNzAwNDQwMDgzLTE3NzQ1ODMyMTEwOTc=")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.scnet.cn/api/llm/v1")  # 如果使用第三方API（如硅基流动），设置此项
MODEL = "MiniMax-M2.5"  # 或其他模型如 "gpt-3.5-turbo", "Qwen/Qwen2.5-72B-Instruct"

# 数据集路径
BASE_DATA_DIR = os.getenv("LLM_AGENT_DATA_DIR", "/data/llm_agent/data")
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")
DATASET_NAME = "SeekBCD/IC-Textbook_Dataset"
CACHE_DIR = os.path.join(BASE_DATA_DIR, "IC_Textbook_Dataset")
OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "sft_data/ic_sft_data.jsonl")

# 处理参数
MAX_WORKERS = 5          # 并行线程数
MAX_ITEMS = 4000         # 处理条目数限制（设为None处理全部数据）
SAVE_INTERVAL = 100     # 每多少条保存一次
MIN_TEXT_LENGTH = 50     # 最小文本长度
DETECT_LANG = False      # 是否使用LLM检测语言（默认使用字符特征检测）


# ==================== 语言检测器 ====================

class LanguageDetector:
    """语言检测器（支持LLM或字符特征检测）"""

    LANG_MAP = {
        'Chinese': 'Chinese',
        'English': 'English',
        'Japanese': 'Japanese',
        'Korean': 'Korean',
        'French': 'French',
        'German': 'German',
        'Spanish': 'Spanish',
        'Russian': 'Russian',
        'Italian': 'Italian',
        'Portuguese': 'Portuguese',
    }

    def __init__(self, client: Optional[OpenAI] = None, model: Optional[str] = None, use_llm: bool = False):
        self.client = client
        self.model = model
        self.use_llm = use_llm

    def detect(self, text: str) -> str:
        """检测文本语言"""
        if self.use_llm and self.client is not None:
            return self._detect_with_llm(text)
        else:
            return self._fallback_detect(text)

    def _detect_with_llm(self, text: str) -> str:
        """使用LLM检测文本语言"""
        try:
            sample = text[:500] if len(text) > 500 else text

            prompt = f"""请判断以下文本的主要语言是什么。

文本内容：
{sample}

请只输出以下选项之一（不要有任何其他文字）：
Chinese, English, Japanese, Korean, French, German, Spanish, Russian, Italian, Portuguese

如果文本是中文（简体或繁体），输出：Chinese
如果文本是英文，输出：English
如果文本是日文，输出：Japanese
如果文本是韩文，输出：Korean
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a language detection assistant. Only respond with the language name in English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=20,
                timeout=30
            )

            if response is None or not response.choices:
                return self._fallback_detect(text)

            message = response.choices[0].message
            if message is None:
                return self._fallback_detect(text)

            content = message.content
            if content is None:
                return self._fallback_detect(text)

            lang_result = content.strip()
            lang_result = lang_result.strip().rstrip('.').rstrip('。')

            for std_name in self.LANG_MAP.keys():
                if std_name.lower() in lang_result.lower():
                    return std_name

            return self._fallback_detect(text)

        except Exception as e:
            print(f"Language detection error: {type(e).__name__}: {e}")
            return self._fallback_detect(text)

    def _fallback_detect(self, text: str) -> str:
        """备用语言检测（统计各类字符数量，返回数量最多的语言）"""
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
        # 返回数量最多的语言（如都为0则默认English）
        max_lang = max(counts, key=counts.get)
        if counts[max_lang] == 0:
            return 'English'
        return max_lang

# ==================== Prompt模板 ====================

class PromptTemplates:
    """不同语言的Prompt模板"""

    @classmethod
    def get_prompt(cls, language: str, text: str) -> str:
        prompts = {
            'Chinese': cls._chinese_prompt(text),
            'English': cls._english_prompt(text),
            'Japanese': cls._japanese_prompt(text),
            'Korean': cls._korean_prompt(text),
        }
        return prompts.get(language, cls._english_prompt(text))

    @staticmethod
    def _chinese_prompt(text: str) -> str:
        return f"""请基于以下文本内容，生成一个高质量的问答对用于模型微调。

文本内容：
{text[:3000]}

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
    def _english_prompt(text: str) -> str:
        return f"""Please generate a high-quality question-answer pair based on the following text for model fine-tuning.

Text content:
{text[:3000]}

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

    @staticmethod
    def _japanese_prompt(text: str) -> str:
        return f"""以下のテキスト内容に基づいて、モデルファインチューニング用の高品質なQ&Aペアを生成してください。

テキスト内容：
{text[:3000]}

以下の要件に従って生成してください：
1. **question**: テキスト内容に基づいて具体的で明確な質問を生成してください。
2. **cot** (思考過程): テキストを段階的に分析して答えに至るまでの詳細な思考過程を提供してください。
3. **answer**: テキスト内容に基づいて完全で正確な回答を提供してください。
4. **category**: 以下のカテゴリーから最も適切なものを選んでください："基本概念", "技術原理", "応用シナリオ", "設計方法", "製造プロセス", "性能分析", "デバイス物理", "回路設計", "EDAツール", "その他"

JSON形式で出力し、説明文は含めないでください：
{{
    "question": "生成された質問",
    "cot": "詳細な思考過程...",
    "answer": "回答内容...",
    "category": "カテゴリー"
}}"""

    @staticmethod
    def _korean_prompt(text: str) -> str:
        return f"""다음 텍스트 내용을 바탕으로 모델 미세조정을 위한 고품질 Q&A 쌍을 생성해주세요.

텍스트 내용:
{text[:3000]}

다음 요구사항에 따라 생성해주세요:
1. **question**: 텍스트 내용을 바탕으로 구체적이고 명확한 질문을 생성해주세요.
2. **cot** (사고 과정): 텍스트를 단계적으로 분석하여 답에 이르는 자세한 사고 과정을 제공해주세요.
3. **answer**: 텍스트 내용을 바탕으로 완전하고 정확한 답변을 제공해주세요.
4. **category**: 다음 카테고리 중 가장 적절한 것을 선택해주세요: "기본 개념", "기술 원리", "응용 시나리오", "설계 방법", "제조 공정", "성능 분석", "소자 물리", "회로 설계", "EDA 도구", "기타"

JSON 형식으로 출력하고 설명은 포함하지 마세요：
{{
    "question": "생성된 질문",
    "cot": "자세한 사고 과정...",
    "answer": "답변 내용...",
    "category": "카테고리"
}}"""


# ==================== 数据处理器 ====================

class DataProcessor:
    """数据处理主类"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o", use_llm_detect: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.use_llm_detect = use_llm_detect
        self.language_detector = LanguageDetector(
            client=self.client if use_llm_detect else None,
            model=model if use_llm_detect else None,
            use_llm=use_llm_detect
        )

    def process_single_item(
        self,
        text: str,
        source: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Optional[Dict]:
        try:
            language = self.language_detector.detect(text)
            prompt = PromptTemplates.get_prompt(language, text)

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
                "domain": domain,
                "original_text_preview": text[:500] if len(text) > 500 else text
            }

            if not result["question"] or not result["answer"]:
                print(f"Warning: Missing question or answer")
                return None

            return result

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content: {content[:200] if 'content' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print(f"Error processing item: {e}")
            return None

    @staticmethod
    def _clean_json_response(content: str) -> str:
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content.strip()

    def _get_text_hash(self, text: str, length: int = 200) -> str:
        """获取文本的标识符（用于去重）"""
        import hashlib
        # 取前N个字符的hash作为唯一标识
        sample = text[:length] if len(text) > length else text
        return hashlib.md5(sample.encode('utf-8')).hexdigest()

    def process_dataset(
        self,
        dataset,
        output_path: str,
        max_workers: int = 5,
        max_items: Optional[int] = None,
        save_interval: int = 100,
        min_text_length: int = 50,
        skip_existing: bool = True
    ) -> str:
        """批量处理数据集

        Args:
            skip_existing: 是否跳过已处理的数据（根据文本内容判断）
        """
        results = []
        processed_count = 0
        success_count = 0

        # 检查已处理的数据（根据文本内容）
        processed_hashes = set()
        if skip_existing and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            # 使用 original_text_preview 或 text 字段生成hash
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
        total_items = len(dataset) if max_items is None else min(max_items, len(dataset))

        print(f"[{datetime.now()}] 开始处理 {total_items} 条数据...")
        if len(processed_hashes) > 0:
            print(f"将根据内容跳过 {len(processed_hashes)} 条已处理数据")
        print(f"语言检测方式: {'LLM' if self.use_llm_detect else '字符特征'}")
        print(f"输出文件: {output_path}")
        print(f"并行工作线程: {max_workers}")

        print("\n第一步：检测所有文本的语言...")
        languages = []
        items_with_lang = []
        skipped_count = 0

        # 直接使用 dataset，不再创建 items_to_process
        data_iterator = dataset.select(range(total_items)) if max_items else dataset

        for idx, item in enumerate(tqdm(data_iterator, desc="Language Detection")):
            text = item.get('text', '')
            source = item.get('source')
            domain = item.get('domain')

            if not text or len(text.strip()) < min_text_length:
                continue

            # 检查是否已处理过（根据内容）
            if len(processed_hashes) > 0:
                text_hash = self._get_text_hash(text)
                if text_hash in processed_hashes:
                    skipped_count += 1
                    continue

            language = self.language_detector.detect(text)
            languages.append(language)
            items_with_lang.append((idx, text, source, domain, language))

        print(f"\n语言检测完成，共 {len(items_with_lang)} 条有效数据（跳过 {skipped_count} 条已处理）")
        if len(items_with_lang) == 0:
            print("没有需要处理的新数据")
            return output_path
        print("语言分布:")
        for lang, count in Counter(languages).most_common():
            print(f"  {lang}: {count}")

        print("\n第二步：并行生成QA对...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for idx, text, source, domain, language in items_with_lang:
                future = executor.submit(
                    self._generate_qa, text, source, domain, language
                )
                future_to_item[future] = (idx, text, source, domain, language)

            with tqdm(total=len(future_to_item), desc="Generating QA") as pbar:
                for future in as_completed(future_to_item):
                    idx, text, source, domain, language = future_to_item[future]
                    try:
                        result = future.result(timeout=120)
                        if result:
                            results.append(result)
                            success_count += 1

                            if len(results) % save_interval == 0:
                                self._append_to_jsonl(results, output_path)
                                results = []
                    except Exception as e:
                        print(f"Error in future {idx}: {e}")

                    processed_count += 1
                    pbar.update(1)

        if results:
            self._append_to_jsonl(results, output_path)

        print(f"\n[{datetime.now()}] 处理完成!")
        print(f"总条目: {total_items}")
        print(f"成功处理: {success_count}")
        print(f"失败/跳过: {total_items - success_count}")
        print(f"输出文件: {output_path}")

        return output_path

    def _generate_qa(
        self,
        text: str,
        source: Optional[str],
        domain: Optional[str],
        language: str
    ) -> Optional[Dict]:
        try:
            prompt = PromptTemplates.get_prompt(language, text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in generating high-quality training data for fine-tuning language models. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            print(f"Generated response: {response.choices[0].message.content[:200] if response and response.choices else 'N/A'}")

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
                "domain": domain,
                "original_text_preview": text[:500] if len(text) > 500 else text
            }

            if not result["question"] or not result["answer"]:
                print(f"Warning: Missing question or answer")
                return None

            return result

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Error generating QA: {e}")
            return None

    @staticmethod
    def _append_to_jsonl(data: List[Dict], filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='IC Textbook Dataset SFT数据处理')
    parser.add_argument('--max-items', type=int, default=MAX_ITEMS, help='处理条目数限制')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='并行线程数')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='输出文件路径')
    parser.add_argument('--test', action='store_true', help='测试模式：只处理1条数据')
    parser.add_argument('--detect-lang', action='store_true', help='使用LLM检测语言（默认使用字符特征）')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的数据（默认会跳过）')
    args = parser.parse_args()

    # 禁用代理
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

    print("=" * 60)
    print("IC Textbook Dataset SFT数据处理")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print(f"API Key: {'已设置' if API_KEY != 'your-api-key-here' else '未设置'}")
    print(f"Model: {MODEL}")
    print(f"Max Items: {args.max_items if args.max_items else '全部'}")
    print(f"Max Workers: {args.max_workers}")
    print(f"语言检测: {'LLM' if args.detect_lang else '字符特征'}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # 加载数据集
    print("\n正在加载数据集...")
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR, split="train")
    print(f"数据集加载完成，共 {len(dataset)} 条数据")

    # 初始化处理器
    processor = DataProcessor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        use_llm_detect=args.detect_lang
    )

    if args.test:
        # 测试模式：处理第一条数据
        print("\n[测试模式] 处理第一条数据...")
        test_item = dataset[0]
        test_result = processor.process_single_item(
            text=test_item['text'],
            source=test_item.get('source'),
            domain=test_item.get('domain')
        )
        if test_result:
            print("\n处理结果:")
            print(json.dumps(test_result, indent=2, ensure_ascii=False))
        else:
            print("处理失败")
    else:
        # 批量处理
        processor.process_dataset(
            dataset=dataset,
            output_path=args.output,
            max_workers=args.max_workers,
            max_items=args.max_items,
            save_interval=SAVE_INTERVAL,
            min_text_length=MIN_TEXT_LENGTH,
            skip_existing=not args.no_skip
        )

        # 验证输出
        print("\n验证输出文件...")
        output_data = []
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                output_data.append(json.loads(line))

        print(f"共生成 {len(output_data)} 条SFT数据")

        # 统计
        languages = [item['language'] for item in output_data]
        categories = [item['category'] for item in output_data]

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
