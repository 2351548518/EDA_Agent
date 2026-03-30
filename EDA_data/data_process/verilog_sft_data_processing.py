#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verilog Dataset SFT数据处理脚本
将MG-Verilog数据集转换为SFT微调格式

后台运行方法：
    nohup python verilog_sft_data_processing.py > process.log 2>&1 &

查看进度：
    tail -f process.log
"""

import os
import sys
import json
import re
import argparse
import hashlib
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from collections import Counter

from datasets import Dataset
from openai import OpenAI


# ==================== 配置参数（可通过命令行覆盖）====================

# OpenAI API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "sk-MTIzLTExNzAwNDQwMDgzLTE3NzQ1ODMyMTEwOTc=")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.scnet.cn/api/llm/v1")
MODEL = "MiniMax-M2.5"

# 数据集路径
BASE_DATA_DIR = os.getenv("LLM_AGENT_DATA_DIR", "/data/llm_agent/data")
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")
DATASET_PATH = os.path.join(BASE_DATA_DIR, "MG_Verilog/merged_dataset/data-00000-of-00001.arrow")
OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "sft_data/verilog_sft_data.jsonl")

# 处理参数
MAX_WORKERS = 4          # 并行线程数
MAX_ITEMS = None         # 处理条目数限制（设为None处理全部数据）
SAVE_INTERVAL = 10       # 每多少条保存一次
MIN_CODE_LENGTH = 50     # 最小代码长度
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
    def get_prompt(cls, language: str, code: str, description: Dict) -> str:
        prompts = {
            'Chinese': cls._chinese_prompt(code, description),
            'English': cls._english_prompt(code, description),
            'Japanese': cls._japanese_prompt(code, description),
            'Korean': cls._korean_prompt(code, description),
        }
        return prompts.get(language, cls._english_prompt(code, description))

    @staticmethod
    def _chinese_prompt(code: str, description: Dict) -> str:
        block_summary = description.get('block_summary', '')
        detailed_summary = description.get('detailed_global_summary', '')
        high_level_summary = description.get('high_level_global_summary', '')

        return f"""请基于以下Verilog代码和描述，生成一个高质量的问答对用于模型微调。

Verilog代码：
```verilog
{code}
```

模块描述信息：
- 高级概述：{high_level_summary}
- 详细描述：{detailed_summary}
- 模块摘要：{block_summary}

请按以下要求生成：
1. **question**: 生成一个具体、清晰的Verilog设计相关问题，并且question中必须包含完整的Verilog代码。问题可以是关于代码功能、设计原理、信号定义、时序逻辑等。
2. **cot** (思维链): 提供详细的思考过程，说明如何分析代码结构和描述信息来理解设计意图。
3. **answer**: 提供完整、准确的回答。回答应包含对代码的详细解释，必要时可以补充Verilog设计建议和最佳实践。
4. **category**: 从以下类别中选择一个最符合的："模块实例化", "组合逻辑", "时序逻辑", "状态机", "数据通路", "控制逻辑", "接口设计", "参数配置", "测试平台", "其他"

请以JSON格式输出，不要包含任何其他说明文字：
{{
    "question": "问题内容...\\n\\nVerilog代码：\\n```verilog\\n{code}\\n```",
    "cot": "详细的思考过程...",
    "answer": "回答内容...",
    "category": "类别"
}}"""

    @staticmethod
    def _english_prompt(code: str, description: Dict) -> str:
        block_summary = description.get('block_summary', '')
        detailed_summary = description.get('detailed_global_summary', '')
        high_level_summary = description.get('high_level_global_summary', '')

        return f"""Please generate a high-quality question-answer pair based on the following Verilog code and description for model fine-tuning.

Verilog Code:
```verilog
{code}
```

Module Description:
- High-level Overview: {high_level_summary}
- Detailed Description: {detailed_summary}
- Block Summary: {block_summary}

Please generate according to the following requirements:
1. **question**: Generate a specific, clear question about Verilog design based on the code and description. The question MUST include the complete Verilog code. The question can be about code functionality, design principles, signal definitions, timing logic, etc.
2. **cot** (Chain of Thought): Provide a detailed reasoning process, explaining how to analyze the code structure and description to understand the design intent.
3. **answer**: Provide a complete and accurate answer. The answer should include a detailed explanation of the code, and can supplement with Verilog design suggestions and best practices if necessary.
4. **category**: Choose the most appropriate one from the following categories: "Module Instantiation", "Combinational Logic", "Sequential Logic", "State Machine", "Data Path", "Control Logic", "Interface Design", "Parameter Configuration", "Testbench", "Others"

Please output in JSON format without any other explanatory text:
{{
    "question": "Question content...\\n\\nVerilog Code:\\n```verilog\\n{code}\\n```",
    "cot": "Detailed reasoning process...",
    "answer": "Answer content...",
    "category": "Category"
}}"""

    @staticmethod
    def _japanese_prompt(code: str, description: Dict) -> str:
        block_summary = description.get('block_summary', '')
        detailed_summary = description.get('detailed_global_summary', '')
        high_level_summary = description.get('high_level_global_summary', '')

        return f"""以下のVerilogコードと説明に基づいて、モデルファインチューニング用の高品質なQ&Aペアを生成してください。

Verilogコード：
```verilog
{code}
```

モジュール説明：
- 高レベル概要：{high_level_summary}
- 詳細説明：{detailed_summary}
- ブロック概要：{block_summary}

以下の要件に従って生成してください：
1. **question**: コードと説明に基づいて、Verilog設計に関する具体的で明確な質問を生成してください。質問には必ず完全なVerilogコードを含めてください。
2. **cot** (思考過程): コード構造と説明を分析して設計意図を理解するための詳細な思考過程を提供してください。
3. **answer**: 完全で正確な回答を提供してください。
4. **category**: 以下のカテゴリーから最も適切なものを選んでください："モジュールインスタンス化", "組み合わせ論理", "順序論理", "状態機械", "データパス", "制御論理", "インターフェース設計", "パラメータ設定", "テストベンチ", "その他"

JSON形式で出力し、説明文は含めないでください：
{{
    "question": "質問内容...\\n\\nVerilogコード：\\n```verilog\\n{code}\\n```",
    "cot": "詳細な思考過程...",
    "answer": "回答内容...",
    "category": "カテゴリー"
}}"""

    @staticmethod
    def _korean_prompt(code: str, description: Dict) -> str:
        block_summary = description.get('block_summary', '')
        detailed_summary = description.get('detailed_global_summary', '')
        high_level_summary = description.get('high_level_global_summary', '')

        return f"""다음 Verilog 코드와 설명을 바탕으로 모델 미세조정을 위한 고품질 Q&A 쌍을 생성해주세요.

Verilog 코드：
```verilog
{code}
```

모듈 설명：
- 고수준 개요：{high_level_summary}
- 상세 설명：{detailed_summary}
- 블록 요약：{block_summary}

다음 요구사항에 따라 생성해주세요:
1. **question**: 코드와 설명을 바탕으로 Verilog 설계에 관한 구체적이고 명확한 질문을 생성해주세요. 질문에는 반드시 완전한 Verilog 코드를 포함해야 합니다.
2. **cot** (사고 과정): 코드 구조와 설명을 분석하여 설계 의도를 이해하는 자세한 사고 과정을 제공해주세요.
3. **answer**: 완전하고 정확한 답변을 제공해주세요.
4. **category**: 다음 카테고리 중 가장 적절한 것을 선택해주세요: "모듈 인스턴스화", "조합 논리", "순차 논리", "상태 머신", "데이터 경로", "제어 논리", "인터페이스 설계", "매개변수 구성", "테스트벤치", "기타"

JSON 형식으로 출력하고 설명은 포함하지 마세요：
{{
    "question": "질문 내용...\\n\\nVerilog 코드：\\n```verilog\\n{code}\\n```",
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
        code: str,
        description: Dict,
        idx: int = 0
    ) -> Optional[Dict]:
        """处理单个Verilog代码条目"""
        try:
            # 从description中提取文本进行语言检测
            desc_text = ""
            if isinstance(description, dict):
                desc_text = description.get('detailed_global_summary', '') or description.get('high_level_global_summary', '') or description.get('block_summary', '')

            # 检测语言（基于描述文本）
            language = self.language_detector.detect(desc_text)

            # 获取对应语言的prompt
            prompt = PromptTemplates.get_prompt(language, code, description)

            # 调用LLM生成QA
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in Verilog design and hardware description languages. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # 解析响应
            content = response.choices[0].message.content.strip()
            content = self._clean_json_response(content)
            qa_data = json.loads(content)

            # 构建最终输出格式
            result = {
                "question": qa_data.get("question", ""),
                "cot": qa_data.get("cot", ""),
                "answer": qa_data.get("answer", ""),
                "language": language,
                "category": qa_data.get("category", "其他"),
                "source": "MG-Verilog",
                "domain": "Verilog",
                "original_text_preview": code[:500] if len(code) > 500 else code
            }

            # 验证必要字段
            if not result["question"] or not result["answer"]:
                print(f"Warning: Missing question or answer for item {idx}")
                return None

            return result

        except json.JSONDecodeError as e:
            print(f"JSON decode error for item {idx}: {e}")
            print(f"Raw content: {content[:200] if 'content' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """清理LLM响应中的markdown代码块"""
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return content.strip()

    def _get_text_hash(self, text: str, length: int = 200) -> str:
        """获取文本的标识符（用于去重）"""
        sample = text[:length] if len(text) > length else text
        return hashlib.md5(sample.encode('utf-8')).hexdigest()

    def process_dataset(
        self,
        dataset,
        output_path: str,
        max_workers: int = 5,
        max_items: Optional[int] = None,
        save_interval: int = 100,
        min_code_length: int = 50,
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
                            # 使用 original_text_preview 字段生成hash
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

        print("\n第一步：检测所有代码的语言...")
        languages = []
        items_with_lang = []
        skipped_count = 0

        for idx in tqdm(range(total_items), desc="Language Detection"):
            item = dataset[idx]
            code = item.get('code', '')
            description = item.get('description', {})

            if not code or len(code.strip()) < min_code_length:
                continue

            # 检查是否已处理过（根据内容）
            if len(processed_hashes) > 0:
                text_hash = self._get_text_hash(code)
                if text_hash in processed_hashes:
                    skipped_count += 1
                    continue

            # 从description中提取文本进行语言检测
            desc_text = ""
            if isinstance(description, dict):
                desc_text = description.get('detailed_global_summary', '') or description.get('high_level_global_summary', '') or description.get('block_summary', '')

            language = self.language_detector.detect(desc_text)
            languages.append(language)
            items_with_lang.append((idx, code, description, language))

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
            for idx, code, description, language in items_with_lang:
                future = executor.submit(
                    self._generate_qa, idx, code, description, language
                )
                future_to_item[future] = (idx, code, description, language)

            with tqdm(total=len(future_to_item), desc="Generating QA") as pbar:
                for future in as_completed(future_to_item):
                    idx, code, description, language = future_to_item[future]
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
        idx: int,
        code: str,
        description: Dict,
        language: str
    ) -> Optional[Dict]:
        """生成QA对（用于并行处理）"""
        try:
            prompt = PromptTemplates.get_prompt(language, code, description)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in Verilog design and hardware description languages. Always respond with valid JSON."},
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
                "source": "MG-Verilog",
                "domain": "Verilog",
                "original_text_preview": code[:500] if len(code) > 500 else code
            }

            if not result["question"] or not result["answer"]:
                print(f"Warning: Missing question or answer for item {idx}")
                return None

            return result

        except json.JSONDecodeError as e:
            print(f"JSON decode error for item {idx}: {e}")
            return None
        except Exception as e:
            print(f"Error generating QA for item {idx}: {e}")
            return None

    @staticmethod
    def _append_to_jsonl(data: List[Dict], filepath: str):
        """将数据追加到jsonl文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_verilog_dataset(dataset_path: str):
    """加载Verilog数据集"""
    return Dataset.from_file(dataset_path)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Verilog Dataset SFT数据处理')
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
    print("Verilog Dataset SFT数据处理")
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
    dataset = load_verilog_dataset(DATASET_PATH)
    print(f"数据集加载完成，共 {len(dataset)} 条数据")

    # 查看样本数据结构
    print("\n样本数据结构:")
    sample = dataset[0]
    print(f"Keys: {list(sample.keys())}")
    if 'code' in sample:
        print(f"Code length: {len(sample['code'])}")
    if 'description' in sample:
        print(f"Description keys: {list(sample['description'].keys()) if isinstance(sample['description'], dict) else 'N/A'}")

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
            code=test_item['code'],
            description=test_item.get('description', {}),
            idx=0
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
            min_code_length=MIN_CODE_LENGTH,
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
