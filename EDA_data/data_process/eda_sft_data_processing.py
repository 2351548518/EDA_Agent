#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG-EDA Dataset SFT数据处理脚本
将RAG-EDA原始数据集转换为SFT微调格式

后台运行方法：
    nohup python eda_sft_data_processing.py > process.log 2>&1 &

查看进度：
    tail -f process.log
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from collections import Counter

from openai import OpenAI


# ==================== 配置参数 ====================

# OpenAI API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "sk-MTIzLTExNzAwNDQwMDgzLTE3NzQ1ODMyMTEwOTc=")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.scnet.cn/api/llm/v1")
MODEL = "MiniMax-M2.5"

# 数据集路径
BASE_DATA_DIR = os.getenv("LLM_AGENT_DATA_DIR", "/data/llm_agent/data")
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")
INPUT_PATH = os.path.join(BASE_DATA_DIR, "RAG_EDA/RAG-EDA/training_dataset/generator_dataset/QA_finetuning_v1v2amend1.jsonl")
OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "sft_data/eda_sft_data.jsonl")

# 处理参数
MAX_WORKERS = 5          # 并行线程数
MAX_ITEMS = None         # 处理条目数限制（设为None处理全部数据）
SAVE_INTERVAL = 10      # 每多少条保存一次

# 固定字段值
LANGUAGE = "English"
SOURCE = "RAG-EDA"
DOMAIN = "EDA"


# ==================== CoT生成器 ====================

class CoTGenerator:
    """使用LLM生成Chain of Thought推理过程"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_cot(self, question: str, answer: str, reference_content: str = "") -> str:
        """基于问题和答案生成思维链推理过程"""
        try:
            prompt = self._build_prompt(question, answer, reference_content)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates detailed reasoning processes (Chain of Thought) for question-answer pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            print(f"Error generating CoT: {e}")
            # 返回一个简单的默认CoT
            return self._generate_default_cot(question, answer)

    def _build_prompt(self, question: str, answer: str, reference_content: str) -> str:
        """构建生成CoT的prompt"""
        ref_part = f"\n\n参考内容：\n{reference_content[:2000]}" if reference_content else ""

        return f"""请为以下问题和答案生成详细的思维链（Chain of Thought）推理过程。

问题：
{question}

答案：
{answer}
{ref_part}

请生成一个详细的思考过程，说明：
1. 如何理解这个问题
2. 需要分析哪些关键点
3. 如何基于参考内容（如果有）推导出答案
4. 最终的结论是如何得出的

请以第一人称（"我"）的角度描述思考过程，用英文输出。"""

    def _generate_default_cot(self, question: str, answer: str) -> str:
        """生成默认的CoT当LLM调用失败时"""
        return f"""To answer this question, I need to analyze what is being asked. The question is about: {question[:200]}...

Based on the reference content and my knowledge, I can deduce the answer by examining the key aspects mentioned in the question.

The final answer is derived from understanding the context and applying relevant knowledge to provide a comprehensive response."""


# ==================== 数据处理器 ====================

class DataProcessor:
    """RAG-EDA数据处理主类"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.cot_generator = CoTGenerator(api_key=api_key, base_url=base_url, model=model)
        self.model = model

    def process_single_item(self, item: Dict) -> Optional[Dict]:
        """处理单个数据条目"""
        try:
            # 提取基本信息
            conversation_id = item.get("conversation_id")
            category = item.get("category", "Others")
            conversation = item.get("conversation", [])

            if not conversation or len(conversation) == 0:
                print(f"Warning: Empty conversation for item {conversation_id}")
                return None

            # 获取第一个对话轮次
            conv = conversation[0]
            question = conv.get("question", "").strip()
            answer = conv.get("answer", "").strip()
            reference_content = conv.get("reference_content", "")

            if not question or not answer:
                print(f"Warning: Missing question or answer for item {conversation_id}")
                return None

            # 生成CoT
            cot = self.cot_generator.generate_cot(question, answer, reference_content)

            # 构建结果
            result = {
                "question": question,
                "cot": cot,
                "answer": answer,
                "language": LANGUAGE,
                "category": category,
                "source": SOURCE,
                "domain": DOMAIN,
                "original_text_preview": reference_content[:500] if len(reference_content) > 500 else reference_content
            }

            return result

        except Exception as e:
            print(f"Error processing item: {e}")
            return None

    def _get_item_hash(self, item: Dict) -> str:
        """获取条目的唯一标识（用于去重）"""
        import hashlib
        conversation_id = str(item.get("conversation_id", ""))
        conversation = item.get("conversation", [])
        if conversation and len(conversation) > 0:
            question = conversation[0].get("question", "")[:100]
            return hashlib.md5(f"{conversation_id}:{question}".encode('utf-8')).hexdigest()
        return hashlib.md5(conversation_id.encode('utf-8')).hexdigest()

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        max_workers: int = 5,
        max_items: Optional[int] = None,
        save_interval: int = 100,
        skip_existing: bool = True
    ) -> str:
        """批量处理数据集"""

        # 加载数据
        print(f"[{datetime.now()}] 正在加载数据集...")
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        print(f"数据集加载完成，共 {len(data)} 条数据")

        # 检查已处理的数据
        processed_hashes = set()
        if skip_existing and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            preview = item.get('original_text_preview', '')
                            if preview:
                                import hashlib
                                text_hash = hashlib.md5(preview[:100].encode('utf-8')).hexdigest()
                                processed_hashes.add(text_hash)
                        except:
                            continue
                print(f"发现已有输出文件，包含 {len(processed_hashes)} 条已处理数据")
            except Exception as e:
                print(f"读取已有文件失败: {e}")

        # 确定要处理的数据范围
        total_items = len(data) if max_items is None else min(max_items, len(data))
        items_to_process = data[:total_items]

        # 过滤已处理的数据
        if processed_hashes:
            filtered_items = []
            for item in items_to_process:
                item_hash = self._get_item_hash(item)
                if item_hash not in processed_hashes:
                    filtered_items.append(item)
            skipped_count = len(items_to_process) - len(filtered_items)
            items_to_process = filtered_items
            print(f"跳过 {skipped_count} 条已处理数据，实际处理 {len(items_to_process)} 条")

        print(f"\n[{datetime.now()}] 开始处理 {len(items_to_process)} 条数据...")
        print(f"输出文件: {output_path}")
        print(f"并行工作线程: {max_workers}")

        results = []
        success_count = 0
        processed_count = 0

        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for item in items_to_process:
                future = executor.submit(self.process_single_item, item)
                future_to_item[future] = item

            with tqdm(total=len(future_to_item), desc="Processing") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result(timeout=120)
                        if result:
                            results.append(result)
                            success_count += 1

                            if len(results) % save_interval == 0:
                                self._append_to_jsonl(results, output_path)
                                results = []
                    except Exception as e:
                        print(f"Error in future: {e}")

                    processed_count += 1
                    pbar.update(1)

        # 保存剩余结果
        if results:
            self._append_to_jsonl(results, output_path)

        print(f"\n[{datetime.now()}] 处理完成!")
        print(f"总条目: {total_items}")
        print(f"成功处理: {success_count}")
        print(f"失败/跳过: {total_items - success_count}")
        print(f"输出文件: {output_path}")

        return output_path

    @staticmethod
    def _append_to_jsonl(data: List[Dict], filepath: str):
        """追加数据到jsonl文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='RAG-EDA Dataset SFT数据处理')
    parser.add_argument('--input', type=str, default=INPUT_PATH, help='输入文件路径')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='输出文件路径')
    parser.add_argument('--max-items', type=int, default=None, help='处理条目数限制')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='并行线程数')
    parser.add_argument('--test', action='store_true', help='测试模式：只处理3条数据')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的数据')
    args = parser.parse_args()

    # 禁用代理
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

    print("=" * 60)
    print("RAG-EDA Dataset SFT数据处理")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print(f"API Key: {'已设置' if API_KEY != 'your-api-key-here' else '未设置'}")
    print(f"Model: {MODEL}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max Items: {args.max_items if args.max_items else '全部'}")
    print(f"Max Workers: {args.max_workers}")
    print("=" * 60)

    # 初始化处理器
    processor = DataProcessor(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )

    if args.test:
        # 测试模式：处理前3条数据
        print("\n[测试模式] 处理前3条数据...")

        # 加载数据
        data = []
        with open(args.input, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for i, item in enumerate(data):
            print(f"\n--- 测试条目 {i+1} ---")
            result = processor.process_single_item(item)
            if result:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("处理失败")
    else:
        # 批量处理
        processor.process_dataset(
            input_path=args.input,
            output_path=args.output,
            max_workers=args.max_workers,
            max_items=args.max_items,
            save_interval=SAVE_INTERVAL,
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
        sources = [item['source'] for item in output_data]
        domains = [item['domain'] for item in output_data]

        print("\n语言分布:")
        for lang, count in Counter(languages).most_common():
            print(f"  {lang}: {count}")

        print("\n类别分布:")
        for cat, count in Counter(categories).most_common():
            print(f"  {cat}: {count}")

        print("\n来源分布:")
        for src, count in Counter(sources).most_common():
            print(f"  {src}: {count}")

        print("\n领域分布:")
        for dom, count in Counter(domains).most_common():
            print(f"  {dom}: {count}")

    print(f"\n结束时间: {datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
