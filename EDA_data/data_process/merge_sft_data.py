#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并中英文SFT数据并划分训练集/测试集
- 从中文数据取30条作为测试，其余训练
- 从英文数据取30条作为测试，其余训练
"""

import json
import random
from pathlib import Path

# 文件路径
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")
ch_file = os.path.join(BASE_OUTPUT_DIR, "sft_data/ch_sft_data.jsonl")
en_file = os.path.join(BASE_OUTPUT_DIR, "sft_data/eda_english.jsonl")
output_dir = os.path.join(BASE_OUTPUT_DIR, "sft_data")

# 设置随机种子保证可复现
random.seed(42)

def load_jsonl(filepath):
    """加载jsonl文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_jsonl(data, filepath):
    """保存为jsonl文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 加载数据
    print("加载中文数据...")
    ch_data = load_jsonl(ch_file)
    print(f"  中文数据: {len(ch_data)} 条")

    print("加载英文数据...")
    en_data = load_jsonl(en_file)
    print(f"  英文数据: {len(en_data)} 条")

    # 随机打乱
    random.shuffle(ch_data)
    random.shuffle(en_data)

    # 划分测试集和训练集
    ch_test = ch_data[:30]
    ch_train = ch_data[30:]

    en_test = en_data[:30]
    en_train = en_data[30:]

    # 合并测试集和训练集
    test_data = ch_test + en_test
    train_data = ch_train + en_train

    # 再次打乱
    random.shuffle(test_data)
    random.shuffle(train_data)

    # 保存文件
    test_file = Path(output_dir) / "eda_test.jsonl"
    train_file = Path(output_dir) / "eda_train.jsonl"

    print(f"\n保存测试集: {test_file}")
    save_jsonl(test_data, test_file)

    print(f"保存训练集: {train_file}")
    save_jsonl(train_data, train_file)

    # 统计信息
    print("\n" + "=" * 50)
    print("数据划分完成!")
    print("=" * 50)
    print(f"中文数据:")
    print(f"  - 测试集: {len(ch_test)} 条")
    print(f"  - 训练集: {len(ch_train)} 条")
    print(f"英文数据:")
    print(f"  - 测试集: {len(en_test)} 条")
    print(f"  - 训练集: {len(en_train)} 条")
    print(f"\n总计:")
    print(f"  - 测试集: {len(test_data)} 条 (中文{len(ch_test)} + 英文{len(en_test)})")
    print(f"  - 训练集: {len(train_data)} 条 (中文{len(ch_train)} + 英文{len(en_train)})")
    print("=" * 50)

if __name__ == "__main__":
    main()
