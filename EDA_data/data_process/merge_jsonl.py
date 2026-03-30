import json
import os
from pathlib import Path

# 基础数据目录，可通过环境变量覆盖
BASE_OUTPUT_DIR = os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output")

# 源目录
data_dir = Path(BASE_OUTPUT_DIR) / "sft_data"

# 输出文件
output_file = data_dir / "eda_english.jsonl"

# 获取所有 jsonl 文件（排除已存在的输出文件）
jsonl_files = sorted([f for f in data_dir.glob("*.jsonl") if f.name != "eda_english.jsonl"])

print(f"找到 {len(jsonl_files)} 个 jsonl 文件:")
for f in jsonl_files:
    print(f"  - {f.name}")

# 合并文件
total_lines = 0
with open(output_file, 'w', encoding='utf-8') as out_f:
    for jsonl_file in jsonl_files:
        print(f"\n正在处理: {jsonl_file.name}")
        line_count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                if line:  # 跳过空行
                    out_f.write(line + '\n')
                    line_count += 1
        print(f"  写入 {line_count} 行")
        total_lines += line_count

print(f"\n✅ 合并完成!")
print(f"输出文件: {output_file}")
print(f"总数据条数: {total_lines:,}")
print(f"文件大小: {output_file.stat().st_size / (1024*1024):.2f} MB")
