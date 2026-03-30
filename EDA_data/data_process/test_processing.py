#!/usr/bin/env python3
"""
IC Textbook Dataset 数据处理测试脚本
用于验证语言检测和Prompt生成功能
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# 设置语言检测的确定性结果
DetectorFactory.seed = 0


# 测试数据
TEST_TEXTS = {
    "English": """Semiconductor devices are the foundation of modern electronics.
    The metal-oxide-semiconductor field-effect transistor (MOSFET) has been the
    dominant device for integrated circuits since the 1960s. In this chapter,
    we will explore the physics and operation principles of these devices.""",

    "Chinese": """半导体器件是现代电子学的基础。金属氧化物半导体场效应晶体管（MOSFET）
    自1960年代以来一直是集成电路的主导器件。在本章中，我们将探讨这些器件的
    物理原理和工作原理。""",
}


def detect_language(text: str) -> str:
    """检测文本语言"""
    try:
        sample = text[:1000] if len(text) > 1000 else text
        lang_code = detect(sample)

        lang_map = {
            'zh': 'Chinese',
            'en': 'English',
            'ja': 'Japanese',
            'ko': 'Korean',
        }
        return lang_map.get(lang_code, 'English')
    except LangDetectException:
        # 通过字符特征判断中文
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return 'Chinese'
        return 'English'


def get_chinese_prompt(text: str) -> str:
    """中文Prompt模板"""
    return f"""请基于以下文本内容，生成一个高质量的问答对用于模型微调。

文本内容：
{text[:3000]}

请按以下要求生成：
1. **question**: 基于文本内容生成一个具体、清晰的问题。
2. **cot** (思维链): 提供详细的思考过程。
3. **answer**: 基于文本内容提供完整、准确的回答。
4. **category**: 从以下类别中选择："基本概念", "技术原理", "应用场景", "设计方法", "制造工艺", "性能分析", "器件物理", "电路设计", "EDA工具", "其他"

请以JSON格式输出：
{{
    "question": "生成的问题",
    "cot": "详细的思考过程...",
    "answer": "回答内容...",
    "category": "类别"
}}"""


def get_english_prompt(text: str) -> str:
    """英文Prompt模板"""
    return f"""Please generate a high-quality question-answer pair based on the following text.

Text content:
{text[:3000]}

Please generate:
1. **question**: A specific, clear question based on the text.
2. **cot** (Chain of Thought): Detailed reasoning process.
3. **answer**: Complete and accurate answer based on the text.
4. **category**: Choose from: "Basic Concepts", "Technical Principles", "Application Scenarios", "Design Methods", "Manufacturing Process", "Performance Analysis", "Device Physics", "Circuit Design", "EDA Tools", "Others"

Output in JSON format:
{{
    "question": "Generated question",
    "cot": "Detailed reasoning process...",
    "answer": "Answer content...",
    "category": "Category"
}}"""


def main():
    print("=" * 60)
    print("IC Textbook Dataset SFT数据处理 - 测试脚本")
    print("=" * 60)

    # 测试语言检测
    print("\n[1] 测试语言检测功能:")
    for expected_lang, text in TEST_TEXTS.items():
        detected = detect_language(text)
        status = "✓" if detected == expected_lang else "✗"
        print(f"  {status} {expected_lang}: 检测结果 = {detected}")

    # 测试Prompt生成
    print("\n[2] 测试Prompt生成功能:")

    for lang, text in TEST_TEXTS.items():
        print(f"\n{'='*40}")
        print(f"语言: {lang}")
        print(f"{'='*40}")

        detected_lang = detect_language(text)

        if detected_lang == "Chinese":
            prompt = get_chinese_prompt(text)
        else:
            prompt = get_english_prompt(text)

        print(f"\n生成的Prompt:\n{prompt}")
        print(f"\nPrompt长度: {len(prompt)} 字符")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
