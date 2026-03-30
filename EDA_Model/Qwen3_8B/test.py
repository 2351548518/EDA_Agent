import os
# 设置 可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextStreamer


# 模型路径（已经合并好的模型）
MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "Qwen3-8B-IC")

# ---------- 1. 加载分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # 直接加载你保存的分词器

# ---------- 2. 加载模型 ----------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # 自动分配设备（GPU/CPU）
)

class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=True, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        text = self.generated_text.strip()
        return text

def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=is_thinking
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    model.eval()  # 确保模型在推理模式
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=1024,
            temperature = 0.6, top_p = 0.95, top_k = 20, # 推理模式参数
            streamer=streamer,  # 关键：使用自定义的 streamer
        )

    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"✅ 成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    return full_output

# 测试集中的数据
ask("在阅读场效应管电路图时，如何通过符号快速判断其类型？符号中的箭头和线条各有什么含义？")

print("\n\n\n")
print("####################################################")
print("####################################################")
print("####################################################")
print("####################################################")
print("\n\n\n")

ask("在阅读场效应管电路图时，如何通过符号快速判断其类型？符号中的箭头和线条各有什么含义？",is_thinking=False)

