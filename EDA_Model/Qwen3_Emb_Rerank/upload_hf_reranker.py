import os
# 设置 可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import tempfile
os.environ["TMPDIR"] = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "tmp")
os.makedirs(os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "tmp"), exist_ok=True)
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "Qwen3-Reranker-4B-IC")
HF_REPO_ID = "ZeeoRe/Qwen3-Reranker-4B-IC"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 2. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 6. 上传到 HF Hub
login()
model.push_to_hub(HF_REPO_ID, private=False)
tokenizer.push_to_hub(HF_REPO_ID, private=False)

print("全部完成！模型已上传。")
