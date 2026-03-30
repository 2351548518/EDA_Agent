import os
# 设置 可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import tempfile
os.environ["TMPDIR"] = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "tmp")
os.makedirs(os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "tmp"), exist_ok=True)
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from huggingface_hub import login


BASE_MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "qwen3_8b")
LORA_MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "qwen3_8b_lora/lora_train")
SAVE_MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "Qwen3-8B-IC")
HF_REPO_ID = "ZeeoRe/Qwen3-8B-IC" 

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH, trust_remote_code=True)

# 2. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 3. 加载 LoRA
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# 4. 合并
merged_model = model.merge_and_unload()

# 5. 保存本地
merged_model.save_pretrained(SAVE_MODEL_PATH, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(SAVE_MODEL_PATH)

# 6. 上传到 HF Hub
login()
merged_model.push_to_hub(HF_REPO_ID, private=False)
tokenizer.push_to_hub(HF_REPO_ID, private=False)

print("全部完成！模型已合并并上传。")