import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
transformers.utils.import_utils._TORCH_GREATER_EQUAL_2_6 = True  

RAW_DATA = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "sft_data/eda_train.jsonl")
OUTPUT_DATA = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "SFT_emb_reranker/data/emb_rank_train_raw.jsonl")
MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "m3e-small")

# 下载模型到指定目录
embedding_model = SentenceTransformer(
    "moka-ai/m3e-small",
    cache_folder=MODEL_PATH,
)

eda_dataset = load_dataset("json", data_files=RAW_DATA, split="train")

#挖掘难负样本
hard_train_dataset = mine_hard_negatives(
    eda_dataset,
    embedding_model,
    anchor_column_name="question",
    positive_column_name="answer",
    num_negatives=5,  # How many negatives per question-answer pair
    range_min=20,  # Skip the x most similar samples
    range_max=50,  # Consider only the x most similar samples
    max_score=0.8,  # Only consider samples with a similarity score of at most x
    absolute_margin=0.1,  # Similarity between query and negative samples should be x lower than query-positive similarity
    sampling_strategy="top",  # Randomly sample negatives from the range
    batch_size=64,  # Use a batch size of 4096 for the embedding model
    output_format="labeled-list",  
    use_faiss=True,  # Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)
)

def convert_format(example):
    # 获取正确答案和被拒绝的答案
    correct_response = next(resp for resp, label in zip(example['answer'], example['labels']) if label == 1)
    rejected_responses = [resp for resp, label in zip(example['answer'], example['labels']) if label == 0]
    return {
        "query": example['question'],
        "response": correct_response,
        "rejected_response": rejected_responses
    }
# 数据格式转换
transformed_dataset = hard_train_dataset.map(convert_format, remove_columns=hard_train_dataset.column_names)
transformed_dataset.to_json(OUTPUT_DATA,force_ascii=False)
