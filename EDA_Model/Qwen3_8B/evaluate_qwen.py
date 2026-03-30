import os
# 设置 可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import langid
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import torch
from datasets import load_dataset,Dataset
from bert_score import score

# 初始化评估指标
rouge = evaluate.load(os.path.join(os.getenv("LLM_AGENT_METRIC_DIR", "/data/llm_agent/metrics"), "rouge.py"))
bleu = evaluate.load(os.path.join(os.getenv("LLM_AGENT_METRIC_DIR", "/data/llm_agent/metrics"), "bleu.py"))


MODEL_PATH = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "qwen3_8b")

TEST_DATA = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "sft_data/eda_test.jsonl")
BERT_BASE_CHINESE = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "bert-base-chinese")
ROBERTA_LARGE = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "roberta-large")
EVALUATE_RESULTS_PATH = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "results/base_qwen8b_evaluation.json")

# 1. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

model.eval()

# 2. 加载测试数据
test_data = load_dataset("json", data_files=TEST_DATA, split="train")
# test_data = test_data.select(range(2))  # 调试用，正式评估时注释掉这一行

# 3. 语言检测函数
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# 4. 生成预测（支持思考模式开关）
def generate_predictions(questions, enable_thinking=True):
    predictions = []
    thinking_contents = []
    
    for q in tqdm(questions, desc=f"Generating (thinking={enable_thinking})"):
        messages = [
            {"role": "system", "content": "You are a helpful integrated circuit assistant. 你是一个有用的集成电路助手。"},
            {"role": "user", "content": q}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            thinking = ""
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        thinking_contents.append(thinking)
        predictions.append(answer)
    
    return predictions, thinking_contents

# 5. 评估函数
def evaluate_metrics(predictions, references):
    # 计算ROUGE
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    # 计算BLEU
    bleu_results = bleu.compute(
        predictions=predictions,
        references=references  # BLEU需要references是列表的列表
    )
    
    # 计算BERTScore（按语言分组）
    def calculate_bertscore(preds, refs):
        # 文本截断函数，保证不超过最大token数
        def safe_truncate(text, tokenizer, max_len=512):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                return tokenizer.decode(tokens, skip_special_tokens=True)
            return text

        lang_groups = defaultdict(list)
        for p, r in zip(preds, refs):
            lang = detect_language(r)
            lang_groups[lang].append((p, r))
        
        scores = {'precision': [], 'recall': [], 'f1': []}
        for lang, group in lang_groups.items():
            if not group:
                continue
            g_preds, g_refs = zip(*group)
            # 针对每个文本做截断（中英文都截断）
            g_preds = [safe_truncate(x, tokenizer, 512) for x in g_preds]
            g_refs = [safe_truncate(x, tokenizer, 512) for x in g_refs]
            try:
                P, R, F1 = score(
                    g_preds, 
                    g_refs,
                    model_type = BERT_BASE_CHINESE if lang == 'zh' else ROBERTA_LARGE,
                    lang='en' if lang == 'en' else 'zh', 
                    verbose=True,
                    num_layers=24 if lang == 'en' else 12,
                )
                scores['precision'].extend(P)
                scores['recall'].extend(R)
                scores['f1'].extend(F1)
            except Exception as e:
                print(f"BERTScore error ({lang}): {str(e)}")
        
        return {
            k: np.mean(v)*100 if v else 0 
            for k, v in scores.items()
        }
    
    bert_scores = calculate_bertscore(predictions, references)
    
    return {
        "rouge1": round(rouge_scores["rouge1"], 2),
        "rouge2": round(rouge_scores["rouge2"], 2),
        "rougeL": round(rouge_scores["rougeL"], 2),
        "bleu": round(bleu_results["bleu"] * 100, 2),
        "bertscore_precision": round(bert_scores['precision'], 2),
        "bertscore_recall": round(bert_scores['recall'], 2),
        "bertscore_f1": round(bert_scores['f1'], 2)
    }

# 6. 主评估流程
def main():
    questions = [d["question"] for d in test_data]
    references = [d["answer"] for d in test_data]
    
    # 添加总进度条
    with tqdm(total=2, desc="Overall Progress") as pbar:
        # 分别测试两种模式
        results = {}
        for thinking_mode in [True, False]:
            preds, thoughts = generate_predictions(questions, enable_thinking=thinking_mode)
            metrics = evaluate_metrics(preds, references)
            
            results[f"thinking_{thinking_mode}"] = {
                "predictions": preds,
                "thinking_contents": thoughts,
                "metrics": metrics
            }
            pbar.update(1)
    
    # 打印对比结果
    print("\n=== 微调模型评估结果（思考模式 vs 非思考模式） ===")
    for mode in [True, False]:
        data = results[f"thinking_{mode}"]
        m = data["metrics"]
        print(f"\n◆ 模式: {'思考模式' if mode else '非思考模式'}")
        print(f"ROUGE-1: {m['rouge1']}% | ROUGE-2: {m['rouge2']}% | ROUGE-L: {m['rougeL']}%")
        print(f"BLEU: {m['bleu']}% | BERTScore-F1: {m['bertscore_f1']}%")
        
        # 打印首条示例
        print("\n示例：")
        print(f"问题: {questions[0]}")
        if mode:
            print(f"思考过程: {data['thinking_contents'][0]}")
        print(f"生成回答: {data['predictions'][0]}")
        print(f"参考答案: {references[0]}")
    
    # 保存完整结果
    os.makedirs(os.path.dirname(EVALUATE_RESULTS_PATH), exist_ok=True)
    with open(EVALUATE_RESULTS_PATH, "w") as f:
        json.dump({
            "test_data": list(test_data),
            "results": results
        }, f, indent=2, ensure_ascii=False, default=lambda o: float(o))

if __name__ == "__main__":
    main()