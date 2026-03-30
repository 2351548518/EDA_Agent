from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer,DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed
from typing import Optional, List, Union
import sys
import json

DS_CONFIG = os.path.join(os.getenv("LLM_AGENT_CONFIG_DIR", "/data/llm_agent/config"), "ds_z2_offload_config.json")
MODEL_NAME = os.path.join(os.getenv("LLM_AGENT_MODEL_DIR", "/data/llm_agent/models"), "qwen3_8b")
TRAIN_DATA = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "sft_data/eda_train.jsonl")
OUTPUT_DIR = os.path.join(os.getenv("LLM_AGENT_OUTPUT_DIR", "/data/llm_agent/output"), "qwen3_8b_lora/lora_train")

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

# 获取LoRA模型
# 转换模型
peft_model = get_peft_model(model, config)
peft_model.config.use_cache = False



# Dataset加载和预处理 =================================================================
def process_func_batch(examples, tokenizer, max_length=2048):
    """
    批次处理函数（不填充到相同长度，但会截断超长部分）
    Args:
        examples: 包含question, cot, answer, type的批次数据
        tokenizer: 分词器对象（需支持apply_chat_template）
        max_length: 最大序列长度
    Returns:
        包含input_ids, attention_mask, labels的字典（各样本长度可能不同）
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for question, cot, answer in zip(examples["question"], examples["cot"], examples["answer"]):
        # 构建对话结构
        messages = [
            {"role": "system", "content": "You are a helpful integrated circuit assistant. 你是一个有用的集成电路助手。"},
            {"role": "user", "content": question}
        ]
        
        try:
            # 生成指令部分
            instruction_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            instruction = tokenizer(instruction_text, add_special_tokens=False)
            
            # 生成响应部分
            response_text = f"<think>{cot}</think>\n{answer}" # 将cot放在特殊标签<think>中，帮助模型区分思考过程和最终答案
            response = tokenizer(response_text, add_special_tokens=False)
            
            # 合并指令和响应

            #明确分界：eos_token_id 作为特殊标记，帮助模型区分一条完整回复的终止点，防止生成时输出多余内容。
            
            # 训练推理一致：训练时加 eos，推理时模型遇到 eos_token_id 会自动停止生成，保证行为一致。
            input_ids = instruction["input_ids"] + response["input_ids"]+[tokenizer.eos_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"]+[1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]+[tokenizer.eos_token_id]
            
            # 截断超长部分
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        except Exception as e:
            print(f"处理样本时出错 - 问题: {question[:50]}... 错误: {str(e)}")
            # 添加空样本以防中断流程
            batch_input_ids.append([])
            batch_attention_mask.append([])
            batch_labels.append([])
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

eda_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
train_dataset = eda_dataset.map(
    lambda x: process_func_batch(x, tokenizer),
    batched=True,
    batch_size=4
)




# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-finetune-IC",
    experiment_name="Qwen3-8B-IC-trainer",
    description="使用通义千问Qwen3-8B模型在集成电路（Integrated Circuit, IC）相关数据集上微调。This experiment fine-tunes Qwen3-8B on integrated circuit (IC) domain datasets.",
    config={
        "model": "Qwen/Qwen3-8B",
        "dataset": "IC相关数据集/IC domain datasets",
        "train_data_number": len(train_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
)


# 配置训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=2e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    # bf16=True,
    fp16=True,
    max_grad_norm=1.0, # 用于深度学习训练过程中，控制梯度裁剪（gradient clipping）的阈值。它的作用是在每次反向传播时，将所有参数的梯度范数（通常是 L2范数）限制在不超过 1.0 的范围内
    deepspeed=DS_CONFIG
)
        


# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # padding= True 会自动将输入填充到当前批次中最长的样本长度
    callbacks=[swanlab_callback],
)


# 开启模型训练
trainer.train()
trainer.save_model(OUTPUT_DIR)
trainer.save_state()
