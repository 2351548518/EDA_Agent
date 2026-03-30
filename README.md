# EDA Agent - 集成电路领域 RAG 智能问答系统

本项目是一个面向集成电路（EDA/IC）领域的智能问答系统，基于 RAG（检索增强生成）架构，结合领域专属的大语言模型，提供专业的技术问答服务。

## 项目架构概览

```
EDA_Agent/
├── EDA_data/          # 数据处理层：多源数据获取与SFT数据构建
├── EDA_Model/         # 模型层：领域模型微调与部署
└── EDA_Agent/         # 应用层：RAG系统与Web服务
    ├── backend/       # FastAPI后端服务
    ├── frontend/      # Vue3前端界面
    └── data/          # 本地数据存储
```

---

## 一、数据处理层 (EDA_data)

数据处理层负责从多源异构数据中获取、清洗并构建高质量的SFT（监督微调）训练数据。

### 1.1 数据源

| 数据集 | 来源 | 类型 | 说明 |
|--------|------|------|------|
| **RAG-EDA** | 学术数据集 | EDA问答对 | 集成电路设计相关的高质量问答对 |
| **IC-Textbook** | HuggingFace | 教材文本 | 集成电路教材知识，支持中英日韩多语言 |
| **MG-Verilog** | HuggingFace | Verilog代码 | 多语言Verilog代码及描述，支持代码生成与解释 |
| **CH-IC-Textbook** | 本地数据 | 中文IC教材 | 中文集成电路教材PDF解析数据 |

**最终SFT数据集**：合并处理后约 **7K** 条高质量训练样本

### 1.2 数据处理流程

```
原始数据 → 语言检测 → QA对生成 → CoT生成 → 格式转换 → SFT数据
```

#### 核心处理脚本

| 脚本 | 功能 |
|------|------|
| `eda_sft_data_processing.py` | RAG-EDA数据集处理，生成带CoT的SFT数据 |
| `ic_sft_data_processing.py` | IC-Textbook教材数据处理，多语言QA生成 |
| `verilog_sft_data_processing.py` | Verilog代码数据处理，代码问答对生成 |
| `merge_sft_data.py` | 多源数据合并与去重 |
| `ch_sft_data_process.py` | 中文医疗数据处理（对比学习） |

#### 数据格式示例

```json
{
  "question": "什么是CMOS反相器的传输特性？",
  "cot": "我需要分析CMOS反相器的工作原理...",
  "answer": "CMOS反相器的传输特性描述了输入电压与输出电压之间的关系...",
  "language": "Chinese",
  "category": "技术原理",
  "source": "IC-Textbook",
  "domain": "IC"
}
```

### 1.3 环境变量配置

```bash
# 数据目录配置
export LLM_AGENT_DATA_DIR="/data/llm_agent/data"      # 原始数据目录
export LLM_AGENT_OUTPUT_DIR="/data/llm_agent/output"  # 处理结果输出目录
```

---

## 二、模型层 (EDA_Model)

模型层包含领域专属模型的微调和部署，包括主模型、Embedding模型和Reranker模型。

### 2.1 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                      模型层架构                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│   主模型 (LLM)   │  Embedding模型  │      Reranker模型       │
├─────────────────┼─────────────────┼─────────────────────────┤
│  Qwen3-8B-IC    │ Qwen3-Embedding │     Qwen3-Reranker      │
│  (对话生成)      │   (向量编码)     │      (相关性排序)        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 2.2 主模型微调 (Qwen3-8B-IC)

基于 Qwen3-8B 进行领域微调，使用 LoRA 技术降低训练成本。

#### 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen3-8B |
| 微调方法 | LoRA |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| 学习率 | 2e-4 |
| 训练轮数 | 3 epochs |
| 批次大小 | 1 (per device) |
| 梯度累积 | 4 steps |
| 优化器 | DeepSpeed ZeRO-2 |

#### 训练脚本

```bash
# 配置环境变量
export LLM_AGENT_MODEL_DIR="/data/llm_agent/models"
export LLM_AGENT_OUTPUT_DIR="/data/llm_agent/output"
export LLM_AGENT_CONFIG_DIR="/data/llm_agent/config"

# 启动训练
cd EDA_Model/Qwen3_8B
python train.py
```

#### 模型评估

```bash
# 运行评估
python evaluate_qwen.py \
  --model_path ${LLM_AGENT_MODEL_DIR}/Qwen3-8B-IC \
  --test_data ${LLM_AGENT_OUTPUT_DIR}/sft_data/test.jsonl
```

#### 实验结果

在EDA/IC领域测试集上对比了基础模型(Qwen3-8B)与微调模型(Qwen3-8B-IC)的表现：

| 指标 | 基础模型(Think) | 基础模型(No-Think) | 微调模型(Think) | 微调模型(No-Think) |
|------|----------------|-------------------|----------------|-------------------|
| ROUGE-1 | 0.24 | 0.25 | **0.43** | **0.42** |
| ROUGE-2 | 0.10 | 0.10 | **0.24** | **0.23** |
| ROUGE-L | 0.18 | 0.19 | **0.34** | **0.33** |
| BLEU | 3.08 | 2.84 | **18.95** | **11.21** |
| BERTScore-P | 75.65 | 76.44 | **90.18** | 80.71 |
| BERTScore-R | 82.46 | 82.33 | **88.03** | 82.28 |
| BERTScore-F1 | 78.87 | 79.25 | **89.06** | 81.45 |

**结论**：
- 微调后的模型在所有指标上均有显著提升
- ROUGE-1从0.24/0.25提升至0.43/0.42，提升约79%
- BLEU指标提升最为显著，Think模式从3.08提升至18.95（+515%），No-Think模式从2.84提升至11.21（+295%）
- BERTScore F1从约79提升至89（Think模式）和81（No-Think模式），语义相似度明显改善
- Think模式在各指标上均优于No-Think模式，说明思考过程有助于提升回答质量

### 2.3 Embedding模型微调 (Qwen3-Embedding-4B-IC)

基于 Qwen3-Embedding-4B 进行领域适配，用于文档向量编码。

#### 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen3-Embedding-4B |
| 任务类型 | embedding |
| 损失函数 | InfoNCE |
| 学习率 | 5e-5 |
| LoRA Rank | 8 |
| 最大长度 | 8192 |

#### 数据格式

```json
{
  "query": "CMOS反相器的工作原理是什么？",
  "pos": ["相关文档1", "相关文档2"],
  "neg": ["不相关文档1", "不相关文档2"]
}
```

### 2.4 Reranker模型微调 (Qwen3-Reranker-4B-IC)

基于 Qwen3-Reranker-4B 进行领域适配，用于精排检索结果。

#### 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen3-Reranker-4B |
| 任务类型 | generative_reranker |
| 损失函数 | pointwise_reranker |
| 学习率 | 5e-5 |
| 最大长度 | 4096 |

### 2.5 模型部署

#### vLLM部署

```bash
# Think模式（带思考过程）
CUDA_VISIBLE_DEVICES=6,7 \
python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3-8B-IC \
  --model ${LLM_AGENT_MODEL_DIR}/Qwen3-8B-IC \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.85

# No-Think模式（直接回答）
CUDA_VISIBLE_DEVICES=6,7 \
python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3-8B-IC \
  --model ${LLM_AGENT_MODEL_DIR}/Qwen3-8B-IC \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

---

## 三、RAG应用层 (EDA_Agent)

RAG应用层提供完整的检索增强生成服务，包括文档管理、智能检索和对话交互。

### 3.1 系统架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline (LangGraph)                        │
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │ retrieve_initial │ 初始检索：Hybrid搜索(Dense+Sparse) + RRF融合      │
│  │   (三级分块召回)  │         + Reranker精排 + Auto-merging合并         │
│  └────────┬─────────┘                                                   │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │ grade_documents  │ 相关性评分：LLM判断文档是否相关(yes/no)            │
│  │   (相关性门控)    │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│     ┌─────┴─────┐                                                       │
│     ▼           ▼                                                       │
│  ┌──────┐   ┌──────────────────┐                                       │
│  │  END │   │ rewrite_question │ 查询重写：Step-Back / HyDE / Complex   │
│  │(生成)│   │   (查询扩展)      │         生成扩展查询或假设文档         │
│  └──────┘   └────────┬─────────┘                                       │
│                      ▼                                                  │
│           ┌──────────────────┐                                         │
│           │ retrieve_expanded│ 扩展检索：使用重写后的查询再次检索        │
│           │   (二次召回)      │          合并结果去重                    │
│           └────────┬─────────┘                                         │
│                    ▼                                                    │
│                   END                                                   │
│                 (生成答案)                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**流程说明**：
1. **初始检索**：对原始问题进行Hybrid检索（稠密向量+稀疏向量），经RRF融合和Reranker精排后，通过Auto-merging合并三级分块
2. **相关性评分**：使用LLM对检索结果进行二进制评分（yes/no），判断是否满足回答需求
3. **查询重写**：若相关性不足，根据问题类型选择Step-Back（生成上层问题）、HyDE（生成假设文档）或Complex策略扩展查询
4. **扩展检索**：使用重写后的查询进行二次检索，合并去重后生成最终答案

### 3.2 核心特性

#### 混合检索 (Hybrid Search)
- **稠密向量检索**：基于 Qwen3-Embedding-4B-IC 的语义匹配
- **稀疏向量检索**：基于 BM25 的关键词匹配
- **RRF融合**：倒数排名融合，无需调节权重参数

#### 三级分块策略
- **L1（粗粒度）**：2000-3000 tokens，主题级分块
- **L2（中粒度）**：1000-1500 tokens，章节级分块
- **L3（细粒度）**：512-1024 tokens，叶子分块，用于向量化

#### Auto-merging机制
- 优先召回L3叶子块
- 满足阈值后自动合并到父块（L3→L2→L1）
- 保留完整上下文，减少向量冗余

#### 查询重写体系
- **Step-Back**：生成更宽泛的上层问题
- **HyDE**：生成假设文档用于扩展检索
- **路由选择**：智能判断是否需要重写

**效果验证**：引入查询重写后，RAG系统评估指标显著提升

| 指标 | 引入前 | 引入后 | 提升 |
|------|--------|--------|------|
| correctness | 0.683 | **0.75** | +9.8% |
| custom_evaluator | 0.80 | **0.867** | +8.4% |

### 3.3 本地部署

#### 环境准备

```bash
# Python 3.12+
# 安装依赖
uv sync

# 或使用pip
pip install -e .
```

#### 配置环境变量

创建 `.env` 文件：

```env
# ===== 模型配置 =====
ARK_API_KEY=your_api_key
MODEL=Qwen3-8B-IC
BASE_URL=http://localhost:8000/v1
EMBEDDER=Qwen3-Embedding-4B-IC

# ===== Rerank配置 =====
RERANK_MODEL=Qwen3-Reranker-4B-IC
RERANK_BINDING_HOST=http://localhost:8001
RERANK_API_KEY=your_key

# ===== Milvus向量库 =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# ===== 路径配置 =====
LLM_AGENT_DATA_DIR=/data/llm_agent/data
LLM_AGENT_OUTPUT_DIR=/data/llm_agent/output
LLM_AGENT_MODEL_DIR=/data/llm_agent/models
```

#### 启动服务

```bash
# 1. 启动Milvus向量库
docker compose up -d

# 2. 启动后端服务
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 3. 访问应用
# 前端界面: http://127.0.0.1:8000/
# API文档: http://127.0.0.1:8000/docs
```

### 3.4 API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 非流式对话 |
| `/chat/stream` | POST | 流式对话（SSE） |
| `/sessions/{user_id}` | GET | 获取会话列表 |
| `/documents` | GET | 获取文档列表 |
| `/documents/upload` | POST | 上传文档 |
| `/documents/{filename}` | DELETE | 删除文档 |

### 3.5 技术栈

- **后端**：FastAPI + LangChain + LangGraph
- **向量库**：Milvus（稠密+稀疏混合索引）
- **前端**：Vue 3 + marked + highlight.js
- **部署**：Docker + vLLM

---

## 四、项目目录结构

```
EDA_Agent/
├── EDA_data/                          # 数据处理层
│   ├── data/                          # 数据获取与清洗
│   │   ├── clean_mineru_md.py         # Markdown清洗
│   │   ├── pdf_mineru_to_clean_json_langchain.py  # PDF解析
│   │   ├── down_IC_Textbook_Dataset.py
│   │   └── down_MG_Verilog.py
│   └── data_process/                  # SFT数据处理
│       ├── eda_sft_data_processing.py
│       ├── ic_sft_data_processing.py
│       ├── verilog_sft_data_processing.py
│       └── merge_sft_data.py
│
├── EDA_Model/                         # 模型层
│   ├── Qwen3_8B/                      # 主模型微调
│   │   ├── train.py                   # LoRA训练脚本
│   │   ├── evaluate_qwen.py           # 模型评估
│   │   ├── merge_lora.py              # LoRA合并
│   │   └── upload_hf.py               # 上传HuggingFace
│   ├── Qwen3_Emb_Rerank/              # Embedding/Reranker微调
│   │   ├── emb_data_process.py
│   │   ├── reranker_data_process.py
│   │   └── sft_emb_rerankder.ipynb
│   └── deploy/vllm/                   # 模型部署
│       └── vllm_Qwen3.ipynb
│
└── EDA_Agent/                         # 应用层
    ├── backend/                       # FastAPI后端
    │   ├── app.py                     # 应用入口
    │   ├── agent.py                   # LangChain Agent
    │   ├── api.py                     # API路由
    │   ├── rag_pipeline.py            # RAG流程
    │   ├── milvus_client.py           # 向量检索
    │   ├── embedding.py               # 向量编码
    │   ├── document_loader.py         # 文档处理
    │   └── tools.py                   # 工具定义
    ├── frontend/                      # Vue3前端
    │   ├── index.html
    │   ├── script.js
    │   └── style.css
    ├── data/                          # 本地数据
    └── docker-compose.yml             # Milvus部署
```

---

## 五、关键创新点

1. **领域专属模型**：基于Qwen3构建IC/EDA领域专属模型，提升专业问答质量
2. **混合检索架构**：Dense + Sparse + Reranker三级检索，兼顾语义与关键词匹配
3. **三级分块策略**：L1/L2/L3分层分块，Auto-merging自动聚合上下文
4. **实时RAG可视化**：检索过程实时推送前端，展示完整思考链路
5. **查询重写体系**：Step-Back + HyDE智能扩展查询，提升召回率

---

## 六、许可证

本项目仅供学术研究使用。
