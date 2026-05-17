# Resource-Constrained Agentic RAG for Grounded Stock Market Analysis
> 面向低資源本地環境的 Agentic RAG 股市分析系統：結合工具呼叫、知識檢索與幻覺防護

本專案打造一個具備自主推理（ReAct）、工具呼叫、RAG 知識檢索與 guardrails 的金融 AI Agent。專題重點不是預測股價或超越 SOTA，而是研究小型量化開源 LLM 在本地 GPU 資源有限的情境下，是否能透過工具與知識 grounding 提升金融問答的可靠性。

系統整合 yfinance 股價與基本面資料、財經新聞、ChromaDB 向量資料庫、BAAI/bge-m3 embeddings、4-bit 開源 LLM，以及自製文字式 ReAct StateGraph，支援台股案例（如 `2330.TW`、`0050.TW`）與中文金融查詢。

For contributors continuing the project:

- Start with this `README.md` for the current project direction.
- Read `week3_progress.md` for the verified Week 1-3 baseline status and recent Agent fixes.
- Follow `model_ablation_plan.md` for Week 4 Llama/Qwen model comparison.
- Historical planning documents are preserved under `docs/`.

---

## Research Question

This project does not aim to outperform state-of-the-art financial forecasting models. Instead, it investigates whether a resource-constrained local LLM can perform more grounded stock market analysis when combined with a ReAct-style agent loop, external financial tools, and a RAG knowledge base.

Specifically, we ask:

> Under the limitation that small quantized open-source LLMs cannot reliably perform JSON-based function calling, can a text-based ReAct agent improve tool usage, evidence grounding, and hallucination control in financial question answering?

中文來說，本專案想回答：

> 在小型量化開源 LLM 無法穩定支援 JSON function calling 的限制下，文字式 ReAct Agent 搭配 RAG 與金融工具，是否能提升股市分析回答的 groundedness 並降低幻覺？

---

## Novelty and Contributions

The main novelty of this project lies in system design and reliability analysis rather than predictive performance.

1. **Resource-constrained local deployment**  
   The system runs a 4-bit quantized open-source LLM on a consumer laptop GPU（RTX 4060 Laptop 8.6GB VRAM），showing how agentic financial analysis can be built without relying on commercial APIs.

2. **Text-based ReAct tool calling**  
   Since small quantized LLMs may not reliably support structured JSON function calling, we implement a custom text-based ReAct loop using LangGraph StateGraph.

3. **Grounded financial analysis with hybrid tools**  
   The agent combines market data tools, fundamental data, financial news retrieval, chart generation, and RAG-based knowledge search.

4. **Hallucination and guardrail evaluation**  
   The project evaluates whether RAG and guardrails reduce unsupported claims, fabricated numbers, and out-of-scope responses.

5. **Taiwan stock market examples**  
   The demo focuses on local stock tickers such as `2330.TW` and `0050.TW`, supporting Chinese financial queries and Taiwan-market use cases.

---

## System Architecture

The core workflow is:

```text
User Query
  -> Text-based ReAct Agent
  -> Tool Selection
  -> Tool Observation
  -> Final Answer with grounded evidence
```

原先計畫使用 LangChain 的 `create_react_agent` 與標準 tool calling，但目前新版 LangChain / LangGraph 的 JSON function calling 對 4-bit Llama 3 8B 不夠穩定。因此本專案改採文字式 ReAct 格式，由 StateGraph 控制推理與工具呼叫流程：

```text
Thought: I need to check the stock price.
Action: tool_get_stock_history
Action Input: 2330.TW, 1mo
Observation: [real tool output]
Thought: I have enough evidence.
Final Answer: ...
```

系統中的工具分成三類：

| 類別 | 工具 | 功能 |
|---|---|---|
| Market data tools | `tool_get_stock_history`, `tool_get_fundamental_data` | 查詢股價、歷史資料、本益比、殖利率、市值等 |
| News and RAG tools | `tool_search_financial_news`, `tool_search_knowledge_base` | 查詢 ticker 新聞與知識庫語意檢索 |
| Charting tool | `tool_plot_stock_chart` | 產生股價走勢圖並存檔 |

`MemorySaver` 用於多輪對話的 thread state 管理，讓不同對話能維持各自的上下文。

---

## Tools

| 工具 | 輸入 | 功能 |
|---|---|---|
| `tool_get_stock_history` | ticker, period | 抓取歷史股價（回傳文字摘要） |
| `tool_get_fundamental_data` | ticker | 本益比、殖利率、市值等基本面 |
| `tool_plot_stock_chart` | ticker, period | 繪製股價走勢圖並存檔 |
| `tool_search_financial_news` | ticker | 抓取最新財經新聞（僅限 ticker） |
| `tool_search_knowledge_base` | query | RAG 向量資料庫語意檢索 |

> 關鍵字查詢（如「半導體」、「AI」）請使用 `tool_search_knowledge_base`，不要用 `tool_search_financial_news`。

---

## Evaluation Plan

We evaluate the system through an ablation-style comparison:

| Setting | Description |
|---|---|
| LLM only | The model answers directly without tools or RAG |
| LLM + tools | The model can use stock, fundamental, news, and chart tools |
| Full system | The model uses ReAct, tools, RAG retrieval, memory, and guardrails |

Evaluation tasks include:

- Single-tool questions, such as recent stock price or P/E ratio lookup.
- Multi-tool questions, such as comparing two stocks and generating charts.
- RAG-based questions about financial concepts or industry trends.
- Out-of-scope or unsafe questions, such as requests for guaranteed investment advice or fabricated data.

Metrics:

| Metric | Description |
|---|---|
| Tool selection accuracy | Whether the agent chooses the correct tool |
| Answer relevance | Whether the final answer addresses the user query |
| Numeric correctness | Whether reported prices or fundamentals match tool outputs |
| Evidence grounding | Whether claims are supported by retrieved data |
| Hallucination count | Number of unsupported or fabricated claims |
| Refusal correctness | Whether unsafe or irrelevant requests are handled properly |

### Model Ablation Study

We also compare multiple 4-bit local LLMs under the same Agentic RAG framework. The goal is not to use the largest or newest model, but to find the best trade-off between local deployability, ReAct-format stability, Chinese financial QA quality, numeric correctness, and hallucination control.

| Role | Model | Purpose |
|---|---|---|
| Baseline | `meta-llama/Meta-Llama-3-8B-Instruct` | Current Week 3 baseline |
| Llama upgrade | `meta-llama/Llama-3.1-8B-Instruct` | Compare against a newer Llama 8B model |
| Qwen primary | `Qwen/Qwen3-4B-Instruct-2507` | Main Qwen candidate for local 4-bit testing |
| Qwen optional | `Qwen/Qwen3-8B` | 8B-scale Qwen comparison if VRAM allows |
| Optional extension | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | Additional newer Qwen candidates if time allows |

Each model will be tested with the same agent, tools, RAG database, and evaluation questions. We record load success, VRAM usage, latency, ReAct format success, tool selection accuracy, numeric correctness, answer relevance, Chinese fluency, hallucination count, and refusal correctness.

`Qwen/Qwen3.6-27B` and `Qwen/Qwen3.6-35B-A3B` are not included in the main local experiment because they are much larger 27B/35B-scale models and are less aligned with the current `AutoModelForCausalLM` + `text-generation` pipeline. They are considered future work or cloud/large-model reference baselines.

Week 4 的評估結果將作為 Week 5 guardrails 與 ablation study 的 baseline。期末報告會將錯誤案例整理成 Error Analysis，說明哪些問題可以透過 RAG、工具呼叫或拒答策略改善。

---

## Current Progress

| 週次 | 狀態 | 內容 |
|---|---|---|
| Week 1 | ✅ 完成 | 技能建置：股價、基本面、走勢圖、4-bit 模型載入 |
| Week 2 | ✅ 完成 | RAG pipeline：新聞載入、ChromaDB、語意檢索 |
| Week 3 | ✅ 完成 | Agent 框架：文字式 ReAct StateGraph + 5 工具整合 + MemorySaver 記憶；2026-05-17 baseline verification 通過 |
| Week 4 | 🔲 待開始 | Evaluation & Error Analysis：單工具、多工具、RAG、越界問題測試 |
| Week 5 | 🔲 待開始 | Guardrails and Ablation Study：幻覺處理、拒答策略、優化前後比較 |
| Week 6 | 🔲 待開始 | Demo UI and Final Report：展示介面、期末報告與結果整理 |

---

## Installation

### 系統需求

- Python 3.11
- CUDA GPU（建議 8GB+ VRAM，本專案使用 RTX 4060 Laptop 8.6GB）
- conda 環境（本專案使用 `pytorch` env）

### 1. 啟動環境並安裝套件

If the `pytorch` conda environment already exists:

```bash
conda activate pytorch
pip install -r requirements.txt
```

If the environment does not exist yet:

```bash
conda create -n pytorch python=3.11 -y
conda activate pytorch
pip install -r requirements.txt
```

If `conda` is not available in the current shell PATH, open Anaconda Prompt or use the Python executable inside your own conda environment directly:

```powershell
C:\Users\<YOUR_USERNAME>\anaconda3\envs\pytorch\python.exe
```

On the verified local machine, the path was:

```powershell
C:\Users\User\anaconda3\envs\pytorch\python.exe
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，填入 HUGGINGFACE_TOKEN
```

Model weights and embedding models are **not included in this repository**. The first run may download them into the local Hugging Face cache, so make sure you have enough disk space.

Required/expected downloads:

| Asset | Why it is needed | Approx. cache size observed |
|---|---|---:|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Week 3 baseline LLM | ~15 GB |
| `BAAI/bge-m3` | RAG embedding model | ~6.4 GB |
| `Qwen/Qwen3-4B-Instruct-2507` | Week 4 model ablation candidate | download only when running Qwen tests |

Llama models may require Hugging Face access approval and a valid `HUGGINGFACE_TOKEN`.

### 3. 確認 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage

### 測試個別工具

```bash
conda activate pytorch
cd C:\Data_science\Final

# 測試股價抓取
python -c "from src.tools.stock_tools import get_stock_history; print(get_stock_history('2330.TW', '1mo').tail())"

# 測試 4-bit 模型載入
python -c "from src.model import load_model_4bit; load_model_4bit()"
```

### 執行 Notebook

```bash
conda activate pytorch
cd C:\Data_science\Final
jupyter notebook
```

在瀏覽器開啟對應 notebook：

- `notebooks/week1_skills.ipynb`：Week 1 完整測試
- `notebooks/week2_rag.ipynb`：Week 2 RAG pipeline
- `notebooks/week3_agent.ipynb`：Week 3 Agent 整合測試

### 建立向量資料庫

```bash
python -c "
from src.rag.loader import load_all_documents
from src.rag.retriever import build_vectordb
docs = load_all_documents()
build_vectordb(docs)
"
```

### 模型設定

預設使用 `meta-llama/Meta-Llama-3-8B-Instruct`（需先在 HuggingFace 申請存取權限）。若要改用其他模型，在 `.env` 中設定：

```text
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
# MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507  # Qwen 主候選，較適合 8GB VRAM 本地測試
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # Llama 系列升級比較
# MODEL_NAME=Qwen/Qwen3-8B  # Qwen 8B 對照，需確認 VRAM 是否穩定
```

---

## Known Limitations

- The system is designed for grounded analysis, not stock price prediction or investment recommendation.
- Small 4-bit LLMs may produce unstable ReAct formatting, so the graph includes fallback handling and final-answer forcing.
- yfinance news search is ticker-based; broader keyword queries should use the RAG knowledge base instead.
- Numeric answers must be checked against tool observations during evaluation because small LLMs may paraphrase or distort numbers.
- Guardrails are planned for Week 5 and are not yet fully implemented.
- Qwen3.6 is considered future work rather than a main local baseline because the available 27B/35B-scale models exceed the project's 8GB VRAM constraint and may require a different image-text-to-text loading path.

---

## Project Structure

```text
agentic-rag-stock/
├── notebooks/
│   ├── week1_skills.ipynb      # Week 1：技能測試（股價、基本面、圖表、模型載入）
│   ├── week2_rag.ipynb         # Week 2：RAG pipeline 測試
│   ├── week3_agent.ipynb       # Week 3：Agent 整合測試（完成）
│   └── week4_reasoning.ipynb   # Week 4：Evaluation & Error Analysis（草稿，尚未納入目前 commit）
│
│   # Planned:
│   # week5_guardrails.ipynb    # Week 5：Guardrails and Ablation Study
│
├── src/
│   ├── config.py               # 集中管理路徑與參數（模型名稱、資料夾路徑）
│   ├── model.py                # 4-bit 量化模型載入（NF4 + bitsandbytes）
│   │
│   ├── tools/
│   │   ├── stock_tools.py      # 股價歷史、基本面、走勢圖（含 @tool 包裝）
│   │   ├── news_tools.py       # yfinance 財經新聞抓取（僅支援 ticker）
│   │   └── rag_tools.py        # RAG 向量資料庫檢索工具
│   │
│   ├── rag/
│   │   ├── loader.py           # 文本載入：yfinance 新聞 + PDF
│   │   ├── embedder.py         # BAAI/bge-m3 embedding（GPU）
│   │   └── retriever.py        # ChromaDB 建立、持久化、similarity search
│   │
│   └── agent/
│       ├── agent.py            # 文字式 ReAct StateGraph（Week 3）
│       ├── memory.py           # MemorySaver 多輪對話記憶（Week 3）
│       └── guardrails.py       # 防護欄與 Output Parser（Week 5，待完成）
│
├── data/
│   ├── raw/                    # 原始資料（PDF 等，不 commit）
│   └── vectordb/               # ChromaDB 持久化目錄（不 commit）
│
├── outputs/
│   └── charts/                 # 生成的股價走勢圖（PNG）
│
├── docs/
│   ├── initial_project_plan.md  # 初版專題規劃（歷史參考，不作為目前實作依據）
│   └── legacy_README.md         # README 舊版備份
│
├── .env                        # API Keys（不 commit）
├── .env.example                # 環境變數範本
├── requirements.txt            # 套件依賴
├── model_ablation_plan.md      # Week 4 Llama/Qwen 模型比較計畫
├── week1,2_progress.md         # Week 1 & 2 進度紀錄與已知問題
└── week3_progress.md           # Week 3 Agent 架構與測試紀錄
```

---

## Roadmap

- **Week 4：Evaluation & Error Analysis**  
  設計約 10 題評估題，涵蓋單工具、多工具、RAG 與越界問題，紀錄工具選擇、答案相關性、數字正確性、幻覺次數，以及 Llama/Qwen 4-bit 模型比較結果。

- **Week 5：Guardrails and Ablation Study**  
  加入拒答策略、金融風險提醒、證據不足時禁止捏造數字，並比較 guardrails 前後差異。

- **Week 6：Demo UI and Final Report**  
  包裝展示介面，整理架構、實驗結果、error analysis 與未來改進方向。
