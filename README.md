# Agentic RAG for Stock Market Analysis
> 基於代理人 RAG 架構之股市分析：結合工具呼叫與大型語言模型推理

本專案打造一個具備自主推理（ReAct）與工具呼叫能力的金融 AI Agent，整合 RAG 知識庫、yfinance 即時數據與開源 LLM，模擬企業級「知識 AI + 技能 AI」協作架構。

---

## 系統需求

- Python 3.11
- CUDA GPU（建議 8GB+ VRAM，本專案使用 RTX 4060 Laptop 8.6GB）
- conda 環境（本專案使用 `pytorch` env）

---

## 安裝步驟

**1. 啟動環境並安裝套件**
```bash
conda activate pytorch
pip install -r requirements.txt
```

**2. 設定環境變數**
```bash
cp .env.example .env
# 編輯 .env，填入 HUGGINGFACE_TOKEN
```

**3. 確認 GPU 可用**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 資料夾結構

```
agentic-rag-stock/
├── notebooks/
│   ├── week1_skills.ipynb      # Week 1：技能測試（股價、基本面、圖表、模型載入）
│   ├── week2_rag.ipynb         # Week 2：RAG pipeline 測試
│   ├── week3_agent.ipynb       # Week 3：Agent 整合測試（待完成）
│   ├── week4_reasoning.ipynb   # Week 4：ReAct 推理測試（待完成）
│   └── week5_guardrails.ipynb  # Week 5：Guardrails 優化（待完成）
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
│       ├── agent.py            # ReAct Agent 主體（Week 3，待完成）
│       ├── memory.py           # 對話記憶（Week 3，待完成）
│       └── guardrails.py       # 防護欄與 Output Parser（Week 5，待完成）
│
├── data/
│   ├── raw/                    # 原始資料（PDF 等，不 commit）
│   └── vectordb/               # ChromaDB 持久化目錄（不 commit）
│
├── outputs/
│   └── charts/                 # 生成的股價走勢圖（PNG）
│
├── .env                        # API Keys（不 commit）
├── .env.example                # 環境變數範本
├── requirements.txt            # 套件依賴
└── week1,2_progress.md         # Week 1 & 2 進度紀錄與已知問題
```

---

## 工具清單

| 工具 | 輸入 | 功能 |
|---|---|---|
| `tool_get_stock_history` | ticker, period | 抓取歷史股價（回傳文字摘要） |
| `tool_get_fundamental_data` | ticker | 本益比、殖利率、市值等基本面 |
| `tool_plot_stock_chart` | ticker, period | 繪製股價走勢圖並存檔 |
| `tool_search_financial_news` | ticker | 抓取最新財經新聞（僅限 ticker） |
| `tool_search_knowledge_base` | query | RAG 向量資料庫語意檢索 |

> 關鍵字查詢（如「半導體」、「AI」）請使用 `tool_search_knowledge_base`，不要用 `tool_search_financial_news`。

---

## 執行方式

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

### 建立向量資料庫（Week 2）
```bash
python -c "
from src.rag.loader import load_all_documents
from src.rag.retriever import build_vectordb
docs = load_all_documents()
build_vectordb(docs)
"
```

---

## 模型設定

預設使用 `meta-llama/Meta-Llama-3-8B-Instruct`（需先在 HuggingFace 申請存取權限）。
若要改用其他模型，在 `.env` 中設定：
```
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
# MODEL_NAME=Qwen/Qwen2.5-7B-Instruct  # 中文能力更強，但需下載 ~15 GB
```

---

## 週次進度

| 週次 | 狀態 | 內容 |
|---|---|---|
| Week 1 | ✅ 完成 | 技能建置：股價、基本面、走勢圖、4-bit 模型載入 |
| Week 2 | ✅ 完成 | RAG pipeline：新聞載入、ChromaDB、語意檢索 |
| Week 3 | 🔲 進行中 | Agent 框架：ReAct Agent + 工具整合 + 對話記憶 |
| Week 4 | 🔲 待開始 | 推理測試：長時程規劃、多工具呼叫、評估框架 |
| Week 5 | 🔲 待開始 | Guardrails：幻覺處理、自我修復、優化對比 |
| Week 6 | 🔲 待開始 | UI 包裝與期末報告 |
