# 專案進度紀錄

## 環境
- 本機：Python 3.11.9, PyTorch 2.5.1+CUDA, RTX 4060 Laptop 8.6GB, conda env: pytorch
- GitHub：https://github.com/ericterng/agentic-rag-stock
- gh CLI 完整路徑：`C:\Program Files\GitHub CLI\gh.exe`
- Deadline：2026-06-04

## 完成進度

### ✅ Week 1 — Skills Setup
- `src/tools/stock_tools.py`：yfinance 股價歷史、基本面、matplotlib 走勢圖（含 @tool 包裝）
- `src/tools/news_tools.py`：yfinance 新聞抓取（含 @tool 包裝）
- `src/model.py`：4-bit 量化模型載入（NF4，RTX 4060 可跑）— ✓ 實測通過，Device: cuda:0
- 本機測試：股價、基本面、圖表、新聞全數通過 ✓

### ✅ Week 2 — RAG Setup
- `src/rag/loader.py`：yfinance 新聞 + PDF 載入
- `src/rag/embedder.py`：BAAI/bge-m3 embedding（GPU）
- `src/rag/retriever.py`：ChromaDB 建立與 similarity search
- `src/tools/rag_tools.py`：RAG 包成 @tool（Week 3 提前實作）
- 本機測試：20 篇新聞 → 28 chunks → ChromaDB → 3 個 query 檢索通過 ✓

## 下一步

### ✅ Week 1 & 2 全部通過
- 執行環境：conda `pytorch`（torch 2.5.1+CUDA，bitsandbytes 0.49.2）
- 模型：meta-llama/Meta-Llama-3-8B-Instruct，4-bit NF4，Device: cuda:0

### 🔲 Week 3 — Agent Framework
1. `src/agent/agent.py`：用 `create_react_agent` 整合模型 + 所有 Tool
2. `src/agent/memory.py`：ConversationBufferMemory 多輪對話記憶
3. `notebooks/week3_agent.ipynb`：端對端測試
4. **先跑 `src/model.py` 確認 4-bit 模型載入正常**（這是 Week 3 的前提）

## 已知問題（已修正）
| 問題 | 修正方式 |
|---|---|
| `langchain.schema.Document` not found | 改用 `langchain_core.documents.Document` |
| `langchain.text_splitter` not found | 改用 `langchain_text_splitters` |
| `langchain_community.HuggingFaceEmbeddings` deprecated | 改用 `langchain_huggingface` |
| PyTorch 2.5.1 torch.load CVE | 升至 2.6.0+cu124，順帶升 torchvision/torchaudio |
