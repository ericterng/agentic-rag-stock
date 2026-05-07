# 專案名稱：Agentic RAG for Stock Market Analysis: Integrating Tool Use and LLM Reasoning
> 中文名稱：基於代理人 RAG 架構之股市分析：結合工具呼叫與大型語言模型推理

## 🎯 專案核心概念
本專案旨在重現企業級（如台積電）的「領域知識型 (Knowledge AI) + 領域技能型 (Skills AI) 協作」架構。
不只是單純的聊天機器人，而是打造一個具備自主推理 (Reasoning) 與工具呼叫 (Tool Use) 能力的金融理財代理人 (Agent)。

## ⚙️ 系統與硬體限制 (Constraints)
*   **運算資源**：Google Colab T4 GPU (免費版，約 15GB VRAM)。
*   **語言模型**：開源小模型（建議 `Meta-Llama-3-8B-Instruct` 或 `Qwen2.5-Instruct`），且必須使用 **4-bit 量化 (Quantization)** 載入，以符合記憶體限制並避免 OOM。
*   **核心框架**：LangChain 或 LlamaIndex。

---

## 📁 專案目錄結構

```
agentic-rag-stock/
├── notebooks/
│   ├── week1_skills.ipynb
│   ├── week2_rag.ipynb
│   ├── week3_agent.ipynb
│   ├── week4_reasoning.ipynb
│   └── week5_guardrails.ipynb
├── src/
│   ├── tools/
│   │   ├── stock_tools.py        # get_stock_history, plot_stock_chart, get_fundamental_data
│   │   ├── news_tools.py         # search_financial_news
│   │   └── rag_tools.py          # search_knowledge_base
│   ├── rag/
│   │   ├── loader.py             # 資料載入與清洗
│   │   ├── embedder.py           # BAAI/bge-m3 embedding
│   │   └── retriever.py          # ChromaDB / FAISS 檢索
│   ├── agent/
│   │   ├── agent.py              # ReAct Agent 主體
│   │   ├── memory.py             # ConversationBufferMemory
│   │   └── guardrails.py         # 防護欄與 Output Parser
│   └── config.py                 # 集中管理路徑與參數
├── data/
│   ├── raw/                      # 原始爬取資料
│   └── vectordb/                 # ChromaDB 持久化目錄
├── outputs/
│   └── charts/                   # 生成的股價圖表
├── .env                          # API keys（不 commit，加入 .gitignore）
├── .env.example                  # 範本，說明需要哪些變數
├── .gitignore
└── requirements.txt
```

---

## 🛠️ 完整工具清單 (Tools)

| 工具名稱 | 輸入 | 功能 |
|---|---|---|
| `get_stock_history(ticker, period)` | 股票代碼、時間區間 | 抓取歷史股價 |
| `get_fundamental_data(ticker)` | 股票代碼 | 本益比、殖利率、市值等基本面 |
| `plot_stock_chart(ticker, data)` | 股票代碼、價格資料 | 繪製走勢圖並存檔 |
| `search_financial_news(query)` | 股票代碼（ticker） | 查詢財經新聞（yfinance news 為主；關鍵字查詢請改用 `search_knowledge_base`） |
| `search_knowledge_base(query)` | 搜尋關鍵字 | RAG 向量資料庫檢索 |

---

## 🗓️ 6 週實作 Pipeline 藍圖

### Week 1：環境建置與 API 技能測試 (Skills Setup)
第一週專注於確立 Agent 的「技能 (Skills)」，先不處理語言模型。
*   **[ ] 任務 0：專案初始化**
    *   建立上述目錄結構，初始化 `git`，建立 `.gitignore`（排除 `.env`、`data/`、`__pycache__`）。
    *   建立 `.env.example`，列出所需環境變數（目前無需 API key，預留 HuggingFace token）。
    *   建立 `requirements.txt`，列出本週需要的套件。
*   **[ ] 任務 1：熟悉 API 與數據抓取**
    *   撰寫 Python 腳本，使用 `yfinance` 開源套件成功抓取指定股票（如 0050.TW、2330.TW）的歷史股價與基本面資料。
*   **[ ] 任務 2：撰寫視覺化函數**
    *   寫一個能接收股價資料的 Python 函數，並使用 `matplotlib` 或 `plotly` 畫出走勢圖後存成圖片檔。
*   **[ ] 任務 3：模型載入與環境測試**
    *   複習作業一 (A1) 的程式碼，在 Colab 上以 4-bit 量化成功載入模型，並進行基礎測試確認環境無誤。

### Week 2：建立領域知識庫 (Knowledge / RAG Setup)
這週要打造 Agent 的「大腦知識區」，實作檢索增強生成系統 (RAG)。
*   **[ ] 任務 1：收集領域文本資料**
    *   **主要來源**：使用 `yfinance` 的新聞 API 抓取財經新聞；下載 ETF 公開說明書（PDF）。
    *   **備用來源**：爬取金管會公告頁面（靜態頁面較穩定）。
    *   *(Novelty 亮點)* 若時間允許，加入 PTT 股版資料進行在地化情緒分析；若爬蟲受阻，改用 PTT 公開 API (`https://www.ptt.cc/bbs/Stock/index.html`)。
*   **[ ] 任務 2：建立向量資料庫 (Vector DB)**
    *   使用 LangChain，搭配開源 Embedding 模型（如 `BAAI/bge-m3`）將文本進行切塊 (Chunking)。
    *   將切塊後的資料存入輕量級免費的向量資料庫（如 ChromaDB 或 FAISS），並**持久化至 `data/vectordb/`**。
*   **[ ] 任務 3：檢索功能測試**
    *   輸入測試問題（如「請問 0050 的成分股有哪些？」），驗證 RAG 系統能精準撈出相關文本。

### Week 3：封裝工具與代理人核心 (Agent Framework)
將 Week 1 的技能庫與 Week 2 的知識庫串聯，賦予模型呼叫工具的能力。
*   **[ ] 任務 1：定義 LangChain 工具 (Tools)**
    *   使用 LangChain 的 `@tool` 裝飾器將前面的 Python 函數包裝起來。
    *   包含：`get_stock_history()`、`get_fundamental_data()`、`plot_stock_chart()`、`search_financial_news()` 與 **`search_knowledge_base()`**（RAG 檢索也必須包成 Tool）。
*   **[ ] 任務 2：建構 ReAct Agent**
    *   使用 LangChain 專為 Llama 設計的 `create_react_agent` 模組，將模型與工具清單整合，讓模型意識到可以使用哪些工具。
*   **[ ] 任務 3：加入對話記憶 (Conversation Memory)**
    *   整合 `ConversationBufferMemory`，讓 Agent 在多輪對話中能記住上下文（如「剛才你分析的那檔股票...」）。

### Week 4：測試思維鏈與規劃能力 (Reasoning & ReAct)
測試 Agent 的長時程規劃 (Long-horizon planning) 與多步驟問題解決能力。
*   **[ ] 任務 1：下達綜合複雜指令**
    *   給予複雜 Prompt，例如：「請幫我分析台積電近三個月的股價走勢，畫出圖表，並結合最近兩篇相關新聞，給我投資建議。」
    *   *(Novelty 亮點)* 測試多重工具的長時程規劃任務（例如自主決定先查台積電、再查聯發科，並綜合比較產出報告）。
*   **[ ] 任務 2：觀察 ReAct 運作邏輯**
    *   確保 Agent 有嚴格遵循 **思考 (Thought) ➡️ 行動 (呼叫 API / RAG) ➡️ 觀察 (Observation) ➡️ 回答 (Answer)** 的邏輯運作。
*   **[ ] 任務 3：建立評估框架**
    *   設計至少 10 道測試問題（涵蓋單工具、多工具、越界問題三類）。
    *   記錄每題的 **Tool Call 正確率**、**答案相關性（1-5 分人工評分）** 與 **幻覺次數**，作為 Week 5 優化與期末報告的量化依據。

### Week 5：除錯、防護欄與優化 (Debugging & Guardrails)
此階段為專案技術含金量最高、期末報告核心亮點所在。
*   **[ ] 任務 1：處理幻覺與對齊 (Alignment & Guardrails)**
    *   透過 **System Prompt** 強制模型僅依據 RAG 檢索結果回答（加入「若知識庫無相關資料，請明確告知使用者」的指令）。
    *   使用 LangChain 的 **`OutputParser`** 驗證模型輸出格式是否符合 ReAct 規範。
    *   若使用者詢問非金融相關問題（如寫遊戲），Agent 需能辨識並禮貌拒絕。
*   **[ ] 任務 2：實作自我反思與錯誤重試 (Self-Reflection & Error Recovery)**
    *   *(Novelty 亮點)* 在程式碼加入 `try-except` 捕捉格式解析錯誤。
    *   若 API 抓不到資料（如代碼輸入錯誤），賦予 Agent 閱讀 Error Log 的能力，讓它自動去 RAG 裡搜尋正確代碼並**自動重新嘗試 (Retry)**。
*   **[ ] 任務 3：對比優化前後**
    *   使用 Week 4 的 10 道測試題，重新跑一次評估，量化 Guardrails 帶來的改善幅度，作為報告的核心數據。

### Week 6：介面包裝與期末報告準備 (UI & Presentation)
完成最後的展示與具備學術深度的報告設計。
*   **[ ] 任務 1：打造視覺化介面 (視時間而定)**
    *   若時間允許，花 1~2 天使用 Streamlit 或 Gradio 建立極簡網頁 UI，左側顯示對話，右側顯示生成的圖表（若時間緊迫，直接展示 Jupyter Notebook 運行過程也完全合格）。
*   **[ ] 任務 2：撰寫期末簡報 (Error Analysis 導向)**
    *   報告重點**不要**只專注於成功率或展示最終結果。
    *   將重點放在「錯誤分析 (Error Analysis)」：探討開源小模型在擔任理財 Agent 時的失敗模式 (Failure Modes)，以及小組如何透過實作防護欄、修改提示詞與自我修復機制來解決痛點。
    *   附上 Week 4 vs Week 5 的量化評估對比表。
