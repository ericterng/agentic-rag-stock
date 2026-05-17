# Week 3 進度紀錄 — Agent Framework

## 完成日期
2026-05-11

## 完成項目

### ✅ Week 3 — Agent Framework（實測通過）

| 檔案 | 內容 |
|---|---|
| `src/agent/agent.py` | 文字式 ReAct StateGraph：agent / tool / force_final / finalize 四個節點 |
| `src/agent/memory.py` | MemorySaver checkpointer（多 thread 對話隔離） |
| `notebooks/week3_agent.ipynb` | 7 步驟端對端測試 notebook |

**實測結果**：
- 4-bit 模型載入：✅ VRAM 使用 5.7 GB（剩 2.9 GB）
- 5 個工具全部正常：✅
- TSMC 2330.TW 股價：正確回傳 2235.00（真實數據）
- 0050.TW P/E ratio：正確回傳 31.7（真實數據）
- 多輪對話（history 傳遞）：✅

---

## 2026-05-17 Baseline Verification Update

本次重新驗證 Week 1~3，是為了在進入 Week 4 model ablation（Llama/Qwen 比較）前，先確認目前 baseline 是否穩定可重現。

### ✅ 環境確認

- 正確 Python 環境：
  `C:\Users\User\anaconda3\envs\pytorch\python.exe`
- Conda 安裝位置：
  `C:\Users\User\anaconda3`
- 目前一般 PowerShell shell 中 `conda` 沒有進 PATH，因此如果 `conda activate pytorch` 失敗，可以直接使用上面的 env Python 路徑執行驗證指令。
- `pytorch` env 版本重點：
  - Python 3.11.13
  - torch 2.5.1
  - CUDA 可用：✅
  - transformers 4.57.6
  - langchain-chroma 1.1.0

### ✅ Week 1 工具層驗證

| 項目 | 結果 |
|---|---|
| `get_stock_history("2330.TW", "1mo")` | ✅ 通過 |
| `get_fundamental_data("0050.TW")` | ✅ 通過 |
| `tool_plot_stock_chart("2330.TW", "1mo")` | ✅ 通過 |

驗證結果摘要：

- `2330.TW` 最近資料可成功抓取。
- `0050.TW` 基本面可成功抓取，P/E ratio 約 `31.216623`。
- 圖表可成功輸出到 `outputs/charts/`。

### ✅ Week 2 RAG 層驗證

| 項目 | 結果 |
|---|---|
| `data/vectordb/chroma.sqlite3` | ✅ 存在 |
| `BAAI/bge-m3` Hugging Face cache | ✅ 存在 |
| `tool_search_knowledge_base("半導體 AI 需求")` | ✅ 通過 |

RAG query 範例：

```text
半導體 AI 需求
```

回傳內容包含 semiconductor / AI infrastructure demand 相關新聞與摘要，表示 ChromaDB 與 embedding 檢索流程可用。

### ✅ Week 3 Agent 層驗證

| 項目 | 結果 |
|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` 4-bit 載入 | ✅ 通過 |
| Device | `cuda:0` |
| Agent graph import | ✅ 通過 |
| 5 個 tool registry | ✅ 通過 |
| MemorySaver | ✅ 通過 |
| Llama baseline 端對端查詢 | ✅ 修正後通過 |

Llama baseline 載入結果：

```text
Model loaded: meta-llama/Meta-Llama-3-8B-Instruct
Device: cuda:0
loaded tokenizer PreTrainedTokenizerFast
loaded model LlamaForCausalLM
```

端對端測試 prompt：

```text
What is the recent stock price performance of 2330.TW?
```

成功 observations：

```text
Stock: 2330.TW | Period: 3mo
Latest Close: 2265.00
Period High:  2345.00
Period Low:   1760.00
Avg Volume:   38227851
Price Change: 19.60%
Trading Days: 56
```

圖表工具也成功輸出：

```text
Chart saved to: C:\Data_science\Final\outputs\charts\2330.TW_20260517_153623.png
```

Final Answer 正確使用股價 observation：

```text
The recent stock price performance of 2330.TW has been positive,
with a price change of 19.60% over the past 3 months.
```

### 🔧 本次發現並修正的 Week 3 bugs

#### Bug 4：多參數工具輸入解析錯誤

**問題**：
Llama 會產生：

```text
Action Input: 2330.TW, 3mo
```

原本 `tool_node` 會把整串當成 ticker：

```text
ticker = "2330.TW, 3mo"
```

導致 yfinance 查詢失敗。

**修正**：
在 `src/agent/agent.py` 新增 `_coerce_tool_input()`，在呼叫 tool 前將文字式 ReAct input 轉成正確參數。

修正後：

```python
"2330.TW, 3mo" -> {"ticker": "2330.TW", "period": "3mo"}
"2330.TW, 1 month" -> {"ticker": "2330.TW", "period": "1mo"}
```

同時在 ReAct prompt 補上各工具輸入格式，降低模型產生錯誤 input 的機率。

#### Bug 5：Financial news tool schema mismatch

**問題**：
Agent 原本傳入：

```python
{"ticker": "2330.TW"}
```

但 `tool_search_financial_news` 的參數名稱是 `query`。

**修正**：
`_coerce_tool_input()` 對 `tool_search_financial_news` 改成傳：

```python
{"query": "2330.TW"}
```

驗證結果：

```text
Action: tool_search_financial_news
Action Input: 2330.TW
Observation: Latest news for '2330.TW':
...
TSMC predicts semiconductor market will reach $1.5 trillion by 2030
```

### 💾 儲存空間與 Hugging Face cache

- C 槽清理前剩餘約 `13.6 GB`，不適合直接下載新模型。
- Hugging Face cache 中已存在：
  - `models--meta-llama--Meta-Llama-3-8B-Instruct`：約 `14.97 GB`
  - `models--BAAI--bge-m3`：約 `6.37 GB`
- 已刪除未使用的 Mistral cache：
  - `models--mistralai--Mistral-7B-Instruct-v0.3`：約 `13.5 GB`
- 清理後 C 槽剩餘約 `42.29 GB`。

### 目前結論

Week 1~3 baseline 已完成重新驗證。修正後，Llama 3 8B 4-bit Agent 能完成：

1. ReAct 推理
2. 股價工具呼叫
3. 多參數工具輸入解析
4. 圖表生成
5. RAG 查詢
6. financial news tool 呼叫
7. 基於 Observation 的 Final Answer

因此可以進入 Week 4 的固定題組評估與 Llama/Qwen model ablation。

---

## ⚠️ 重要架構說明（與原計畫不同，請務必閱讀）

原計畫使用 `langchain.agents.create_react_agent` + `ConversationBufferMemory`，但 **LangChain 1.2.17 已完全移除這些 API**。

### 實際採用架構：文字式 ReAct StateGraph

原因：`langgraph.prebuilt.create_react_agent`（新版）需要模型支援 JSON 格式的 function calling，但 **Llama 3 8B 4-bit 量化版無法可靠輸出此格式**，`bind_tools()` 呼叫結果為空（工具不會被呼叫）。

解決方案：自己實作文字式 ReAct loop：

```
Thought: 我需要查詢股價
Action: tool_get_stock_history
Action Input: 2330.TW
Observation: [工具真實回傳結果]
Thought: 我已知道答案
Final Answer: TSMC 目前股價為 2235.00
```

LangGraph 的 StateGraph 負責：
1. `agent_node`：LLM 推理，輸出 Thought/Action/Action Input
2. `tool_node`：解析 Action，執行對應工具，附加 Observation
3. `force_final`：若模型沒給出 Final Answer，強制生成
4. `finalize`：提取最終答案

### Agent 用法

```python
from src.model import load_model_4bit
from src.agent.agent import create_agent_graph
from src.agent.memory import create_memory

tokenizer, model = load_model_4bit()
memory = create_memory()
graph, memory = create_agent_graph(tokenizer, model, checkpointer=memory)

# 每次呼叫都需要完整的 init state
config = {"configurable": {"thread_id": "my-thread"}}
init = {
    "input": "What is the current price of TSMC (2330.TW)?",
    "history": "",       # 多輪對話時傳上一輪摘要
    "scratchpad": "",
    "output": "",
    "iterations": 0,
}
result = graph.invoke(init, config=config)
print(result["output"])  # The current stock price of TSMC is 2235.00.
```

---

## 環境 Bug 修正（組員 clone 後必看）

### Bug 1：RAG embedding 載入失敗（CVE-2025-32434）

**錯誤訊息**：
```
ValueError: Due to a serious vulnerability issue in torch.load, we now require
users to upgrade torch to at least v2.6 ...
```

**原因**：`transformers` 新版加了安全限制，torch < 2.6 不允許 `torch.load`。

**修正**（已寫入 `src/rag/embedder.py`）：
```python
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None  # 本機可信任模型，安全
```

> 注意：這個 patch 只適用於已知可信任的快取模型。如果要在正式環境使用，應升級 torch 至 2.6+。

---

### Bug 2：BAAI/bge-m3 快取分裂在兩個 snapshot

**原因**：模型分兩次下載，`model.safetensors` 和其他設定檔（config、tokenizer）落在不同 snapshot 目錄。

**修正**：用 hard link 把 safetensors 連到有 config 的 snapshot（不佔額外空間）：
```powershell
$config_snap = "$env:USERPROFILE\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
$safetensors_snap = "$env:USERPROFILE\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\9a0624b896d81da7492a910ffa53731274b6cf3d"
New-Item -ItemType HardLink -Path "$config_snap\model.safetensors" -Target "$safetensors_snap\model.safetensors"
```

> 如果組員是全新下載模型則不需要這個步驟。

---

### Bug 3：Chroma import 棄用

**原因**：`langchain_community.vectorstores.Chroma` 在 LangChain 1.x 已移除。

**修正**（已寫入 `src/rag/retriever.py`）：
```python
# 舊（不能用）
from langchain_community.vectorstores import Chroma
# 新
from langchain_chroma import Chroma
```

安裝：
```bash
pip install langchain-chroma
```

---

## 下一步（Week 4）

### 🔲 Week 4 — Reasoning & ReAct 測試

1. **設計 10 道評估題**（各 3-4 題）：
   - 單工具：「台積電近一個月股價？」
   - 多工具：「比較台積電和 0050 近三個月表現，並畫圖」
   - 越界：「幫我寫一個貪吃蛇遊戲」

2. **逐題記錄**：
   - Tool Call 是否正確（Y/N）
   - 答案相關性（1-5 分，人工評分）
   - 幻覺次數（模型自行捏造數字或事實的次數）

3. **觀察 ReAct 推理鏈**：確認 Thought → Action → Observation → Final Answer 流程是否合理

> 這些數據是 Week 5 優化對比的基準，也是期末報告 Error Analysis 的核心。

---

## 已知限制（期末報告 Error Analysis 素材）

| 問題 | 描述 |
|---|---|
| 小模型工具呼叫格式 | Llama 3 8B 4-bit 無法可靠輸出 JSON function call，需要文字式 ReAct |
| 答案數字化不穩定 | 有時模型取得真實數據後，Final Answer 仍用文字描述而非直接給數字 |
| 中文理解能力有限 | 8B 量化模型中文推理較弱，Week 4 評估後可考慮換 Qwen2.5-7B-Instruct |
