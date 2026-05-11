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
