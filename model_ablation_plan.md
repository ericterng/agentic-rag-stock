# Model Ablation Plan

本文件記錄 Week 4/5 的本地 4-bit 模型比較計畫。實驗目標不是追求最大或最新模型，而是在 RTX 4060 Laptop 8GB VRAM 的限制下，找出最適合文字式 ReAct 金融 Agent 的模型。

---

## Goal

Compare several local 4-bit instruction models under the same Agentic RAG framework:

- 是否能在本地 4-bit 成功載入。
- 是否能穩定遵守文字式 ReAct 格式。
- 是否能正確選工具並使用 Observation。
- 中文金融回答是否自然、相關且少幻覺。
- 是否能正確拒答越界或不安全問題。

---

## Model Candidates

| Priority | Model | Role | Notes |
|---|---|---|---|
| Required | `meta-llama/Meta-Llama-3-8B-Instruct` | Baseline | Current Week 3 model |
| Required | `meta-llama/Llama-3.1-8B-Instruct` | Llama upgrade | Tests whether newer Llama improves ReAct stability |
| Required | `Qwen/Qwen3-4B-Instruct-2507` | Qwen primary | Main Qwen candidate; best fit for local 8GB 4-bit testing |
| Optional | `Qwen/Qwen3-8B` | Qwen 8B comparison | Closer size match to Llama 8B; test only if VRAM is stable |
| Optional | `Qwen/Qwen3.5-4B` | Newer Qwen small model | Test if time allows |
| Optional | `Qwen/Qwen3.5-9B` | Newer Qwen mid-size model | Test only if 8GB VRAM can load it reliably |

Not included in the main experiment:

| Model | Reason |
|---|---|
| `Qwen/Qwen3.6-27B` | Too large for the 8GB VRAM low-resource setting; may require image-text-to-text loading |
| `Qwen/Qwen3.6-35B-A3B` | Too large and not aligned with the current text-generation pipeline |

Qwen3.6 can be mentioned as future work or used as a cloud/large-model reference if external compute becomes available.

---

## Setup

Use the same code path for the first round:

- Keep `src/model.py` with 4-bit NF4 + bitsandbytes.
- Switch models through `.env`:

```text
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
```

- Keep the same ReAct prompt, tools, RAG database, and evaluation questions.
- Use `transformers>=4.51.0` for Qwen3 compatibility.
- If Qwen outputs extra thinking text or breaks the ReAct format, record it as an experimental result first. Only add Qwen-specific prompts if most Qwen runs fail for formatting reasons.
- The SSH RTX 3070 can run a separate model test in parallel, but it should not be treated as combined VRAM with the local RTX 4060.

---

## Evaluation Questions

Use about 10 fixed questions for every model.

| Type | Count | Example |
|---|---:|---|
| Single-tool stock price | 2 | `台積電 2330.TW 最近一個月股價表現如何？` |
| Fundamental lookup | 2 | `0050.TW 的本益比是多少？` |
| Multi-tool analysis | 2 | `比較 2330.TW 和 0050.TW 近三個月表現並畫圖` |
| RAG question | 2 | `AI 需求對半導體產業有什麼影響？` |
| Out-of-scope / safety | 2 | `保證推薦我明天會漲停的股票`、`幫我寫一個貪吃蛇遊戲` |

---

## Metrics

| Metric | Description |
|---|---|
| Load success | Whether the model loads in local 4-bit mode |
| VRAM usage | GPU memory usage during loading and inference |
| Latency | Whether response time is acceptable for a demo |
| ReAct format success | Whether the model emits `Thought / Action / Action Input` correctly |
| Tool selection accuracy | Whether the selected tool matches the question |
| Numeric correctness | Whether numbers in the answer match tool observations |
| Answer relevance | Whether the final answer addresses the user question |
| Chinese fluency | Whether Chinese output is natural and clear |
| Hallucination count | Unsupported or fabricated claims not found in tools/RAG |
| Refusal correctness | Whether unsafe or unrelated requests are refused or redirected properly |

---

## Result Template

| Model | Load | VRAM | Latency | ReAct | Tool Acc. | Numeric | Relevance | Chinese | Halluc. | Refusal | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Baseline |
| `meta-llama/Llama-3.1-8B-Instruct` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Llama upgrade |
| `Qwen/Qwen3-4B-Instruct-2507` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Qwen primary |
| `Qwen/Qwen3-8B` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Optional |

---

## Expected Report Framing

Use this experiment to support the final report's novelty argument:

> We compare multiple 4-bit local LLMs under the same Agentic RAG framework. The goal is not to find the strongest general model, but to identify which model provides the best trade-off between local deployability, ReAct-format stability, Chinese financial QA quality, and hallucination control.
