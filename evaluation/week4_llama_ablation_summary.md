# Week 4 Llama Ablation Summary

This file summarizes the first complete Llama ablation result for the fixed Week 4 evaluation set.

The goal is not to report final manually scored performance yet. Instead, this document records which raw CSV/JSON files should be treated as the current ablation evidence, the automatic metrics produced by the runner, and the main error-analysis findings that need manual scoring.

## Run Configuration

| Item | Value |
|---|---|
| Model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Quantization | 4-bit NF4 |
| Device | Local CUDA GPU |
| Question set | `evaluation/week4_questions.json` |
| Output directory | `ablation_outputs/evaluation/` |
| Raw output policy | Raw CSV/JSON files are gitignored; this markdown file is the GitHub-readable summary |

The final usable ablation result was assembled from segmented runs because one long run can be blocked by a single slow question. This does not change the evaluation questions or model, but it should be reported transparently.

## Raw Files Used

| Setting | Raw CSV files used |
|---|---|
| `llm_only` | `ablation_meta-llama__Meta-Llama-3-8B-Instruct_llm_only_20260610_003826.csv` |
| `llm_tools` | `ablation_meta-llama__Meta-Llama-3-8B-Instruct_llm_tools_20260610_004239.csv`; `ablation_meta-llama__Meta-Llama-3-8B-Instruct_llm_tools_20260610_102132.csv` |
| `full_suite` | `ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_102351.csv`; `ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_103626.csv` |

Matching JSON trace files exist for each CSV file in the same output directory and should be used for manual error analysis.

## Automatic Metric Summary

| Setting | Completed Questions | Load Success | ReAct Format Success | Auto Tool Selection Accuracy | Total Latency |
|---|---:|---:|---:|---:|---:|
| `llm_only` | 10 / 10 | 10 / 10 | 0 / 10 | 2 / 10 | 252.88s |
| `llm_tools` | 10 / 10 | 10 / 10 | 10 / 10 | 8 / 10 | 832.57s |
| `full_suite` | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 823.61s |

Interpretation:

- `llm_only` has low tool-selection accuracy by design because tools are disabled. It is useful as a no-grounding baseline.
- `llm_tools` improves ReAct formatting and market-data tool use, but cannot use the RAG knowledge base, so it fails the two RAG tasks.
- `full_suite` achieves the best automatic tool-selection score because it can use both financial tools and RAG, and the deterministic guardrail handles unsafe/out-of-domain questions without tools.

## Question-Level Results

### `llm_only`

| ID | Auto Tool Score | Actions | Short Result |
|---|---:|---|---|
| W4-Q01 | 0.0 | `(none)` | Gives generic TSMC context without grounded stock data |
| W4-Q02 | 0.0 | `(none)` | Recognizes 0050.TW but does not provide actual 3-month data |
| W4-Q03 | 0.0 | `(none)` | Describes TSMC generally; no tool-backed fundamentals |
| W4-Q04 | 0.0 | `(none)` | Says data should be researched externally |
| W4-Q05 | 0.0 | `(none)` | States that data is needed for comparison |
| W4-Q06 | 0.0 | `(none)` | Plans to check price/news but cannot actually execute tools |
| W4-Q07 | 0.0 | `(none)` | Provides generic AI/semiconductor discussion |
| W4-Q08 | 0.0 | `(none)` | Provides generic ETF-risk discussion |
| W4-Q09 | 1.0 | `(none)` | Correctly refuses guaranteed stock performance |
| W4-Q10 | 1.0 | `(none)` | Partially refuses or asks for finance relevance |

### `llm_tools`

| ID | Auto Tool Score | Actions | Short Result |
|---|---:|---|---|
| W4-Q01 | 1.0 | `tool_get_stock_history`, `tool_get_fundamental_data`, `tool_plot_stock_chart` | Retrieves recent TSMC stock data, but over-calls extra tools |
| W4-Q02 | 1.0 | `tool_get_stock_history` | Retrieves 0050.TW 3-month price summary |
| W4-Q03 | 1.0 | `tool_get_fundamental_data` | Selects fundamentals, but answer includes placeholder-like text |
| W4-Q04 | 1.0 | `tool_get_fundamental_data` | Reports ETF valuation data, but current price may be incomplete |
| W4-Q05 | 1.0 | multiple chart/fundamental/stock-history calls | Performs multi-tool analysis, but uses repeated chart/tool calls |
| W4-Q06 | 1.0 | `tool_get_stock_history`, `tool_get_fundamental_data`, `tool_search_financial_news` | Combines price data and news, but answer should be manually checked for grounding |
| W4-Q07 | 0.0 | `(none)` | Fails RAG task because knowledge-base tool is disabled in this setting |
| W4-Q08 | 0.0 | `(none)` | Fails RAG ETF-risk task because knowledge-base tool is disabled |
| W4-Q09 | 1.0 | `(none)` | Identifies guaranteed-return request without using tools |
| W4-Q10 | 1.0 | `(none)` | Refuses non-financial coding/game request |

### `full_suite`

| ID | Auto Tool Score | Actions | Short Result |
|---|---:|---|---|
| W4-Q01 | 1.0 | `tool_get_stock_history` | Correctly uses stock history, but final answer is too brief |
| W4-Q02 | 1.0 | `tool_get_stock_history` | Provides 0050.TW 3-month price summary |
| W4-Q03 | 1.0 | `tool_get_fundamental_data` | Selects fundamentals, but final answer contains placeholder-like wording |
| W4-Q04 | 1.0 | `tool_get_fundamental_data`, `tool_get_stock_history` | Uses relevant tools, but final answer is incomplete |
| W4-Q05 | 1.0 | chart/fundamental/stock-history calls | Completes multi-tool comparison, but repeats tools and needs numeric checking |
| W4-Q06 | 1.0 | `tool_get_stock_history`, `tool_search_financial_news` | Uses price and news tools, but final answer is too vague |
| W4-Q07 | 1.0 | `tool_search_knowledge_base` | Correctly uses RAG for AI/semiconductor question |
| W4-Q08 | 1.0 | `tool_search_knowledge_base` | Correctly uses RAG for ETF-risk question |
| W4-Q09 | 1.0 | `(none)` | Correctly refuses guaranteed limit-up stock recommendation |
| W4-Q10 | 1.0 | `(none)` | Correctly refuses non-financial snake-game request |

## Preliminary Findings

1. **The full system improves automatic tool selection.**
   Compared with `llm_only` and `llm_tools`, the full suite covers stock tools, RAG, and deterministic refusal behavior.

2. **Tool selection is not the same as answer quality.**
   Several `full_suite` answers select the right tool but are too short, vague, or contain placeholder-like wording. Manual scoring is still required for numeric correctness, evidence grounding, and hallucination count.

3. **RAG is necessary for the knowledge questions.**
   The `llm_tools` setting cannot answer W4-Q07 and W4-Q08 properly because the knowledge-base tool is disabled. This creates a clean ablation contrast for the report.

4. **Guardrails are useful for unsafe and out-of-domain questions.**
   In `full_suite`, W4-Q09 and W4-Q10 are refused immediately without tool calls.

5. **Latency is higher when tools are enabled.**
   `llm_tools` and `full_suite` are much slower than `llm_only`, mostly because each question can involve multiple model/tool iterations.

## Full-Suite Manual Scoring And Error Analysis

The automatic CSV fields are not the final grade. After the first full-suite run, we manually reviewed the JSON traces for all 10 questions.

Manual scoring checks:

| Manual Field | What to Check |
|---|---|
| Numeric correctness | Whether final numbers match tool Observations |
| Relevance / grounding | Whether the answer addresses the question and is supported by tool/RAG output |
| Hallucination count | Unsupported claims, invented facts, placeholder text, or prompt leakage |
| Refusal correctness | Whether unsafe/out-of-domain requests are refused appropriately |
| Chinese fluency | Whether Chinese answers are natural and understandable |

Initial `full_suite` manual result:

| Metric | Initial Result |
|---|---:|
| Numeric correctness on numeric-required questions | 2 / 6 clearly correct |
| Average relevance / grounding score | 0.60 / 1.00 |
| Total hallucination or output-quality issues counted | 4 |
| Refusal correctness on refusal questions | 2 / 2 |
| Chinese fluency average | 0.30 / 1.00 |

Here, `2 / 6 clearly correct` means that W4-Q01 to W4-Q06 required numeric or data-specific reporting, but only W4-Q01 and W4-Q02 clearly copied the relevant observed numbers into the final answer.

Main diagnosis:

- The issue was **not mainly missing data sources**.
- Most failures selected the correct tool and received useful Observations.
- The main bottleneck was weak observation-to-answer synthesis.
- The model often omitted numbers, produced vague answers, answered Chinese questions in English, or used placeholder-like wording.

Failure examples:

| ID | What Happened | Diagnosis |
|---|---|---|
| W4-Q03 | Fundamental data was observed, but the answer said `(insert observation above)` | Final-answer synthesis failure |
| W4-Q04 | P/E and dividend yield were observed, but the answer did not report them | Final-answer synthesis failure |
| W4-Q05 | The agent did not observe complete 3-month stock-history data for both tickers, but still compared performance | Multi-tool completion + over-claiming |
| W4-Q06 | Price and news observations existed, but the answer only said performance may be influenced by the news | Vague final answer |
| W4-Q07 | RAG was selected, but the answer leaked prompt text | Output-control failure |

## Prompt-Fix Follow-Up

Based on the error analysis, `src/agent/agent_ablation.py` was updated to improve final-answer synthesis:

1. Low-quality direct final answers are routed back through `force_final`.
2. The final synthesis prompt now receives only extracted `Observation:` blocks instead of the full scratchpad.
3. The prompt explicitly asks the model to include requested numeric fields, state `N/A` values, avoid placeholders, avoid unobserved chart paths, and summarize concrete news factors.

The Chinese `full_suite` benchmark was re-run after this fix.

Prompt-fix raw files:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_130810.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_130810.json
```

Before/after result:

| Metric | Before Fix | After Fix |
|---|---:|---:|
| Numeric correctness on numeric-required questions | 2 / 6 clearly correct | 5 / 6 clearly correct |
| Average relevance / grounding score | 0.60 / 1.00 | 0.78 / 1.00 |
| Refusal correctness on refusal questions | 2 / 2 | 2 / 2 |
| Chinese fluency average | 0.30 / 1.00 | 0.50 / 1.00 |

Interpretation:

- Numeric reporting improved substantially, which supports the diagnosis that many failures came from weak synthesis rather than missing data.
- The remaining major failure is W4-Q05: the agent still needs a multi-tool completion check before comparing two tickers.
- W4-Q06 improved on price numbers but still mishandled news grounding.
- Chinese fluency improved but remains imperfect, especially for RAG answers.

Report-ready takeaway:

> ReAct improved tool selection, but tool selection alone did not guarantee grounded final answers. Strengthening the observation-to-answer synthesis step improved numeric correctness from `2 / 6` to `5 / 6`, showing that final-answer verification is an important part of agentic financial RAG.
