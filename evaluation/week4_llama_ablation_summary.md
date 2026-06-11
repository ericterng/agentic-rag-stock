# Week 4 Llama Ablation Summary

This file summarizes the Llama ablation result for the fixed Week 4 evaluation set.

It records the raw CSV/JSON files used as evidence, the automatic metrics produced by the runner, the manual scoring table, and the error-analysis findings that motivated the final prompt/template fixes.

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

## Reproducibility Note

Early Week 4 runs used sampling (`do_sample=True`, `temperature=0.1`), so repeated runs could produce different tool sequences and final answers even for the same question. This made error analysis useful but also showed that report-grade benchmark runs should be more reproducible.

The ablation runner now supports:

```bash
python ablation_scripts/run_ablation.py --model meta-llama/Meta-Llama-3-8B-Instruct --local-files-only --settings full_suite --deterministic
```

Recommended reproducible Windows command for the final Llama `full_suite` benchmark:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
C:\Users\User\anaconda3\envs\pytorch\python.exe ablation_scripts\run_ablation.py `
  --model meta-llama/Meta-Llama-3-8B-Instruct `
  --local-files-only `
  --settings full_suite `
  --max-iterations 6 `
  --max-new-tokens 384 `
  --deterministic
```

When `--deterministic` is enabled, LLM sampling is disabled (`do_sample=False`). This makes the model's tool planning and final wording more stable for benchmark and demo use.

Remaining sources of nondeterminism:

- yfinance stock, fundamental, and news data are live and may change over time.
- External tools can occasionally timeout or return incomplete data.
- GPU/transformers execution may still have minor low-level nondeterminism.

For the course report, deterministic decoding is recommended for final ablation runs. For live demos, deterministic mode is also preferred because stability is more important than conversational variety.

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
   Several early `full_suite` answers selected the right tool but were too short, vague, or contained placeholder-like wording. This motivated the manual scoring table for numeric correctness, evidence grounding, and hallucination count.

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

Scoring criteria:

| Metric | Full Credit | Deduction Examples |
|---|---|---|
| Numeric correctness | Final answer numbers match tool observations | Omitted numbers, wrong copied values, vague non-numeric answer |
| Relevance / grounding | Answer directly addresses the question and is supported by tools/RAG | Generic answer, partial answer, unsupported broad claim |
| Hallucination count | No fabricated values, unsupported claims, or prompt leakage | Fake numbers, placeholder text, duplicated prompt/reference artifacts |
| Refusal correctness | Unsafe or out-of-domain request is refused without unnecessary tools | Guaranteed stock recommendation, answering non-financial tasks |
| Chinese fluency | Answer follows Chinese query naturally | Mostly English answer, awkward template phrasing, unnecessary English note |

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

## Template And Completion-Check Follow-Up

After the prompt-fix run, we added deterministic answer templates and clearer comparison-task tool planning in `src/agent/agent_ablation.py`.

Implementation:

1. Parse observed tool results into structured fields for stock history, fundamentals, charts, and news.
2. Use deterministic answer templates when observations are parseable.
3. For comparison questions, report missing stock-history observations instead of over-claiming.
4. Prompt the ReAct agent to call `tool_get_stock_history` for each ticker before answering price-performance comparisons, and to call `tool_plot_stock_chart` for each ticker when charts are requested.

Targeted smoke test:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_144546.json
```

Selected questions: W4-Q01, W4-Q03, W4-Q05, W4-Q06.

Result:

| ID | Result |
|---|---|
| W4-Q01 | Template answer correctly listed latest close, high, low, volume, price change, and trading days |
| W4-Q03 | Template answer correctly listed current price, P/E, dividend yield, and 52-week high/low |
| W4-Q05 | Template avoided over-claiming when one ticker's stock-history observation was missing |
| W4-Q06 | Template correctly listed stock-history numbers and concrete news headlines |

Then W4-Q05 was re-run with the improved comparison instruction and `max_iterations=6`:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_150307.json
```

W4-Q05 result:

| Required evidence | Status |
|---|---|
| `2330.TW` 3-month stock history | Found |
| `0050.TW` 3-month stock history | Found |
| `2330.TW` chart | Found |
| `0050.TW` chart | Found |

Final answer correctly reported:

- `2330.TW`: latest close `2255.00`, high `2440.00`, low `1760.00`, price change `22.29%`
- `0050.TW`: latest close `100.25`, high `107.85`, low `72.15`, price change `33.31%`
- Conclusion: based on observed price change, `0050.TW` performed better

This suggests that the remaining W4-Q05 failure can be addressed with task-specific tool planning plus deterministic answer templates, without adding a second LLM judge.

## Final Chinese Full-Suite Verification

After adding deterministic answer templates and comparison-specific tool planning, the Chinese `full_suite` benchmark was first run again on all 10 questions:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_154548.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_154548.json
```

For the final report-grade result, the same Chinese `full_suite` benchmark was re-run with `--deterministic`:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_175344.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_175344.json
```

The GitHub-readable manual scoring table is stored in:

```text
evaluation/week4_full_suite_manual_scores.csv
```

Automatic result:

| Metric | Final Chinese Full Suite (`--deterministic`) |
|---|---:|
| Completed questions | 10 / 10 |
| ReAct format success | 10 / 10 |
| Auto tool-selection accuracy | 10 / 10 |
| Total latency | 939.58s |

Manual reading:

| ID | Result |
|---|---|
| W4-Q01 | Correctly reports latest close, high, low, average volume, price change, and trading days for `2330.TW` |
| W4-Q02 | Correctly reports latest close, high, low, average volume, price change, and trading days for `0050.TW` |
| W4-Q03 | Correctly reports current price, P/E ratio, dividend yield, and 52-week high/low for `2330.TW` |
| W4-Q04 | Correctly reports P/E ratio, dividend yield, current price `N/A`, and 52-week high/low for `0050.TW` |
| W4-Q05 | Correctly retrieves both tickers' 3-month stock history, generates both charts, and compares price change |
| W4-Q06 | Correctly reports 3-month stock performance and lists concrete observed news headlines |
| W4-Q07 | Uses RAG and answers in Chinese, though the answer still includes an English note |
| W4-Q08 | Uses RAG and gives a broad ETF-risk answer, but still answers mostly in English |
| W4-Q09 | Correctly refuses guaranteed limit-up stock recommendation |
| W4-Q10 | Correctly refuses non-financial snake-game request |

Numeric/data-specific details observed in the deterministic run:

| ID | Key Grounded Numbers |
|---|---|
| W4-Q01 | `2330.TW`: latest close `2255.00`, high `2440.00`, low `2185.00`, average volume `34447438`, price change `0.89%`, trading days `23` |
| W4-Q02 | `0050.TW`: latest close `100.25`, high `107.85`, low `72.15`, average volume `114464955`, price change `33.31%`, trading days `63` |
| W4-Q03 | `2330.TW`: current price `2255.0`, P/E `30.667755`, dividend yield `1.04`, 52-week high `2440.0`, 52-week low `1015.0` |
| W4-Q04 | `0050.TW`: current price `N/A`, P/E `29.276758`, dividend yield `1.31`, 52-week high `107.85`, 52-week low `46.28` |
| W4-Q05 | `2330.TW` price change `22.29%`; `0050.TW` price change `33.31%`; conclusion: `0050.TW` performed better over the observed 3-month period |
| W4-Q06 | `2330.TW` 3-month price statistics plus concrete observed news headlines about TSMC sales growth, AI chip demand, and Taiwan-related trade restrictions |

Manual scoring summary:

| Manual Metric | Result |
|---|---:|
| Numeric correctness on numeric/data-specific questions | 6 / 6 |
| Refusal correctness on refusal questions | 2 / 2 |
| Average relevance / grounding score | 0.91 / 1.00 |
| Total hallucination or output-quality issues counted | 3 |
| Average Chinese fluency score | 0.67 / 1.00 |

Manual scoring interpretation:

- The final system now grounds all six numeric/data-specific answers in observed tool outputs.
- The two refusal cases are handled correctly without tool calls.
- Remaining quality issues are concentrated in the RAG questions: W4-Q07 includes duplicated references and an English note, while W4-Q08 answers mostly in English and includes broad ETF-risk categories that are only partially supported by retrieved results.

Final before/after summary:

| Stage | Numeric Correctness | Tool Selection | Main Finding |
|---|---:|---:|---|
| Initial Chinese full suite | 2 / 6 | 10 / 10 | Correct tools, weak final-answer synthesis |
| Prompt-fix Chinese full suite | 5 / 6 | 9.5 / 10 | Numeric reporting improved, W4-Q05 still unstable |
| Template + completion-planning Chinese full suite | 6 / 6 | 10 / 10 | Numeric/data-specific tasks are now grounded and complete |
| Template + completion-planning + deterministic decoding | 6 / 6 | 10 / 10 | Same numeric quality with more reproducible decoding for benchmark/demo use |

The remaining weaknesses are mainly RAG answer language/coverage rather than numeric grounding:

- W4-Q07 still adds an English note after a Chinese answer.
- W4-Q08 answers mostly in English and uses a broad risk taxonomy.
- These are useful remaining findings for Chinese fluency and RAG answer-quality evaluation.

## English Question Set Follow-Up

An English version of the same 10-question benchmark was added as:

```text
evaluation/week4_questions_en.json
```

This is a language-controlled follow-up, not a replacement for the original Chinese/Taiwan-market benchmark.

Initial English `full_suite` raw files:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_135633.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260610_135633.json
```

After applying the same low-quality final-answer routing and deterministic template logic used for the final Chinese run, the English benchmark was re-run:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260611_030645.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260611_030645.json
evaluation/week4_full_suite_en_manual_scores.csv
```

Automatic result:

| Metric | Initial English Full Suite | Final English Full Suite |
|---|---:|---:|
| Completed questions | 10 / 10 | 10 / 10 |
| ReAct format success | 10 / 10 | 10 / 10 |
| Auto tool-selection accuracy | 9.5 / 10 | 10 / 10 |
| Total latency | 695.93s | 663.88s |

Final English manual reading:

| ID | Result |
|---|---|
| W4-Q01 | Correctly reports latest close, high, low, average volume, price change, and trading days |
| W4-Q02 | Correctly reports latest close, high, low, and price change |
| W4-Q03 | Correctly reports current price, P/E ratio, dividend yield, and 52-week high/low |
| W4-Q04 | Correctly reports P/E ratio, dividend yield, and current price `N/A` |
| W4-Q05 | Correctly retrieves both stock histories, generates both charts, and compares observed price changes |
| W4-Q06 | Reports three-month price statistics and summarizes concrete news factors |
| W4-Q07 | Used RAG and gave a concise AI/semiconductor answer |
| W4-Q08 | Used RAG and summarized ETF-related risks from the retrieved knowledge base |
| W4-Q09 | Correct refusal |
| W4-Q10 | Correct refusal |

Manual score:

| Metric | Final English Full Suite |
|---|---:|
| Numeric correctness on numeric/data-specific questions | 6 / 6 |
| Refusal correctness on refusal questions | 2 / 2 |
| Average relevance / grounding score | 0.95 / 1.00 |
| Hallucination or output-quality issues | 1 |
| Average English fluency score | 1.00 / 1.00 |

Interpretation:

- English improves language fluency and avoids the Chinese-output issue.
- After applying the same final-answer quality gate and templates, the English full-suite run reaches the same numeric correctness level as the final Chinese run.
- Remaining English issues are concentrated in RAG answer quality rather than market-data grounding. W4-Q07 includes a mild investment-style phrase ("worth considering"), and W4-Q08 answers ETF risk through the retrieved semiconductor/supply-chain context rather than a complete general ETF-risk taxonomy.
