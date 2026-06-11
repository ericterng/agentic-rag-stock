# Resource-Constrained Agentic RAG for Grounded Stock Market Analysis

Author: 滕彥宜 (Terng Yen-Yi)

This project builds a transparent text-based ReAct financial agent for grounded stock-market question answering. The goal is not to forecast stock prices or beat SOTA trading systems. Instead, we evaluate whether a local 4-bit Llama agent can choose financial tools, ground numeric answers in tool observations, use RAG for financial knowledge questions, and refuse unsafe or out-of-domain requests.

For the English final report and presentation, the main demonstration result is the fixed English Week 4 full-suite benchmark. The original Chinese/Taiwan-market benchmark is still kept as the original project setting and supporting result.

## What to Read

| File | Purpose |
|---|---|
| `docs/final_report_draft.md` | Main final-report draft |
| `docs/presentation_outline.md` | 22-slide presentation outline with English slide text |
| `evaluation/week4_llama_ablation_summary.md` | Full Week 4 ablation summary and error analysis |
| `evaluation/week4_full_suite_en_manual_scores.csv` | Manual scores for the fixed English full-suite run |
| `evaluation/week4_full_suite_manual_scores.csv` | Manual scores for the Chinese/Taiwan-market run |
| `week3_progress.md` | Verified Week 1-3 baseline: tools, RAG, Llama 3 8B 4-bit, and ReAct |

## Main Report Result

Fixed English full-suite run:

```text
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260611_030645.csv
ablation_outputs/evaluation/ablation_meta-llama__Meta-Llama-3-8B-Instruct_full_suite_20260611_030645.json
```

Reported result:

| Metric | Result |
|---|---:|
| Completed questions | 10 / 10 |
| ReAct format success | 10 / 10 |
| Tool selection accuracy | 10 / 10 |
| Manual numeric correctness | 6 / 6 |
| Refusal correctness | 2 / 2 |
| English fluency | 1.00 / 1.00 |

## Setup

Use a Python 3.11 CUDA environment:

```powershell
conda create -n pytorch python=3.11 -y
conda activate pytorch
pip install -r requirements.txt
copy .env.example .env
```

Required cached/downloaded models:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `BAAI/bge-m3`

If `data/vectordb/` is missing, rebuild the RAG database:

```powershell
python -c "from src.rag.loader import load_all_documents; from src.rag.retriever import build_vectordb; docs = load_all_documents(); build_vectordb(docs)"
```

## Reproduce English Result

On Windows, Chroma/yfinance SQLite caches may fail inside the project folder with `disk I/O error`. The command below copies the vector DB to a temp folder and points `VECTORDB_PATH` there. yfinance cache is already configured to use the system temp directory.

```powershell
$runtime = Join-Path $env:TEMP 'agentic_rag_stock_vectordb'
if (Test-Path $runtime) { Remove-Item -LiteralPath $runtime -Recurse -Force }
Copy-Item -Path data\vectordb -Destination $runtime -Recurse

$env:VECTORDB_PATH = $runtime
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

python ablation_scripts\run_ablation.py `
  --model meta-llama/Meta-Llama-3-8B-Instruct `
  --local-files-only `
  --settings full_suite `
  --questions evaluation/week4_questions_en.json `
  --max-iterations 6 `
  --max-new-tokens 384 `
  --deterministic
```

## Reproduce Original Chinese Benchmark

Use the same temp vector DB setup if starting from a new PowerShell session:

```powershell
$runtime = Join-Path $env:TEMP 'agentic_rag_stock_vectordb'
if (Test-Path $runtime) { Remove-Item -LiteralPath $runtime -Recurse -Force }
Copy-Item -Path data\vectordb -Destination $runtime -Recurse

$env:VECTORDB_PATH = $runtime
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

python ablation_scripts\run_ablation.py `
  --model meta-llama/Meta-Llama-3-8B-Instruct `
  --local-files-only `
  --settings full_suite `
  --questions evaluation/week4_questions.json `
  --max-iterations 6 `
  --max-new-tokens 384 `
  --deterministic
```

Note: yfinance stock/news data are live, so exact prices, headlines, chart filenames, and latency may differ across reruns. Model weights, `.env`, Hugging Face caches, and generated zip files are not included in the repository.
