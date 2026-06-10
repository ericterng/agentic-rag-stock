# Resource-Constrained Agentic RAG for Grounded Stock Market Analysis

Author: 滕彥宜 (Teng Yan-Yi)

This project builds a transparent text-based ReAct financial agent for grounded stock-market question answering. The goal is not to predict stock prices or beat SOTA trading systems, but to evaluate whether a local 4-bit Llama agent can choose tools correctly, ground numeric answers in observations, use RAG for financial knowledge questions, and refuse unsafe or out-of-domain requests.

## What to Read

| File | Purpose |
|---|---|
| `docs/final_report_draft.md` | Main final-report draft and latest research narrative |
| `evaluation/week4_llama_ablation_summary.md` | Full Week 4 ablation results, error analysis, and final deterministic run |
| `evaluation/week4_full_suite_manual_scores.csv` | Manual scoring table for the final full-suite run |
| `week4_evaluation.md` | Week 4 evaluation overview and validation notes |
| `week3_progress.md` | Verified Week 1-3 tools, RAG, and ReAct baseline |
| `model_ablation_plan.md` | Planned Llama/Qwen model comparison |

## Reproduce the Final Week 4 Result

Install dependencies in a Python 3.11 CUDA environment:

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

Run the final deterministic benchmark:

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

Expected reported result: completed questions `10/10`, ReAct format `10/10`, tool selection `10/10`, manual numeric correctness `6/6`, refusal correctness `2/2`, average relevance/grounding `0.91`, and Chinese fluency `0.67`.

Note: yfinance stock/news data are live, so exact prices, headlines, and latency may differ across reruns. Model weights, `.env`, and Hugging Face caches are not included in the repository or submission zip.
