import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ablation_engine import execute_ablation_suite
from src.model import load_model_4bit
import torch


def main():
    parser = argparse.ArgumentParser(description="Run complete ablation loops for a specific model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The Hugging Face model ID (e.g., meta-llama/Meta-Llama-3-8B-Instruct or Qwen/Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="evaluation/week4_questions.json",
        help="Path to the evaluation json file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to test (useful for fast debugging)"
    )
    parser.add_argument(
        "--only-ids",
        default="",
        help="Comma-separated question IDs to run, such as W4-Q09,W4-Q10."
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local Hugging Face cache."
    )

    args = parser.parse_args()

    print(f"\nPreparing ablation loop matrix for {args.model}")

    tokenizer, model = load_model_4bit(
        model_name=args.model,
        local_files_only=args.local_files_only,
    )
    load_memory = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    for setting in ["llm_only", "llm_tools", "full_suite"]:
        execute_ablation_suite(
            tokenizer=tokenizer,
            model=model,
            model_name=args.model,
            load_memory=load_memory,
            setting=setting,
            questions_path=args.questions,
            limit=args.limit,
            only_ids=args.only_ids,
        )

    print(f"\n   All 3 ablation settings completed for {args.model}!")

if __name__ == "__main__":
    main()
