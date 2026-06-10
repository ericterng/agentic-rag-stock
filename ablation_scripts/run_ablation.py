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
    parser.add_argument(
        "--settings",
        default="llm_only,llm_tools,full_suite",
        help="Comma-separated ablation settings to run: llm_only,llm_tools,full_suite."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum ReAct tool iterations per question."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens generated per model call."
    )

    args = parser.parse_args()

    print(f"\nPreparing ablation loop matrix for {args.model}")

    tokenizer, model = load_model_4bit(
        model_name=args.model,
        local_files_only=args.local_files_only,
    )
    load_memory = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    settings = [item.strip() for item in args.settings.split(",") if item.strip()]
    valid_settings = {"llm_only", "llm_tools", "full_suite"}
    invalid_settings = [item for item in settings if item not in valid_settings]
    if invalid_settings:
        raise ValueError(f"Invalid ablation settings: {', '.join(invalid_settings)}")

    for setting in settings:
        execute_ablation_suite(
            tokenizer=tokenizer,
            model=model,
            model_name=args.model,
            load_memory=load_memory,
            setting=setting,
            questions_path=args.questions,
            limit=args.limit,
            only_ids=args.only_ids,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
        )

    print(f"\n   Ablation settings completed for {args.model}: {', '.join(settings)}")

if __name__ == "__main__":
    main()
