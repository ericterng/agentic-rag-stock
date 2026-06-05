import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.agent import create_agent_graph
from src.config import MODEL_NAME
from src.model import load_model_4bit


VALID_TOOLS = {
    "tool_get_stock_history",
    "tool_get_fundamental_data",
    "tool_plot_stock_chart",
    "tool_search_financial_news",
    "tool_search_knowledge_base",
}
ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE)
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
OBS_RE = re.compile(r"Observation:\s*(.*?)(?=\nThought:|\nAction:|\nFinal Answer:|\Z)", re.IGNORECASE | re.DOTALL)

CSV_FIELDS = [
    "question_id",
    "category",
    "question",
    "expected_tools",
    "expected_refusal",
    "model_name",
    "load_success",
    "vram_after_load_mb",
    "peak_vram_reserved_mb",
    "latency_seconds",
    "iterations",
    "actions",
    "raw_actions",
    "action_inputs",
    "react_format_success_auto",
    "tool_selection_accuracy_auto",
    "final_answer",
    "numeric_correctness_manual",
    "answer_relevance_manual",
    "evidence_grounding_manual",
    "hallucination_count_manual",
    "refusal_correctness_manual",
    "chinese_fluency_manual",
    "notes",
]


def gpu_memory_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    return round(torch.cuda.memory_allocated() / 1024 / 1024)


def gpu_peak_reserved_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    return round(torch.cuda.max_memory_reserved() / 1024 / 1024)


def parse_scratchpad(scratchpad: str) -> dict:
    raw_actions = [a.strip() for a in ACTION_RE.findall(scratchpad)]
    actions = [a for a in raw_actions if a in VALID_TOOLS]
    action_inputs = [a.strip() for a in INPUT_RE.findall(scratchpad)]
    observations = [o.strip() for o in OBS_RE.findall(scratchpad)]
    final_match = FINAL_RE.search(scratchpad)
    final_answer = final_match.group(1).strip() if final_match else ""
    return {
        "actions": actions,
        "raw_actions": raw_actions,
        "action_inputs": action_inputs,
        "observations": observations,
        "final_answer_from_scratchpad": final_answer,
    }


def tool_selection_accuracy(expected_tools: list[str], actual_tools: list[str], expected_refusal: bool) -> str:
    if expected_refusal:
        return "pass" if not actual_tools else "fail"
    if not expected_tools:
        return "n/a"
    actual_set = set(actual_tools)
    expected_set = set(expected_tools)
    return "pass" if expected_set.issubset(actual_set) else "fail"


def react_format_success(actual_tools: list[str], final_answer: str, expected_refusal: bool) -> str:
    if actual_tools:
        return "pass"
    if expected_refusal and final_answer:
        return "pass"
    return "fail"


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Week 4 evaluation set against the ReAct agent.")
    parser.add_argument("--questions", default="evaluation/week4_questions.json")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true", help="Load model files only from the local Hugging Face cache.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    args = parser.parse_args()

    question_path = ROOT / args.questions
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(question_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        questions = questions[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.model_name.replace("/", "__")
    json_path = output_dir / f"week4_baseline_{slug}_{timestamp}.json"
    csv_path = output_dir / f"week4_baseline_{slug}_{timestamp}.csv"

    print(f"Loading model: {args.model_name}")
    load_start = time.perf_counter()
    tokenizer, model = load_model_4bit(args.model_name, local_files_only=args.local_files_only)
    load_seconds = round(time.perf_counter() - load_start, 2)
    load_memory = gpu_memory_mb()
    print(f"Load completed in {load_seconds}s; allocated VRAM: {load_memory} MB")

    graph, _ = create_agent_graph(
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
        max_iterations=args.max_iterations,
    )
    results = []

    for index, item in enumerate(questions, 1):
        question_id = item["id"]
        print(f"[{index}/{len(questions)}] {question_id}: {item['question']}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        state = {
            "input": item["question"],
            "history": "",
            "scratchpad": "",
            "output": "",
            "iterations": 0,
        }
        config = {"configurable": {"thread_id": f"week4-{timestamp}-{question_id}"}}

        error = ""
        try:
            result = graph.invoke(state, config)
        except Exception as exc:
            result = {
                "input": item["question"],
                "history": "",
                "scratchpad": "",
                "output": "",
                "iterations": 0,
            }
            error = repr(exc)

        latency = round(time.perf_counter() - start, 2)
        scratchpad = result.get("scratchpad", "")
        parsed = parse_scratchpad(scratchpad)
        final_answer = result.get("output") or parsed["final_answer_from_scratchpad"]
        actions = parsed["actions"]
        expected_tools = item.get("expected_tools", [])
        expected_refusal = bool(item.get("expected_refusal", False))

        row = {
            "question_id": question_id,
            "category": item.get("category", ""),
            "question": item["question"],
            "expected_tools": ";".join(expected_tools),
            "expected_refusal": expected_refusal,
            "model_name": args.model_name,
            "load_success": True,
            "vram_after_load_mb": load_memory,
            "peak_vram_reserved_mb": gpu_peak_reserved_mb(),
            "latency_seconds": latency,
            "iterations": result.get("iterations", 0),
            "actions": ";".join(actions),
            "raw_actions": ";".join(parsed["raw_actions"]),
            "action_inputs": ";".join(parsed["action_inputs"]),
            "react_format_success_auto": react_format_success(actions, final_answer, expected_refusal),
            "tool_selection_accuracy_auto": tool_selection_accuracy(expected_tools, actions, expected_refusal),
            "final_answer": final_answer,
            "numeric_correctness_manual": "",
            "answer_relevance_manual": "",
            "evidence_grounding_manual": "",
            "hallucination_count_manual": "",
            "refusal_correctness_manual": "",
            "chinese_fluency_manual": "",
            "notes": error,
        }

        results.append({
            "metadata": item,
            "row": row,
            "scratchpad": scratchpad,
            "observations": parsed["observations"],
        })

        print(
            f"  actions={row['actions'] or '(none)'} "
            f"latency={latency}s "
            f"tool_auto={row['tool_selection_accuracy_auto']}"
        )

        write_json(json_path, {
            "model_name": args.model_name,
            "load_seconds": load_seconds,
            "vram_after_load_mb": load_memory,
            "created_at": timestamp,
            "max_new_tokens": args.max_new_tokens,
            "max_iterations": args.max_iterations,
            "results": results,
        })
        write_csv(csv_path, [result["row"] for result in results])

    write_json(json_path, {
        "model_name": args.model_name,
        "load_seconds": load_seconds,
        "vram_after_load_mb": load_memory,
        "created_at": timestamp,
        "max_new_tokens": args.max_new_tokens,
        "max_iterations": args.max_iterations,
        "results": results,
    })

    write_csv(csv_path, [result["row"] for result in results])

    print(f"JSON results: {json_path}")
    print(f"CSV results:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
