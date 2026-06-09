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

from src.agent.agent_ablation import create_agent_graph


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
    "ablation_setting",
    "load_success",
    "vram_after_load_mb",
    "peak_vram_reserved_mb",
    "latency_seconds",
    "iterations",
    "actions",
    "raw_actions",
    "action_inputs",
    "final_answer",
    "react_format_success",
    "tool_selection_accuracy_auto",
    "numeric_correctness_manual",
    "relevance_grounding_manual",
    "hallucination_count_manual",
    "refusal_correctness_manual",
    "chinese_fluency_manual",
    "notes",
]


def enabled_tools_for_setting(setting: str) -> set[str]:
    if setting == "llm_only":
        return set()
    if setting == "llm_tools":
        return VALID_TOOLS - {"tool_search_knowledge_base"}
    return set(VALID_TOOLS)


def parse_scratchpad(scratchpad: str, enabled_tools: set[str]) -> dict:
    raw_actions = [a.strip() for a in ACTION_RE.findall(scratchpad)]
    actions = [a for a in raw_actions if a in enabled_tools]
    action_inputs = [a.strip() for a in INPUT_RE.findall(scratchpad)]
    final_match = FINAL_RE.search(scratchpad)
    final_answer = final_match.group(1).strip() if final_match else ""

    has_thought = "thought:" in scratchpad.lower()
    has_action_or_final = bool(raw_actions) or final_match is not None or "final answer:" in scratchpad.lower()
    react_format = 1.0 if has_thought and has_action_or_final else 0.0

    return {
        "actions": actions,
        "raw_actions": raw_actions,
        "action_inputs": action_inputs,
        "final_answer": final_answer,
        "react_format_success": react_format,
        "observations": [o.strip() for o in OBS_RE.findall(scratchpad)],
    }


def evaluate_tools(predicted_actions: list[str], expected_str: str) -> float:
    if not expected_str or expected_str.strip().lower() == "none":
        return 1.0 if len(predicted_actions) == 0 else 0.0

    expected_set = {t.strip() for t in re.split(r"[,;]", expected_str) if t.strip()}
    pred_set = set(predicted_actions)

    if len(expected_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    return float(len(expected_set.intersection(pred_set)) / len(expected_set))


def write_outputs(csv_path: Path, json_path: Path, results: list[dict]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([item["row"] for item in results])

    with open(json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=2)


def execute_ablation_suite(
    tokenizer,
    model,
    model_name,
    load_memory,
    setting: str,
    questions_path: str = "evaluation/week4_questions.json",
    limit: int = None,
    only_ids: str = "",
):
    print("\n=======================================================")
    print(f"STARTING ABLATION: {model_name} | MODE: {setting}")
    print("=======================================================")

    graph, _ = create_agent_graph(tokenizer, model, ablation_mode=setting)
    enabled_tools = enabled_tools_for_setting(setting)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = model_name.replace("/", "__")
    output_dir = Path("ablation_outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"ablation_{sanitized_model}_{setting}_{timestamp}.json"
    csv_path = output_dir / f"ablation_{sanitized_model}_{setting}_{timestamp}.csv"

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data["questions"]
    if only_ids:
        wanted = {item.strip() for item in only_ids.split(",") if item.strip()}
        items = [item for item in items if item.get("id", item.get("question_id", "")) in wanted]
    if limit:
        items = items[:limit]
    if not items:
        raise ValueError("No questions selected for ablation evaluation.")

    results = []

    for i, item in enumerate(items):
        response = None
        q_id = item.get("id", item.get("question_id", f"Q_{i}"))
        category = item.get("category", "General")
        question = item.get("question", "")
        expected_tools_raw = item.get("expected_tools", [])
        expected_tools = ";".join(expected_tools_raw) if isinstance(expected_tools_raw, list) else expected_tools_raw
        expected_refusal = item.get("expected_refusal", "")

        print(f"[{i + 1}/{len(items)}] Evaluating {q_id}...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        try:
            response = graph.invoke({
                "input": question,
                "history": "",
                "scratchpad": "",
                "output": "",
                "iterations": 0,
            }, config={"configurable": {"thread_id": f"{setting}-{q_id}"}})

            scratchpad = response.get("scratchpad", "")
            parsed = parse_scratchpad(scratchpad, enabled_tools)

            latency = round(time.time() - t0, 2)
            peak_vram = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
            iterations = response.get("iterations", len(parsed["actions"]) + 1)
            tool_acc = evaluate_tools(parsed["actions"], expected_tools)
            error_msg = ""
        except Exception as e:
            print(f"  Error executing {q_id}: {e}")
            parsed = {
                "actions": [],
                "raw_actions": [],
                "action_inputs": [],
                "final_answer": "",
                "react_format_success": 0.0,
                "observations": [],
            }
            latency, peak_vram, iterations, tool_acc = 0.0, 0.0, 0, 0.0
            scratchpad = ""
            error_msg = str(e)

        row = {
            "question_id": q_id,
            "category": category,
            "question": question,
            "expected_tools": expected_tools,
            "expected_refusal": expected_refusal,
            "model_name": model_name,
            "ablation_setting": setting,
            "load_success": 1.0,
            "vram_after_load_mb": load_memory,
            "peak_vram_reserved_mb": peak_vram,
            "latency_seconds": latency,
            "iterations": iterations,
            "actions": ",".join(parsed["actions"]),
            "raw_actions": ",".join(parsed["raw_actions"]),
            "action_inputs": ";".join(parsed["action_inputs"]),
            "final_answer": (response.get("output") if response is not None else "") or parsed["final_answer"],
            "react_format_success": parsed["react_format_success"],
            "tool_selection_accuracy_auto": tool_acc,
            "numeric_correctness_manual": "",
            "relevance_grounding_manual": "",
            "hallucination_count_manual": "",
            "refusal_correctness_manual": "",
            "chinese_fluency_manual": "",
            "notes": error_msg,
        }

        results.append({
            "row": row,
            "scratchpad": scratchpad,
            "observations": parsed["observations"],
        })

        write_outputs(csv_path, json_path, results)

    print(f"Finished baseline evaluation matrix output saved to: {csv_path}")
    print(f"Finished baseline trace output saved to: {json_path}")
