#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from answer_extractors import AR_LSAT_AnswerExtractor
from reasoners import AdaptiveAgentReasoner


@dataclass
class SamplePostprocessResult:
    sample_id: str
    gold: str
    original_success: bool
    postprocessed_success: bool
    original_extracted_answer: str
    final_extracted_answer: str
    z3_called: bool
    z3_call_count: int
    final_answer_source: Optional[str]
    tool_parsed_answers: List[str]
    run_name: str


def parse_entry(entry: str) -> tuple[str, List[Path]]:
    if "=" not in entry:
        raise ValueError(f"Invalid --entry value: {entry}")
    model_name, runs_part = entry.split("=", 1)
    run_paths = [Path(part).expanduser() for part in runs_part.split(",") if part.strip()]
    if not run_paths:
        raise ValueError(f"No run paths provided for model: {model_name}")
    return model_name, run_paths


def extract_final_answer(reasoning_output: Optional[str], label_idx: int) -> tuple[bool, str]:
    if not isinstance(reasoning_output, str) or not reasoning_output.strip():
        return False, "[]"
    ok, extracted, _ = AR_LSAT_AnswerExtractor().extract_answer(
        reasoning_output,
        str(label_idx),
        reasoning_method="cot",
    )
    return ok, extracted


def postprocess_sample(summary_path: Path) -> SamplePostprocessResult:
    obj = json.loads(summary_path.read_text())
    problem = obj["problem"]
    aro = obj["all_reasoning_outputs"][0]
    label_idx = int(problem["label"])
    gold = f"[{label_idx}]"

    reasoning_output = aro.get("reasoning_output")
    final_ok, final_extracted = extract_final_answer(reasoning_output, label_idx)

    tool_parsed_answers: List[str] = []
    for tool_log in aro.get("tool_logs", []):
        solver_output = tool_log.get("solver_output")
        parsed = AdaptiveAgentReasoner._extract_answer_from_solver_output(
            solver_output,
            problem.get("answers", []),
        )
        if parsed:
            tool_parsed_answers.append(parsed)

    postprocessed_success = final_ok or (gold in tool_parsed_answers)
    return SamplePostprocessResult(
        sample_id=problem["id_string"],
        gold=gold,
        original_success=bool(aro.get("success")),
        postprocessed_success=postprocessed_success,
        original_extracted_answer=aro.get("answer") or "[]",
        final_extracted_answer=final_extracted,
        z3_called=bool(aro.get("z3_called")),
        z3_call_count=int(aro.get("z3_call_count") or 0),
        final_answer_source=aro.get("final_answer_source"),
        tool_parsed_answers=tool_parsed_answers,
        run_name=summary_path.parent.parent.name,
    )


def aggregate_model(model_name: str, run_paths: List[Path]) -> Dict:
    by_sample: Dict[str, SamplePostprocessResult] = {}
    for run_path in run_paths:
        summary_dir = run_path / "summary"
        for summary_path in sorted(summary_dir.glob("*.json")):
            result = postprocess_sample(summary_path)
            by_sample[result.sample_id] = result

    samples = list(by_sample.values())
    total = len(samples)
    original_correct = sum(1 for s in samples if s.original_success)
    postprocessed_correct = sum(1 for s in samples if s.postprocessed_success)
    z3_called = sum(1 for s in samples if s.z3_called)
    total_z3_calls = sum(s.z3_call_count for s in samples)

    return {
        "model": model_name,
        "sample_count": total,
        "original_accuracy": (original_correct / total) if total else 0.0,
        "postprocessed_accuracy": (postprocessed_correct / total) if total else 0.0,
        "recovered_samples": postprocessed_correct - original_correct,
        "z3_called_frequency": (z3_called / total) if total else 0.0,
        "total_z3_calls": total_z3_calls,
        "runs": [str(p) for p in run_paths],
        "sample_details": samples,
    }


def write_outputs(results: List[Dict], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    serializable = []
    for row in results:
        serializable.append({
            "model": row["model"],
            "sample_count": row["sample_count"],
            "original_accuracy": row["original_accuracy"],
            "postprocessed_accuracy": row["postprocessed_accuracy"],
            "recovered_samples": row["recovered_samples"],
            "z3_called_frequency": row["z3_called_frequency"],
            "total_z3_calls": row["total_z3_calls"],
            "runs": row["runs"],
            "sample_details": [
                {
                    "sample_id": s.sample_id,
                    "gold": s.gold,
                    "original_success": s.original_success,
                    "postprocessed_success": s.postprocessed_success,
                    "final_extracted_answer": s.final_extracted_answer,
                    "tool_parsed_answers": s.tool_parsed_answers,
                    "z3_called": s.z3_called,
                    "z3_call_count": s.z3_call_count,
                    "final_answer_source": s.final_answer_source,
                    "run_name": s.run_name,
                }
                for s in row["sample_details"]
            ],
        })

    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "model",
                "sample_count",
                "original_accuracy",
                "postprocessed_accuracy",
                "recovered_samples",
                "z3_called_frequency",
                "total_z3_calls",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(results, start=1):
            writer.writerow({
                "rank": idx,
                "model": row["model"],
                "sample_count": row["sample_count"],
                "original_accuracy": f"{row['original_accuracy']:.4f}",
                "postprocessed_accuracy": f"{row['postprocessed_accuracy']:.4f}",
                "recovered_samples": row["recovered_samples"],
                "z3_called_frequency": f"{row['z3_called_frequency']:.4f}",
                "total_z3_calls": row["total_z3_calls"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process AR-LSAT adaptive-agent runs and create a leaderboard.")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Model entry in the form model_name=run_dir or model_name=run_dir1,run_dir2",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output path prefix without extension, e.g. results/postprocessed/leaderboard_v1",
    )
    args = parser.parse_args()

    results = []
    for entry in args.entry:
        model_name, run_paths = parse_entry(entry)
        results.append(aggregate_model(model_name, run_paths))

    results.sort(key=lambda row: (-row["postprocessed_accuracy"], -row["sample_count"], row["model"]))
    write_outputs(results, Path(args.output_prefix))

    for idx, row in enumerate(results, start=1):
        print(
            f"{idx}. {row['model']}: "
            f"postprocessed_acc={row['postprocessed_accuracy']:.4f}, "
            f"original_acc={row['original_accuracy']:.4f}, "
            f"samples={row['sample_count']}, "
            f"recovered={row['recovered_samples']}, "
            f"z3_freq={row['z3_called_frequency']:.4f}"
        )


if __name__ == "__main__":
    main()
