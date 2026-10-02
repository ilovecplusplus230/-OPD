from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


CLOSEOUT_ROOT = Path(__file__).resolve().parents[1]
if str(CLOSEOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLOSEOUT_ROOT))

from experiment.answer_verifier import AnswerVerifier
from experiment.runner import ReasoningGraphParser, run_record, select_mask_node
from opsd_comparison.dataset_bridge import read_jsonl
from opsd_comparison.local_model import LocalQwenBackend, default_model_specs


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_expansion_records(
    records: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep only records with a safe middle node, before any model is compared."""
    parser = ReasoningGraphParser()
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        try:
            graph = parser.parse(str(record["question"]), str(record["solution"]))
            select_mask_node(graph, str(record["id"]), seed)
            eligible.append(record)
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"id": str(record.get("id", "")), "reason": str(exc)})
    return eligible, skipped


def direct_result(backend: LocalQwenBackend, record: dict[str, Any], seed: int, samples: int) -> dict[str, Any]:
    verifier = AnswerVerifier()
    outputs = []
    for index in range(samples):
        text = backend.solve(record["question"], seed + index, sample=index > 0)
        verification = verifier.verify_text(text, str(record["answer"])).to_dict()
        outputs.append({"index": index, "response": text, "verification": verification})
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "model_variant": backend.spec.name,
        "reference_answer": record["answer"],
        "pass_at_1": bool(outputs[0]["verification"].get("correct")),
        "pass_at_k": any(bool(item["verification"].get("correct")) for item in outputs),
        "k": samples,
        "outputs": outputs,
    }


def summarize(output_dir: Path, variants: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        direct_path = output_dir / f"direct_{variant}.jsonl"
        if direct_path.exists():
            records = read_jsonl(direct_path)
            if not records:
                continue
            rows.append(
                {
                    "task": "direct_solve",
                    "model_variant": variant,
                    "sample_count": len(records),
                    "primary_rate": sum(bool(r["pass_at_1"]) for r in records) / len(records),
                    "secondary_rate": sum(bool(r["pass_at_k"]) for r in records) / len(records),
                    "primary_metric": "pass@1",
                    "secondary_metric": f"pass@{records[0]['k']}",
                    "retained_quality_score": "",
                    "retained_quality_gain": "",
                }
            )
        expansion_path = output_dir / f"expansion_{variant}.jsonl"
        if expansion_path.exists():
            records = read_jsonl(expansion_path)
            if not records:
                continue
            accepted = [bool(r["methods"]["feedback"]["accepted"]) for r in records]
            one_shot = [bool(r["methods"]["one_shot"]["accepted"]) for r in records]
            retained_scores = []
            retained_gains = []
            for record in records:
                original = float(record["methods"]["original"]["aggregate_score"])
                feedback = record["methods"]["feedback"]
                retained = float(feedback["aggregate_score"]) if feedback["accepted"] else original
                retained_scores.append(retained)
                retained_gains.append(retained - original)
            rows.append(
                {
                    "task": "reasoning_expansion",
                    "model_variant": variant,
                    "sample_count": len(records),
                    "primary_rate": sum(accepted) / len(accepted),
                    "secondary_rate": sum(one_shot) / len(one_shot),
                    "primary_metric": "feedback_accept_rate",
                    "secondary_metric": "one_shot_accept_rate",
                    "retained_quality_score": sum(retained_scores) / len(retained_scores),
                    "retained_quality_gain": sum(retained_gains) / len(retained_gains),
                }
            )
    write_csv(output_dir / "model_comparison_summary.csv", rows)
    report = [
        "# Base / OPSD 数学模型统一对照",
        "",
        "该报告由统一评测脚本生成。所有模型使用相同题目、顺序、提示词和验证器。",
        "",
        "| 任务 | 模型 | 样本 | 主指标 | 数值 | 次指标 | 数值 | 保留后质量分 | 质量提升 |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        score_text = (
            ""
            if row["retained_quality_score"] == ""
            else f"{float(row['retained_quality_score']):.3f}"
        )
        gain_text = (
            ""
            if row["retained_quality_gain"] == ""
            else f"{float(row['retained_quality_gain']):+.3f}"
        )
        report.append(
            f"| {row['task']} | {row['model_variant']} | {row['sample_count']} | "
            f"{row['primary_metric']} | {float(row['primary_rate']):.3f} | "
            f"{row['secondary_metric']} | {float(row['secondary_rate']):.3f} | "
            f"{score_text} | {gain_text} |"
        )
    (output_dir / "模型对照报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Base Qwen and trained MATHOPD adapters")
    parser.add_argument("--input", type=Path, default=CLOSEOUT_ROOT / "opsd_comparison/data/mathopd_heldout_40.jsonl")
    parser.add_argument("--mathopd-root", type=Path, default=CLOSEOUT_ROOT.parent / "MATHOPD")
    parser.add_argument("--output-dir", type=Path, default=CLOSEOUT_ROOT / "opsd_comparison/outputs")
    parser.add_argument("--variants", nargs="+", default=["base", "opsd_original", "opsd_history"])
    parser.add_argument("--task", choices=["direct", "expansion", "both"], default="both")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    all_records = read_jsonl(args.input)
    records = all_records[: args.limit]
    eligible_expansion_records, skipped_records = prepare_expansion_records(all_records, args.seed)
    expansion_records = eligible_expansion_records[: args.limit]
    specs = default_model_specs(args.mathopd_root)
    unknown = sorted(set(args.variants) - set(specs))
    if unknown:
        raise ValueError(f"未知模型组: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task in {"expansion", "both"}:
        expansion_manifest = {
            "source_count": len(all_records),
            "eligible_count": len(eligible_expansion_records),
            "selected_count": len(expansion_records),
            "selection_rule": "at least three parsed nodes and one safe middle node",
            "skipped": skipped_records,
        }
        (args.output_dir / "expansion_dataset_manifest.json").write_text(
            json.dumps(expansion_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for variant in args.variants:
        backend = LocalQwenBackend(specs[variant], max_new_tokens=args.max_new_tokens)
        backend.load()
        try:
            if args.task in {"direct", "both"}:
                path = args.output_dir / f"direct_{variant}.jsonl"
                if args.fresh:
                    path.unlink(missing_ok=True)
                done = {row["id"] for row in read_jsonl(path)} if path.exists() else set()
                for index, record in enumerate(records, start=1):
                    if record["id"] in done:
                        continue
                    print(f"[{variant} direct {index}/{len(records)}] {record['id']}", flush=True)
                    append_jsonl(path, direct_result(backend, record, args.seed + index * 10, args.pass_k))
            if args.task in {"expansion", "both"}:
                path = args.output_dir / f"expansion_{variant}.jsonl"
                if args.fresh:
                    path.unlink(missing_ok=True)
                done = {row["id"] for row in read_jsonl(path)} if path.exists() else set()
                for index, record in enumerate(expansion_records, start=1):
                    if record["id"] in done:
                        continue
                    print(
                        f"[{variant} expansion {index}/{len(expansion_records)}] {record['id']}",
                        flush=True,
                    )
                    result = run_record(
                        record,
                        backend,
                        seed=args.seed,
                        max_attempts=args.max_attempts,
                        threshold=0.80,
                        min_gain=0.01,
                    )
                    result["model_variant"] = variant
                    append_jsonl(path, result)
        finally:
            usage_path = args.output_dir / f"usage_{args.task}_{variant}.json"
            previous_usage: dict[str, Any] = {}
            if usage_path.exists() and not args.fresh:
                previous_usage = json.loads(usage_path.read_text(encoding="utf-8-sig"))
            usage = {
                "model_variant": variant,
                "calls": int(previous_usage.get("calls", 0)) + backend.calls,
                "prompt_tokens": int(previous_usage.get("prompt_tokens", 0)) + backend.prompt_tokens,
                "completion_tokens": int(previous_usage.get("completion_tokens", 0)) + backend.completion_tokens,
            }
            usage_path.write_text(
                json.dumps(usage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            backend.close()
    summarize(args.output_dir, args.variants)


if __name__ == "__main__":
    main()

