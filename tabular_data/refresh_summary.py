"""从已完成报告生成当前结果索引及可直接打开的网页结果快照。"""
from __future__ import annotations

import json
from pathlib import Path

from .dataset_registry import DATASETS, SPECS
from .paths import OUTPUT_DIR, TABULAR_DIR
from .research import write_summary

NAMES = {key: SPECS[key]["name"] for key in DATASETS}


def collect_reports(root):
    reports = []
    for path in sorted(Path(root).rglob("training_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("dataset") not in DATASETS or result.get("feature_generation", {}).get("mode") != "reasoned":
            continue
        status_file = path.with_name("run_status.json")
        if not status_file.is_file():
            continue
        state = json.loads(status_file.read_text(encoding="utf-8"))
        if state.get("status") != "completed" or state.get("run_id") != result.get("run_id"):
            continue
        result["output_dir"] = str(path.parent.resolve())
        reports.append(result)
    return sorted(reports, key=lambda r: (r["model"]["profile"] != "weak", DATASETS.index(r["dataset"]),
                                          r["model"]["parameters"]["random_state"], r["output_dir"]))


def refresh_summary(root=OUTPUT_DIR):
    root = Path(root)
    reports = collect_reports(root)
    write_summary(reports, root)
    # Website snapshot always uses one predeclared configuration, never the best test score.
    if root.resolve() == (TABULAR_DIR / "outputs").resolve():
        snapshots = {}
        for result in sorted(reports, key=lambda r: Path(r["output_dir"], "training_results.json").stat().st_mtime):
            generation, model, experiment = (result[k] for k in ("feature_generation", "model", "experiment"))
            if (model["profile"] != "reference" or model["parameters"]["random_state"] != 42
                    or generation["rounds"] != 5 or generation["candidates_per_round"] != 5
                    or experiment["training_budget"]["requested_fraction_of_training_split"] != 1
                    or result["noise"]["strength_in_train_std"] != 0):
                continue
            baseline, optimized = result["test"]["baseline"], result["test"]["optimized"]
            trace = [result["validation"]["baseline"], *[r["metrics_after"] for r in result["iterations"]]]
            snapshots[result["dataset"]] = {
                "name": NAMES[result["dataset"]], "run_id": result["run_id"],
                "baseline": {"f1": baseline["f1_macro"], "log_loss": baseline["log_loss"]},
                "final": {"f1": optimized["f1_macro"], "log_loss": optimized["log_loss"]},
                "improvement": {"f1": round(100*(optimized["f1_macro"]-baseline["f1_macro"]), 2),
                                "log_loss": round(optimized["log_loss"]-baseline["log_loss"], 4)},
                "iter_f1": [r["f1_macro"] for r in trace], "iter_log_loss": [r["log_loss"] for r in trace],
                "rules": [{"desc": p["hypothesis"], "formula": p["display_expression"]}
                          for p in result["accepted_feature_proposals"]],
                "top_features": list(dict.fromkeys(c for p in result["accepted_feature_proposals"] for c in p["input_columns"])),
            }
        content = json.dumps(snapshots, ensure_ascii=False, indent=2).replace("</", "<\\/")
        (root / "benchmark_results.js").write_text(
            "// 自动生成：reference / reasoned / clean / seed42；最终分数为测试，逐轮轨迹为验证。\nwindow.TabularBenchmarks = " + content + ";\n", encoding="utf-8")
    return reports


if __name__ == "__main__":
    refresh_summary()
