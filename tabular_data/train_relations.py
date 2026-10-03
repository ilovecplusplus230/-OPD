"""组合关系四数据集实验：python tabular_data/train_relations.py。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabular_data.dataset_registry import RELATIONAL_DATASETS, SPECS
from tabular_data.evaluation import PROFILES
from tabular_data.feature_proposals import GENERATOR_VERSION
from tabular_data.paths import DATASET_DIR, OUTPUT_DIR, TABULAR_DIR
from tabular_data.run_logging import atomic_json, training_output_dir, utc_now


CONFIGURATION = dict(profile="reference", feature_mode="reasoned", iterations=5,
                     candidates_per_round=5, seed=42, noise_std=0., train_fraction=1., dataset_view="full")


def make_plan(datasets=RELATIONAL_DATASETS, balanced_control=None):
    plan = {"selected_at_utc": utc_now(), "selection_before_training": True,
            "test_scores_used_for_selection": False,
            "criteria": {"priority": "multi_column_relationships", "allow_more_than_three_classes": True,
                         "preserve_original_labels": True, "prefer_rare_classes": True,
                         "balanced_control": balanced_control, "sparse_group_max_training_fraction": .05},
            "configuration": {**CONFIGURATION, "parameters": PROFILES["reference"],
                              "generator_version": GENERATOR_VERSION, "offline": True},
            "source_sha256": {name: hashlib.sha256((TABULAR_DIR/name).read_bytes()).hexdigest()
                              for name in ("feature_proposals.py", "model_guidance.py", "evaluation.py",
                                           "training.py", "prepare_datasets.py", "dataset_registry.py")},
            "datasets": {}, "status": "prepared", "completed": []}
    for key in datasets:
        m = json.loads((DATASET_DIR/key/"manifest.json").read_text(encoding="utf-8"))
        spec = SPECS[key]
        counts = m["preparation"]["raw_class_counts"]
        if len(counts) != spec["classes"] or sum(counts.values()) != spec["rows"]:
            raise ValueError(f"{key} 数据版本与预先声明不符。")
        plan["datasets"][key] = {
            "provider": m["provider"], "openml_id": m["openml_id"], "uci_id": m.get("uci_id"),
            "source_rows": m["source_rows"], "raw_class_counts": counts,
            "removed_duplicates": m["preparation"]["rows_removed_as_duplicates"],
            "class_counts": m["class_counts"], "csv_sha256": {s: v["sha256"] for s, v in m["files"].items()},
            "literature_scope": m["literature_scope"], "benchmark_reference": m["benchmark_reference"],
            "relationship_basis": spec["relationship_basis"], "minority_tier": m["minority_tier"]}
    return plan


def run_cohort(datasets, plan_name, balanced_control):
    if not datasets:
        raise ValueError("此历史批次的数据已移除，请使用 tabular_data/train.py 运行当前三个数据集。")
    plan = make_plan(datasets, balanced_control)
    path = OUTPUT_DIR/plan_name
    atomic_json(path, plan)
    for key in datasets:
        plan.update(status="running", current_dataset=key)
        atomic_json(path, plan)
        command = [sys.executable, "-u", str(TABULAR_DIR/"train.py"), "--dataset", key, "--offline"]
        for name, value in CONFIGURATION.items():
            option = "noise" if name == "noise_std" else name.replace("_", "-")
            command += ["--" + option, str(value)]
        code = subprocess.call(command)
        if code:
            plan.update(status="failed", failed_dataset=key, exit_code=code)
            atomic_json(path, plan)
            return code
        directory = training_output_dir(OUTPUT_DIR, key, **CONFIGURATION)
        result = json.loads((directory/"training_results.json").read_text(encoding="utf-8"))
        plan["completed"].append({"dataset": key, "run_id": result["run_id"], "output_dir": str(directory)})
        atomic_json(path, plan)
    plan.update(status="completed", completed_at_utc=utc_now())
    plan.pop("current_dataset", None)
    atomic_json(path, plan)
    print(f"{len(datasets)} 个任务已完成：{OUTPUT_DIR/'latest_summary.md'}", flush=True)
    return 0


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    return run_cohort(RELATIONAL_DATASETS, "relational_experiment_plan.json", None)


if __name__ == "__main__":
    raise SystemExit(main())
