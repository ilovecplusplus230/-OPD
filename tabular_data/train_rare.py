"""运行本轮三个稀少类别数据集：python tabular_data/train_rare.py。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabular_data.dataset_registry import RARE_CLASS_DATASETS, SPECS
from tabular_data.evaluation import PROFILES
from tabular_data.feature_proposals import GENERATOR_VERSION
from tabular_data.paths import DATASET_DIR, OUTPUT_DIR, TABULAR_DIR
from tabular_data.run_logging import atomic_json, training_output_dir, utc_now


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    configuration = dict(profile="reference", feature_mode="reasoned", iterations=5,
                         candidates_per_round=5, seed=42, noise_std=0., train_fraction=1., dataset_view="full")
    plan = {"selected_at_utc": utc_now(), "selection_before_training": True,
            "test_scores_used_for_selection": False,
            "criteria": {"classes": 3, "preferred_raw_minority_fraction_max": .05,
                         "permitted_raw_minority_fraction_max": .1, "preserve_original_labels": True},
            "configuration": {**configuration, "parameters": PROFILES["reference"],
                              "generator_version": GENERATOR_VERSION, "offline": True},
            "generator_source_sha256": {name: hashlib.sha256((TABULAR_DIR/name).read_bytes()).hexdigest()
                                        for name in ("feature_proposals.py", "model_guidance.py", "evaluation.py")},
            "datasets": {}, "status": "prepared", "completed": []}
    for key in RARE_CLASS_DATASETS:
        manifest = json.loads((DATASET_DIR/key/"manifest.json").read_text(encoding="utf-8"))
        counts = manifest["preparation"]["raw_class_counts"]
        if len(counts) != 3 or min(counts.values()) / sum(counts.values()) > .1:
            raise ValueError(f"{key} 不符合预设的三分类及最少类占比条件。")
        plan["datasets"][key] = {
            "openml_id": SPECS[key]["id"], "source_rows": manifest["source_rows"], "raw_class_counts": counts,
            "removed_duplicates": manifest["preparation"]["rows_removed_as_duplicates"],
            "class_counts": manifest["class_counts"], "csv_sha256": {s: v["sha256"] for s, v in manifest["files"].items()},
            "literature_scope": manifest["literature_scope"], "benchmark_reference": manifest["benchmark_reference"],
            "minority_tier": manifest["minority_tier"]}
    path = OUTPUT_DIR/"rare_class_experiment_plan.json"
    atomic_json(path, plan)
    for key in RARE_CLASS_DATASETS:
        plan.update(status="running", current_dataset=key)
        atomic_json(path, plan)
        code = subprocess.call([sys.executable, "-u", str(TABULAR_DIR/"train.py"), "--dataset", key,
                                "--profile", "reference", "--offline", "--iterations", "5",
                                "--candidates-per-round", "5", "--seed", "42", "--noise", "0",
                                "--train-fraction", "1", "--feature-mode", "reasoned"])
        if code:
            plan.update(status="failed", failed_dataset=key, exit_code=code)
            atomic_json(path, plan)
            return code
        directory = training_output_dir(OUTPUT_DIR, key, **configuration)
        result = json.loads((directory/"training_results.json").read_text(encoding="utf-8"))
        plan["completed"].append({"dataset": key, "run_id": result["run_id"], "output_dir": str(directory)})
        atomic_json(path, plan)
    plan.update(status="completed", completed_at_utc=utc_now())
    plan.pop("current_dataset", None)
    atomic_json(path, plan)
    print(f"三分类稀少类别实验完成：{OUTPUT_DIR/'latest_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
