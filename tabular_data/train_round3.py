"""第三轮四数据集：python tabular_data/train_round3.py。"""
import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabular_data.dataset_registry import ROUND3_DATASETS
from tabular_data.train_relations import run_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not ROUND3_DATASETS:
        parser.error("第三轮数据已按用户要求移除；请使用 python tabular_data/train.py 运行当前三个数据集。")
    return run_cohort(ROUND3_DATASETS, "round3_experiment_plan.json", "mfeat_morphological")


if __name__ == "__main__":
    raise SystemExit(main())
