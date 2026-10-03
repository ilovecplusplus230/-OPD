"""表格训练入口：python tabular_data/train.py。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]

# 同时支持直接运行文件和 python -m tabular_data.train。
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_DIR))

from tabular_data.dataset_registry import DATASETS
from tabular_data.feature_feedback import DEFAULT_CANDIDATES, DEFAULT_ROUNDS
from tabular_data.feature_explanations import display_values


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("迭代次数不能小于 0")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 GPU XGBoost 训练表格数据并扩充特征。")
    parser.add_argument(
        "--dataset", choices=("all", *DATASETS), default="all",
        help="选择数据集，默认依次训练全部三个数据集",
    )
    parser.add_argument(
        "--iterations", type=nonnegative_int, default=DEFAULT_ROUNDS,
        help="特征扩充轮数，默认 5；设为 0 时只评估原始特征",
    )
    parser.add_argument("--profile", choices=("weak", "reference"), default="weak", help="默认使用弱交互模型")
    parser.add_argument("--noise", type=float, default=0.0, help="独立噪声实验的强度（训练集标准差倍数），默认 0")
    parser.add_argument("--seed", type=nonnegative_int, default=42, help="模型和噪声随机种子，默认 42；数据划分固定为 42")
    parser.add_argument("--offline", action="store_true", help="只用本地规则生成特征，不请求 LLM API")
    parser.add_argument("--feature-mode", choices=("reasoned",), default="reasoned", help="只使用 CART/模型误差指导的当前生成器")
    parser.add_argument("--dataset-view", choices=("full",), default="full", help="固定使用完整原始特征")
    parser.add_argument("--train-fraction", type=float, default=1., help="在原 20%% 训练集合内分层抽取的预算，默认全量；不改变验证/测试")
    parser.add_argument("--candidates-per-round", type=int, default=DEFAULT_CANDIDATES, help="每轮候选数，默认 5，最多接受 1 个")
    parser.add_argument("--permutation-repeats", type=int, default=10, help="消融置换次数，默认 10")
    parser.add_argument("--importance-threshold", type=float, default=.001, help="重要性阈值：Macro-F1 绝对降幅，默认 0.001")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import math
    if not math.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise 必须为有限的非负数")
    if not math.isfinite(args.train_fraction) or not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction 必须在 (0,1] 内")
    if args.candidates_per_round < 1 or args.permutation_repeats < 2:
        parser.error("候选数至少为 1，置换次数至少为 2")
    if not math.isfinite(args.importance_threshold) or args.importance_threshold < 0:
        parser.error("重要性阈值必须为有限非负数")

    if args._worker and args.dataset == "all":
        parser.error("内部 worker 每次只训练一个数据集。")
    if not args._worker:
        from tabular_data.paths import OUTPUT_DIR
        from tabular_data.run_logging import run_logged_process, training_output_dir
        # 每个数据集单独记录完整 stdout/stderr，同时在终端同步显示。
        for name in DATASETS if args.dataset == "all" else (args.dataset,):
            directory = training_output_dir(OUTPUT_DIR, name, profile=args.profile, noise_std=args.noise,
                feature_mode=args.feature_mode, iterations=args.iterations,
                candidates_per_round=args.candidates_per_round, seed=args.seed,
                dataset_view=args.dataset_view, train_fraction=args.train_fraction)
            exit_code = run_logged_process([
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--_worker", "--dataset", name, "--iterations", str(args.iterations),
                "--profile", args.profile, "--noise", str(args.noise), "--seed", str(args.seed),
                "--feature-mode", args.feature_mode, "--candidates-per-round", str(args.candidates_per_round),
                "--dataset-view", args.dataset_view, "--train-fraction", str(args.train_fraction),
                "--permutation-repeats", str(args.permutation_repeats), "--importance-threshold", str(args.importance_threshold),
                *(["--offline"] if args.offline else []),
            ], directory, metadata={k: v for k, v in dict(vars(args), dataset=name).items() if k != "_worker"})
            if exit_code != 0:
                print(f"{name} 训练失败，已停止后续训练。", file=sys.stderr, flush=True)
                return exit_code
        from tabular_data.refresh_summary import refresh_summary
        refresh_summary(OUTPUT_DIR)
        print(f"当前结果索引：{OUTPUT_DIR / 'latest_summary.md'}", flush=True)
        return 0

    from tabular_data.training import run_training

    result = run_training(
        args.dataset, iterations=args.iterations, profile=args.profile,
        noise_std=args.noise, seed=args.seed, offline=args.offline, feature_mode=args.feature_mode,
        candidates_per_round=args.candidates_per_round, permutation_repeats=args.permutation_repeats,
        importance_threshold=args.importance_threshold,
        dataset_view=args.dataset_view, train_fraction=args.train_fraction,
    )
    print(json.dumps(display_values({
        "dataset": args.dataset,
        "model": result["model"],
        "validation": result["validation"],
        "test": result["test"],
        "accepted_features": result["accepted_features"],
        "feature_generation": result["feature_generation"],
        "ablation_status": result["ablation"]["status"],
    }), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
