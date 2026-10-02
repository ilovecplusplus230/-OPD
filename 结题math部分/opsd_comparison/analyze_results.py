from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from opsd_comparison.dataset_bridge import read_jsonl


LABELS = {
    "base": "Base Qwen3-1.7B",
    "opsd_original": "Original OPSD",
    "opsd_history": "History OPSD",
}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    baseline: list[bool], candidate: list[bool], samples: int = 20000, seed: int = 20261002
) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas = []
    size = len(baseline)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        delta = sum(int(candidate[i]) - int(baseline[i]) for i in indices) / size
        deltas.append(delta)
    deltas.sort()
    return deltas[int(0.025 * samples)], deltas[int(0.975 * samples)]


def load_variant(output_dir: Path, variant: str) -> dict[str, bool]:
    rows = read_jsonl(output_dir / f"direct_{variant}.jsonl")
    results = {str(row["id"]): bool(row["pass_at_1"]) for row in rows}
    if len(rows) != len(results):
        raise RuntimeError(f"{variant} 存在重复题目 ID。")
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_chart(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [LABELS[row["variant"]] for row in summaries]
    values = [row["accuracy"] for row in summaries]
    lower = [row["accuracy"] - row["ci_low"] for row in summaries]
    upper = [row["ci_high"] - row["accuracy"] for row in summaries]
    colors = ["#708090", "#2878B5", "#2E8B57"]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    bars = axis.bar(labels, values, color=colors, width=0.62, yerr=[lower, upper], capsize=5)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Pass@1 accuracy")
    axis.set_title("Held-out MATH comparison (n=40)")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the unified Base/OPSD comparison")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = ["base", "opsd_original", "opsd_history"]
    results = {variant: load_variant(args.output_dir, variant) for variant in variants}
    shared_ids = sorted(results["base"])
    for variant in variants[1:]:
        if sorted(results[variant]) != shared_ids:
            raise RuntimeError(f"{variant} 与 Base 的题目集合不一致。")
    if len(shared_ids) != 40:
        raise RuntimeError(f"统一模型对照必须包含40题，实际为 {len(shared_ids)} 题。")

    summaries = []
    for variant in variants:
        correct = sum(results[variant].values())
        low, high = wilson_interval(correct, len(shared_ids))
        summaries.append(
            {
                "variant": variant,
                "correct": correct,
                "total": len(shared_ids),
                "accuracy": correct / len(shared_ids),
                "ci_low": low,
                "ci_high": high,
            }
        )

    paired_rows = []
    case_rows = []
    base_values = [results["base"][sample_id] for sample_id in shared_ids]
    for variant in variants[1:]:
        candidate_values = [results[variant][sample_id] for sample_id in shared_ids]
        improved = sum(not before and after for before, after in zip(base_values, candidate_values))
        regressed = sum(before and not after for before, after in zip(base_values, candidate_values))
        ci_low, ci_high = paired_bootstrap_ci(base_values, candidate_values)
        paired_rows.append(
            {
                "baseline": "base",
                "candidate": variant,
                "accuracy_gain": sum(candidate_values) / 40 - sum(base_values) / 40,
                "improved_cases": improved,
                "regressed_cases": regressed,
                "unchanged_cases": 40 - improved - regressed,
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
            }
        )
        for sample_id in shared_ids:
            before = results["base"][sample_id]
            after = results[variant][sample_id]
            if before != after:
                case_rows.append(
                    {
                        "id": sample_id,
                        "candidate": variant,
                        "base_correct": before,
                        "candidate_correct": after,
                        "change": "improved" if after else "regressed",
                    }
                )

    write_csv(args.output_dir / "model_accuracy_with_ci.csv", summaries)
    write_csv(args.output_dir / "paired_model_comparison.csv", paired_rows)
    write_csv(args.output_dir / "paired_changed_cases.csv", case_rows)
    draw_chart(args.output_dir / "opsd_model_comparison.png", summaries)

    payload = {"models": summaries, "paired_vs_base": paired_rows}
    (args.output_dir / "model_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OPSD 训练前后统一模型对照",
        "",
        "三组模型在同一批40道 MATH 测试题上评测，题目已与完整 OPSD 训练集做规范化精确去重。",
        "提示词、解码方式、最大生成长度和答案验证器保持一致。",
        "",
        "| 模型 | 正确数 | Pass@1 | 相对 Base |",
        "|---|---:|---:|---:|",
    ]
    base_accuracy = summaries[0]["accuracy"]
    for row in summaries:
        lines.append(
            f"| {LABELS[row['variant']]} | {row['correct']}/{row['total']} | "
            f"{row['accuracy']:.1%} | {row['accuracy'] - base_accuracy:+.1%} |"
        )
    lines.extend(["", "## 配对比较", ""])
    for row in paired_rows:
        lines.append(
            f"- {LABELS[row['candidate']]} 相对 Base：提升 {row['accuracy_gain']:+.1%}，"
            f"纠正 {row['improved_cases']} 题、退化 {row['regressed_cases']} 题；"
            f"配对 bootstrap 95% CI [{row['paired_bootstrap_ci_low']:+.1%}, "
            f"{row['paired_bootstrap_ci_high']:+.1%}]，精确 McNemar p={row['exact_mcnemar_p']:.3f}。"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "结果显示 OPSD 训练后正确率呈上升趋势，History OPSD 在本次40题中最好。",
            "由于样本量较小且置信区间仍覆盖0，当前应表述为方向性证据，不能写成已经达到统计显著。",
        ]
    )
    (args.output_dir / "OPSD训练前后对照结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
