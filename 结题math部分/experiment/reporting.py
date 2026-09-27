from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import write_csv


METRICS = ["similarity", "logic_consistency", "formula_correctness", "completeness", "reasoning_gain"]
METHODS = ["original", "one_shot", "feedback"]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset = record["dataset"]
        baseline_score = float(record["methods"]["original"].get("aggregate_score", 0))
        for method in METHODS:
            item = dict(record["methods"][method])
            retained_score = float(item.get("aggregate_score", 0))
            if method != "original" and not item.get("accepted"):
                retained_score = baseline_score
            item["retained_aggregate_score"] = retained_score
            item["retained_quality_gain"] = retained_score - baseline_score
            groups[(dataset, method)].append(item)
            groups[("ALL", method)].append(item)
    rows: list[dict[str, Any]] = []
    for (dataset, method), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "sample_count": len(items),
            "accepted_count": sum(bool(item.get("accepted")) for item in items),
            "accepted_rate": mean([float(bool(item.get("accepted"))) for item in items]),
            "answer_consistency_rate": mean([float(bool(item.get("answer_verification", {}).get("correct"))) for item in items]),
            "manual_review_rate": mean([float(item.get("answer_verification", {}).get("status") == "manual_required") for item in items]),
            "avg_aggregate_score": mean([float(item.get("aggregate_score", 0)) for item in items]),
            "avg_quality_gain": mean([float(item.get("quality_gain", 0)) for item in items]),
            "avg_retained_aggregate_score": mean([float(item.get("retained_aggregate_score", 0)) for item in items]),
            "avg_retained_quality_gain": mean([float(item.get("retained_quality_gain", 0)) for item in items]),
            "avg_added_nodes": mean([float(item.get("added_node_count", 0)) for item in items]),
            "avg_attempts": mean([float(item.get("attempts", 0)) for item in items]),
        }
        for metric in METRICS:
            row[f"avg_{metric}"] = mean([float(item.get("scores", {}).get(metric, 0)) for item in items])
        rows.append(row)
    return rows


def build_figures(summary: list[dict[str, Any]], records: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    overall = {row["method"]: row for row in summary if row["dataset"] == "ALL"}
    labels = ["Original", "One-shot", "Feedback"]
    methods = METHODS

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    score = [overall[m]["avg_retained_aggregate_score"] for m in methods]
    consistency = [overall[m]["answer_consistency_rate"] for m in methods]
    bars1 = ax.bar(x - 0.18, score, 0.36, label="Retained quality score", color="#4C78A8")
    bars2 = ax.bar(x + 0.18, consistency, 0.36, label="Answer consistency", color="#59A14F")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Math Expansion Results")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.bar_label(bars1, fmt="%.3f", fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "01_overall_metrics.png", dpi=220)
    fig.savefig(output_dir / "01_overall_metrics.svg")
    plt.close(fig)

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for method, label, color in zip(methods, labels, ["#777777", "#F28E2B", "#4C78A8"]):
        values = [overall[method][f"avg_{metric}"] for metric in METRICS]
        values += values[:1]
        ax.plot(angles, values, label=label, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], ["Novelty", "Logic", "Formula", "Completeness", "Reasoning gain"])
    ax.set_ylim(0, 1)
    ax.set_title("Five-dimensional Quality Comparison", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    fig.tight_layout()
    fig.savefig(output_dir / "02_five_metric_radar.png", dpi=220)
    fig.savefig(output_dir / "02_five_metric_radar.svg")
    plt.close(fig)

    datasets = sorted({row["dataset"] for row in summary if row["dataset"] != "ALL"})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    x = np.arange(len(datasets))
    for offset, method, label, color in zip([-width, 0, width], methods, labels, ["#999999", "#F28E2B", "#4C78A8"]):
        values = [next(row["avg_retained_aggregate_score"] for row in summary if row["dataset"] == ds and row["method"] == method) for ds in datasets]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Aggregate score")
    ax.set_title("Results by Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_dataset_comparison.png", dpi=220)
    fig.savefig(output_dir / "03_dataset_comparison.svg")
    plt.close(fig)

    added = [record["methods"]["feedback"].get("added_node_count", 0) for record in records]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, max(added + [1]) + 2)
    ax.hist(added, bins=bins, align="left", rwidth=0.8, color="#4C78A8")
    ax.set_xlabel("Added reasoning nodes")
    ax.set_ylabel("Sample count")
    ax.set_title("Reasoning Node Expansion Distribution")
    ax.set_xticks(list(bins)[:-1])
    fig.tight_layout()
    fig.savefig(output_dir / "04_added_nodes_distribution.png", dpi=220)
    fig.savefig(output_dir / "04_added_nodes_distribution.svg")
    plt.close(fig)


def write_report(root: Path, records: list[dict[str, Any]], summary: list[dict[str, Any]], run_meta: dict[str, Any]) -> None:
    overall = {row["method"]: row for row in summary if row["dataset"] == "ALL"}
    feedback = overall["feedback"]
    one_shot = overall["one_shot"]
    lines = [
        "# 数学推理链扩充实验报告",
        "",
        "## 实验目的",
        "",
        "在原有 Tabular 特征扩充闭环基础上，将生成、验证、反馈和保留机制扩展到数学 CoT。",
        "本实验不生成新题，而是在已有正确解答中遮盖中间节点，让模型根据前后文恢复并细化推理。",
        "",
        "## 数据与方法",
        "",
        f"- 数据量：{run_meta['sample_count']} 条。",
        f"- 模型：`{run_meta['model']}`。",
        f"- 随机种子：{run_meta['seed']}。",
        "- 对照一：原始 CoT。",
        "- 对照二：只恢复一次，不使用评价反馈。",
        "- 完整方法：共享第一次恢复，在失败时最多继续反馈优化两次。",
        "- 严格保留条件：综合分达到阈值、相对原节点有提升、最终答案一致、且新增推理节点大于 0。",
        "",
        "## 总体结果",
        "",
        "| 方法 | 候选综合分 | 门控保留后综合分 | 保留后提升 | 答案一致率 | 接受率 | 平均新增节点 | 平均尝试次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, name in [("original", "原始 CoT"), ("one_shot", "单次恢复"), ("feedback", "反馈优化")]:
        row = overall[method]
        lines.append(
            f"| {name} | {row['avg_aggregate_score']:.3f} | {row['avg_retained_aggregate_score']:.3f} | "
            f"{row['avg_retained_quality_gain']:+.3f} | {row['answer_consistency_rate']:.3f} | {row['accepted_rate']:.3f} | "
            f"{row['avg_added_nodes']:.3f} | {row['avg_attempts']:.3f} |"
        )
    lines.extend(["", "## 分数据集结果", "", "| 数据集 | 样本 | 原始分 | 单次恢复 | 反馈优化 | 反馈接受率 | 答案一致率 |", "|---|---:|---:|---:|---:|---:|---:|"])
    datasets = sorted({row["dataset"] for row in summary if row["dataset"] != "ALL"})
    for dataset in datasets:
        by_method = {row["method"]: row for row in summary if row["dataset"] == dataset}
        lines.append(
            f"| {dataset} | {by_method['feedback']['sample_count']} | {by_method['original']['avg_retained_aggregate_score']:.3f} | "
            f"{by_method['one_shot']['avg_retained_aggregate_score']:.3f} | {by_method['feedback']['avg_retained_aggregate_score']:.3f} | "
            f"{by_method['feedback']['accepted_rate']:.3f} | {by_method['feedback']['answer_consistency_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"反馈优化相对单次恢复的门控保留后综合分变化为 "
            f"`{feedback['avg_retained_aggregate_score'] - one_shot['avg_retained_aggregate_score']:+.3f}`，",
            f"严格接收率为 `{feedback['accepted_rate']:.1%}`，最终答案一致率为 `{feedback['answer_consistency_rate']:.1%}`。",
            "这些结果用于说明工程闭环是否能稳定筛选出有价值的中间推理补充，不把启发式质量分解释为形式化证明正确率。",
            "无法自动解析的复杂答案单独计入人工复核比例，不会默认判为正确。",
            "",
            "## 输出说明",
            "",
            "- `outputs/experiment_records.jsonl`：逐题完整轨迹。",
            "- `outputs/per_sample_metrics.csv`：逐题指标。",
            "- `outputs/summary.csv`：总体和分数据集汇总。",
            "- `outputs/figures/`：四张 PNG 图，同时保留 SVG 便于无损编辑。",
            "- `report/典型案例.md`：成功与失败案例。",
        ]
    )
    (root / "report" / "数学扩充实验报告.md").write_text("\n".join(lines), encoding="utf-8")

    successes = [r for r in records if r["methods"]["feedback"]["accepted"]][:3]
    failures = [r for r in records if not r["methods"]["feedback"]["accepted"]][:2]
    case_lines = ["# 数学推理扩充典型案例", ""]
    for title, cases in [("成功案例", successes), ("失败案例", failures)]:
        case_lines.extend([f"## {title}", ""])
        for index, record in enumerate(cases, start=1):
            method = record["methods"]["feedback"]
            case_lines.extend(
                [
                    f"### {title}{index} {record['dataset']} {record['id']}",
                    "",
                    f"**题目：** {record['question']}",
                    "",
                    f"**被遮盖节点：** {record['masked_original']}",
                    "",
                    f"**第一次恢复：** {' '.join(record['methods']['one_shot']['steps'])}",
                    "",
                    f"**最终恢复：** {' '.join(method['steps'])}",
                    "",
                    f"**结果：** 综合分 {method['aggregate_score']:.3f}，质量提升 {method['quality_gain']:+.3f}，接受={method['accepted']}。",
                    "",
                ]
            )
    (root / "report" / "典型案例.md").write_text("\n".join(case_lines), encoding="utf-8")


def save_all(root: Path, records: list[dict[str, Any]], run_meta: dict[str, Any]) -> list[dict[str, Any]]:
    summary = summarize(records)
    write_csv(root / "outputs" / "summary.csv", summary)
    per_sample: list[dict[str, Any]] = []
    for record in records:
        for method in METHODS:
            item = record["methods"][method]
            per_sample.append(
                {
                    "id": record["id"],
                    "dataset": record["dataset"],
                    "subject": record.get("subject", ""),
                    "difficulty": record.get("difficulty", ""),
                    "method": method,
                    "accepted": item.get("accepted"),
                    "aggregate_score": item.get("aggregate_score"),
                    "quality_gain": item.get("quality_gain"),
                    "retained_aggregate_score": (
                        item.get("aggregate_score")
                        if method == "original" or item.get("accepted")
                        else record["methods"]["original"].get("aggregate_score")
                    ),
                    "answer_correct": item.get("answer_verification", {}).get("correct"),
                    "answer_status": item.get("answer_verification", {}).get("status"),
                    "added_node_count": item.get("added_node_count"),
                    "attempts": item.get("attempts"),
                    **{metric: item.get("scores", {}).get(metric) for metric in METRICS},
                }
            )
    write_csv(root / "outputs" / "per_sample_metrics.csv", per_sample)
    (root / "outputs" / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    build_figures(summary, records, root / "outputs" / "figures")
    write_report(root, records, summary, run_meta)
    return summary

