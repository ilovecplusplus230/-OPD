"""分层划分的显式策略、类别覆盖与分布审计，不根据测试效果调整划分。"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd


POLICY_PATH = Path(__file__).resolve().parent / "datasets" / "split_policy.json"
SPLITS = ("train", "validation", "test")


def load_split_policy():
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if (policy["ratios"] != {"train": .2, "validation": .4, "test": .4}
            or policy["class_ratio_policy"] != "preserve_original"
            or policy["stratify_column"] != "target"
            or policy["protocol"] != "holdout_20_40_40"):
        raise ValueError("当前正式训练要求 target 分层、保留原始类别比例和 20/40/40 划分。")
    if (type(policy["seed"]) is not int or policy["seed"] < 0
            or type(policy["minimum_class_count_per_split"]) is not int
            or policy["minimum_class_count_per_split"] < 1
            or policy["maximum_rounding_error_in_samples"] not in (1, 2)):
        raise ValueError("划分种子、每类最小样本数或取整误差配置无效。")
    return policy


def label_counts(frame):
    if frame.target.isna().any():
        raise ValueError("标签不能缺失。")
    return {str(k): int(v) for k, v in frame.target.astype(str).value_counts().sort_index().items()}


def audit_class_counts(class_counts, *, policy=None, original_counts=None):
    policy = policy or load_split_policy()
    if set(class_counts) != set(SPLITS):
        raise ValueError("必须同时提供 train、validation、test 类别计数。")
    labels = sorted(set().union(*(set(x) for x in class_counts.values())))
    if len(labels) < 2:
        raise ValueError("分类任务至少需要两个类别。")
    if any(type(n) is not int or n < 0 for counts in class_counts.values() for n in counts.values()):
        raise ValueError("类别数量必须是非负整数。")
    merged = {label: sum(class_counts[s].get(label, 0) for s in SPLITS) for label in labels}
    original = original_counts or merged
    if original != merged:
        raise ValueError("划分后的类别总数与原始数据不符，存在遗漏、重复或标签修改。")
    total = sum(original.values())
    records = []
    for split in SPLITS:
        counts = class_counts[split]
        rows = sum(counts.values())
        if abs(rows - total * policy["ratios"][split]) > 2:
            raise ValueError(f"{split} 总样本数不满足 20/40/40 及取整容差。")
        for label in labels:
            count = counts.get(label, 0)
            expected = rows * original[label] / total
            if count < policy["minimum_class_count_per_split"]:
                raise ValueError(f"{split} 类别 {label} 只有 {count} 条，需至少 {policy['minimum_class_count_per_split']} 条。")
            if abs(count - expected) > policy["maximum_rounding_error_in_samples"] + 1e-9:
                raise ValueError(f"{split} 类别 {label} 占比偏离原始分布：实际 {count} 条、期望 {expected:.2f} 条。")
            records.append({"split": split, "class": label, "count": count, "split_rows": rows,
                "proportion": count / rows, "percentage": round(100 * count / rows, 4),
                "original_count": original[label], "original_proportion": original[label] / total,
                "expected_count": expected, "rounding_error_samples": count - expected,
                "difference_percentage_points": 100 * (count / rows - original[label] / total)})
    return {"status": "passed", "policy": policy, "original_class_counts": original,
            "total_rows": total, "records": records,
            "note": "类别占比一致指原始分布在三个集合中保持；有限整数样本不保证比例小数完全相等。"}


def audit_frames(frames, *, policy=None, original_counts=None):
    if set(frames) != set(SPLITS):
        raise ValueError("数据划分不完整。")
    hashes = {}
    columns = list(frames["train"].columns)
    for split, frame in frames.items():
        if list(frame.columns) != columns:
            raise ValueError(f"{split} 特征列与训练集不一致。")
        hashes[split] = set(pd.util.hash_pandas_object(frame, index=False).tolist())
        if len(hashes[split]) != len(frame):
            raise ValueError(f"{split} 存在重复样本。")
    for a, b in itertools.combinations(SPLITS, 2):
        if hashes[a] & hashes[b]:
            raise ValueError(f"{a} 与 {b} 存在相同样本，不能跨集合复用。")
    audit = audit_class_counts({s: label_counts(f) for s, f in frames.items()},
                               policy=policy, original_counts=original_counts)
    audit["cross_split_overlap"] = 0
    audit["within_split_duplicates"] = 0
    return audit


def write_distribution(directory, audit):
    (directory / "class_distribution.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(audit["records"]).to_csv(directory / "class_distribution.csv", index=False, encoding="utf-8-sig", float_format="%.6f")


def print_distribution(audit):
    print("类别分布校验通过：按原始类别比例分层；允许每类至多 "
          f"{audit['policy']['maximum_rounding_error_in_samples']} 条样本的取整误差。", flush=True)
    for split in SPLITS:
        print(f"  {split}: " + "；".join(f"{r['class']}={r['count']:,} ({r['percentage']:.2f}%)"
              for r in audit["records"] if r["split"] == split), flush=True)
