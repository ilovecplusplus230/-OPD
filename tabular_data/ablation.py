"""冻结模型后的特征使用检查、置换重要性和删除重训消融。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import applicable_metrics, compute_metrics, make_model, metric_improvement, predict_model


def feature_usage(model, columns):
    booster = model.get_booster()
    counts = booster.get_score(importance_type="weight")
    gains = booster.get_score(importance_type="gain")
    totals = booster.get_score(importance_type="total_gain")
    denominator = sum(totals.values())
    return {col: {"split_count": int(counts.get(col, 0)), "used_by_xgboost": counts.get(col, 0) > 0,
                  "gain": float(gains.get(col, 0)), "total_gain": float(totals.get(col, 0)),
                  "total_gain_fraction": float(totals.get(col, 0) / denominator) if denominator else 0.0}
            for col in columns}


def dependency_group(feature, proposals):
    """删除根节点时同时删除引用它的后代，避免其信息经复合特征留下。"""
    dropped = {feature}
    changed = True
    while changed:
        changed = False
        for item in proposals:
            if item["name"] not in dropped and dropped.intersection(item["input_columns"]):
                dropped.add(item["name"])
                changed = True
    return [item["name"] for item in proposals if item["name"] in dropped]


def _metrics(model, frame, task_type):
    pred, probability = predict_model(model, frame.drop(columns="target"), task_type)
    return applicable_metrics(compute_metrics(frame.target.to_numpy(), pred, probability, task_type))


def _contribution(full, without):
    # 正数始终表示完整特征模型更好；损失指标采用反方向。
    return applicable_metrics(metric_improvement(without, full))


def permutation_scores(model, test, columns, task_type, *, repeats=10, seed=42, reference_metrics=None):
    if repeats < 2:
        raise ValueError("置换次数至少为 2，才能报告置换波动。")
    full = reference_metrics or _metrics(model, test, task_type)
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(repeats):
        shuffled = test.copy()
        # 同一行置换整个依赖组，保留组内关系；原始列保持不变。
        order = rng.permutation(len(test))
        shuffled[columns] = test[columns].to_numpy()[order]
        effects.append(_contribution(full, _metrics(model, shuffled, task_type)))
    means = {key: float(np.mean([row[key] for row in effects])) for key in full}
    deviations = {key: float(np.std([row[key] for row in effects], ddof=1)) for key in full}
    return {"repeats": repeats, "mean_score_drop": means, "std_score_drop": deviations,
            "individual_score_drops": effects,
            "interpretation": "边际置换敏感度；std 仅反映置换波动，不是数据抽样置信区间。"}


def run_ablation(model, baseline_model, train, test, proposals, *, task_type="classification", profile="weak",
                 seed=42, repeats=10, importance_threshold=.001):
    if repeats < 2 or not np.isfinite(importance_threshold) or importance_threshold < 0:
        raise ValueError("消融参数无效。")
    primary = "f1_macro" if task_type == "classification" else "mse"
    features = [item["name"] for item in proposals]
    result = {
        "evaluation_split": "test", "used_for_feature_selection": False, "primary_metric": primary,
        "importance_threshold": importance_threshold, "permutation_repeats": repeats, "seed": seed,
        "importance_definition": "完整模型 Macro-F1 − 置换特征后的 Macro-F1（回归使用 MSE 增量）",
        "decision_rule": "split_count>0 表示被使用；mean_drop>阈值 且 mean_drop−2×std>0 表示稳定置换贡献；再结合删除重训降幅。",
        "caveat": "相关/冗余特征可能掩盖个体贡献；边际置换可能产生不符合原关系的样本；本检查不证明因果重要性或统计显著性。",
        "feature_rows": [], "group_rows": [],
    }
    if not features:
        result["status"] = "no_accepted_features"
        result["message"] = "没有被接受的新特征，无法对新特征进行消融；未伪造重要性。"
        return result
    full = _metrics(model, test, task_type)
    usage = feature_usage(model, features)
    groups = [(name, dependency_group(name, proposals)) for name in features]
    groups.append(("__all_generated__", features))
    cache = {}
    for name, columns in groups:
        key = tuple(sorted(columns))
        if key not in cache:
            permutation = permutation_scores(model, test, columns, task_type, repeats=repeats, seed=seed,
                                             reference_metrics=full)
            reduced_train = train.drop(columns=columns)
            reduced_test = test.drop(columns=columns)
            if set(columns) == set(features):
                reduced_model = baseline_model
                model_source = "baseline_model_same_original_features"
            else:
                n_classes = train.target.nunique() if task_type == "classification" else 0
                reduced_model = make_model(task_type, n_classes, profile, seed)
                reduced_model.fit(reduced_train.drop(columns="target"), reduced_train.target.to_numpy())
                model_source = "gpu_retrained_without_feature_and_descendants"
            without = _metrics(reduced_model, reduced_test, task_type)
            cache[key] = {"permutation": permutation, "drop_retrain_metrics": without,
                          "drop_retrain_score_drop": _contribution(full, without), "retrain_model_source": model_source}
        effect = cache[key]
        mean = effect["permutation"]["mean_score_drop"][primary]
        std = effect["permutation"]["std_score_drop"][primary]
        stable = mean > importance_threshold and mean - 2 * std > 0
        group_used = any(usage[col]["used_by_xgboost"] for col in columns)
        row = {"feature": name, "removed_or_permuted_columns": columns, "full_metrics": full, **effect,
               "importance_score": mean, "importance_std": std, "mean_minus_2std": mean - 2 * std,
               "passes_permutation_threshold": bool(stable), "group_used_by_xgboost": group_used,
               "positive_drop_retrain": effect["drop_retrain_score_drop"][primary] > importance_threshold,
               "supported_predictive_contribution": bool(group_used and stable and effect["drop_retrain_score_drop"][primary] > importance_threshold)}
        if name == "__all_generated__":
            result["group_rows"].append(row)
        else:
            row.update(usage[name])
            # 依赖组与单列置换分开记录：前者衡量整个派生子树，后者回答当前模型是否依赖该列。
            single = (effect["permutation"] if len(columns) == 1 else
                      permutation_scores(model, test, [name], task_type, repeats=repeats, seed=seed, reference_metrics=full))
            row["single_column_permutation"] = single
            row["single_column_importance_score"] = single["mean_score_drop"][primary]
            result["feature_rows"].append(row)
        print(f"  消融 {name}: 置换 Δ{primary}={mean:.4f} ± {std:.4f}；"
              f"删除重训 Δ{primary}={effect['drop_retrain_score_drop'][primary]:.4f}；"
              f"贡献证据={'支持' if row['supported_predictive_contribution'] else '不足'}", flush=True)
    result["status"] = "completed"
    return result


def ablation_table(result):
    rows = []
    primary = result["primary_metric"]
    for item in result["feature_rows"] + result["group_rows"]:
        rows.append({"feature": item["feature"], "split_count": item.get("split_count"),
                     "used_by_xgboost": item.get("used_by_xgboost"), "gain": item.get("gain"),
                     "total_gain_fraction": item.get("total_gain_fraction"), "primary_metric": primary,
                     "importance_score": item["importance_score"], "importance_std": item["importance_std"],
                     "single_column_importance_score": item.get("single_column_importance_score"),
                     "drop_retrain_score_drop": item["drop_retrain_score_drop"][primary],
                     "supported_predictive_contribution": item["supported_predictive_contribution"],
                     "dependency_group": " | ".join(item["removed_or_permuted_columns"])})
    return pd.DataFrame(rows, columns=["feature", "split_count", "used_by_xgboost", "gain", "total_gain_fraction",
                                      "primary_metric", "importance_score", "importance_std", "single_column_importance_score",
                                      "drop_retrain_score_drop", "supported_predictive_contribution", "dependency_group"])
