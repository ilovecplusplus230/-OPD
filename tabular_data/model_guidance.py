"""训练集 CART 解释与模型条件候选预筛；不是独立验证或 OpenFE 的复现。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


def build_guidance(frame, columns, *, predictions, probabilities=None,
                   task_type="classification", seed=42, round_index=1):
    x = frame[columns].astype(float)
    y = frame.target.to_numpy()
    min_leaf = max(5, int(np.ceil(len(frame) * .01)))
    cls = DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor
    cart = cls(max_depth=4, min_samples_leaf=min_leaf, random_state=seed).fit(x, y)
    if task_type == "classification":
        probabilities = np.asarray(probabilities)
        residuals = np.eye(probabilities.shape[1])[y.astype(int)] - probabilities
        diagnostics = [{"class": int(c), "rows": int((y == c).sum()),
                        "recall": float(np.mean(np.asarray(predictions)[y == c] == c))}
                       for c in np.unique(y)]
    else:
        residuals = (y - predictions)[:, None]
        diagnostics = []
    tree, parameters, paths = cart.tree_, {}, []

    def visit(node, conditions, mask):
        if tree.children_left[node] == tree.children_right[node]:
            if len(conditions) >= 2:
                paths.append({"leaf": int(node), "conditions": conditions, "rows": int(mask.sum()),
                              "mean_squared_residual": float(np.mean(residuals[mask] ** 2))})
            return
        col = columns[tree.feature[node]]
        threshold = float(tree.threshold[node])
        symbol = f"cart_r{round_index}_node{node}"
        values = x[col].to_numpy()
        left, right = mask & (values <= threshold), mask & (values > threshold)
        # sklearn CART 使用 float32 输入寻找分裂；保存实际实现边界和两侧观测。
        low, high = float(values[left].max()), float(values[right].min())
        parameters[symbol] = {"value": threshold, "kind": "training_tree_threshold",
            "source": "train", "column": col, "statistic": "CART split",
            "calculation": (f"CART 深度上限 4、最小叶样本 {min_leaf}，在本节点 {int(mask.sum())} 行训练样本上"
                f"按加权子节点不纯度选择分裂；边界两侧观测为 {low:.4g} 和 {high:.4g}。"
                "阈值取 CART 实际保存的中点（训练内部 float32）；完整精度保存在 value。"),
            "purpose": "冻结训练得到的分段条件，不在验证或测试集重估。",
            "node": int(node), "boundary_values": [low, high], "node_rows": int(mask.sum()),
            "criterion": cart.criterion, "impurity_before": float(tree.impurity[node]),
            "weighted_impurity_after": float((left.sum()*tree.impurity[tree.children_left[node]] +
                                               right.sum()*tree.impurity[tree.children_right[node]]) / mask.sum())}
        visit(tree.children_left[node], conditions + [(col, "<=", symbol)], left)
        visit(tree.children_right[node], conditions + [(col, ">", symbol)], right)

    visit(0, [], np.ones(len(frame), dtype=bool))
    return {"fit_split": "train", "method": "CART paths + current XGBoost training residuals",
            "cart_rules": export_text(cart, feature_names=list(columns), decimals=4),
            "cart_parameters": {"max_depth": 4, "min_samples_leaf": min_leaf, "seed": seed},
            "thresholds": parameters, "paths": paths, "class_diagnostics": diagnostics,
            "ranking_method": "8-bin training residual explained variance; diversity by family and correlation",
            "limitation": "训练内预筛启发式，残差为当前模型的训练预测残差；不是 OOF、验证增益或显著性证据。",
            "_residuals": residuals}


def public_guidance(context):
    return {k: v for k, v in context.items() if not k.startswith("_")}


def tree_specs(context):
    for path in context["paths"]:
        conditions = " & ".join(f"(df[{col!r}] {op} {symbol})" for col, op, symbol in path["conditions"])
        yield (f"oct_cart_leaf_{path['leaf']}", "piecewise", f"np.where({conditions}, 1, 0)",
               "训练决策树中多个条件共同出现的子群，可能需要一个联合指示列才能被浅层 XGBoost 表达。",
               f"将 CART 根节点到叶 {path['leaf']} 的条件逐项求交；满足所有条件输出 1，否则 0。"
               f"训练支持数为 {path['rows']}，阈值逐项保留训练来源。", path)


def screen_candidates(pool, frame, context):
    """只访问训练样本。分箱数固定 8，不用验证/测试分数调整预筛。"""
    residuals = context["_residuals"]
    residuals = residuals - residuals.mean(axis=0)
    total = float(np.square(residuals).sum())
    original = frame.drop(columns="target").rank().to_numpy(dtype=float, copy=True)
    original -= original.mean(axis=0)
    original_norm = np.linalg.norm(original, axis=0)
    for proposal in pool:
        with np.errstate(all="ignore"):
            values = np.asarray(eval(proposal.expression, {"__builtins__": {}, "np": np, "df": frame}), dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all() or np.unique(values).size < 2:
            proposal.evidence["prescreen"] = {"valid": False, "score": -1., "reason": "训练变换非有限或恒定"}
            continue
        ranked = pd.Series(values).rank().to_numpy(copy=True)
        ranked -= ranked.mean()
        correlation = np.abs(ranked @ original) / np.maximum(np.linalg.norm(ranked)*original_norm, 1e-12)
        redundancy = float(np.max(correlation)) if correlation.size else 0.
        bins = pd.qcut(values, q=min(8, np.unique(values).size), labels=False, duplicates="drop")
        # 二值指示列的分位点可能塌缩，使用离散取值分组。
        if np.unique(values).size <= 8:
            _, bins = np.unique(values, return_inverse=True)
        between = 0.
        for group in np.unique(bins):
            selected = residuals[bins == group]
            between += len(selected) * float(np.square(selected.mean(axis=0)).sum())
        score = between / max(total, 1e-12)
        proposal.evidence["prescreen"] = {"valid": True, "score": score, "fit_split": "train",
            "between_bin_residual_sum_squares": between, "total_residual_sum_squares": total,
            "bins": int(np.unique(bins).size), "max_abs_spearman_existing": redundancy,
            "formula": "score = Σ_b n_b ||mean(r_b) − mean(r)||² / Σ_i ||r_i − mean(r)||²; r=onehot(y)−p_XGBoost",
            "interpretation": "训练内误差分组能力；不等于验证增益，不用于直接接受特征。"}
        proposal.evidence["_screen_values"] = values
    return sorted(pool, key=lambda p: p.evidence["prescreen"]["score"], reverse=True)
