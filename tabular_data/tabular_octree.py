from __future__ import annotations

"""Tabular OCTree 主线。

这一部分是中期工作的主体：LLM 不是直接改标签，而是生成可执行的新特征代码。
每个候选特征都要先过沙盒检查，再用模型和多指标评估决定是否保留。
整体思路保留 test_advanced / test_final 的闭环，只是整理成 Flask 可以调用的函数。
"""

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from xgboost import build_info, __version__ as XGBOOST_VERSION

from tabular_data.feature_feedback import DEFAULT_CANDIDATES, DEFAULT_ROUNDS, round_feedback
from tabular_data.feature_explanations import display_values
from llm_client import LLMClient
from result_schema import ResultRecord, write_json

from .paths import DATASET_DIR, OUTPUT_DIR
from .evaluation import (
    _RequireCudaTraining, accept_candidate, compute_metrics, make_model,
    metric_improvement, predict_model,
)
from .feature_proposals import propose_feature_candidates, print_proposal
from .ablation import feature_usage

TARGET_CANDIDATES = ["target", "label", "class", "y", "is_phishing", "phishing", "quality"]


def get_data() -> pd.DataFrame:
    """网页演示：从 Jungle 正式训练集分层抽取 600 行，不读取验证/测试。"""
    frame = pd.read_csv(DATASET_DIR / "jungle_chess" / "train.csv")
    if len(frame) > 600:
        frame, _ = train_test_split(frame, train_size=600, stratify=frame.target, random_state=42)
    return frame.reset_index(drop=True)


def get_data_with_llm(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """兼容旧代码函数名，实际预处理交给 preprocess_dataframe。"""
    return preprocess_dataframe(df, target_col=target_col)


def infer_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """自动识别目标列。用户指定优先，其次匹配常见名字，最后退回最后一列。"""
    if target_col and target_col in df.columns:
        return target_col
    lowered = {str(col).lower(): col for col in df.columns}
    for name in TARGET_CANDIDATES:
        if name in lowered:
            return str(lowered[name])
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in ["target", "label", "class", "是否", "类别", "结果"]):
            return str(col)
    return str(df.columns[-1])


def preprocess_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """清洗数据并修复 target leakage。

    关键点：先识别 target_col，再从特征矩阵中 drop 掉原始目标列，
    不能让目标列既当标签又当特征。
    """
    raw = df.copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    detected_target = infer_target_column(raw, target_col)
    y_raw = raw[detected_target].copy()
    feature_df = raw.drop(columns=[detected_target]).copy()

    metadata: Dict[str, Any] = {
        "target_col": detected_target,
        "original_columns": list(raw.columns),
        "feature_columns": list(feature_df.columns),
        "rows": int(len(raw)),
        "target_leakage_fixed": True,
        "encoders": {},
    }

    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            median = feature_df[col].median()
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(median if pd.notna(median) else 0)
        else:
            feature_df[col] = feature_df[col].fillna("missing").astype(str)
            encoder = LabelEncoder()
            feature_df[col] = encoder.fit_transform(feature_df[col])
            metadata["encoders"][col] = list(map(str, encoder.classes_[:20]))

    if len(feature_df.columns) > 0:
        scaler = MinMaxScaler()
        feature_df[feature_df.columns] = scaler.fit_transform(feature_df[feature_df.columns])

    numeric_target = pd.to_numeric(y_raw, errors="coerce")
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > max(20, int(math.sqrt(max(len(y_raw), 1)))):
        y = numeric_target.fillna(numeric_target.median())
        task_type = "regression"
    else:
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y_raw.fillna("missing").astype(str)), index=raw.index)
        task_type = "classification"
        metadata["target_classes"] = list(map(str, target_encoder.classes_))

    numeric = feature_df.copy()
    numeric["target"] = y
    metadata["task_type"] = task_type
    metadata["processed_columns"] = list(numeric.columns)
    return numeric, detected_target, metadata


def dataframe_stats(df: pd.DataFrame, max_cols: int = 20) -> Dict[str, Any]:
    """给 LLM 的简要数据画像，只传必要统计量，避免 prompt 太长。"""
    stats: Dict[str, Any] = {"shape": list(df.shape), "columns": list(df.columns)}
    desc = df.drop(columns=["target"], errors="ignore").describe().T.head(max_cols)
    stats["describe"] = json.loads(desc.to_json(orient="index"))
    return stats


def generate_domain_knowledge(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """生成轻量领域知识，模拟原项目里的知识库提示。"""
    cols = ", ".join(metadata.get("feature_columns", [])[:20])
    return (
        f"数据共有 {metadata.get('rows')} 行，目标列为 {metadata.get('target_col')}。"
        f"可用特征包括：{cols}。OCTree 检验多列复合、非线性和分段关系，输出假说、公式和验证依据。"
    )


def check_feature_robustness(
    df: pd.DataFrame, feature_code: str, allow_constant: bool = False,
) -> Tuple[bool, pd.DataFrame, str, List[str]]:
    """执行 LLM 特征代码，并检查所有新增列是否有效。"""
    before = df.copy()
    before_columns = list(before.columns)
    # 特征代码无法读取标签，且不能覆盖已有特征或改变样本顺序。
    safe_df = before.drop(columns=["target"], errors="ignore").copy()
    namespace = {
        "__builtins__": {},
        "df": safe_df,
        "np": np,
        "pd": pd,
        "math": math,
    }
    try:
        parsed = ast.parse(feature_code)
        forbidden = {"mean", "median", "std", "var", "sum", "min", "max", "quantile", "rank",
                     "groupby", "rolling", "expanding", "shift", "diff", "sample", "sort_values"}
        if any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(parsed)):
            return False, before, "特征必须逐行计算，不能重新拟合统计量或依赖其他行。", []
        exec(feature_code, namespace, namespace)
    except Exception as exc:
        return False, before, f"特征代码执行失败：{type(exc).__name__}: {exc}", []

    original_features = before.drop(columns=["target"], errors="ignore")
    if (namespace["df"] is not safe_df or not safe_df.index.equals(before.index)
            or "target" in safe_df.columns or safe_df.columns.duplicated().any()
            or not set(original_features.columns).issubset(safe_df.columns)
            or not safe_df[list(original_features.columns)].equals(original_features)):
        return False, before, "特征代码不能修改标签、已有特征或样本顺序。", []
    new_columns = [col for col in safe_df.columns if col not in before_columns]
    if not new_columns:
        return False, before, "特征代码没有生成任何新列。", []

    for col in new_columns:
        series = safe_df[col]
        if len(series) != len(before):
            return False, before, f"新特征 {col} 长度不一致。", new_columns
        if series.isna().any():
            return False, before, f"新特征 {col} 存在 NaN。", new_columns
        if not pd.api.types.is_numeric_dtype(series):
            return False, before, f"新特征 {col} 不是数值类型。", new_columns
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            return False, before, f"新特征 {col} 无法稳定转成数值。", new_columns
        if np.isinf(values.to_numpy()).any():
            return False, before, f"新特征 {col} 存在 Inf，可能有除零风险。", new_columns
        if not allow_constant and values.nunique(dropna=False) <= 1:
            return False, before, f"新特征 {col} 是常数列。", new_columns
        for old_col in ([] if allow_constant else before_columns):
            if old_col == "target":
                continue
            old_values = pd.to_numeric(before[old_col], errors="coerce")
            if values.reset_index(drop=True).equals(old_values.reset_index(drop=True)):
                return False, before, f"新特征 {col} 与已有列 {old_col} 完全重复。", new_columns
        safe_df[col] = values
    if "target" in before:
        safe_df["target"] = before["target"]
    return True, safe_df, f"通过沙盒检查，新增列：{', '.join(map(str, new_columns))}", new_columns


def _classification_model(n_classes: int):
    return make_model("classification", n_classes)


def _regression_model():
    return make_model("regression")


def evaluate_and_reason(
    df: pd.DataFrame,
    task_type: Optional[str] = None,
    model_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """五折交叉验证评估当前特征集合，并导出 CART 规则作为自然语言反馈基础。"""
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()
    if task_type is None:
        task_type = "classification" if len(np.unique(y)) <= max(20, int(math.sqrt(max(len(y), 1)))) else "regression"

    is_classification = task_type == "classification"
    backend = model_backend or "xgboost"
    if backend != "xgboost":
        raise ValueError(f"未知评估模型：{backend}")

    if is_classification:
        n_classes = len(np.unique(y))
        min_count = int(pd.Series(y).value_counts().min())
        folds = max(2, min(5, min_count))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        model = _classification_model(n_classes)
    else:
        folds = max(2, min(5, len(df)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        model = _regression_model()

    pred = np.empty(len(y), dtype=float)
    # 保持 XGBoost 概率的 float32 精度，避免 sklearn 按 float64 精度误报归一化警告。
    proba = np.empty((len(y), n_classes), dtype=np.float32) if is_classification else None
    usage = {col: {"split_count": 0, "total_gain": 0.0, "used_by_xgboost": False} for col in X}
    for train_ids, validation_ids in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_ids], y[train_ids])
        for col, item in feature_usage(fitted, list(X.columns)).items():
            usage[col]["split_count"] += item["split_count"]
            usage[col]["total_gain"] += item["total_gain"]
            usage[col]["used_by_xgboost"] |= item["used_by_xgboost"]
        fold_pred, fold_proba = predict_model(fitted, X.iloc[validation_ids], task_type)
        pred[validation_ids] = fold_pred
        if proba is not None:
            proba[validation_ids] = fold_proba
    metrics = compute_metrics(y, pred, proba, task_type)
    tree = (DecisionTreeClassifier if is_classification else DecisionTreeRegressor)(max_depth=3, random_state=42)
    tree.fit(X, y)

    try:
        cart_rules = export_text(tree, feature_names=list(X.columns))
    except Exception:
        cart_rules = "决策树规则导出失败。"
    return {
        "metrics": metrics, "cart_rules": cart_rules, "feature_names": list(X.columns),
        "model_backend": backend, "device": "cuda:0", "feature_usage": usage,
    }


def _metric_improvement(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    return metric_improvement(baseline, optimized)


def evaluate_feature_round(
    train, validation, evaluate, current_metrics, task_type, client, round_index, history, seen,
    *, candidate_count=DEFAULT_CANDIDATES, feature_mode="reasoned", eligible_columns=None, guidance=None,
):
    """OCTree 共用节点：提出可解释候选→检查→XGBoost 验证→保留最佳可接受分支。"""
    proposals = propose_feature_candidates(
        train, client, round_index, history, seen, count=candidate_count, mode=feature_mode,
        eligible_columns=eligible_columns, task_type=task_type, guidance=guidance,
    )
    evaluated, candidates = [], []
    primary = "f1_macro" if task_type == "classification" else "mse"
    for index, proposal in enumerate(proposals, 1):
        seen.add(proposal.signature)
        print_proposal(proposal, round_index, index)
        valid, candidate_train, reason, new_columns = check_feature_robustness(train, proposal.code)
        candidate_validation, report, metrics, usage, passes = None, None, None, {}, False
        if valid and new_columns != [proposal.name]:
            valid, reason = False, "每个候选必须恰好创建其声明的一列。"
        if valid and validation is not None:
            valid, candidate_validation, validation_reason, validation_columns = check_feature_robustness(
                validation, proposal.code, allow_constant=True)
            if not valid or validation_columns != new_columns:
                valid, reason = False, f"验证集特征变换失败：{validation_reason}"
        if valid:
            report = evaluate(candidate_train, candidate_validation)
            metrics = report["metrics"]
            passes, decision = accept_candidate(current_metrics, metrics, task_type)
            if passes:
                decision = decision.replace("接受：", "符合接受条件，待本轮候选比较：", 1)
            reason += "\n" + decision
            if report.get("model") is not None:
                usage = feature_usage(report["model"], new_columns)
            else:
                usage = {col: report.get("feature_usage", {}).get(col, {}) for col in new_columns}
        candidate = {"proposal": proposal.to_dict(), "feature_code": proposal.code, "new_columns": new_columns,
                     "valid": valid, "passes_metric_guard": passes, "accepted": False, "reason": reason,
                     "metrics": metrics, "model_usage": usage}
        candidates.append(candidate)
        evaluated.append((proposal, candidate_train, candidate_validation, report))
        if metrics is not None:
            print(f"    候选评估指标：{display_values(metrics)}", flush=True)
        print(f"    候选检查：{reason.splitlines()[-1]}；XGBoost 使用：{display_values(usage)}", flush=True)
    eligible = [i for i, row in enumerate(candidates) if row["passes_metric_guard"]]
    best = (max(eligible, key=lambda i: candidates[i]["metrics"][primary]) if task_type == "classification" else
            min(eligible, key=lambda i: candidates[i]["metrics"][primary])) if eligible else None
    selected = None
    next_train, next_validation, selected_report, after = train, validation, None, current_metrics.copy()
    if best is not None:
        candidates[best]["accepted"] = True
        selected, next_train, next_validation, selected_report = evaluated[best]
        after = candidates[best]["metrics"].copy()
        for index in eligible:
            if index != best:
                candidates[index]["reason"] += "\n本轮另一个可接受候选的主指标更优，故未保留。"
    representative = candidates[best] if best is not None else (candidates[0] if candidates else {})
    reason = representative.get("reason", "候选库已耗尽，未产生新特征。")
    iteration = {"round": round_index, "model_backend": "xgboost", "evaluation_split": "validation" if validation is not None else "cross_validation_demo",
                 "accepted": best is not None, "valid": representative.get("valid", False),
                 "feature_code": representative.get("feature_code", ""), "new_columns": representative.get("new_columns", []),
                 "explanation": representative.get("proposal", {}), "reason": reason,
                 "metrics_before": current_metrics.copy(), "metrics": representative.get("metrics") or current_metrics.copy(),
                 "metrics_after": after, "candidates": candidates, "feedback": reason}
    iteration["learning_feedback"] = round_feedback(iteration, task_type)
    if guidance is not None:
        from .model_guidance import public_guidance
        iteration["training_model_guidance"] = public_guidance(guidance)
    print(f"  第 {round_index} 轮结论：" + (f"接受 {selected.name}" if selected else "未接受新特征"), flush=True)
    return next_train, next_validation, selected_report, selected, iteration


def run_octree_analysis(
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    max_iterations: int = DEFAULT_ROUNDS,
    llm_client: Optional[LLMClient] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Flask 调用的 Tabular 主函数。"""
    llm_client = llm_client or LLMClient()
    raw_df = df.copy() if df is not None else get_data()
    processed, detected_target, metadata = get_data_with_llm(raw_df, target_col=target_col)
    task_type = metadata.get("task_type", "classification")

    baseline_report = evaluate_and_reason(processed, task_type=task_type)
    model_backend = baseline_report["model_backend"]
    model_info = {
        "backend": model_backend, "device": baseline_report["device"],
        "xgboost_version": XGBOOST_VERSION,
        "cuda_version": ".".join(map(str, build_info().get("CUDA_VERSION", []))),
    }
    baseline_metrics = baseline_report["metrics"]
    current_df = processed.copy()
    best_metrics = baseline_metrics.copy()
    history_log: List[str] = []
    learning_history = []
    iterations: List[Dict[str, Any]] = []
    accepted_features: List[str] = []
    generated_rules: List[str] = []
    seen = set()

    from .model_guidance import build_guidance
    for round_idx in range(1, max_iterations + 1):
        features = current_df.drop(columns="target")
        fitted = make_model(task_type, n_classes=current_df.target.nunique()).fit(features, current_df.target)
        predictions, probabilities = predict_model(fitted, features, task_type)
        guidance = build_guidance(current_df, list(features), predictions=predictions,
                                  probabilities=probabilities, task_type=task_type, round_index=round_idx)
        def evaluate(candidate_train, unused):
            return evaluate_and_reason(candidate_train, task_type=task_type, model_backend=model_backend)
        current_df, _, _, selected, iteration = evaluate_feature_round(
            current_df, None, evaluate, best_metrics, task_type, llm_client, round_idx, learning_history, seen, guidance=guidance)
        best_metrics = iteration["metrics_after"].copy()
        if selected:
            accepted_features.append(selected.name)
        generated_rules.extend(row["feature_code"] for row in iteration["candidates"])
        history_log.append(f"第 {round_idx} 轮：{iteration['reason']}；指标：{display_values(best_metrics)}")
        learning_history.append(iteration["learning_feedback"])
        iterations.append(iteration)

    optimized_report = evaluate_and_reason(current_df, task_type=task_type, model_backend=model_backend)
    optimized_metrics = optimized_report["metrics"]
    improvement = _metric_improvement(baseline_metrics, optimized_metrics)
    accepted = len(accepted_features) > 0
    feedback = "\n\n".join(history_log) or "未生成反馈。"

    record = ResultRecord.create(
        modality="tabular",
        input_summary={
            "rows": int(len(raw_df)),
            "columns": list(map(str, raw_df.columns)),
            "target_col": detected_target,
            "task_type": task_type,
        },
        structured_representation={
            "processed_columns": list(map(str, current_df.columns)),
            "domain_knowledge": generate_domain_knowledge(current_df, metadata),
            "decision_tree_rules": optimized_report["cart_rules"],
        },
        generated={
            "feature_codes": generated_rules,
            "accepted_features": accepted_features,
            "iterations": iterations,
        },
        verification={
            "model": model_info,
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "improvement": improvement,
        },
        feedback=feedback,
        accepted=accepted,
        round=max_iterations,
        metrics={"baseline": baseline_metrics, "optimized": optimized_metrics, "improvement": improvement},
    )

    result = {
        "success": True,
        "model": model_info,
        "message": f"分析完成：共评估 {len(iterations)} 轮，保留 {len(accepted_features)} 个新特征。",
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvement": improvement,
        "iterations": iterations,
        "generated_rules": generated_rules,
        "accepted_features": accepted_features,
        "decision_tree_explanation": optimized_report["cart_rules"],
        "history_log": history_log,
        "record": record.to_dict(),
    }

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        augmented_path = OUTPUT_DIR / "tabular_augmented.csv"
        results_path = OUTPUT_DIR / "tabular_results.json"
        metrics_path = OUTPUT_DIR / "metrics.csv"
        current_df.to_csv(augmented_path, index=False, encoding="utf-8-sig")
        metrics_rows = []
        for stage, values in [("baseline", baseline_metrics), ("optimized", optimized_metrics), ("improvement", improvement)]:
            row = {"stage": stage}
            row.update(values)
            metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        record.saved_path = str(results_path)
        result["record"] = record.to_dict()
        write_json(results_path, result)
    return result


def run_tabular_demo() -> ResultRecord:
    result = run_octree_analysis()
    return ResultRecord(**result["record"])


def load_table_file(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path)
