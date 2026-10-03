"""GPU 模型配置、分类/回归指标及统一特征接受规则。"""
from __future__ import annotations

import json
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from xgboost import DMatrix, XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback
from tabular_data.feature_explanations import number


PROFILES = {
    "weak": {"n_estimators": 40, "max_depth": 1, "learning_rate": 0.08,
             "subsample": 1.0, "colsample_bytree": 1.0},
    "reference": {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.08,
                  "subsample": 1.0, "colsample_bytree": 1.0},
}
HIGHER_IS_BETTER = ("accuracy", "balanced_accuracy", "f1", "f1_macro", "auc", "r2")
LOWER_IS_BETTER = ("log_loss", "mse", "rmse", "mae")
SELECTION_POLICY = {
    "classification_primary": "f1_macro", "classification_min_gain": 0.0001,
    "regression_primary": "mse", "regression_min_relative_gain": 0.001,
    "guard": "all_reported_metrics_non_decreasing", "numeric_tolerance": 1e-10,
    "selection_split": "validation", "test_used_for_selection": False,
}


class _RequireCudaTraining(TrainingCallback):
    def after_training(self, model):
        device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
        if not device.startswith("cuda"):
            raise RuntimeError("XGBoost 未在 GPU 上训练，请检查 CUDA 12 安装包和显卡可见性。")
        return model


def make_model(task_type: str, n_classes: int = 2, profile: str = "weak", seed: int = 42):
    params = dict(PROFILES[profile], random_state=seed, device="cuda:0", tree_method="hist",
                  callbacks=[_RequireCudaTraining()])
    if task_type == "classification":
        return XGBClassifier(**params, eval_metric="mlogloss" if n_classes > 2 else "logloss")
    if task_type == "regression":
        return XGBRegressor(**params, objective="reg:squarederror")
    raise ValueError(f"不支持的任务类型：{task_type}")


def predict_model(model, features, task_type: str):
    """明确使用 DMatrix 预测，避免 CPU pandas 触发 inplace_predict 设备回退警告。"""
    values = model.get_booster().predict(DMatrix(features), strict_shape=True)
    if task_type == "regression":
        return values[:, 0], None
    probabilities = np.column_stack((1 - values[:, 0], values[:, 0])) if values.shape[1] == 1 else values
    return probabilities.argmax(axis=1), probabilities


def compute_metrics(y, predictions, probabilities, task_type: str):
    metrics = {name: None for name in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)}
    if task_type == "regression":
        mse = float(mean_squared_error(y, predictions))
        metrics.update(mse=mse, rmse=math.sqrt(mse), mae=float(mean_absolute_error(y, predictions)),
                       r2=float(r2_score(y, predictions)))
        return metrics
    metrics.update(
        accuracy=float(accuracy_score(y, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
        f1=float(f1_score(y, predictions, average="weighted", zero_division=0)),
        f1_macro=float(f1_score(y, predictions, average="macro", zero_division=0)),
    )
    if probabilities is not None:
        labels = np.arange(probabilities.shape[1])
        metrics["log_loss"] = float(log_loss(y, probabilities, labels=labels))
        if len(np.unique(y)) == len(labels):
            metrics["auc"] = float(roc_auc_score(y, probabilities[:, 1]) if len(labels) == 2 else
                                   roc_auc_score(y, probabilities, multi_class="ovr", labels=labels))
    # 分类标签的任意整数编码没有距离含义，因此不计算分类 MSE。
    return metrics


def metric_improvement(baseline, candidate):
    result = {}
    for key in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        before, after = baseline.get(key), candidate.get(key)
        result[key] = None if before is None or after is None else float(
            after - before if key in HIGHER_IS_BETTER else before - after)
    return result


def accept_candidate(baseline, candidate, task_type: str):
    """固定主指标有实质改善，且所有可用指标不退化；只应传入验证集指标。"""
    primary = "f1_macro" if task_type == "classification" else "mse"
    # 兼容旧报告中只有 weighted F1 的情况。
    if task_type == "classification" and baseline.get(primary) is None:
        primary = "f1"
    if baseline.get(primary) is None or candidate.get(primary) is None:
        return False, f"缺少主指标 {primary}。"
    changes = metric_improvement(baseline, candidate)
    for key, before in baseline.items():
        if before is None or key not in changes:
            continue
        after = candidate.get(key)
        if after is None or not math.isfinite(before) or not math.isfinite(after):
            return False, f"指标 {key} 缺失或不是有限数值。"
        tolerance = SELECTION_POLICY["numeric_tolerance"] * max(1.0, abs(before))
        if changes[key] < -tolerance:
            return False, f"拒绝：{key} 退化（{number(before)} → {number(after)}；改善量 {number(changes[key])}）。"
    threshold = (SELECTION_POLICY["classification_min_gain"] if task_type == "classification" else
                 max(1e-8, abs(baseline[primary]) * SELECTION_POLICY["regression_min_relative_gain"]))
    if changes[primary] <= threshold:
        return False, f"拒绝：主指标 {primary} 改善 {number(changes[primary])}，未超过阈值 {number(threshold)}。"
    return True, f"接受：主指标 {primary} 改善 {number(changes[primary])}，其余可用指标均未退化。"


def applicable_metrics(metrics):
    return {key: value for key, value in metrics.items() if value is not None}
