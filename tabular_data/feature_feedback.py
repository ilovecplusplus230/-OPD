"""本次训练内的验证反馈记忆；不读取测试集、不更新 LLM 权重。"""
from __future__ import annotations

from collections import Counter

from tabular_data.evaluation import metric_improvement, SELECTION_POLICY


DEFAULT_ROUNDS = 5
DEFAULT_CANDIDATES = 5
FEEDBACK_POLICY = {
    "source": "prior_round_validation_only", "retrain_llm": False,
    "actions": {"invalid": "改用稳定的标准化、log1p/tanh 或有下界的分母。",
                "unused": "简化复合关系并更换列组合，让浅树更容易使用。",
                "metric_regression": "压缩极值或改变分段门槛，并降低相同失败列组合的优先级。",
                "insufficient_gain": "更换列组合或函数形式，避免只重复已有信号。",
                "not_selected": "保留为有潜力方向；相对于下一轮新基线重新验证。",
                "accepted": "可探索其输入列的其他组合，仍须通过完整验证。"},
    "ranking_weights": {"accepted": 1.0, "not_selected": 0.5, "invalid": -2.0,
                        "unused": -1.5, "metric_regression": -1.5, "insufficient_gain": -1.0},
    "ranking_note": "预设搜索启发式，权重不是学得的因果效应；保留家族覆盖和探索名额。",
}


def round_feedback(iteration, task_type):
    records = []
    before = iteration["metrics_before"]
    primary = "f1_macro" if task_type == "classification" else "mse"
    for candidate in iteration["candidates"]:
        proposal = candidate["proposal"]
        changes = metric_improvement(before, candidate.get("metrics") or {})
        degraded = [key for key, value in changes.items() if value is not None and
                    value < -SELECTION_POLICY["numeric_tolerance"] * max(1., abs(before.get(key) or 0))]
        usage = candidate.get("model_usage", {}).get(proposal["name"], {})
        # 无模型证据时保持未知，不能误判成“未使用”。
        used = usage.get("used_in_splits")
        if used is None and "split_count" in usage:
            used = usage["split_count"] > 0
        if candidate["accepted"]:
            outcome = "accepted"
        elif not candidate["valid"]:
            outcome = "invalid"
        elif candidate["passes_metric_guard"]:
            outcome = "not_selected"
        elif degraded:
            outcome = "metric_regression"
        elif used is False:
            outcome = "unused"
        else:
            outcome = "insufficient_gain"
        record = {"round": iteration["round"], "name": proposal["name"], "family": proposal["family"],
                  "input_columns": proposal["input_columns"], "expression": proposal["expression"],
                  "display_expression": proposal.get("display_expression", proposal["expression"]),
                  "outcome": outcome, "reason": candidate["reason"], "degraded_metrics": degraded,
                  "metric_changes": {k: v for k, v in changes.items() if v is not None},
                  "primary_gain": changes.get(primary), "used_by_model": used,
                  "action": FEEDBACK_POLICY["actions"][outcome]}
        candidate["learning_feedback"] = record
        records.append(record)
    return {"round": iteration["round"], "evaluation_split": iteration["evaluation_split"], "candidates": records}


def prior_feedback(history, round_index):
    # 只允许更早轮次，明确拒绝测试/消融记录流入候选生成。
    return [row for item in history if isinstance(item, dict)
            and item.get("round", round_index) < round_index
            and item.get("evaluation_split") in {"validation", "cross_validation_demo"}
            for row in item.get("candidates", []) if "outcome" in row]


def feedback_summary(records):
    return {"observations": len(records), "outcomes": dict(Counter(r["outcome"] for r in records)),
            "policy": FEEDBACK_POLICY}


def candidate_priority(proposal, records):
    """可解释的失败频次惩罚，避免重复试验同一列集合和家族。"""
    score = 0.
    columns = set(proposal.input_columns)
    for row in records:
        old = set(row["input_columns"])
        overlap = len(columns & old) / max(1, len(columns | old))
        related = 1.0 if columns == old else 0.25 * overlap
        if proposal.family != row["family"]:
            related *= .25
        score += related * FEEDBACK_POLICY["ranking_weights"][row["outcome"]]
    return score
