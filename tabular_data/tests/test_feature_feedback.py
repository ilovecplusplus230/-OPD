"""验证反馈真正改变搜索、公式来源可复核、显示精度不影响执行。"""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from llm_client import LLMClient
from tabular_data.evaluation import accept_candidate
from tabular_data.feature_explanations import number, parameter_registry
from tabular_data.feature_feedback import prior_feedback, round_feedback
from tabular_data.feature_generation import OfflineFeatureClient
from tabular_data.feature_proposals import (_domain_candidates, _proposal, propose_feature_candidates,
                                           training_profile, print_proposal)
from tabular_data.tabular_octree import evaluate_feature_round


class FeedbackTests(unittest.TestCase):
    def propose(self, *args, **kwargs):
        from tabular_data.model_guidance import build_guidance
        frame = args[0]
        probabilities = np.full((len(frame), 2), .5)
        context = build_guidance(frame, [c for c in frame if c != "target"],
                                 predictions=np.zeros(len(frame)), probabilities=probabilities)
        return propose_feature_candidates(*args, guidance=context, **kwargs)

    def setUp(self):
        rng = np.random.default_rng(31)
        self.frame = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
        self.frame["target"] = (self.frame.a * self.frame.b > 0).astype(int)
        self.plan = next(p for p in self.propose(self.frame, OfflineFeatureClient(), 1, [], set()) if p.family == "multicolumn")

    def feedback(self, outcome):
        return [{"round": 1, "evaluation_split": "validation", "candidates": [{
            "round": 1, "name": self.plan.name, "family": self.plan.family,
            "input_columns": self.plan.input_columns, "expression": self.plan.expression,
            "display_expression": self.plan.display_expression, "outcome": outcome,
            "reason": outcome, "degraded_metrics": ["accuracy"] if outcome == "metric_regression" else [],
            "metric_changes": {}, "primary_gain": 0., "used_by_model": outcome != "unused"}]}]

    def test_rejection_type_changes_repair_pool_before_residual_screening(self):
        from tabular_data.feature_proposals import _repair_candidates
        results = {}
        for kind in ("unused", "metric_regression", "insufficient_gain", "invalid"):
            records = prior_feedback(self.feedback(kind), 2)
            repairs = _repair_candidates(training_profile(self.frame, list("abcde")), records)
            self.assertEqual(repairs[0].adaptation["source_candidate"], self.plan.name)
            self.assertEqual(repairs[0].adaptation["outcome"], kind)
            results[kind] = repairs[0].signature
        self.assertEqual(len(set(results.values())), 4)

    def test_feedback_ignores_test_and_future_rounds(self):
        history = self.feedback("unused")
        polluted = copy.deepcopy(history)
        polluted[0]["evaluation_split"] = "test"
        future = copy.deepcopy(history)
        future[0]["round"] = 3
        self.assertEqual(len(prior_feedback(history + polluted + future, 2)), 1)
        self.assertEqual(prior_feedback(polluted + future, 2), [])

    def test_round_captures_all_degraded_metrics_and_unused_evidence(self):
        proposal = self.plan.to_dict()
        item = {"round": 1, "evaluation_split": "validation", "metrics_before": {"accuracy": .8, "f1_macro": .7, "log_loss": .4},
                "candidates": [{"proposal": proposal, "accepted": False, "valid": True, "passes_metric_guard": False,
                    "metrics": {"accuracy": .79, "f1_macro": .69, "log_loss": .42}, "reason": "退化",
                    "model_usage": {proposal["name"]: {"split_count": 2}}}]}
        record = round_feedback(item, "classification")["candidates"][0]
        self.assertEqual(record["outcome"], "metric_regression")
        self.assertEqual(set(record["degraded_metrics"]), {"accuracy", "f1_macro", "log_loss"})
        item["candidates"][0]["metrics"] = item["metrics_before"].copy()
        item["candidates"][0]["model_usage"][proposal["name"]]["split_count"] = 0
        self.assertEqual(round_feedback(item, "classification")["candidates"][0]["outcome"], "unused")

    def test_five_rounds_use_feedback_and_produce_twenty_five_distinct_candidates(self):
        from tabular_data.model_guidance import build_guidance
        guidance = build_guidance(self.frame, list("abcde"), predictions=np.zeros(len(self.frame)),
                                  probabilities=np.full((len(self.frame), 2), .5))
        history, seen = [], set()
        baseline = {"accuracy": .7, "f1_macro": .6}
        def evaluate(train, validation):
            name = train.columns[-1]
            return {"metrics": baseline.copy(), "feature_usage": {name: {"split_count": 0}}}
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(1, 6):
                _, _, _, _, iteration = evaluate_feature_round(self.frame, self.frame.copy(), evaluate, baseline,
                    "classification", OfflineFeatureClient(), index, history, seen, guidance=guidance)
                self.assertEqual(len(iteration["candidates"]), 5)
                if index > 1:
                    self.assertTrue(all(c["proposal"]["adaptation"]["history_summary"] for c in iteration["candidates"]))
                history.append(iteration["learning_feedback"])
        self.assertEqual(len(seen), 25)
        self.assertEqual(len(prior_feedback(history, 6)), 25)

    def test_llm_receives_structured_feedback_and_must_reference_it(self):
        client = LLMClient(api_key="test-placeholder")
        payload = {"features": [{"name": "bounded", "family": "nonlinear",
            "expression": "np.tanh((df['a'] - median_a) / scale_a) * df['b']",
            "hypothesis": "饱和关系待验证", "construction": "训练中位数居中后按离散程度缩放，再以 tanh 限幅。",
            "feedback_reference": self.plan.name, "adaptation_action": "针对退化压缩第一列幅度。"}]}
        with patch.object(client, "chat", return_value=json.dumps(payload)) as chat:
            plans = self.propose(self.frame, client, 2, self.feedback("metric_regression"), set(), count=1)
        self.assertEqual(plans[0].source, "llm")
        self.assertEqual(plans[0].adaptation["source_candidate"], self.plan.name)
        self.assertIn('"outcome": "metric_regression"', chat.call_args.args[0])
        self.assertIn("median_a", plans[0].display_expression)
        self.assertNotIn("median_a", plans[0].expression)

    def test_unexplained_llm_constants_fail_instead_of_inventing_provenance(self):
        profile = training_profile(self.frame, list("abcde"))
        with self.assertRaisesRegex(ValueError, "无可审计来源"):
            _proposal("unknown", "nonlinear", "df['a'] / 17.328721", "假说", "说明", profile)

    def test_llm_missing_history_reference_falls_back_to_actual_repairs(self):
        client = LLMClient(api_key="test-placeholder")
        payload = {"features": [{"name": "unexplained", "family": "nonlinear", "expression": "df['a'] * df['b']",
                                "hypothesis": "假说", "construction": "相乘"}]}
        with patch.object(client, "chat", return_value=json.dumps(payload)):
            plans = self.propose(self.frame, client, 2, self.feedback("unused"), set(), count=5)
        self.assertEqual(client.mock_fallback_count, 1)
        self.assertTrue(all(p.source in {"local_template", "cart_rule"} for p in plans))
        self.assertTrue(all(p.adaptation["history_summary"] for p in plans))

    def test_quantile_and_scale_provenance_reconstruct_actual_constants(self):
        profile = training_profile(self.frame, list("abcde"))
        registry = parameter_registry(profile)
        for col in "abcde":
            stats = profile["columns"][col]
            for stat in ("q25", "median", "q75"):
                detail = stats["quantile_details"][stat]
                actual = (1-detail["weight"])*detail["lower_value"] + detail["weight"]*detail["upper_value"]
                self.assertAlmostEqual(registry[f"{stat}_{col}"]["value"], actual)
            self.assertAlmostEqual(registry[f"scale_{col}"]["value"], max(stats["q75"]-stats["q25"],
                                  np.sqrt(stats["squared_deviations"]/len(self.frame)), 1e-6))
        self.assertTrue(self.plan.derivation)
        self.assertTrue(all(p["source"] and p["calculation"] and p["purpose"] for p in self.plan.constant_provenance))

    def test_display_rounding_does_not_change_code_or_decision(self):
        code = self.plan.code
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_proposal(self.plan, 1, 1)
        self.assertEqual(code, self.plan.code)
        self.assertIn("公式构造步骤", stream.getvalue())
        self.assertNotIn(repr(float(self.frame.a.median())), stream.getvalue())
        self.assertEqual(number(.123456789), "0.1235")
        self.assertNotEqual(number(-1e-8), "0")
        # 四舍五入后相等也必须按未舍入的值拒绝退化。
        self.assertFalse(accept_candidate({"accuracy": .80000003, "f1_macro": .7},
            {"accuracy": .80000001, "f1_macro": .71}, "classification")[0])

    def test_chess_domain_geometry_and_constants_are_explained(self):
        columns = [f"{color}_piece0_{field}" for color in ("white", "black")
                   for field in ("strength", "file", "rank")]
        frame = pd.DataFrame(np.random.default_rng(7).integers(0, 8, (100, 6)), columns=columns)
        plans = _domain_candidates(training_profile(frame, columns))
        self.assertEqual(len(plans), 3)
        self.assertTrue(all(p.derivation for p in plans))
        self.assertTrue(any(len(p.input_columns) == 6 for p in plans))
        self.assertTrue(any(p.constant_provenance for p in plans))


if __name__ == "__main__":
    unittest.main()
