"""验证训练限定的解释依据、误差预筛和独立难度实验。"""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabular_data.feature_generation import OfflineFeatureClient
from tabular_data.feature_proposals import propose_feature_candidates
from tabular_data.model_guidance import build_guidance, public_guidance, screen_candidates
from tabular_data.research_protocol import training_budget
from tabular_data.research import experiment_plan
from tabular_data.tabular_octree import check_feature_robustness
from llm_client import LLMClient


class ResearchGuidanceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.frame = pd.DataFrame(rng.normal(size=(600, 5)), columns=list("abcde"))
        self.frame["target"] = (self.frame.a * self.frame.b > 0).astype(int)
        self.probabilities = np.tile([.5, .5], (len(self.frame), 1))
        self.context = build_guidance(self.frame, list("abcde"), predictions=np.zeros(len(self.frame)),
                                      probabilities=self.probabilities)

    def test_cart_thresholds_have_training_boundaries_and_frozen_rowwise_output(self):
        candidates = propose_feature_candidates(self.frame, OfflineFeatureClient(), 1, [], set(),
                                                mode="reasoned", guidance=self.context)
        self.assertEqual(len(candidates), 5)
        tree = next(p for p in candidates if p.source == "cart_rule")
        self.assertTrue(any(p["kind"] == "training_tree_threshold" for p in tree.constant_provenance))
        for parameter in self.context["thresholds"].values():
            low, high = parameter["boundary_values"]
            self.assertLessEqual(low, parameter["value"])
            self.assertLess(parameter["value"], high)
            self.assertGreaterEqual(parameter["impurity_before"] + 1e-10, parameter["weighted_impurity_after"])
        for candidate in candidates:
            passed, whole, _, _ = check_feature_robustness(self.frame, candidate.code)
            self.assertTrue(passed)
            passed, single, _, _ = check_feature_robustness(self.frame.iloc[[5]], candidate.code, allow_constant=True)
            self.assertTrue(passed)
            self.assertEqual(whole[candidate.name].iloc[5], single[candidate.name].iloc[0])
            json.dumps(candidate.to_dict())
        self.assertNotIn("_residuals", json.dumps(public_guidance(self.context)))

    def test_residual_screen_prefers_interaction_over_irrelevant_column(self):
        from tabular_data.feature_proposals import FeatureProposal
        pool = [FeatureProposal(name, "nonlinear", inputs, expression, "fixture", "fixture", {})
                for name, inputs, expression in (("signal", ["a", "b"], "df['a'] * df['b']"),
                                                 ("noise", ["e"], "df['e']"))]
        ranked = screen_candidates(pool, self.frame, self.context)
        self.assertEqual(ranked[0].name, "signal")
        self.assertGreater(ranked[0].evidence["prescreen"]["score"], .5)
        self.assertLess(ranked[1].evidence["prescreen"]["score"], .1)

    def test_llm_receives_tree_context_and_can_use_audited_thresholds(self):
        symbol, threshold = next(iter(self.context["thresholds"].items()))
        column = threshold["column"]
        client = LLMClient(api_key="test-placeholder")
        payload = {"features": [{"name": "tree_gate", "family": "piecewise",
            "expression": f"np.where(df[{column!r}] <= {symbol}, df['d'] * df['e'], 0)",
            "hypothesis": "训练树指示的子群可能存在条件交互。", "construction": "冻结训练树门槛，分段计算乘积。"}]}
        with patch.object(client, "chat", return_value=json.dumps(payload)) as chat:
            plans = propose_feature_candidates(self.frame, client, 1, [], set(), count=1,
                                                mode="reasoned", guidance=self.context)
        self.assertEqual(plans[0].source, "llm")
        self.assertIn("cart_rules", chat.call_args.args[0])
        self.assertIn("class_diagnostics", chat.call_args.args[0])
        self.assertNotIn("_residuals", chat.call_args.args[0])
        self.assertIn(symbol, plans[0].display_expression)
        self.assertNotIn(symbol, plans[0].expression)
        self.assertEqual(next(p for p in plans[0].constant_provenance if p["symbol"] == symbol)["source"], "train")

    def test_reasoned_five_rounds_are_distinct_and_explained(self):
        seen = set()
        for round_index in range(1, 6):
            candidates = propose_feature_candidates(self.frame, OfflineFeatureClient(), round_index, [], seen,
                                                    mode="reasoned", guidance=self.context)
            self.assertEqual(len(candidates), 5)
            for candidate in candidates:
                self.assertNotIn(candidate.signature, seen)
                seen.add(candidate.signature)
                self.assertIn("prescreen", candidate.evidence)
                self.assertTrue(candidate.derivation)

    def test_repair_skips_unavailable_old_tree_parameter_without_losing_round(self):
        from tabular_data.feature_proposals import _repair_candidates, training_profile
        profile = training_profile(self.frame, list("abcde"))
        record = {"round": 1, "name": "previous_llm_gate", "family": "piecewise",
                  "input_columns": ["a", "b", "c"], "outcome": "metric_regression",
                  "reason": "fixture regression", "degraded_metrics": ["accuracy"],
                  "display_expression": "np.where(df['a'] <= cart_r1_node0, df['b'] - median_b, df['c'])"}
        repaired = _repair_candidates(profile, [record])
        self.assertTrue(repaired)
        self.assertTrue(all("cart_r1_node0" not in p.display_expression for p in repaired))


    def test_training_budget_reproducible_stratified_not_rebalanced(self):
        frame = pd.DataFrame({"x": np.arange(1000), "target": [0]*850 + [1]*150})
        a, meta_a = training_budget(frame, .5)
        b, meta_b = training_budget(frame, .5)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(meta_a, meta_b)
        self.assertEqual(meta_a["class_counts"], {"0": 425, "1": 75})
        self.assertEqual(len(frame), 1000)
        self.assertFalse(meta_a["unused_rows_used_elsewhere"])
        with self.assertRaises(ValueError):
            training_budget(frame, .01)

    def test_research_plan_uses_current_multiclass_datasets_and_generator(self):
        from tabular_data.dataset_registry import DATASETS
        plan = experiment_plan()
        self.assertEqual(len(plan), len(DATASETS))
        self.assertEqual({p["dataset"] for p in plan}, set(DATASETS))
        self.assertEqual({p["feature_mode"] for p in plan}, {"reasoned"})
        self.assertTrue(all(p["iterations"] == p["candidates_per_round"] == 5 for p in plan))
        reference = experiment_plan(seeds=(42, 43), profile="reference")
        self.assertEqual(len(reference), 2 * len(DATASETS))
        self.assertTrue(all(p["profile"] == "reference" for p in reference))

    def test_training_applies_budget_before_fit_but_keeps_holdout_rows(self):
        from tabular_data import training
        frame = pd.DataFrame(np.random.default_rng(1).normal(size=(500, 6)),
                             columns=list("abcdef"))
        frame["target"] = np.tile([0, 1], 250)
        frames = {"train": frame.iloc[:100], "validation": frame.iloc[100:300], "test": frame.iloc[300:]}
        counts = {s: {"0": len(f)//2, "1": len(f)//2} for s, f in frames.items()}
        manifest = {"task_type": "classification", "preparation": {"split": {"protocol": training.SPLIT_PROTOCOL, "seed": 42}},
                    "class_counts": counts, "files": {s: {"rows": len(f), "sha256": "fixture"} for s,f in frames.items()}}
        events = []
        def load(directory, metadata, split):
            events.append(split)
            return frames[split].copy()
        def fit(data, *args):
            events.append("fit")
            self.assertEqual(len(data), 50)
            self.assertEqual(set(data.columns), set("abcdef") | {"target"})
            return object()
        def evaluate(model, data, task):
            self.assertEqual(len(data), 200)
            return {"f1_macro": .8, "accuracy": .8}, data.target.to_numpy()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(folder)/"jungle_chess"
            directory.mkdir()
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with patch.object(training, "DATASET_DIR", Path(folder)), patch.object(training, "_load_split", side_effect=load), \
                 patch.object(training, "_fit", side_effect=fit), patch.object(training, "_evaluate", side_effect=evaluate), \
                 patch.object(training, "run_ablation", return_value={"status": "fixture"}):
                result = training.run_training("jungle_chess", iterations=0, offline=True, save_outputs=False,
                                               train_fraction=.5)
        self.assertEqual(events, ["train", "validation", "fit", "test"])
        self.assertEqual(result["rows"], {"train": 50, "validation": 200, "test": 200})
        self.assertEqual(result["experiment"]["effective_fraction_of_all_rows"], .1)
        self.assertEqual(result["experiment"]["training_budget"]["class_counts"], {"0": 25, "1": 25})


if __name__ == "__main__":
    unittest.main()
