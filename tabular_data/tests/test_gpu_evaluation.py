"""GPU XGBoost 训练、交叉验证及禁止 CPU 回退的检查。"""
import json
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from xgboost.core import XGBoostError

from tabular_data import tabular_octree as tabular
from tabular_data.feature_generation import OfflineFeatureClient
from tabular_data.ablation import run_ablation
from tabular_data.evaluation import make_model


class GpuEvaluationTests(unittest.TestCase):
    def test_cpu_training_is_rejected(self):
        booster = Mock()
        booster.save_config.return_value = json.dumps({
            "learner": {"generic_param": {"device": "cpu"}},
        })
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            tabular._RequireCudaTraining().after_training(booster)

    def test_cuda_failure_remains_visible(self):
        data, _, _ = tabular.preprocess_dataframe(tabular.get_data())
        error = XGBoostError("cudaErrorInsufficientDriver")
        with patch.object(tabular, "_classification_model", side_effect=error):
            with self.assertRaises(XGBoostError):
                tabular.evaluate_and_reason(data)

    def test_only_xgboost_can_be_selected(self):
        data, _, _ = tabular.preprocess_dataframe(tabular.get_data())
        with self.assertRaises(ValueError):
            tabular.evaluate_and_reason(data, model_backend="random_forest")

    def test_classification_trains_on_the_actual_gpu(self):
        data, _, _ = tabular.preprocess_dataframe(tabular.get_data())
        model = tabular._classification_model(data.target.nunique())
        model.fit(data.drop(columns="target"), data.target)
        config = json.loads(model.get_booster().save_config())
        self.assertEqual(config["learner"]["generic_param"]["device"], "cuda:0")

    def test_regression_gpu_cross_validation(self):
        data = pd.DataFrame({
            "x": np.linspace(0, 1, 30),
            "z": np.linspace(0, 1, 30) ** 2,
            "target": np.linspace(1, 5, 30) ** 2,
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = tabular.evaluate_and_reason(data, task_type="regression")
        self.assertFalse(any("mismatched devices" in str(item.message) for item in caught))
        self.assertEqual(report["model_backend"], "xgboost")
        self.assertEqual(report["device"], "cuda:0")
        self.assertTrue(np.isfinite(report["metrics"]["mse"]))
        self.assertTrue(np.isfinite(report["metrics"]["rmse"]))
        self.assertTrue(np.isfinite(report["metrics"]["mae"]))
        self.assertTrue(np.isfinite(report["metrics"]["r2"]))
        self.assertIsNone(report["metrics"]["accuracy"])

    def test_all_feature_rounds_use_gpu_xgboost(self):
        result = tabular.run_octree_analysis(
            llm_client=OfflineFeatureClient(), save_outputs=False,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["model"]["backend"], "xgboost")
        self.assertEqual(result["model"]["device"], "cuda:0")
        self.assertEqual(result["model"]["xgboost_version"], "3.4.1")
        self.assertTrue(result["model"]["cuda_version"].startswith("12."))
        self.assertEqual(result["record"]["verification"]["model"], result["model"])
        self.assertTrue(all(row["model_backend"] == "xgboost" for row in result["iterations"]))
        self.assertEqual(len(result["iterations"]), 5)
        self.assertTrue(all(len(row["candidates"]) == 5 for row in result["iterations"]))
        self.assertTrue(all(row["learning_feedback"]["evaluation_split"] == "cross_validation_demo"
                            for row in result["iterations"]))

    def test_gpu_ablation_detects_a_used_interaction_and_an_unused_constant(self):
        rng = np.random.default_rng(19)
        frame = pd.DataFrame(rng.uniform(-1, 1, size=(1000, 2)), columns=["a", "b"])
        frame["target"] = (frame.a * frame.b > 0).astype(int)
        train, test = frame.iloc[:500].copy(), frame.iloc[500:].copy()
        baseline = make_model("classification")
        baseline.fit(train.drop(columns="target"), train.target)
        for part in (train, test):
            part["interaction"] = part.a * part.b
            part["unused"] = 0.
        model = make_model("classification")
        model.fit(train.drop(columns="target"), train.target)
        proposals = [{"name": "interaction", "input_columns": ["a", "b"]},
                     {"name": "unused", "input_columns": ["a"]}]
        result = run_ablation(model, baseline, train, test, proposals, repeats=5)
        rows = {row["feature"]: row for row in result["feature_rows"]}
        self.assertTrue(rows["interaction"]["used_by_xgboost"])
        self.assertTrue(rows["interaction"]["supported_predictive_contribution"])
        self.assertGreater(rows["interaction"]["importance_score"], .2)
        self.assertFalse(rows["unused"]["used_by_xgboost"])
        self.assertEqual(rows["unused"]["importance_score"], 0.)


if __name__ == "__main__":
    unittest.main()
