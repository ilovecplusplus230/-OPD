"""正式训练的指标、防泄漏、划分和噪声实验回归检查（无需 GPU）。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tabular_data.evaluation import accept_candidate, compute_metrics
from tabular_data.paths import DATASET_DIR
from tabular_data.prepare_datasets import prepare_frames
from tabular_data.tabular_octree import check_feature_robustness
from tabular_data import training, tabular_octree
from tabular_data.feature_proposals import FeatureProposal


class TrainingProtocolTests(unittest.TestCase):
    def test_auc_gain_cannot_hide_accuracy_or_f1_regression(self):
        baseline = dict(accuracy=.9233893299, f1=.9234479610, auc=.9946432045, log_loss=.2285651565)
        candidate = dict(accuracy=.9228355178, f1=.9228835679, auc=.9948376950, log_loss=.2226327658)
        accepted, reason = accept_candidate(baseline, candidate, "classification")
        self.assertFalse(accepted)
        self.assertIn("退化", reason)
        better = dict(baseline, accuracy=.924, f1=.924, auc=.995, log_loss=.22)
        self.assertTrue(accept_candidate(baseline, better, "classification")[0])

    def test_regression_requires_relative_mse_gain_and_no_mae_regression(self):
        baseline = dict(mse=100., rmse=10., mae=7., r2=.4)
        self.assertFalse(accept_candidate(baseline, dict(baseline, mse=99.99), "regression")[0])
        self.assertFalse(accept_candidate(baseline, dict(mse=90., rmse=9.5, mae=8., r2=.5), "regression")[0])
        self.assertTrue(accept_candidate(baseline, dict(mse=90., rmse=9.5, mae=6., r2=.5), "regression")[0])
        self.assertFalse(accept_candidate(baseline, dict(mse=90., rmse=9.5, mae=float("nan"), r2=.5), "regression")[0])

    def test_regression_reports_meaningful_metrics(self):
        metrics = compute_metrics(np.array([1., 2., 3.]), np.array([1., 3., 2.]), None, "regression")
        self.assertAlmostEqual(metrics["mse"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["r2"], 0.)
        self.assertIsNone(metrics["accuracy"])

    def test_preprocessing_fits_training_data_only(self):
        train = pd.DataFrame(dict(x=[1., 3., np.nan], cat=["b", "a", "b"], target=["no", "yes", "no"]))
        validation = pd.DataFrame(dict(x=[1000., np.nan], cat=["new", "b"], target=["yes", "no"]))
        processor = training.TrainPreprocessor().fit(train, "classification")
        result = processor.transform(validation)
        self.assertEqual(result.x.tolist(), [1000., 2.])
        self.assertEqual(result.cat.tolist(), [-1., 1.])
        self.assertEqual(processor.medians["x"], 2.)
        self.assertEqual(processor.classes, ["no", "yes"])
        with self.assertRaisesRegex(ValueError, "未见"):
            processor.transform(validation.assign(target="unknown"))

    def test_noise_is_repeatable_and_preserves_labels_and_source(self):
        frame = pd.DataFrame(dict(x=[1., 3., 6., 8.], target=[0, 1, 0, 1]))
        original = frame.copy(deep=True)
        noise = training.FeatureNoise(frame, "pendigits", .1, 42)
        first = noise.transform(frame, "train")
        pd.testing.assert_frame_equal(first, noise.transform(frame, "train"))
        self.assertFalse(first.x.equals(noise.transform(frame, "test").x))
        self.assertFalse(first.x.equals(training.FeatureNoise(frame, "pendigits", .1, 43).transform(frame, "train").x))
        pd.testing.assert_series_equal(first.target, frame.target)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(frame, training.FeatureNoise(frame, "pendigits", 0, 42).transform(frame, "train"))

    def test_features_cannot_read_labels_or_overwrite_input(self):
        frame = pd.DataFrame(dict(a=[1., 3., 5.], b=[2., 4., 8.], target=[0, 1, 0]))
        for code in ["df['leak'] = df['target']", "df['a'] = df['b']; df['new'] = df['b'] * 2",
                     "df['target'] = df['a']", "df['new'] = df['a'] / df['a'].mean()"]:
            self.assertFalse(check_feature_robustness(frame, code)[0])
        passed, extended, _, _ = check_feature_robustness(frame, "df['sum'] = df['a'] + df['b']")
        self.assertTrue(passed)
        pd.testing.assert_frame_equal(extended[frame.columns], frame)

    def test_prepared_splits_are_reproducible_and_disjoint(self):
        for name in ("jungle_chess", "balance_scale", "chess_krk"):
            with self.subTest(dataset=name):
                train, validation, test, _, details = prepare_frames(name, DATASET_DIR / "_raw" / f"{name}.arff")
                total = len(train) + len(validation) + len(test)
                self.assertLess(abs(len(train) / total - .2), .0001)
                self.assertLess(abs(len(validation) / total - .4), .0001)
                self.assertFalse(set(train.index) & set(validation.index))
                self.assertFalse(set(train.index) & set(test.index))
                self.assertFalse(set(validation.index) & set(test.index))
                for split, frame in (("train", train), ("validation", validation), ("test", test)):
                    saved = pd.read_csv(DATASET_DIR / name / f"{split}.csv")
                    pd.testing.assert_frame_equal(frame.reset_index(drop=True).assign(target=frame.target.astype(str).to_numpy()), saved.assign(target=saved.target.astype(str)))
                    self.assertEqual(set(train.target), set(frame.target))
                self.assertEqual(details["split"]["protocol"], training.SPLIT_PROTOCOL)

    def test_test_set_is_opened_after_selection_and_never_selects_features(self):
        frame = pd.DataFrame(dict(a=np.arange(20, dtype=float), b=np.arange(20, dtype=float)*2+1,
                                  target=np.tile([0, 1], 10)))
        events = []
        client = Mock()
        client.real_call_count = client.mock_fallback_count = 0
        def generate(*args, **kwargs):
            events.append("generate")
            self.assertNotIn("test", events)
            return [FeatureProposal("sum", "multicolumn", ["a", "b"], "df['a'] + df['b']", "fixture", "fixture",
                                    {"column_statistics": {}})]
        def load(directory, manifest, split):
            events.append(split)
            return (frame.iloc[:10] if split == "train" else frame).copy()
        predictions = frame.target.to_numpy()
        scores = [(dict(accuracy=.7, f1_macro=.7), predictions),
                  (dict(accuracy=.8, f1_macro=.8), predictions),
                  (dict(accuracy=.9, f1_macro=.9), predictions),
                  (dict(accuracy=.6, f1_macro=.6), predictions)]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "jungle_chess"
            directory.mkdir()
            manifest = {"task_type": "classification", "preparation": {"split": {"protocol": training.SPLIT_PROTOCOL, "seed": 42}},
                        "class_counts": {s: {"0": n//2, "1": n//2} for s,n in (("train",10),("validation",20),("test",20))},
                        "files": {split: {"rows": n, "sha256": "fixture"} for split,n in (("train",10),("validation",20),("test",20))}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(training, "DATASET_DIR", Path(folder)), patch.object(training, "_load_split", side_effect=load), \
                 patch.object(training, "_fit"), patch.object(training, "_evaluate", side_effect=scores), \
                 patch.object(training, "predict_model", return_value=(np.zeros(10), np.full((10,2), .5))), \
                 patch.object(tabular_octree, "propose_feature_candidates", side_effect=generate), \
                 patch.object(tabular_octree, "feature_usage", return_value={}), \
                 patch.object(training, "run_ablation", return_value={"status": "fixture"}):
                result = training.run_training("jungle_chess", iterations=1, llm_client=client, save_outputs=False)
        self.assertEqual(events, ["train", "validation", "generate", "test"])
        self.assertEqual(result["accepted_features"], ["sum"])
        self.assertLess(result["test"]["improvement"]["f1_macro"], 0)

    def test_completed_round_is_saved_before_later_round_failure(self):
        frame = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(20, dtype=float)*2+1,
                              "target": np.tile([0, 1], 10)})
        proposal = FeatureProposal("sum", "nonlinear", ["a", "b"], "df['a'] + df['b']", "fixture", "fixture",
                                   {"column_statistics": {}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root/"jungle_chess"
            directory.mkdir()
            manifest = {"task_type":"classification", "preparation":{"split":{"protocol":training.SPLIT_PROTOCOL,"seed":42}},
                "class_counts":{s:{"0":n//2,"1":n//2} for s,n in (("train",10),("validation",20),("test",20))},
                "files":{s:{"rows":n,"sha256":"fixture"} for s,n in (("train",10),("validation",20),("test",20))}}
            (directory/"manifest.json").write_text(json.dumps(manifest))
            def load(directory, manifest, split):
                self.assertNotEqual(split, "test")
                return (frame.iloc[:10] if split=="train" else frame).copy()
            with patch.object(training,"DATASET_DIR",root), patch.object(training,"OUTPUT_DIR",root/"outputs"), \
                 patch.object(training,"_load_split",side_effect=load), patch.object(training,"_fit"), \
                 patch.object(training, "predict_model", return_value=(np.zeros(10), np.full((10,2), .5))), \
                 patch.object(training,"_evaluate",return_value=({"accuracy":.6,"f1_macro":.6},frame.target.to_numpy())), \
                 patch.object(tabular_octree,"feature_usage",return_value={}), \
                 patch.object(tabular_octree,"propose_feature_candidates",side_effect=[[proposal],RuntimeError("second round failed")]), \
                 self.assertRaisesRegex(RuntimeError,"second round failed"):
                training.run_training("jungle_chess",iterations=2,offline=True)
            progress_path = next((root/"outputs").rglob("progress.json"))
            progress = json.loads(progress_path.read_text())
            self.assertEqual(progress["completed_rounds"],1)
            self.assertEqual(progress["iterations"][0]["round"],1)
            explanations = json.loads((progress_path.parent/"feature_explanations.json").read_text())
            self.assertEqual(explanations[0]["candidates"][0]["proposal"]["name"],"sum")
            self.assertFalse((progress_path.parent/"training_results.json").exists())


if __name__ == "__main__":
    unittest.main()
