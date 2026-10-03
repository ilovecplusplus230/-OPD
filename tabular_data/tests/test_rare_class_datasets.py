"""稀少类别数据版本、分层与训练专属独热编码的回归检查。"""
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from tabular_data.dataset_registry import RARE_CLASS_DATASETS, SPECS
from tabular_data.paths import DATASET_DIR
from tabular_data.prepare_datasets import prepare_frames
from tabular_data.training import TrainPreprocessor


class RareClassDatasetTests(unittest.TestCase):
    def test_three_class_sources_and_reproducible_splits(self):
        for key in RARE_CLASS_DATASETS:
            with self.subTest(dataset=key):
                source = DATASET_DIR / "_raw" / f"{key}.arff"
                self.assertEqual(hashlib.md5(source.read_bytes()).hexdigest(), SPECS[key]["md5"])
                train, validation, test, rows, detail = prepare_frames(key, source)
                self.assertEqual(rows, SPECS[key]["rows"])
                self.assertEqual(len(detail["raw_class_counts"]), 3)
                self.assertLessEqual(min(detail["raw_class_counts"].values()) / rows, .1)
                self.assertEqual(detail["class_distribution"]["cross_split_overlap"], 0)
                for split, expected in (("train", train), ("validation", validation), ("test", test)):
                    saved = pd.read_csv(DATASET_DIR / key / f"{split}.csv")
                    pd.testing.assert_frame_equal(expected.reset_index(drop=True).astype({"target": str}),
                                                  saved.astype({"target": str}))
                    self.assertEqual(saved.target.nunique(), 3)
                    self.assertGreaterEqual(saved.target.value_counts().min(), 5)
                manifest = json.loads((DATASET_DIR / key / "manifest.json").read_text())
                self.assertEqual(manifest["preparation"]["raw_class_counts"], detail["raw_class_counts"])

    def test_onehot_is_fit_on_training_only_and_unknown_category_is_zero(self):
        train = pd.DataFrame({"a1": ["blank", "token_1", "token_2"], "x": [1., 3., np.nan],
                              "target": ["draw", "loss", "win"]})
        holdout = pd.DataFrame({"a1": ["unseen", "token_1"], "x": [np.nan, 999.], "target": ["win", "draw"]})
        processor = TrainPreprocessor().fit(train, "classification", categorical_encoding="onehot")
        a, b = processor.transform(train), processor.transform(holdout)
        self.assertEqual(list(a.columns), list(b.columns))
        self.assertNotIn("a1", a)
        self.assertNotIn("unseen", processor.categories["a1"])
        self.assertEqual(b.loc[0, list(processor.onehot_columns)].sum(), 0)
        self.assertEqual(b.loc[1, "a1__is_token_1"], 1)
        self.assertEqual(b.x.tolist(), [2., 999.])
        self.assertEqual(set(processor.generator_columns), set(a.columns) - {"target"})
        self.assertEqual(processor.to_dict()["unknown_category_value"], "all_zero")

    @unittest.skipUnless("connect4" in SPECS, "该历史数据集已移除")
    def test_connect4_has_no_arbitrary_numeric_cell_codes_after_preprocessing(self):
        frame = pd.read_csv(DATASET_DIR / "connect4" / "train.csv")
        processor = TrainPreprocessor().fit(frame, "classification", categorical_encoding="onehot")
        transformed = processor.transform(frame)
        self.assertFalse(processor.numeric)
        self.assertEqual(len(processor.categories), 42)
        self.assertTrue(set(np.unique(transformed.drop(columns="target"))) <= {0., 1.})
        np.testing.assert_array_equal(transformed.drop(columns="target").sum(axis=1), 42)
        self.assertTrue(all("__is_" in c for c in processor.generator_columns))


if __name__ == "__main__":
    unittest.main()
