"""新任务的原始标签、KRK 坐标语义与训练定义的稀少类别诊断。"""
import hashlib
from io import StringIO
import json
import re
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from tabular_data.dataset_registry import RELATIONAL_DATASETS, SPECS
from tabular_data.paths import DATASET_DIR
from tabular_data.prepare_datasets import normalize, prepare_frames
from tabular_data.research import sparse_labels, sparse_class_scores
from tabular_data.training import TrainPreprocessor


class RelationalDatasetTests(unittest.TestCase):
    def test_original_classes_and_reproducible_nonoverlapping_splits(self):
        for key in RELATIONAL_DATASETS:
            with self.subTest(dataset=key):
                spec = SPECS[key]
                source = DATASET_DIR / "_raw" / f"{key}.arff"
                self.assertEqual(hashlib.md5(source.read_bytes()).hexdigest(), spec["md5"])
                train, validation, test, rows, detail = prepare_frames(key, source)
                self.assertEqual(rows, spec["rows"])
                self.assertEqual(len(detail["raw_class_counts"]), spec["classes"])
                self.assertEqual(detail["class_distribution"]["cross_split_overlap"], 0)
                for split, expected in (("train", train), ("validation", validation), ("test", test)):
                    saved = pd.read_csv(DATASET_DIR / key / f"{split}.csv")
                    pd.testing.assert_frame_equal(expected.reset_index(drop=True).astype({"target": str}),
                                                  saved.astype({"target": str}))
                    self.assertEqual(set(saved.target.astype(str)), set(detail["raw_class_counts"]))
                    self.assertGreaterEqual(saved.target.value_counts().min(), 5)

    def test_krk_coordinates_and_labels_match_every_uci_row(self):
        raw = DATASET_DIR / "_raw"
        audit = json.loads((raw / "chess_krk_coordinate_audit.json").read_text())
        archive = raw / "chess_krk_uci.zip"
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), audit["uci_archive_sha256"])
        with ZipFile(archive) as source:
            official = pd.read_csv(source.open("krkopt.data"), header=None)
        text = re.sub(r"(?<=,) +", "", (raw / "chess_krk.arff").read_text())
        records, _ = loadarff(StringIO(text))
        mirrored = normalize(pd.DataFrame(records), "Class")
        for index, column in enumerate(mirrored.columns[:-1]):
            expected = official[index].map(audit["coordinate_mapping"]) if index % 2 == 0 else official[index]
            np.testing.assert_array_equal(pd.to_numeric(mirrored[column]), expected)
        spec = SPECS["chess_krk"]
        np.testing.assert_array_equal(mirrored.target.map(spec["class_names"]), official[6])
        self.assertEqual(len(official), 28056)
        prepared = pd.read_csv(DATASET_DIR / "chess_krk" / "train.csv")
        self.assertEqual(list(prepared.columns[:-1]), spec["coordinate_columns"])
        self.assertTrue(prepared.iloc[:, :-1].isin(range(1, 9)).all().all())

    @unittest.skipUnless("car_evaluation" in SPECS, "该历史数据集已移除")
    def test_car_uses_only_training_fit_onehot_and_keeps_all_six_fields(self):
        frame = pd.read_csv(DATASET_DIR / "car_evaluation" / "train.csv")
        processor = TrainPreprocessor().fit(frame, "classification", categorical_encoding="onehot")
        transformed = processor.transform(frame)
        self.assertEqual(set(processor.categories), set(frame.columns) - {"target"})
        self.assertFalse(processor.numeric)
        self.assertTrue(set(np.unique(transformed.drop(columns="target"))) <= {0., 1.})
        np.testing.assert_array_equal(transformed.drop(columns="target").sum(axis=1), 6)

    def test_sparse_group_is_defined_by_training_counts_and_averaged_equally(self):
        result = {"experiment": {"training_budget": {"class_counts": {"a": 94, "b": 5, "c": 1}}},
                  "test_class_diagnostics": {
                      "baseline": {"a": {"f1-score": 0.}, "b": {"f1-score": .2}, "c": {"f1-score": .8}},
                      "optimized": {"a": {"f1-score": 1.}, "b": {"f1-score": .4}, "c": {"f1-score": .6}}}}
        self.assertEqual(sparse_labels(result), ["b", "c"])
        self.assertEqual(sparse_class_scores(result), {"baseline": .5, "optimized": .5})
        result["experiment"]["training_budget"]["class_counts"] = {"a": 34, "b": 33, "c": 33}
        self.assertEqual(sparse_labels(result), [])
        self.assertIsNone(sparse_class_scores(result))


if __name__ == "__main__":
    unittest.main()
