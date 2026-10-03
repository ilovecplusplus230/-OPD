import copy
import unittest

import pandas as pd

from tabular_data.split_validation import audit_class_counts, audit_frames


class SplitValidationTests(unittest.TestCase):
    def setUp(self):
        self.counts = {"train": {"yes": 10, "no": 10}, "validation": {"yes": 20, "no": 20}, "test": {"yes": 20, "no": 20}}

    def test_each_split_preserves_the_original_distribution(self):
        counts = {"train": {"yes": 5, "no": 15}, "validation": {"yes": 10, "no": 30}, "test": {"yes": 10, "no": 30}}
        audit = audit_class_counts(counts)
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(all(r["proportion"]==.25 for r in audit["records"] if r["class"]=="yes"))

    def test_missing_class_and_distribution_drift_fail(self):
        for train_count in (0, 6):
            counts = copy.deepcopy(self.counts)
            counts["train"] = {"yes":train_count,"no":20-train_count}
            counts["validation"] = {"yes":30-train_count,"no":10+train_count}
            with self.assertRaises(ValueError):
                audit_class_counts(counts)

    def test_multiclass_is_supported_and_all_classes_must_be_present(self):
        counts = {"train":{"a":5,"b":10,"c":15}, "validation":{"a":10,"b":20,"c":30}, "test":{"a":10,"b":20,"c":30}}
        self.assertEqual(len(audit_class_counts(counts)["records"]),9)
        counts["train"]["b"] += counts["train"].pop("a")
        with self.assertRaises(ValueError):
            audit_class_counts(counts)

    def test_no_samples_can_disappear_or_be_duplicated_between_splits(self):
        with self.assertRaisesRegex(ValueError,"原始数据不符"):
            audit_class_counts(self.counts, original_counts={"yes":49,"no":51})
        all_data = pd.DataFrame({"x":range(100),"target":["yes","no"]*50})
        frames={"train":all_data.iloc[:20],"validation":all_data.iloc[20:60],"test":all_data.iloc[60:]}
        self.assertEqual(audit_frames(frames)["cross_split_overlap"],0)
        frames["test"] = frames["validation"].copy()
        with self.assertRaisesRegex(ValueError,"相同样本"):
            audit_frames(frames)

    def test_total_train_validation_test_ratio_cannot_drift(self):
        with self.assertRaisesRegex(ValueError,"20/40/40"):
            audit_class_counts({s:{"yes":10,"no":10} for s in ("train","validation","test")})


if __name__ == "__main__":
    unittest.main()
