"""新数据的版本、标签泄漏防护和官方波段顺序检查。"""
import json
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from tabular_data.dataset_registry import ROUND3_DATASETS
from tabular_data.paths import DATASET_DIR
from tabular_data.prepare_datasets import download, prepare_frames
from tabular_data.train_relations import make_plan


@unittest.skipUnless(ROUND3_DATASETS, "原第三轮数据已按用户要求移除")
class Round3DatasetTests(unittest.TestCase):
    def test_splits_match_pretraining_screening_and_saved_csvs(self):
        audit = json.loads((DATASET_DIR/'_screening_round3'/'audit.json').read_text())
        for key in ROUND3_DATASETS:
            with self.subTest(dataset=key):
                train, val, test, rows, detail = prepare_frames(key, download(key, True))
                self.assertEqual(rows, audit['datasets'][key]['rows_raw'])
                self.assertEqual(detail['raw_class_counts'], audit['datasets'][key]['raw_class_counts'])
                self.assertEqual(detail['class_distribution']['cross_split_overlap'], 0)
                for split, frame in (('train', train), ('validation', val), ('test', test)):
                    saved = pd.read_csv(DATASET_DIR/key/f'{split}.csv')
                    pd.testing.assert_frame_equal(frame.reset_index(drop=True).astype({'target': str}),
                                                  saved.astype({'target': str}))
                    self.assertEqual(saved.target.nunique(), audit['datasets'][key]['classes'])
                    self.assertGreaterEqual(saved.target.value_counts().min(), 5)

    def test_steel_has_seven_classes_and_no_fault_indicators_as_inputs(self):
        frame = pd.read_csv(DATASET_DIR/'steel_plates'/'train.csv')
        self.assertEqual(list(frame.columns[:-1]), [f'V{i}' for i in range(1, 28)])
        self.assertEqual(set(frame.target), {'Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'})
        manifest = json.loads((DATASET_DIR/'steel_plates'/'manifest.json').read_text())
        self.assertEqual((manifest['openml_id'], manifest['source_version']), (40982, 3))

    def test_satimage_preserves_all_official_rows_and_center_bands(self):
        with ZipFile(DATASET_DIR/'_raw'/'satimage_uci.zip') as archive:
            raw = pd.concat([pd.read_csv(archive.open(n), sep=r'\s+', header=None) for n in ('sat.trn', 'sat.tst')], ignore_index=True)
        frames = prepare_frames('satimage', download('satimage', True))[:3]
        recovered = pd.concat(frames).sort_index()
        self.assertEqual(len(recovered), 6435)
        np.testing.assert_array_equal(recovered.iloc[:, :-1], raw.iloc[:, :-1])
        np.testing.assert_array_equal(recovered.target.astype(int), raw[36])
        self.assertEqual(list(recovered.columns[16:20]), [f'pixel_5_band_{i}' for i in range(1, 5)])
        self.assertEqual(set(recovered.target), {'1', '2', '3', '4', '5', '7'})
        manifest = json.loads((DATASET_DIR/'satimage'/'manifest.json').read_text())
        self.assertIsNone(manifest['openml_id'])
        self.assertEqual(manifest['uci_id'], 146)

    def test_new_plan_does_not_include_previous_cohort(self):
        plan = make_plan(ROUND3_DATASETS, 'mfeat_morphological')
        self.assertEqual(tuple(plan['datasets']), ROUND3_DATASETS)
        self.assertEqual(plan['criteria']['balanced_control'], 'mfeat_morphological')
        self.assertEqual(plan['configuration']['profile'], 'reference')
        self.assertFalse(plan['test_scores_used_for_selection'])


if __name__ == '__main__':
    unittest.main()
