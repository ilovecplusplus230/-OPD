"""当前入口和运行状态索引，防止把失败/过期报告展示成成功结果。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tabular_data.dataset_registry import DATASETS
from tabular_data.refresh_summary import collect_reports, refresh_summary


class CurrentReportingTests(unittest.TestCase):
    def test_completed_matching_run_is_the_only_report_indexed(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, status, state_id, report_id in [
                ("complete", "completed", "a", "a"),
                ("failed", "failed", "b", "b"),
                ("stale", "completed", "c", "older"),
                ("running", "running", "d", "d"),
            ]:
                path = Path(folder) / name
                path.mkdir()
                result = {"dataset": "jungle_chess", "run_id": report_id,
                          "feature_generation": {"mode": "reasoned"},
                          "model": {"profile": "weak", "parameters": {"random_state": 42}}}
                (path / "training_results.json").write_text(json.dumps(result))
                (path / "run_status.json").write_text(json.dumps({"status": status, "run_id": state_id}))
            reports = collect_reports(folder)
            self.assertEqual([r["run_id"] for r in reports], ["a"])
            self.assertEqual(Path(reports[0]["output_dir"]).name, "complete")

    def test_empty_results_create_an_honest_empty_index(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(refresh_summary(folder), [])
            summary = (Path(folder) / "latest_summary.md").read_text()
            self.assertIn("Macro F1", summary)
            self.assertNotIn("0.8000", summary)

    def test_short_training_entry_help_lists_only_current_datasets(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "tabular_data/train.py", "--help"],
                                cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for dataset in DATASETS:
            self.assertIn(dataset, result.stdout)
        self.assertIn("{reasoned}", result.stdout)
        self.assertIn("20%", result.stdout)


if __name__ == "__main__":
    unittest.main()
