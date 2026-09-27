from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.io_utils import read_jsonl
from experiment.reporting import save_all


def main() -> None:
    records_path = ROOT / "outputs" / "experiment_records.jsonl"
    metadata_path = ROOT / "outputs" / "run_metadata.json"
    if not records_path.exists():
        raise FileNotFoundError("缺少 outputs/experiment_records.jsonl，请先运行实验。")
    records = read_jsonl(records_path)
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    metadata["sample_count"] = len(records)
    metadata.setdefault("model", "unknown")
    metadata.setdefault("seed", "unknown")
    save_all(ROOT, records, metadata)
    print(f"已根据 {len(records)} 条记录重建 CSV、PNG、SVG 和 Markdown 报告。")


if __name__ == "__main__":
    main()
