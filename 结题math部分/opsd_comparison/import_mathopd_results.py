from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opsd_comparison.dataset_bridge import read_jsonl, write_jsonl


ARM_TO_VARIANT = {
    "A": "opsd_original",
    "B": "opsd_history",
    "C": "opsd_error_step",
    "D": "opsd_reflection",
}


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    variant = ARM_TO_VARIANT[str(row["arm"])]
    return {
        "id": f"mathopd_heldout_{int(row['eval_index']):03d}",
        "dataset": "MATH-OPSD-heldout",
        "model_variant": variant,
        "reference_answer": str(row["reference_answer"]),
        "pass_at_1": bool(row["is_correct"]),
        "pass_at_k": bool(row["is_correct"]),
        "k": 1,
        "outputs": [
            {
                "index": 0,
                "response": row["student_answer"],
                "verification": {
                    "correct": bool(row["is_correct"]),
                    "status": row.get("verification_status", ""),
                    "extracted_answer": row.get("extracted_answer", ""),
                },
                "generated_tokens": row.get("generated_tokens"),
                "hit_token_cap": row.get("hit_token_cap"),
                "terminated_by_eos": row.get("terminated_by_eos"),
            }
        ],
        "provenance": {
            "source": "MATHOPD V2 deterministic held-out evaluation",
            "source_method": row["method"],
            "source_id": row.get("original_sample_id", ""),
            "decoding": "do_sample=false, thinking=false, max_new_tokens=1536",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse verified MATHOPD V2 held-out outputs")
    parser.add_argument("--mathopd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["opsd_original", "opsd_history"])
    args = parser.parse_args()

    unknown = sorted(set(args.variants) - set(ARM_TO_VARIANT.values()))
    if unknown:
        raise ValueError(f"未知模型组: {unknown}")

    source = args.mathopd_root / "results/v2_long_completion/evaluation/held_out_outputs.jsonl"
    converted = [convert_row(row) for row in read_jsonl(source)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source": str(source), "variants": {}}
    for variant in args.variants:
        rows = [row for row in converted if row["model_variant"] == variant]
        rows.sort(key=lambda row: row["id"])
        if len(rows) != 40:
            raise RuntimeError(f"{variant} 应有40条结果，实际为 {len(rows)} 条。")
        destination = args.output_dir / f"direct_{variant}.jsonl"
        write_jsonl(destination, rows)
        manifest["variants"][variant] = {
            "count": len(rows),
            "pass_at_1": sum(bool(row["pass_at_1"]) for row in rows) / len(rows),
            "destination": str(destination),
        }
    (args.output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
