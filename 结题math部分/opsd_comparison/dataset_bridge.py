from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def question_hash(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def convert_mathopd_eval(source: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(source)):
        question = str(row["question"]).strip()
        output.append(
            {
                "id": f"mathopd_heldout_{index:03d}",
                "dataset": "MATH-OPSD-heldout",
                "source_url": row.get("source_url", "https://github.com/hendrycks/math"),
                "source_id": row.get("original_sample_id", ""),
                "question": question,
                "solution": str(row["reference_solution"]).strip(),
                "answer": str(row["reference_answer"]).strip(),
                "difficulty": row.get("difficulty", ""),
                "subject": row.get("subject", ""),
                "question_sha256": question_hash(question),
                "training_overlap_audit": "checked against the complete OPSD train split by MATHOPD",
            }
        )
    return output


def audit_overlap(
    heldout: list[dict[str, Any]],
    closeout_pool: list[dict[str, Any]],
    formal_records: list[dict[str, Any]],
) -> dict[str, Any]:
    pool_by_hash = {question_hash(row.get("question", "")): row for row in closeout_pool}
    formal_by_hash = {question_hash(row.get("question", "")): row for row in formal_records}
    pool_matches = []
    formal_matches = []
    for row in heldout:
        digest = question_hash(row["question"])
        if digest in pool_by_hash:
            matched = pool_by_hash[digest]
            pool_matches.append(
                {
                    "heldout_id": row["id"],
                    "heldout_source_id": row.get("source_id", ""),
                    "closeout_id": matched.get("id", ""),
                    "closeout_dataset": matched.get("dataset", ""),
                }
            )
        if digest in formal_by_hash:
            matched = formal_by_hash[digest]
            formal_matches.append(
                {
                    "heldout_id": row["id"],
                    "formal_id": matched.get("id", ""),
                    "formal_dataset": matched.get("dataset", ""),
                }
            )
    return {
        "mathopd_heldout_count": len(heldout),
        "closeout_candidate_pool_count": len(closeout_pool),
        "closeout_formal_record_count": len(formal_records),
        "candidate_pool_exact_normalized_overlap_count": len(pool_matches),
        "formal_90_exact_normalized_overlap_count": len(formal_matches),
        "candidate_pool_matches": pool_matches,
        "formal_90_matches": formal_matches,
        "recommended_shared_benchmark": "MATHOPD audited held-out 40",
        "reason": (
            "These 40 questions were already checked against the complete OPSD training split. "
            "They should be reused for Base/OPSD adapter comparison without entering training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the audited shared Math benchmark")
    parser.add_argument("--mathopd-root", type=Path, required=True)
    parser.add_argument("--closeout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    heldout_source = args.mathopd_root / "results/v2_long_completion/evaluation/eval_set.jsonl"
    heldout = convert_mathopd_eval(heldout_source)
    pool = read_jsonl(args.closeout_root / "datasets/processed/combined.jsonl")
    formal = read_jsonl(args.closeout_root / "outputs/experiment_records.jsonl")
    audit = audit_overlap(heldout, pool, formal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "mathopd_heldout_40.jsonl", heldout)
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

