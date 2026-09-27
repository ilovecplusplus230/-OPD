from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "original_math_package"))

from experiment.answer_verifier import AnswerVerifier, reference_from_record
from experiment.io_utils import read_jsonl, write_json
from math_reasoning_expander.parser import ReasoningGraphParser


def main() -> None:
    source = ROOT / "datasets" / "processed" / "combined.jsonl"
    records = read_jsonl(source)
    verifier = AnswerVerifier()
    parser = ReasoningGraphParser()
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rejected: list[dict[str, str]] = []

    for record in records:
        dataset = str(record.get("dataset", "unknown"))
        stats[dataset]["total"] += 1
        missing = [name for name in ("question", "solution") if not str(record.get(name, "")).strip()]
        reference = reference_from_record(record)
        if not reference:
            missing.append("answer")
        if missing:
            stats[dataset]["missing_fields"] += 1
            rejected.append({"id": str(record.get("id")), "reason": "missing: " + ", ".join(missing)})
            continue

        graph = parser.parse(str(record["question"]), str(record["solution"]))
        if len(graph.nodes) < 3:
            stats[dataset]["too_short"] += 1
            rejected.append({"id": str(record.get("id")), "reason": "fewer than 3 reasoning nodes"})
            continue

        verification = verifier.verify_text(str(record["solution"]), reference)
        if verification.correct:
            stats[dataset]["verifier_pass"] += 1
        elif verification.status == "manual_required":
            stats[dataset]["manual_required"] += 1
        else:
            stats[dataset]["verifier_fail"] += 1

        stats[dataset]["usable"] += 1

    payload = {
        "source": str(source),
        "stats": {dataset: dict(values) for dataset, values in sorted(stats.items())},
        "rejected": rejected,
    }
    write_json(ROOT / "datasets" / "audit.json", payload)
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
