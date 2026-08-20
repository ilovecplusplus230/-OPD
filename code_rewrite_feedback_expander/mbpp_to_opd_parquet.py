from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SPLITS = ("train", "validation", "test")


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _read_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    paths = sorted(dataset_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No MBPP parquet shard found for split={split!r} in {dataset_dir}")
    table = pq.read_table(paths)
    return table.to_pylist()


def _prompt_text(row: dict[str, Any]) -> str:
    task = str(row.get("text") or row.get("prompt") or "").strip()
    return (
        "Solve the following MBPP Python programming task. Reason step by step before writing the solution, "
        "then return the complete implementation in one fenced ```python``` block. Preserve the requested "
        "function signature and do not include placeholder code.\n\n"
        f"Task:\n{task}"
    )


def _setup_code(row: dict[str, Any]) -> str:
    setup = str(row.get("test_setup_code") or "").strip()
    imports = _as_strings(row.get("test_imports"))
    return "\n".join(part for part in [setup, *imports] if part).strip()


def convert_row(
    row: dict[str, Any],
    *,
    split: str,
    row_index: int,
    include_challenge_tests: bool,
) -> dict[str, Any]:
    tests = _as_strings(row.get("test_list"))
    challenge_tests = _as_strings(row.get("challenge_test_list")) if include_challenge_tests else []
    all_tests = tests + challenge_tests
    canonical_code = str(row.get("code") or "").strip()
    task_id = str(row.get("task_id") if row.get("task_id") is not None else row_index)
    setup_code = _setup_code(row)

    ground_truth = json.dumps(
        {
            "task_id": task_id,
            "canonical_code": canonical_code,
            "setup_code": setup_code,
            "tests": all_tests,
        },
        ensure_ascii=False,
    )

    return {
        "data_source": "mbpp",
        "prompt": [{"role": "user", "content": _prompt_text(row)}],
        "ability": "code",
        # OPD uses a fresh student rollout. This is retained only for auditing/evaluation.
        "response": canonical_code,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": row_index,
            "task_id": task_id,
            "split": split,
            "dataset": "mbpp",
            "language": "python",
            "setup_code": setup_code,
            "tests": all_tests,
            "canonical_code": canonical_code,
        },
    }


def convert_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    *,
    include_challenge_tests: bool,
) -> Path:
    source_rows = _read_split(dataset_dir, split)
    rows = [
        convert_row(
            row,
            split=split,
            row_index=index,
            include_challenge_tests=include_challenge_tests,
        )
        for index, row in enumerate(source_rows)
    ]
    rows = [row for row in rows if row["prompt"][0]["content"] and row["response"]]
    if not rows:
        raise ValueError(f"No usable MBPP rows found for split={split!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="snappy")
    print(f"{split}: {len(rows)} rows -> {output_path}")
    return output_path


def parse_splits(values: Iterable[str]) -> tuple[str, ...]:
    splits = tuple(dict.fromkeys(item.strip() for item in values if item.strip()))
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Hugging Face MBPP parquet files to verl OPD format.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing MBPP split parquet shards")
    parser.add_argument("--output-dir", required=True, help="Destination directory for verl-format parquet files")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--include-challenge-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include challenge_test_list in the executable reward payload (default: true)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for split in parse_splits(args.splits):
        convert_split(
            dataset_dir,
            output_dir,
            split,
            include_challenge_tests=args.include_challenge_tests,
        )


if __name__ == "__main__":
    main()
