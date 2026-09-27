from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
PROCESSED_DIR = ROOT / "datasets" / "processed"

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
NUMINA_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "HIT-undergraduate-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    return random.Random(seed).sample(rows, count)


def fetch_gsm8k(count: int, seed: int) -> list[dict[str, Any]]:
    text = download_text(GSM8K_URL)
    (RAW_DIR / "gsm8k_test.jsonl").write_text(text, encoding="utf-8")
    rows = deterministic_sample(parse_json_lines(text), count, seed)
    return [
        {
            "id": f"gsm8k_test_{index}",
            "dataset": "GSM8K",
            "source_url": GSM8K_URL,
            "question": row["question"],
            "solution": row["answer"],
            "answer": _gsm_answer(row["answer"]),
            "difficulty": "grade_school",
            "subject": "arithmetic_word_problem",
        }
        for index, row in enumerate(rows)
    ]


def fetch_math500(count: int, seed: int) -> list[dict[str, Any]]:
    text = download_text(MATH500_URL)
    (RAW_DIR / "math500_test.jsonl").write_text(text, encoding="utf-8")
    rows = deterministic_sample(parse_json_lines(text), count, seed)
    return [
        {
            "id": f"math500_{row.get('unique_id', index)}",
            "dataset": "MATH-500",
            "source_url": MATH500_URL,
            "question": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "answer": row.get("answer", ""),
            "difficulty": str(row.get("level", "unknown")),
            "subject": str(row.get("subject", "unknown")),
        }
        for index, row in enumerate(rows)
    ]


def fetch_numina(count: int, seed: int) -> list[dict[str, Any]]:
    # 分散读取，避免只取数据集开头同一来源的一组题。
    offsets = [0, 100_000, 250_000, 450_000, 700_000]
    per_request = max(10, (count + len(offsets) - 1) // len(offsets))
    collected: list[dict[str, Any]] = []
    for offset in offsets:
        query = urllib.parse.urlencode(
            {
                "dataset": "AI-MO/NuminaMath-CoT",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": per_request,
            }
        )
        try:
            payload = json.loads(download_text(f"{NUMINA_ROWS_URL}?{query}"))
        except Exception:
            continue
        for item in payload.get("rows", []):
            row = item.get("row", {})
            if row.get("problem") and row.get("solution"):
                collected.append(row)
    if not collected:
        raise RuntimeError("NuminaMath-CoT 下载失败，未取得任何样本。")
    rows = deterministic_sample(collected, min(count, len(collected)), seed)
    source_url = "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT"
    return [
        {
            "id": f"numina_{index}",
            "dataset": "NuminaMath-CoT",
            "source_url": source_url,
            "question": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "answer": _boxed_answer(row.get("solution", "")),
            "difficulty": "mixed",
            "subject": str(row.get("source", "unknown")),
        }
        for index, row in enumerate(rows)
    ]


def _gsm_answer(text: str) -> str:
    return text.rsplit("####", 1)[-1].strip() if "####" in text else ""


def _boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    start += len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并统一结题 Math 数据集格式。")
    parser.add_argument("--count", type=int, default=100, help="每个数据集保留的样本数。")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "gsm8k": fetch_gsm8k(args.count, args.seed),
        "math500": fetch_math500(args.count, args.seed),
        "numinamath_cot": fetch_numina(args.count, args.seed),
    }
    combined: list[dict[str, Any]] = []
    for name, rows in datasets.items():
        write_jsonl(PROCESSED_DIR / f"{name}.jsonl", rows)
        combined.extend(rows)
        print(f"{name}: {len(rows)}")
    write_jsonl(PROCESSED_DIR / "combined.jsonl", combined)

    manifest = {
        "seed": args.seed,
        "requested_per_dataset": args.count,
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "sources": {
            "GSM8K": GSM8K_URL,
            "MATH-500": MATH500_URL,
            "NuminaMath-CoT": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
        },
    }
    (ROOT / "datasets" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

