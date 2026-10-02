from pathlib import Path

from opsd_comparison.dataset_bridge import audit_overlap, convert_mathopd_eval, question_hash
from opsd_comparison.import_mathopd_results import convert_row


def test_question_hash_ignores_case_and_whitespace() -> None:
    assert question_hash("  What is 2 + 2? ") == question_hash("what  is  2 + 2?")


def test_convert_and_overlap(tmp_path: Path) -> None:
    source = tmp_path / "eval.jsonl"
    source.write_text(
        '{"question":"What is 2 + 2?","reference_solution":"2+2=4","reference_answer":"4",'
        '"original_sample_id":"test/a.json","subject":"Algebra","difficulty":"Level 1"}\n',
        encoding="utf-8",
    )
    heldout = convert_mathopd_eval(source)
    assert heldout[0]["solution"] == "2+2=4"
    audit = audit_overlap(
        heldout,
        [{"id": "pool", "dataset": "MATH-500", "question": "what  is  2 + 2?"}],
        [],
    )
    assert audit["candidate_pool_exact_normalized_overlap_count"] == 1
    assert audit["formal_90_exact_normalized_overlap_count"] == 0


def test_convert_existing_mathopd_result() -> None:
    converted = convert_row(
        {
            "arm": "B",
            "method": "B_history",
            "eval_index": 2,
            "original_sample_id": "test/algebra/176.json",
            "reference_answer": "x^3 + 2x^2 + x",
            "student_answer": "\\boxed{x^3 + 2x^2 + x}",
            "is_correct": True,
            "verification_status": "correct",
            "extracted_answer": "x^3 + 2x^2 + x",
            "generated_tokens": 42,
            "hit_token_cap": False,
            "terminated_by_eos": True,
        }
    )
    assert converted["id"] == "mathopd_heldout_002"
    assert converted["model_variant"] == "opsd_history"
    assert converted["pass_at_1"] is True

