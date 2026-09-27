from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "original_math_package"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from math_reasoning_expander.parser import ReasoningGraphParser

from experiment.answer_verifier import AnswerVerifier
from experiment.llm_client import _parse_steps
from experiment.runner import _strict_acceptance, make_task, select_mask_node


def test_answer_verifier_handles_gsm8k_marker() -> None:
    result = AnswerVerifier().verify_text("We obtain 42.\n#### 42", "42")
    assert result.correct is True


def test_answer_verifier_handles_boxed_fraction() -> None:
    result = AnswerVerifier().verify_text(r"Therefore the answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}")
    assert result.correct is True


def test_mask_never_selects_first_or_final_node() -> None:
    graph = ReasoningGraphParser().parse(
        "Find x.",
        "Let x be unknown.\nWe have x + 2 = 5.\nThus x = 3.\nTherefore the answer is 3.",
    )
    target = select_mask_node(graph, "sample", 7)
    assert target.node_id not in {graph.nodes[0].node_id, graph.nodes[-1].node_id}
    task = make_task(graph, target)
    assert task.target_nodes == [target]


def test_strict_acceptance_requires_answer_and_new_node() -> None:
    accepted, reasons = _strict_acceptance(
        aggregate_score=0.91,
        baseline_score=0.80,
        answer_result={"correct": False},
        added_nodes=1,
        threshold=0.80,
        min_gain=0.01,
    )
    assert accepted is False
    assert any("最终答案" in reason for reason in reasons)

    accepted, reasons = _strict_acceptance(
        aggregate_score=0.91,
        baseline_score=0.80,
        answer_result={"correct": True},
        added_nodes=0,
        threshold=0.80,
        min_gain=0.01,
    )
    assert accepted is False
    assert any("新增推理节点" in reason for reason in reasons)


def test_llm_json_parser_tolerates_latex_backslashes() -> None:
    raw = r'{"steps":["Use \cos(180+45).","Then \frac{1}{2} follows."]}'
    assert _parse_steps(raw) == [r"Use \cos(180+45).", r"Then \frac{1}{2} follows."]
