from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "original_math_package"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from math_reasoning_expander.evaluators import MultiFeedbackEvaluator
from math_reasoning_expander.models import FillCandidate, MaskedTask, ReasoningGraph, ReasoningNode
from math_reasoning_expander.parser import ReasoningGraphParser

from .answer_verifier import AnswerVerifier, reference_from_record
from .io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from .llm_client import Generation, MathFillClient
from .reporting import save_all


PRIORITY_TYPES = {
    "equation_transform": 0,
    "inference": 1,
    "explanation": 2,
    "definition": 9,
    "conclusion": 10,
}


def _stable_index(text: str, size: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def select_mask_node(graph: ReasoningGraph, sample_id: str, seed: int) -> ReasoningNode:
    """Select one useful middle node while avoiding definitions and final answers."""
    if len(graph.nodes) < 3:
        raise ValueError("推理步骤少于 3 个，无法安全遮盖中间节点。")
    middle = graph.nodes[1:-1]
    eligible = [node for node in middle if node.node_type not in {"definition", "conclusion"}]
    if not eligible:
        eligible = middle
    best_priority = min(PRIORITY_TYPES.get(node.node_type, 5) for node in eligible)
    preferred = [node for node in eligible if PRIORITY_TYPES.get(node.node_type, 5) == best_priority]
    return preferred[_stable_index(sample_id, len(preferred), seed)]


def make_task(graph: ReasoningGraph, target: ReasoningNode, context_window: int = 4) -> MaskedTask:
    index = graph.nodes.index(target)
    return MaskedTask(
        graph=graph,
        masked_node_ids=[target.node_id],
        mask_strategy="typed_middle_node",
        prefix_nodes=graph.nodes[max(0, index - context_window) : index],
        suffix_nodes=graph.nodes[index + 1 : index + 1 + context_window],
        target_nodes=[target],
    )


def _candidate(generation: Generation) -> FillCandidate:
    steps = [step.strip() for step in generation.steps if step.strip()]
    return FillCandidate(
        text="\n".join(steps),
        steps=steps,
        raw_response=generation.raw,
        metadata={
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
        },
    )


def _scores(evaluation: Any) -> dict[str, float]:
    return {score.name: round(float(score.score), 6) for score in evaluation.scores}


def merge_solution(graph: ReasoningGraph, target: ReasoningNode, steps: list[str]) -> str:
    output: list[str] = []
    for node in graph.nodes:
        if node.node_id == target.node_id:
            output.extend(steps)
        else:
            output.append(node.content)
    return "\n".join(output)


def _strict_acceptance(
    aggregate_score: float,
    baseline_score: float,
    answer_result: dict[str, Any],
    added_nodes: int,
    threshold: float,
    min_gain: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if aggregate_score < threshold:
        reasons.append(f"综合分 {aggregate_score:.3f} 低于阈值 {threshold:.3f}")
    if aggregate_score - baseline_score <= min_gain:
        reasons.append(f"质量提升 {aggregate_score - baseline_score:+.3f} 未超过 {min_gain:.3f}")
    if not answer_result.get("correct"):
        reasons.append("最终答案未通过自动一致性验证")
    if added_nodes <= 0:
        reasons.append("没有形成新增推理节点")
    return not reasons, reasons


def _method_result(
    graph: ReasoningGraph,
    task: MaskedTask,
    target: ReasoningNode,
    candidate: FillCandidate,
    baseline_score: float,
    verifier: AnswerVerifier,
    reference_answer: str,
    threshold: float,
    min_gain: float,
    attempts: int,
    feedback_history: list[str],
) -> dict[str, Any]:
    evaluator = MultiFeedbackEvaluator(accept_threshold=threshold)
    evaluation = evaluator.evaluate(task, candidate)
    expanded = merge_solution(graph, target, candidate.steps)
    verification = verifier.verify_text(expanded, reference_answer).to_dict()
    added_nodes = max(0, len(candidate.steps) - len(task.target_nodes))
    accepted, rejection_reasons = _strict_acceptance(
        evaluation.aggregate_score,
        baseline_score,
        verification,
        added_nodes,
        threshold,
        min_gain,
    )
    return {
        "accepted": accepted,
        "aggregate_score": round(float(evaluation.aggregate_score), 6),
        "quality_gain": round(float(evaluation.aggregate_score - baseline_score), 6),
        "scores": _scores(evaluation),
        "answer_verification": verification,
        "added_node_count": added_nodes,
        "attempts": attempts,
        "steps": candidate.steps,
        "expanded_solution": expanded,
        "evaluator_feedback": evaluation.feedback,
        "feedback_history": feedback_history,
        "rejection_reasons": rejection_reasons,
        "raw_response": candidate.raw_response,
        "token_usage": candidate.metadata,
    }


def run_record(
    record: dict[str, Any],
    client: MathFillClient,
    seed: int,
    max_attempts: int,
    threshold: float,
    min_gain: float,
) -> dict[str, Any]:
    parser = ReasoningGraphParser()
    verifier = AnswerVerifier()
    question = str(record.get("question", "")).strip()
    solution = str(record.get("solution", "")).strip()
    reference_answer = reference_from_record(record)
    if not question or not solution:
        raise ValueError("记录缺少 question 或 solution。")
    if not reference_answer:
        raise ValueError("无法从记录中获得参考答案。")

    graph = parser.parse(question, solution)
    target = select_mask_node(graph, str(record["id"]), seed)
    task = make_task(graph, target)

    original_candidate = FillCandidate(text=target.content, steps=[target.content], raw_response="")
    evaluator = MultiFeedbackEvaluator(accept_threshold=threshold)
    original_eval = evaluator.evaluate(task, original_candidate)
    original_verification = verifier.verify_text(solution, reference_answer).to_dict()
    original = {
        "accepted": bool(original_verification.get("correct")),
        "aggregate_score": round(float(original_eval.aggregate_score), 6),
        "quality_gain": 0.0,
        "scores": _scores(original_eval),
        "answer_verification": original_verification,
        "added_node_count": 0,
        "attempts": 0,
        "steps": [target.content],
        "expanded_solution": solution,
        "evaluator_feedback": original_eval.feedback,
        "feedback_history": [],
        "rejection_reasons": [],
        "raw_response": "",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    first_generation = client.generate(
        question,
        [node.content for node in task.prefix_nodes],
        [node.content for node in task.suffix_nodes],
        target.node_type,
    )
    first_candidate = _candidate(first_generation)
    one_shot = _method_result(
        graph,
        task,
        target,
        first_candidate,
        original_eval.aggregate_score,
        verifier,
        reference_answer,
        threshold,
        min_gain,
        attempts=1,
        feedback_history=[],
    )

    candidates = [one_shot]
    feedback_history: list[str] = []
    current = one_shot
    for attempt in range(2, max_attempts + 1):
        if current["accepted"]:
            break
        feedback = current["evaluator_feedback"]
        if current["rejection_reasons"]:
            feedback += " Strict checks: " + "; ".join(current["rejection_reasons"])
        feedback_history.append(feedback)
        generation = client.generate(
            question,
            [node.content for node in task.prefix_nodes],
            [node.content for node in task.suffix_nodes],
            target.node_type,
            feedback=feedback,
        )
        current = _method_result(
            graph,
            task,
            target,
            _candidate(generation),
            original_eval.aggregate_score,
            verifier,
            reference_answer,
            threshold,
            min_gain,
            attempts=attempt,
            feedback_history=list(feedback_history),
        )
        candidates.append(current)

    accepted_candidates = [item for item in candidates if item["accepted"]]
    feedback_result = max(
        accepted_candidates or candidates,
        key=lambda item: (bool(item["accepted"]), item["aggregate_score"]),
    )
    feedback_result = dict(feedback_result)
    feedback_result["attempt_trace"] = candidates

    return {
        "id": record["id"],
        "dataset": record.get("dataset", "unknown"),
        "source_url": record.get("source_url", ""),
        "subject": record.get("subject", ""),
        "difficulty": record.get("difficulty", ""),
        "question": question,
        "reference_answer": reference_answer,
        "original_solution": solution,
        "original_node_count": len(graph.nodes),
        "masked_node_id": target.node_id,
        "masked_node_type": target.node_type,
        "masked_original": target.content,
        "reasoning_graph": graph.to_dict(),
        "methods": {
            "original": original,
            "one_shot": one_shot,
            "feedback": feedback_result,
        },
    }


def stratified_sample(records: list[dict[str, Any]], per_dataset: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    parser = ReasoningGraphParser()
    verifier = AnswerVerifier()
    for record in records:
        question = str(record.get("question", "")).strip()
        solution = str(record.get("solution", "")).strip()
        reference = reference_from_record(record)
        if not question or not solution or not reference:
            continue
        if len(parser.parse(question, solution).nodes) < 3:
            continue
        if not verifier.verify_text(solution, reference).correct:
            continue
        grouped.setdefault(str(record.get("dataset", "unknown")), []).append(record)
    selected: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        items = list(grouped[dataset])
        random.Random(f"{seed}:{dataset}").shuffle(items)
        selected.extend(items[:per_dataset])
    return selected


def run_experiment(args: argparse.Namespace) -> None:
    root = ROOT
    input_path = Path(args.input).resolve()
    output_path = root / "outputs" / "experiment_records.jsonl"
    error_path = root / "outputs" / "errors.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        output_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    source_records = read_jsonl(input_path)
    selected = stratified_sample(source_records, args.per_dataset, args.seed)
    existing = read_jsonl(output_path) if output_path.exists() else []
    completed = {str(item.get("id")) for item in existing}
    client = MathFillClient(root, mock=args.mock)
    started = time.time()

    print(f"Selected {len(selected)} samples from {input_path.name}.")
    for index, record in enumerate(selected, start=1):
        sample_id = str(record.get("id", f"sample-{index}"))
        if sample_id in completed:
            print(f"[{index}/{len(selected)}] skip {sample_id} (already completed)")
            continue
        try:
            result = run_record(
                record,
                client,
                seed=args.seed,
                max_attempts=args.max_attempts,
                threshold=args.accept_threshold,
                min_gain=args.min_gain,
            )
            append_jsonl(output_path, result)
            existing.append(result)
            print(
                f"[{index}/{len(selected)}] {sample_id}: "
                f"accepted={result['methods']['feedback']['accepted']} "
                f"score={result['methods']['feedback']['aggregate_score']:.3f}"
            )
        except Exception as exc:
            append_jsonl(
                error_path,
                {"id": sample_id, "dataset": record.get("dataset"), "error": f"{type(exc).__name__}: {exc}"},
            )
            print(f"[{index}/{len(selected)}] {sample_id}: ERROR {type(exc).__name__}: {exc}")

    selected_ids = {str(item.get("id")) for item in selected}
    final_records = [item for item in read_jsonl(output_path) if str(item.get("id")) in selected_ids]
    completed_ids = {str(item.get("id")) for item in final_records}
    if error_path.exists():
        unresolved = [item for item in read_jsonl(error_path) if str(item.get("id")) not in completed_ids]
        if unresolved:
            write_jsonl(error_path, unresolved)
        else:
            error_path.unlink()

    traces = [
        attempt
        for record in final_records
        for attempt in record.get("methods", {}).get("feedback", {}).get("attempt_trace", [])
    ]
    total_api_calls = 0 if args.mock else len(traces)
    total_prompt_tokens = sum(int(item.get("token_usage", {}).get("prompt_tokens", 0) or 0) for item in traces)
    total_completion_tokens = sum(int(item.get("token_usage", {}).get("completion_tokens", 0) or 0) for item in traces)
    run_meta = {
        "sample_count": len(final_records),
        "requested_count": len(selected),
        "model": "mock" if args.mock else client.model,
        "provider": "mock" if args.mock else client.base_url,
        "seed": args.seed,
        "accept_threshold": args.accept_threshold,
        "min_gain": args.min_gain,
        "max_attempts": args.max_attempts,
        "api_calls_this_process": client.calls,
        "prompt_tokens_this_process": client.prompt_tokens,
        "completion_tokens_this_process": client.completion_tokens,
        "api_responses_in_saved_records": total_api_calls,
        "prompt_tokens_in_saved_records": total_prompt_tokens,
        "completion_tokens_in_saved_records": total_completion_tokens,
        "elapsed_seconds_this_process": round(time.time() - started, 3),
        "input_file": str(input_path),
    }
    write_json(root / "outputs" / "run_metadata.json", run_meta)
    if final_records:
        save_all(root, final_records, run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数学 CoT 节点扩充对照实验")
    parser.add_argument("--input", required=True, help="标准化后的 JSONL 数据")
    parser.add_argument("--per-dataset", type=int, default=20, help="每个数据集抽取条数")
    parser.add_argument("--max-attempts", type=int, default=3, help="反馈方法最多尝试次数")
    parser.add_argument("--accept-threshold", type=float, default=0.80)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--mock", action="store_true", help="不调用 API，仅检查流程")
    parser.add_argument("--fresh", action="store_true", help="清空当前实验结果后重跑")
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
