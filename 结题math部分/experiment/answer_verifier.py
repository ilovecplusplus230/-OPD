from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class VerificationResult:
    correct: bool
    status: str
    method: str
    extracted_answer: str
    normalized_student: str
    normalized_reference: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerVerifier:
    """验证扩充后的最终答案是否仍与数据集参考答案等价。"""

    def __init__(self, numeric_tolerance: float = 1e-8) -> None:
        self.numeric_tolerance = numeric_tolerance

    def extract_final_answer(self, text: str, explicit: str = "") -> tuple[str, str]:
        if str(explicit or "").strip():
            return str(explicit).strip(), "explicit_field"
        raw = str(text or "").strip()
        gsm_matches = re.findall(r"####\s*([^\n]+)", raw)
        if gsm_matches:
            return gsm_matches[-1].strip(), "gsm8k_marker"
        boxes = _extract_balanced_command(raw, "boxed")
        if boxes:
            return boxes[-1].strip(), "boxed_extraction"
        patterns = [
            r"(?is)(?:final\s*answer|answer)\s*[:：]\s*([^\n]+)",
            r"(?is)答案\s*[:：]\s*([^\n]+)",
            r"(?is)(?:therefore|thus)[^\n]*?\bis\s+([^\n.]+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, raw)
            if matches:
                return matches[-1].strip(), "label_extraction"
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return (lines[-1] if lines else ""), "last_line_fallback"

    def verify_text(self, solution: str, reference_answer: str) -> VerificationResult:
        extracted, method = self.extract_final_answer(solution)
        result = self.verify(extracted, reference_answer)
        result.extracted_answer = extracted
        if result.method == "missing_answer":
            result.method = method
        return result

    def verify(self, student_answer: str, reference_answer: str) -> VerificationResult:
        student = _clean_answer(student_answer)
        reference = _clean_answer(reference_answer)
        if not student or not reference:
            return VerificationResult(False, "manual_required", "missing_answer", student_answer, student, reference, "答案为空。")
        if student.casefold() == reference.casefold():
            return VerificationResult(True, "correct", "normalized_exact", student_answer, student, reference, "规范化文本一致。")

        left_assignment = _split_assignment(student)
        right_assignment = _split_assignment(reference)
        if left_assignment:
            student = left_assignment[1]
        if right_assignment:
            reference = right_assignment[1]

        collection = self._compare_collection(student, reference)
        if collection is not None:
            return VerificationResult(collection, "correct" if collection else "incorrect", "collection_equivalence", student_answer, student, reference, "按元素比较集合或元组。")
        symbolic = self._compare_sympy(student, reference)
        if symbolic is not None:
            return VerificationResult(symbolic, "correct" if symbolic else "incorrect", "sympy", student_answer, student, reference, "使用 SymPy 比较数学等价性。")
        numeric = self._compare_numeric(student, reference)
        if numeric is not None:
            return VerificationResult(numeric, "correct" if numeric else "incorrect", "numeric_tolerance", student_answer, student, reference, "使用数值容差比较。")
        return VerificationResult(False, "manual_required", "unparsed", student_answer, student, reference, "无法可靠自动解析，需要人工复核。")

    def _compare_numeric(self, left: str, right: str) -> Optional[bool]:
        try:
            return math.isclose(float(left), float(right), rel_tol=self.numeric_tolerance, abs_tol=self.numeric_tolerance)
        except ValueError:
            return None

    def _compare_sympy(self, left: str, right: str) -> Optional[bool]:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations
            transforms = standard_transformations + (convert_xor, implicit_multiplication_application)
            left_expr = parse_expr(_latex_to_sympy(left), transformations=transforms, evaluate=True)
            right_expr = parse_expr(_latex_to_sympy(right), transformations=transforms, evaluate=True)
            difference = sp.simplify(left_expr - right_expr)
            if difference == 0:
                return True
            if not difference.free_symbols:
                return abs(float(sp.N(difference))) <= self.numeric_tolerance
            return False
        except Exception:
            return None

    def _compare_collection(self, left: str, right: str) -> Optional[bool]:
        left_parts = _collection_parts(left)
        right_parts = _collection_parts(right)
        if left_parts is None or right_parts is None:
            return None
        if len(left_parts) != len(right_parts):
            return False
        results = [self.verify(a, b) for a, b in zip(left_parts, right_parts)]
        if any(item.status == "manual_required" for item in results):
            return None
        return all(item.correct for item in results)


def reference_from_record(record: dict[str, Any]) -> str:
    for key in ("expected_answer", "final_answer", "answer"):
        value = str(record.get(key, "") or "").strip()
        if value:
            if key == "answer" and ("####" in value or len(value.splitlines()) > 2):
                extracted, _ = AnswerVerifier().extract_final_answer(value)
                return extracted
            return value
    solution = str(record.get("solution") or record.get("cot") or "")
    extracted, _ = AnswerVerifier().extract_final_answer(solution)
    return extracted


def _extract_balanced_command(text: str, command: str) -> list[str]:
    marker = "\\" + command
    results: list[str] = []
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return results
        brace = text.find("{", index + len(marker))
        if brace < 0:
            return results
        depth = 0
        for position in range(brace, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[brace + 1 : position])
                    start = position + 1
                    break
        else:
            return results


def _clean_answer(value: str) -> str:
    text = str(value or "").strip()
    boxes = _extract_balanced_command(text, "boxed")
    if boxes:
        text = boxes[-1]
    text = text.replace("−", "-").replace("π", "pi").replace("°", "")
    text = text.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    text = text.strip(" $\n\t.,;：。")
    text = re.sub(r"(?i)^(?:the\s+answer\s+is|answer|final\s+answer|答案)\s*[:：]?\s*", "", text)
    text = re.sub(r"\\text\{\s*(?:units?|degrees?|cm|mm|km|m|dollars?|hours?|minutes?)\s*\}", "", text, flags=re.I)
    text = re.sub(r"(?i)\s+(?:units?|degrees?|cm|mm|km|meters?|dollars?|hours?|minutes?)\.?$", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    if text.endswith("\\%"):
        text = f"({text[:-2]})/100"
    elif text.endswith("%"):
        text = f"({text[:-1]})/100"
    return re.sub(r"\s+", " ", text).strip()


def _split_assignment(text: str) -> Optional[tuple[str, str]]:
    if text.count("=") != 1 or any(op in text for op in ("<=", ">=", "!=")):
        return None
    left, right = [part.strip() for part in text.split("=", 1)]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", left) and right:
        return left, right
    return None


def _collection_parts(text: str) -> Optional[list[str]]:
    stripped = text.strip()
    if len(stripped) < 2 or (stripped[0], stripped[-1]) not in {("(", ")"), ("[", "]"), ("{", "}")}:
        return None
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in stripped[1:-1]:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts if len(parts) >= 2 and all(parts) else None


def _balanced_group(text: str, open_index: int) -> tuple[str, int]:
    depth = 0
    for position in range(open_index, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : position], position
    raise ValueError("Unbalanced braces")


def _replace_frac(text: str) -> str:
    while "\\frac" in text:
        start = text.find("\\frac")
        first_open = text.find("{", start + 5)
        if first_open < 0:
            break
        numerator, first_end = _balanced_group(text, first_open)
        second_open = text.find("{", first_end + 1)
        if second_open < 0:
            break
        denominator, second_end = _balanced_group(text, second_open)
        text = text[:start] + f"(({numerator})/({denominator}))" + text[second_end + 1 :]
    return text


def _latex_to_sympy(text: str) -> str:
    value = text.strip().replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = _replace_frac(value)
    while "\\sqrt" in value:
        start = value.find("\\sqrt")
        open_index = value.find("{", start + 5)
        if open_index < 0:
            break
        inner, end = _balanced_group(value, open_index)
        value = value[:start] + f"sqrt({inner})" + value[end + 1 :]
    value = value.replace("\\cdot", "*").replace("\\times", "*")
    value = value.replace("\\pi", "pi").replace("π", "pi").replace("^", "**")
    value = value.replace("{", "(").replace("}", ")").replace("$", "")
    return re.sub(r"\\[A-Za-z]+", "", value).strip()

