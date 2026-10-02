from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Generation:
    steps: list[str]
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_local_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class MathFillClient:
    def __init__(self, root: Path, mock: bool = False) -> None:
        load_local_env(root)
        self.mock = mock
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.gpt.ge/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if not mock and not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY。请在结题math/.env 中配置，不要写进源码。")

    def generate(
        self,
        question: str,
        prefix: list[str],
        suffix: list[str],
        node_type: str,
        feedback: str = "",
    ) -> Generation:
        if self.mock:
            bridge = "Therefore, combine the previous result with the stated condition."
            return Generation([bridge, "This gives the intermediate result required by the next step."], bridge)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120)
        system = (
            "You restore and expand a missing middle segment in a correct mathematical chain of thought. "
            "Preserve the original problem and final answer. Return JSON only."
        )
        user = (
            "Recover the hidden transition using two concise and independently useful reasoning steps.\n"
            "Do not repeat the visible suffix. Do not introduce unsupported assumptions.\n"
            "Return exactly: {\"steps\": [\"step 1\", \"step 2\"]}.\n\n"
            f"Question:\n{question}\n\n"
            f"Visible prefix:\n{json.dumps(prefix, ensure_ascii=False)}\n\n"
            f"Hidden node type: {node_type}\n\n"
            f"Visible suffix:\n{json.dumps(suffix, ensure_ascii=False)}\n\n"
            f"Evaluator feedback:\n{feedback or 'None'}"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.2,
                )
                self.calls += 1
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                raw = response.choices[0].message.content or ""
                return Generation(_parse_steps(raw), raw, prompt_tokens, completion_tokens)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用连续失败: {type(last_error).__name__}: {last_error}")


def _parse_steps(raw: str) -> list[str]:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data: Any = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            steps = [str(item).strip() for item in data["steps"] if str(item).strip()]
            if steps:
                return steps
    except json.JSONDecodeError:
        # LLMs often place LaTeX such as \cos or \frac directly inside JSON.
        # Escape those backslashes once and retry instead of treating the whole
        # JSON object as one reasoning step.
        # Escape isolated LaTeX commands before JSON treats sequences such as
        # ``\\frac`` as control characters. Keep existing ``\\\\sqrt`` intact.
        repaired = re.sub(r'(?<!\\)\\(?=[A-Za-z]{2,})', r'\\\\', text)
        repaired = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', repaired)
        try:
            data = json.loads(repaired)
            if isinstance(data, dict) and isinstance(data.get("steps"), list):
                steps = [str(item).strip() for item in data["steps"] if str(item).strip()]
                if steps:
                    return steps
        except json.JSONDecodeError:
            pass
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines or [text]

