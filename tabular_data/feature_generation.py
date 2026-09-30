"""表格特征生成提示词及离线回退规则。"""

import json
from typing import Any, Dict, List

from llm_client import LLMError, strip_code_fence


class OfflineFeatureClient:
    """模拟测试使用本地特征规则，不请求外部模型。"""

    def generate_feature_code(self, columns: List[str], stats: Dict[str, Any], history: List[str]) -> str:
        return mock_feature_code(columns, history)


def generate_feature_code(client, columns: List[str], stats: Dict[str, Any], history: List[str]) -> str:
    """通过共享 API 客户端生成 pandas 特征代码。"""
    system_prompt = "You generate short pandas feature-engineering code. Return only Python code."
    user_prompt = (
        "Given a pandas DataFrame named df, create one useful numeric feature column.\n"
        "Allowed objects: df, np, pd, math. Do not import modules. Do not read files.\n"
        f"Columns: {columns}\n"
        f"Stats: {json.dumps(stats, ensure_ascii=False)[:3000]}\n"
        f"Previous feedback: {history[-5:]}\n"
        "Return code like: df['new_feature'] = ..."
    )
    try:
        raw = client.chat(user_prompt, system_prompt=system_prompt, temperature=0.3)
        return strip_code_fence(raw)
    except LLMError:
        if not client.use_mock_when_fails:
            raise
        return mock_feature_code(columns, history)


def mock_feature_code(columns: List[str], history: List[str] | None = None) -> str:
    """本地 mock 特征生成，按轮次生成比例、差值、交互项。"""
    usable = [col for col in columns if col != "target"]
    history = history or []
    idx = max(0, len(history) - 1)
    if len(usable) >= 2:
        a = usable[idx % len(usable)]
        b = usable[(idx + 1) % len(usable)]
        if idx % 3 == 0:
            return (
                f"denom = df[{b!r}].replace(0, np.nan)\n"
                f"df['llm_ratio_{a}_to_{b}_{idx}'] = (df[{a!r}] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)"
            )
        if idx % 3 == 1:
            return f"df['llm_diff_{a}_minus_{b}_{idx}'] = df[{a!r}] - df[{b!r}]"
        return f"df['llm_interaction_{a}_{b}_{idx}'] = df[{a!r}] * df[{b!r}]"
    if usable:
        a = usable[0]
        return f"df['llm_squared_{a}'] = df[{a!r}] * df[{a!r}]"
    return "df['llm_constant_guard'] = 0"
