from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    model_id: str = "Qwen/Qwen3-1.7B"
    adapter_path: Path | None = None


def default_model_specs(mathopd_root: Path) -> dict[str, ModelSpec]:
    adapter_root = mathopd_root / "adapters/v2_long_completion"
    return {
        "base": ModelSpec("base", "Base Student"),
        "opsd_original": ModelSpec(
            "opsd_original", "Original OPSD Student", adapter_path=adapter_root / "A_original"
        ),
        "opsd_history": ModelSpec(
            "opsd_history", "History-enhanced OPSD Student", adapter_path=adapter_root / "B_history"
        ),
        "opsd_error_step": ModelSpec(
            "opsd_error_step", "Error-Step OPSD Student", adapter_path=adapter_root / "C_error_step"
        ),
        "opsd_reflection": ModelSpec(
            "opsd_reflection", "Reflection OPSD Student", adapter_path=adapter_root / "D_reflection"
        ),
    }


class LocalQwenBackend:
    """Load one Qwen base/LoRA arm and expose direct-solving plus FIM generation."""

    def __init__(
        self,
        spec: ModelSpec,
        max_new_tokens: int = 1536,
        expansion_max_new_tokens: int = 256,
    ) -> None:
        self.spec = spec
        self.max_new_tokens = max_new_tokens
        self.expansion_max_new_tokens = expansion_max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("本地模型对照需要 CUDA GPU，当前未检测到可用 CUDA。")
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to("cuda")
        if self.spec.adapter_path is not None:
            if not (self.spec.adapter_path / "adapter_model.safetensors").exists():
                raise FileNotFoundError(f"缺少 adapter: {self.spec.adapter_path}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, self.spec.adapter_path, is_trainable=False)
        else:
            self.model = base_model
        self.model.eval()

    def close(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, messages: list[dict[str, str]], max_new_tokens: int, seed: int, sample: bool) -> str:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未加载。")
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.95})
        else:
            kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **kwargs)
        prompt_length = int(encoded["input_ids"].shape[1])
        completion_ids = generated[0, prompt_length:]
        self.calls += 1
        self.prompt_tokens += prompt_length
        self.completion_tokens += int(completion_ids.shape[0])
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def solve(self, question: str, seed: int, sample: bool = False) -> str:
        prompt = (
            f"Problem: {question}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        return self._generate([{"role": "user", "content": prompt}], self.max_new_tokens, seed, sample)

    def generate(
        self,
        question: str,
        prefix: list[str],
        suffix: list[str],
        node_type: str,
        feedback: str = "",
    ):
        # Import here so this bridge remains inspectable without installing Torch/Transformers.
        from experiment.llm_client import Generation, _parse_steps

        system = (
            "You restore and expand a missing middle segment in a correct mathematical chain of thought. "
            "Preserve the original problem and final answer. Return JSON only."
        )
        user = (
            "Recover the hidden transition using two concise and independently useful reasoning steps.\n"
            "Do not repeat the visible suffix. Do not introduce unsupported assumptions.\n"
            'Return one JSON object with the key "steps" and an array of exactly two strings.\n'
            "Each string must contain the actual mathematical derivation for this problem. "
            "Never output placeholder labels such as 'step 1' or 'step 2'.\n\n"
            f"Question:\n{question}\n\n"
            f"Visible prefix:\n{json.dumps(prefix, ensure_ascii=False)}\n\n"
            f"Hidden node type: {node_type}\n\n"
            f"Visible suffix:\n{json.dumps(suffix, ensure_ascii=False)}\n\n"
            f"Evaluator feedback:\n{feedback or 'None'}"
        )
        raw = self._generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.expansion_max_new_tokens,
            seed=20261002 + self.calls,
            sample=False,
        )
        steps = _parse_steps(raw)
        return Generation(steps=steps, raw=raw)

