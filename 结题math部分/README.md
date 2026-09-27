# 结题 Math 实验

这个文件夹专门用于大创结题阶段的数学推理链扩充实验。原始 `Math.zip` 已完整保存在
`original_math_package/`，实验代码只在外层调用它，没有覆盖队友原来的 parser、evaluator 和数据结构。

当前正式结果包含 90 条真实数据：GSM8K、MATH-500、NuminaMath-CoT 各 30 条。反馈优化严格接收
19 条，单次恢复接收 6 条；门控保留后的平均综合分由原始 `0.794` 提升到 `0.810`。

## 这次验证什么

对同一道题、同一个遮盖节点，比较三种方法：

1. `original`：原始 CoT，不扩充。
2. `one_shot`：LLM 只恢复一次，不接收评价反馈。
3. `feedback`：第一次结果与 one-shot 完全相同；若未通过，最多再反馈优化两次。

完整方法只有同时满足下面四项才会保留：

- 综合质量分不低于 `0.80`；
- 相比原节点质量分提升超过 `0.01`；
- 扩充后最终答案仍与数据集参考答案一致；
- 一个遮盖节点至少恢复为两个有效步骤，确实增加了推理粒度。

这修复了旧版本“只要 gain 大于 0 就接收”的宽松规则，也避免质量分提高但答案被改错。

## 数据来源

- GSM8K：OpenAI 官方仓库的 `test.jsonl`。
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
- MATH-500：Hugging Face 上的 `HuggingFaceH4/MATH-500`。
  https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- NuminaMath-CoT：Hugging Face 上的 `AI-MO/NuminaMath-CoT`，作为长 CoT 补充集。
  https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

下载脚本会保存原始响应、标准化 JSONL 和来源清单，不会把手写样例冒充真实数据。

## 第一次运行

在 VS Code 中打开本文件夹，然后在终端逐行执行：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

打开 `.env`，只填写自己的 API Key。不要把 `.env` 发到 GitHub 或交给别人。

```text
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.gpt.ge/v1
OPENAI_MODEL=gpt-5.6-terra
```

## 下载真实数据

```powershell
python scripts/download_datasets.py --count 100
```

下载后重点看：

- `datasets/processed/gsm8k.jsonl`
- `datasets/processed/math500.jsonl`
- `datasets/processed/numinamath_cot.jsonl`
- `datasets/processed/combined.jsonl`
- `datasets/manifest.json`

## 先做不花钱的流程检查

```powershell
pytest -q
python run_experiment.py --input datasets/processed/combined.jsonl --per-dataset 2 --mock --fresh
```

## 正式调用 API

先小规模检查 5 条，再根据预算扩大：

```powershell
python run_experiment.py --input datasets/processed/combined.jsonl --per-dataset 5 --fresh
python run_experiment.py --input datasets/processed/combined.jsonl --per-dataset 20 --fresh
```

实验每完成一题就追加保存，所以中途断网或关机不会丢掉已经完成的题。不使用 `--fresh` 再运行时，
程序会跳过已完成的样本。三个数据集每个 20 条时最多 60 条，每条最多调用 3 次 API。

只重画图表和重建报告，不重新调用 API：

```powershell
python scripts/rebuild_reports.py
```

## 输出文件

- `outputs/experiment_records.jsonl`：每道题的完整恢复和反馈轨迹。
- `outputs/errors.jsonl`：无法处理的题和错误原因。
- `outputs/per_sample_metrics.csv`：逐题、逐方法指标。
- `outputs/summary.csv`：总体和各数据集汇总。
- `outputs/run_metadata.json`：模型、阈值、随机种子、调用量和耗时。
- `outputs/run_history.json`：正式运行与断点补跑的实际调用量、Token 和耗时。
- `outputs/figures/`：四张 PNG 图，可直接用于报告；同时保留 SVG 便于无损编辑。
- `report/数学扩充实验报告.md`：自动汇总的数学章节草稿。
- `report/典型案例.md`：成功和失败案例。

## 文件结构

- `original_math_package/`：队友提供的原始数学扩充包，只作为基础模块使用。
- `experiment/answer_verifier.py`：提取最终答案并做数值、集合、SymPy 等价判断。
- `experiment/runner.py`：遮盖节点、三组对照、反馈迭代和严格接收规则。
- `experiment/reporting.py`：汇总 CSV、图表和报告。
- `experiment/llm_client.py`：统一调用 OpenAI-compatible API。
- `scripts/download_datasets.py`：下载并标准化三个公开数据集。
- `scripts/audit_datasets.py`：检查缺失字段、节点数量和参考答案可验证性。
- `scripts/rebuild_reports.py`：根据已有轨迹重建图表和报告，不调用 API。
- `tests/`：答案验证和接收规则测试。

## 结题时怎么解释

数学部分不是生成新题，而是对已有正确 CoT 做中间节点扩充。Reasoning Graph 用于明确遮盖位置，LLM
根据前后文恢复更细的中间步骤，五维 evaluator 提供反馈；最终是否保留还要经过质量提升、答案一致性和
节点增量三层硬约束。报告同时给出原始、单次恢复和反馈优化三组结果，不只展示成功案例。
