# MATHOPD 与结题 Math 串联对照

这个目录只负责评测，不重新训练，也不修改 `MATHOPD` 和原有90条 API 实验。

## 两边数据是否相同

- `MATHOPD` 训练：`siyanzhao/Openthoughts_math_30k_opsd`，正式 A/B/C/D 每组固定80条。
- `MATHOPD` 评测：40道官方 MATH test，已和完整 OPSD 训练集做规范化精确去重。
- 原结题 Math：GSM8K、MATH-500、NuminaMath-CoT 候选池各100条，正式结果各30条。

两套测试候选池只有1道题精确重合，而且没有进入原90条正式结果。本目录使用
`MATHOPD` 已审计的40道题作为三模型共同测试集，避免重新挑题造成偏差。

## 三组模型

1. `base`：原始 `Qwen/Qwen3-1.7B`，不加载 LoRA。
2. `opsd_original`：加载 `A_original`，标准 Original OPSD。
3. `opsd_history`：加载 `B_history`，V2 中表现最好的 History-enhanced OPSD。

可选的 `opsd_error_step` 和 `opsd_reflection` 仅作为消融，不作为主结论。

## 安装

建议在有 NVIDIA GPU 的环境运行。Windows RTX 4060 8GB 可逐个加载1.7B模型做小规模验证，
完整实验更适合原 A800 环境。

```powershell
python -m pip install -r opsd_comparison/requirements_opsd.txt
```

## 先跑一题

```powershell
python -m opsd_comparison.run_comparison --limit 1 --task both --pass-k 1 --fresh
```

现有 A/B 两组已经在 A800 上完成40题确定性评测，可以先导入，避免重复计算：

```powershell
python -m opsd_comparison.import_mathopd_results `
  --mathopd-root ..\MATHOPD `
  --output-dir opsd_comparison\outputs
```

然后只补跑未训练模型的直接解题结果：

```powershell
python -m opsd_comparison.run_comparison `
  --variants base --limit 40 --task direct --pass-k 1 --fresh
```

## 正式40题

先做确定性 `pass@1`：

```powershell
python -m opsd_comparison.run_comparison --limit 40 --task both --pass-k 1 --fresh
```

如需 `pass@3`，后两次会使用固定随机种子的采样解答，耗时和生成量约增加三倍：

```powershell
python -m opsd_comparison.run_comparison --limit 40 --task direct --pass-k 3 --fresh
```

输出保存在 `opsd_comparison/outputs/`。LoRA 权重仍保存在相邻的 `MATHOPD/adapters/`，
不会复制到本目录或普通 GitHub 提交中。

三组模型结果都齐全后，生成配对统计和 PNG 图：

```powershell
python -m opsd_comparison.analyze_results --output-dir opsd_comparison\outputs
```

重点查看：

- `outputs/OPSD训练前后对照结论.md`
- `outputs/opsd_model_comparison.png`
- `outputs/paired_model_comparison.csv`
- `outputs/paired_changed_cases.csv`

## 与节点扩充流程的关系

40题直接解题对照和节点扩充不是同一个指标。40题中有25题的参考解答能切出至少3个节点，
适合继续做遮盖恢复；另外15题过短，会在模型运行前记录为不适用，不能强行遮盖。

本机已经用三组模型完成1题串联冒烟测试，结果放在
`outputs/expansion_smoke/`。这次测试证明 LoRA 可以直接接入原 Reasoning Graph、反馈和
严格接收门控，但1题中三组候选都未达到接收条件。因此它只能证明接口跑通，不能证明
OPSD 已经提升节点扩充能力。正式比较应在25道适用题上完整运行后另行汇报。

