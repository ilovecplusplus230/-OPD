# 特征生成研究结果

以下为已完成运行；所有数值来自各运行的独立测试报告。
同一行内 baseline 与 optimized 的数据、模型配置相同。跨数据集的绝对分数不能当作统一难度标尺。
主指标为 Macro F1；Log Loss 越低越好，其余越高越好。分数显示四位小数，增益用百分点。

| 数据集/视图 | 模型 | 生成器 | 种子 | 训练行数 | 接受数 | 测试 Macro F1：基线 → 优化 | Δ百分点 | 实际 LLM 调用 |
|---|---|---|---:|---:|---:|---|---:|---:|
| jungle_chess/full | reference | reasoned | 42 | 8963 | 4 | 0.5603 → 0.7428 | +18.2492 | 0 |
| balance_scale/full | reference | reasoned | 42 | 125 | 4 | 0.5824 → 0.8047 | +22.2362 | 0 |
| chess_krk/full | reference | reasoned | 42 | 5611 | 4 | 0.3705 → 0.4352 | +6.4689 | 0 |

## jungle_chess/full/reference/reasoned/seed_42

[完整日志](research/runs/jungle_chess/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](research/runs/jungle_chess/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.7716 | 0.8264 | +0.0548 |
| balanced_accuracy | 0.5753 | 0.7105 | +0.1352 |
| f1 | 0.7374 | 0.8182 | +0.0808 |
| f1_macro | 0.5603 | 0.7428 | +0.1825 |
| auc | 0.9132 | 0.9513 | +0.0380 |
| log_loss | 0.5498 | 0.4565 | +0.0934 |

接受特征：oct_composite_19_r1, oct_simple_difference_17_r2, oct_signed_log_33_r3, oct_chess_manhattan_r4
实际训练预算：8963/8963。
训练流程版本：research_guided_v5；生成器版本：cart_residual_v5。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| b | 3484 / 38.87% | 6969 | 0.7669 → 0.8210 | 0.8047 → 0.8295 | 0.7853 → 0.8253 |
| d | 867 / 9.67% | 1734 | 0.6364 → 0.8311 | 0.0363 → 0.3973 | 0.0687 → 0.5377 |
| w | 4612 / 51.46% | 9225 | 0.7762 → 0.8297 | 0.8848 → 0.9046 | 0.8269 → 0.8655 |

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_composite_19_r1 | 156 | 0.1186 ± 0.0032 | +0.0405 | 通过 |
| oct_simple_difference_17_r2 | 138 | 0.1296 ± 0.0031 | +0.0361 | 通过 |
| oct_signed_log_33_r3 | 45 | 0.0209 ± 0.0014 | +0.0191 | 通过 |
| oct_chess_manhattan_r4 | 141 | 0.0216 ± 0.0016 | +0.0075 | 通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## balance_scale/full/reference/reasoned/seed_42

[完整日志](balance_scale/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](balance_scale/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.8280 | 0.9160 | +0.0880 |
| balanced_accuracy | 0.6000 | 0.7739 | +0.1739 |
| f1 | 0.8037 | 0.9078 | +0.1042 |
| f1_macro | 0.5824 | 0.8047 | +0.2224 |
| auc | 0.8412 | 0.9105 | +0.0694 |
| log_loss | 0.5076 | 0.3471 | +0.1606 |

接受特征：oct_simple_difference_5_r1, oct_composite_sum_0_r2, oct_signed_log_4_r3, oct_composite_sum_13_r4
实际训练预算：125/125。
训练流程版本：multiclass_reasoned_v7；生成器版本：cart_residual_v6。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| B | 10 / 8.00% | 20 | 0.0000 → 0.8000 | 0.0000 → 0.4000 | 0.0000 → 0.5333 |
| L | 57 / 45.60% | 115 | 0.8268 → 0.9106 | 0.9130 → 0.9739 | 0.8678 → 0.9412 |
| R | 58 / 46.40% | 115 | 0.8718 → 0.9316 | 0.8870 → 0.9478 | 0.8793 → 0.9397 |

[论文依据](https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf#page=9)：ICML 2023 OpenFE 自动特征生成的直接实验数据集。
Small mechanistic benchmark (625 rows); minority training count is about 10. Not evidence of large-scale real-world generalization.

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_simple_difference_5_r1 | 98 | 0.4283 ± 0.0124 | +0.2224 | 通过 |
| oct_composite_sum_0_r2 | 197 | 0.4132 ± 0.0100 | +0.1635 | 通过 |
| oct_signed_log_4_r3 | 134 | 0.1065 ± 0.0300 | +0.0965 | 通过 |
| oct_composite_sum_13_r4 | 95 | 0.0230 ± 0.0241 | -0.0143 | 未通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## chess_krk/full/reference/reasoned/seed_42

[完整日志](chess_krk/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](chess_krk/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.4276 | 0.4571 | +0.0295 |
| balanced_accuracy | 0.3579 | 0.4106 | +0.0527 |
| f1 | 0.4097 | 0.4418 | +0.0321 |
| f1_macro | 0.3705 | 0.4352 | +0.0647 |
| auc | 0.9176 | 0.9232 | +0.0056 |
| log_loss | 1.6419 | 1.5812 | +0.0607 |

接受特征：oct_composite_sum_11_r1, oct_composite_sum_28_r2, oct_simple_product_17_r3, oct_piecewise_32_median_r4
实际训练预算：5611/5611。
训练流程版本：multiclass_reasoned_v7；生成器版本：cart_residual_v6。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| 1 | 559 / 9.96% | 1119 | 0.5069 → 0.4339 | 0.4924 → 0.5693 | 0.4995 → 0.4925 |
| 10 | 5 / 0.09% | 11 | 0.6667 → 1.0000 | 0.1818 → 0.2727 | 0.2857 → 0.4286 |
| 11 | 287 / 5.11% | 573 | 0.4047 → 0.4378 | 0.5410 → 0.5096 | 0.4630 → 0.4710 |
| 12 | 571 / 10.18% | 1142 | 0.4348 → 0.5000 | 0.2513 → 0.3170 | 0.3185 → 0.3880 |
| 13 | 433 / 7.72% | 867 | 0.4661 → 0.5111 | 0.4198 → 0.5029 | 0.4417 → 0.5070 |
| 14 | 94 / 1.68% | 188 | 0.5023 → 0.5819 | 0.5745 → 0.5479 | 0.5360 → 0.5644 |
| 15 | 40 / 0.71% | 79 | 0.6250 → 0.8000 | 0.3165 → 0.6076 | 0.4202 → 0.6906 |
| 16 | 911 / 16.24% | 1821 | 0.4195 → 0.4510 | 0.7243 → 0.7397 | 0.5313 → 0.5603 |
| 17 | 342 / 6.10% | 685 | 0.4040 → 0.4748 | 0.2920 → 0.3168 | 0.3390 → 0.3800 |
| 18 | 16 / 0.29% | 31 | 0.0000 → 0.5833 | 0.0000 → 0.2258 | 0.0000 → 0.3256 |
| 2 | 137 / 2.44% | 273 | 0.3000 → 0.4805 | 0.0440 → 0.1355 | 0.0767 → 0.2114 |
| 3 | 118 / 2.10% | 237 | 0.5587 → 0.4852 | 0.5021 → 0.4852 | 0.5289 → 0.4852 |
| 4 | 78 / 1.39% | 156 | 0.8310 → 0.6698 | 0.3782 → 0.4551 | 0.5198 → 0.5420 |
| 5 | 397 / 7.08% | 794 | 0.3640 → 0.3827 | 0.2040 → 0.1725 | 0.2615 → 0.2378 |
| 6 | 839 / 14.95% | 1678 | 0.3761 → 0.4357 | 0.4678 → 0.4440 | 0.4170 → 0.4398 |
| 7 | 16 / 0.29% | 32 | 0.5000 → 0.3333 | 0.0625 → 0.0938 | 0.1111 → 0.1463 |
| 8 | 719 / 12.81% | 1439 | 0.4128 → 0.4232 | 0.2960 → 0.3523 | 0.3448 → 0.3845 |
| 9 | 49 / 0.87% | 98 | 0.4892 → 0.5250 | 0.6939 → 0.6429 | 0.5738 → 0.5780 |

[论文依据](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf#page=5)：JMLR 分类基准；另有 FLAIRS 2012 的距离特征研究，不能称其为 AAAI 主会。
IID endgame positions; 18 nominal labels are retained, not regression on encoded class IDs. Rare classes have very few test cases; not independent-game/trajectory generalization.

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_composite_sum_11_r1 | 772 | 0.1649 ± 0.0040 | +0.0399 | 通过 |
| oct_composite_sum_28_r2 | 823 | 0.1257 ± 0.0092 | +0.0078 | 通过 |
| oct_simple_product_17_r3 | 409 | 0.0482 ± 0.0096 | +0.0180 | 通过 |
| oct_piecewise_32_median_r4 | 612 | 0.0729 ± 0.0044 | +0.0187 | 通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## 解释边界

CART 和残差预筛只使用训练集；正式接受仍要求验证主指标改善且其他指标不退化。测试退化照实保留。
这是固定外层划分的实验。多个 seed 改变模型/CART/预算采样，不代表多个独立数据划分；同一测试集反复评估也不构成独立重复。
局部模板与 CART 生成结果不等于真实 LLM 成绩。训练残差预筛不是 OpenFE FeatureBoost 的原样复现。
当前生成器包含候选池、CART 路径与预筛策略；不能将收益单独归因于任一组件。CART 是从训练标签学习的监督式特征构造。
当前是固定划分的三个已筛选任务，不能推广为所有表格任务的有效性证据；棋盘任务的随机划分不证明对未见棋子组合的泛化。
