# 当前三个分类数据集

按用户要求仅保留以下三个数据集；其他任务的 CSV、原始文件、筛选文件与训练结果均已清理。此次没有修改保留任务的原始划分或已有训练结果。

| 目录 | 来源 | 行数 | 特征数 | 类别数 | train / validation / test |
|---|---|---:|---:|---:|---|
| `jungle_chess/` | [OpenML 41027](https://www.openml.org/d/41027)，RAW 六字段 | 44,819 | 6 | 3 | 8,963 / 17,928 / 17,928 |
| `balance_scale/` | [OpenML 11](https://www.openml.org/d/11) / [UCI](https://archive.ics.uci.edu/dataset/12/balance+scale) | 625 | 4 | 3 | 125 / 250 / 250 |
| `chess_krk/` | [OpenML 1481](https://www.openml.org/d/1481) / [UCI](https://archive.ics.uci.edu/dataset/23/chess+king+rook+vs+king) | 28,056 | 6 | 18 | 5,611 / 11,222 / 11,223 |

Jungle Chess 的位置及棋子强度、Balance Scale 的重量及力臂、Chess KRK 的三个棋子坐标均适合考察多列组合关系。Jungle 和 Balance 有 [OpenFE（ICML 2023）Table 7](https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf#page=9) 的直接实验依据；KRK 有 [JMLR 分类基准](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf#page=5) 依据，以及 [FLAIRS 2012 距离特征研究](https://cdn.aaai.org/ocs/4430/4430-21453-1-PB.pdf)。FLAIRS 不是 AAAI 主会。

每个数据集目录包含 `train.csv`、`validation.csv`、`test.csv`、`manifest.json`、`class_distribution.json/csv`。`_raw/` 仅保留这三个任务的官方原始数据、元信息及 KRK 坐标核对材料。KRK 的全部坐标和类别编码曾与 UCI 28,056 行逐行核对，见 [核对记录](_raw/chess_krk_coordinate_audit.json)。保留 18 个原始类别，不将编号当作回归目标。

`split_policy.json` 固定 seed=42、20%/40%/40% 分层划分，保留原始类别比例，不做类别重采样。`catalog.json`、`validation.json` 仅索引这三个任务。三个任务无完全重复行需要删除；训练时继续核对 CSV SHA-256、行数和各类计数。

在项目根目录可用 `python -m tabular_data.prepare_datasets --offline` 离线重建三个任务；只准备一个任务可加 `--dataset chess_krk`。GPU 完成状态以各运行目录中的 `run_status.json` 为准，不能把数据加载检查等同于训练通过。

训练指令和输出文件说明见 [项目说明](../README.md)，三个任务的横向对比见 [latest_summary.md](../outputs/latest_summary.md)。清理没有重跑训练；历史报告继续保留真实的生成器版本及运行标识。

这些数据是用户依据已观察结果选择保留的研究任务，不能把三者当作未筛选的总体基准。当前固定随机划分也不证明跨棋子组合或空间隔离的泛化能力。数据来源及署名继续保存在各 manifest 和原始元信息中。
