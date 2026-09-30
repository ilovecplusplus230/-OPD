# 表格数据模块

本目录集中管理表格数据预处理、LLM 特征生成、GPU XGBoost 训练评估和表格数据集。

```text
tabular_data/
├── __init__.py
├── tabular_octree.py       # 数据清洗、特征检查、交叉验证和迭代主流程
├── feature_generation.py   # 表格特征生成提示词及离线回退规则
├── web.js                  # 模拟测试、上传、分类/回归报告与下载
├── paths.py                # 表格数据、上传文件和输出路径
├── datasets/
│   └── sample_tabular.csv   # 12 条 Wine 分类样例，target 为标签
├── uploads/                # 表格上传时自动创建
└── outputs/                # 保存分析结果时自动创建
```

## 使用方式

在项目根目录（`-OPD/`）使用新的导入路径：

```python
from tabular_data.tabular_octree import load_table_file, run_octree_analysis

df = load_table_file("tabular_data/datasets/sample_tabular.csv")
result = run_octree_analysis(df=df, target_col="target")
```

网页的 `/api/upload`、`/api/run/tabular` 接口也使用此模块。上传文件默认保存到 `tabular_data/uploads/`。

“运行表格模拟测试”调用 `/api/simulate`，使用内置 Wine 样例和本地特征规则进行模型交叉验证，不请求外部 LLM。页面上的四个数据集卡片及性能对比图是预置示意数据；模拟测试和上传分析的弹窗显示本次实际计算的指标、每轮接受/拒绝结果和决策树规则。

回归任务不适用的分类指标显示为“不适用”。迭代曲线跟踪已保留特征的指标，被拒绝的候选只显示在迭代记录中。图表库未加载时会显示指标表格，报告仍可下载。

分类和回归均使用 GPU XGBoost：`device="cuda:0"`、`tree_method="hist"`。Linux 依赖固定为 `xgboost-cu12==3.4.1`，该安装包使用 CUDA 12.9，已在本机 RTX 4060、Windows 驱动 566.26 / WSL 环境实际验证 GPU 分类和回归。CUDA 12 的小版本兼容机制允许此安装包在该驱动上运行。

基线、候选特征各轮和最终评估均使用 GPU XGBoost；训练结束时检查实际设备，拒绝自动切到 CPU 的结果。GPU 或驱动异常会正常报告，不再使用随机森林回退。报告记录 GPU 设备、XGBoost 版本和安装包的 CUDA 版本。用于生成解释规则的浅层 CART 树仍由 sklearn 构建。

已有环境需先卸载 `xgboost`，再安装 `xgboost-cu12==3.4.1`，避免两个发行包同时提供同名的 `xgboost` 模块。新环境可直接安装根目录的 `requirements.txt`。

模块复用项目根目录的 `config.py`（共享配置）、`llm_client.py`（API 客户端）和 `result_schema.py`（结果结构），不需要复制这些公共文件。表格特征生成的具体实现位于本目录的 `feature_generation.py`，共享客户端保留转发接口。

## 输出和路径配置

默认结果保存在 `tabular_data/outputs/`：

- `tabular_augmented.csv`：预处理并增加已接受特征后的表格。
- `tabular_results.json`：每轮特征代码、检查结果、指标和解释规则。
- `metrics.csv`：原始特征、扩充后特征及改善量的指标。

下载地址保持 `/outputs/tabular_augmented.csv`、`/outputs/tabular_results.json`、`/outputs/metrics.csv`。

可以在项目根目录的 `.env` 中设置 `TABULAR_DATASET_DIR`、`TABULAR_UPLOAD_DIR`、`TABULAR_OUTPUT_DIR` 覆盖上述目录。迭代次数继续使用 `MAX_ITERATIONS`，默认 3 轮。
