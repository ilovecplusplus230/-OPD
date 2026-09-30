# 文件说明

这个文档给队友看每个文件大概是干什么的，不需要先懂完整后端框架。

## app.py

Flask 后端入口。网站打开、文件上传、三个 demo 接口都从这里进来。

保留了原来的：

- `GET /`
- `POST /api/upload`
- `POST /api/simulate`

新增或整理了：

- `GET /api/health`
- `GET /api/ping`（供本地 HTML 和启动器快速识别网站）
- `GET /api/results`
- `POST /api/run/tabular`
- `POST /api/run/math`
- `POST /api/run/code`
- `POST /api/run/all_demo`

## index.html

前端页面。它负责展示页面、按钮、图表和控制台输出。

前端同学如果要改图片路径、页面样式、展示区域，主要改这个文件。
后端接口地址保持 `/api/...` 不变即可。

## website_entry.js / start_website.py / 一键启动脚本

`website_entry.js` 让直接打开的 `index.html` 自动连接并跳转到完整网站；服务未启动时显示操作提示并重试。通过网站地址访问时不做跳转。

`start_website.py` 检查可用 Python 环境，启动独立后台进程，等待网站就绪后打开浏览器；日志在 `logs/website.log`。已有服务会直接复用。

Windows 双击 `一键启动.bat`，由 `start_website_windows.ps1` 判断使用 Windows 还是项目所在的 WSL 环境；Linux 使用 `一键启动.sh`。电脑或 WSL 重启后需重新启动，浏览器不能仅靠 HTML 启动 Python。

## config.py

配置文件。集中管理路径、模型名、API 地址、最大迭代次数、mock 开关。

它会自动创建：

- `outputs/`
- `uploads/`
- `sample_data/`

## llm_client.py

统一 LLM 调用封装。

Tabular、Math、Code 都通过它调用模型。真实 API 失败时会自动 fallback 到 mock，保证 demo 不崩。

表格专用的提示词和回退规则已移到 `tabular_data/feature_generation.py`，此文件保留兼容调用接口。

## result_schema.py

统一结果结构。三类 pipeline 都返回 `ResultRecord`，前端就不用为每种数据猜不同字段。

核心字段包括：

- `input_summary`
- `structured_representation`
- `generated`
- `verification`
- `feedback`
- `accepted`
- `metrics`

## tabular_data/

表格 OCTree 主线与数据集的统一目录：

- `tabular_octree.py`：原根目录表格主流程，导入路径为 `tabular_data.tabular_octree`。
- `feature_generation.py`：LLM 表格特征生成提示词及离线回退规则。
- `web.js`：统一表格上传、模拟测试和结果展示，兼容分类与回归指标。
- `paths.py`：表格专用数据、上传文件及输出路径，可用 `TABULAR_DATASET_DIR`、`TABULAR_UPLOAD_DIR`、`TABULAR_OUTPUT_DIR` 配置。
- `datasets/sample_tabular.csv`：从原 `sample_data/` 移入的表格样例。
- `uploads/`：表格上传文件，上传时自动创建。
- `outputs/`：表格运行结果，保存时自动创建。

核心流程：

```text
读取 CSV/DataFrame
-> 自动识别 target
-> 清洗、编码、归一化
-> LLM 生成新特征代码
-> 沙盒检查 NaN/Inf/常数列/重复列
-> XGBoost 或 RandomForest 交叉验证
-> DecisionTree 规则反馈
-> 接受或拒绝新特征
```

输出：

- `tabular_data/outputs/tabular_results.json`
- `tabular_data/outputs/tabular_augmented.csv`
- `tabular_data/outputs/metrics.csv`

对应的 `/outputs/...` 下载接口保持兼容。

## math_adapter.py

数学扩充适配器。

它负责把队友 Math.zip 里的 `math_reasoning_expander` 接到 Flask、统一 LLM 客户端和统一结果结构。如果完整数学包临时不可用，它会自动使用 fallback demo。

核心流程：

```text
原始数学题 + 原始 CoT
-> 切分步骤
-> 构建 Reasoning Graph
-> 随机遮盖中间推理节点
-> 每个节点最多反馈恢复 3 次
-> 生成质量高于原节点时保留
-> 最多继续处理 10 个遮盖点
-> 合成 expanded_cot 并记录完整轨迹
```

输出：

- `outputs/math_expanded.jsonl`
- `outputs/math_quality.svg`

## Math/

从队友 `Math.zip` 保留下来的完整数学扩充包。主工程实际使用的是 `Math/math_reasoning_expander/`：

- `parser.py`：切分 CoT 并构造 Reasoning Graph
- `masking.py`：随机选择中间节点、公式节点或路径进行遮盖
- `llm.py`：定义数学 Fill-in-the-Middle 所需的 LLM 接口
- `evaluators.py`：评价去重复性、逻辑一致性、公式正确性、完整性和推理增益
- `feedback.py`：把弱项转换成下一轮恢复提示
- `pipeline.py`：控制外层选点和内层反馈优化
- `visualization.py`：生成数学质量对比 SVG

平时只需运行根目录的 `python app.py`，不需要进入 `Math/` 单独运行。

## code_expander.py

Code 数据扩充主入口。

现在有两条路径：

1. 只有 `prompt / starter_code / tests`：走原来的代码生成、测试、错误反馈、修复。
2. 有 `prompt / reasoning / code / tests`：走队友的 code rewrite 扩充流程。

这样老 demo 不坏，新 code rewrite 思路也能展示。

## code_rewrite_feedback_expander/

队友 code rewrite 思路的工程化包。

每个文件作用：

- `models.py`：定义 CodeRecord、RewriteCandidate、ExpansionRecord 等数据结构
- `io_utils.py`：读写 JSONL，并兼容 reasoning、cot、chain_of_thought 等字段
- `llm.py`：独立 rewrite LLM 客户端，支持 mock 和 OpenAI-compatible API
- `semantic.py`：语义等价检查，包括 AST、签名、安全、单测、相似度
- `quality.py`：质量评分，包括可读性、复杂度、长度平衡、多样性、风格
- `feedback.py`：把语义和质量问题转成自然语言反馈
- `pipeline.py`：核心多策略、多轮 rewrite 控制器
- `visualization.py`：生成 `outputs/code_quality.svg`

## safe_exec.py

代码沙盒执行工具。

它用 subprocess 跑临时代码文件，不直接裸 exec，可以捕获：

- `SyntaxError`
- `AssertionError`
- `Timeout`
- `RuntimeError`
- `ImportError`
- `UnknownError`

## sample_data/

数学和代码的本地 demo 数据。表格样例已移至 `tabular_data/datasets/`。

- `sample_math.jsonl`：数学 CoT 样例
- `sample_code.json`：代码样例，第一条会走 rewrite，第二条保留生成修复路径

## outputs/

数学和代码的运行结果目录。表格结果集中在 `tabular_data/outputs/`。

## legacy/

旧代码备份目录。如果后续需要对照中期脚本，可以把原来的 `test_advanced2.py`、`test_final.py` 等放这里。
