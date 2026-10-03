# 大创结题版工程系统使用说明

本项目是“面向生成式模型的高质量数据特征扩充技术研究”的结题版工程化系统。
它不是另起炉灶，而是在原 OCTree 表格特征扩充基础上，继续扩展 Math 和 Code 两类数据。

## 三类数据分别扩充什么

Tabular：扩充新特征列和特征生成代码。流程是 LLM 生成候选特征，沙盒检查，模型评估，再根据指标和决策树规则反馈下一轮。

Math：扩充数学 CoT 的中间推理步骤。流程是切分原始解答，构建 Reasoning Graph，遮盖中间节点，让 LLM 恢复，再做质量评估。数学模块已经接入队友的 `Math/math_reasoning_expander/`，支持多遮盖点和同一节点多轮反馈优化。

Code：现在支持两种模式。没有原始代码时，走“生成代码 -> 跑测试 -> 错误反馈 -> 修复”；有原始 code 和 reasoning 时，走队友的 rewrite 扩充流程，同步改写 reasoning 和 code，并做语义等价检查和质量评分。

表格专用代码与数据集集中在 `tabular_data/`：主流程为 `tabular_data/tabular_octree.py`，特征生成逻辑为 `tabular_data/feature_proposals.py`，正式数据集为 `tabular_data/datasets/` 下的 Jungle Chess、Balance Scale、Chess KRK；网页样例来自 Jungle 训练集。表格上传文件和分析结果分别保存到该目录的 `uploads/`、`outputs/`，详见 [表格模块说明](tabular_data/README.md)。

## 安装依赖

在 VSCode 终端进入本目录：

```powershell
python -m pip install -r requirements.txt
```

## 启动网站

Windows 用户在项目文件夹中双击 **`一键启动.bat`**。它会启动后台服务并自动打开 [完整网站](http://127.0.0.1:5000/)，关闭启动窗口不会停止服务。项目位于 `\\wsl.localhost\发行版\...` 或 `\\wsl$\发行版\...` 时，会自动使用对应 WSL 环境，无需另装一份 Windows 依赖。

之后直接双击 **`index.html`** 即可：本地 HTML 会检测服务并自动进入完整网站，Tabular 上传与模拟测试、Math、Code 和一键 Demo 都从网站调用后端。也可以直接收藏上面的地址。

电脑或 WSL 重启后需要再次双击 `一键启动.bat`。浏览器无法自行启动 Python；若服务尚未启动，本地 HTML 会显示启动提示并持续重试，服务就绪后自动进入。若浏览器拦截本地连接检测，点击提示中的“进入完整网站”。

Linux 用户可运行 `./一键启动.sh`。启动器会查找当前 Python、项目虚拟环境及本机 `llm_reviewer` Conda 环境；本机已有可用依赖。其他机器需先完成上面的依赖安装。可以用 `OPD_PYTHON` 指定已有 Python 环境。

启动日志位于 `logs/website.log`，后台进程编号记录在 `logs/website.pid`。重复点击启动脚本会复用已有服务。只启动后台、不打开浏览器时可使用 `python3 start_website.py --no-browser`。需要在终端查看实时日志或用 `Ctrl+C` 停止时，仍可用已装依赖的 Python 执行 `python app.py`（需先停止已有后台服务）。

## 前后端关系

`app.py` 是后端，负责运行 Python 算法、保存 outputs、返回 JSON。

`index.html` 是前端，负责页面展示、按钮点击、上传文件、把后端返回的结果显示出来。

前端点击按钮后，会调用这些接口：

- `/api/run/tabular`
- `/api/run/math`
- `/api/run/code`
- `/api/run/all_demo`

所以前端同学主要改 `index.html` 的样式和展示逻辑，不需要动算法代码。

## outputs 结果文件

运行 demo 后会生成：

- `tabular_data/outputs/tabular_results.json`：表格特征扩充结果
- `tabular_data/outputs/tabular_augmented.csv`：增强后的表格
- `tabular_data/outputs/metrics.csv`：baseline / optimized / improvement 指标
- `outputs/math_expanded.jsonl`：数学推理链扩充结果
- `outputs/math_quality.svg`：数学质量图
- `outputs/code_repair_traces.jsonl`：代码生成、修复或 rewrite 扩充轨迹
- `outputs/code_quality.svg`：代码 rewrite 质量对比图

表格文件的原有下载地址 `/outputs/tabular_results.json`、`/outputs/tabular_augmented.csv` 和 `/outputs/metrics.csv` 保持可用。

## Code 新增 rewrite 流程

队友的代码包已经接入到 `code_rewrite_feedback_expander/`。

如果输入里有：

```json
{
  "prompt": "...",
  "reasoning": ["..."],
  "code": "def ...",
  "tests": ["assert ..."]
}
```

就会走 rewrite 扩充：

```text
原始 code + reasoning
-> LLM Rewrite
-> 语义等价检查
-> 质量评分
-> 自然语言反馈
-> 再次 Rewrite
-> 有提升才保留
```

rewrite 策略包括：

- `cot`：让 reasoning 更清楚
- `style`：改善代码风格
- `ast`：做 AST 等价重构
- `variable`：替换更清晰的变量名
- `control_flow`：调整控制流

保留条件是：

```text
语义检查通过 + 质量分提升
```

语义检查包括 AST 解析、函数签名、危险调用过滤、单元测试、AST 相似度、CodeBLEU-like 相似度。

质量评分包括可读性、复杂度、长度平衡、多样性和代码风格。

## Math 完整迭代流程

队友的数学包保留在 `Math/math_reasoning_expander/`，主工程通过 `math_adapter.py` 调用，不需要单独启动 Math 目录。

```text
原始题目 + 原始 CoT
-> 切分推理节点并构建 Reasoning Graph
-> 外层选择一个中间节点
-> 内层最多反馈改进 3 次
-> 生成节点质量高于原节点时保留并合入 CoT
-> 继续选择下一个节点
-> 输出扩充结果、每轮轨迹和质量图
```

默认配置是外层最多 10 个遮盖点、内层每个点最多 3 次、连续 2 个节点没有提升就提前停止。可以在 `.env` 中调整：

```text
MATH_MAX_MASK_ROUNDS=10
MATH_MAX_REFINE_ROUNDS=3
MATH_PATIENCE=2
MATH_ACCEPT_THRESHOLD=0.80
```

真实 API 不可用时仍会自动使用 mock；完整数学包运行异常时会自动切回 `math_adapter.py` 的单轮兼容流程。

## 常见问题

1. `ModuleNotFoundError: No module named 'flask'`

说明依赖没装到当前 Python 环境，重新运行：

```powershell
python -m pip install -r requirements.txt
```

2. 网站打不开

先确认 `python app.py` 的终端没有关闭，然后打开：

```text
http://127.0.0.1:5000
```

3. LLM 调用失败

项目会自动使用 mock 兜底，所以 demo 仍然能跑。正式使用时可以在 `.env` 里配置：

```text
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=你的base_url
OPENAI_MODEL=模型名
```

可以先复制 `.env.example` 并改名为 `.env`，再填写自己的配置。`.env` 已被 `.gitignore` 排除，不要把完整 API Key 发到群里、写进截图或放进压缩包。
