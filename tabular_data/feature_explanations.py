"""可审计的公式说明；人读近似值与机器执行的全精度表达式分离。"""
from __future__ import annotations

import ast
import copy
import math
import re

import numpy as np


def number(value):
    value = float(value)
    if value and abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def display_values(value):
    """仅用于显示，不能把返回值用于模型训练或指标决策。"""
    if isinstance(value, dict):
        return {k: display_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [display_values(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return number(value)
    return value


def quantile_details(values, probability):
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = (len(ordered) - 1) * probability
    low, high = math.floor(rank), math.ceil(rank)
    return {"probability": probability, "rank_zero_based": rank, "lower_index": low,
            "upper_index": high, "lower_value": float(ordered[low]),
            "upper_value": float(ordered[high]), "weight": rank - low}


def symbol_name(column, stat):
    # 编号由列名的字节编码保底，避免不同标点清理后重名。
    safe = column if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", column) else "col_" + column.encode().hex()
    return f"{stat}_{safe}"


def _quantile_calculation(record, count):
    p, low, high, w = (record[k] for k in ("probability", "lower_value", "upper_value", "weight"))
    return (f"训练列升序排列，p={number(p)}；零基位置 h=(n−1)p=({count}−1)×{number(p)}"
            f"={number(record['rank_zero_based'])}；相邻值 x[{record['lower_index']}]={number(low)}、"
            f"x[{record['upper_index']}]={number(high)}；线性插值 Q=(1−{number(w)})×{number(low)}"
            f"+{number(w)}×{number(high)}={number((1-w)*low+w*high)}。")


def parameter_registry(profile):
    if "_parameter_registry" in profile:
        return profile["_parameter_registry"]
    registry = {}
    for column, stats in profile["columns"].items():
        for stat in ("median", "q25", "q75", "scale"):
            if stat in ("median", "q25", "q75"):
                calculation = _quantile_calculation(stats["quantile_details"][stat], profile["rows"])
                purpose = ("用中位数作为居中基准或均分门槛，减少极端值影响。" if stat == "median" else
                           "预先固定的四分位门槛，用于探索不同区间；不是从测试集挑出的最优阈值。")
                value = stats[stat]
            else:
                value = stats["scale"]
                calculation = (f"Q25: {_quantile_calculation(stats['quantile_details']['q25'], profile['rows'])} "
                    f"Q75: {_quantile_calculation(stats['quantile_details']['q75'], profile['rows'])} "
                    f"IQR=Q75−Q25={number(stats['q75'])}−{number(stats['q25'])}={number(stats['iqr'])}；"
                    f"均值 μ=Σx/n={number(stats['sum'])}/{profile['rows']}={number(stats['mean'])}；"
                    f"总体标准差 σ=√(Σ(x−μ)²/n)=√({number(stats['squared_deviations'])}/{profile['rows']})"
                    f"={number(stats['std'])}（ddof=0）；scale=max(IQR,σ,ε)={number(stats['scale'])}。"
                    "选择两种离散程度中较大的值以避免放大微小波动；ε=1e-6 是固定数值下限，并非拟合结果。")
                purpose = "无量纲化或设定衰减尺度，使用训练集冻结值；不在验证/测试重新估计。"
            registry[symbol_name(column, stat)] = {"value": value, "kind": "training_statistic",
                "column": column, "statistic": stat, "source": "train", "calculation": calculation, "purpose": purpose}
    registry.update(profile.get("_extra_parameters", {}))
    profile["_parameter_registry"] = registry
    return registry


def explain_expression(expression, profile):
    """符号参数必须能追溯到训练统计；未知裸常数拒绝，不能事后编造来源。"""
    registry = parameter_registry(profile)
    tree = ast.parse(expression, mode="eval")
    constants, used = [], {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in registry:
            used[node.id] = registry[node.id]
        elif isinstance(node, ast.Name) and node.id not in {"np", "df"}:
            raise ValueError(f"未声明的公式参数：{node.id}；请使用训练统计符号。")
    literal_rules = {
        0: "零基准：作为分段判断/空历史输出或非负裁剪下界；由表达式所在分支决定，不从数据拟合。",
        1: "固定常数：分段指示列的 1 表示条件成立；用于 1+|z| 时保证分母至少为 1。具体含义由出现位置决定，不是估计系数。",
        2: "固定二次幂：x²=x×x；几何周长平方用于消去与面积的量纲差。不是训练得到的权重。",
        4: "几何常数：椭圆半轴为长短轴/2，面积=π×长轴×短轴/4；圆的周长平方/面积=4π。",
        180: "角度单位换算：180 度=π 弧度，因此 θ_rad=θ_deg×π/180。",
        1e-6: "固定数值保护 ε=10⁻⁶=0.000001，避免零分母或 log(0)；工程设定，未通过验证/测试调参。",
    }
    seen_literals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in literal_rules:
                raise ValueError(f"裸常数 {number(node.value)} 无可审计来源；请改用训练统计符号。")
            if node.value not in seen_literals:
                constants.append({"symbol": number(node.value), "value": node.value, "kind": "fixed_design",
                                  "source": "预先规定的数学或数值规则", "calculation": literal_rules[node.value],
                                  "purpose": "参见显示公式中的具体位置。",
                                  "occurrences": list(dict.fromkeys(ast.unparse(parents[item]) for item in ast.walk(tree)
                                      if isinstance(item, ast.Constant) and isinstance(item.value, (int, float))
                                      and item.value == node.value))})
                seen_literals.add(node.value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np":
            descriptions = {"pi": (np.pi, "π 是圆周率，来自圆/椭圆几何或弧度换算，不从数据估计。"),
                "log1p": (1, "log1p(x)=ln(1+x)：固定加 1 使 x=0 时输出 0，ln 的底数为自然常数 e。"),
                "log10": (10, "log10 的底数固定为 10，是十进对数定义，不是拟合参数。"),
                "tanh": (2, "tanh(z)=(exp(2z)−1)/(exp(2z)+1)，2 来自双曲正切恒等式，输出在 (−1,1)。"),
                "exp": (math.e, "exp(z)=e^z，e 为自然常数；exp(−t/s) 在 t=s 时为 e⁻¹，表达平滑衰减。"),
                "sqrt": (.5, "sqrt(x)=x^(1/2)，1/2 是平方根定义；用于把面积还原为长度量纲。"),
                "log": (math.e, "log(x)=ln(x)，底数为自然常数 e，不是训练参数。")}
            if node.attr in descriptions and node.attr not in seen_literals:
                value, detail = descriptions[node.attr]
                constants.append({"symbol": "np." + node.attr, "value": float(value), "kind": "mathematical_definition",
                                  "source": "数学定义", "calculation": detail, "purpose": "解释函数隐含常数。"})
                seen_literals.add(node.attr)

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(float(used[node.id]["value"])), node) if node.id in used else node

    compiled = ast.unparse(ast.fix_missing_locations(Substitute().visit(copy.deepcopy(tree))))
    constants = [{"symbol": name, **item} for name, item in used.items()] + constants
    return compiled, ast.unparse(tree), constants, used


def derivation_steps(expression, construction):
    """输出可复核的运算分解，而不是语言模型的内部思维链。"""
    tree = ast.parse(expression, mode="eval")
    steps = ["选列依据：训练集互信息反映单列与目标的统计关联；几何/业务定义给出组合假说，不能据此保证提升。",
             "构造依据：" + construction]
    counter = 0

    def visit(node):
        nonlocal counter
        if isinstance(node, ast.Subscript):
            return ast.unparse(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call)):
            clone = copy.deepcopy(node)
            for field, value in ast.iter_fields(clone):
                if field == "func":
                    continue
                if isinstance(value, ast.expr):
                    setattr(clone, field, ast.parse(visit(value), mode="eval").body)
                elif isinstance(value, list):
                    setattr(clone, field, [ast.parse(visit(v), mode="eval").body if isinstance(v, ast.expr) else v for v in value])
            counter += 1
            name = f"u{counter}"
            steps.append(f"{name} = {ast.unparse(clone)}")
            return name
        return ast.unparse(node)

    final = visit(tree.body)
    steps.append(f"新列 = {final}；所有输入逐行计算，统计参数仅在训练集计算一次。")
    steps.append("检验假说：保持 XGBoost 配置不变，用验证集主指标增益及其他指标不退化共同决定是否接受。")
    return steps
