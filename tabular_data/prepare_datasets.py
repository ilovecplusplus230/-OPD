"""下载 OpenML 官方多分类表格数据，生成本项目可直接读取的训练集和独立测试集。

在项目根目录执行：python -m tabular_data.prepare_datasets
离线重建：python -m tabular_data.prepare_datasets --offline
检查 GPU 特征扩充：python -m tabular_data.prepare_datasets --offline --verify-gpu
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import re
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

from .paths import DATASET_DIR
from .tabular_octree import load_table_file, preprocess_dataframe
from .split_validation import audit_frames, label_counts, load_split_policy, write_distribution


from .dataset_registry import SPECS
SEED = 42


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(key: str, offline: bool) -> Path:
    spec = SPECS[key]
    archive = DATASET_DIR / "_raw" / spec.get("archive_name", f"{key}.arff")
    algorithm = "sha256" if spec.get("sha256") else "md5"
    expected = spec[algorithm]
    if not archive.is_file():
        if offline:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        request = urllib.request.Request(spec["download_url"], headers={"User-Agent": "OPD-tabular/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            temporary.write_bytes(response.read())
        if hashlib.new(algorithm, temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise ValueError(f"{key} 下载内容与官方 MD5 不符。")
        temporary.replace(archive)
    if hashlib.new(algorithm, archive.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{key} 原始文件与官方 MD5 不符。")
    return archive


def normalize(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(name).strip() for name in frame.columns]
    frame = frame.rename(columns={target: "target"})
    for name in frame.select_dtypes(include=["object", "string"]).columns:
        frame[name] = frame[name].map(lambda value: value.decode() if isinstance(value, bytes) else value)
        frame[name] = frame[name].str.strip().replace("?", np.nan)
    return frame[[name for name in frame.columns if name != "target"] + ["target"]]


def prepare_frames(key: str, archive: Path):
    data = archive.read_bytes()
    details = {"removed_columns": {}, "normalization": ["UTF-8、逗号分隔、有表头，标签列统一为 target。"]}
    spec = SPECS[key]
    if spec.get("archive_format") == "satimage_zip":
        with ZipFile(archive) as source:
            parts = [pd.read_csv(source.open(name), sep=r"\s+", header=None) for name in ("sat.trn", "sat.tst")]
        if [len(part) for part in parts] != [4435, 2000] or any(part.shape[1] != 37 for part in parts):
            raise ValueError("Satimage 原始文件维度与声明不符。")
        frame = pd.concat(parts, ignore_index=True)
        frame.columns = [f"pixel_{i//4+1}_band_{i%4+1}" for i in range(36)] + ["target"]
        frame["target"] = frame.target.astype(str)
        if set(frame.target) != {"1", "2", "3", "4", "5", "7"}:
            raise ValueError("Satimage 原始六类标签不符。")
        details["original_partitions"] = {"sat.trn": 4435, "sat.tst": 2000}
        details["normalization"].append("合并 UCI 官方 train/test 后按项目协议重分；36 列依官方像素及波段顺序命名，数值不变。")
    else:
        text = re.sub(r"(?<=,) +", "", data.decode("utf-8-sig"))
        records, _ = loadarff(StringIO(text))
        frame = normalize(pd.DataFrame(records), spec["target"])
    if len(frame.columns) - 1 != spec["features"] or frame.target.nunique() != spec["classes"]:
        raise ValueError(f"{key} 官方版本的特征数或类别数不符。")
    if spec.get("rename_columns"):
        frame = frame.rename(columns=spec["rename_columns"])
        details["column_name_mapping"] = spec["rename_columns"]
    if spec.get("coordinate_columns"):
        for column in spec["coordinate_columns"]:
            coordinates = pd.to_numeric(frame[column], errors="raise")
            if not coordinates.isin(range(1, 9)).all():
                raise ValueError(f"{key}/{column} 不是 1–8 的棋盘坐标。")
            frame[column] = coordinates.astype(float)
        details["coordinate_columns"] = spec["coordinate_columns"]
        details["normalization"].append("KRK 三个 file 字段按已与 UCI 全量逐行核对的 1–8 坐标读取；不添加人工距离特征。")
    if spec.get("categorical_value_names"):
        # 可逆地命名格子状态，避免 CSV 读取器把无序类别重新推断为连续数值。
        mapping = spec["categorical_value_names"]
        for column in frame.columns[:-1]:
            values = frame[column].astype(str)
            if not values.isin(mapping).all():
                raise ValueError(f"{key}/{column} 存在未声明的类别编码。")
            frame[column] = values.map(mapping)
        details["categorical_value_names"] = mapping
        details["normalization"].append("无序特征编码按公开映射可逆命名；标签不变，训练时才拟合独热编码。")
    details.update(original_target=spec["target"], source_files=[archive.name], scope=spec["scope"])
    details["normalization"].append("保留官方多分类标签与所有特征，ARFF 转 CSV；不合并类别，不人为改写数值。")

    raw_rows = len(frame)
    details["raw_class_counts"] = label_counts(frame)
    frame = frame.drop(columns=list(details["removed_columns"]))
    if SPECS[key]["task"] == "classification":
        frame = frame.drop_duplicates().copy()
        details["rows_removed_as_duplicates"] = raw_rows - len(frame)
        details["normalization"].append("按保留的特征及标签，在划分前删除完全相同的重复行。")
    policy = load_split_policy()
    ratios = policy["ratios"]
    train, remaining = train_test_split(frame, train_size=ratios["train"], random_state=policy["seed"], stratify=frame.target)
    validation, test = train_test_split(remaining, test_size=ratios["test"]/(ratios["validation"]+ratios["test"]),
                                      random_state=policy["seed"], stratify=remaining.target)
    details["split"] = {"method": "stratified_random_20_40_40", "seed": policy["seed"],
                        "protocol": policy["protocol"], "ratios": ratios,
                        "class_ratio_policy": policy["class_ratio_policy"], "policy_file": "split_policy.json"}
    details["class_distribution"] = audit_frames({"train": train, "validation": validation, "test": test},
                                                 policy=policy, original_counts=label_counts(frame))
    return train, validation, test, raw_rows, details


def validate_csv(path: Path, task: str):
    frame = load_table_file(path)
    if frame.columns[-1] != "target" or frame.columns.duplicated().any():
        raise ValueError(f"标签或列名不符合要求：{path}")
    if frame.target.isna().any():
        raise ValueError(f"标签不能包含缺失值：{path}")
    processed, target, metadata = preprocess_dataframe(frame)
    if target != "target" or metadata["task_type"] != task:
        raise ValueError(f"项目自动识别的任务类型不正确：{path}")
    if not np.isfinite(processed.to_numpy(dtype=float)).all():
        raise ValueError(f"预处理后仍有非有限数值：{path}")
    if task == "classification" and frame.target.value_counts().min() < 5:
        raise ValueError(f"每个类别必须至少有 5 条记录：{path}")
    return frame, {
        "rows": len(frame), "columns": len(frame.columns), "bytes": path.stat().st_size,
        "sha256": sha256(path), "missing_feature_cells": int(frame.drop(columns="target").isna().sum().sum()),
        "task_detected": metadata["task_type"], "preprocess_passed": True,
    }


def prepare(key: str, archive: Path):
    train, validation, test, raw_rows, details = prepare_frames(key, archive)
    spec = SPECS[key]
    if raw_rows != spec["rows"]:
        raise ValueError(f"{key} 官方文件记录数不符：{raw_rows} != {spec['rows']}")
    directory = DATASET_DIR / key
    directory.mkdir(parents=True, exist_ok=True)
    checks = {}
    for split, frame in (("train", train), ("validation", validation), ("test", test)):
        path = directory / f"{split}.csv"
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        _, checks[split] = validate_csv(path, spec["task"])
    manifest = {
        "name": spec["name"], "key": key, "task_type": spec["task"], "description": spec["description"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "provider": spec["provider"],
        "openml_id": spec["id"] if spec["provider"] == "OpenML" else None,
        "uci_id": spec["id"] if spec["provider"] == "UCI" else None,
        "source_version": spec.get("version"),
        "source_url": f"https://www.openml.org/d/{spec['id']}" if spec["provider"] == "OpenML" else spec["original_source_url"],
        "original_source_url": spec.get("original_source_url"),
        "download_url": spec["download_url"], "doi": spec["doi"], "citation": spec["citation"],
        "license": "Public (OpenML metadata)" if spec["provider"] == "OpenML" else "CC BY 4.0 (UCI)",
        "license_url": f"https://www.openml.org/api/v1/json/data/{spec['id']}" if spec["provider"] == "OpenML" else spec["original_source_url"],
        "benchmark_reference": spec.get("benchmark_reference", "https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf#page=9"),
        "literature_scope": spec.get("literature_scope", "ICML 2023 OpenFE 自动特征生成实验。"),
        "minority_tier": spec.get("minority_tier"),
        "categorical_encoding": spec.get("categorical_encoding", "ordinal"),
        "class_names": spec.get("class_names", {}),
        "relationship_basis": spec.get("relationship_basis"),
        "additional_reference": spec.get("additional_reference"),
        "raw_archive": {"path": str(archive.relative_to(DATASET_DIR)), "sha256": sha256(archive),
                        "bytes": archive.stat().st_size, "official_md5": spec.get("md5"),
                        "screening_sha256": spec.get("sha256")},
        "source_rows": raw_rows, "feature_count": len(train.columns) - 1,
        "features": list(train.columns[:-1]), "target_column": "target", "preparation": details,
        "files": checks,
    }
    if spec["task"] == "classification":
        manifest["class_counts"] = {split: {str(label): int(count) for label, count in frame.target.value_counts().items()}
                                    for split, frame in (("train", train), ("validation", validation), ("test", test))}
    write_json(directory / "manifest.json", manifest)
    write_distribution(directory, details["class_distribution"])
    print(f"验证通过：{key}, train={len(train):,}, validation={len(validation):,}, test={len(test):,}, features={len(train.columns)-1}, task={spec['task']}", flush=True)
    return manifest


def verify_gpu(manifests):
    from .training import run_training

    validation = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for manifest in manifests:
        result = run_training(manifest["key"], iterations=1, offline=True, save_outputs=False)
        if not result["success"] or result["model"]["device"] != "cuda:0":
            raise RuntimeError(f"GPU 特征扩充失败：{manifest['key']}")
        validation["datasets"][manifest["key"]] = {
            "rows_checked": result["rows"], "model": result["model"], "protocol": result["protocol"],
            "source_files": result["source_files"], "validation": result["validation"], "test": result["test"],
            "iteration_count": len(result["iterations"]), "passed": True,
        }
        write_json(DATASET_DIR / "validation.json", validation)
        print(f"GPU 验证通过：{manifest['key']}，全量 20/40/40 划分，1 轮实际特征扩充", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="只使用 datasets/_raw 中已下载的官方文件")
    parser.add_argument("--verify-gpu", action="store_true", help="每个数据集做 1 轮 GPU 特征扩充和独立测试")
    parser.add_argument("--dataset", choices=("all", *SPECS), default="all", help="可单独准备新数据，不重写已有数据")
    args = parser.parse_args()
    keys = list(SPECS) if args.dataset == "all" else [args.dataset]
    with ThreadPoolExecutor(max_workers=3) as executor:
        archives = list(executor.map(lambda key: download(key, args.offline), keys))
    manifests = [prepare(key, archive) for key, archive in zip(keys, archives)]
    catalog = [json.loads(path.read_text(encoding="utf-8")) for path in [DATASET_DIR / key / "manifest.json" for key in SPECS if (DATASET_DIR / key / "manifest.json").exists()]]
    write_json(DATASET_DIR / "catalog.json", {"schema_version": 7, "datasets": catalog})
    write_json(DATASET_DIR / "validation.json", {
        "status": "split_checks_passed", "gpu_status": "not_run_by_this_preparation", "protocol": "holdout_20_40_40",
        "message": "已检查每类覆盖、原始类别比例、三份集合互斥及数据校验值；GPU 结果以各次训练报告为准。",
        "datasets": {m["key"]: m["preparation"]["class_distribution"] for m in manifests},
    })
    if args.verify_gpu:
        verify_gpu(manifests)


if __name__ == "__main__":
    main()
