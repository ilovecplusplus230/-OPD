"""正式训练日志：子进程 stdout/stderr 同步显示、逐行落盘及运行状态。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid


TRAINING_VERSION = "multiclass_reasoned_v7"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def training_output_dir(root, dataset, *, profile="weak", noise_std=0., feature_mode="reasoned",
                        iterations=5, candidates_per_round=5, seed=42, dataset_view="full", train_fraction=1.):
    if feature_mode != "reasoned" or dataset_view != "full":
        raise ValueError("只支持 reasoned 生成器及完整特征视图。")
    variant = "clean" if noise_std == 0 else f"noise_{noise_std:g}"
    root = Path(root)
    if dataset_view != "full" or train_fraction != 1.:
        root = root / "research" / f"view_{dataset_view}_budget_{train_fraction:g}"
    return (root / dataset / "holdout_20_40_40" / profile / variant / f"features_{feature_mode}"
            / f"rounds_{iterations}_candidates_{candidates_per_round}" / f"seed_{seed}")


@contextmanager
def _output_lock(directory):
    # Linux/WSL 正式训练环境；防止两次训练混写同一个日志与模型目录。
    with (directory / ".training.lock").open("a") as stream:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"该目录已有训练进程运行：{directory}") from exc
        yield


def _interrupt(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_logged_process(command, output_dir, *, metadata=None):
    """使用独立 worker 捕获 Python/C/CUDA 输出；不用用户另加 tee 命令。"""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    with _output_lock(output_dir), log_path.open("w", encoding="utf-8", buffering=1) as log:
        state = {"run_id": uuid.uuid4().hex, "training_version": TRAINING_VERSION,
                 "status": "running", "started_at_utc": utc_now(), "output_dir": str(output_dir),
                 "log_file": str(log_path), "configuration": metadata or {},
                 "note": "运行中或失败时，training_results.json 如存在可能属于上一次成功运行；请比较 run_id，以本文件和 progress.json 为准。"}
        atomic_json(output_dir / "run_status.json", state)
        atomic_json(output_dir / "progress.json", {**state, "stage": "starting", "completed_rounds": 0, "iterations": []})

        def emit(message):
            log.write(message)
            log.flush()
            try:
                sys.stdout.write(message)
                sys.stdout.flush()
            except BrokenPipeError:
                pass  # 终端管道关闭仍继续保存本地日志。

        process = None
        code = 1
        try:
            emit(f"训练版本：{TRAINING_VERSION}\n开始时间：{state['started_at_utc']}\n结果目录：{output_dir}\n完整日志：{log_path}\n")
            env = dict(os.environ, OPD_TRAINING_RUN_ID=state["run_id"], PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, start_new_session=(os.name == "posix"))
            for line in process.stdout:
                emit(line)
            code = process.wait()
            state["status"] = "completed" if code == 0 else ("interrupted" if code in (-2, 130) else "failed")
        except KeyboardInterrupt:
            if process is not None:
                _interrupt(process)
                emit(process.stdout.read())
            code, state["status"] = 130, "interrupted"
            emit("\n训练已中断；已输出的日志和最近完成轮次的进度已保留。\n")
        except Exception:
            if process is not None:
                _interrupt(process)
            state["status"] = "failed"
            emit(traceback.format_exc())
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            state.update(exit_code=code, finished_at_utc=utc_now())
            atomic_json(output_dir / "run_status.json", state)
            progress_path = output_dir / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.update(status=state["status"], exit_code=code, finished_at_utc=state["finished_at_utc"])
            atomic_json(progress_path, progress)
            emit(f"\n运行状态：{state['status']}；退出码：{code}；结束时间：{state['finished_at_utc']}\n日志已保存：{log_path}\n")
        return code if code >= 0 else 128 + abs(code)
