"""用已有 Python 环境启动本地网站，等待就绪后打开浏览器。

仅依赖标准库；不会安装依赖，也不会结束占用端口的其他程序。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


BASE_DIR = Path(__file__).resolve().parent
SITE_URL = "http://127.0.0.1:5000/"
LOG_DIR = BASE_DIR / "logs"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def website_ready() -> bool:
    try:
        with OPENER.open(SITE_URL + "api/ping", timeout=1) as response:
            data = json.load(response)
        return data.get("service") == "opd-web" and data.get("status") == "ok"
    except (OSError, ValueError, AttributeError, urllib.error.URLError):
        return False


def port_in_use() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 5000), timeout=1):
            return True
    except OSError:
        return False


def find_python() -> str:
    """优先复用当前环境，再尝试项目环境和本机已安装的 Conda 环境。"""
    relative_python = Path("python.exe") if os.name == "nt" else Path("bin/python")
    venv_python = Path("Scripts/python.exe") if os.name == "nt" else relative_python
    candidates = [os.getenv("OPD_PYTHON"), sys.executable]
    candidates += [str(BASE_DIR / name / venv_python) for name in (".venv", "venv")]
    candidates += [
        str(Path.home() / distribution / "envs" / "llm_reviewer" / relative_python)
        for distribution in ("miniconda3", "anaconda3", "miniforge3")
    ]
    seen = set()
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        executable = shutil.which(candidate)
        if not executable or executable in seen:
            continue
        seen.add(executable)
        try:
            result = subprocess.run(
                [executable, "-c", "import app"], cwd=BASE_DIR,
                capture_output=True, text=True, errors="replace", timeout=45,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{executable}: {exc}")
            continue
        if result.returncode == 0:
            return executable
        detail = (result.stderr or result.stdout).strip().splitlines()
        errors.append(f"{executable}: {detail[-1] if detail else '导入失败'}")
    raise RuntimeError(
        "未找到能运行网站的 Python 环境。请先用目标环境安装 requirements.txt，"
        "或设置 OPD_PYTHON 指向已有环境的 Python。\n" + "\n".join(errors)
    )


def ensure_website() -> bool:
    """返回是否启动了新进程。现有本站服务直接复用。"""
    if website_ready():
        return False
    if port_in_use():
        raise RuntimeError("5000 端口已被其他服务占用。请关闭该服务后重试。")

    python = find_python()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "website.log"
    environment = dict(os.environ, FLASK_DEBUG="0", PYTHONUNBUFFERED="1")
    process_options = {"start_new_session": True}
    if os.name == "nt":
        process_options = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [python, "-u", str(BASE_DIR / "app.py")], cwd=BASE_DIR,
            env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, close_fds=True, **process_options,
        )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if website_ready():
            if process.poll() is None:
                (LOG_DIR / "website.pid").write_text(str(process.pid), encoding="utf-8")
            return True
        if process.poll() is not None:
            raise RuntimeError(f"网站启动失败，详情请查看：{log_path}")
        time.sleep(0.3)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise RuntimeError(f"网站启动超时，详情请查看：{log_path}")


def open_browser() -> None:
    # WSL 中直接调用 Windows 默认浏览器，避免 Linux 缺少桌面浏览器。
    windows_cmd = Path("/mnt/c/Windows/System32/cmd.exe")
    if os.getenv("WSL_DISTRO_NAME") and windows_cmd.exists():
        subprocess.run(
            [str(windows_cmd), "/c", "start", "", SITE_URL],
            cwd=windows_cmd.parent, check=True,
        )
    elif not webbrowser.open(SITE_URL):
        print(f"请点击或在浏览器中打开：{SITE_URL}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="仅启动后台服务")
    args = parser.parse_args()
    try:
        started = ensure_website()
    except (OSError, RuntimeError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    print(f"{'网站已在后台启动' if started else '网站已在运行'}：{SITE_URL}")
    print("现在可以直接打开 index.html；关闭启动窗口不会停止网站。")
    if not args.no_browser:
        try:
            open_browser()
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"自动打开浏览器失败：{exc}\n请打开 {SITE_URL}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
