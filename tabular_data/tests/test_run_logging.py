"""实际子进程验证完整输出落盘、失败保留及中断；不需要 GPU。"""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tabular_data.run_logging import _output_lock, run_logged_process


class TrainingLogTests(unittest.TestCase):
    def test_stdout_stderr_native_output_and_run_id_are_captured(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as terminal:
            code = run_logged_process([sys.executable, "-u", "-c",
                "import os,sys; print('公式推导'); print('Python warning',file=sys.stderr); os.write(2,b'native CUDA message\\n'); print(os.environ['OPD_TRAINING_RUN_ID'])"], folder)
            self.assertEqual(code, 0)
            log = (Path(folder)/"training.log").read_text()
            for text in ("公式推导", "Python warning", "native CUDA message"):
                self.assertIn(text, log)
                self.assertIn(text, terminal.getvalue())
            state = json.loads((Path(folder)/"run_status.json").read_text())
            self.assertIn(state["run_id"], log)
            self.assertEqual(state["status"], "completed")

    def test_failure_preserves_output_and_traceback(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            code = run_logged_process([sys.executable, "-u", "-c", "print('候选 1 已输出'); raise RuntimeError('fixture failure')"], folder)
            self.assertNotEqual(code, 0)
            log = (Path(folder)/"training.log").read_text()
            self.assertIn("候选 1 已输出", log)
            self.assertIn("Traceback", log)
            self.assertIn("fixture failure", log)
            self.assertEqual(json.loads((Path(folder)/"progress.json").read_text())["status"], "failed")

    def test_launch_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_logged_process([str(Path(folder)/"missing-executable")], folder), 1)
            self.assertIn("FileNotFoundError", (Path(folder)/"training.log").read_text())

    @unittest.skipUnless(os.name == "posix", "Linux/WSL 进程锁与 SIGINT")
    def test_same_directory_cannot_be_overwritten_by_concurrent_training(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"training.log"
            log.write_text("keep")
            with _output_lock(Path(folder)), self.assertRaisesRegex(RuntimeError, "已有训练进程"):
                run_logged_process([sys.executable, "-c", "pass"], folder)
            self.assertEqual(log.read_text(), "keep")

    @unittest.skipUnless(os.name == "posix", "Linux/WSL SIGINT")
    def test_live_log_and_progress_survive_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            worker = ("import json,os,time; from pathlib import Path; "
                f"p=Path({folder!r})/'progress.json'; "
                "p.write_text(json.dumps({'run_id':os.environ['OPD_TRAINING_RUN_ID'],'completed_rounds':1,'iterations':[{'round':1}]})); "
                "print('round-one-saved',flush=True); time.sleep(30)")
            wrapper = "from tabular_data.run_logging import run_logged_process; import sys; sys.exit(run_logged_process(" + repr([sys.executable,"-u","-c",worker]) + "," + repr(folder) + "))"
            proc = subprocess.Popen([sys.executable,"-u","-c",wrapper], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    if (path/"training.log").exists() and "round-one-saved" in (path/"training.log").read_text():
                        break
                    if proc.poll() is not None:
                        self.fail("日志进程提前退出")
                    time.sleep(.02)
                else:
                    self.fail("训练输出未实时写入")
                proc.send_signal(signal.SIGINT)
                self.assertEqual(proc.wait(timeout=8), 130)
                self.assertIn("训练已中断", (path/"training.log").read_text())
                progress = json.loads((path/"progress.json").read_text())
                self.assertEqual(progress["completed_rounds"], 1)
                self.assertEqual(progress["status"], "interrupted")
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    unittest.main()
