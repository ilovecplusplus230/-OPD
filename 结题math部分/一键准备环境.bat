@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
echo.
echo 环境准备完成。请参考 README.md 配置 .env 并下载数据。
pause
