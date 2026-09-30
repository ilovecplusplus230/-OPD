@echo off
chcp 65001 >nul
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_website_windows.ps1"
if errorlevel 1 (
    echo.
    echo Website startup failed. Please check the message above.
    pause
    exit /b 1
)
exit /b 0
