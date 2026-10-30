@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [Portfolio Risk Lab] 缺少项目 Python 环境，请先按 README 安装。
  pause
  exit /b 1
)
if not exist ".cache\tmp" mkdir ".cache\tmp"
if not exist ".cache\tmp" (
  echo [Portfolio Risk Lab] 无法创建项目临时目录 .cache\tmp。
  pause
  exit /b 1
)
set "TEMP=%~dp0.cache\tmp"
set "TMP=%TEMP%"
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats=false %*
set "PORTFOLIO_EXIT_CODE=%ERRORLEVEL%"
if not "%PORTFOLIO_EXIT_CODE%"=="0" pause
exit /b %PORTFOLIO_EXIT_CODE%
