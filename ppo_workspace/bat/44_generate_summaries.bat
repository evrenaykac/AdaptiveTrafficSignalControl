@echo off
setlocal
cd /d "%~dp0..\.."

echo --- Generating PPO and PPO+LLM Summaries ---
py ppo_workspace\scripts\generate_summaries.py

if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo Summaries generated successfully in ppo_workspace\results\
pause
