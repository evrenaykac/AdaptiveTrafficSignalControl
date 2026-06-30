@echo off
cd /d "%~dp0..\.."

echo ====================================================
echo SUMO Single Intersection - DDPG Benchmark Automator
echo ====================================================

echo [1/4] Testing Ollama Connection...
curl -s http://localhost:11434/api/tags >nul
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is NOT running! 
    echo Please start the Ollama application and try again.
    pause
    exit /b
)
echo [OK] Ollama is running.

set PYTHONPATH=%cd%

:: Generate safe timestamp format YYYYMMDD_HHMMSS
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set dt=%%I
set TIMESTAMP=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%

echo.
echo ====================================================
echo [2/4] PHASE 1: Baseline (Static Traffic Light)
echo ====================================================
python ddpg_workspace\scripts\train_ddpg_single.py --mode baseline --out ddpg_workspace\results\Baseline_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo [3/4] PHASE 2: Standard DDPG (Fixed Reward Weights)
echo ====================================================
python ddpg_workspace\scripts\train_ddpg_single.py --mode ddpg --out ddpg_workspace\results\DDPG_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo [4/4] PHASE 3: DDPG + LLM (Dynamic Reward Tuning)
echo ====================================================
python ddpg_workspace\scripts\train_ddpg_single.py --mode ddpg_llm --out ddpg_workspace\results\DDPGAndLLM_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo All tests completed successfully!
echo Results saved to: ddpg_workspace\results\
echo - Baseline.csv
echo - DDPG.csv
echo - DDPGAndLLM.csv
echo ====================================================
pause
