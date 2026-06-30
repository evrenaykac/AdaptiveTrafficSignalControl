@echo off
cd /d "%~dp0..\.."

echo ====================================================
echo SUMO Single Intersection - A2C Benchmark Automator
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
python a2c_workspace\scripts\train_a2c_single.py --mode baseline --out a2c_workspace\results\Baseline_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo [3/4] PHASE 2: Standard A2C (Fixed Reward Weights)
echo ====================================================
python a2c_workspace\scripts\train_a2c_single.py --mode a2c --out a2c_workspace\results\A2C_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo [4/4] PHASE 3: A2C + LLM (Dynamic Reward Tuning)
echo ====================================================
python a2c_workspace\scripts\train_a2c_single.py --mode a2c_llm --out a2c_workspace\results\A2CAndLLM_%TIMESTAMP%.csv --episodes 10 --duration 1000

echo.
echo ====================================================
echo All tests completed successfully!
echo Results saved to: a2c_workspace\results\
echo - Baseline.csv
echo - A2C.csv
echo - A2CAndLLM.csv
echo ====================================================
pause
