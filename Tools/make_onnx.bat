@echo off
setlocal
title Sentis Test ONNX Export

cd /d "%~dp0"

echo ========================================
echo Sentis test ONNX export
echo ========================================
echo.

if not exist "Retrieval-based-Voice-Conversion-WebUI\webui.py" (
    echo [ERROR] Retrieval-based-Voice-Conversion-WebUI was not found.
    echo Place this BAT next to the repository folder.
    pause
    exit /b 1
)

if not exist "export_sentis_test.py" (
    echo [ERROR] export_sentis_test.py was not found.
    echo Place export_sentis_test.py next to this BAT.
    pause
    exit /b 1
)

cd /d "Retrieval-based-Voice-Conversion-WebUI"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe was not found.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [2/3] Installing ONNX...
".venv\Scripts\python.exe" -m pip install onnx
if errorlevel 1 (
    echo [ERROR] Failed to install ONNX.
    pause
    exit /b 1
)

echo [3/3] Exporting Sentis test model...
".venv\Scripts\python.exe" "..\export_sentis_test.py"
if errorlevel 1 (
    echo [ERROR] Failed to export the ONNX model.
    pause
    exit /b 1
)

echo.
echo [OK] Export completed.
echo Output: sentis_test.onnx
echo.

pause
exit /b 0
