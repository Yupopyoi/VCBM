@echo off
setlocal
title Export RVC v2 F0 + HuBERT for Sentis

cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "RVC_DIR=%ROOT_DIR%Retrieval-based-Voice-Conversion-WebUI"

set "RVC_SCRIPT_PATH=%ROOT_DIR%export_rvc_v2_f0_sentis.py"
set "HUBERT_SCRIPT_PATH=%ROOT_DIR%export_hubert_rvc_v2_transformers.py"

set "MODEL_PATH=%RVC_DIR%\assets\weights\god.pth"

rem Current RVC uses a Transformers-format directory, not hubert_base.pt.
set "HUBERT_MODEL_DIR=%RVC_DIR%\assets\hubert_base"

set "PYTHON_PATH=%RVC_DIR%\.venv\Scripts\python.exe"

set "OUTPUT_DIR=%ROOT_DIR%sentis_models"
set "RVC_OUTPUT_FILE=%OUTPUT_DIR%\rvc_v2_f0_fixed.onnx"
set "HUBERT_OUTPUT_FILE=%OUTPUT_DIR%\hubert_rvc_v2_2s.onnx"

if not exist "%OUTPUT_DIR%" (
    echo [INFO] Creating output folder...
    mkdir "%OUTPUT_DIR%"
)

echo.
echo ============================================================
echo RVC v2 F0 + HuBERT Sentis Export
echo ============================================================
echo.

if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python was not found:
    echo %PYTHON_PATH%
    pause
    exit /b 1
)

echo [DEBUG] Python:
"%PYTHON_PATH%" --version

if errorlevel 1 (
    echo [ERROR] Python could not be started.
    pause
    exit /b 1
)

echo.

if not exist "%MODEL_PATH%" (
    echo [ERROR] RVC model was not found:
    echo %MODEL_PATH%
    pause
    exit /b 1
)

if not exist "%RVC_SCRIPT_PATH%" (
    echo [ERROR] RVC exporter was not found:
    echo %RVC_SCRIPT_PATH%
    pause
    exit /b 1
)

if not exist "%HUBERT_MODEL_DIR%\config.json" (
    echo [ERROR] HuBERT config.json was not found:
    echo %HUBERT_MODEL_DIR%\config.json
    pause
    exit /b 1
)

if not exist "%HUBERT_MODEL_DIR%\preprocessor_config.json" (
    echo [ERROR] HuBERT preprocessor_config.json was not found:
    echo %HUBERT_MODEL_DIR%\preprocessor_config.json
    pause
    exit /b 1
)

if not exist "%HUBERT_MODEL_DIR%\pytorch_model.bin" (
    echo [ERROR] HuBERT pytorch_model.bin was not found:
    echo %HUBERT_MODEL_DIR%\pytorch_model.bin
    pause
    exit /b 1
)

if not exist "%HUBERT_SCRIPT_PATH%" (
    echo [ERROR] HuBERT exporter was not found:
    echo %HUBERT_SCRIPT_PATH%
    pause
    exit /b 1
)

echo ============================================================
echo [1/2] Exporting RVC v2 F0 Generator
echo ============================================================
echo.

pushd "%RVC_DIR%"

"%PYTHON_PATH%" "%RVC_SCRIPT_PATH%" ^
    --model "%MODEL_PATH%" ^
    --output "%RVC_OUTPUT_FILE%" ^
    --frames 200 ^
    --speaker-id 0

set "RVC_EXIT_CODE=%ERRORLEVEL%"

popd

if not "%RVC_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] RVC export failed with code %RVC_EXIT_CODE%.
    pause
    exit /b %RVC_EXIT_CODE%
)

if not exist "%RVC_OUTPUT_FILE%" (
    echo.
    echo [ERROR] RVC ONNX was not created:
    echo %RVC_OUTPUT_FILE%
    pause
    exit /b 1
)

echo.
echo [OK] RVC ONNX:
echo %RVC_OUTPUT_FILE%
echo.

echo ============================================================
echo [2/2] Exporting HuBERT
echo ============================================================
echo.

pushd "%RVC_DIR%"

"%PYTHON_PATH%" "%HUBERT_SCRIPT_PATH%" ^
    "%HUBERT_MODEL_DIR%" ^
    "%HUBERT_OUTPUT_FILE%"

set "HUBERT_EXIT_CODE=%ERRORLEVEL%"

popd

if not "%HUBERT_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] HuBERT export failed with code %HUBERT_EXIT_CODE%.
    pause
    exit /b %HUBERT_EXIT_CODE%
)

if not exist "%HUBERT_OUTPUT_FILE%" (
    echo.
    echo [ERROR] HuBERT ONNX was not created:
    echo %HUBERT_OUTPUT_FILE%
    pause
    exit /b 1
)

echo.
echo [OK] HuBERT ONNX:
echo %HUBERT_OUTPUT_FILE%
echo.

echo ============================================================
echo [SUCCESS] All Sentis models were exported.
echo ============================================================
echo.
echo RVC:
echo   %RVC_OUTPUT_FILE%
echo.
echo HuBERT:
echo   %HUBERT_OUTPUT_FILE%
echo.

pause
exit /b 0
