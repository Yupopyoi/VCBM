@echo off
setlocal
title Export Sentis-compatible RMVPE

cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "RVC_DIR=%ROOT_DIR%Retrieval-based-Voice-Conversion-WebUI"
set "PYTHON=%RVC_DIR%\.venv\Scripts\python.exe"

set "SCRIPT=%ROOT_DIR%export_rmvpe_sentis_unrolled_fixed3.py"
set "RMVPE_PT=%RVC_DIR%\assets\rmvpe\rmvpe.pt"

set "OUTPUT_DIR=%ROOT_DIR%sentis_models"
set "OUTPUT=%OUTPUT_DIR%\rmvpe_sentis_2s_padded224.onnx"

echo.
echo ============================================================
echo Export Sentis-compatible RMVPE
echo ============================================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Python was not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Export script was not found:
    echo %SCRIPT%
    pause
    exit /b 1
)

if not exist "%RVC_DIR%\infer\lib\rmvpe.py" (
    echo [WARN] Standard RMVPE source path was not found:
    echo %RVC_DIR%\infer\lib\rmvpe.py
    echo [WARN] The Python exporter will search the RVC tree automatically.
    echo.
)

if not exist "%RMVPE_PT%" (
    echo [ERROR] rmvpe.pt was not found:
    echo %RMVPE_PT%
    echo.
    echo This exporter needs the PyTorch RMVPE checkpoint,
    echo not the official rmvpe.onnx.
    pause
    exit /b 1
)

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo [INFO] Python:
"%PYTHON%" --version
echo.

"%PYTHON%" -c "import onnx" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing onnx...
    "%PYTHON%" -m pip install onnx
    if errorlevel 1 (
        echo [ERROR] Failed to install onnx.
        pause
        exit /b 1
    )
)

pushd "%RVC_DIR%"

"%PYTHON%" "%SCRIPT%" ^
    --rvc-dir "%RVC_DIR%" ^
    --checkpoint "%RMVPE_PT%" ^
    --output "%OUTPUT%" ^
    --frames 224

set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] RMVPE Sentis export failed with code %EXIT_CODE%.
    pause
    exit /b %EXIT_CODE%
)

if not exist "%OUTPUT%" (
    echo [ERROR] ONNX was not created:
    echo %OUTPUT%
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS]
echo ============================================================
echo %OUTPUT%
echo.
echo Import THIS model into Unity instead of the official rmvpe.onnx.
echo.

pause
exit /b 0
