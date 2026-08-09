@echo off
setlocal
title Export RVC FAISS index for Unity

cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "RVC_DIR=%ROOT_DIR%Retrieval-based-Voice-Conversion-WebUI"
set "PYTHON=%RVC_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%ROOT_DIR%export_rvc_index_unity.py"

set "OUTPUT_DIR=%ROOT_DIR%sentis_models"
set "OUTPUT=%OUTPUT_DIR%\rvc_retrieval_index.bytes"

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

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo.
echo ============================================================
echo Export RVC retrieval index for Unity
echo ============================================================
echo.
echo You can:
echo   1. Double-click this BAT if there is exactly one .index under logs
echo   2. Drag the desired .index file onto this BAT
echo.

if "%~1"=="" (
    "%PYTHON%" "%SCRIPT%" ^
        --rvc-dir "%RVC_DIR%" ^
        --output "%OUTPUT%"
) else (
    "%PYTHON%" "%SCRIPT%" ^
        --rvc-dir "%RVC_DIR%" ^
        --index "%~1" ^
        --output "%OUTPUT%"
)

set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Index export failed with code %EXIT_CODE%.
    pause
    exit /b %EXIT_CODE%
)

if not exist "%OUTPUT%" (
    echo.
    echo [ERROR] Output was not created:
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
echo Copy this .bytes file into Unity Assets and assign it to:
echo   Retrieval Index Asset
echo.

pause
exit /b 0
