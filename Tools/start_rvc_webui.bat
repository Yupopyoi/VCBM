@echo off
setlocal

title RVC WebUI Launcher
echo ========================================
echo RVC WebUI launcher started
echo ========================================
echo.
timeout /t 1 /nobreak >nul

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%Retrieval-based-Voice-Conversion-WebUI\webui.py" (
    set "RVC_DIR=%SCRIPT_DIR%Retrieval-based-Voice-Conversion-WebUI"
) else if exist "%SCRIPT_DIR%webui.py" (
    set "RVC_DIR=%SCRIPT_DIR%"
) else (
    echo [ERROR] Retrieval-based-Voice-Conversion-WebUI was not found.
    echo Place this BAT next to the repository folder or inside the repository folder.
    pause
    exit /b 1
)

pushd "%RVC_DIR%"

echo [1/4] Repository found

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe was not found.
    echo Create the virtual environment before running this BAT.
    popd
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
timeout /t 1 /nobreak >nul

call ".venv\Scripts\activate.bat"

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo       Virtual environment activated.
echo.

set "PYTHONSAFEPATH=1"
set "PYTHONPATH=%CD%"
set "PYTHONNOUSERSITE=1"

echo [3/4] Environment variables configured
echo       PYTHONSAFEPATH=%PYTHONSAFEPATH%
echo       PYTHONPATH=%PYTHONPATH%
echo       PYTHONNOUSERSITE=%PYTHONNOUSERSITE%
echo.

echo [4/4] Starting RVC WebUI...
echo.

".venv\Scripts\python.exe" webui.py
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] webui.py exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
