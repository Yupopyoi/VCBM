@echo off
cd /d "%~dp0"

".\VoxCPM\.venv\Scripts\python.exe" ".\VoxCPMService\server.py"

pause