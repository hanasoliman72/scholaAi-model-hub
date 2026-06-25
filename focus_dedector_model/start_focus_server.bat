@echo off
echo Starting ScholaAi Focus Server on all interfaces (LAN-accessible)...
cd /d "%~dp0"
call venv\Scripts\activate.bat
uvicorn focus_server:app --host 0.0.0.0 --port 8000 --reload
pause
