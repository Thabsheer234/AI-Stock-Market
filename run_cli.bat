@echo off
title FinAgent AI - CLI Interactive Loop
echo =====================================================================
echo  Starting FinAgent AI - CLI Interactive Financial Desk
echo =====================================================================
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py
pause
