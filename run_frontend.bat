@echo off
title FinAgent AI - Autonomous Stock Market Desk
echo =====================================================================
echo  Starting FinAgent AI - Autonomous Stock Market Financial Terminal
echo =====================================================================
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" (
    set "VENV_DIR=%~dp0venv"
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    set "STREAMLIT_EXE=%~dp0venv\Scripts\streamlit.exe"
) else if exist "C:\Stock_market_agentic_ai\venv\Scripts\python.exe" (
    set "VENV_DIR=C:\Stock_market_agentic_ai\venv"
    set "PYTHON_EXE=C:\Stock_market_agentic_ai\venv\Scripts\python.exe"
    set "STREAMLIT_EXE=C:\Stock_market_agentic_ai\venv\Scripts\streamlit.exe"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Virtual environment not found at .\venv or C:\Stock_market_agentic_ai\venv
    echo Please make sure the virtual environment is set up.
    pause
    exit /b 1
)

echo Activating Python 3.11 virtual environment (%VENV_DIR%)...
call "%VENV_DIR%\Scripts\activate.bat"

echo Launching Streamlit Front-End Interface...
echo App will open at: http://localhost:8501
"%STREAMLIT_EXE%" run app.py

pause
