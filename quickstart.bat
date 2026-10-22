@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo RestoreAI SRGAN - Quickstart
echo ==========================================

if not exist "app.py" (
    echo ERROR: app.py was not found. Run this script from the project folder.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a local Python virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 py -3.11 -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Python 3.11 was unavailable or failed. Trying the default Python installation...
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Could not create .venv using the Python launcher or default Python.
    echo Install a supported Python version and ensure python.exe is on PATH.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing project dependencies. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: pip upgrade failed.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed. Check your internet connection and requirements.txt.
        pause
        exit /b 1
    )
)

echo Starting RestoreAI. Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo ERROR: RestoreAI did not start successfully.
    pause
    exit /b 1
)

endlocal
