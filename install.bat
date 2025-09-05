@echo off
cd /d "%~dp0"

:: Install UV if not already present
where uv >nul 2>nul || (
    echo Installing UV (Python package installer)...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo Restarting to use UV...
    call "%~f0"
    exit /b
)

:: REM :: Check for Git
:: REM where git >nul 2>nul || (
:: REM     echo ERROR: Git is not installed
:: REM     echo Please install Git from: https://git-scm.com/download/win
:: REM     pause
:: REM     exit /b 1
:: REM )

:: REM :: Setup and run project
:: REM if not exist FGD_presentation git clone https://github.com/Osetrovie-Podeba/FGD_presentation
:: REM cd FGD_presentation
uv sync
call .venv\Scripts\activate.bat
uvicorn app:app --reload --host 0.0.0.0 --port 8000
