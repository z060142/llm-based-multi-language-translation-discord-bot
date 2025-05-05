@echo off
REM Discord Translation Bot Install Script for Windows
REM This script sets up the virtual environment and installs dependencies

echo Discord Translation Bot Setup Script for Windows
echo ==============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Display Python version
echo Found Python version:
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install "discord.py>=2.3.2"
pip install "openai>=1.40.3"
pip install "pyyaml>=6.0.1"
pip install "aiosqlite>=0.19.0"
pip install "aiohttp>=3.9.5"

REM Create start scripts
echo.
echo Creating start scripts...

REM Create start script for bot only
echo @echo off > start-bot.bat
echo cd /d "%%~dp0" >> start-bot.bat
echo call venv\Scripts\activate.bat >> start-bot.bat
echo python bot.py >> start-bot.bat
echo pause >> start-bot.bat

REM Create start script for UI
echo @echo off > start-ui.bat
echo cd /d "%%~dp0" >> start-ui.bat
echo call venv\Scripts\activate.bat >> start-ui.bat
echo python bot_ui.py >> start-ui.bat
echo pause >> start-ui.bat

echo.
echo Installation completed successfully!
echo.
echo Available start scripts:
echo   start-bot.bat          - Start bot only
echo   start-ui.bat           - Start UI (can control bot from within UI)
echo.
echo Next steps:
echo 1. Configure your settings using the UI by double-clicking start-ui.bat
echo 2. Use the UI to start/stop the bot as needed
echo.
echo Note: If this is your first time, the bot will create a template config.yaml
echo       You'll need to fill in your Discord token and other API keys
echo.
pause