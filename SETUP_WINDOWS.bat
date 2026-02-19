@echo off
setlocal EnableDelayedExpansion
title  Installer
cls

cd /d "%~dp0"

echo ========================================================
echo      pvBG - AUTOMATIC SETUP
echo      (Windows Edition)
echo ========================================================
echo.

:: ---------------------------------------------------------
:: CHECK PYTHON
:: ---------------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b
)

echo [OK] Python detected.
echo.

:: ---------------------------------------------------------
:: CREATE VIRTUAL ENVIRONMENT
:: ---------------------------------------------------------
if not exist "venv" (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b
    )
) else (
    echo [1/3] Virtual environment already exists.
)

echo.

:: ---------------------------------------------------------
:: INSTALL DEPENDENCIES
:: ---------------------------------------------------------
echo [2/3] Installing dependencies...
call "venv\Scripts\activate.bat"
pip install --upgrade pip >nul
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b
)

echo.

:: ---------------------------------------------------------
:: CREATE DESKTOP SHORTCUT
:: ---------------------------------------------------------
echo [3/3] Creating Desktop shortcut...

set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

set "TARGET_PYTHON=!SCRIPT_DIR!\venv\Scripts\pythonw.exe"
set "TARGET_SCRIPT=!SCRIPT_DIR!\src\gui.py"
set "ICON_PATH=!SCRIPT_DIR!\assets\icon.ico"
set "SHORTCUT_NAME=%USERPROFILE%\Desktop\pvBG.lnk"

if exist "!SHORTCUT_NAME!" del "!SHORTCUT_NAME!"

(
echo Set oWS = WScript.CreateObject("WScript.Shell"^)
echo sLinkFile = "!SHORTCUT_NAME!"
echo Set oLink = oWS.CreateShortcut(sLinkFile^)
echo oLink.TargetPath = "!TARGET_PYTHON!"
echo oLink.Arguments = """!TARGET_SCRIPT!"""
echo oLink.WorkingDirectory = "!SCRIPT_DIR!"
echo oLink.Description = "pvBG - Offline Background Remover"
echo oLink.IconLocation = "!ICON_PATH!"
echo oLink.Save
) > CreateShortcut.vbs

cscript /nologo CreateShortcut.vbs >nul
del CreateShortcut.vbs

echo.
echo ========================================================
echo      SUCCESS! pvBG IS READY.
echo ========================================================
echo Check your Desktop for the "pvBG" icon.
echo.
pause

endlocal