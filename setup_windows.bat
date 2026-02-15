@echo off
title reMBG Installer
cls

echo ========================================================
echo      reMBG - AUTOMATIC SETUP
echo      (Windows Edition)
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python from the Microsoft Store or python.org.
    pause
    exit
)

if not exist "venv" (
    echo [1/3] Creating Python virtual environment (venv)...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

echo [2/3] Installing dependencies (reMBG requires this)...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

echo [3/3] Creating "reMBG" Shortcut on Desktop...

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "TARGET_PYTHON=%SCRIPT_DIR%\venv\Scripts\pythonw.exe"
set "TARGET_SCRIPT=%SCRIPT_DIR%\src\gui.py"

set "ICON_PATH=%SCRIPT_DIR%\assets\icon.ico"
set "SHORTCUT_NAME=%UserProfile%\Desktop\reMBG.lnk"

if exist "%SHORTCUT_NAME%" del "%SHORTCUT_NAME%"

(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%SHORTCUT_NAME%"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%TARGET_PYTHON%"
echo oLink.Arguments = """%TARGET_SCRIPT%"""
echo oLink.WorkingDirectory = "%SCRIPT_DIR%"
echo oLink.Description = "reMBG - Offline Background Remover"
echo oLink.IconLocation = "%ICON_PATH%"  
echo oLink.Save
) > CreateShortcut.vbs

cscript /nologo CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo ========================================================
echo      SUCCESS! reMBG IS READY.
echo ========================================================
echo Check your Desktop for the "reMBG" icon.
pause