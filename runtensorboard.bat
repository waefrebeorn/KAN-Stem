@echo off
set SCRIPT_DIR=%~dp0
call %SCRIPT_DIR%venv\Scripts\activate

powershell.exe -NoLogo -NoExit -Command "& { %SCRIPT_DIR%tensorboard.ps1 }"
