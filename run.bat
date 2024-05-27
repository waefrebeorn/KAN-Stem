@echo off
REM Get the directory of the script
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Activate the virtual environment
call %SCRIPT_DIR%venv\Scripts\activate

REM Run the PowerShell script with arguments
powershell.exe -NoLogo -NoExit -Command "& { %SCRIPT_DIR%rungradioapp.ps1 -checkpointDir '.\checkpoints' -filePath '.\path\to\your\audio_file.wav' }"

pause
