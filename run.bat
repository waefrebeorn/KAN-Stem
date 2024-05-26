@echo off
set SCRIPT_DIR=%~dp0
call %SCRIPT_DIR%venv\Scripts\activate

powershell.exe -NoLogo -NoExit -Command "& { %SCRIPT_DIR%rungradioapp.ps1 -checkpointDir '.\checkpoints' -filePath '.\path\to\your\audio_file.wav' }"
