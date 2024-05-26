@echo off
REM Set the project directory to the script's directory
set SCRIPT_DIR=%~dp0

REM Ensure the virtual environment is activated
call %SCRIPT_DIR%venv\Scripts\activate

REM Run the PowerShell script to launch the Gradio app with arguments
powershell.exe -File %SCRIPT_DIR%rungradioapp.ps1 -checkpointDir ".\checkpoints" -filePath ".\path\to\your\audio_file.wav"

REM Deactivate the virtual environment after running the script
deactivate
