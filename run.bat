@echo off
REM Get the directory of the script
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Set the environment variable to suppress oneDNN custom operations messages
set TF_ENABLE_ONEDNN_OPTS=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

REM Check if Python is installed and get the path
for /f "tokens=*" %%p in ('where python') do (
    set PYTHON_PATH=%%p
    goto :foundPython
)

echo Python not found in PATH. Please install Python or add it to your PATH.
exit /b 1

:foundPython
echo Python found at %PYTHON_PATH%

REM Activate the virtual environment
call %SCRIPT_DIR%venv\Scripts\activate

REM Ensure the path to the PowerShell script is correctly formatted
set PS_SCRIPT_PATH=%SCRIPT_DIR%rungradioapp.ps1

REM Run the PowerShell script with arguments
powershell.exe -NoLogo -NoProfile -Command "& { . '%PS_SCRIPT_PATH%' -checkpointDir '.\checkpoints' -filePath '.\path\to\your\audio_file.wav' -pythonPath '%PYTHON_PATH%' }"

REM End of script
pause