@echo off
REM Ensure the virtual environment is activated
call venv\Scripts\activate

REM Run the PowerShell script to launch the Gradio app
powershell.exe -File .\runwebapp.ps1

REM Deactivate the virtual environment after running the script
deactivate