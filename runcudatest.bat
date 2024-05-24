@echo off
REM Ensure the virtual environment is activated
call venv\Scripts\activate

REM Run the PowerShell script and wait for it to complete
powershell.exe -NoExit -File .\runcudatest.ps1

REM Deactivate the virtual environment after running the script
deactivate
