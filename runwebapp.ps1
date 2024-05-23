# Explicitly set the project directory
$projectDir = "C:\projects\KAN-Stem"
Set-Location -Path $projectDir

# Ensure the virtual environment activation script exists and activate it
$venvPath = Join-Path -Path $projectDir -ChildPath "venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript

    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "$projectDir\src;$projectDir"

    # Run the gradio_app.py script
    Write-Output "Running Gradio app..."
    python .\src\gradio_app.py
} else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}

# Pause to see any output or error messages
Write-Output "Press Enter to continue..."
[System.Console]::ReadLine() | Out-Null
