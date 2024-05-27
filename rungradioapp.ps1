param(
    [string]$checkpointDir = "./checkpoints",
    [string]$filePath
)

# Set the project directory to the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

Write-Output "Project directory set to: $scriptDir"

# Ensure the virtual environment activation script exists and activate it
$venvPath = Join-Path -Path $scriptDir -ChildPath "venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    Write-Output "Activating virtual environment..."
    & $activateScript
    Write-Output "Virtual environment activated."

    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "$scriptDir\src;$scriptDir"
    Write-Output "PYTHONPATH set to: $env:PYTHONPATH"

    # Run the gradio_app.py script with arguments
    $pythonScript = Join-Path -Path $scriptDir -ChildPath "src\gradio_app.py"
    Write-Output "Running Gradio app with checkpoint directory: $checkpointDir and file path: $filePath"
    python $pythonScript --checkpoint_dir "$checkpointDir" --file_path "$filePath"
}
else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}
