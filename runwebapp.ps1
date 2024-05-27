# Get the current script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set the project directory relative to the script location
$projectDir = Join-Path -Path $scriptDir -ChildPath "KAN-Stem"

# Set the PYTHONPATH to include the src directory
$env:PYTHONPATH = "$projectDir\src;$projectDir"

# Print PYTHONPATH and working directory
Write-Output "PYTHONPATH: $env:PYTHONPATH"
Write-Output "Current working directory: $(Get-Location)"

# Run the gradio_app.py script
Write-Output "Running Gradio app..."
python "$projectDir\src\gradio_app.py"

# Pause to see any output or error messages
Write-Output "Press Enter to continue..."
[System.Console]::ReadLine() | Out-Null
