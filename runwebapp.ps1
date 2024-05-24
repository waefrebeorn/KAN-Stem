# Explicitly set the project directory
$projectDir = "C:\projects\KAN-Stem"
Set-Location -Path $projectDir

# Set the PYTHONPATH to include the src directory
$env:PYTHONPATH = "$projectDir\src;$projectDir"

# Print PYTHONPATH and working directory
Write-Output "PYTHONPATH: $env:PYTHONPATH"
Write-Output "Current working directory: $(Get-Location)"


# Run the gradio_app.py script
Write-Output "Running Gradio app..."
python .\src\gradio_app.py

# Pause to see any output or error messages
Write-Output "Press Enter to continue..."
[System.Console]::ReadLine() | Out-Null
