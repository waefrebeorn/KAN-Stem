# Explicitly set the project directory
$projectDir = "C:\projects\KAN-Stem"
Set-Location -Path $projectDir

# Ensure the virtual environment activation script exists and activate it
$venvPath = "C:\projects\KAN-Stem\venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript

    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "C:\projects\KAN-Stem\src;C:\projects\KAN-Stem"

    # Run the gradio_app.py script
    python .\src\testcudaapp.py

    Write-Output "Finished running testcudaapp.py. Press Enter to exit..."
    [System.Console]::ReadLine() | Out-Null

} else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
    Write-Output "Press Enter to exit..."
    [System.Console]::ReadLine() | Out-Null
}
