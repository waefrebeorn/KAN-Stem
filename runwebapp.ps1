param (
    [string]$checkpointDir = ".\checkpoints",
    [string]$filePath = ".\path\to\your\audio_file.wav",
    [string]$pythonPath
)

try {
    # Get the current script directory
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

    # Set the project directory relative to the script location
    $projectDir = Join-Path -Path $scriptDir -ChildPath "KAN-Stem"

    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "$projectDir\src;$projectDir"

    # Print PYTHONPATH and working directory
    Write-Output "PYTHONPATH: $env:PYTHONPATH"
    Write-Output "Current working directory: $(Get-Location)"

    # If a custom Python path is provided, use it
    if ($pythonPath) {
        $pythonExe = $pythonPath
    } else {
        # Find Python executable in the virtual environment
        $pythonExe = Join-Path -Path $projectDir -ChildPath "venv\Scripts\python.exe"

        if (-Not (Test-Path $pythonExe)) {
            Write-Error "Python executable not found at $pythonExe"
            exit 1
        }
    }

    # Run the gradio_app.py script using the found Python executable
    Write-Output "Running Gradio app with $pythonExe ..."
    & $pythonExe "$projectDir\src\gradio_app.py" -checkpointDir $checkpointDir -filePath $filePath
} catch {
    Write-Error "An error occurred: $_"
    pause
    exit 1
}

# Pause to see any output or error messages
Write-Output "Press Enter to continue..."
[System.Console]::ReadLine() | Out-Null
