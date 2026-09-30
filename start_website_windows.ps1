param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot

try {
    # Reuse the project's WSL distribution when opened from Explorer.
    if ($projectDir -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)\\(.*)$') {
        $distribution = $Matches[1]
        $linuxDir = '/' + $Matches[2].Replace('\', '/')
        & wsl.exe --distribution $distribution --cd $linuxDir --exec python3 start_website.py --no-browser
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 (Join-Path $projectDir 'start_website.py') --no-browser
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python (Join-Path $projectDir 'start_website.py') --no-browser
    } else {
        throw 'Python not found. Install Python and requirements.txt first.'
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Website startup failed. See the message above.'
    }
    if (-not $NoBrowser) {
        Start-Process 'http://127.0.0.1:5000/'
    }
    exit 0
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
