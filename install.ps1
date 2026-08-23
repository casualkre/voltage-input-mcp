# One command from a fresh clone to a working install, on Windows.
#
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#
# Safe to re-run: every step checks before acting.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Head($t) { Write-Host $t -ForegroundColor White }
function Ok($t)   { Write-Host "  [ok] $t" -ForegroundColor Green }
function No($t)   { Write-Host "  [--] $t" -ForegroundColor Red }
function Cmd($t)  { Write-Host "      $t" -ForegroundColor Cyan }

Write-Host @"
 ##   ##  #####  ##  ######  #####   ####  #####
 ##   ## ##   ## ##    ##   ##   ## ##     ##
 ##   ## ##   ## ##    ##   ####### ##  ## ####
  ## ##  ##   ## ##    ##   ##   ## ##   ## ##
   ###    #####  #####  ##  ##   ##  ##### #####
"@ -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------- 1. python
Head "1/4  Python"
$py = $null
foreach ($c in @("py -3", "python", "python3")) {
    $exe, $arg = $c.Split(" ", 2)
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $v = & $exe $arg -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($v) { $py = $c; break }
    }
}
if (-not $py) {
    No "Python not found. Install 3.11 or newer:"
    Cmd "winget install Python.Python.3.12"
    Cmd "  ...then re-run this script."
    exit 1
}
$exe, $arg = $py.Split(" ", 2)
$ver = & $exe $arg -c "import sys;print('%d.%d'%sys.version_info[:2])"
$okv = & $exe $arg -c "import sys;print(1 if sys.version_info>=(3,11) else 0)"
if ($okv -ne "1") { No "Python $ver is too old; 3.11+ required"; exit 1 }
Ok "python $ver"

# ------------------------------------------------------------------ 2. venv
Head "2/4  Python environment"
if (-not (Test-Path ".venv")) {
    & $exe $arg -m venv .venv
    if (-not (Test-Path ".venv")) { No "venv creation failed"; exit 1 }
    Ok "created .venv"
} else { Ok ".venv exists" }

$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip 2>&1 | Out-Null
& $vpy -m pip install -q -e . 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -eq 0) { Ok "package installed" } else { No "pip install failed"; exit 1 }

# ------------------------------------------------------------------ 3. PATH
Head "3/4  PATH"
$scripts = (Resolve-Path ".\.venv\Scripts").Path
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$scripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$scripts", "User")
    Ok "added to your user PATH"
    Write-Host "      Open a new terminal for this to take effect." -ForegroundColor Yellow
} else { Ok "already on PATH" }

# ------------------------------------------------------------------ 4. input
Head "4/4  Input and capture"
# Nothing to install: SendInput and GDI BitBlt are part of Windows. The one caveat is
# UIPI -- a non-elevated process cannot drive an elevated window, and that fails
# silently rather than erroring.
Ok "SendInput and GDI capture need no setup"
Write-Host "      Note: input cannot reach windows owned by an elevated process." -ForegroundColor Yellow
Write-Host "      If a burst seems to do nothing over an admin window, that is why." -ForegroundColor Yellow

Write-Host ""
Head "Next -- models and connecting to your AI client:"
Write-Host ""
Cmd "voltage setup"
Write-Host ""
Write-Host "  That downloads the models and registers the server."
Write-Host "  You will also need llama-server.exe on your PATH -- a CUDA build from:"
Cmd "https://github.com/ggml-org/llama.cpp/releases"
Write-Host ""
Cmd "voltage"
Write-Host ""
