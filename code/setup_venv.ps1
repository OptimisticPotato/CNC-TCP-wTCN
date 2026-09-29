<#
  TCN CNC->TCP : virtual-environment bootstrap  (Windows / PowerShell)

      powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1

  Creates .venv next to this script, installs CUDA torch + requirements, and
  verifies every step. Options:

      -Python "C:\path\to\python.exe"   use a specific interpreter
      -CudaTag cpu                      CPU-only torch (no NVIDIA GPU here)
      -CudaTag cu124                    newer CUDA build
      -Force                            delete an existing .venv first

  Note on PowerShell: a failing .exe does NOT raise an error, not even with
  $ErrorActionPreference = "Stop". Every external call below is therefore
  followed by an explicit exit-code check -- otherwise a Microsoft Store python
  stub silently does nothing and the script cheerfully reports success.
#>
param(
    # cu121 works on every RTX 30xx driver >= 530. "cpu" for a machine
    # without an NVIDIA GPU.
    [string]$CudaTag = "cu121",
    [string]$Python = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"

function Fail($msg) {
    Write-Host ""
    Write-Host "FAILED: $msg" -ForegroundColor Red
    exit 1
}

function Test-Interpreter($exe) {
    if (-not $exe) { return $false }
    if ($exe -like "*WindowsApps*") { return $false }   # Microsoft Store stub
    try { $v = & $exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null }
    catch { return $false }
    if ($LASTEXITCODE -ne 0 -or -not $v) { return $false }
    return $true
}

# --------------------------------------------------------------------------- #
# 1. find a usable interpreter
# --------------------------------------------------------------------------- #
Write-Host "[1/5] looking for python"
$candidates = @()
if ($Python) { $candidates += $Python }
foreach ($tag in @("-3.10", "-3.11", "-3.12", "-3")) {
    $p = (Get-Command py -ErrorAction SilentlyContinue)
    if ($p) {
        $found = & py $tag -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { $candidates += $found }
    }
}
$onPath = (Get-Command python -ErrorAction SilentlyContinue)
if ($onPath) { $candidates += $onPath.Source }

$exe = $null
foreach ($c in $candidates) {
    if (Test-Interpreter $c) { $exe = $c; break }
    if ($c -like "*WindowsApps*") {
        Write-Host "      skipping the Microsoft Store stub: $c" -ForegroundColor Yellow
    }
}
if (-not $exe) {
    Fail ("no usable python found. Install Python 3.10 from python.org " +
          "(tick 'Add python.exe to PATH'), or pass -Python C:\path\to\python.exe. " +
          "Candidates tried: " + ($candidates -join ", "))
}
$ver = & $exe -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
Write-Host "      using $exe  (python $ver)"

# --------------------------------------------------------------------------- #
# 2. create the venv
# --------------------------------------------------------------------------- #
if ((Test-Path $venv) -and $Force) {
    Write-Host "[2/5] removing the existing .venv (-Force)"
    Remove-Item -Recurse -Force $venv
}
$py = Join-Path $venv "Scripts\python.exe"
if (Test-Path $py) {
    Write-Host "[2/5] .venv already exists, reusing it (-Force to rebuild)"
} else {
    Write-Host "[2/5] creating venv at $venv"
    & $exe -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "'$exe -m venv' returned $LASTEXITCODE" }
    if (-not (Test-Path $py)) {
        Fail "venv creation reported success but $py does not exist"
    }
}

# --------------------------------------------------------------------------- #
# 3..5 install
# --------------------------------------------------------------------------- #
Write-Host "[3/5] upgrading pip"
& $py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade returned $LASTEXITCODE" }

Write-Host "[4/5] installing torch ($CudaTag)"
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"
if ($LASTEXITCODE -ne 0) {
    Fail ("torch install returned $LASTEXITCODE. Try another build, " +
          "e.g. -CudaTag cpu or -CudaTag cu124.")
}

Write-Host "[5/5] installing requirements"
& $py -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "requirements install returned $LASTEXITCODE" }

# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
Write-Host ""
& $py -c @"
import torch, numpy, scipy, pandas, matplotlib
print('torch      ', torch.__version__)
print('cuda       ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU only)')
print('numpy/scipy', numpy.__version__, scipy.__version__)
"@
if ($LASTEXITCODE -ne 0) { Fail "the installed packages do not import" }

$activate = Join-Path $venv "Scripts\Activate.ps1"
Write-Host ""
Write-Host "done." -ForegroundColor Green
Write-Host "activate with:"
Write-Host "    $activate"
Write-Host "or skip activation entirely and call the interpreter directly:"
Write-Host "    .venv\Scripts\python.exe scripts\00_check_env.py --selftest"
