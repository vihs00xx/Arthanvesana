# Phase 2: regenerate the two remaining corrected experiments, then verify.
#
# Run AFTER the Phase 1 calibration grid has finished.
#
# Usage (Windows PowerShell 5.1 or PowerShell 7+, from the repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_remaining_grid.ps1
#
# Estimated wall time: ~90 minutes (direction diagnostics dominates).

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python not found at $py" }
Set-Location $root

Write-Host "Working directory: $root"
Write-Host ""

function Step($name, $block) {
    Write-Host "=== $name ==="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $block
    $sw.Stop()
    Write-Host ("--- {0} finished in {1:N1} min ---" -f $name, ($sw.Elapsed.TotalMinutes))
    Write-Host ""
}

# 1. Direction diagnostics: aligned union groups, OOV-separated transfer,
#    three boundary models, first/interior/last/singleton breakdown.
#    Measured: --repeats 1 costs ~8 min, so --repeats 10 is ~80 min.
Step "Direction diagnostics (repeats=10, ~80 min)" {
    & $py scripts/run_direction_diagnostics.py --repeats 10
}

# 2. Same-family transcription sensitivity. Measured: --repeats 1 costs ~46 s,
#    so --repeats 10 is ~8 min.
#
#    NOTE: the Step 6 repair to this runner is NOT yet implemented. It still
#    prints count-only placeholders ("descriptive; per-label metrics inherit the
#    aligned design") instead of real out-of-fold subgroup metrics, and it does
#    not build the explicit artifact/inscription pair table. Running it now
#    refreshes the existing behaviour; it does NOT satisfy the brief's Step 6.
Step "Transcription sensitivity (repeats=10, ~8 min) -- PRE-CORRECTION BEHAVIOUR" {
    & $py scripts/run_transcription_sensitivity.py --repeats 10
}

# 3. Verification.
Step "Verification" {
    & $py -m pytest -q
    & $py -m ruff check .
    & $py -m mypy
}

Write-Host "Phase 2 done. Results:"
Write-Host "  outputs\direction_diagnostics\direction_diagnostics_report.txt"
Write-Host "  outputs\transcription_sensitivity\transcription_sensitivity_report.txt"