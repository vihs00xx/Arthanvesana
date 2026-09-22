# Resumable synthetic-calibration grid for the bigram-vs-trigram inference.
#
# Runs the calibration shards in parallel (one per core-group), then rebuilds the
# summary from the per-replicate cache. Safe to re-run: every replicate that
# already finished is reused, so an interrupted run simply continues.
#
# Usage (from the repository root, works in Windows PowerShell 5.1 and PowerShell 7+):
#   powershell -ExecutionPolicy Bypass -File scripts\run_calibration_grid.ps1
# or, if you are already in a PowerShell session:
#   & .\scripts\run_calibration_grid.ps1
#
# Outputs:
#   outputs/power_analysis/replicates/            per-replicate cache
#   outputs/power_analysis/power_analysis_summary.json
#   outputs/power_analysis/power_analysis_report.txt

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python not found at $py" }
Set-Location $root

Write-Host "Working directory: $root"
Write-Host "Interpreter:       $py"
Write-Host ""

# Shared settings for the grid cells.
$common = @(
    "--sizes", "1.0",
    "--replicates", "6",
    "--permutations", "2000",
    "--bootstrap", "2000",
    "--cache-only",
    "--quiet"
)

# Disjoint scenario subsets, so shards never duplicate work.
$shards = @(
    @("--scenarios", "unigram", "position_only", "--lambdas", "0.0"),
    @("--scenarios", "markov", "hmm_slots", "--lambdas", "0.0"),
    @("--scenarios", "shuffled_real", "--lambdas", "0.0", "--null-replicates", "20"),
    @("--scenarios", "trigram_mixture", "--lambdas", "0.0", "0.25", "0.30"),
    @("--scenarios", "trigram_mixture", "--lambdas", "0.35", "0.40", "0.5", "1.0")
)

$cached = @(Get-ChildItem "outputs\power_analysis\replicates" -Recurse -File `
    -ErrorAction SilentlyContinue).Count
Write-Host "Replicates already cached: $cached"
Write-Host "Starting $($shards.Count) shards in parallel. This is the long part."
Write-Host ""

$jobs = foreach ($shard in $shards) {
    $cliArgs = $shard + $common
    Start-Job -ScriptBlock {
        param($exe, $dir, $argv)
        Set-Location $dir
        & $exe scripts/run_power_analysis.py @argv 2>&1
    } -ArgumentList $py, $root, $cliArgs
}

$done = 0
foreach ($job in $jobs) {
    Wait-Job $job | Out-Null
    $out = Receive-Job $job
    if ($out) { $out | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" } }
    Remove-Job $job
    $done++
    $n = @(Get-ChildItem "outputs\power_analysis\replicates" -Recurse -File `
        -ErrorAction SilentlyContinue).Count
    Write-Host "  shard $done/$($jobs.Count) finished; $n replicates cached"
}

Write-Host ""
Write-Host "Rebuilding the summary from cache (this also runs the structural test)..."
& $py scripts/run_power_analysis.py `
    --sizes 1.0 `
    --replicates 6 `
    --scenarios unigram position_only markov hmm_slots shuffled_real trigram_mixture `
    --lambdas 0.0 0.25 0.30 0.35 0.40 0.5 1.0 `
    --null-replicates 20 `
    --permutations 2000 `
    --bootstrap 2000

Write-Host ""
Write-Host "Done. Results:"
Write-Host "  outputs\power_analysis\power_analysis_report.txt"
Write-Host "  outputs\power_analysis\power_analysis_summary.json"