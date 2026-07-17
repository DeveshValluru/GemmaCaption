# Local acceptance-test harness (BLUEPRINT §9). Run from repo root:
#   pwsh tests/run_local.ps1 -Build                       # build + happy path
#   pwsh tests/run_local.ps1 -InputDir tests/edge_input   # a specific scenario
#   pwsh tests/run_local.ps1 -EnvFile .env                # with real creds
param(
    [string]$InputDir = "tests/test_input",
    [string]$Tag = "stylecap:dev",
    [string]$EnvFile = "",
    [switch]$Build
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $root

if ($Build) {
    Write-Host "==> building $Tag" -ForegroundColor Cyan
    docker build -t $Tag .
}

$inAbs = (Resolve-Path $InputDir).Path
$outAbs = Join-Path $root "tests/out"
New-Item -ItemType Directory -Force -Path $outAbs | Out-Null
Remove-Item -Force (Join-Path $outAbs "results.json") -ErrorAction SilentlyContinue

$envArgs = @()
if ($EnvFile -and (Test-Path $EnvFile)) { $envArgs = @("--env-file", (Resolve-Path $EnvFile).Path) }

Write-Host "==> running $Tag on $InputDir" -ForegroundColor Cyan
$t0 = Get-Date
docker run --rm @envArgs -v "${inAbs}:/input:ro" -v "${outAbs}:/output" $Tag
$elapsed = (Get-Date) - $t0
Write-Host ("==> container finished in {0:N1}s (limit 600s)" -f $elapsed.TotalSeconds) -ForegroundColor Cyan

Write-Host "==> validating output" -ForegroundColor Cyan
python tests/validate_output.py (Join-Path $inAbs "tasks.json") (Join-Path $outAbs "results.json")
