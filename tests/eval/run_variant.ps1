# Run one pipeline variant on the 11-clip devset, then judge it.
#   pwsh tests/eval/run_variant.ps1 -Label grid2x2 -EnvOverrides @{FRAME_TILE="2x2"; GRID_TOTAL_FRAMES="16"}
param(
    [Parameter(Mandatory=$true)][string]$Label,
    [hashtable]$EnvOverrides = @{},
    [string]$Image = "stylecap:dev6"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/../..").Path
Set-Location $root
$ev = (Resolve-Path "tests/eval").Path
$outDir = Join-Path $ev "out_$Label"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Remove-Item -Force (Join-Path $outDir "results.json") -ErrorAction SilentlyContinue

$envArgs = @()
foreach ($k in $EnvOverrides.Keys) { $envArgs += @("-e", "$k=$($EnvOverrides[$k])") }

Write-Host "==> [$Label] running devset (11 clips)" -ForegroundColor Cyan
$t0 = Get-Date
$log = docker run --rm @envArgs -v "${ev}:/input:ro" -v "${outDir}:/output" `
    -e INPUT_PATH=/input/tasks.json $Image 2>&1
$viaHf = ($log | Select-String "done via hf").Count
$viaFw = ($log | Select-String "done via fireworks").Count
$tpl = ($log | Select-String "done via template").Count
Write-Host ("==> [{0}] {1:N0}s | hf:{2} fw:{3} tpl:{4}" -f $Label, ((Get-Date)-$t0).TotalSeconds, $viaHf, $viaFw, $tpl)
python tests/eval/judge.py (Join-Path $outDir "results.json") $Label
