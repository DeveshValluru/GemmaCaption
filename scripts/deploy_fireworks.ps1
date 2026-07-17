# Create the Fireworks on-demand Gemma 4 deployment (Phase-0 brain + $3k Gemma prize runtime).
# Prereq: FIREWORKS_API_KEY in .env (or pass -ApiKey). Uses tools/firectl.exe.
# Billed by GPU-second while READY — remember to `firectl deployment delete <id>` when idle.
param(
    [string]$Model = "accounts/fireworks/models/gemma-4-26b-a4b-it",
    [string]$Accelerator = "NVIDIA_H100_80GB",
    [string]$Region = "GLOBAL",
    [string]$ApiKey = ""
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path
$firectl = Join-Path $root "tools/firectl.exe"

if (-not $ApiKey) {
    $line = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*FIREWORKS_API_KEY\s*=' } | Select-Object -First 1
    if ($line) { $ApiKey = ($line -split '=', 2)[1].Trim() }
}
if (-not $ApiKey) { throw "No FIREWORKS_API_KEY found in .env or -ApiKey." }

Write-Host "==> authenticating firectl" -ForegroundColor Cyan
& $firectl set-api-key $ApiKey
& $firectl whoami --api-key $ApiKey

Write-Host "==> creating deployment: $Model on $Accelerator ($Region)" -ForegroundColor Cyan
Write-Host "    (this reserves a GPU and bills by the second until deleted)" -ForegroundColor Yellow
& $firectl deployment create $Model --accelerator-type $Accelerator --region $Region --wait --api-key $ApiKey

Write-Host "`n==> deployments:" -ForegroundColor Cyan
& $firectl list deployments --api-key $ApiKey
Write-Host "`nNEXT: copy the deployment model id into .env as:" -ForegroundColor Green
Write-Host "  FIREWORKS_VLM_MODEL=accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>" -ForegroundColor Green
Write-Host "  FIREWORKS_TEXT_MODEL=accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>" -ForegroundColor Green
