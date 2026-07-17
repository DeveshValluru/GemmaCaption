# Build for linux/amd64 (R-07) and push a PUBLIC image to GHCR (R-08).
# Prereq: a GitHub Personal Access Token with write:packages.
#   pwsh scripts/push_ghcr.ps1 -User <github-user> -Token <PAT> -Tag v1
# After first push: make the package PUBLIC at github.com/users/<user>/packages (or it PULL_ERRORs).
param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Tag = "v1",
    [string]$Name = "stylecap"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $root
$image = "ghcr.io/$($User.ToLower())/${Name}:$Tag"

Write-Host "==> logging into ghcr.io as $User" -ForegroundColor Cyan
$Token | docker login ghcr.io -u $User --password-stdin

# Bake the Fireworks key from .env into the image (Track 2 injects nothing).
$fwKey = ""
$line = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*FIREWORKS_API_KEY\s*=' } | Select-Object -First 1
if ($line) { $fwKey = ($line -split '=', 2)[1].Trim() }
if (-not $fwKey) { throw "No FIREWORKS_API_KEY in .env — needed to bake into the submission image." }
$hfKey = ""
$hfline = Get-Content (Join-Path $root ".env") | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
if ($hfline) { $hfKey = ($hfline -split '=', 2)[1].Trim() }

Write-Host "==> buildx build + push (linux/amd64): $image" -ForegroundColor Cyan
docker buildx create --use --name stylecap-builder 2>$null | Out-Null
docker buildx build --platform linux/amd64 --build-arg FIREWORKS_API_KEY=$fwKey --build-arg HF_TOKEN=$hfKey -t $image --push .

Write-Host "`n==> verifying manifest arch" -ForegroundColor Cyan
docker buildx imagetools inspect $image | Select-String -Pattern "Platform|linux/amd64"

Write-Host "`nDONE: $image" -ForegroundColor Green
Write-Host "1) Make the package PUBLIC: https://github.com/users/$User/packages/container/$Name/settings" -ForegroundColor Yellow
Write-Host "2) Submit this exact reference on lablab; then watch the leaderboard scores it (R-08/R-15)." -ForegroundColor Yellow
