# KinetiMesh — PowerShell Setup Script
# Run this in PowerShell as Administrator in your KinetiMesh folder

Write-Host "KinetiMesh Project Setup" -ForegroundColor Cyan

# Remove malformed brace-expanded folders (PowerShell syntax)
$badFolders = Get-ChildItem -Path . -Directory | Where-Object { $_.Name -like "{*" }
foreach ($f in $badFolders) {
    Remove-Item -Recurse -Force $f.FullName
    Write-Host "Removed: $($f.Name)" -ForegroundColor Yellow
}

# Create clean directory structure
$dirs = @(
    "frontend\src\components\dashboard",
    "frontend\src\components\charts",
    "frontend\src\components\city3d",
    "frontend\src\components\quantum",
    "frontend\src\hooks",
    "frontend\src\pages",
    "frontend\src\store",
    "frontend\src\utils",
    "frontend\public",
    "backend\app\api",
    "backend\app\core",
    "backend\app\ml",
    "backend\app\db",
    "backend\app\services",
    "backend\tests",
    "ml\federated",
    "ml\rl",
    "ml\gnn",
    "ml\quantum",
    "blockchain\chaincode",
    "blockchain\network",
    "infra\k8s",
    "infra\nginx",
    "infra\grafana",
    "docs\api",
    "docs\architecture"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    Write-Host "Created: $d" -ForegroundColor Green
}

Write-Host ""
Write-Host "Structure ready! Now run:" -ForegroundColor Cyan
Write-Host "  cd frontend && npm create vite@latest . -- --template react && npm install" -ForegroundColor White
Write-Host "  cd ../backend && pip install -r requirements.txt" -ForegroundColor White
Write-Host "  cd ../infra && docker compose up -d" -ForegroundColor White
