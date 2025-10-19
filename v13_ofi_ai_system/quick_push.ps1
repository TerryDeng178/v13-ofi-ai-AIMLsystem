# V13 OFI+CVD 快速推送脚本 (PowerShell版本)

Write-Host "========================================" -ForegroundColor Green
Write-Host "V13 OFI+CVD 快速推送脚本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`n正在检查Git状态..." -ForegroundColor Yellow
git status

Write-Host "`n正在添加所有更改..." -ForegroundColor Yellow
git add .

Write-Host "`n请输入提交信息（或按回车使用默认信息）:" -ForegroundColor Cyan
$commitMsg = Read-Host "提交信息"

if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "feat: Update V13 OFI+CVD system"
}

Write-Host "`n正在提交更改..." -ForegroundColor Yellow
git commit -m $commitMsg

Write-Host "`n正在推送到GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "推送完成！" -ForegroundColor Green
Write-Host "仓库地址: https://github.com/TerryDeng178/v13-ofi-ai-AIMLsystem" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green

