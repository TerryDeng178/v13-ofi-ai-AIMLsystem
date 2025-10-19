# V13 策略模式监控仪表盘启动脚本
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "V13 策略模式监控仪表盘启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "1. 检查Docker状态..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker已安装: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker未安装或未启动" -ForegroundColor Red
    Write-Host "请安装Docker Desktop并启动后重试" -ForegroundColor Yellow
    Read-Host "按Enter键退出"
    exit 1
}

Write-Host ""
Write-Host "2. 启动监控服务..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "3. 等待服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "4. 检查服务状态..." -ForegroundColor Yellow

# 检查Prometheus
try {
    $prometheusResponse = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=up" -TimeoutSec 5
    Write-Host "✅ Prometheus: http://localhost:9090" -ForegroundColor Green
} catch {
    Write-Host "❌ Prometheus启动失败" -ForegroundColor Red
}

# 检查Grafana
try {
    $grafanaResponse = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -TimeoutSec 5
    Write-Host "✅ Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Green
} catch {
    Write-Host "❌ Grafana启动失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "仪表盘访问地址：" -ForegroundColor Cyan
Write-Host "- Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "按Enter键退出"
