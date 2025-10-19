@echo off
echo ========================================
echo V13 策略模式监控仪表盘启动脚本
echo ========================================

echo.
echo 1. 检查Docker状态...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker未安装或未启动
    echo 请安装Docker Desktop并启动后重试
    pause
    exit /b 1
)

echo ✅ Docker已安装

echo.
echo 2. 启动监控服务...
docker-compose up -d

echo.
echo 3. 等待服务启动...
timeout /t 10 /nobreak >nul

echo.
echo 4. 检查服务状态...
curl -s http://localhost:9090/api/v1/query?query=up >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus: http://localhost:9090
) else (
    echo ❌ Prometheus启动失败
)

curl -s http://localhost:3000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Grafana: http://localhost:3000 (admin/admin)
) else (
    echo ❌ Grafana启动失败
)

echo.
echo ========================================
echo 仪表盘访问地址：
echo - Grafana: http://localhost:3000 (admin/admin)
echo - Prometheus: http://localhost:9090
echo ========================================
echo.
echo 按任意键退出...
pause >nul
