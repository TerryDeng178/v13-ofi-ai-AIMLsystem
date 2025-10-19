@echo off
chcp 65001 >nul
echo ========================================
echo V13 策略模式监控仪表盘启动脚本
echo ========================================

echo.
echo 1. 启动指标生成器...
start "Metrics Generator" python grafana/metrics_server.py 8000

echo.
echo 2. 等待指标服务器启动...
timeout /t 3 /nobreak >nul

echo.
echo 3. 启动Docker监控服务...
docker-compose up -d

echo.
echo 4. 等待服务启动...
timeout /t 15 /nobreak >nul

echo.
echo 5. 检查服务状态...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 指标服务器: http://localhost:8000/metrics
) else (
    echo ❌ 指标服务器启动失败
)

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
echo 仪表盘启动完成！
echo ========================================
echo.
echo 访问地址：
echo - Grafana仪表盘: http://localhost:3000
echo - 用户名: admin
echo - 密码: admin
echo.
echo - Prometheus: http://localhost:9090
echo - 指标端点: http://localhost:8000/metrics
echo.
echo 导入的仪表盘：
echo - Strategy Mode Overview (策略模式概览)
echo - Strategy Performance (策略性能)
echo - Strategy Alerts (策略告警)
echo.
echo ========================================
echo 按任意键退出...
pause >nul
