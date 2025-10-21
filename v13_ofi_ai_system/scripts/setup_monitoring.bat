@echo off
REM 设置OFI+CVD数据采集监控环境

echo ========================================
echo OFI+CVD监控环境设置
echo ========================================

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Docker环境
    echo 请先安装Docker Desktop
    pause
    exit /b 1
)

echo 启动Prometheus和Grafana...

REM 创建docker-compose.yml
echo version: '3.8' > docker-compose.monitoring.yml
echo services: >> docker-compose.monitoring.yml
echo   prometheus: >> docker-compose.monitoring.yml
echo     image: prom/prometheus:latest >> docker-compose.monitoring.yml
echo     container_name: ofi_cvd_prometheus >> docker-compose.monitoring.yml
echo     ports: >> docker-compose.monitoring.yml
echo       - "9090:9090" >> docker-compose.monitoring.yml
echo     volumes: >> docker-compose.monitoring.yml
echo       - "./grafana/prometheus.yml:/etc/prometheus/prometheus.yml" >> docker-compose.monitoring.yml
echo       - "./grafana/ofi_cvd_alerts.yml:/etc/prometheus/ofi_cvd_alerts.yml" >> docker-compose.monitoring.yml
echo       - "prometheus_data:/prometheus" >> docker-compose.monitoring.yml
echo     command: >> docker-compose.monitoring.yml
echo       - '--config.file=/etc/prometheus/prometheus.yml' >> docker-compose.monitoring.yml
echo       - '--storage.tsdb.path=/prometheus' >> docker-compose.monitoring.yml
echo       - '--web.console.libraries=/etc/prometheus/console_libraries' >> docker-compose.monitoring.yml
echo       - '--web.console.templates=/etc/prometheus/consoles' >> docker-compose.monitoring.yml
echo       - '--web.enable-lifecycle' >> docker-compose.monitoring.yml
echo. >> docker-compose.monitoring.yml
echo   grafana: >> docker-compose.monitoring.yml
echo     image: grafana/grafana:latest >> docker-compose.monitoring.yml
echo     container_name: ofi_cvd_grafana >> docker-compose.monitoring.yml
echo     ports: >> docker-compose.monitoring.yml
echo       - "3000:3000" >> docker-compose.monitoring.yml
echo     volumes: >> docker-compose.monitoring.yml
echo       - "grafana_data:/var/lib/grafana" >> docker-compose.monitoring.yml
echo       - "./grafana/dashboards:/var/lib/grafana/dashboards" >> docker-compose.monitoring.yml
echo     environment: >> docker-compose.monitoring.yml
echo       - GF_SECURITY_ADMIN_PASSWORD=admin123 >> docker-compose.monitoring.yml
echo       - GF_USERS_ALLOW_SIGN_UP=false >> docker-compose.monitoring.yml
echo. >> docker-compose.monitoring.yml
echo volumes: >> docker-compose.monitoring.yml
echo   prometheus_data: >> docker-compose.monitoring.yml
echo   grafana_data: >> docker-compose.monitoring.yml

REM 启动监控服务
docker-compose -f docker-compose.monitoring.yml up -d

echo.
echo 监控服务已启动:
echo   Prometheus: http://localhost:9090
echo   Grafana: http://localhost:3000 (admin/admin123)
echo.
echo 等待服务启动完成...
timeout /t 10 /nobreak >nul

REM 导入Grafana仪表板
echo 导入Grafana仪表板...
curl -X POST "http://admin:admin123@localhost:3000/api/dashboards/db" ^
     -H "Content-Type: application/json" ^
     -d @grafana/dashboards/ofi_cvd_harvest.json >nul 2>&1

echo.
echo 监控环境设置完成！
echo.
echo 使用说明:
echo   1. 启动数据采集: scripts\start_harvest.bat
echo   2. 查看监控: http://localhost:3000
echo   3. 停止监控: docker-compose -f docker-compose.monitoring.yml down
echo.

pause
