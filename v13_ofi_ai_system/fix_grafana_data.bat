@echo off
echo ========================================
echo 修复Grafana数据显示问题
echo ========================================

echo.
echo 1. 重启Prometheus服务...
docker-compose restart prometheus

echo.
echo 2. 等待服务启动...
timeout /t 15 /nobreak >nul

echo.
echo 3. 检查连接状态...
python check_prometheus_data.py

echo.
echo 4. 如果仍有问题，请尝试以下操作：
echo    - 重启所有Docker服务: docker-compose restart
echo    - 检查指标服务器: python grafana/simple_metrics_server.py 8000
echo    - 在Grafana中测试查询: strategy_mode_active
echo.
echo ========================================
echo 修复完成！
echo ========================================
pause
