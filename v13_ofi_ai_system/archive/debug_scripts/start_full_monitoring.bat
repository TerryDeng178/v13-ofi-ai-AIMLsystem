@echo off
echo ========================================
echo V13 完整监控系统启动脚本
echo ========================================

echo 1. 检查Docker是否运行...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：Docker未安装或未运行
    pause
    exit /b 1
)

echo 2. 创建环境变量文件...
if not exist .env (
    copy env.example .env
    echo 已创建.env文件，请根据需要修改密码
)

echo 3. 启动完整监控栈...
docker compose up -d

echo 4. 等待服务启动...
timeout /t 10 /nobreak >nul

echo 5. 检查服务状态...
docker compose ps

echo ========================================
echo 服务访问地址：
echo - Grafana: http://localhost:3000 (admin/从.env文件读取密码)
echo - Prometheus: http://localhost:9090
echo - Alertmanager: http://localhost:9093
echo - Loki: http://localhost:3100
echo ========================================

echo 6. 启动指标服务器...
start "Metrics Server" cmd /k "cd grafana && python simple_metrics_server.py 8000"

echo 监控系统启动完成！
pause
