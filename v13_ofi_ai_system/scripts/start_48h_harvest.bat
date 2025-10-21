@echo off
echo ========================================
echo 启动48小时OFI+CVD数据收集
echo ========================================
echo 开始时间: %date% %time%
echo 预计结束时间: 48小时后
echo 交易对: BTCUSDT, ETHUSDT
echo ========================================

REM 设置环境变量
set RUN_HOURS=48
set SYMBOLS=BTCUSDT,ETHUSDT
set PARQUET_ROTATE_SEC=60
set WSS_PING_INTERVAL=20
set DEDUP_LRU=8192

REM 创建必要的目录
if not exist "data\ofi_cvd" mkdir "data\ofi_cvd"
if not exist "artifacts\run_logs" mkdir "artifacts\run_logs"
if not exist "artifacts\dq_reports" mkdir "artifacts\dq_reports"

REM 启动数据收集（后台运行）
echo 正在启动数据收集进程...
start /B python examples\run_success_harvest.py > artifacts\run_logs\harvest_48h_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log 2>&1

echo 数据收集已启动，进程在后台运行
echo 日志文件: artifacts\run_logs\harvest_48h_*.log
echo 数据目录: data\ofi_cvd\
echo.
echo 要监控进度，请运行: type artifacts\run_logs\harvest_48h_*.log
echo 要停止收集，请运行: taskkill /f /im python.exe
echo.
pause

