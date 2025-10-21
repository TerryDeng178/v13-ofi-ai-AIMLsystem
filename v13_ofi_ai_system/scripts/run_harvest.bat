@echo off
echo 启动OFI+CVD数据采集系统...

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=2
set PARQUET_ROTATE_SEC=60
set WSS_PING_INTERVAL=20
set DEDUP_LRU=8192
set Z_MODE=delta
set SCALE_MODE=hybrid
set MAD_MULTIPLIER=1.8
set SCALE_FAST_WEIGHT=0.20
set HALF_LIFE_SEC=600
set WINSOR_LIMIT=8.0
set OUTPUT_DIR=data\ofi_cvd
set ARTIFACTS_DIR=artifacts

REM 创建必要目录
if not exist "data\ofi_cvd" mkdir "data\ofi_cvd"
if not exist "artifacts\run_logs" mkdir "artifacts\run_logs"
if not exist "artifacts\dq_reports" mkdir "artifacts\dq_reports"

REM 生成日志文件名
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "stamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"
set "logPath=artifacts\run_logs\harvest_%stamp%.log"

echo 配置: 运行 %RUN_HOURS% 小时
echo 日志: %logPath%

REM 启动数据采集进程
echo 启动数据采集进程...
start /B python examples\run_realtime_harvest.py > "%logPath%" 2>&1

echo 数据采集进程已启动
echo 日志文件: %logPath%
echo 进程将在后台运行 %RUN_HOURS% 小时

pause

