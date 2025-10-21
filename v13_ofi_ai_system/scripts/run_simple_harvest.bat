@echo off
echo 启动简化版OFI+CVD数据采集系统...

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=2
set OUTPUT_DIR=data\ofi_cvd

REM 创建必要目录
if not exist "data\ofi_cvd" mkdir "data\ofi_cvd"
if not exist "artifacts\run_logs" mkdir "artifacts\run_logs"

REM 生成日志文件名
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "stamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"
set "logPath=artifacts\run_logs\simple_harvest_%stamp%.log"

echo 配置: 运行 %RUN_HOURS% 小时
echo 日志: %logPath%

REM 启动数据采集进程
echo 启动简化版数据采集进程...
start /B python examples\simple_harvest.py > "%logPath%" 2>&1

echo 简化版数据采集进程已启动
echo 日志文件: %logPath%
echo 进程将在后台运行 %RUN_HOURS% 小时

pause

