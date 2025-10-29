@echo off
REM harvestd - Windows启动脚本
REM OFI+CVD Data Collection Daemon Launcher

setlocal enabledelayedexpansion

echo ========================================
echo harvestd - OFI+CVD Data Collection Daemon
echo ========================================
echo.

REM 设置脚本所在目录为deploy目录
set DEPLOY_DIR=%~dp0
set SCRIPT_DIR=%DEPLOY_DIR%..
cd /d "%SCRIPT_DIR%"

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python未安装或不在PATH中
    echo 请先安装Python 3.11+
    pause
    exit /b 1
)

echo [INFO] 当前目录: %CD%
echo [INFO] Python版本:
python --version
echo.

REM 检查必要文件是否存在
if not exist "tools\harvestd.py" (
    echo [ERROR] 找不到 tools\harvestd.py
    pause
    exit /b 1
)

if not exist "deploy\run_success_harvest.py" (
    echo [ERROR] 找不到 deploy\run_success_harvest.py
    pause
    exit /b 1
)

if not exist "scripts\validate_ofi_cvd_harvest.py" (
    echo [ERROR] 找不到 scripts\validate_ofi_cvd_harvest.py
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist "deploy\artifacts\run_logs" mkdir "deploy\artifacts\run_logs"
if not exist "deploy\artifacts\dq_reports" mkdir "deploy\artifacts\dq_reports"
if not exist "deploy\data\ofi_cvd" mkdir "deploy\data\ofi_cvd"

echo [INFO] 目录结构检查完成
echo.

REM 设置环境变量（可选，如果已设置则使用已设置的值）
if not defined HARVESTD_PORT set HARVESTD_PORT=8088
if not defined VALIDATE_INTERVAL_MIN set VALIDATE_INTERVAL_MIN=60
if not defined SYMBOLS set SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT
if not defined RUN_HOURS set RUN_HOURS=72
if not defined PYTHONIOENCODING set PYTHONIOENCODING=UTF-8

REM 新增：订单簿数据收集配置
if not defined ENABLE_ORDERBOOK set ENABLE_ORDERBOOK=1
if not defined ORDERBOOK_ROTATE_SEC set ORDERBOOK_ROTATE_SEC=60

echo [INFO] 配置信息:
echo   HTTP端口: %HARVESTD_PORT%
echo   数据质量检查间隔: %VALIDATE_INTERVAL_MIN% 分钟
echo   交易对: %SYMBOLS%
echo   运行时长: %RUN_HOURS% 小时
echo   订单簿数据收集: %ENABLE_ORDERBOOK%
echo   订单簿轮转间隔: %ORDERBOOK_ROTATE_SEC% 秒
echo.

REM 启动守护进程
echo [INFO] 启动 harvestd daemon...
echo [INFO] 访问监控UI: http://localhost:%HARVESTD_PORT%/
echo [INFO] 按 Ctrl+C 停止守护进程
echo.

python tools\harvestd.py

echo.
echo [INFO] harvestd daemon 已停止
pause
