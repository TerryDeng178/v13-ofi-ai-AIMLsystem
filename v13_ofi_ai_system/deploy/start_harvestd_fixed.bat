@echo off
REM harvestd - 增强版Windows启动脚本
REM 修复重启机制失效问题，增强稳定性和自动恢复能力

setlocal enabledelayedexpansion

echo ========================================
echo harvestd - OFI+CVD Data Collection Daemon (增强版)
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
    echo 请确保在正确的项目根目录运行此脚本
    pause
    exit /b 1
)

if not exist "deploy\run_success_harvest.py" (
    echo [ERROR] 找不到 deploy\run_success_harvest.py
    echo 请确保在正确的项目根目录运行此脚本
    pause
    exit /b 1
)

if not exist "scripts\validate_ofi_cvd_harvest.py" (
    echo [ERROR] 找不到 scripts\validate_ofi_cvd_harvest.py
    echo 请确保在正确的项目根目录运行此脚本
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist "deploy\artifacts\run_logs" mkdir "deploy\artifacts\run_logs"
if not exist "deploy\artifacts\dq_reports" mkdir "deploy\artifacts\dq_reports"
if not exist "deploy\data\ofi_cvd" mkdir "deploy\data\ofi_cvd"

echo [INFO] 目录结构检查完成
echo.

REM 设置环境变量（修复版配置）
if not defined HARVESTD_PORT set HARVESTD_PORT=8088
if not defined VALIDATE_INTERVAL_MIN set VALIDATE_INTERVAL_MIN=60
if not defined SYMBOLS set SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT
if not defined RUN_HOURS set RUN_HOURS=168
if not defined PYTHONIOENCODING set PYTHONIOENCODING=UTF-8

REM 新增：订单簿数据收集配置
if not defined ENABLE_ORDERBOOK set ENABLE_ORDERBOOK=1
if not defined ORDERBOOK_ROTATE_SEC set ORDERBOOK_ROTATE_SEC=60

REM 新增：重启机制配置
if not defined RESTART_BACKOFF_MAX_SEC set RESTART_BACKOFF_MAX_SEC=60
if not defined DQ_FAIL_MAX_TOL set DQ_FAIL_MAX_TOL=3

echo [INFO] 配置信息:
echo   HTTP端口: %HARVESTD_PORT%
echo   数据质量检查间隔: %VALIDATE_INTERVAL_MIN% 分钟
echo   交易对: %SYMBOLS%
echo   运行时长: %RUN_HOURS% 小时 (7天)
echo   订单簿数据收集: %ENABLE_ORDERBOOK%
echo   订单簿轮转间隔: %ORDERBOOK_ROTATE_SEC% 秒
echo   最大重启延迟: %RESTART_BACKOFF_MAX_SEC% 秒
echo   数据质量失败容忍: %DQ_FAIL_MAX_TOL% 次
echo.

REM 检查端口是否被占用
netstat -ano | findstr ":%HARVESTD_PORT%" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] 端口 %HARVESTD_PORT% 被占用，尝试释放...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%HARVESTD_PORT%"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

REM 启动守护进程
echo [INFO] 启动 harvestd daemon (修复版)...
echo [INFO] 访问监控UI: http://localhost:%HARVESTD_PORT%/
echo [INFO] 按 Ctrl+C 停止守护进程
echo [INFO] 守护进程将自动重启采集器，即使正常退出
echo.

REM 使用无限循环确保守护进程持续运行
:restart_loop
python tools\harvestd.py
if errorlevel 1 (
    echo [ERROR] harvestd 进程异常退出，5秒后重启...
    timeout /t 5 /nobreak >nul
    goto restart_loop
) else (
    echo [INFO] harvestd 正常退出，5秒后重启...
    timeout /t 5 /nobreak >nul
    goto restart_loop
)

echo.
echo [INFO] harvestd daemon 已停止
pause
