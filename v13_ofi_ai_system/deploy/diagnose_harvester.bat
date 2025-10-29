@echo off
REM harvestd 诊断脚本
REM 检查采集器状态和问题

setlocal enabledelayedexpansion

echo ========================================
echo harvestd 诊断工具
echo ========================================
echo.

REM 设置脚本所在目录
set DEPLOY_DIR=%~dp0
set SCRIPT_DIR=%DEPLOY_DIR%..
cd /d "%SCRIPT_DIR%"

echo [1/6] 检查Python进程...
tasklist | findstr python
if errorlevel 1 (
    echo [INFO] 没有Python进程在运行
) else (
    echo [WARN] 发现Python进程在运行
)
echo.

echo [2/6] 检查端口8088状态...
netstat -ano | findstr ":8088"
if errorlevel 1 (
    echo [INFO] 端口8088未被占用
) else (
    echo [WARN] 端口8088被占用
)
echo.

echo [3/6] 检查最新数据文件...
if exist "deploy\data\ofi_cvd\date=2025-10-26\symbol=BTCUSDT\kind=prices\" (
    echo [INFO] 检查BTCUSDT价格数据...
    for /f "tokens=*" %%i in ('dir "deploy\data\ofi_cvd\date=2025-10-26\symbol=BTCUSDT\kind=prices\" /T:W /B ^| findstr /R "part-.*\.parquet$"') do (
        set LATEST_FILE=%%i
    )
    echo [INFO] 最新文件: !LATEST_FILE!
    
    REM 获取文件时间
    for /f "tokens=1,2" %%a in ('dir "deploy\data\ofi_cvd\date=2025-10-26\symbol=BTCUSDT\kind=prices\%LATEST_FILE%" /T:W') do (
        echo [INFO] 文件时间: %%a %%b
    )
) else (
    echo [ERROR] 数据目录不存在
)
echo.

echo [4/6] 检查日志文件...
if exist "deploy\artifacts\run_logs\" (
    echo [INFO] 检查运行日志...
    dir "deploy\artifacts\run_logs\" /T:W /B | findstr /R "harvester_.*\.log$"
) else (
    echo [WARN] 日志目录不存在
)
echo.

echo [5/6] 检查数据质量报告...
if exist "deploy\artifacts\dq_reports\" (
    echo [INFO] 检查数据质量报告...
    dir "deploy\artifacts\dq_reports\" /T:W /B | findstr /R "dq_.*\.json$"
) else (
    echo [WARN] 数据质量报告目录不存在
)
echo.

echo [6/6] 检查网络连接...
ping -n 1 binance.com >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 无法连接到binance.com
) else (
    echo [OK] 网络连接正常
)
echo.

echo ========================================
echo 诊断完成
echo ========================================
echo.
echo 建议操作:
echo 1. 如果采集器停止，运行: deploy\start_harvestd_fixed.bat
echo 2. 如果端口被占用，运行: deploy\restart_harvestd.bat
echo 3. 查看监控界面: http://localhost:8088/
echo.

pause
