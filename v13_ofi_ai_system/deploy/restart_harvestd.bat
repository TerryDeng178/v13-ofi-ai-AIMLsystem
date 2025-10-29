@echo off
REM harvestd - 重启数据收集进程脚本
REM 停止现有进程并启动新的数据收集进程，支持订单簿数据收集

setlocal enabledelayedexpansion

echo ========================================
echo harvestd - 重启数据收集进程
echo ========================================
echo.

REM 设置脚本所在目录为deploy目录
set DEPLOY_DIR=%~dp0
set SCRIPT_DIR=%DEPLOY_DIR%..
cd /d "%SCRIPT_DIR%"

echo [INFO] 当前目录: %CD%
echo.

REM 1. 停止现有的harvestd进程
echo [1/4] 停止现有harvestd进程...
tasklist | findstr "python" | findstr "harvestd" >nul 2>&1
if errorlevel 1 (
    echo [INFO] 未发现运行中的harvestd进程
) else (
    echo [INFO] 发现运行中的harvestd进程，正在停止...
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq harvestd*" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] 无法通过窗口标题停止进程，尝试通过端口停止...
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8088"') do (
            taskkill /F /PID %%a >nul 2>&1
        )
    )
    echo [OK] 已停止现有进程
)
echo.

REM 2. 等待进程完全停止
echo [2/4] 等待进程完全停止...
timeout /t 3 /nobreak >nul
echo [OK] 等待完成
echo.

REM 3. 检查端口是否释放
echo [3/4] 检查端口状态...
netstat -ano | findstr ":8088" >nul 2>&1
if errorlevel 1 (
    echo [OK] 端口8088已释放
) else (
    echo [WARN] 端口8088仍被占用，强制释放...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8088"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)
echo.

REM 4. 启动新的harvestd进程
echo [4/4] 启动新的harvestd进程...

REM 设置环境变量，启用订单簿数据收集
set ENABLE_ORDERBOOK=1
set ORDERBOOK_ROTATE_SEC=60
set HARVESTD_PORT=8088
set VALIDATE_INTERVAL_MIN=60
set SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT
set RUN_HOURS=72
set PYTHONIOENCODING=UTF-8

echo [INFO] 配置信息:
echo   订单簿数据收集: %ENABLE_ORDERBOOK%
echo   订单簿轮转间隔: %ORDERBOOK_ROTATE_SEC% 秒
echo   HTTP端口: %HARVESTD_PORT%
echo   交易对: %SYMBOLS%
echo   运行时长: %RUN_HOURS% 小时
echo.

echo [INFO] 启动harvestd守护进程...
echo [INFO] 访问监控UI: http://localhost:%HARVESTD_PORT%/
echo [INFO] 按 Ctrl+C 停止守护进程
echo.

REM 启动守护进程
python tools\harvestd.py

echo.
echo [INFO] harvestd守护进程已停止
pause

