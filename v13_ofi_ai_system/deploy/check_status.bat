@echo off
REM harvestd - 状态检查脚本
REM 检查守护进程运行状态和数据质量

setlocal

set HARVESTD_PORT=8088

echo ========================================
echo harvestd Status Checker
echo ========================================
echo.

REM 检查HTTP服务是否可用
echo [1/3] 检查HTTP服务状态...
curl -s http://localhost:%HARVESTD_PORT%/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] HTTP服务未运行
    echo 请先运行 deploy\start_harvestd.bat
    echo.
) else (
    echo [OK] HTTP服务正常运行
    curl -s http://localhost:%HARVESTD_PORT%/health
    echo.
)

REM 检查健康状态
echo [2/3] 检查健康状态...
curl -s http://localhost:%HARVESTD_PORT%/health 2>nul | findstr /C:"ok" >nul
if errorlevel 1 (
    echo [WARN] 服务状态: degraded
) else (
    echo [OK] 服务状态: healthy
)
echo.

REM 检查最新DQ报告
echo [3/4] 检查数据质量报告...
if exist "deploy\artifacts\dq_reports\dq_*.json" (
    echo [OK] 找到DQ报告文件
    for %%F in ("deploy\artifacts\dq_reports\dq_*.json") do (
        echo   最新报告: %%~nF
    )
) else (
    echo [INFO] 暂无DQ报告（可能还未运行第一次验证）
)
echo.

REM 检查订单簿数据收集状态
echo [4/4] 检查订单簿数据收集状态...
if exist "deploy\data\ofi_cvd\date=*\symbol=*\kind=orderbook\*.parquet" (
    echo [OK] 找到订单簿数据文件
    for /r "deploy\data\ofi_cvd" %%F in (*orderbook*.parquet) do (
        echo   订单簿文件: %%~nF
    )
) else (
    echo [INFO] 暂无订单簿数据（可能还未开始收集或配置未启用）
)
echo.

REM 显示关键信息
echo ========================================
echo Quick Links:
echo   监控UI:  http://localhost:%HARVESTD_PORT%/
echo   健康检查: http://localhost:%HARVESTD_PORT%/health
echo   查看日志: http://localhost:%HARVESTD_PORT%/logs
echo   数据质量: http://localhost:%HARVESTD_PORT%/dq
echo   订单簿状态: http://localhost:%HARVESTD_PORT%/orderbook
echo   指标:    http://localhost:%HARVESTD_PORT%/metrics
echo ========================================
echo.

pause
