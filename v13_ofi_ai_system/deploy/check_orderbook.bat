@echo off
REM harvestd - 订单簿数据收集状态检查脚本
REM 专门检查订单簿数据收集的状态和进度

setlocal

set HARVESTD_PORT=8088

echo ========================================
echo harvestd - 订单簿数据收集状态检查
echo ========================================
echo.

REM 检查HTTP服务是否可用
echo [1/3] 检查HTTP服务状态...
curl -s http://localhost:%HARVESTD_PORT%/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] HTTP服务未运行
    echo 请先运行 deploy\start_harvestd.bat 或 deploy\restart_harvestd.bat
    echo.
    pause
    exit /b 1
) else (
    echo [OK] HTTP服务正常运行
)
echo.

REM 获取订单簿数据收集状态
echo [2/3] 获取订单簿数据收集状态...
curl -s http://localhost:%HARVESTD_PORT%/orderbook >nul 2>&1
if errorlevel 1 (
    echo [FAIL] 无法获取订单簿状态
    echo 请检查服务是否正常运行
    echo.
    pause
    exit /b 1
) else (
    echo [OK] 订单簿状态信息:
    curl -s http://localhost:%HARVESTD_PORT%/orderbook
    echo.
)
echo.

REM 检查本地订单簿数据文件
echo [3/3] 检查本地订单簿数据文件...
if exist "data\ofi_cvd\date=*\symbol=*\kind=orderbook\*.parquet" (
    echo [OK] 找到订单簿数据文件
    echo.
    echo 订单簿文件列表:
    for /r "data\ofi_cvd" %%F in (*orderbook*.parquet) do (
        echo   %%F
    )
    echo.
    
    REM 统计文件数量和大小
    set /a file_count=0
    set /a total_size=0
    for /r "data\ofi_cvd" %%F in (*orderbook*.parquet) do (
        set /a file_count+=1
        for %%S in ("%%F") do set /a total_size+=%%~zS
    )
    echo 统计信息:
    echo   文件数量: %file_count%
    echo   总大小: %total_size% 字节
    echo   总大小: %/1000000% MB
) else (
    echo [INFO] 暂无订单簿数据文件
    echo 可能原因:
    echo   1. 数据收集进程未启动
    echo   2. 订单簿数据收集未启用 (ENABLE_ORDERBOOK=0)
    echo   3. 数据收集时间太短，还未生成文件
    echo   4. 数据保存在其他位置
)
echo.

REM 显示关键信息
echo ========================================
echo 订单簿数据收集监控:
echo   状态页面: http://localhost:%HARVESTD_PORT%/orderbook
echo   主监控UI: http://localhost:%HARVESTD_PORT%/
echo   健康检查: http://localhost:%HARVESTD_PORT%/health
echo ========================================
echo.

REM 提供操作建议
echo 操作建议:
echo   1. 如果未启用订单簿收集，运行: deploy\restart_harvestd.bat
echo   2. 如果数据收集异常，检查: deploy\check_status.bat
echo   3. 如果需要诊断问题，运行: deploy\diagnose.bat
echo.

pause

