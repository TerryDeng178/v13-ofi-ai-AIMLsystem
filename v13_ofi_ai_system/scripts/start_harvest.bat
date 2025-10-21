@echo off
REM OFI+CVD数据采集启动脚本 (Task 1.3.1 v2)
REM 支持Windows环境的一键启动

echo ========================================
echo OFI+CVD数据采集系统 (Task 1.3.1 v2)
echo ========================================

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=72
set PARQUET_ROTATE_SEC=60
set WSS_PING_INTERVAL=20
set DEDUP_LRU=8192
set Z_MODE=delta
set SCALE_MODE=hybrid
set MAD_MULTIPLIER=1.8
set SCALE_FAST_WEIGHT=0.20
set HALF_LIFE_SEC=600
set WINSOR_LIMIT=8
set PROMETHEUS_PORT=8009
set LOG_LEVEL=INFO
set OUTPUT_DIR=data/ofi_cvd
set ARTIFACTS_DIR=artifacts

echo 配置参数:
echo   SYMBOLS=%SYMBOLS%
echo   RUN_HOURS=%RUN_HOURS%
echo   PARQUET_ROTATE_SEC=%PARQUET_ROTATE_SEC%
echo   Z_MODE=%Z_MODE%
echo   SCALE_MODE=%SCALE_MODE%
echo   MAD_MULTIPLIER=%MAD_MULTIPLIER%
echo   SCALE_FAST_WEIGHT=%SCALE_FAST_WEIGHT%
echo   HALF_LIFE_SEC=%HALF_LIFE_SEC%
echo   WINSOR_LIMIT=%WINSOR_LIMIT%
echo   PROMETHEUS_PORT=%PROMETHEUS_PORT%
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import pandas, pyarrow, prometheus_client, websockets" >nul 2>&1
if errorlevel 1 (
    echo 错误: 缺少必要的依赖包
    echo 请运行: pip install pandas pyarrow prometheus_client websockets
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%ARTIFACTS_DIR%" mkdir "%ARTIFACTS_DIR%"

echo 开始数据采集...
echo 监控地址: http://localhost:%PROMETHEUS_PORT%/metrics
echo 按 Ctrl+C 停止采集
echo.

REM 运行采集脚本
cd /d "%~dp0.."
python examples/run_realtime_harvest.py

echo.
echo 数据采集完成
echo 输出目录: %OUTPUT_DIR%
echo 日志目录: %ARTIFACTS_DIR%\run_logs
echo 报告目录: %ARTIFACTS_DIR%\dq_reports
echo.

REM 运行数据质量验证
echo 运行数据质量验证...
python scripts/validate_ofi_cvd_harvest.py --base-dir "%OUTPUT_DIR%" --output-dir "%ARTIFACTS_DIR%\dq_reports"

echo.
echo 任务完成！
pause
