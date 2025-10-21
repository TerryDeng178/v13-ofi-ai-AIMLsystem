@echo off
echo 启动稳定版OFI+CVD数据采集...
echo.

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=1
set OUTPUT_DIR=data/ofi_cvd

echo 配置参数:
echo - 交易对: %SYMBOLS%
echo - 运行时间: %RUN_HOURS% 小时
echo - 输出目录: %OUTPUT_DIR%
echo.

REM 启动稳定版采集脚本
echo 启动稳定版数据采集脚本...
python examples/run_stable_harvest.py

echo.
echo 稳定版数据采集完成！
pause

