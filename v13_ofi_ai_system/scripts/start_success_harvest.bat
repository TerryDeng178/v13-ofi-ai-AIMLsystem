@echo off
echo 启动成功版OFI+CVD数据采集（基于Task 1.2.5成功实现）...
echo.

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=1
set OUTPUT_DIR=data/ofi_cvd

echo 配置参数:
echo - 交易对: %SYMBOLS%
echo - 运行时间: %RUN_HOURS% 小时
echo - 输出目录: %OUTPUT_DIR%
echo - 订单簿流: Binance Futures (wss://fstream.binancefuture.com/stream?streams=...)
echo - 交易流: Binance Spot (wss://stream.binance.com)
echo - 消息解析: 基于Task 1.2.5成功实现
echo.

REM 启动成功版采集脚本
echo 启动成功版数据采集脚本...
python examples/run_success_harvest.py

echo.
echo 成功版数据采集完成！
pause

