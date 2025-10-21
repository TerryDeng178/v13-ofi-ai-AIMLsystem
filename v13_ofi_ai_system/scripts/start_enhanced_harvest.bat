@echo off
echo 启动增强版OFI+CVD数据采集（基于Task 1.2.4成功实现）...
echo.

REM 设置环境变量
set SYMBOLS=BTCUSDT,ETHUSDT
set RUN_HOURS=1
set OUTPUT_DIR=data/ofi_cvd

echo 配置参数:
echo - 交易对: %SYMBOLS%
echo - 运行时间: %RUN_HOURS% 小时
echo - 输出目录: %OUTPUT_DIR%
echo - OFI计算器: 基于Task 1.2.4成功实现
echo.

REM 启动增强版采集脚本
echo 启动增强版数据采集脚本...
python examples/run_enhanced_harvest.py

echo.
echo 增强版数据采集完成！
pause

