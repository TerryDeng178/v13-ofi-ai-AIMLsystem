@echo off
REM 本周三件事快速启动脚本 (Windows版本)

echo 🎯 本周三件事快速启动脚本
echo ================================

REM 检查参数
if "%~2"=="" (
    echo 用法: %0 ^<数据路径^> ^<输出目录^> [任务名称]
    echo.
    echo 参数说明:
    echo   数据路径    - 回放数据文件或目录
    echo   输出目录    - 结果输出目录
    echo   任务名称    - 可选，指定运行的任务
    echo.
    echo 可用任务:
    echo   tune_params        - 参数调优
    echo   score_monotonicity - 单调性验证
    echo   metrics_alignment  - 指标对齐
    echo   config_hot_update  - 配置热更新
    echo   all               - 运行所有任务（默认）
    echo.
    echo 示例:
    echo   %0 data\replay\btcusdt_2025-10-01_2025-10-19.parquet runs\weekly_tasks
    echo   %0 data\replay\btcusdt_2025-10-01_2025-10-19.parquet runs\tune_only tune_params
    exit /b 1
)

set DATA_PATH=%~1
set OUTPUT_DIR=%~2
set TASK=%~3
if "%TASK%"=="" set TASK=all

echo 📁 数据路径: %DATA_PATH%
echo 📁 输出目录: %OUTPUT_DIR%
echo 📋 任务: %TASK%
echo.

REM 检查数据路径
if not exist "%DATA_PATH%" (
    echo ❌ 数据路径不存在: %DATA_PATH%
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或不在PATH中
    exit /b 1
)

REM 检查必要的Python包
echo 🔍 检查Python依赖...
python -c "import pandas, numpy, scipy, sklearn, yaml, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo ❌ 缺少必要的Python包，请安装：
    echo    pip install pandas numpy scipy scikit-learn pyyaml matplotlib
    exit /b 1
)

REM 检查脚本文件
echo 🔍 检查脚本文件...
set SCRIPTS_DIR=%~dp0
set REQUIRED_SCRIPTS=tune_divergence.py score_monotonicity.py metrics_alignment.py config_hot_update.py run_weekly_tasks.py

for %%s in (%REQUIRED_SCRIPTS%) do (
    if not exist "%SCRIPTS_DIR%%%s" (
        echo ❌ 脚本文件不存在: %SCRIPTS_DIR%%%s
        exit /b 1
    )
)

echo ✅ 环境检查通过
echo.

REM 运行任务
if "%TASK%"=="all" (
    echo 🚀 开始执行所有任务...
    python "%SCRIPTS_DIR%run_weekly_tasks.py" --data "%DATA_PATH%" --out "%OUTPUT_DIR%"
) else (
    echo 🚀 开始执行任务: %TASK%
    python "%SCRIPTS_DIR%run_weekly_tasks.py" --data "%DATA_PATH%" --out "%OUTPUT_DIR%" --task "%TASK%"
)

echo.
echo 🎉 任务执行完成！
echo 📊 结果目录: %OUTPUT_DIR%
echo.

REM 显示结果摘要
if exist "%OUTPUT_DIR%\weekly_tasks_report.json" (
    echo 📋 执行摘要:
    python -c "import json; report=json.load(open('%OUTPUT_DIR%\weekly_tasks_report.json', 'r')); print('总体状态: ✅ 成功' if report['overall_success'] else '总体状态: ❌ 失败'); [print(f'{\"✅\" if task_result[\"status\"] == \"success\" else \"❌\"} {task_name}: {task_result[\"status\"]} ({task_result.get(\"duration\", 0):.1f}s)') for task_name, task_result in report['tasks'].items()]"
)

echo.
echo 📚 更多信息请查看:
echo    - 调优指南: docs\divergence_tuning.md
echo    - 验收标准: docs\weekly_tasks_acceptance.md
echo    - 结果报告: %OUTPUT_DIR%\weekly_tasks_report.json
