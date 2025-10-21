@echo off
REM Task_1.2.13 修正评估执行脚本
REM 按照任务卡要求进行完整的修正评估流程

echo ========================================
echo Task_1.2.13 修正评估执行脚本
echo ========================================

REM 设置环境变量
set SYMBOL=BTCUSDT
set DURATION=3600
set OUTPUT_DIR=data/cvd_corrected_evaluation

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo 步骤1: 修正网格搜索 (3x3x3, 15分钟每个测试)
echo ========================================
python examples/cvd_corrected_grid_search.py
if %ERRORLEVEL% neq 0 (
    echo 网格搜索失败，退出
    exit /b 1
)

echo.
echo 步骤2: 软冻结A/B测试 (20+20分钟)
echo ========================================
python examples/cvd_soft_freeze_ab_test.py
if %ERRORLEVEL% neq 0 (
    echo A/B测试失败，退出
    exit /b 1
)

echo.
echo 步骤3: 60分钟最终验证
echo ========================================
python examples/cvd_corrected_evaluation.py
if %ERRORLEVEL% neq 0 (
    echo 最终验证失败，退出
    exit /b 1
)

echo.
echo 步骤4: 生成分析报告
echo ========================================
python generate_analysis_charts.py
if %ERRORLEVEL% neq 0 (
    echo 分析报告生成失败，但继续
)

echo.
echo 步骤5: 检查回滚条件
echo ========================================
python -c "
import json
import os
from pathlib import Path

# 检查最新结果
output_dir = Path('data/cvd_corrected_evaluation')
latest_file = max(output_dir.glob('*.json'), key=os.path.getctime, default=None)

if latest_file:
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    overall = results.get('overall', {})
    p_z_gt_2 = overall.get('p_z_gt_2', 1.0)
    p_z_gt_3 = overall.get('p_z_gt_3', 1.0)
    
    print(f'P(|Z|>2): {p_z_gt_2:.4f}')
    print(f'P(|Z|>3): {p_z_gt_3:.4f}')
    
    if p_z_gt_2 > 0.08:
        print('警告: P(|Z|>2) > 8%%，建议回滚')
    else:
        print('P(|Z|>2) 在可接受范围内')
        
    if p_z_gt_3 > 0.02:
        print('警告: P(|Z|>3) > 2%%，建议回滚')
    else:
        print('P(|Z|>3) 在可接受范围内')
else:
    print('未找到评估结果文件')
"

echo.
echo ========================================
echo 修正评估完成
echo ========================================
echo 请检查以下文件:
echo - data/cvd_corrected_grid_search/grid_rank_table_*.csv
echo - data/cvd_soft_freeze_ab/soft_freeze_ab_*.json
echo - data/cvd_corrected_evaluation/corrected_evaluation_*.json
echo - docs/reports/task_1_2_13_analysis_charts.png
echo ========================================

pause
