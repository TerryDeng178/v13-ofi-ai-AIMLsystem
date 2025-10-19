#!/bin/bash
# 本周三件事快速启动脚本

set -e

echo "🎯 本周三件事快速启动脚本"
echo "================================"

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <数据路径> <输出目录> [任务名称]"
    echo ""
    echo "参数说明:"
    echo "  数据路径    - 回放数据文件或目录"
    echo "  输出目录    - 结果输出目录"
    echo "  任务名称    - 可选，指定运行的任务"
    echo ""
    echo "可用任务:"
    echo "  tune_params        - 参数调优"
    echo "  score_monotonicity - 单调性验证"
    echo "  metrics_alignment  - 指标对齐"
    echo "  config_hot_update  - 配置热更新"
    echo "  all               - 运行所有任务（默认）"
    echo ""
    echo "示例:"
    echo "  $0 data/replay/btcusdt_2025-10-01_2025-10-19.parquet runs/weekly_tasks"
    echo "  $0 data/replay/btcusdt_2025-10-01_2025-10-19.parquet runs/tune_only tune_params"
    exit 1
fi

DATA_PATH="$1"
OUTPUT_DIR="$2"
TASK="${3:-all}"

echo "📁 数据路径: $DATA_PATH"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📋 任务: $TASK"
echo ""

# 检查数据路径
if [ ! -e "$DATA_PATH" ]; then
    echo "❌ 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查必要的Python包
echo "🔍 检查Python依赖..."
python -c "import pandas, numpy, scipy, sklearn, yaml, matplotlib" 2>/dev/null || {
    echo "❌ 缺少必要的Python包，请安装："
    echo "   pip install pandas numpy scipy scikit-learn pyyaml matplotlib"
    exit 1
}

# 检查脚本文件
echo "🔍 检查脚本文件..."
SCRIPTS_DIR="$(dirname "$0")"
REQUIRED_SCRIPTS=(
    "tune_divergence.py"
    "score_monotonicity.py"
    "metrics_alignment.py"
    "config_hot_update.py"
    "run_weekly_tasks.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPTS_DIR/$script" ]; then
        echo "❌ 脚本文件不存在: $SCRIPTS_DIR/$script"
        exit 1
    fi
done

echo "✅ 环境检查通过"
echo ""

# 运行任务
if [ "$TASK" = "all" ]; then
    echo "🚀 开始执行所有任务..."
    python "$SCRIPTS_DIR/run_weekly_tasks.py" \
        --data "$DATA_PATH" \
        --out "$OUTPUT_DIR"
else
    echo "🚀 开始执行任务: $TASK"
    python "$SCRIPTS_DIR/run_weekly_tasks.py" \
        --data "$DATA_PATH" \
        --out "$OUTPUT_DIR" \
        --task "$TASK"
fi

echo ""
echo "🎉 任务执行完成！"
echo "📊 结果目录: $OUTPUT_DIR"
echo ""

# 显示结果摘要
if [ -f "$OUTPUT_DIR/weekly_tasks_report.json" ]; then
    echo "📋 执行摘要:"
    python -c "
import json
with open('$OUTPUT_DIR/weekly_tasks_report.json', 'r') as f:
    report = json.load(f)
print(f'总体状态: {\"✅ 成功\" if report[\"overall_success\"] else \"❌ 失败\"}')
for task_name, task_result in report['tasks'].items():
    status_icon = '✅' if task_result['status'] == 'success' else '❌'
    duration = task_result.get('duration', 0)
    print(f'{status_icon} {task_name}: {task_result[\"status\"]} ({duration:.1f}s)')
"
fi

echo ""
echo "📚 更多信息请查看:"
echo "   - 调优指南: docs/divergence_tuning.md"
echo "   - 验收标准: docs/weekly_tasks_acceptance.md"
echo "   - 结果报告: $OUTPUT_DIR/weekly_tasks_report.json"
