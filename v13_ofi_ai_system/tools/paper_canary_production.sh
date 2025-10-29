#!/bin/bash
# -*- coding: utf-8 -*-
"""
生产环境纸上交易金丝雀测试脚本（60分钟完整版）
用于合并后主线验证
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "生产环境纸上交易金丝雀测试（60分钟完整版）"
echo "============================================================"
echo ""
echo "目标："
echo "  - error_rate=0"
echo "  - latency.p99 < 500ms"
echo "  - 指纹无漂移"
echo "  - 4类信号在活跃时段触发率 > 0"
echo ""
echo "开始时间: $(date)"
echo ""

python tools/paper_canary.py --mins 60 --p99-limit-ms 500

echo ""
echo "完成时间: $(date)"
echo "============================================================"

# 检查结果
if [ -f "reports/paper_canary_report.json" ]; then
    python -c "
import json
with open('reports/paper_canary_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
if data.get('overall_pass'):
    print('[SUCCESS] 生产金丝雀测试通过')
    exit(0)
else:
    print('[FAIL] 生产金丝雀测试失败')
    print(json.dumps(data, indent=2, ensure_ascii=False))
    exit(1)
"
else
    echo "[ERROR] 报告文件不存在"
    exit 1
fi

