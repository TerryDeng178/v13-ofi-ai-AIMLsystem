#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析OFI测试结果，诊断问题并提供改进建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_ofi_test_results():
    """分析OFI测试结果"""
    
    print("=" * 60)
    print("OFI测试结果分析")
    print("=" * 60)
    
    # 读取测试报告
    report_file = "examples/figs/TASK_1_2_5_REAL_DATA_TEST_BTCUSDT.md"
    
    if not os.path.exists(report_file):
        print(f"错误: 找不到测试报告文件 {report_file}")
        return
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("测试报告摘要:")
    print("-" * 40)
    
    # 提取关键信息
    lines = content.split('\n')
    for line in lines:
        if '总数据点' in line or '有效记录' in line or '坏数据点' in line or 'Warmup期' in line or '标准差为0' in line:
            print(line.strip())
        elif '最大间隔' in line or 'P99间隔' in line:
            print(line.strip())
        elif '中位数' in line or 'IQR' in line or 'P95' in line or 'P99' in line:
            print(line.strip())
        elif 'P(|z| > 2)' in line or 'P(|z| > 3)' in line or 'P(|z| > 4)' in line:
            print(line.strip())
        elif '总体通过率' in line:
            print(line.strip())
    
    print("\n" + "=" * 60)
    print("问题诊断")
    print("=" * 60)
    
    # 问题1: 标准差为0问题
    print("\n1. 标准差为0问题 (99.27%的数据点)")
    print("   原因: OFI计算器使用了标准差下限保护 (std_floor=1e-4)")
    print("   影响: 所有Z-score都被设为0，无法进行有效的标准化")
    print("   建议: 调整std_floor参数或增加数据量以建立更稳定的统计基线")
    
    # 问题2: 数据点不足
    print("\n2. 数据点不足 (38,779 < 300,000)")
    print("   原因: 测试只使用了100,000个数据点，且大部分被过滤掉")
    print("   影响: 无法建立足够的统计基线")
    print("   建议: 增加测试数据量，使用更多orderbook数据")
    
    # 问题3: Z-score全部为0
    print("\n3. Z-score全部为0")
    print("   原因: 由于标准差为0，所有Z-score都被设为0")
    print("   影响: 无法进行有效的异常检测和信号分析")
    print("   建议: 修复标准差计算问题")
    
    print("\n" + "=" * 60)
    print("改进建议")
    print("=" * 60)
    
    print("\n1. 调整OFI计算器参数:")
    print("   - 降低std_floor: 从1e-4降到1e-6")
    print("   - 增加z_window: 从80增加到200-300")
    print("   - 调整winsorize_ofi_delta: 从0.9降到0.5")
    
    print("\n2. 增加测试数据量:")
    print("   - 使用更多orderbook文件")
    print("   - 延长测试时间")
    print("   - 确保有足够的数据建立统计基线")
    
    print("\n3. 优化数据处理:")
    print("   - 检查orderbook数据质量")
    print("   - 确保价格和数量数据有效")
    print("   - 添加数据预处理步骤")
    
    print("\n4. 监控和调试:")
    print("   - 添加详细的调试日志")
    print("   - 监控OFI值的分布")
    print("   - 跟踪标准差计算过程")
    
    print("\n" + "=" * 60)
    print("下一步行动")
    print("=" * 60)
    
    print("\n1. 立即修复:")
    print("   - 调整RealOFICalculator的std_floor参数")
    print("   - 增加测试数据量到至少100,000个有效数据点")
    
    print("\n2. 重新测试:")
    print("   - 使用修复后的参数重新运行测试")
    print("   - 确保Z-score分布正常")
    print("   - 验证验收标准通过率")
    
    print("\n3. 长期优化:")
    print("   - 建立更稳定的统计基线")
    print("   - 优化OFI计算算法")
    print("   - 改进数据处理流程")

if __name__ == "__main__":
    analyze_ofi_test_results()
