#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成V13项目所有剩余任务卡文件
包含Stage1剩余11个任务 + Stage2-4的所有任务
"""

import os
from pathlib import Path

# 任务卡模板
TASK_TEMPLATE = """# Task {task_id}: {task_name}

## 📋 任务信息
- **任务编号**: Task_{task_id}
- **所属阶段**: {stage_name}
- **任务状态**: ⏳ 待开始
- **优先级**: {priority}
- **预计时间**: {estimated_time}
- **实际时间**: (完成后填写)

## 🎯 任务目标
{objective}

## 📝 任务清单
{checklist}

## 📦 Allowed Files
{allowed_files}

## 📚 依赖项
- **前置任务**: {prerequisite}
- **依赖包**: {dependencies}

## ✅ 验证标准
{validation}

## 🧪 测试结果
**测试执行时间**: (填写)

### 测试项1: {test_name_1}
- **状态**: ⏳ 未测试
- **结果**: (填写)
- **测试方法**: (填写)

### 测试项2: {test_name_2}
- **状态**: ⏳ 未测试
- **结果**: (填写)
- **测试方法**: (填写)

## 📊 DoD检查清单
- [ ] 代码无语法错误
- [ ] 通过 lint 检查
- [ ] 通过所有测试
- [ ] 无 mock/占位/跳过
- [ ] 产出真实验证结果
- [ ] 性能达标（如适用）
- [ ] 更新相关文档

## 📝 执行记录
**开始时间**: (填写)  
**完成时间**: (填写)  
**执行者**: AI Assistant

### 遇到的问题
- (记录问题)

### 解决方案
- (记录解决方案)

### 经验教训
- (记录经验)

## 🔗 相关链接
- 上一个任务: {prev_task}
- 下一个任务: {next_task}
- 阶段总览: [📋V13_TASK_CARD.md](../../📋V13_TASK_CARD.md)
- 任务系统: [TASKS/README.md](../README.md)

## ⚠️ 注意事项
{notes}

---
**任务状态**: ⏳ 待{prerequisite}完成后开始  
**质量评分**: (完成后填写)  
**是否可以继续下一个任务**: ❓ 待测试通过后确定
"""

# 定义所有任务信息
TASKS_DATA = [
    # ====== Stage1剩余任务 ======
    {
        "task_id": "1.1.6",
        "task_name": "测试和验证",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "1-2小时",
        "objective": "运行WebSocket客户端，连续接收1小时数据，验证数据完整性和延迟。",
        "checklist": "- [ ] 运行WebSocket客户端\n- [ ] 连续接收1小时数据\n- [ ] 验证数据完整性 >95%\n- [ ] 测量延迟 <500ms",
        "allowed_files": "- `v13_ofi_ai_system/src/binance_websocket_client.py` (验证)\n- `v13_ofi_ai_system/examples/` (测试脚本)",
        "prerequisite": "Task_1.1.5",
        "dependencies": "无",
        "validation": "1. 连续接收1小时以上\n2. 数据完整性 >95%\n3. 延迟 <500ms\n4. 无连接中断",
        "test_name_1": "连续运行测试",
        "test_name_2": "数据完整性测试",
        "notes": "- 必须使用真实币安数据\n- 测试时间至少1小时\n- 记录所有异常情况",
        "prev_task": "[Task_1.1.5_实现实时打印和日志](./Task_1.1.5_实现实时打印和日志.md)",
        "next_task": "[Task_1.2.1_创建OFI计算器基础类](./Task_1.2.1_创建OFI计算器基础类.md)"
    },
    {
        "task_id": "1.2.1",
        "task_name": "创建OFI计算器基础类",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "30分钟",
        "objective": "创建OFI计算器的基础类结构，定义档位权重和历史数据缓存。",
        "checklist": "- [ ] 创建文件 `v13_ofi_ai_system/src/real_ofi_calculator.py`\n- [ ] 实现 `RealOFICalculator` 类基础结构\n- [ ] 定义档位权重: [0.4, 0.25, 0.2, 0.1, 0.05]\n- [ ] 初始化历史数据缓存",
        "allowed_files": "- `v13_ofi_ai_system/src/real_ofi_calculator.py` (新建)",
        "prerequisite": "Task_1.1.6",
        "dependencies": "numpy (已在requirements.txt)",
        "validation": "1. 文件创建成功\n2. 类结构正确\n3. 参数定义合理\n4. 无语法错误",
        "test_name_1": "文件创建验证",
        "test_name_2": "类结构验证",
        "notes": "- 档位权重必须符合OFI算法\n- 缓存大小要合理",
        "prev_task": "[Task_1.1.6_测试和验证](./Task_1.1.6_测试和验证.md)",
        "next_task": "[Task_1.2.2_实现OFI核心算法](./Task_1.2.2_实现OFI核心算法.md)"
    },
    {
        "task_id": "1.2.2",
        "task_name": "实现OFI核心算法",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "2小时",
        "objective": "实现OFI核心计算算法，包括买卖单变化和档位权重应用。",
        "checklist": "- [ ] 实现 `calculate_ofi()` 方法\n- [ ] 计算买单变化: ΔBid\n- [ ] 计算卖单变化: ΔAsk\n- [ ] 应用档位权重\n- [ ] 计算最终OFI值",
        "allowed_files": "- `v13_ofi_ai_system/src/real_ofi_calculator.py` (修改)",
        "prerequisite": "Task_1.2.1",
        "dependencies": "numpy",
        "validation": "1. 算法实现正确\n2. OFI值在合理范围（-5到+5）\n3. 与公式一致\n4. 计算效率高",
        "test_name_1": "算法正确性验证",
        "test_name_2": "OFI值范围验证",
        "notes": "- 必须严格按照OFI公式实现\n- 注意数值稳定性",
        "prev_task": "[Task_1.2.1_创建OFI计算器基础类](./Task_1.2.1_创建OFI计算器基础类.md)",
        "next_task": "[Task_1.2.3_实现OFI_Z-score标准化](./Task_1.2.3_实现OFI_Z-score标准化.md)"
    },
    {
        "task_id": "1.2.3",
        "task_name": "实现OFI Z-score标准化",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "1小时",
        "objective": "实现OFI的Z-score标准化，使用滚动窗口计算。",
        "checklist": "- [ ] 实现 `get_ofi_zscore()` 方法\n- [ ] 计算OFI均值和标准差\n- [ ] 滚动窗口大小: 1200个数据点\n- [ ] 计算Z-score",
        "allowed_files": "- `v13_ofi_ai_system/src/real_ofi_calculator.py` (修改)",
        "prerequisite": "Task_1.2.2",
        "dependencies": "numpy",
        "validation": "1. Z-score分布接近标准正态分布\n2. 强信号（|Z|>2）频率 5-10%\n3. 计算稳定\n4. 窗口大小合理",
        "test_name_1": "Z-score分布验证",
        "test_name_2": "信号频率验证",
        "notes": "- 窗口大小影响信号灵敏度\n- 注意边界情况处理",
        "prev_task": "[Task_1.2.2_实现OFI核心算法](./Task_1.2.2_实现OFI核心算法.md)",
        "next_task": "[Task_1.2.4_集成WebSocket和OFI计算](./Task_1.2.4_集成WebSocket和OFI计算.md)"
    },
    {
        "task_id": "1.2.4",
        "task_name": "集成WebSocket和OFI计算",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "1小时",
        "objective": "创建集成示例，实时计算OFI并打印结果。",
        "checklist": "- [ ] 创建集成示例 `v13_ofi_ai_system/examples/run_realtime_ofi.py`\n- [ ] 连接WebSocket数据流\n- [ ] 实时计算OFI\n- [ ] 实时打印OFI和Z-score",
        "allowed_files": "- `v13_ofi_ai_system/examples/run_realtime_ofi.py` (新建)",
        "prerequisite": "Task_1.2.3",
        "dependencies": "所有已实现模块",
        "validation": "1. 实时OFI计算正常\n2. 数据流畅不中断\n3. 无内存泄漏\n4. 计算延迟低",
        "test_name_1": "实时计算验证",
        "test_name_2": "性能验证",
        "notes": "- 集成时注意模块耦合\n- 性能优化很重要",
        "prev_task": "[Task_1.2.3_实现OFI_Z-score标准化](./Task_1.2.3_实现OFI_Z-score标准化.md)",
        "next_task": "[Task_1.2.5_OFI计算测试](./Task_1.2.5_OFI计算测试.md)"
    },
    {
        "task_id": "1.2.5",
        "task_name": "OFI计算测试",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "2-4小时",
        "objective": "运行实时OFI计算，收集并分析OFI数据。",
        "checklist": "- [ ] 运行实时OFI计算\n- [ ] 收集2-4小时OFI数据\n- [ ] 分析OFI分布\n- [ ] 验证OFI合理性",
        "allowed_files": "- `v13_ofi_ai_system/examples/run_realtime_ofi.py` (运行)\n- `v13_ofi_ai_system/data/` (数据文件)",
        "prerequisite": "Task_1.2.4",
        "dependencies": "无",
        "validation": "1. OFI值范围合理\n2. Z-score分布正态\n3. 无异常值\n4. 数据质量高",
        "test_name_1": "OFI分布验证",
        "test_name_2": "数据质量验证",
        "notes": "- 测试时间要足够长\n- 记录所有异常情况",
        "prev_task": "[Task_1.2.4_集成WebSocket和OFI计算](./Task_1.2.4_集成WebSocket和OFI计算.md)",
        "next_task": "[Task_1.3.1_收集历史OFI数据](./Task_1.3.1_收集历史OFI数据.md)"
    },
    {
        "task_id": "1.3.1",
        "task_name": "收集历史OFI数据",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "24-72小时（自动运行）",
        "objective": "连续运行OFI计算24-72小时，收集完整的历史数据。",
        "checklist": "- [ ] 连续运行OFI计算24-72小时\n- [ ] 保存完整的OFI历史数据\n- [ ] 同步保存价格数据",
        "allowed_files": "- `v13_ofi_ai_system/examples/run_realtime_ofi.py` (运行)\n- `v13_ofi_ai_system/data/` (数据文件)",
        "prerequisite": "Task_1.2.5",
        "dependencies": "无",
        "validation": "1. 数据完整无缺失\n2. 至少1000个OFI信号\n3. 数据质量高\n4. 价格数据同步",
        "test_name_1": "数据完整性验证",
        "test_name_2": "信号数量验证",
        "notes": "- 需要长时间运行\n- 确保系统稳定性",
        "prev_task": "[Task_1.2.5_OFI计算测试](./Task_1.2.5_OFI计算测试.md)",
        "next_task": "[Task_1.3.2_创建OFI信号分析工具](./Task_1.3.2_创建OFI信号分析工具.md)"
    },
    {
        "task_id": "1.3.2",
        "task_name": "创建OFI信号分析工具",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "2小时",
        "objective": "创建OFI信号分析工具，评估信号预测能力。",
        "checklist": "- [ ] 创建文件 `v13_ofi_ai_system/tests/test_ofi_signal_validity.py`\n- [ ] 实现OFI信号提取功能\n- [ ] 实现价格变化计算功能\n- [ ] 实现准确率评估功能",
        "allowed_files": "- `v13_ofi_ai_system/tests/test_ofi_signal_validity.py` (新建)",
        "prerequisite": "Task_1.3.1",
        "dependencies": "pandas, numpy",
        "validation": "1. 工具功能完整\n2. 分析逻辑正确\n3. 输出清晰\n4. 无错误",
        "test_name_1": "功能完整性验证",
        "test_name_2": "分析逻辑验证",
        "notes": "- 分析工具要严谨\n- 输出要清晰易懂",
        "prev_task": "[Task_1.3.1_收集历史OFI数据](./Task_1.3.1_收集历史OFI数据.md)",
        "next_task": "[Task_1.3.3_分析OFI预测能力](./Task_1.3.3_分析OFI预测能力.md)"
    },
    {
        "task_id": "1.3.3",
        "task_name": "分析OFI预测能力",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "2小时",
        "objective": "分析OFI信号的预测能力，计算准确率。",
        "checklist": "- [ ] 提取强买入信号（OFI Z > 2）\n- [ ] 提取强卖出信号（OFI Z < -2）\n- [ ] 计算后续价格变化（5秒、10秒、30秒）\n- [ ] 计算信号准确率",
        "allowed_files": "- `v13_ofi_ai_system/tests/test_ofi_signal_validity.py` (运行)\n- `v13_ofi_ai_system/data/` (数据文件)",
        "prerequisite": "Task_1.3.2",
        "dependencies": "pandas, numpy",
        "validation": "1. 买入信号准确率 >55%\n2. 卖出信号准确率 >55%\n3. 强信号准确率 >60%\n4. 分析结果可靠",
        "test_name_1": "信号准确率验证",
        "test_name_2": "分析结果验证",
        "notes": "- 准确率是关键指标\n- 如果<55%需要优化算法",
        "prev_task": "[Task_1.3.2_创建OFI信号分析工具](./Task_1.3.2_创建OFI信号分析工具.md)",
        "next_task": "[Task_1.3.4_生成OFI验证报告](./Task_1.3.4_生成OFI验证报告.md)"
    },
    {
        "task_id": "1.3.4",
        "task_name": "生成OFI验证报告",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "1小时",
        "objective": "创建OFI验证报告，记录所有关键发现。",
        "checklist": "- [ ] 创建 `v13_ofi_ai_system/reports/Stage1_OFI_Validation_Report.md`\n- [ ] 记录OFI信号统计\n- [ ] 记录预测准确率\n- [ ] 记录关键发现和建议",
        "allowed_files": "- `v13_ofi_ai_system/reports/Stage1_OFI_Validation_Report.md` (新建)",
        "prerequisite": "Task_1.3.3",
        "dependencies": "无",
        "validation": "1. 报告完整\n2. 数据真实\n3. 结论明确\n4. 建议可行",
        "test_name_1": "报告完整性验证",
        "test_name_2": "数据真实性验证",
        "notes": "- 报告要客观真实\n- 数据不能造假",
        "prev_task": "[Task_1.3.3_分析OFI预测能力](./Task_1.3.3_分析OFI预测能力.md)",
        "next_task": "[Task_1.3.5_阶段1总结和决策](./Task_1.3.5_阶段1总结和决策.md)"
    },
    {
        "task_id": "1.3.5",
        "task_name": "阶段1总结和决策",
        "stage_name": "阶段1 - 真实OFI核心",
        "stage_dir": "Stage1_真实OFI核心",
        "priority": "高",
        "estimated_time": "1小时",
        "objective": "评估阶段1成果，决定是否进入阶段2。",
        "checklist": "- [ ] 评估OFI信号有效性\n- [ ] 决定是否进入阶段2\n- [ ] 调整阈值参数（如需要）",
        "allowed_files": "- `v13_ofi_ai_system/reports/Stage1_OFI_Validation_Report.md` (查看)",
        "prerequisite": "Task_1.3.4",
        "dependencies": "无",
        "validation": "1. 如果OFI准确率 >55%，进入阶段2\n2. 如果OFI准确率 <55%，优化算法\n3. 用户确认后继续",
        "test_name_1": "决策标准验证",
        "test_name_2": "用户确认",
        "notes": "- 必须基于真实数据决策\n- 必须经用户确认",
        "prev_task": "[Task_1.3.4_生成OFI验证报告](./Task_1.3.4_生成OFI验证报告.md)",
        "next_task": "[Task_2.1.1_创建测试网交易客户端](../Stage2_简单真实交易/Task_2.1.1_创建测试网交易客户端.md)"
    },
]

def create_task_file(task_data):
    """创建单个任务文件"""
    base_dir = Path(__file__).parent
    stage_dir = base_dir / task_data["stage_dir"]
    stage_dir.mkdir(exist_ok=True)
    
    filename = f"Task_{task_data['task_id']}_{task_data['task_name']}.md"
    filepath = stage_dir / filename
    
    content = TASK_TEMPLATE.format(**task_data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 创建任务文件: {filename}")

def main():
    """主函数"""
    print("开始批量创建任务文件...\n")
    
    for task in TASKS_DATA:
        create_task_file(task)
    
    print(f"\n✅ 成功创建 {len(TASKS_DATA)} 个任务文件")
    print(f"位置: v13_ofi_ai_system/TASKS/")

if __name__ == "__main__":
    main()

