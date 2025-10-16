#!/usr/bin/env python3
"""
批量生成剩余的任务卡文件
本脚本用于快速生成V13项目的所有任务卡文件
"""

import os
from pathlib import Path

# 定义任务模板
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

### 测试项1: {test_name}
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

# 定义所有任务
TASKS = [
    # Stage1 remaining tasks
    {
        "task_id": "1.1.3",
        "task_name": "实现订单簿数据解析",
        "stage_name": "阶段1 - 真实OFI核心",
        "priority": "高",
        "estimated_time": "1小时",
        "objective": "解析币安WebSocket返回的订单簿JSON数据，提取bids、asks和时间戳信息。",
        "checklist": "- [ ] 解析订单簿JSON数据\n- [ ] 提取bids（买单）数据\n- [ ] 提取asks（卖单）数据\n- [ ] 提取时间戳",
        "allowed_files": "- `v13_ofi_ai_system/src/binance_websocket_client.py` (修改)",
        "prerequisite": "Task_1.1.2",
        "dependencies": "json (标准库)",
        "validation": "1. 数据格式正确\n2. 包含5档买卖单\n3. 时间戳准确",
        "test_name": "数据解析验证",
        "notes": "- 必须解析真实的币安数据\n- 数据结构要清晰"
    },
    # ... 更多任务定义
]

def generate_task_file(task_info, output_dir):
    """生成单个任务文件"""
    task_id = task_info["task_id"]
    filename = f"Task_{task_id}_{task_info['task_name']}.md"
    filepath = os.path.join(output_dir, filename)
    
    content = TASK_TEMPLATE.format(**task_info)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 创建任务文件: {filename}")

def main():
    """主函数"""
    base_dir = Path(__file__).parent
    
    for task in TASKS:
        stage_dir = base_dir / task["stage_name"].split(" - ")[1]
        stage_dir.mkdir(exist_ok=True)
        generate_task_file(task, stage_dir)
    
    print(f"\n✅ 共生成 {len(TASKS)} 个任务文件")

if __name__ == "__main__":
    main()

