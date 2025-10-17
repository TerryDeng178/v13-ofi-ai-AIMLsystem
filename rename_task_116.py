# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# 设置UTF-8输出
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 项目根目录
base_dir = Path(__file__).parent / "v13_ofi_ai_system" / "TASKS" / "Stage1_真实OFI核心"

# 旧文件名
old_name = "Task_1.1.6_测试和验证.md"
# 新文件名
new_name = "✅Task_1.1.6_测试和验证.md"

old_path = base_dir / old_name
new_path = base_dir / new_name

try:
    print("Checking file...")
    print(f"Old path exists: {old_path.exists()}")
    
    if old_path.exists():
        old_path.rename(new_path)
        print("Success! File renamed with check mark prefix.")
    else:
        print("Error: File not found")
except Exception as e:
    print(f"Error: {e}")

