#!/usr/bin/env python3
"""
修复Unicode字符问题
"""

import re

def fix_unicode_in_file(file_path):
    """修复文件中的Unicode字符"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换常见的Unicode字符
    replacements = {
        '📊': '[CHART]',
        '🔍': '[SEARCH]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '📈': '[GRAPH]',
        '📋': '[LIST]',
        '🏷️': '[TAG]',
        '📡': '[SIGNAL]',
        '🚀': '[ROCKET]',
        '📁': '[FOLDER]',
        '🔧': '[TOOL]',
        '🧠': '[BRAIN]',
        '📝': '[NOTE]',
        '🎯': '[TARGET]',
        '📦': '[PACKAGE]',
        '🔗': '[LINK]',
        '⚠️': '[WARNING]',
        '🧪': '[TEST]',
        '🔄': '[REFRESH]',
        '📊': '[CHART]',
        '🎉': '[CELEBRATE]'
    }
    
    for unicode_char, replacement in replacements.items():
        content = content.replace(unicode_char, replacement)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {file_path}")

if __name__ == "__main__":
    files_to_fix = [
        'analysis/ofi_cvd_signal_eval.py',
        'analysis/utils_labels.py',
        'analysis/plots.py'
    ]
    
    for file_path in files_to_fix:
        try:
            fix_unicode_in_file(file_path)
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
