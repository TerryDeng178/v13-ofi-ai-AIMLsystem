"""
调试输入验证
"""

import sys
import os
import io
import math

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ofi_cvd_divergence import DivergenceDetector, DivergenceConfig


def debug_input_validation():
    """调试输入验证"""
    print("=== 调试输入验证 ===\n")
    
    # 创建配置
    config = DivergenceConfig(
        swing_L=2,
        min_separation=1,
        cooldown_secs=0.1,
        warmup_min=5,
        z_hi=1.0,
        z_mid=0.5,
        weak_threshold=20.0,
        use_fusion=False
    )
    
    detector = DivergenceDetector(config)
    
    # 测试不同的输入组合
    test_cases = [
        (1.0, 100.0, 1.0, 1.0, 0.1, "正常输入"),
        (0.0, 100.0, 1.0, 1.0, 0.1, "ts=0"),
        (1.0, 0.0, 1.0, 1.0, 0.1, "price=0"),
        (1.0, 100.0, float('nan'), 1.0, 0.1, "z_ofi=NaN"),
        (1.0, 100.0, 1.0, float('inf'), 0.1, "z_cvd=inf"),
        (1.0, 100.0, 1.0, 1.0, -0.1, "lag_sec<0"),
        (1.0, 100.0, 1.0, 1.0, 0.0, "lag_sec=0"),
    ]
    
    for ts, price, z_ofi, z_cvd, lag_sec, description in test_cases:
        print(f"测试: {description}")
        print(f"  输入: ts={ts}, price={price}, z_ofi={z_ofi}, z_cvd={z_cvd}, lag_sec={lag_sec}")
        
        # 手动验证
        is_valid = True
        for x in [ts, price, z_ofi, z_cvd, lag_sec]:
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                is_valid = False
                print(f"    类型检查失败: {x} (type: {type(x)}, isnan: {math.isnan(x) if isinstance(x, float) else False}, isinf: {math.isinf(x) if isinstance(x, float) else False})")
                break
        
        if is_valid and (ts <= 0 or price <= 0):
            is_valid = False
            print(f"    范围检查失败: ts={ts}, price={price}")
        
        print(f"  手动验证结果: {is_valid}")
        
        # 使用检测器验证
        result = detector.update(ts, price, z_ofi, z_cvd, lag_sec)
        if result is None:
            print(f"  检测器结果: None (无效输入)")
        else:
            print(f"  检测器结果: {result['type']} (原因: {result['reason_codes']})")
        
        print()


if __name__ == '__main__':
    debug_input_validation()
