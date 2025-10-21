"""
最小化调试
"""

import sys
import os
import io

# 修复Windows编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ofi_cvd_divergence import DivergenceDetector, DivergenceConfig


def debug_minimal():
    """最小化调试"""
    print("=== 最小化调试 ===\n")
    
    # 创建最简单的配置
    config = DivergenceConfig(
        swing_L=1,  # 最小窗口
        min_separation=1,
        cooldown_secs=0.0,
        warmup_min=1,  # 最小预热
        z_hi=0.5,
        z_mid=0.2,
        weak_threshold=10.0,
        use_fusion=False
    )
    
    detector = DivergenceDetector(config)
    
    print(f"配置: swing_L={config.swing_L}, warmup_min={config.warmup_min}")
    print(f"需要最少数据点: {config.swing_L * 2 + 1}")
    print()
    
    # 测试单个输入
    test_inputs = [
        (1.0, 100.0, 1.0, 1.0, 0.0),
        (2.0, 101.0, 1.5, 1.5, 0.0),
        (3.0, 102.0, 2.0, 2.0, 0.0),
        (4.0, 103.0, 2.5, 2.5, 0.0),
        (5.0, 104.0, 3.0, 3.0, 0.0),
    ]
    
    for i, (ts, price, z_ofi, z_cvd, lag_sec) in enumerate(test_inputs):
        print(f"输入 {i+1}: ts={ts}, price={price}, z_ofi={z_ofi}, z_cvd={z_cvd}, lag_sec={lag_sec}")
        
        result = detector.update(ts, price, z_ofi, z_cvd, lag_sec=lag_sec)
        
        if result is None:
            print(f"  结果: None")
        else:
            print(f"  结果: {result['type']} (原因: {result['reason_codes']})")
            if result['type'] is not None:
                print(f"  分数: {result['score']}")
                print(f"  枢轴: {result['pivots']}")
        
        print(f"  样本计数: {detector._sample_count}")
        print(f"  统计: {detector._stats}")
        print()


if __name__ == '__main__':
    debug_minimal()
