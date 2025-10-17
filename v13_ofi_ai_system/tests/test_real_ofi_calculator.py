# -*- coding: utf-8 -*-
"""
Real OFI Calculator 测试文件 - Task 1.2.1
测试OFI计算器的核心功能

测试项:
1. 权重归一化和有效性
2. OFI方向正确性
3. Warmup期行为
4. Z-score计算
5. EMA平滑
6. 状态管理
"""
import sys
import io
from pathlib import Path

# Windows UTF-8 输出兼容
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from v13_ofi_ai_system.src.real_ofi_calculator import RealOFICalculator, OFIConfig

def test_weights_valid():
    """测试权重归一化和有效性"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=5))
    assert len(calc.w) == 5, f"权重数量应为5，实际为{len(calc.w)}"
    assert abs(sum(calc.w) - 1.0) < 1e-6, f"权重总和应为1.0，实际为{sum(calc.w)}"
    assert all(x >= 0 for x in calc.w), "所有权重应为非负数"
    print("✓ test_weights_valid 通过")

def test_ofi_direction():
    """测试OFI方向正确性（买入压力>卖出压力，OFI>0）"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=3))
    
    # 初始状态
    b1 = [(100.0, 5.0), (99.9, 3.0), (99.8, 2.0)]
    a1 = [(100.1, 4.0), (100.2, 3.5), (100.3, 2.5)]
    calc.update_with_snapshot(b1, a1)
    
    # 买单增加1.0，卖单减少0.4 → 买入压力增强 → OFI应>0
    b2 = [(100.0, 6.0), (99.9, 3.0), (99.8, 2.0)]
    a2 = [(100.1, 3.6), (100.2, 3.5), (100.3, 2.5)]
    r = calc.update_with_snapshot(b2, a2)
    
    assert r["ofi"] > 0.0, f"买入压力增强时OFI应>0，实际为{r['ofi']}"
    print(f"✓ test_ofi_direction 通过: OFI={r['ofi']:.4f}")

def test_warmup_behavior():
    """测试warmup期行为（warmup期z_ofi为None）"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=2, z_window=20))
    b = [(100.0, 1.0), (99.9, 1.0)]
    a = [(100.1, 1.0), (100.2, 1.0)]
    
    # 第一次更新应在warmup期
    r = calc.update_with_snapshot(b, a)
    assert r["meta"]["warmup"] is True, "首次更新应在warmup期"
    assert r["z_ofi"] is None, "warmup期z_ofi应为None"
    
    # 更新足够多次后应退出warmup期
    for _ in range(20):
        r = calc.update_with_snapshot(b, a)
    
    assert r["meta"]["warmup"] is False, "更新20次后应退出warmup期"
    assert r["z_ofi"] is not None, "退出warmup期后z_ofi应有值"
    print(f"✓ test_warmup_behavior 通过: warmup={r['meta']['warmup']}, z_ofi={r['z_ofi']}")

def test_z_score_calculation():
    """测试Z-score计算正确性"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=2, z_window=10))
    b = [(100.0, 1.0), (99.9, 1.0)]
    a = [(100.1, 1.0), (100.2, 1.0)]
    
    # warmup期
    for _ in range(5):
        calc.update_with_snapshot(b, a)
    
    # 创建一个明显偏离的OFI值
    b_large = [(100.0, 10.0), (99.9, 10.0)]  # 大量买入
    r = calc.update_with_snapshot(b_large, a)
    
    assert r["z_ofi"] is not None, "退出warmup期后应计算z_ofi"
    assert abs(r["z_ofi"]) > 1.0, f"明显偏离应有较大z_ofi，实际为{r['z_ofi']}"
    print(f"✓ test_z_score_calculation 通过: z_ofi={r['z_ofi']:.4f}")

def test_ema_smoothing():
    """测试EMA平滑功能"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=2, ema_alpha=0.5))
    b1 = [(100.0, 1.0), (99.9, 1.0)]
    a1 = [(100.1, 1.0), (100.2, 1.0)]
    
    r1 = calc.update_with_snapshot(b1, a1)
    ema1 = r1["ema_ofi"]
    ofi1 = r1["ofi"]
    
    # 第一次EMA应等于OFI
    assert abs(ema1 - ofi1) < 1e-9, f"首次EMA应等于OFI，EMA={ema1}, OFI={ofi1}"
    
    # 第二次更新
    b2 = [(100.0, 2.0), (99.9, 2.0)]
    r2 = calc.update_with_snapshot(b2, a1)
    ema2 = r2["ema_ofi"]
    ofi2 = r2["ofi"]
    
    # EMA应在两个OFI值之间
    expected_ema2 = 0.5 * ofi2 + 0.5 * ema1
    assert abs(ema2 - expected_ema2) < 1e-9, f"EMA计算错误，期望{expected_ema2}，实际{ema2}"
    print(f"✓ test_ema_smoothing 通过: EMA={ema2:.4f}")

def test_reset_and_state():
    """测试reset和get_state功能"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=3))
    b = [(100.0, 1.0), (99.9, 1.0), (99.8, 1.0)]
    a = [(100.1, 1.0), (100.2, 1.0), (100.3, 1.0)]
    
    # 更新几次
    for _ in range(5):
        calc.update_with_snapshot(b, a)
    
    # 获取状态
    state1 = calc.get_state()
    assert state1["ofi_hist_len"] > 0, "应有OFI历史记录"
    assert state1["ema_ofi"] is not None, "应有EMA值"
    
    # 重置
    calc.reset()
    state2 = calc.get_state()
    assert state2["ofi_hist_len"] == 0, "重置后OFI历史应为空"
    assert state2["ema_ofi"] is None, "重置后EMA应为None"
    assert state2["bad_points"] == 0, "重置后bad_points应为0"
    print(f"✓ test_reset_and_state 通过")

def test_k_components():
    """测试各档OFI分量计算"""
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=3))
    b1 = [(100.0, 5.0), (99.9, 3.0), (99.8, 2.0)]
    a1 = [(100.1, 4.0), (100.2, 3.5), (100.3, 2.5)]
    calc.update_with_snapshot(b1, a1)
    
    # 只改变第一档
    b2 = [(100.0, 6.0), (99.9, 3.0), (99.8, 2.0)]
    a2 = [(100.1, 4.0), (100.2, 3.5), (100.3, 2.5)]
    r = calc.update_with_snapshot(b2, a2)
    
    # 检查k_components
    assert len(r["k_components"]) == 3, "应有3档分量"
    # 第一档权重最大，贡献应最大
    assert abs(r["k_components"][0]) > abs(r["k_components"][1]), "第一档贡献应大于第二档"
    # OFI应等于所有分量之和
    assert abs(sum(r["k_components"]) - r["ofi"]) < 1e-9, "OFI应等于所有分量之和"
    print(f"✓ test_k_components 通过: components={r['k_components']}")

if __name__ == "__main__":
    print("=" * 60)
    print("运行 Real OFI Calculator 测试")
    print("=" * 60)
    
    try:
        test_weights_valid()
        test_ofi_direction()
        test_warmup_behavior()
        test_z_score_calculation()
        test_ema_smoothing()
        test_reset_and_state()
        test_k_components()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

