#!/usr/bin/env python3
"""
背离检测配置集成测试脚本

测试背离检测组件与统一配置系统的集成功能
包括配置加载、参数验证、环境变量覆盖等
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.divergence_config_loader import DivergenceConfigLoader, DivergenceConfig
from src.ofi_cvd_divergence import DivergenceDetector, DivergenceConfig as OriginalDivergenceConfig

def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试背离检测配置加载功能 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 创建背离检测配置加载器
        divergence_loader = DivergenceConfigLoader(config_loader)
        
        # 加载配置
        config = divergence_loader.load_config()
        
        # 验证基础配置
        assert hasattr(config, 'swing_L'), "缺少swing_L属性"
        assert hasattr(config, 'ema_k'), "缺少ema_k属性"
        assert hasattr(config, 'z_hi'), "缺少z_hi属性"
        assert hasattr(config, 'z_mid'), "缺少z_mid属性"
        assert hasattr(config, 'min_separation'), "缺少min_separation属性"
        assert hasattr(config, 'cooldown_secs'), "缺少cooldown_secs属性"
        assert hasattr(config, 'warmup_min'), "缺少warmup_min属性"
        assert hasattr(config, 'max_lag'), "缺少max_lag属性"
        assert hasattr(config, 'use_fusion'), "缺少use_fusion属性"
        
        print("配置加载成功")
        print(f"  - Swing L: {config.swing_L}")
        print(f"  - EMA K: {config.ema_k}")
        print(f"  - Z Hi: {config.z_hi}")
        print(f"  - Z Mid: {config.z_mid}")
        print(f"  - Min Separation: {config.min_separation}")
        print(f"  - Cooldown Secs: {config.cooldown_secs}")
        print(f"  - Warmup Min: {config.warmup_min}")
        print(f"  - Max Lag: {config.max_lag}")
        print(f"  - Use Fusion: {config.use_fusion}")
        
        return True
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False

def test_divergence_detector_creation():
    """测试背离检测器创建"""
    print("\n=== 测试背离检测器创建 ===")
    
    try:
        # 创建配置加载器
        config_loader = ConfigLoader()
        
        # 使用配置加载器创建背离检测器
        detector = DivergenceDetector(config_loader=config_loader)
        
        # 验证配置
        assert detector.cfg is not None, "配置对象为空"
        assert detector.cfg.swing_L == 12, f"swing_L配置错误: {detector.cfg.swing_L}"
        assert detector.cfg.ema_k == 5, f"ema_k配置错误: {detector.cfg.ema_k}"
        assert detector.cfg.z_hi == 1.5, f"z_hi配置错误: {detector.cfg.z_hi}"
        assert detector.cfg.z_mid == 0.7, f"z_mid配置错误: {detector.cfg.z_mid}"
        assert detector.cfg.use_fusion == True, f"use_fusion配置错误: {detector.cfg.use_fusion}"
        
        print("背离检测器创建成功")
        print(f"  - 枢轴检测器数量: 3 (price_ofi, price_cvd, price_fusion)")
        print(f"  - 配置来源: 统一配置系统")
        print(f"  - Swing L: {detector.cfg.swing_L}")
        print(f"  - 使用融合指标: {detector.cfg.use_fusion}")
        
        return True
        
    except Exception as e:
        print(f"背离检测器创建失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    try:
        # 测试使用原始配置对象创建
        original_config = OriginalDivergenceConfig(
            swing_L=8,
            ema_k=3,
            z_hi=2.0,
            z_mid=1.0
        )
        
        detector1 = DivergenceDetector(config=original_config)
        assert detector1.cfg.swing_L == 8, "原始配置未正确传递"
        assert detector1.cfg.ema_k == 3, "原始配置未正确传递"
        assert detector1.cfg.z_hi == 2.0, "原始配置未正确传递"
        assert detector1.cfg.z_mid == 1.0, "原始配置未正确传递"
        
        # 测试使用默认配置创建
        detector2 = DivergenceDetector()
        assert detector2.cfg.swing_L == 12, "默认配置未正确应用"
        assert detector2.cfg.ema_k == 5, "默认配置未正确应用"
        
        print("向后兼容性测试成功")
        print("  - 原始配置对象: 支持")
        print("  - 默认配置: 支持")
        print("  - 统一配置系统: 支持")
        
        return True
        
    except Exception as e:
        print(f"向后兼容性测试失败: {e}")
        return False

def test_environment_override():
    """测试环境变量覆盖功能"""
    print("\n=== 测试环境变量覆盖功能 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__DIVERGENCE_DETECTION__PIVOT_DETECTION__SWING_L'] = '15'
        os.environ['V13__DIVERGENCE_DETECTION__THRESHOLDS__Z_HI'] = '2.0'
        os.environ['V13__DIVERGENCE_DETECTION__DENOISING__COOLDOWN_SECS'] = '2.0'
        os.environ['V13__DIVERGENCE_DETECTION__FUSION__USE_FUSION'] = 'False'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        detector = DivergenceDetector(config_loader=config_loader)
        
        # 验证环境变量覆盖
        if detector.cfg.swing_L == 15:
            print("Swing L环境变量覆盖成功")
        else:
            print(f"Swing L环境变量覆盖失败，期望: 15，实际: {detector.cfg.swing_L}")
            return False
        
        if detector.cfg.z_hi == 2.0:
            print("Z Hi环境变量覆盖成功")
        else:
            print(f"Z Hi环境变量覆盖失败，期望: 2.0，实际: {detector.cfg.z_hi}")
            return False
        
        if detector.cfg.cooldown_secs == 2.0:
            print("Cooldown Secs环境变量覆盖成功")
        else:
            print(f"Cooldown Secs环境变量覆盖失败，期望: 2.0，实际: {detector.cfg.cooldown_secs}")
            return False
        
        if detector.cfg.use_fusion == False:
            print("Use Fusion环境变量覆盖成功")
        else:
            print(f"Use Fusion环境变量覆盖失败，期望: False，实际: {detector.cfg.use_fusion}")
            return False
        
        # 清理环境变量
        del os.environ['V13__DIVERGENCE_DETECTION__PIVOT_DETECTION__SWING_L']
        del os.environ['V13__DIVERGENCE_DETECTION__THRESHOLDS__Z_HI']
        del os.environ['V13__DIVERGENCE_DETECTION__DENOISING__COOLDOWN_SECS']
        del os.environ['V13__DIVERGENCE_DETECTION__FUSION__USE_FUSION']
        
        return True
        
    except Exception as e:
        print(f"环境变量覆盖测试失败: {e}")
        return False

def test_config_methods():
    """测试配置方法"""
    print("\n=== 测试配置方法 ===")
    
    try:
        config_loader = ConfigLoader()
        divergence_loader = DivergenceConfigLoader(config_loader)
        
        # 测试枢轴检测配置
        pivot_config = divergence_loader.get_pivot_config()
        assert 'swing_L' in pivot_config, "枢轴检测配置缺少swing_L"
        assert 'ema_k' in pivot_config, "枢轴检测配置缺少ema_k"
        print("枢轴检测配置方法正常")
        
        # 测试阈值配置
        thresholds = divergence_loader.get_thresholds()
        assert 'z_hi' in thresholds, "阈值配置缺少z_hi"
        assert 'z_mid' in thresholds, "阈值配置缺少z_mid"
        print("阈值配置方法正常")
        
        # 测试去噪配置
        denoising = divergence_loader.get_denoising_config()
        assert 'min_separation' in denoising, "去噪配置缺少min_separation"
        assert 'cooldown_secs' in denoising, "去噪配置缺少cooldown_secs"
        print("去噪配置方法正常")
        
        # 测试融合配置
        fusion = divergence_loader.get_fusion_config()
        assert 'use_fusion' in fusion, "融合配置缺少use_fusion"
        print("融合配置方法正常")
        
        # 测试性能配置
        performance = divergence_loader.get_performance_config()
        assert 'max_events_per_second' in performance, "性能配置缺少max_events_per_second"
        print("性能配置方法正常")
        
        # 测试监控配置
        monitoring = divergence_loader.get_monitoring_config()
        assert 'prometheus_port' in monitoring, "监控配置缺少prometheus_port"
        print("监控配置方法正常")
        
        # 测试热更新配置
        hot_reload = divergence_loader.get_hot_reload_config()
        assert 'enabled' in hot_reload, "热更新配置缺少enabled"
        print("热更新配置方法正常")
        
        return True
        
    except Exception as e:
        print(f"配置方法测试失败: {e}")
        return False

def test_divergence_detection_functionality():
    """测试背离检测功能"""
    print("\n=== 测试背离检测功能 ===")
    
    try:
        config_loader = ConfigLoader()
        detector = DivergenceDetector(config_loader=config_loader)
        
        # 模拟一些数据
        import time
        current_time = time.time()
        
        # 模拟背离检测数据
        for i in range(50):
            ts = current_time + i
            price = 50000 + i * 10  # 上升趋势
            z_ofi = 2.0 - i * 0.04  # 下降趋势，形成背离
            z_cvd = 1.5 - i * 0.03  # 下降趋势，形成背离
            fusion_score = 0.8 - i * 0.01 if detector.cfg.use_fusion else None
            consistency = 0.7 - i * 0.01 if detector.cfg.use_fusion else None
            
            # 使用update方法
            result = detector.update(
                ts=ts,
                price=price,
                z_ofi=z_ofi,
                z_cvd=z_cvd,
                fusion_score=fusion_score,
                consistency=consistency,
                warmup=i < 20,  # 前20个样本为暖启动
                lag_sec=0.1
            )
        
        # 检查状态
        assert detector._sample_count >= 50, f"样本数量不足: {detector._sample_count}"
        print(f"样本数量: {detector._sample_count}")
        
        # 检查枢轴检测器状态
        assert hasattr(detector, 'price_ofi_detector'), "价格OFI枢轴检测器属性不存在"
        assert hasattr(detector, 'price_cvd_detector'), "价格CVD枢轴检测器属性不存在"
        assert detector.price_ofi_detector is not None, "价格OFI枢轴检测器未初始化"
        assert detector.price_cvd_detector is not None, "价格CVD枢轴检测器未初始化"
        if detector.cfg.use_fusion:
            assert detector.price_fusion_detector is not None, "价格融合枢轴检测器未初始化"
        
        print("背离检测功能正常")
        print(f"  - 样本数量: {detector._sample_count}")
        print(f"  - 枢轴检测器: 已初始化")
        print(f"  - 融合检测器: {'已启用' if detector.cfg.use_fusion else '已禁用'}")
        
        return True
        
    except Exception as e:
        print(f"背离检测功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("背离检测配置集成测试开始")
    print("=" * 60)
    
    tests = [
        ("配置加载功能", test_config_loading),
        ("背离检测器创建", test_divergence_detector_creation),
        ("向后兼容性", test_backward_compatibility),
        ("环境变量覆盖", test_environment_override),
        ("配置方法", test_config_methods),
        ("背离检测功能", test_divergence_detection_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} 测试失败")
        except Exception as e:
            print(f"[FAIL] {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("[SUCCESS] 所有测试通过！背离检测配置集成功能正常")
        return True
    else:
        print("[ERROR] 部分测试失败，请检查配置和代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
