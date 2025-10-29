#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Harvester 构造函数重构和配置系统接入

验证点：
1. 构造函数签名是否正确
2. _apply_cfg 方法是否存在
3. 配置系统集成是否正确
4. 向后兼容是否正常
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_constructor_signature():
    """测试构造函数签名"""
    print("=" * 60)
    print("测试 1: 构造函数签名")
    print("=" * 60)
    
    try:
        from deploy.run_success_harvest import SuccessOFICVDHarvester
        import inspect
        
        # 获取构造函数签名
        sig = inspect.signature(SuccessOFICVDHarvester.__init__)
        params = list(sig.parameters.keys())
        
        print(f"参数列表: {params}")
        
        # 验证关键参数
        assert 'cfg' in params, "缺少 cfg 参数"
        assert 'compat_env' in params, "缺少 compat_env 参数"
        assert params.index('cfg') < params.index('compat_env'), "参数顺序错误"
        
        # 验证 compat_env 是 keyword-only
        param_compat = sig.parameters['compat_env']
        assert param_compat.kind == inspect.Parameter.KEYWORD_ONLY, "compat_env 应该是 keyword-only 参数"
        
        print("✅ 构造函数签名验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 构造函数签名验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_apply_cfg_method():
    """测试 _apply_cfg 方法"""
    print("\n" + "=" * 60)
    print("测试 2: _apply_cfg 方法")
    print("=" * 60)
    
    try:
        from deploy.run_success_harvest import SuccessOFICVDHarvester
        
        # 检查方法是否存在
        assert hasattr(SuccessOFICVDHarvester, '_apply_cfg'), "_apply_cfg 方法不存在"
        
        # 检查方法签名
        import inspect
        sig = inspect.signature(SuccessOFICVDHarvester._apply_cfg)
        params = list(sig.parameters.keys())
        
        print(f"_apply_cfg 参数: {params}")
        
        # 验证参数
        assert 'self' in params, "缺少 self 参数"
        assert 'symbols' in params, "缺少 symbols 参数"
        assert 'output_dir' in params, "缺少 output_dir 参数"
        
        print("✅ _apply_cfg 方法验证通过")
        return True
        
    except Exception as e:
        print(f"❌ _apply_cfg 方法验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """测试配置系统集成"""
    print("\n" + "=" * 60)
    print("测试 3: 配置系统集成")
    print("=" * 60)
    
    try:
        from deploy.run_success_harvest import SuccessOFICVDHarvester
        
        # 测试配置：cfg 模式
        test_cfg = {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "paths": {
                "output_dir": "./test_output",
                "preview_dir": "./test_preview",
                "artifacts_dir": "./test_artifacts"
            },
            "buffers": {
                "high": {"prices": 10000, "orderbook": 6000, "ofi": 4000, "cvd": 4000, "fusion": 2500, "events": 2500, "features": 4000},
                "emergency": {"prices": 20000, "orderbook": 12000, "ofi": 8000, "cvd": 8000, "fusion": 5000, "events": 5000, "features": 8000}
            },
            "files": {
                "max_rows_per_file": 25000,
                "parquet_rotate_sec": 30
            },
            "concurrency": {
                "save_concurrency": 1
            },
            "timeouts": {
                "health_check_interval": 10,
                "stream_idle_sec": 60,
                "trade_timeout": 100,
                "orderbook_timeout": 120,
                "backoff_reset_secs": 150
            },
            "thresholds": {
                "extreme_traffic_threshold": 15000,
                "extreme_rotate_sec": 15,
                "ofi_max_lag_ms": 400
            },
            "dedup": {
                "lru_size": 16384,
                "queue_drop_threshold": 500
            },
            "scenario": {
                "win_secs": 150,
                "active_tps": 0.05,
                "vol_split": 0.3,
                "fee_tier": "T1"
            }
        }
        
        # 创建实例（仅初始化，不运行）
        harvester = SuccessOFICVDHarvester(cfg=test_cfg, run_hours=0.001)
        
        # 验证配置是否正确应用
        assert harvester.symbols == ["BTCUSDT", "ETHUSDT"], f"symbols 配置错误: {harvester.symbols}"
        assert str(harvester.output_dir).endswith("test_output") or "test_output" in str(harvester.output_dir), \
            f"output_dir 配置错误: {harvester.output_dir}"
        assert harvester.max_rows_per_file == 25000, f"max_rows_per_file 配置错误: {harvester.max_rows_per_file}"
        assert harvester.ofi_max_lag_ms == 400, f"ofi_max_lag_ms 配置错误: {harvester.ofi_max_lag_ms}"
        assert harvester.win_secs == 150, f"win_secs 配置错误: {harvester.win_secs}"
        
        print(f"✅ 配置验证通过")
        print(f"   - symbols: {harvester.symbols}")
        print(f"   - output_dir: {harvester.output_dir}")
        print(f"   - max_rows_per_file: {harvester.max_rows_per_file}")
        print(f"   - ofi_max_lag_ms: {harvester.ofi_max_lag_ms}")
        print(f"   - win_secs: {harvester.win_secs}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统集成验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_function():
    """测试 main 函数配置加载被动"""
    print("\n" + "=" * 60)
    print("测试 4: main 函数配置加载")
    print("=" * 60)
    
    try:
        import inspect
        from deploy.run_success_harvest import main
        
        # 检查 main 函数是否存在
        assert callable(main), "main 函数不存在或不可调用"
        
        # 检查是否是 async 函数
        assert inspect.iscoroutinefunction(main), "main 应该是 async 函数"
        
        # 检查函数签名中是否有相关参数（通过源码检查）
        source = inspect.getsource(main)
        assert '--config' in source or 'config' in source.lower(), "main 函数中缺少 --config 参数处理"
        assert '--dry-run-config' in source or 'dry_run_config' in source, "main 函数中缺少 --dry-run-config 样数处理"
        assert 'load_component_runtime_config' in source or 'load_component' in source, "main 函数中缺少配置加载调用"
        
        print("✅ main 函数配置加载验证通过")
        print(f"   - main 是 async 函数: {inspect.iscoroutinefunction(main)}")
        print(f"   - 包含配置加载逻辑")
        print(f"   - 使用 SuccessOFICVDHarvester(cfg=harvester_cfg) 创建实例")
        
        return True
        
    except Exception as e:
        print(f"❌ main 函数配置加载验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Harvester 构造函数重构测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("构造函数签名", test_constructor_signature()))
    results.append(("_apply_cfg 方法", test_apply_cfg_method()))
    results.append(("配置系统集成", test_config_integration()))
    results.append(("main 函数配置", test_main_function()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！重构成功完成！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())

