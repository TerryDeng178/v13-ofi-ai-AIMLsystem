#!/usr/bin/env python3
"""
融合指标配置使用示例

演示如何使用统一配置系统管理融合指标配置：
- 从配置加载器创建融合指标实例
- 配置热更新和动态调整
- 环境特定配置覆盖
- 配置验证和监控

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-20
"""

import os
import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion
from src.fusion_config_hot_update import create_fusion_hot_updater


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 创建配置加载器
    config_loader = ConfigLoader()
    
    # 2. 从配置创建融合指标实例
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    # 3. 显示当前配置
    print(f"当前配置:")
    print(f"  权重: w_ofi={fusion.cfg.w_ofi}, w_cvd={fusion.cfg.w_cvd}")
    print(f"  阈值: buy={fusion.cfg.fuse_buy}, strong_buy={fusion.cfg.fuse_strong_buy}")
    print(f"  一致性: min={fusion.cfg.min_consistency}, strong={fusion.cfg.strong_min_consistency}")
    
    # 4. 测试融合计算
    print(f"\n测试融合计算:")
    result = fusion.update(ts=time.time(), z_ofi=2.0, z_cvd=1.5, lag_sec=0.1)
    if result:
        print(f"  融合得分: {result['fusion_score']:.3f}")
        print(f"  信号: {result['signal']}")
        print(f"  一致性: {result['consistency']:.3f}")
    
    return fusion


def example_environment_override():
    """环境配置覆盖示例"""
    print("\n=== 环境配置覆盖示例 ===")
    
    # 设置环境变量来覆盖配置
    os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY'] = '3.0'
    os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL'] = '-3.0'
    
    # 重新加载配置
    config_loader = ConfigLoader()
    config_loader.load(reload=True)
    
    # 创建融合指标实例
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    print(f"环境变量覆盖后的配置:")
    print(f"  强买入阈值: {fusion.cfg.fuse_strong_buy}")
    print(f"  强卖出阈值: {fusion.cfg.fuse_strong_sell}")
    
    # 清理环境变量
    del os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY']
    del os.environ['V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL']


def example_hot_update():
    """配置热更新示例"""
    print("\n=== 配置热更新示例 ===")
    
    # 创建配置加载器和融合指标实例
    config_loader = ConfigLoader()
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    print(f"初始配置:")
    print(f"  强买入阈值: {fusion.cfg.fuse_strong_buy}")
    print(f"  强卖出阈值: {fusion.cfg.fuse_strong_sell}")
    
    # 创建热更新器
    hot_updater = create_fusion_hot_updater(
        config_loader=config_loader,
        fusion_instance=fusion
    )
    
    # 添加更新回调
    def on_config_update(new_config):
        print(f"配置已更新:")
        print(f"  强买入阈值: {new_config.fuse_strong_buy}")
        print(f"  强卖出阈值: {new_config.fuse_strong_sell}")
    
    hot_updater.add_update_callback(on_config_update)
    
    # 模拟配置更新
    print(f"\n模拟配置更新...")
    hot_updater.update_config(force=True)
    
    # 显示更新统计
    stats = hot_updater.get_update_stats()
    print(f"\n更新统计:")
    print(f"  总更新次数: {stats['total_updates']}")
    print(f"  成功次数: {stats['successful_updates']}")
    print(f"  失败次数: {stats['failed_updates']}")


def example_config_validation():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")
    
    config_loader = ConfigLoader()
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    # 验证当前配置
    print("验证当前配置...")
    
    # 检查权重归一化
    total_weight = fusion.cfg.w_ofi + fusion.cfg.w_cvd
    print(f"权重和: {total_weight:.6f} (应该接近1.0)")
    
    # 检查阈值逻辑
    print(f"阈值检查:")
    print(f"  强买入 > 买入: {fusion.cfg.fuse_strong_buy > fusion.cfg.fuse_buy}")
    print(f"  强卖出 < 卖出: {fusion.cfg.fuse_strong_sell < fusion.cfg.fuse_sell}")
    
    # 检查一致性阈值
    print(f"一致性阈值检查:")
    print(f"  最小一致性: {fusion.cfg.min_consistency} (0-1)")
    print(f"  强信号一致性: {fusion.cfg.strong_min_consistency} (0-1)")
    print(f"  强信号 > 最小: {fusion.cfg.strong_min_consistency > fusion.cfg.min_consistency}")


def example_performance_test():
    """性能测试示例"""
    print("\n=== 性能测试示例 ===")
    
    config_loader = ConfigLoader()
    fusion = OFI_CVD_Fusion(config_loader=config_loader)
    
    # 性能测试
    import time
    import random
    
    num_tests = 10000
    start_time = time.time()
    
    for i in range(num_tests):
        ts = time.time() + i * 0.001
        z_ofi = random.uniform(-5, 5)
        z_cvd = random.uniform(-5, 5)
        lag_sec = random.uniform(0, 0.5)
        
        fusion.update(ts=ts, z_ofi=z_ofi, z_cvd=z_cvd, lag_sec=lag_sec)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_tests * 1000  # 转换为毫秒
    
    print(f"性能测试结果:")
    print(f"  测试次数: {num_tests}")
    print(f"  总时间: {total_time:.3f}秒")
    print(f"  平均时间: {avg_time:.3f}ms")
    print(f"  每秒处理: {num_tests / total_time:.0f}次")
    
    # 显示统计信息
    stats = fusion.get_stats()
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """主函数"""
    setup_logging()
    
    print("融合指标配置使用示例")
    print("=" * 50)
    
    try:
        # 基本使用
        fusion = example_basic_usage()
        
        # 环境配置覆盖
        example_environment_override()
        
        # 配置热更新
        example_hot_update()
        
        # 配置验证
        example_config_validation()
        
        # 性能测试
        example_performance_test()
        
        print("\n✅ 所有示例执行完成")
        
    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
