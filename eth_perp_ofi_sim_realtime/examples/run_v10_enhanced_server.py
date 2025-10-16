#!/usr/bin/env python3
"""
V10.0 增强服务器启动脚本
启动集成深度学习模型和3级加权OFI的实时市场模拟器
"""

import asyncio
import yaml
import argparse
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stream_v10_enhanced import V10EnhancedWSHub

def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params.yaml')
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V10.0 增强实时市场模拟器服务器')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8765, help='服务器端口')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if config is None:
        print("无法加载配置文件，使用默认配置")
        config = {
            "sim": {
                "seed": 42,
                "seconds": 300,
                "init_mid": 2500.0,
                "tick_size": 0.1,
                "base_spread_ticks": 2,
                "base_depth": 30.0,
                "depth_jitter": 0.6,
                "levels": 5
            },
            "ofi": {
                "micro_window_ms": 100,
                "z_window_seconds": 900,
                "ofi_long_z": 2.0,
                "ofi_short_z": -2.0
            },
            "signals": {
                "thin_spread_bps_max": 16.0,
                "use_reclaim": True,
                "reclaim_lookback_ms": 1500,
                "reclaim_confirm_ms": 100
            }
        }
    
    if args.verbose:
        print("V10.0 增强实时市场模拟器服务器配置:")
        print(f"  主机: {args.host}")
        print(f"  端口: {args.port}")
        print(f"  配置文件: {args.config or '默认'}")
        print(f"  模拟时长: {config['sim']['seconds']}秒")
        print(f"  初始价格: {config['sim']['init_mid']}")
        print(f"  OFI窗口: {config['ofi']['micro_window_ms']}ms")
        print(f"  OFI Z窗口: {config['ofi']['z_window_seconds']}秒")
    
    # 创建V10增强WebSocket Hub
    hub = V10EnhancedWSHub(
        host=args.host,
        port=args.port,
        config=config
    )
    
    print("="*60)
    print("V10.0 增强实时市场模拟器服务器")
    print("="*60)
    print(f"服务器地址: ws://{args.host}:{args.port}")
    print("支持功能:")
    print("  ✅ 3级加权OFI计算")
    print("  ✅ 深度学习信号生成")
    print("  ✅ 实时优化算法")
    print("  ✅ 自适应阈值调整")
    print("  ✅ 性能监控")
    print("="*60)
    
    try:
        # 启动服务器
        asyncio.run(hub.serve())
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
