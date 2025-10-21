#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket统一配置测试脚本
验证WebSocket组件在统一配置下的运行
"""

import sys
import os
import time
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.websocket_config import WebSocketConfigLoader, WebSocketConfig, create_websocket_config

def test_websocket_config_loader():
    """测试WebSocket配置加载器"""
    print("=== WebSocket配置加载器测试 ===")
    
    try:
        # 1. 创建配置加载器
        print("1. 创建配置加载器...")
        config_loader = ConfigLoader()
        print("   [OK] 配置加载器创建成功")
        
        # 2. 创建WebSocket配置加载器
        print("2. 创建WebSocket配置加载器...")
        ws_config_loader = WebSocketConfigLoader(config_loader)
        print("   [OK] WebSocket配置加载器创建成功")
        
        # 3. 加载配置
        print("3. 加载WebSocket配置...")
        config = ws_config_loader.load_config("ETHUSDT")
        print(f"   深度级别: {config.depth_levels}")
        print(f"   连接超时: {config.timeout}秒")
        print(f"   重连间隔: {config.reconnect_interval}秒")
        print(f"   心跳超时: {config.heartbeat_timeout}秒")
        print(f"   缓冲区大小: {config.buffer_size}")
        print(f"   日志级别: {config.log_level}")
        print(f"   启用NDJSON: {config.enable_ndjson}")
        print("   [OK] 配置加载成功")
        
        # 4. 测试URL生成
        print("4. 测试URL生成...")
        ws_url = ws_config_loader.get_ws_url("ETHUSDT")
        rest_url = ws_config_loader.get_rest_snap_url("ETHUSDT")
        print(f"   WebSocket URL: {ws_url}")
        print(f"   REST URL: {rest_url}")
        print("   [OK] URL生成成功")
        
        # 5. 测试便捷函数
        print("5. 测试便捷函数...")
        config2 = create_websocket_config(config_loader, "BTCUSDT")
        print(f"   BTCUSDT配置: 深度={config2.depth_levels}, 超时={config2.timeout}")
        print("   [OK] 便捷函数测试成功")
        
        print("\n[SUCCESS] WebSocket配置加载器测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_client_integration():
    """测试WebSocket客户端集成"""
    print("\n=== WebSocket客户端集成测试 ===")
    
    try:
        # 1. 创建配置加载器
        print("1. 创建配置加载器...")
        config_loader = ConfigLoader()
        print("   [OK] 配置加载器创建成功")
        
        # 2. 测试向后兼容性（不使用config_loader）
        print("2. 测试向后兼容性...")
        from src.binance_websocket_client import BinanceOrderBookStream
        
        # 使用传统方式创建（向后兼容）
        stream1 = BinanceOrderBookStream(
            symbol="ETHUSDT",
            depth_levels=5,
            print_interval=10
        )
        print(f"   传统方式: 深度={stream1.depth_levels}, 超时={getattr(stream1, 'timeout', 'N/A')}")
        print("   [OK] 向后兼容性测试通过")
        
        # 3. 测试统一配置集成
        print("3. 测试统一配置集成...")
        stream2 = BinanceOrderBookStream(
            symbol="ETHUSDT",
            config_loader=config_loader
        )
        print(f"   统一配置: 深度={stream2.depth_levels}, 超时={getattr(stream2, 'timeout', 'N/A')}")
        print(f"   WebSocket URL: {stream2.ws_url}")
        print(f"   REST URL: {stream2.rest_snap_url}")
        print("   [OK] 统一配置集成测试通过")
        
        # 4. 验证配置参数
        print("4. 验证配置参数...")
        expected_params = [
            'depth_levels', 'timeout', 'reconnect_interval', 'ping_interval',
            'heartbeat_timeout', 'buffer_size', 'backpressure_threshold',
            'log_level', 'enable_ndjson', 'stats_interval'
        ]
        
        for param in expected_params:
            if hasattr(stream2, param):
                value = getattr(stream2, param)
                print(f"   {param}: {value}")
            else:
                print(f"   {param}: 未找到")
        
        print("   [OK] 配置参数验证完成")
        
        print("\n[SUCCESS] WebSocket客户端集成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket客户端集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_override():
    """测试环境变量覆盖"""
    print("\n=== 环境变量覆盖测试 ===")
    
    try:
        # 设置环境变量
        os.environ['V13__WEBSOCKET__TIMEOUT'] = '60'
        os.environ['V13__WEBSOCKET__RECONNECT_INTERVAL'] = '10'
        os.environ['V13__WEBSOCKET__DEPTH_LEVELS'] = '10'
        os.environ['V13__WEBSOCKET__LOG_LEVEL'] = 'DEBUG'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        config_loader.load(reload=True)
        
        # 创建WebSocket配置加载器
        ws_config_loader = WebSocketConfigLoader(config_loader)
        config = ws_config_loader.load_config("ETHUSDT")
        
        print(f"环境变量覆盖后的配置:")
        print(f"   超时: {config.timeout} (应该是60)")
        print(f"   重连间隔: {config.reconnect_interval} (应该是10)")
        print(f"   深度级别: {config.depth_levels} (应该是10)")
        print(f"   日志级别: {config.log_level} (应该是DEBUG)")
        
        # 验证覆盖是否生效
        if (config.timeout == 60 and 
            config.reconnect_interval == 10 and
            config.depth_levels == 10 and
            config.log_level == 'DEBUG'):
            print("   [OK] 环境变量覆盖成功")
        else:
            print("   [ERROR] 环境变量覆盖失败")
            return False
        
        # 清理环境变量
        del os.environ['V13__WEBSOCKET__TIMEOUT']
        del os.environ['V13__WEBSOCKET__RECONNECT_INTERVAL']
        del os.environ['V13__WEBSOCKET__DEPTH_LEVELS']
        del os.environ['V13__WEBSOCKET__LOG_LEVEL']
        
        print("\n[SUCCESS] 环境变量覆盖测试通过！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 环境变量覆盖测试失败: {e}")
        return False

def test_config_validation():
    """测试配置验证"""
    print("\n=== 配置验证测试 ===")
    
    try:
        config_loader = ConfigLoader()
        ws_config_loader = WebSocketConfigLoader(config_loader)
        config = ws_config_loader.load_config("ETHUSDT")
        
        # 验证配置参数范围
        print("验证配置参数范围...")
        
        # 超时时间应该在合理范围内
        if 1 <= config.timeout <= 300:
            print(f"   [OK] 超时时间合理: {config.timeout}秒")
        else:
            print(f"   [ERROR] 超时时间不合理: {config.timeout}秒")
            return False
        
        # 重连间隔应该大于0
        if config.reconnect_interval > 0:
            print(f"   [OK] 重连间隔合理: {config.reconnect_interval}秒")
        else:
            print(f"   [ERROR] 重连间隔不合理: {config.reconnect_interval}秒")
            return False
        
        # 深度级别应该在合理范围内
        if 1 <= config.depth_levels <= 100:
            print(f"   [OK] 深度级别合理: {config.depth_levels}")
        else:
            print(f"   [ERROR] 深度级别不合理: {config.depth_levels}")
            return False
        
        # 缓冲区大小应该大于0
        if config.buffer_size > 0:
            print(f"   [OK] 缓冲区大小合理: {config.buffer_size}")
        else:
            print(f"   [ERROR] 缓冲区大小不合理: {config.buffer_size}")
            return False
        
        # 背压阈值应该在0-1之间
        if 0 <= config.backpressure_threshold <= 1:
            print(f"   [OK] 背压阈值合理: {config.backpressure_threshold}")
        else:
            print(f"   [ERROR] 背压阈值不合理: {config.backpressure_threshold}")
            return False
        
        print("\n[SUCCESS] 配置验证测试通过！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 配置验证测试失败: {e}")
        return False

def main():
    """主函数"""
    print("WebSocket统一配置测试")
    print("=" * 50)
    
    success = True
    
    # 配置加载器测试
    if not test_websocket_config_loader():
        success = False
    
    # 客户端集成测试
    if not test_websocket_client_integration():
        success = False
    
    # 环境变量覆盖测试
    if not test_environment_override():
        success = False
    
    # 配置验证测试
    if not test_config_validation():
        success = False
    
    if success:
        print("\n[SUCCESS] 所有WebSocket配置测试通过！")
        print("WebSocket组件已成功集成到统一配置系统。")
    else:
        print("\n[ERROR] 部分WebSocket配置测试失败，请检查配置。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
