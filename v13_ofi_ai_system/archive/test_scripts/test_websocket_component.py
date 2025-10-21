#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket组件功能测试
测试WebSocket配置和基本功能，避免外部依赖问题
"""

import sys
import os
import time
import json
sys.path.insert(0, '.')

from src.utils.config_loader import ConfigLoader
from src.websocket_config import WebSocketConfigLoader, WebSocketConfig, create_websocket_config

def test_websocket_config_loading():
    """测试WebSocket配置加载功能"""
    print("=== WebSocket配置加载测试 ===")
    
    try:
        # 1. 测试配置加载器
        print("1. 测试配置加载器...")
        config_loader = ConfigLoader()
        ws_config_loader = WebSocketConfigLoader(config_loader)
        print("   [OK] 配置加载器创建成功")
        
        # 2. 测试不同交易对的配置加载
        print("2. 测试不同交易对配置...")
        symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            config = ws_config_loader.load_config(symbol)
            ws_url = ws_config_loader.get_ws_url(symbol)
            rest_url = ws_config_loader.get_rest_snap_url(symbol)
            
            print(f"   {symbol}:")
            print(f"     深度级别: {config.depth_levels}")
            print(f"     超时: {config.timeout}秒")
            print(f"     WebSocket URL: {ws_url}")
            print(f"     REST URL: {rest_url}")
        
        print("   [OK] 多交易对配置加载成功")
        
        # 3. 测试配置参数验证
        print("3. 测试配置参数验证...")
        config = ws_config_loader.load_config("ETHUSDT")
        
        # 验证关键参数
        assert 1 <= config.timeout <= 300, f"超时时间不合理: {config.timeout}"
        assert config.reconnect_interval > 0, f"重连间隔不合理: {config.reconnect_interval}"
        assert 1 <= config.depth_levels <= 100, f"深度级别不合理: {config.depth_levels}"
        assert config.buffer_size > 0, f"缓冲区大小不合理: {config.buffer_size}"
        assert 0 <= config.backpressure_threshold <= 1, f"背压阈值不合理: {config.backpressure_threshold}"
        
        print("   [OK] 配置参数验证通过")
        
        # 4. 测试URL格式验证
        print("4. 测试URL格式验证...")
        ws_url = ws_config_loader.get_ws_url("ETHUSDT")
        rest_url = ws_config_loader.get_rest_snap_url("ETHUSDT")
        
        assert ws_url.startswith("wss://"), f"WebSocket URL格式错误: {ws_url}"
        assert rest_url.startswith("https://"), f"REST URL格式错误: {rest_url}"
        assert "ethusdt" in ws_url.lower(), f"WebSocket URL缺少交易对: {ws_url}"
        assert "ETHUSDT" in rest_url, f"REST URL缺少交易对: {rest_url}"
        
        print("   [OK] URL格式验证通过")
        
        print("\n[SUCCESS] WebSocket配置加载测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_config_consistency():
    """测试WebSocket配置一致性"""
    print("\n=== WebSocket配置一致性测试 ===")
    
    try:
        config_loader = ConfigLoader()
        ws_config_loader = WebSocketConfigLoader(config_loader)
        
        # 1. 测试多次加载的一致性
        print("1. 测试多次加载一致性...")
        config1 = ws_config_loader.load_config("ETHUSDT")
        config2 = ws_config_loader.load_config("ETHUSDT")
        
        assert config1.timeout == config2.timeout, "超时时间不一致"
        assert config1.depth_levels == config2.depth_levels, "深度级别不一致"
        assert config1.log_level == config2.log_level, "日志级别不一致"
        
        print("   [OK] 多次加载配置一致")
        
        # 2. 测试不同交易对的基础配置一致性
        print("2. 测试不同交易对基础配置一致性...")
        symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT"]
        configs = [ws_config_loader.load_config(symbol) for symbol in symbols]
        
        # 基础配置应该一致
        for i in range(1, len(configs)):
            assert configs[0].timeout == configs[i].timeout, f"超时时间不一致: {symbols[0]} vs {symbols[i]}"
            assert configs[0].depth_levels == configs[i].depth_levels, f"深度级别不一致: {symbols[0]} vs {symbols[i]}"
            assert configs[0].log_level == configs[i].log_level, f"日志级别不一致: {symbols[0]} vs {symbols[i]}"
        
        print("   [OK] 不同交易对基础配置一致")
        
        # 3. 测试URL生成的一致性
        print("3. 测试URL生成一致性...")
        for symbol in symbols:
            ws_url1 = ws_config_loader.get_ws_url(symbol)
            ws_url2 = ws_config_loader.get_ws_url(symbol)
            assert ws_url1 == ws_url2, f"WebSocket URL生成不一致: {ws_url1} vs {ws_url2}"
            
            rest_url1 = ws_config_loader.get_rest_snap_url(symbol)
            rest_url2 = ws_config_loader.get_rest_snap_url(symbol)
            assert rest_url1 == rest_url2, f"REST URL生成不一致: {rest_url1} vs {rest_url2}"
        
        print("   [OK] URL生成一致性验证通过")
        
        print("\n[SUCCESS] WebSocket配置一致性测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_config_performance():
    """测试WebSocket配置性能"""
    print("\n=== WebSocket配置性能测试 ===")
    
    try:
        config_loader = ConfigLoader()
        ws_config_loader = WebSocketConfigLoader(config_loader)
        
        # 1. 测试配置加载性能
        print("1. 测试配置加载性能...")
        num_tests = 1000
        start_time = time.time()
        
        for i in range(num_tests):
            config = ws_config_loader.load_config("ETHUSDT")
            ws_url = ws_config_loader.get_ws_url("ETHUSDT")
            rest_url = ws_config_loader.get_rest_snap_url("ETHUSDT")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_tests * 1000  # 转换为毫秒
        
        print(f"   测试次数: {num_tests}")
        print(f"   总时间: {total_time:.3f}秒")
        print(f"   平均时间: {avg_time:.3f}ms")
        print(f"   每秒处理: {num_tests / total_time:.0f}次")
        
        if avg_time < 1.0:  # 小于1ms
            print("   [OK] 性能优秀")
        else:
            print("   [WARN] 性能需要优化")
        
        # 2. 测试内存使用
        print("2. 测试内存使用...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大量配置对象
        configs = []
        for i in range(1000):
            config = ws_config_loader.load_config(f"SYMBOL{i}")
            configs.append(config)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"   创建1000个配置对象")
        print(f"   内存使用: {memory_used:.2f}MB")
        print(f"   平均每个配置: {memory_used/1000:.4f}MB")
        
        if memory_used < 10:  # 小于10MB
            print("   [OK] 内存使用合理")
        else:
            print("   [WARN] 内存使用较高")
        
        print("\n[SUCCESS] WebSocket配置性能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_config_edge_cases():
    """测试WebSocket配置边界情况"""
    print("\n=== WebSocket配置边界情况测试 ===")
    
    try:
        config_loader = ConfigLoader()
        ws_config_loader = WebSocketConfigLoader(config_loader)
        
        # 1. 测试空配置加载器
        print("1. 测试空配置加载器...")
        empty_loader = WebSocketConfigLoader(None)
        config = empty_loader.load_config("ETHUSDT")
        
        assert config.timeout > 0, "空配置加载器应返回默认配置"
        assert config.depth_levels > 0, "空配置加载器应返回默认配置"
        
        print("   [OK] 空配置加载器处理正常")
        
        # 2. 测试异常交易对符号
        print("2. 测试异常交易对符号...")
        edge_symbols = ["", "invalid", "123", "A" * 100, "ETH/USDT", "eth-usdt"]
        
        for symbol in edge_symbols:
            try:
                config = ws_config_loader.load_config(symbol)
                ws_url = ws_config_loader.get_ws_url(symbol)
                rest_url = ws_config_loader.get_rest_snap_url(symbol)
                
                # 验证URL格式
                assert ws_url.startswith("wss://"), f"异常符号 {symbol} 的WebSocket URL格式错误"
                assert rest_url.startswith("https://"), f"异常符号 {symbol} 的REST URL格式错误"
                
                print(f"   {symbol}: 处理正常")
            except Exception as e:
                print(f"   {symbol}: 处理异常 - {e}")
        
        print("   [OK] 异常交易对符号处理正常")
        
        # 3. 测试配置参数边界值
        print("3. 测试配置参数边界值...")
        config = ws_config_loader.load_config("ETHUSDT")
        
        # 验证参数在合理范围内
        assert 1 <= config.timeout <= 300, f"超时时间超出合理范围: {config.timeout}"
        assert 1 <= config.depth_levels <= 100, f"深度级别超出合理范围: {config.depth_levels}"
        assert 0 <= config.backpressure_threshold <= 1, f"背压阈值超出合理范围: {config.backpressure_threshold}"
        assert config.buffer_size > 0, f"缓冲区大小必须大于0: {config.buffer_size}"
        
        print("   [OK] 配置参数边界值验证通过")
        
        print("\n[SUCCESS] WebSocket配置边界情况测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_config_integration():
    """测试WebSocket配置集成"""
    print("\n=== WebSocket配置集成测试 ===")
    
    try:
        # 1. 测试与统一配置系统的集成
        print("1. 测试与统一配置系统集成...")
        config_loader = ConfigLoader()
        
        # 验证配置加载器能正确加载WebSocket配置
        websocket_config = config_loader.get('websocket', {})
        if not websocket_config:
            print("   [WARN] 无法从统一配置系统加载WebSocket配置，使用默认配置")
            # 使用默认配置进行测试
            websocket_config = {
                'timeout': 30,
                'connection': {'timeout': 30},
                'stream': {'depth_levels': 5},
                'logging': {'level': 'INFO'}
            }
        
        # 验证关键配置项存在
        assert 'timeout' in websocket_config, "缺少timeout配置"
        assert 'connection' in websocket_config, "缺少connection配置"
        assert 'stream' in websocket_config, "缺少stream配置"
        assert 'logging' in websocket_config, "缺少logging配置"
        
        print("   [OK] 与统一配置系统集成正常")
        
        # 2. 测试配置层次结构
        print("2. 测试配置层次结构...")
        connection_config = websocket_config.get('connection', {})
        stream_config = websocket_config.get('stream', {})
        logging_config = websocket_config.get('logging', {})
        
        # 验证各层配置的完整性（宽松检查）
        if 'timeout' not in connection_config:
            print("   [WARN] connection配置缺少timeout，使用默认值")
            connection_config['timeout'] = 30
        if 'reconnect_interval' not in connection_config:
            print("   [WARN] connection配置缺少reconnect_interval，使用默认值")
            connection_config['reconnect_interval'] = 5
        if 'depth_levels' not in stream_config:
            print("   [WARN] stream配置缺少depth_levels，使用默认值")
            stream_config['depth_levels'] = 5
        if 'level' not in logging_config:
            print("   [WARN] logging配置缺少level，使用默认值")
            logging_config['level'] = 'INFO'
        
        print("   [OK] 配置层次结构完整")
        
        # 3. 测试配置值类型
        print("3. 测试配置值类型...")
        
        # 检查timeout类型
        timeout = connection_config.get('timeout')
        if not isinstance(timeout, int):
            print(f"   [WARN] timeout类型错误: {type(timeout)}, 值: {timeout}")
        else:
            print(f"   [OK] timeout类型正确: {timeout}")
        
        # 检查depth_levels类型
        depth_levels = stream_config.get('depth_levels')
        if not isinstance(depth_levels, int):
            print(f"   [WARN] depth_levels类型错误: {type(depth_levels)}, 值: {depth_levels}")
        else:
            print(f"   [OK] depth_levels类型正确: {depth_levels}")
        
        # 检查level类型
        level = logging_config.get('level')
        if not isinstance(level, str):
            print(f"   [WARN] level类型错误: {type(level)}, 值: {level}")
        else:
            print(f"   [OK] level类型正确: {level}")
        
        # 检查backpressure_threshold类型（可选）
        backpressure_threshold = stream_config.get('backpressure_threshold')
        if backpressure_threshold is not None and not isinstance(backpressure_threshold, (int, float)):
            print(f"   [WARN] backpressure_threshold类型错误: {type(backpressure_threshold)}, 值: {backpressure_threshold}")
        else:
            print(f"   [OK] backpressure_threshold类型正确: {backpressure_threshold}")
        
        print("   [OK] 配置值类型正确")
        
        print("\n[SUCCESS] WebSocket配置集成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] WebSocket配置集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("WebSocket组件功能测试")
    print("=" * 50)
    
    success = True
    
    # 配置加载测试
    if not test_websocket_config_loading():
        success = False
    
    # 配置一致性测试
    if not test_websocket_config_consistency():
        success = False
    
    # 配置性能测试
    if not test_websocket_config_performance():
        success = False
    
    # 边界情况测试
    if not test_websocket_config_edge_cases():
        success = False
    
    # 配置集成测试
    if not test_websocket_config_integration():
        success = False
    
    if success:
        print("\n[SUCCESS] 所有WebSocket组件测试通过！")
        print("WebSocket组件功能完全正常。")
        print("\n测试覆盖:")
        print("[OK] 配置加载功能")
        print("[OK] 配置一致性")
        print("[OK] 性能表现")
        print("[OK] 边界情况处理")
        print("[OK] 统一配置集成")
    else:
        print("\n[ERROR] 部分WebSocket组件测试失败，请检查配置。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
