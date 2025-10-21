"""
交易流处理配置集成测试

测试交易流处理模块与统一配置系统的集成
包括配置加载、环境变量覆盖、功能测试等

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
import sys
import pytest
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.trade_stream_config_loader import TradeStreamConfigLoader, TradeStreamConfig
from src.binance_trade_stream import TradeStreamProcessor

def print_test_section(title):
    print(f"\n{'='*60}\n=== {title} ===\n{'='*60}")

def test_trade_stream_config_loading():
    """测试交易流处理配置加载功能"""
    print_test_section("测试交易流处理配置加载功能")
    
    config_loader = ConfigLoader()
    config = config_loader.get('trade_stream')
    
    assert config is not None, "配置加载失败"
    print("配置加载成功")
    print(f"  - 启用状态: {config.get('enabled')}")
    print(f"  - 队列大小: {config.get('queue', {}).get('size')}")
    print(f"  - 最大队列大小: {config.get('queue', {}).get('max_size')}")
    print(f"  - 背压阈值: {config.get('queue', {}).get('backpressure_threshold')}")
    print(f"  - 打印间隔: {config.get('logging', {}).get('print_every')}")
    print(f"  - 统计间隔: {config.get('logging', {}).get('stats_interval')}")
    print(f"  - 日志级别: {config.get('logging', {}).get('log_level')}")
    print(f"  - 心跳超时: {config.get('websocket', {}).get('heartbeat_timeout')}")
    print(f"  - 最大退避: {config.get('websocket', {}).get('backoff_max')}")
    print(f"  - Ping间隔: {config.get('websocket', {}).get('ping_interval')}")
    print(f"  - 关闭超时: {config.get('websocket', {}).get('close_timeout')}")
    print(f"  - 重连延迟: {config.get('websocket', {}).get('reconnect_delay')}")
    print(f"  - 最大重连次数: {config.get('websocket', {}).get('max_reconnect_attempts')}")
    print(f"  - 水位线毫秒: {config.get('performance', {}).get('watermark_ms')}")
    print(f"  - 批处理大小: {config.get('performance', {}).get('batch_size')}")
    print(f"  - 最大处理速率: {config.get('performance', {}).get('max_processing_rate')}")
    print(f"  - 内存限制MB: {config.get('performance', {}).get('memory_limit_mb')}")
    print(f"  - Prometheus端口: {config.get('monitoring', {}).get('prometheus', {}).get('port')}")
    print(f"  - 告警启用: {config.get('monitoring', {}).get('alerts', {}).get('enabled')}")
    print(f"  - 热更新启用: {config.get('hot_reload', {}).get('enabled')}")
    
    # 验证关键配置值
    assert config.get('enabled') is True
    assert config.get('queue', {}).get('size') == 1024
    assert config.get('queue', {}).get('max_size') == 2048
    assert config.get('queue', {}).get('backpressure_threshold') == 0.8
    assert config.get('logging', {}).get('print_every') == 100
    assert config.get('logging', {}).get('stats_interval') == 60.0
    assert config.get('logging', {}).get('log_level') == 'INFO'
    assert config.get('websocket', {}).get('heartbeat_timeout') == 30
    assert config.get('websocket', {}).get('backoff_max') == 15
    assert config.get('websocket', {}).get('ping_interval') == 20
    assert config.get('websocket', {}).get('close_timeout') == 10
    assert config.get('websocket', {}).get('reconnect_delay') == 1.0
    assert config.get('websocket', {}).get('max_reconnect_attempts') == 10
    assert config.get('performance', {}).get('watermark_ms') == 1000
    assert config.get('performance', {}).get('batch_size') == 10
    assert config.get('performance', {}).get('max_processing_rate') == 1000
    assert config.get('performance', {}).get('memory_limit_mb') == 100
    assert config.get('monitoring', {}).get('prometheus', {}).get('port') == 8008
    assert config.get('monitoring', {}).get('alerts', {}).get('enabled') is True
    assert config.get('hot_reload', {}).get('enabled') is True

def test_trade_stream_config_loader():
    """测试交易流处理配置加载器"""
    print_test_section("测试交易流处理配置加载器")
    
    config_loader = ConfigLoader()
    trade_config_loader = TradeStreamConfigLoader(config_loader)
    config = trade_config_loader.load_config()
    
    assert config is not None, "配置加载器创建失败"
    assert isinstance(config, TradeStreamConfig), "配置类型错误"
    
    print("配置加载器创建成功")
    print(f"  - 启用状态: {config.enabled}")
    print(f"  - 队列大小: {config.queue.size}")
    print(f"  - 最大队列大小: {config.queue.max_size}")
    print(f"  - 背压阈值: {config.queue.backpressure_threshold}")
    print(f"  - 打印间隔: {config.logging.print_every}")
    print(f"  - 统计间隔: {config.logging.stats_interval}")
    print(f"  - 日志级别: {config.logging.log_level}")
    print(f"  - 心跳超时: {config.websocket.heartbeat_timeout}")
    print(f"  - 最大退避: {config.websocket.backoff_max}")
    print(f"  - Ping间隔: {config.websocket.ping_interval}")
    print(f"  - 关闭超时: {config.websocket.close_timeout}")
    print(f"  - 重连延迟: {config.websocket.reconnect_delay}")
    print(f"  - 最大重连次数: {config.websocket.max_reconnect_attempts}")
    print(f"  - 水位线毫秒: {config.performance.watermark_ms}")
    print(f"  - 批处理大小: {config.performance.batch_size}")
    print(f"  - 最大处理速率: {config.performance.max_processing_rate}")
    print(f"  - 内存限制MB: {config.performance.memory_limit_mb}")
    print(f"  - Prometheus端口: {config.monitoring.prometheus_port}")
    print(f"  - 告警启用: {config.monitoring.alerts_enabled}")
    print(f"  - 热更新启用: {config.hot_reload.enabled}")
    
    # 验证配置值
    assert config.enabled is True
    assert config.queue.size == 1024
    assert config.queue.max_size == 2048
    assert config.queue.backpressure_threshold == 0.8
    assert config.logging.print_every == 100
    assert config.logging.stats_interval == 60.0
    assert config.logging.log_level == 'INFO'
    assert config.websocket.heartbeat_timeout == 30
    assert config.websocket.backoff_max == 15
    assert config.websocket.ping_interval == 20
    assert config.websocket.close_timeout == 10
    assert config.websocket.reconnect_delay == 1.0
    assert config.websocket.max_reconnect_attempts == 10
    assert config.performance.watermark_ms == 1000
    assert config.performance.batch_size == 10
    assert config.performance.max_processing_rate == 1000
    assert config.performance.memory_limit_mb == 100
    assert config.monitoring.prometheus_port == 8008
    assert config.monitoring.alerts_enabled is True
    assert config.hot_reload.enabled is True

def test_trade_stream_processor_creation():
    """测试交易流处理器创建"""
    print_test_section("测试交易流处理器创建")
    
    # 1. 使用配置加载器
    config_loader = ConfigLoader()
    processor_with_config = TradeStreamProcessor(config_loader=config_loader)
    
    assert processor_with_config is not None, "带配置的处理器创建失败"
    assert processor_with_config.config is not None, "配置未加载"
    print("带配置的处理器创建成功")
    
    # 2. 不使用配置加载器（默认配置）
    processor_default = TradeStreamProcessor()
    
    assert processor_default is not None, "默认处理器创建失败"
    print("默认处理器创建成功")
    
    # 3. 测试配置获取方法
    websocket_config = processor_with_config.get_websocket_config()
    queue_config = processor_with_config.get_queue_config()
    logging_config = processor_with_config.get_logging_config()
    performance_config = processor_with_config.get_performance_config()
    
    assert websocket_config is not None, "WebSocket配置获取失败"
    assert queue_config is not None, "队列配置获取失败"
    assert logging_config is not None, "日志配置获取失败"
    assert performance_config is not None, "性能配置获取失败"
    
    print("配置获取方法测试成功")
    print(f"  - WebSocket心跳超时: {websocket_config.heartbeat_timeout}")
    print(f"  - 队列大小: {queue_config.size}")
    print(f"  - 打印间隔: {logging_config.print_every}")
    print(f"  - 水位线毫秒: {performance_config.watermark_ms}")

def test_backward_compatibility():
    """测试向后兼容性"""
    print_test_section("测试向后兼容性")
    
    # 1. 默认配置
    processor_default = TradeStreamProcessor()
    assert processor_default.config is None, "默认配置应该为None"
    print("  - 默认配置: 支持")
    
    # 2. 统一配置系统
    config_loader = ConfigLoader()
    processor_unified = TradeStreamProcessor(config_loader=config_loader)
    assert processor_unified.config is not None, "统一配置系统支持失败"
    print("  - 统一配置系统: 支持")
    
    # 3. 配置获取方法在两种模式下都能工作
    websocket_config_default = processor_default.get_websocket_config()
    websocket_config_unified = processor_unified.get_websocket_config()
    
    assert websocket_config_default is not None, "默认配置获取失败"
    assert websocket_config_unified is not None, "统一配置获取失败"
    print("  - 配置获取方法: 支持")
    
    print("向后兼容性测试成功")

def test_environment_override():
    """测试环境变量覆盖功能"""
    print_test_section("测试环境变量覆盖功能")
    
    try:
        # 设置环境变量
        os.environ['V13__TRADE_STREAM__QUEUE__SIZE'] = '2048'
        os.environ['V13__TRADE_STREAM__LOGGING__PRINT_EVERY'] = '200'
        os.environ['V13__TRADE_STREAM__WEBSOCKET__HEARTBEAT_TIMEOUT'] = '60'
        os.environ['V13__TRADE_STREAM__PERFORMANCE__WATERMARK_MS'] = '3000'
        os.environ['V13__TRADE_STREAM__MONITORING__PROMETHEUS__PORT'] = '9008'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        processor = TradeStreamProcessor(config_loader=config_loader)
        
        # 验证环境变量覆盖
        assert processor.config.queue.size == 2048, f"队列大小环境变量覆盖失败，期望: 2048，实际: {processor.config.queue.size}"
        print("队列大小环境变量覆盖成功")
        
        assert processor.config.logging.print_every == 200, f"打印间隔环境变量覆盖失败，期望: 200，实际: {processor.config.logging.print_every}"
        print("打印间隔环境变量覆盖成功")
        
        assert processor.config.websocket.heartbeat_timeout == 60, f"心跳超时环境变量覆盖失败，期望: 60，实际: {processor.config.websocket.heartbeat_timeout}"
        print("心跳超时环境变量覆盖成功")
        
        assert processor.config.performance.watermark_ms == 3000, f"水位线毫秒环境变量覆盖失败，期望: 3000，实际: {processor.config.performance.watermark_ms}"
        print("水位线毫秒环境变量覆盖成功")
        
        assert processor.config.monitoring.prometheus_port == 9008, f"Prometheus端口环境变量覆盖失败，期望: 9008，实际: {processor.config.monitoring.prometheus_port}"
        print("Prometheus端口环境变量覆盖成功")
        
    finally:
        # 清理环境变量
        del os.environ['V13__TRADE_STREAM__QUEUE__SIZE']
        del os.environ['V13__TRADE_STREAM__LOGGING__PRINT_EVERY']
        del os.environ['V13__TRADE_STREAM__WEBSOCKET__HEARTBEAT_TIMEOUT']
        del os.environ['V13__TRADE_STREAM__PERFORMANCE__WATERMARK_MS']
        del os.environ['V13__TRADE_STREAM__MONITORING__PROMETHEUS__PORT']
    
    print("环境变量覆盖测试成功")

def test_config_methods():
    """测试配置方法"""
    print_test_section("测试配置方法")
    
    config_loader = ConfigLoader()
    processor = TradeStreamProcessor(config_loader=config_loader)
    
    # 队列配置
    queue_config = processor.get_queue_config()
    assert queue_config.size == 1024
    assert queue_config.max_size == 2048
    assert queue_config.backpressure_threshold == 0.8
    print("队列配置方法正常")
    
    # 日志配置
    logging_config = processor.get_logging_config()
    assert logging_config.print_every == 100
    assert logging_config.stats_interval == 60.0
    assert logging_config.log_level == 'INFO'
    print("日志配置方法正常")
    
    # WebSocket配置
    websocket_config = processor.get_websocket_config()
    assert websocket_config.heartbeat_timeout == 30
    assert websocket_config.backoff_max == 15
    assert websocket_config.ping_interval == 20
    assert websocket_config.close_timeout == 10
    assert websocket_config.reconnect_delay == 1.0
    assert websocket_config.max_reconnect_attempts == 10
    print("WebSocket配置方法正常")
    
    # 性能配置
    performance_config = processor.get_performance_config()
    assert performance_config.watermark_ms == 1000
    assert performance_config.batch_size == 10
    assert performance_config.max_processing_rate == 1000
    assert performance_config.memory_limit_mb == 100
    print("性能配置方法正常")
    
    # 监控配置
    monitoring_config = processor.get_monitoring_config()
    assert monitoring_config.prometheus_port == 8008
    assert monitoring_config.prometheus_path == '/metrics'
    assert monitoring_config.prometheus_scrape_interval == '5s'
    assert monitoring_config.alerts_enabled is True
    print("监控配置方法正常")
    
    print("配置方法测试成功")

def test_trade_stream_functionality():
    """测试交易流处理功能（不实际连接WebSocket）"""
    print_test_section("测试交易流处理功能")
    
    config_loader = ConfigLoader()
    processor = TradeStreamProcessor(config_loader=config_loader)
    
    # 测试配置获取
    websocket_config = processor.get_websocket_config()
    queue_config = processor.get_queue_config()
    logging_config = processor.get_logging_config()
    performance_config = processor.get_performance_config()
    
    # 验证配置完整性
    assert websocket_config.heartbeat_timeout > 0, "心跳超时必须大于0"
    assert websocket_config.backoff_max > 0, "最大退避必须大于0"
    assert websocket_config.ping_interval > 0, "Ping间隔必须大于0"
    assert websocket_config.close_timeout > 0, "关闭超时必须大于0"
    assert websocket_config.reconnect_delay > 0, "重连延迟必须大于0"
    assert websocket_config.max_reconnect_attempts > 0, "最大重连次数必须大于0"
    
    assert queue_config.size > 0, "队列大小必须大于0"
    assert queue_config.max_size > 0, "最大队列大小必须大于0"
    assert 0 < queue_config.backpressure_threshold <= 1, "背压阈值必须在0-1之间"
    
    assert logging_config.print_every > 0, "打印间隔必须大于0"
    assert logging_config.stats_interval > 0, "统计间隔必须大于0"
    assert logging_config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], "日志级别无效"
    
    assert performance_config.watermark_ms > 0, "水位线毫秒必须大于0"
    assert performance_config.batch_size > 0, "批处理大小必须大于0"
    assert performance_config.max_processing_rate > 0, "最大处理速率必须大于0"
    assert performance_config.memory_limit_mb > 0, "内存限制必须大于0"
    
    print("交易流处理功能配置验证成功")
    print(f"  - WebSocket配置: 心跳超时={websocket_config.heartbeat_timeout}s, 最大退避={websocket_config.backoff_max}s")
    print(f"  - 队列配置: 大小={queue_config.size}, 最大={queue_config.max_size}, 背压阈值={queue_config.backpressure_threshold}")
    print(f"  - 日志配置: 打印间隔={logging_config.print_every}, 统计间隔={logging_config.stats_interval}s, 级别={logging_config.log_level}")
    print(f"  - 性能配置: 水位线={performance_config.watermark_ms}ms, 批处理={performance_config.batch_size}, 最大速率={performance_config.max_processing_rate}")
    
    print("交易流处理功能测试成功")

if __name__ == "__main__":
    print("交易流处理配置集成测试开始")
    test_trade_stream_config_loading()
    test_trade_stream_config_loader()
    test_trade_stream_processor_creation()
    test_backward_compatibility()
    test_environment_override()
    test_config_methods()
    test_trade_stream_functionality()
    print("\n============================================================\n所有测试通过！交易流处理配置集成功能正常")
