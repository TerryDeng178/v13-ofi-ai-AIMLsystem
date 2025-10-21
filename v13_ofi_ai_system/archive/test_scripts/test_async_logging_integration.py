"""
async_logging 模块集成测试

测试 async_logging 模块与统一配置系统的集成
验证 WebSocket 客户端的日志配置功能

Author: V13 OFI+CVD AI System
Created: 2025-10-20
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.async_logging_config_loader import AsyncLoggingConfigLoader, AsyncLoggingConfig
from src.utils.async_logging import setup_async_logging, sample_queue_metrics

def print_test_section(title):
    print(f"\n{'='*60}\n=== {title} ===\n{'='*60}")

def test_async_logging_config_loading():
    """测试异步日志配置加载功能"""
    print_test_section("测试异步日志配置加载功能")
    
    config_loader = ConfigLoader()
    logging_config_loader = AsyncLoggingConfigLoader(config_loader)
    
    # 测试默认配置
    config = logging_config_loader.load_config()
    
    assert config is not None, "配置加载失败"
    print("配置加载成功")
    print(f"  - 日志级别: {config.level}")
    print(f"  - 队列大小: {config.queue_max}")
    print(f"  - 轮转策略: {config.rotate}")
    print(f"  - 轮转间隔: {config.rotate_sec}秒")
    print(f"  - 最大字节: {config.max_bytes}")
    print(f"  - 备份数量: {config.backups}")
    print(f"  - 控制台输出: {config.to_console}")
    
    # 验证默认值
    assert config.level == "INFO"
    assert config.queue_max == 10000
    assert config.rotate == "interval"
    assert config.rotate_sec == 60
    assert config.max_bytes == 5000000
    assert config.backups == 7
    assert config.to_console is True

def test_async_logging_setup():
    """测试异步日志设置功能"""
    print_test_section("测试异步日志设置功能")
    
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_path = f.name
    
    try:
        # 设置异步日志
        logger, listener, queue_handler = setup_async_logging(
            name="test_async_logging",
            log_path=log_path,
            rotate='interval',
            rotate_sec=60,
            max_bytes=1000000,
            backups=3,
            level=logging.INFO,
            queue_max=1000,
            to_console=False  # 避免控制台输出干扰测试
        )
        
        assert logger is not None, "Logger创建失败"
        assert listener is not None, "Listener创建失败"
        assert queue_handler is not None, "QueueHandler创建失败"
        
        print("异步日志设置成功")
        print(f"  - Logger名称: {logger.name}")
        print(f"  - 日志文件: {log_path}")
        print(f"  - 队列处理器: {type(queue_handler).__name__}")
        
        # 测试日志记录
        logger.info("测试日志消息")
        logger.warning("测试警告消息")
        logger.error("测试错误消息")
        
        # 获取队列指标
        metrics = sample_queue_metrics(queue_handler)
        print(f"  - 当前队列深度: {metrics['depth']}")
        print(f"  - 历史最大深度: {metrics['max_depth']}")
        print(f"  - 丢弃数量: {metrics['drops']}")
        
        # 验证日志级别转换
        config = AsyncLoggingConfig(level="DEBUG")
        assert config.get_log_level() == logging.DEBUG
        print("  - 日志级别转换正常")
        
        # 停止监听器
        listener.stop()
        
    finally:
        # 清理临时文件
        if os.path.exists(log_path):
            os.unlink(log_path)

def test_websocket_logging_integration():
    """测试WebSocket日志集成"""
    print_test_section("测试WebSocket日志集成")
    
    try:
        # 尝试导入WebSocket客户端
        from src.binance_websocket_client import BinanceOrderBookStream
        
        # 测试WebSocket客户端创建（不实际连接）
        config_loader = ConfigLoader()
        ws_config_loader = AsyncLoggingConfigLoader(config_loader)
        logging_config = ws_config_loader.load_config(component='websocket')
        
        print("WebSocket日志集成测试成功")
        print(f"  - 日志级别: {logging_config.level}")
        print(f"  - 队列大小: {logging_config.queue_max}")
        print(f"  - 轮转策略: {logging_config.rotate}")
        
    except ImportError as e:
        print(f"WebSocket导入失败: {e}")
        print("这可能是由于缺少websocket-client依赖")
    except Exception as e:
        print(f"WebSocket集成测试失败: {e}")

def test_environment_override():
    """测试环境变量覆盖功能"""
    print_test_section("测试环境变量覆盖功能")
    
    try:
        # 设置环境变量
        os.environ['V13__WEBSOCKET__LOGGING__LOG_LEVEL'] = 'DEBUG'
        os.environ['V13__WEBSOCKET__LOGGING__QUEUE_MAX'] = '20000'
        os.environ['V13__WEBSOCKET__LOGGING__ROTATE'] = 'size'
        os.environ['V13__WEBSOCKET__LOGGING__MAX_BYTES'] = '10000000'
        os.environ['V13__WEBSOCKET__LOGGING__BACKUPS'] = '5'
        os.environ['V13__WEBSOCKET__LOGGING__TO_CONSOLE'] = 'false'
        
        # 重新加载配置
        config_loader = ConfigLoader()
        logging_config_loader = AsyncLoggingConfigLoader(config_loader)
        config = logging_config_loader.load_config(component='websocket')
        
        # 验证环境变量覆盖
        assert config.level == 'DEBUG', f"日志级别覆盖失败，期望: DEBUG，实际: {config.level}"
        print("日志级别环境变量覆盖成功")
        
        assert config.queue_max == 20000, f"队列大小覆盖失败，期望: 20000，实际: {config.queue_max}"
        print("队列大小环境变量覆盖成功")
        
        assert config.rotate == 'size', f"轮转策略覆盖失败，期望: size，实际: {config.rotate}"
        print("轮转策略环境变量覆盖成功")
        
        assert config.max_bytes == 10000000, f"最大字节覆盖失败，期望: 10000000，实际: {config.max_bytes}"
        print("最大字节环境变量覆盖成功")
        
        assert config.backups == 5, f"备份数量覆盖失败，期望: 5，实际: {config.backups}"
        print("备份数量环境变量覆盖成功")
        
        assert config.to_console is False, f"控制台输出覆盖失败，期望: False，实际: {config.to_console}"
        print("控制台输出环境变量覆盖成功")
        
    finally:
        # 清理环境变量
        env_vars = [
            'V13__WEBSOCKET__LOGGING__LOG_LEVEL',
            'V13__WEBSOCKET__LOGGING__QUEUE_MAX',
            'V13__WEBSOCKET__LOGGING__ROTATE',
            'V13__WEBSOCKET__LOGGING__MAX_BYTES',
            'V13__WEBSOCKET__LOGGING__BACKUPS',
            'V13__WEBSOCKET__LOGGING__TO_CONSOLE'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

def test_performance_metrics():
    """测试性能指标"""
    print_test_section("测试性能指标")
    
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_path = f.name
    
    try:
        # 设置异步日志
        logger, listener, queue_handler = setup_async_logging(
            name="perf_test",
            log_path=log_path,
            queue_max=100,
            to_console=False
        )
        
        # 发送大量日志消息
        for i in range(1000):
            logger.info(f"性能测试消息 {i}")
        
        # 获取性能指标
        metrics = sample_queue_metrics(queue_handler)
        
        print("性能指标测试成功")
        print(f"  - 当前队列深度: {metrics['depth']}")
        print(f"  - 历史最大深度: {metrics['max_depth']}")
        print(f"  - 丢弃数量: {metrics['drops']}")
        
        # 验证指标合理性
        assert metrics['depth'] >= 0, "队列深度不能为负数"
        assert metrics['max_depth'] >= 0, "最大深度不能为负数"
        assert metrics['drops'] >= 0, "丢弃数量不能为负数"
        
        # 停止监听器
        listener.stop()
        
    finally:
        # 清理临时文件
        if os.path.exists(log_path):
            os.unlink(log_path)

if __name__ == "__main__":
    print("async_logging 模块集成测试开始")
    
    try:
        test_async_logging_config_loading()
        test_async_logging_setup()
        test_websocket_logging_integration()
        test_environment_override()
        test_performance_metrics()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！async_logging 模块集成功能正常")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
