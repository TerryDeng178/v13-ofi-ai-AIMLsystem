"""
V12高频交易执行引擎测试脚本
测试毫秒级订单执行、风险管理、性能指标等核心功能
"""

import sys
import os
import time
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v12_high_frequency_execution_engine import (
    V12HighFrequencyExecutionEngine, 
    Order, OrderType, OrderSide, OrderStatus
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    return {
        'max_slippage_bps': 5,  # 最大滑点5bps
        'max_execution_time_ms': 100,  # 最大执行时间100ms
        'max_position_size': 10000,  # 最大仓位
        'tick_size': 0.01,  # 最小价格变动
        'lot_size': 0.001,  # 最小数量变动
        'max_daily_volume': 100000,  # 最大日交易量
        'max_daily_trades': 1000,  # 最大日交易次数
        'max_daily_loss': 5000,  # 最大日损失
    }

def test_basic_order_execution():
    """测试基本订单执行"""
    logger.info("=" * 80)
    logger.info("测试1: 基本订单执行")
    logger.info("=" * 80)
    
    config = create_test_config()
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)  # 等待引擎启动
        
        # 创建市价买单
        buy_order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 1.0)
        logger.info(f"创建市价买单: {buy_order.order_id}")
        
        # 提交订单
        success = engine.submit_order(buy_order)
        logger.info(f"订单提交结果: {success}")
        
        # 等待执行
        time.sleep(0.5)
        
        # 检查订单状态
        status = engine.get_order_status(buy_order.order_id)
        if status:
            logger.info(f"订单状态: {status.status.value}")
            logger.info(f"成交价格: {status.average_price}")
            logger.info(f"滑点: {status.slippage:.4f}")
            logger.info(f"手续费: {status.fees:.4f}")
        else:
            logger.warning("订单状态获取失败")
        
        # 创建限价卖单
        sell_order = engine.create_limit_order("ETHUSDT", OrderSide.SELL, 0.5, 3005.0)
        logger.info(f"创建限价卖单: {sell_order.order_id}")
        
        # 提交订单
        success = engine.submit_order(sell_order)
        logger.info(f"订单提交结果: {success}")
        
        # 等待执行
        time.sleep(0.5)
        
        # 检查订单状态
        status = engine.get_order_status(sell_order.order_id)
        if status:
            logger.info(f"订单状态: {status.status.value}")
        else:
            logger.warning("订单状态获取失败")
        
        # 获取执行指标
        metrics = engine.get_execution_metrics()
        logger.info(f"总订单数: {metrics.total_orders}")
        logger.info(f"成交订单数: {metrics.filled_orders}")
        logger.info(f"成功率: {metrics.success_rate:.2%}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def test_high_frequency_execution():
    """测试高频执行性能"""
    logger.info("=" * 80)
    logger.info("测试2: 高频执行性能")
    logger.info("=" * 80)
    
    config = create_test_config()
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)
        
        # 创建大量订单
        num_orders = 100
        orders = []
        
        start_time = time.time()
        
        for i in range(num_orders):
            # 随机创建买单或卖单
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            quantity = np.random.uniform(0.1, 2.0)
            
            if i % 3 == 0:
                # 市价单
                order = engine.create_market_order("ETHUSDT", side, quantity)
            else:
                # 限价单
                price = 3000 + np.random.uniform(-10, 10)
                order = engine.create_limit_order("ETHUSDT", side, quantity, price)
            
            orders.append(order)
            engine.submit_order(order)
        
        submission_time = time.time() - start_time
        logger.info(f"提交{num_orders}个订单耗时: {submission_time:.3f}秒")
        
        # 等待所有订单执行
        time.sleep(2.0)
        
        execution_time = time.time() - start_time
        logger.info(f"总执行时间: {execution_time:.3f}秒")
        
        # 统计结果
        filled_count = 0
        total_slippage = 0.0
        total_fees = 0.0
        
        for order in orders:
            status = engine.get_order_status(order.order_id)
            if status and status.status == OrderStatus.FILLED:
                filled_count += 1
                total_slippage += abs(status.slippage)
                total_fees += status.fees
        
        logger.info(f"成交订单数: {filled_count}/{num_orders}")
        logger.info(f"成交率: {filled_count/num_orders:.2%}")
        logger.info(f"平均滑点: {total_slippage/filled_count:.4f}" if filled_count > 0 else "平均滑点: 0")
        logger.info(f"总手续费: {total_fees:.4f}")
        logger.info(f"订单处理速度: {num_orders/execution_time:.1f} 订单/秒")
        
        # 获取性能指标
        metrics = engine.get_execution_metrics()
        logger.info(f"平均执行时间: {metrics.average_execution_time:.2f}ms")
        logger.info(f"平均滑点: {metrics.average_slippage:.4f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def test_risk_management():
    """测试风险管理"""
    logger.info("=" * 80)
    logger.info("测试3: 风险管理")
    logger.info("=" * 80)
    
    config = create_test_config()
    config['max_daily_volume'] = 10  # 设置较小的日交易量限制
    config['max_daily_trades'] = 5   # 设置较小的日交易次数限制
    
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)
        
        # 测试超过日交易量限制
        logger.info("测试日交易量限制...")
        large_order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 20.0)  # 超过限制
        success = engine.submit_order(large_order)
        logger.info(f"大额订单提交结果: {success}")
        
        status = engine.get_order_status(large_order.order_id)
        if status:
            logger.info(f"大额订单状态: {status.status.value}")
        else:
            logger.warning("大额订单状态获取失败")
        
        # 测试超过日交易次数限制
        logger.info("测试日交易次数限制...")
        for i in range(7):  # 超过5次限制
            order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 0.1)
            success = engine.submit_order(order)
            logger.info(f"订单{i+1}提交结果: {success}")
            time.sleep(0.1)
        
        # 测试超过仓位限制
        logger.info("测试仓位限制...")
        oversized_order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 15000)  # 超过最大仓位
        success = engine.submit_order(oversized_order)
        logger.info(f"超大仓位订单提交结果: {success}")
        
        status = engine.get_order_status(oversized_order.order_id)
        if status:
            logger.info(f"超大仓位订单状态: {status.status.value}")
        else:
            logger.warning("超大仓位订单状态获取失败")
        
        # 获取指标
        metrics = engine.get_execution_metrics()
        logger.info(f"总订单数: {metrics.total_orders}")
        logger.info(f"成交订单数: {metrics.filled_orders}")
        logger.info(f"拒绝订单数: {metrics.rejected_orders}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def test_order_types():
    """测试不同订单类型"""
    logger.info("=" * 80)
    logger.info("测试4: 不同订单类型")
    logger.info("=" * 80)
    
    config = create_test_config()
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)
        
        # 测试市价单
        logger.info("测试市价单...")
        market_order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 1.0)
        engine.submit_order(market_order)
        time.sleep(0.2)
        
        status = engine.get_order_status(market_order.order_id)
        if status:
            logger.info(f"市价单状态: {status.status.value}")
        else:
            logger.warning("市价单状态获取失败")
        
        # 测试限价单
        logger.info("测试限价单...")
        limit_order = engine.create_limit_order("ETHUSDT", OrderSide.SELL, 1.0, 2990.0)  # 低于市价
        engine.submit_order(limit_order)
        time.sleep(0.2)
        
        status = engine.get_order_status(limit_order.order_id)
        if status:
            logger.info(f"限价单状态: {status.status.value}")
        else:
            logger.warning("限价单状态获取失败")
        
        # 测试IOC订单
        logger.info("测试IOC订单...")
        ioc_order = engine.create_ioc_order("ETHUSDT", OrderSide.BUY, 1.0, 3010.0)  # 高于市价
        engine.submit_order(ioc_order)
        time.sleep(0.2)
        
        status = engine.get_order_status(ioc_order.order_id)
        if status:
            logger.info(f"IOC订单状态: {status.status.value}")
        else:
            logger.warning("IOC订单状态获取失败")
        
        # 测试FOK订单
        logger.info("测试FOK订单...")
        fok_order = engine.create_fok_order("ETHUSDT", OrderSide.SELL, 1.0, 3020.0)  # 高于市价
        engine.submit_order(fok_order)
        time.sleep(0.2)
        
        status = engine.get_order_status(fok_order.order_id)
        if status:
            logger.info(f"FOK订单状态: {status.status.value}")
        else:
            logger.warning("FOK订单状态获取失败")
        
        # 获取指标
        metrics = engine.get_execution_metrics()
        logger.info(f"总订单数: {metrics.total_orders}")
        logger.info(f"成交订单数: {metrics.filled_orders}")
        logger.info(f"取消订单数: {metrics.cancelled_orders}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def test_order_cancellation():
    """测试订单取消"""
    logger.info("=" * 80)
    logger.info("测试5: 订单取消")
    logger.info("=" * 80)
    
    config = create_test_config()
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)
        
        # 创建限价单（不太可能立即成交）
        limit_order = engine.create_limit_order("ETHUSDT", OrderSide.SELL, 1.0, 3200.0)  # 远高于市价
        engine.submit_order(limit_order)
        
        # 立即取消订单
        time.sleep(0.05)  # 短暂等待
        success = engine.cancel_order(limit_order.order_id)
        logger.info(f"订单取消结果: {success}")
        
        # 检查订单状态
        status = engine.get_order_status(limit_order.order_id)
        if status:
            logger.info(f"订单状态: {status.status.value}")
        else:
            logger.warning("订单状态获取失败")
        
        # 尝试取消不存在的订单
        success = engine.cancel_order("NON_EXISTENT_ORDER")
        logger.info(f"取消不存在订单结果: {success}")
        
        # 获取指标
        metrics = engine.get_execution_metrics()
        logger.info(f"总订单数: {metrics.total_orders}")
        logger.info(f"取消订单数: {metrics.cancelled_orders}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def test_performance_monitoring():
    """测试性能监控"""
    logger.info("=" * 80)
    logger.info("测试6: 性能监控")
    logger.info("=" * 80)
    
    config = create_test_config()
    engine = V12HighFrequencyExecutionEngine(config)
    
    try:
        # 启动引擎
        engine.start()
        time.sleep(0.1)
        
        # 执行一些订单
        for i in range(10):
            order = engine.create_market_order("ETHUSDT", OrderSide.BUY, 0.5)
            engine.submit_order(order)
            time.sleep(0.05)
        
        time.sleep(1.0)  # 等待执行完成
        
        # 获取性能摘要
        summary = engine.get_performance_summary()
        logger.info("性能摘要:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # 获取执行指标
        metrics = engine.get_execution_metrics()
        logger.info("执行指标:")
        logger.info(f"  总订单数: {metrics.total_orders}")
        logger.info(f"  成交订单数: {metrics.filled_orders}")
        logger.info(f"  成功率: {metrics.success_rate:.2%}")
        logger.info(f"  平均执行时间: {metrics.average_execution_time:.2f}ms")
        logger.info(f"  平均滑点: {metrics.average_slippage:.4f}")
        logger.info(f"  总手续费: {metrics.total_fees:.4f}")
        logger.info(f"  总执行成本: {metrics.execution_cost:.4f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        engine.stop()

def main():
    """主测试函数"""
    logger.info("=" * 80)
    logger.info("V12高频交易执行引擎测试开始")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 运行所有测试
        test_basic_order_execution()
        time.sleep(1)
        
        test_high_frequency_execution()
        time.sleep(1)
        
        test_risk_management()
        time.sleep(1)
        
        test_order_types()
        time.sleep(1)
        
        test_order_cancellation()
        time.sleep(1)
        
        test_performance_monitoring()
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"V12高频交易执行引擎测试完成，总耗时: {total_time:.2f}秒")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
