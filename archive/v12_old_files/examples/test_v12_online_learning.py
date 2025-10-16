"""
V12在线学习系统测试脚本
测试实时模型更新、增量学习、性能监控等核心功能
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v12_online_learning_system import V12OnlineLearningSystem, LearningMetrics, ModelPerformance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    return {
        'learning_interval': 5,  # 学习间隔5秒
        'batch_size': 50,  # 批次大小
        'min_samples_for_update': 20,  # 最小更新样本数
        'performance_threshold': 0.01,  # 性能阈值1%
        'max_models': 5,  # 最大模型数量
    }

def generate_mock_training_data(num_samples: int = 100) -> list:
    """生成模拟训练数据"""
    data_list = []
    
    for i in range(num_samples):
        # 模拟市场数据
        base_price = 3000 + np.random.normal(0, 50)
        
        # 模拟特征数据
        data = {
            'timestamp': datetime.now() - timedelta(seconds=num_samples-i),
            'ofi_z': np.random.normal(0, 2),
            'cvd_z': np.random.normal(0, 2),
            'real_ofi_z': np.random.normal(0, 2),
            'real_cvd_z': np.random.normal(0, 2),
            'ofi_momentum_1s': np.random.normal(0, 1),
            'ofi_momentum_5s': np.random.normal(0, 1),
            'cvd_momentum_1s': np.random.normal(0, 1),
            'cvd_momentum_5s': np.random.normal(0, 1),
            'spread_bps': np.random.uniform(0.1, 5.0),
            'depth_ratio': np.random.uniform(0.5, 2.0),
            'price_volatility': np.random.uniform(0.01, 0.05),
            'ofi_volatility': np.random.uniform(0.1, 2.0),
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 1),
            'bollinger_upper': base_price * 1.02,
            'bollinger_lower': base_price * 0.98,
            'trend_strength': np.random.uniform(-1, 1),
            'volatility_regime': np.random.choice([0, 1, 2]),
            'bid1_size': np.random.uniform(10, 100),
            'ask1_size': np.random.uniform(10, 100),
            'bid_ask_ratio': np.random.uniform(0.5, 2.0),
            'mid_price_change_1s': np.random.normal(0, 0.001),
            'volume_change_1s': np.random.normal(0, 0.1),
            'num_trades_change_1s': np.random.normal(0, 0.05),
            'taker_buy_sell_ratio': np.random.uniform(0.3, 0.7),
            'vwap_deviation': np.random.normal(0, 0.002),
            'atr_normalized': np.random.uniform(0.005, 0.02),
            'z_score_spread': np.random.normal(0, 1),
            'z_score_depth': np.random.normal(0, 1),
            'z_score_volume': np.random.normal(0, 1),
            'close': base_price,
            'future_close': base_price * (1 + np.random.normal(0, 0.01)),  # 模拟未来价格
            'metadata': {
                'symbol': 'ETHUSDT',
                'timeframe': '1m',
                'data_quality': 'high'
            }
        }
        data_list.append(data)
    
    return data_list

def test_basic_learning_functionality():
    """测试基本学习功能"""
    logger.info("=" * 80)
    logger.info("测试1: 基本学习功能")
    logger.info("=" * 80)
    
    config = create_test_config()
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)  # 等待系统启动
        
        # 生成并添加训练数据
        logger.info("生成模拟训练数据...")
        training_data = generate_mock_training_data(100)
        
        logger.info("添加训练数据到学习系统...")
        for data in training_data:
            learning_system.add_training_data(data)
        
        # 等待学习周期完成
        logger.info("等待学习周期完成...")
        time.sleep(10)  # 等待两个学习周期
        
        # 检查学习指标
        metrics = learning_system.get_learning_metrics()
        logger.info(f"学习周期数: {metrics.learning_cycles}")
        logger.info(f"模型更新数: {metrics.model_updates}")
        logger.info(f"准确度提升次数: {metrics.accuracy_improvements}")
        logger.info(f"平均准确度: {metrics.average_accuracy:.4f}")
        logger.info(f"最佳准确度: {metrics.best_accuracy:.4f}")
        logger.info(f"性能趋势: {metrics.performance_trend}")
        
        # 检查模型性能
        for model_name in ['ofi_expert', 'lstm', 'transformer', 'cnn']:
            performance = learning_system.get_model_performance(model_name)
            if performance:
                logger.info(f"模型 {model_name}: 准确度={performance.accuracy:.4f}, "
                           f"损失={performance.loss:.4f}, 置信度={performance.confidence:.4f}")
        
        # 获取性能摘要
        summary = learning_system.get_performance_summary()
        logger.info("性能摘要:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def test_incremental_learning():
    """测试增量学习"""
    logger.info("=" * 80)
    logger.info("测试2: 增量学习")
    logger.info("=" * 80)
    
    config = create_test_config()
    config['learning_interval'] = 3  # 更短的学习间隔
    config['min_samples_for_update'] = 10  # 更小的最小样本数
    
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)
        
        # 分批添加数据，模拟增量学习
        logger.info("开始增量学习测试...")
        
        for batch in range(5):
            logger.info(f"添加第 {batch+1} 批数据...")
            batch_data = generate_mock_training_data(20)
            
            for data in batch_data:
                learning_system.add_training_data(data)
            
            # 等待学习周期
            time.sleep(4)
            
            # 检查学习进度
            metrics = learning_system.get_learning_metrics()
            logger.info(f"批次 {batch+1}: 学习周期={metrics.learning_cycles}, "
                       f"平均准确度={metrics.average_accuracy:.4f}, "
                       f"性能趋势={metrics.performance_trend}")
        
        # 最终统计
        summary = learning_system.get_performance_summary()
        logger.info("增量学习最终结果:")
        logger.info(f"  总处理样本数: {summary['total_samples_processed']}")
        logger.info(f"  完成学习周期: {summary['learning_cycles_completed']}")
        logger.info(f"  模型更新次数: {summary['model_updates']}")
        logger.info(f"  准确度提升次数: {summary['accuracy_improvements']}")
        logger.info(f"  学习效率: {summary['learning_efficiency']:.4f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def test_model_performance_monitoring():
    """测试模型性能监控"""
    logger.info("=" * 80)
    logger.info("测试3: 模型性能监控")
    logger.info("=" * 80)
    
    config = create_test_config()
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)
        
        # 添加大量数据以触发多次学习
        logger.info("添加大量训练数据...")
        training_data = generate_mock_training_data(200)
        
        for data in training_data:
            learning_system.add_training_data(data)
        
        # 等待学习完成
        time.sleep(15)
        
        # 详细检查每个模型的性能
        logger.info("模型性能详细分析:")
        for model_name in ['ofi_expert', 'lstm', 'transformer', 'cnn']:
            performance = learning_system.get_model_performance(model_name)
            if performance:
                logger.info(f"模型: {model_name}")
                logger.info(f"  准确度: {performance.accuracy:.4f}")
                logger.info(f"  损失: {performance.loss:.4f}")
                logger.info(f"  置信度: {performance.confidence:.4f}")
                logger.info(f"  预测时间: {performance.prediction_time:.4f}ms")
                logger.info(f"  处理样本数: {performance.samples_processed}")
                logger.info(f"  改进率: {performance.improvement_rate:.4f}")
                logger.info(f"  更新时间: {performance.update_time}")
                logger.info("")
        
        # 学习指标分析
        metrics = learning_system.get_learning_metrics()
        logger.info("学习指标分析:")
        logger.info(f"  总样本数: {metrics.total_samples}")
        logger.info(f"  学习周期数: {metrics.learning_cycles}")
        logger.info(f"  模型更新数: {metrics.model_updates}")
        logger.info(f"  准确度提升数: {metrics.accuracy_improvements}")
        logger.info(f"  当前准确度: {metrics.last_accuracy:.4f}")
        logger.info(f"  最佳准确度: {metrics.best_accuracy:.4f}")
        logger.info(f"  平均准确度: {metrics.average_accuracy:.4f}")
        logger.info(f"  学习率: {metrics.learning_rate:.6f}")
        logger.info(f"  收敛率: {metrics.convergence_rate:.6f}")
        logger.info(f"  性能趋势: {metrics.performance_trend}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def test_model_persistence():
    """测试模型持久化"""
    logger.info("=" * 80)
    logger.info("测试4: 模型持久化")
    logger.info("=" * 80)
    
    config = create_test_config()
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)
        
        # 添加训练数据并学习
        logger.info("训练模型...")
        training_data = generate_mock_training_data(100)
        
        for data in training_data:
            learning_system.add_training_data(data)
        
        time.sleep(10)  # 等待学习完成
        
        # 保存模型
        save_path = "models/v12_online_learning"
        logger.info(f"保存模型到: {save_path}")
        learning_system.save_models(save_path)
        
        # 检查保存的文件
        if os.path.exists(save_path):
            files = os.listdir(save_path)
            logger.info(f"保存的文件: {files}")
        
        # 创建新的学习系统实例
        logger.info("创建新的学习系统实例...")
        new_learning_system = V12OnlineLearningSystem(config)
        
        # 加载模型
        logger.info("加载模型...")
        new_learning_system.load_models(save_path)
        
        # 检查加载的模型
        logger.info("检查加载的模型:")
        for model_name in ['ofi_expert', 'lstm', 'transformer', 'cnn']:
            model = new_learning_system.get_model(model_name)
            if model:
                logger.info(f"  模型 {model_name}: 已加载")
            else:
                logger.info(f"  模型 {model_name}: 未找到")
        
        new_learning_system.stop()
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def test_learning_efficiency():
    """测试学习效率"""
    logger.info("=" * 80)
    logger.info("测试5: 学习效率")
    logger.info("=" * 80)
    
    config = create_test_config()
    config['learning_interval'] = 2  # 更短的学习间隔
    config['min_samples_for_update'] = 15  # 更小的最小样本数
    
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)
        
        # 持续添加数据并监控学习效率
        logger.info("开始学习效率测试...")
        
        start_time = time.time()
        total_samples = 0
        
        for cycle in range(10):
            # 添加一批数据
            batch_data = generate_mock_training_data(25)
            for data in batch_data:
                learning_system.add_training_data(data)
                total_samples += 1
            
            # 等待学习周期
            time.sleep(3)
            
            # 获取性能指标
            summary = learning_system.get_performance_summary()
            metrics = learning_system.get_learning_metrics()
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"周期 {cycle+1}:")
            logger.info(f"  处理样本数: {total_samples}")
            logger.info(f"  学习周期数: {summary['learning_cycles_completed']}")
            logger.info(f"  模型更新数: {summary['model_updates']}")
            logger.info(f"  准确度提升: {summary['accuracy_improvements']}")
            logger.info(f"  学习效率: {summary['learning_efficiency']:.4f}")
            logger.info(f"  平均准确度: {metrics.average_accuracy:.4f}")
            logger.info(f"  性能趋势: {metrics.performance_trend}")
            logger.info(f"  样本处理速度: {total_samples/elapsed_time:.2f} 样本/秒")
            logger.info("")
        
        # 最终效率分析
        final_summary = learning_system.get_performance_summary()
        final_metrics = learning_system.get_learning_metrics()
        
        logger.info("学习效率最终分析:")
        logger.info(f"  总处理时间: {time.time() - start_time:.2f}秒")
        logger.info(f"  总处理样本: {total_samples}")
        logger.info(f"  平均处理速度: {total_samples/(time.time() - start_time):.2f} 样本/秒")
        logger.info(f"  学习效率: {final_summary['learning_efficiency']:.4f}")
        logger.info(f"  最终准确度: {final_metrics.average_accuracy:.4f}")
        logger.info(f"  最佳准确度: {final_metrics.best_accuracy:.4f}")
        logger.info(f"  收敛状态: {final_metrics.performance_trend}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def test_error_handling():
    """测试错误处理"""
    logger.info("=" * 80)
    logger.info("测试6: 错误处理")
    logger.info("=" * 80)
    
    config = create_test_config()
    learning_system = V12OnlineLearningSystem(config)
    
    try:
        # 启动学习系统
        learning_system.start()
        time.sleep(0.5)
        
        # 测试无效数据
        logger.info("测试无效数据处理...")
        invalid_data_list = [
            {},  # 空数据
            {'invalid_key': 'invalid_value'},  # 无效键
            None,  # None数据
            {'close': 0, 'future_close': 0},  # 零价格数据
        ]
        
        for invalid_data in invalid_data_list:
            learning_system.add_training_data(invalid_data)
        
        # 测试正常数据
        logger.info("添加正常数据...")
        normal_data = generate_mock_training_data(50)
        for data in normal_data:
            learning_system.add_training_data(data)
        
        time.sleep(8)  # 等待学习周期
        
        # 检查系统是否正常运行
        summary = learning_system.get_performance_summary()
        logger.info(f"系统正常运行: 处理样本数={summary['total_samples_processed']}")
        logger.info(f"学习周期完成: {summary['learning_cycles_completed']}")
        
        # 测试模型获取
        logger.info("测试模型获取...")
        for model_name in ['ofi_expert', 'lstm', 'transformer', 'cnn']:
            model = learning_system.get_model(model_name)
            if model:
                logger.info(f"模型 {model_name}: 获取成功")
            else:
                logger.info(f"模型 {model_name}: 未找到")
        
        # 测试性能获取
        logger.info("测试性能获取...")
        for model_name in ['ofi_expert', 'lstm', 'transformer', 'cnn']:
            performance = learning_system.get_model_performance(model_name)
            if performance:
                logger.info(f"性能 {model_name}: 获取成功")
            else:
                logger.info(f"性能 {model_name}: 未找到")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    finally:
        learning_system.stop()

def main():
    """主测试函数"""
    logger.info("=" * 80)
    logger.info("V12在线学习系统测试开始")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 运行所有测试
        test_basic_learning_functionality()
        time.sleep(2)
        
        test_incremental_learning()
        time.sleep(2)
        
        test_model_performance_monitoring()
        time.sleep(2)
        
        test_model_persistence()
        time.sleep(2)
        
        test_learning_efficiency()
        time.sleep(2)
        
        test_error_handling()
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"V12在线学习系统测试完成，总耗时: {total_time:.2f}秒")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
