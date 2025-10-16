"""
V12 OFI+AI融合策略完整测试
测试WebSocket数据收集 + OFI计算 + AI模型 + 信号融合
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# 导入V12组件
from src.v12_binance_websocket_collector import V12BinanceWebSocketCollector
from src.v12_real_ofi_calculator import V12RealOFICalculator
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V12CompleteTestSystem:
    """
    V12完整测试系统
    """
    
    def __init__(self, config_file="config/params_v12_ofi_ai.yaml"):
        """
        初始化V12完整测试系统
        
        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 初始化组件
        self.websocket_collector = None
        self.ofi_calculator = None
        self.ai_model = None
        self.signal_fusion_system = None
        
        # 测试数据
        self.test_data = []
        self.signal_results = []
        
        # 统计信息
        self.test_stats = {
            'data_points_processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'test_duration': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("V12完整测试系统初始化完成")
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'features': {
                'ofi_levels': 5,
                'ofi_window_seconds': 2,
                'z_window': 1200
            },
            'signals': {
                'ai_enhanced': {
                    'ofi_z_min': 1.4,
                    'min_ai_prediction': 0.7,
                    'min_signal_strength': 1.8,
                    'high_freq_threshold': 1.2
                },
                'real_time_optimization': {
                    'update_frequency': 10,
                    'adaptation_rate': 0.1,
                    'min_performance_threshold': 0.6
                }
            },
            'high_frequency': {
                'max_daily_trades': 200,
                'min_trade_interval': 10
            },
            'ofi_ai_fusion': {
                'ai_models': {
                    'v9_ml_weight': 0.5,
                    'lstm_weight': 0.2,
                    'transformer_weight': 0.2,
                    'cnn_weight': 0.1
                }
            }
        }
    
    def initialize_components(self):
        """初始化所有组件"""
        try:
            logger.info("初始化V12组件...")
            
            # 1. WebSocket收集器
            self.websocket_collector = V12BinanceWebSocketCollector(
                symbol="ETHUSDT",
                depth_levels=self.config['features']['ofi_levels']
            )
            
            # 2. OFI计算器
            self.ofi_calculator = V12RealOFICalculator(
                levels=self.config['features']['ofi_levels'],
                window_seconds=self.config['features']['ofi_window_seconds'],
                z_window=self.config['features']['z_window']
            )
            
            # 3. AI模型
            self.ai_model = V12EnsembleAIModel(self.config)
            
            # 4. 信号融合系统
            self.signal_fusion_system = V12SignalFusionSystem(self.config)
            
            logger.info("V12组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise
    
    def create_mock_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        创建模拟训练数据
        
        Args:
            n_samples: 样本数量
            
        Returns:
            训练数据框
        """
        try:
            logger.info(f"创建模拟训练数据: {n_samples} 样本")
            
            np.random.seed(42)
            
            # 生成时间序列
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
            
            # 生成价格数据
            price_base = 3000.0
            price_changes = np.random.randn(n_samples) * 0.1
            prices = price_base + np.cumsum(price_changes)
            
            # 生成订单簿数据
            mock_data = {
                'timestamp': timestamps,
                'price': prices,
                'bid1': prices - 0.5 + np.random.randn(n_samples) * 0.1,
                'ask1': prices + 0.5 + np.random.randn(n_samples) * 0.1,
                'bid1_size': 100 + np.random.randn(n_samples) * 10,
                'ask1_size': 100 + np.random.randn(n_samples) * 10,
                'size': 50 + np.random.randn(n_samples) * 5,
                'ofi_z': np.random.randn(n_samples) * 2,
                'cvd_z': np.random.randn(n_samples) * 2,
                'ret_1s': np.random.randn(n_samples) * 0.001,
                'atr': 1.0 + np.random.randn(n_samples) * 0.1,
                'vwap': prices + np.random.randn(n_samples) * 0.05,
                'signal_quality': np.random.rand(n_samples)
            }
            
            # 添加更多档位数据
            for level in range(2, 6):
                mock_data[f'bid{level}'] = prices - 0.5 - (level-1) * 0.1 + np.random.randn(n_samples) * 0.1
                mock_data[f'ask{level}'] = prices + 0.5 + (level-1) * 0.1 + np.random.randn(n_samples) * 0.1
                mock_data[f'bid{level}_size'] = 80 / level + np.random.randn(n_samples) * 5
                mock_data[f'ask{level}_size'] = 80 / level + np.random.randn(n_samples) * 5
            
            df = pd.DataFrame(mock_data)
            
            logger.info(f"模拟训练数据创建完成: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            logger.error(f"创建模拟训练数据失败: {e}")
            return pd.DataFrame()
    
    def train_ai_models(self, training_data: pd.DataFrame):
        """
        训练AI模型
        
        Args:
            training_data: 训练数据
        """
        try:
            logger.info("开始训练AI模型...")
            
            # 训练集成AI模型
            v12_params = {
                'signals': {
                    'momentum': {
                        'ofi_z_min': self.config['signals']['ai_enhanced']['ofi_z_min'],
                        'cvd_z_min': 0.6,
                        'min_signal_strength': self.config['signals']['ai_enhanced']['min_signal_strength']
                    }
                }
            }
            
            self.ai_model.train_ensemble_model(training_data, v12_params)
            
            logger.info("AI模型训练完成")
            
        except Exception as e:
            logger.error(f"训练AI模型失败: {e}")
    
    def simulate_real_time_processing(self, duration_seconds: int = 60):
        """
        模拟实时数据处理
        
        Args:
            duration_seconds: 测试持续时间（秒）
        """
        try:
            logger.info(f"开始模拟实时数据处理: {duration_seconds} 秒")
            
            self.test_stats['start_time'] = datetime.now()
            
            # 模拟订单簿数据生成
            base_price = 3000.0
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration_seconds:
                # 生成模拟订单簿数据
                order_book_data = self._create_mock_order_book_data(base_price)
                
                # 处理数据
                fusion_signal = self.signal_fusion_system.process_market_data(order_book_data)
                
                # 记录结果
                if fusion_signal:
                    self.test_data.append(order_book_data)
                    self.signal_results.append(fusion_signal)
                    
                    if fusion_signal.get('execute_trade'):
                        self.test_stats['trades_executed'] += 1
                        logger.info(f"交易信号: {fusion_signal['signal_type']}, "
                                   f"方向: {fusion_signal['signal_side']}, "
                                   f"融合评分: {fusion_signal['fusion_score']:.4f}")
                
                self.test_stats['data_points_processed'] += 1
                self.test_stats['signals_generated'] = len(self.signal_results)
                
                # 模拟价格变化
                base_price += np.random.randn() * 0.1
                
                # 控制处理频率 (模拟10ms间隔)
                time.sleep(0.01)
            
            self.test_stats['end_time'] = datetime.now()
            self.test_stats['test_duration'] = (self.test_stats['end_time'] - self.test_stats['start_time']).total_seconds()
            
            logger.info("实时数据处理模拟完成")
            
        except Exception as e:
            logger.error(f"模拟实时数据处理失败: {e}")
    
    def _create_mock_order_book_data(self, price_base: float) -> Dict:
        """创建模拟订单簿数据"""
        timestamp = datetime.now()
        
        # 生成价格变化
        price_change = np.random.randn() * 0.1
        current_price = price_base + price_change
        
        # 创建5档订单簿数据
        order_book_data = {
            'timestamp': timestamp,
            'mid_price': current_price,
            'spread_bps': 1.0 + np.random.randn() * 0.1
        }
        
        # 生成5档数据
        for level in range(1, 6):
            level_spread = level * 0.1
            order_book_data[f'bid{level}_price'] = current_price - level_spread + np.random.randn() * 0.05
            order_book_data[f'ask{level}_price'] = current_price + level_spread + np.random.randn() * 0.05
            order_book_data[f'bid{level}_size'] = 100 / level + np.random.randn() * 5
            order_book_data[f'ask{level}_size'] = 100 / level + np.random.randn() * 5
        
        return order_book_data
    
    def generate_test_report(self):
        """生成测试报告"""
        try:
            logger.info("生成V12测试报告...")
            
            # 计算性能指标
            duration = self.test_stats['test_duration']
            data_rate = self.test_stats['data_points_processed'] / duration if duration > 0 else 0
            signal_rate = self.test_stats['signals_generated'] / duration if duration > 0 else 0
            trade_rate = self.test_stats['trades_executed'] / duration if duration > 0 else 0
            
            # 信号分析
            if self.signal_results:
                signal_types = [s.get('signal_type', 'none') for s in self.signal_results]
                signal_sides = [s.get('signal_side', 0) for s in self.signal_results]
                fusion_scores = [s.get('fusion_score', 0.0) for s in self.signal_results]
                ai_confidences = [s.get('ai_confidence', 0.0) for s in self.signal_results]
                
                avg_fusion_score = np.mean(fusion_scores)
                avg_ai_confidence = np.mean(ai_confidences)
                long_signals = sum(1 for side in signal_sides if side > 0)
                short_signals = sum(1 for side in signal_sides if side < 0)
            else:
                avg_fusion_score = 0.0
                avg_ai_confidence = 0.0
                long_signals = 0
                short_signals = 0
            
            # 生成报告
            report = {
                'test_summary': {
                    'test_duration_seconds': duration,
                    'data_points_processed': self.test_stats['data_points_processed'],
                    'signals_generated': self.test_stats['signals_generated'],
                    'trades_executed': self.test_stats['trades_executed'],
                    'data_processing_rate': data_rate,
                    'signal_generation_rate': signal_rate,
                    'trade_execution_rate': trade_rate
                },
                'signal_analysis': {
                    'total_signals': len(self.signal_results),
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'avg_fusion_score': avg_fusion_score,
                    'avg_ai_confidence': avg_ai_confidence,
                    'high_quality_signals': sum(1 for s in fusion_scores if s > 0.7)
                },
                'component_performance': {
                    'ofi_calculator_stats': self.ofi_calculator.get_statistics() if self.ofi_calculator else {},
                    'ai_model_stats': self.ai_model.get_statistics() if self.ai_model else {},
                    'signal_fusion_stats': self.signal_fusion_system.get_statistics() if self.signal_fusion_system else {}
                },
                'performance_targets': {
                    'target_daily_trades': 100,
                    'target_win_rate': 0.65,
                    'target_data_rate': 100,  # 100 points/second
                    'achieved_data_rate': data_rate,
                    'achieved_trade_rate': trade_rate * 86400  # 转换为日交易数
                }
            }
            
            # 打印报告
            self._print_test_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
            return {}
    
    def _print_test_report(self, report: Dict):
        """打印测试报告"""
        logger.info("=" * 80)
        logger.info("V12 OFI+AI融合策略完整测试报告")
        logger.info("=" * 80)
        
        # 测试摘要
        summary = report['test_summary']
        logger.info(f"测试持续时间: {summary['test_duration_seconds']:.2f} 秒")
        logger.info(f"数据处理点数: {summary['data_points_processed']}")
        logger.info(f"生成信号数: {summary['signals_generated']}")
        logger.info(f"执行交易数: {summary['trades_executed']}")
        logger.info(f"数据处理速率: {summary['data_processing_rate']:.2f} 点/秒")
        logger.info(f"信号生成速率: {summary['signal_generation_rate']:.2f} 信号/秒")
        logger.info(f"交易执行速率: {summary['trade_execution_rate']:.2f} 交易/秒")
        
        # 信号分析
        signal_analysis = report['signal_analysis']
        logger.info(f"总信号数: {signal_analysis['total_signals']}")
        logger.info(f"多头信号: {signal_analysis['long_signals']}")
        logger.info(f"空头信号: {signal_analysis['short_signals']}")
        logger.info(f"平均融合评分: {signal_analysis['avg_fusion_score']:.4f}")
        logger.info(f"平均AI置信度: {signal_analysis['avg_ai_confidence']:.4f}")
        logger.info(f"高质量信号数: {signal_analysis['high_quality_signals']}")
        
        # 性能目标
        targets = report['performance_targets']
        logger.info(f"目标日交易数: {targets['target_daily_trades']}")
        logger.info(f"实际日交易数: {targets['achieved_trade_rate']:.2f}")
        logger.info(f"目标数据处理速率: {targets['target_data_rate']}")
        logger.info(f"实际数据处理速率: {targets['achieved_data_rate']:.2f}")
        
        logger.info("=" * 80)


def main():
    """主函数"""
    try:
        logger.info("开始V12 OFI+AI融合策略完整测试")
        
        # 创建测试系统
        test_system = V12CompleteTestSystem()
        
        # 初始化组件
        test_system.initialize_components()
        
        # 创建训练数据
        training_data = test_system.create_mock_training_data(1000)
        
        # 训练AI模型
        test_system.train_ai_models(training_data)
        
        # 模拟实时数据处理
        test_system.simulate_real_time_processing(30)  # 30秒测试
        
        # 生成测试报告
        report = test_system.generate_test_report()
        
        logger.info("V12 OFI+AI融合策略完整测试完成")
        
        return report
        
    except Exception as e:
        logger.error(f"V12完整测试失败: {e}")
        return None


if __name__ == "__main__":
    main()
