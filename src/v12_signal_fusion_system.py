"""
V12 信号融合系统
智能融合OFI信号与AI信号，实现高频交易决策
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import deque
import threading
import time

# 导入V12组件
try:
    from .v12_real_ofi_calculator import V12RealOFICalculator
    from .v12_ensemble_ai_model import V12EnsembleAIModel
except ImportError:
    from v12_real_ofi_calculator import V12RealOFICalculator
    from v12_ensemble_ai_model import V12EnsembleAIModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12SignalFusionSystem:
    """
    V12 信号融合系统
    智能融合OFI信号与AI信号
    """
    
    def __init__(self, config: Dict):
        """
        初始化信号融合系统
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 核心组件
        self.ofi_calculator = V12RealOFICalculator(
            levels=config.get('features', {}).get('ofi_levels', 5),
            window_seconds=config.get('features', {}).get('ofi_window_seconds', 2),
            z_window=config.get('features', {}).get('z_window', 1200)
        )
        
        self.ai_model = V12EnsembleAIModel(config)
        
        # 信号融合参数
        self.fusion_params = config.get('signals', {}).get('ai_enhanced', {})
        self.real_time_params = config.get('signals', {}).get('real_time_optimization', {})
        
        # 高频交易参数
        self.hf_params = config.get('high_frequency', {})
        self.max_daily_trades = self.hf_params.get('max_daily_trades', 200)
        self.min_trade_interval = self.hf_params.get('min_trade_interval', 10)
        
        # 信号历史
        self.signal_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=self.max_daily_trades)
        
        # 实时优化状态
        self.optimization_state = {
            'performance_history': [],
            'current_thresholds': {
                'ofi_z_min': self.fusion_params.get('ofi_z_min', 1.4),
                'ai_prediction_min': self.fusion_params.get('min_ai_prediction', 0.7),
                'signal_strength_min': self.fusion_params.get('min_signal_strength', 1.8)
            },
            'adaptation_rate': self.real_time_params.get('adaptation_rate', 0.1),
            'update_counter': 0,
            'last_optimization': datetime.now()
        }
        
        # 线程安全锁
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'signals_generated': 0,
            'signals_filtered': 0,
            'trades_executed': 0,
            'daily_trades': 0,
            'last_reset_date': datetime.now().date(),
            'win_rate': 0.0,
            'avg_signal_strength': 0.0,
            'avg_ai_confidence': 0.0
        }
        
        logger.info("V12信号融合系统初始化完成")
        logger.info(f"高频交易配置: 最大日交易数={self.max_daily_trades}, 最小间隔={self.min_trade_interval}ms")
    
    def process_market_data(self, order_book_data: Dict) -> Dict:
        """
        处理市场数据并生成融合信号
        
        Args:
            order_book_data: 订单簿数据
            
        Returns:
            融合信号结果
        """
        try:
            with self.data_lock:
                # 更新OFI计算
                ofi_result = self.ofi_calculator.update_order_book(order_book_data)
                
                if not ofi_result:
                    return {}
                
                # 准备AI预测数据
                ai_data = self._prepare_ai_data(order_book_data, ofi_result)
                
                # AI模型预测
                ai_prediction = self.ai_model.predict_ensemble(ai_data)
                
                # 生成融合信号
                fusion_signal = self._generate_fusion_signal(ofi_result, ai_prediction)
                
                # 应用高频过滤
                if self._should_execute_trade(fusion_signal):
                    fusion_signal['execute_trade'] = True
                    self._record_trade(fusion_signal)
                else:
                    fusion_signal['execute_trade'] = False
                
                # 记录信号历史
                self._record_signal(fusion_signal)
                
                # 实时优化
                self._update_real_time_optimization()
                
                # 更新统计信息
                self._update_statistics(fusion_signal)
                
                return fusion_signal
                
        except Exception as e:
            logger.error(f"处理市场数据失败: {e}")
            return {}
    
    def _prepare_ai_data(self, order_book_data: Dict, ofi_result: Dict) -> pd.DataFrame:
        """
        准备AI预测数据
        
        Args:
            order_book_data: 订单簿数据
            ofi_result: OFI计算结果
            
        Returns:
            AI预测数据框
        """
        try:
            # 合并数据
            combined_data = {
                'timestamp': order_book_data.get('timestamp', datetime.now()),
                'price': order_book_data.get('mid_price', 0.0),
                'bid1': order_book_data.get('bid1_price', 0.0),
                'ask1': order_book_data.get('ask1_price', 0.0),
                'bid1_size': order_book_data.get('bid1_size', 0.0),
                'ask1_size': order_book_data.get('ask1_size', 0.0),
                'size': order_book_data.get('size', 0.0),
                'ofi': ofi_result.get('ofi', 0.0),
                'cvd': ofi_result.get('cvd', 0.0),
                'ofi_zscore': ofi_result.get('ofi_zscore', 0.0),
                'cvd_zscore': ofi_result.get('cvd_zscore', 0.0),
                'real_ofi_z': ofi_result.get('ofi_zscore', 0.0),
                'real_cvd_z': ofi_result.get('cvd_zscore', 0.0),
                'signal_strength': ofi_result.get('signal_strength', 0.0),
                'quality_score': ofi_result.get('quality_score', 0.0),
                'ret_1s': 0.0,  # 需要计算
                'atr': 1.0,     # 需要计算
                'vwap': order_book_data.get('mid_price', 0.0)
            }
            
            # 转换为DataFrame
            df = pd.DataFrame([combined_data])
            
            # 计算缺失的技术指标
            df['ret_1s'] = df['price'].pct_change().fillna(0.0)
            df['atr'] = df['ret_1s'].rolling(14).std().fillna(1.0)
            
            return df
            
        except Exception as e:
            logger.error(f"准备AI数据失败: {e}")
            return pd.DataFrame()
    
    def _generate_fusion_signal(self, ofi_result: Dict, ai_prediction: pd.Series) -> Dict:
        """
        生成融合信号
        
        Args:
            ofi_result: OFI计算结果
            ai_prediction: AI预测结果
            
        Returns:
            融合信号字典
        """
        try:
            # 获取当前阈值
            thresholds = self.optimization_state['current_thresholds']
            
            # OFI信号检查
            ofi_z = ofi_result.get('ofi_zscore', 0.0)
            cvd_z = ofi_result.get('cvd_zscore', 0.0)
            signal_strength = ofi_result.get('signal_strength', 0.0)
            
            # AI信号检查
            ai_confidence = ai_prediction.iloc[0] if len(ai_prediction) > 0 else 0.5
            
            # 信号强度检查
            strong_signal = signal_strength >= thresholds['signal_strength_min']
            
            # OFI信号检查
            ofi_signal_long = ofi_z >= thresholds['ofi_z_min']
            ofi_signal_short = ofi_z <= -thresholds['ofi_z_min']
            
            # AI增强检查
            ai_enhanced = ai_confidence >= thresholds['ai_prediction_min']
            
            # 高频信号检查
            high_freq_signal = signal_strength >= self.fusion_params.get('high_freq_threshold', 1.2)
            
            # 组合信号
            long_signal = ofi_signal_long and strong_signal and ai_enhanced and high_freq_signal
            short_signal = ofi_signal_short and strong_signal and ai_enhanced and high_freq_signal
            
            # 构建融合信号
            fusion_signal = {
                'timestamp': datetime.now(),
                'signal_type': None,
                'signal_side': 0,
                'signal_strength': signal_strength,
                'quality_score': ofi_result.get('quality_score', 0.0),
                'ai_confidence': ai_confidence,
                'fusion_score': self._calculate_fusion_score(signal_strength, ai_confidence),
                'ofi_zscore': ofi_z,
                'cvd_zscore': cvd_z,
                'ofi': ofi_result.get('ofi', 0.0),
                'cvd': ofi_result.get('cvd', 0.0),
                'mid_price': ofi_result.get('mid_price', 0.0),
                'spread_bps': ofi_result.get('spread_bps', 0.0),
                'execute_trade': False,
                'trade_reason': 'none'
            }
            
            if long_signal:
                fusion_signal['signal_type'] = 'v12_ofi_ai_long'
                fusion_signal['signal_side'] = 1
                fusion_signal['trade_reason'] = 'ofi_ai_fusion_long'
            elif short_signal:
                fusion_signal['signal_type'] = 'v12_ofi_ai_short'
                fusion_signal['signal_side'] = -1
                fusion_signal['trade_reason'] = 'ofi_ai_fusion_short'
            
            return fusion_signal
            
        except Exception as e:
            logger.error(f"生成融合信号失败: {e}")
            return {}
    
    def _calculate_fusion_score(self, signal_strength: float, ai_confidence: float) -> float:
        """
        计算融合评分
        
        Args:
            signal_strength: 信号强度
            ai_confidence: AI置信度
            
        Returns:
            融合评分
        """
        try:
            # 归一化信号强度
            normalized_strength = min(signal_strength / 3.0, 1.0)
            
            # 融合评分
            fusion_score = (normalized_strength * 0.6 + ai_confidence * 0.4)
            
            return min(fusion_score, 1.0)
            
        except Exception as e:
            logger.error(f"计算融合评分失败: {e}")
            return 0.0
    
    def _should_execute_trade(self, fusion_signal: Dict) -> bool:
        """
        判断是否应该执行交易
        
        Args:
            fusion_signal: 融合信号
            
        Returns:
            是否执行交易
        """
        try:
            # 检查信号有效性
            if fusion_signal.get('signal_side', 0) == 0:
                return False
            
            # 检查日交易数限制
            current_date = datetime.now().date()
            if current_date != self.stats['last_reset_date']:
                self.stats['daily_trades'] = 0
                self.stats['last_reset_date'] = current_date
            
            if self.stats['daily_trades'] >= self.max_daily_trades:
                logger.warning(f"达到日交易数限制: {self.max_daily_trades}")
                return False
            
            # 检查交易间隔
            if len(self.trade_history) > 0:
                last_trade_time = self.trade_history[-1].get('timestamp', datetime.min)
                time_diff = (datetime.now() - last_trade_time).total_seconds() * 1000  # 转换为毫秒
                
                if time_diff < self.min_trade_interval:
                    return False
            
            # 检查融合评分
            fusion_score = fusion_signal.get('fusion_score', 0.0)
            if fusion_score < 0.6:  # 融合评分阈值
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"判断交易执行失败: {e}")
            return False
    
    def _record_trade(self, fusion_signal: Dict):
        """记录交易"""
        try:
            trade_record = {
                'timestamp': fusion_signal['timestamp'],
                'signal_type': fusion_signal['signal_type'],
                'signal_side': fusion_signal['signal_side'],
                'fusion_score': fusion_signal['fusion_score'],
                'ai_confidence': fusion_signal['ai_confidence'],
                'signal_strength': fusion_signal['signal_strength'],
                'price': fusion_signal['mid_price'],
                'trade_reason': fusion_signal['trade_reason']
            }
            
            self.trade_history.append(trade_record)
            self.stats['trades_executed'] += 1
            self.stats['daily_trades'] += 1
            
        except Exception as e:
            logger.error(f"记录交易失败: {e}")
    
    def _record_signal(self, fusion_signal: Dict):
        """记录信号"""
        try:
            self.signal_history.append(fusion_signal.copy())
            self.stats['signals_generated'] += 1
            
        except Exception as e:
            logger.error(f"记录信号失败: {e}")
    
    def _update_real_time_optimization(self):
        """更新实时优化"""
        try:
            self.optimization_state['update_counter'] += 1
            
            update_frequency = self.real_time_params.get('update_frequency', 10)
            
            if self.optimization_state['update_counter'] >= update_frequency:
                # 重置计数器
                self.optimization_state['update_counter'] = 0
                
                # 执行优化逻辑
                self._optimize_thresholds()
                
                # 更新最后优化时间
                self.optimization_state['last_optimization'] = datetime.now()
                
        except Exception as e:
            logger.error(f"更新实时优化失败: {e}")
    
    def _optimize_thresholds(self):
        """优化阈值"""
        try:
            # 基于性能历史优化阈值
            if len(self.optimization_state['performance_history']) < 10:
                return
            
            # 获取最近的性能数据
            recent_performance = self.optimization_state['performance_history'][-10:]
            avg_performance = np.mean([p['performance'] for p in recent_performance])
            
            # 如果性能低于阈值，调整参数
            min_performance = self.real_time_params.get('min_performance_threshold', 0.6)
            
            if avg_performance < min_performance:
                adaptation_rate = self.optimization_state['adaptation_rate']
                
                # 调整阈值
                current_thresholds = self.optimization_state['current_thresholds']
                
                # 降低阈值以增加信号频率
                current_thresholds['ofi_z_min'] *= (1 - adaptation_rate)
                current_thresholds['ai_prediction_min'] *= (1 - adaptation_rate)
                current_thresholds['signal_strength_min'] *= (1 - adaptation_rate)
                
                logger.info(f"实时优化调整阈值: {current_thresholds}")
                
        except Exception as e:
            logger.error(f"优化阈值失败: {e}")
    
    def _update_statistics(self, fusion_signal: Dict):
        """更新统计信息"""
        try:
            # 更新平均信号强度
            if fusion_signal.get('signal_strength'):
                self.stats['avg_signal_strength'] = (
                    self.stats['avg_signal_strength'] * 0.9 + 
                    fusion_signal['signal_strength'] * 0.1
                )
            
            # 更新平均AI置信度
            if fusion_signal.get('ai_confidence'):
                self.stats['avg_ai_confidence'] = (
                    self.stats['avg_ai_confidence'] * 0.9 + 
                    fusion_signal['ai_confidence'] * 0.1
                )
            
            # 更新胜率
            if len(self.trade_history) > 0:
                # 这里需要实际的交易结果来计算胜率
                # 简化处理，使用融合评分作为代理
                recent_trades = list(self.trade_history)[-10:]
                if recent_trades:
                    avg_fusion_score = np.mean([t['fusion_score'] for t in recent_trades])
                    self.stats['win_rate'] = avg_fusion_score
            
        except Exception as e:
            logger.error(f"更新统计信息失败: {e}")
    
    def get_current_signals(self) -> Dict:
        """获取当前信号状态"""
        try:
            current_values = self.ofi_calculator.get_current_values()
            
            return {
                'timestamp': datetime.now(),
                'ofi_zscore': current_values.get('ofi_zscore', 0.0),
                'cvd_zscore': current_values.get('cvd_zscore', 0.0),
                'signal_strength': current_values.get('signal_strength', 0.0),
                'quality_score': current_values.get('quality_score', 0.0),
                'current_thresholds': self.optimization_state['current_thresholds'],
                'daily_trades': self.stats['daily_trades'],
                'max_daily_trades': self.max_daily_trades
            }
            
        except Exception as e:
            logger.error(f"获取当前信号状态失败: {e}")
            return {}
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        try:
            return {
                'signals_generated': self.stats['signals_generated'],
                'trades_executed': self.stats['trades_executed'],
                'daily_trades': self.stats['daily_trades'],
                'max_daily_trades': self.max_daily_trades,
                'win_rate': self.stats['win_rate'],
                'avg_signal_strength': self.stats['avg_signal_strength'],
                'avg_ai_confidence': self.stats['avg_ai_confidence'],
                'signal_history_count': len(self.signal_history),
                'trade_history_count': len(self.trade_history),
                'last_reset_date': self.stats['last_reset_date']
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def reset_daily_counters(self):
        """重置日计数器"""
        try:
            self.stats['daily_trades'] = 0
            self.stats['last_reset_date'] = datetime.now().date()
            logger.info("日计数器已重置")
            
        except Exception as e:
            logger.error(f"重置日计数器失败: {e}")


def test_v12_signal_fusion_system():
    """测试V12信号融合系统"""
    logger.info("开始测试V12信号融合系统...")
    
    # 配置参数
    config = {
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
    
    # 创建信号融合系统
    fusion_system = V12SignalFusionSystem(config)
    
    # 模拟订单簿数据
    def create_mock_order_book(timestamp, price_base=3000.0):
        return {
            'timestamp': timestamp,
            'bid1_price': price_base - 0.5,
            'ask1_price': price_base + 0.5,
            'bid1_size': 100.0,
            'ask1_size': 100.0,
            'size': 50.0,
            'mid_price': price_base,
            'spread_bps': 1.0
        }
    
    # 模拟数据流
    base_price = 3000.0
    trade_signals = 0
    
    for i in range(100):
        timestamp = datetime.now() + timedelta(seconds=i)
        
        # 模拟价格变化
        price_change = np.random.randn() * 0.1
        base_price += price_change
        
        # 创建订单簿数据
        order_book = create_mock_order_book(timestamp, base_price)
        
        # 处理市场数据
        fusion_signal = fusion_system.process_market_data(order_book)
        
        if fusion_signal and fusion_signal.get('execute_trade'):
            trade_signals += 1
            logger.info(f"交易信号 {trade_signals}: {fusion_signal['signal_type']}, "
                       f"方向: {fusion_signal['signal_side']}, "
                       f"融合评分: {fusion_signal['fusion_score']:.4f}")
        
        time.sleep(0.01)  # 模拟10ms间隔
    
    # 获取统计信息
    stats = fusion_system.get_statistics()
    logger.info(f"信号融合系统统计: {stats}")
    
    # 获取当前信号状态
    current_signals = fusion_system.get_current_signals()
    logger.info(f"当前信号状态: {current_signals}")
    
    logger.info("V12信号融合系统测试完成")


if __name__ == "__main__":
    test_v12_signal_fusion_system()
