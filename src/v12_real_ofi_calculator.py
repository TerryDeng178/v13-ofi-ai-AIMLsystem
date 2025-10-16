"""
V12 真实OFI计算引擎
基于V9 OFI策略参数，实现真实的订单流不平衡计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import deque
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12RealOFICalculator:
    """
    V12 真实OFI计算引擎
    基于V9 OFI策略参数设计
    """
    
    def __init__(self, levels=5, window_seconds=2, z_window=1200):
        """
        初始化OFI计算引擎
        
        Args:
            levels: 订单簿深度级别 (基于V9: 5档)
            window_seconds: OFI滚动窗口 (基于V9: 2秒)
            z_window: Z-score计算窗口 (基于V9: 1200秒/20分钟)
        """
        self.levels = levels
        self.window_seconds = window_seconds
        self.z_window = z_window
        
        # V9权重配置
        self.weights = [1.0, 0.5, 0.33, 0.25, 0.2]  # 5档权重
        
        # 历史数据存储
        self.order_book_history = deque(maxlen=10000)
        self.ofi_history = deque(maxlen=7200)  # 2小时历史
        self.cvd_history = deque(maxlen=7200)  # 2小时历史
        
        # 实时计算状态
        self.last_order_book = None
        self.current_ofi = 0.0
        self.current_cvd = 0.0
        self.ofi_zscore = 0.0
        self.cvd_zscore = 0.0
        
        # 线程安全锁
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'ofis_calculated': 0,
            'cvds_calculated': 0,
            'z_scores_calculated': 0,
            'last_update': None
        }
        
        logger.info(f"V12真实OFI计算引擎初始化完成 - 档位: {self.levels}, 窗口: {self.window_seconds}s, Z窗口: {self.z_window}s")
    
    def update_order_book(self, order_book_data: Dict) -> Dict:
        """
        更新订单簿数据并计算OFI
        
        Args:
            order_book_data: 订单簿数据字典
            
        Returns:
            包含OFI计算结果的数据字典
        """
        with self.data_lock:
            try:
                # 存储历史数据
                self.order_book_history.append(order_book_data.copy())
                
                # 计算OFI
                ofi_value = self._calculate_ofi(order_book_data)
                
                # 计算CVD
                cvd_value = self._calculate_cvd(order_book_data)
                
                # 更新当前值
                self.current_ofi = ofi_value
                self.current_cvd = cvd_value
                
                # 存储历史OFI和CVD
                ofi_record = {
                    'timestamp': order_book_data['timestamp'],
                    'ofi': ofi_value,
                    'mid_price': order_book_data.get('mid_price', 0.0),
                    'spread_bps': order_book_data.get('spread_bps', 0.0)
                }
                self.ofi_history.append(ofi_record)
                
                cvd_record = {
                    'timestamp': order_book_data['timestamp'],
                    'cvd': cvd_value,
                    'mid_price': order_book_data.get('mid_price', 0.0)
                }
                self.cvd_history.append(cvd_record)
                
                # 计算Z-score
                self.ofi_zscore = self._calculate_ofi_zscore()
                self.cvd_zscore = self._calculate_cvd_zscore()
                
                # 更新统计信息
                self.stats['ofis_calculated'] += 1
                self.stats['cvds_calculated'] += 1
                self.stats['z_scores_calculated'] += 1
                self.stats['last_update'] = datetime.now()
                
                # 构建返回数据
                result = {
                    'timestamp': order_book_data['timestamp'],
                    'ofi': ofi_value,
                    'cvd': cvd_value,
                    'ofi_zscore': self.ofi_zscore,
                    'cvd_zscore': self.cvd_zscore,
                    'mid_price': order_book_data.get('mid_price', 0.0),
                    'spread_bps': order_book_data.get('spread_bps', 0.0),
                    'signal_strength': self._calculate_signal_strength(),
                    'quality_score': self._calculate_quality_score()
                }
                
                # 更新最后订单簿数据
                self.last_order_book = order_book_data.copy()
                
                return result
                
            except Exception as e:
                logger.error(f"更新订单簿数据失败: {e}")
                return {}
    
    def _calculate_ofi(self, current_order_book: Dict) -> float:
        """
        计算真实OFI值 (基于V9算法)
        
        Args:
            current_order_book: 当前订单簿数据
            
        Returns:
            OFI值
        """
        try:
            if self.last_order_book is None:
                return 0.0
            
            ofi_total = 0.0
            
            # 基于V9的5档加权计算
            for level in range(self.levels):
                weight = self.weights[level]
                
                # 获取当前档位数据
                bid_price_key = f'bid{level+1}_price'
                ask_price_key = f'ask{level+1}_price'
                bid_size_key = f'bid{level+1}_size'
                ask_size_key = f'ask{level+1}_size'
                
                # 检查数据完整性
                if not all(key in current_order_book and key in self.last_order_book 
                          for key in [bid_price_key, ask_price_key, bid_size_key, ask_size_key]):
                    continue
                
                # 当前数据
                curr_bid_price = current_order_book[bid_price_key]
                curr_ask_price = current_order_book[ask_price_key]
                curr_bid_size = current_order_book[bid_size_key]
                curr_ask_size = current_order_book[ask_size_key]
                
                # 前一时刻数据
                prev_bid_price = self.last_order_book[bid_price_key]
                prev_ask_price = self.last_order_book[ask_price_key]
                prev_bid_size = self.last_order_book[bid_size_key]
                prev_ask_size = self.last_order_book[ask_size_key]
                
                # 检查价格改进 (V9逻辑)
                bid_improved = curr_bid_price > prev_bid_price
                ask_improved = curr_ask_price > prev_ask_price
                
                # 计算数量变化
                bid_delta = curr_bid_size - prev_bid_size
                ask_delta = curr_ask_size - prev_ask_size
                
                # OFI贡献 (V9公式)
                ofi_contribution = weight * (
                    bid_delta * bid_improved - ask_delta * ask_improved
                )
                
                ofi_total += ofi_contribution
            
            return ofi_total
            
        except Exception as e:
            logger.error(f"计算OFI失败: {e}")
            return 0.0
    
    def _calculate_cvd(self, current_order_book: Dict) -> float:
        """
        计算CVD (累积成交量差值)
        
        Args:
            current_order_book: 当前订单簿数据
            
        Returns:
            CVD值
        """
        try:
            if self.last_order_book is None:
                return 0.0
            
            # 基于订单簿变化估算CVD
            bid_size_change = 0.0
            ask_size_change = 0.0
            
            for level in range(self.levels):
                bid_size_key = f'bid{level+1}_size'
                ask_size_key = f'ask{level+1}_size'
                
                if bid_size_key in current_order_book and bid_size_key in self.last_order_book:
                    bid_size_change += current_order_book[bid_size_key] - self.last_order_book[bid_size_key]
                
                if ask_size_key in current_order_book and ask_size_key in self.last_order_book:
                    ask_size_change += current_order_book[ask_size_key] - self.last_order_book[ask_size_key]
            
            # CVD变化
            cvd_change = bid_size_change - ask_size_change
            
            # 累积CVD
            prev_cvd = self.cvd_history[-1]['cvd'] if self.cvd_history else 0.0
            cvd_total = prev_cvd + cvd_change
            
            return cvd_total
            
        except Exception as e:
            logger.error(f"计算CVD失败: {e}")
            return 0.0
    
    def _calculate_ofi_zscore(self) -> float:
        """
        计算OFI Z-score (基于V9参数)
        
        Returns:
            OFI Z-score值
        """
        try:
            if len(self.ofi_history) < 10:
                return 0.0
            
            # 转换为DataFrame
            df = pd.DataFrame(list(self.ofi_history))
            
            # 按时间窗口过滤
            cutoff_time = df['timestamp'].max() - timedelta(seconds=self.z_window)
            window_data = df[df['timestamp'] >= cutoff_time]
            
            if len(window_data) < 5:
                return 0.0
            
            # 计算Z-score
            latest_ofi = df['ofi'].iloc[-1]
            window_mean = window_data['ofi'].mean()
            window_std = window_data['ofi'].std()
            
            if window_std == 0:
                return 0.0
            
            z_score = (latest_ofi - window_mean) / window_std
            return z_score
            
        except Exception as e:
            logger.error(f"计算OFI Z-score失败: {e}")
            return 0.0
    
    def _calculate_cvd_zscore(self) -> float:
        """
        计算CVD Z-score (基于V9参数)
        
        Returns:
            CVD Z-score值
        """
        try:
            if len(self.cvd_history) < 10:
                return 0.0
            
            # 转换为DataFrame
            df = pd.DataFrame(list(self.cvd_history))
            
            # 按时间窗口过滤
            cutoff_time = df['timestamp'].max() - timedelta(seconds=self.z_window)
            window_data = df[df['timestamp'] >= cutoff_time]
            
            if len(window_data) < 5:
                return 0.0
            
            # 计算Z-score
            latest_cvd = df['cvd'].iloc[-1]
            window_mean = window_data['cvd'].mean()
            window_std = window_data['cvd'].std()
            
            if window_std == 0:
                return 0.0
            
            z_score = (latest_cvd - window_mean) / window_std
            return z_score
            
        except Exception as e:
            logger.error(f"计算CVD Z-score失败: {e}")
            return 0.0
    
    def _calculate_signal_strength(self) -> float:
        """
        计算信号强度 (基于V9逻辑)
        
        Returns:
            信号强度值
        """
        try:
            # 基于V9的信号强度计算
            signal_strength = (abs(self.ofi_zscore) + abs(self.cvd_zscore)) / 2
            return signal_strength
            
        except Exception as e:
            logger.error(f"计算信号强度失败: {e}")
            return 0.0
    
    def _calculate_quality_score(self) -> float:
        """
        计算质量评分
        
        Returns:
            质量评分 (0-1)
        """
        try:
            # 基于多个因子的质量评分
            ofi_quality = min(abs(self.ofi_zscore) / 3.0, 1.0)  # 归一化OFI强度
            cvd_quality = min(abs(self.cvd_zscore) / 3.0, 1.0)  # 归一化CVD强度
            spread_quality = max(0, 1.0 - (self.last_order_book.get('spread_bps', 0) / 50.0))  # 价差质量
            
            # 综合质量评分
            quality_score = (ofi_quality * 0.5 + cvd_quality * 0.3 + spread_quality * 0.2)
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"计算质量评分失败: {e}")
            return 0.0
    
    def get_current_values(self) -> Dict:
        """
        获取当前OFI相关值
        
        Returns:
            当前值字典
        """
        with self.data_lock:
            return {
                'ofi': self.current_ofi,
                'cvd': self.current_cvd,
                'ofi_zscore': self.ofi_zscore,
                'cvd_zscore': self.cvd_zscore,
                'signal_strength': self._calculate_signal_strength(),
                'quality_score': self._calculate_quality_score(),
                'last_update': self.stats['last_update']
            }
    
    def get_historical_data(self, window_seconds: int = None) -> Dict:
        """
        获取历史数据
        
        Args:
            window_seconds: 时间窗口 (秒)
            
        Returns:
            历史数据字典
        """
        if not window_seconds:
            window_seconds = self.window_seconds
        
        with self.data_lock:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            
            # 过滤OFI历史数据
            ofi_data = [record for record in self.ofi_history 
                       if record['timestamp'] >= cutoff_time]
            
            # 过滤CVD历史数据
            cvd_data = [record for record in self.cvd_history 
                       if record['timestamp'] >= cutoff_time]
            
            return {
                'ofi_data': ofi_data,
                'cvd_data': cvd_data,
                'window_seconds': window_seconds,
                'data_count': len(ofi_data)
            }
    
    def generate_v9_signals(self, params: Dict) -> Dict:
        """
        生成V9风格的信号 (基于V9参数)
        
        Args:
            params: V9信号参数
            
        Returns:
            信号字典
        """
        try:
            # V9参数
            ofi_z_min = params.get('ofi_z_min', 1.4)
            cvd_z_min = params.get('cvd_z_min', 0.6)
            min_signal_strength = params.get('min_signal_strength', 1.8)
            
            # 信号强度检查
            signal_strength = self._calculate_signal_strength()
            strong_signal = signal_strength >= min_signal_strength
            
            # OFI信号检查
            ofi_signal_long = self.ofi_zscore >= ofi_z_min
            ofi_signal_short = self.ofi_zscore <= -ofi_z_min
            
            # CVD信号检查
            cvd_signal_long = self.cvd_zscore >= cvd_z_min
            cvd_signal_short = self.cvd_zscore <= -cvd_z_min
            
            # 组合信号
            long_signal = ofi_signal_long and cvd_signal_long and strong_signal
            short_signal = ofi_signal_short and cvd_signal_short and strong_signal
            
            # 构建信号结果
            signal_result = {
                'timestamp': datetime.now(),
                'signal_type': None,
                'signal_side': 0,
                'signal_strength': signal_strength,
                'quality_score': self._calculate_quality_score(),
                'ofi_zscore': self.ofi_zscore,
                'cvd_zscore': self.cvd_zscore,
                'ofi': self.current_ofi,
                'cvd': self.current_cvd
            }
            
            if long_signal:
                signal_result['signal_type'] = 'v9_momentum'
                signal_result['signal_side'] = 1
            elif short_signal:
                signal_result['signal_type'] = 'v9_momentum'
                signal_result['signal_side'] = -1
            
            return signal_result
            
        except Exception as e:
            logger.error(f"生成V9信号失败: {e}")
            return {}
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self.data_lock:
            return {
                'ofis_calculated': self.stats['ofis_calculated'],
                'cvds_calculated': self.stats['cvds_calculated'],
                'z_scores_calculated': self.stats['z_scores_calculated'],
                'last_update': self.stats['last_update'],
                'order_book_history_count': len(self.order_book_history),
                'ofi_history_count': len(self.ofi_history),
                'cvd_history_count': len(self.cvd_history),
                'current_ofi': self.current_ofi,
                'current_cvd': self.current_cvd,
                'ofi_zscore': self.ofi_zscore,
                'cvd_zscore': self.cvd_zscore
            }
    
    def export_data(self, filepath_prefix="v12_real_ofi_data"):
        """
        导出数据
        
        Args:
            filepath_prefix: 文件路径前缀
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.data_lock:
            # 导出OFI历史数据
            if self.ofi_history:
                ofi_df = pd.DataFrame(list(self.ofi_history))
                ofi_file = f"{filepath_prefix}_ofi_{timestamp}.csv"
                ofi_df.to_csv(ofi_file, index=False)
                logger.info(f"OFI历史数据已导出: {ofi_file}")
            
            # 导出CVD历史数据
            if self.cvd_history:
                cvd_df = pd.DataFrame(list(self.cvd_history))
                cvd_file = f"{filepath_prefix}_cvd_{timestamp}.csv"
                cvd_df.to_csv(cvd_file, index=False)
                logger.info(f"CVD历史数据已导出: {cvd_file}")
            
            # 导出订单簿历史数据
            if self.order_book_history:
                order_book_df = pd.DataFrame(list(self.order_book_history))
                order_book_file = f"{filepath_prefix}_orderbook_{timestamp}.csv"
                order_book_df.to_csv(order_book_file, index=False)
                logger.info(f"订单簿历史数据已导出: {order_book_file}")


def test_v12_real_ofi_calculator():
    """测试V12真实OFI计算引擎"""
    logger.info("开始测试V12真实OFI计算引擎...")
    
    # 创建计算引擎
    calculator = V12RealOFICalculator(levels=5, window_seconds=2, z_window=1200)
    
    # 模拟订单簿数据
    def create_mock_order_book(timestamp, price_base=3000.0, spread=0.5):
        """创建模拟订单簿数据"""
        return {
            'timestamp': timestamp,
            'bid1_price': price_base - spread/2,
            'bid1_size': 100.0 + np.random.randn() * 10,
            'bid2_price': price_base - spread - 0.1,
            'bid2_size': 80.0 + np.random.randn() * 8,
            'bid3_price': price_base - spread - 0.2,
            'bid3_size': 60.0 + np.random.randn() * 6,
            'bid4_price': price_base - spread - 0.3,
            'bid4_size': 40.0 + np.random.randn() * 4,
            'bid5_price': price_base - spread - 0.4,
            'bid5_size': 20.0 + np.random.randn() * 2,
            'ask1_price': price_base + spread/2,
            'ask1_size': 100.0 + np.random.randn() * 10,
            'ask2_price': price_base + spread + 0.1,
            'ask2_size': 80.0 + np.random.randn() * 8,
            'ask3_price': price_base + spread + 0.2,
            'ask3_size': 60.0 + np.random.randn() * 6,
            'ask4_price': price_base + spread + 0.3,
            'ask4_size': 40.0 + np.random.randn() * 4,
            'ask5_price': price_base + spread + 0.4,
            'ask5_size': 20.0 + np.random.randn() * 2,
            'mid_price': price_base,
            'spread_bps': spread * 10000 / price_base
        }
    
    # 模拟数据更新
    base_price = 3000.0
    for i in range(100):
        timestamp = datetime.now() + timedelta(seconds=i)
        
        # 模拟价格变化
        price_change = np.random.randn() * 0.1
        base_price += price_change
        
        # 创建订单簿数据
        order_book = create_mock_order_book(timestamp, base_price)
        
        # 更新OFI计算
        result = calculator.update_order_book(order_book)
        
        if result:
            logger.info(f"更新 {i+1}: OFI={result['ofi']:.4f}, OFI_z={result['ofi_zscore']:.4f}, "
                       f"CVD={result['cvd']:.4f}, CVD_z={result['cvd_zscore']:.4f}, "
                       f"信号强度={result['signal_strength']:.4f}")
        
        time.sleep(0.1)  # 模拟100ms间隔
    
    # 测试V9信号生成
    v9_params = {
        'ofi_z_min': 1.4,
        'cvd_z_min': 0.6,
        'min_signal_strength': 1.8
    }
    
    signal = calculator.generate_v9_signals(v9_params)
    logger.info(f"V9信号生成测试: {signal}")
    
    # 获取统计信息
    stats = calculator.get_statistics()
    logger.info(f"统计信息: {stats}")
    
    # 导出数据
    calculator.export_data()
    
    logger.info("V12真实OFI计算引擎测试完成")


if __name__ == "__main__":
    test_v12_real_ofi_calculator()
