"""
V12 写实化数据模拟器
解决数据泄漏、过拟合、成本低估等系统性风险
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import random
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12RealisticDataSimulator:
    """V12写实化数据模拟器"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化模拟器
        
        Args:
            seed: 随机种子，确保每次生成不同的数据
        """
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # 市场参数
        self.base_price = 3000.0
        self.volatility = 0.002  # 2% 日内波动率
        self.trend_strength = 0.0001  # 轻微趋势
        self.noise_level = 0.0005  # 噪声水平
        
        # 微观结构参数
        self.tick_size = 0.01
        self.spread_mean = 0.5  # 平均点差
        self.spread_std = 0.2   # 点差标准差
        self.liquidity_factor = 0.8  # 流动性因子
        
        # 时间参数
        self.data_points = 1440  # 24小时，每分钟一个点
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info(f"V12写实化数据模拟器初始化完成 - 种子: {self.seed}")
    
    def generate_realistic_price_series(self) -> pd.DataFrame:
        """生成写实化的价格序列"""
        logger.info("生成写实化价格序列...")
        
        # 生成基础价格随机游走
        returns = np.random.normal(self.trend_strength, self.volatility, self.data_points)
        
        # 添加市场状态变化
        market_states = self._generate_market_states()
        
        # 根据市场状态调整波动率
        adjusted_returns = []
        for i, state in enumerate(market_states):
            if state == 'trending':
                returns[i] *= 1.5  # 趋势市场波动更大
            elif state == 'choppy':
                returns[i] *= 0.7  # 震荡市场波动较小
            elif state == 'breakout':
                returns[i] *= 2.0  # 突破市场波动最大
            adjusted_returns.append(returns[i])
        
        # 计算价格序列
        prices = [self.base_price]
        for ret in adjusted_returns:
            new_price = prices[-1] * (1 + ret)
            # 确保价格在合理范围内
            new_price = max(new_price, self.base_price * 0.9)
            new_price = min(new_price, self.base_price * 1.1)
            prices.append(new_price)
        
        prices = prices[1:]  # 移除初始价格
        
        # 生成时间序列
        timestamps = [self.start_time + timedelta(minutes=i) for i in range(self.data_points)]
        
        # 生成买卖价差
        spreads = np.random.normal(self.spread_mean, self.spread_std, self.data_points)
        spreads = np.maximum(spreads, 0.1)  # 最小点差
        
        bid_prices = [p - s/2 for p, s in zip(prices, spreads)]
        ask_prices = [p + s/2 for p, s in zip(prices, spreads)]
        
        # 生成成交量
        volumes = self._generate_realistic_volumes(market_states)
        
        # 生成订单簿数据
        order_book_data = self._generate_order_book_data(prices, spreads)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'bid': bid_prices,
            'ask': ask_prices,
            'spread': spreads,
            'volume': volumes,
            'market_state': market_states,
            **order_book_data
        })
        
        logger.info(f"生成了{len(df)}个数据点，价格范围: {min(prices):.2f}-{max(prices):.2f}")
        return df
    
    def _generate_market_states(self) -> List[str]:
        """生成市场状态序列"""
        states = []
        current_state = 'normal'
        state_duration = 0
        
        for i in range(self.data_points):
            # 状态持续时间
            if state_duration <= 0:
                # 随机选择新状态
                state_probs = {'normal': 0.5, 'trending': 0.2, 'choppy': 0.2, 'breakout': 0.1}
                current_state = np.random.choice(list(state_probs.keys()), p=list(state_probs.values()))
                state_duration = np.random.randint(60, 300)  # 1-5小时
            
            states.append(current_state)
            state_duration -= 1
        
        return states
    
    def _generate_realistic_volumes(self, market_states: List[str]) -> List[float]:
        """生成写实化成交量"""
        volumes = []
        base_volume = 100.0
        
        for state in market_states:
            if state == 'breakout':
                # 突破时成交量放大
                volume = base_volume * np.random.uniform(2.0, 4.0)
            elif state == 'trending':
                # 趋势时成交量增加
                volume = base_volume * np.random.uniform(1.2, 2.0)
            elif state == 'choppy':
                # 震荡时成交量减少
                volume = base_volume * np.random.uniform(0.5, 0.8)
            else:
                # 正常成交量
                volume = base_volume * np.random.uniform(0.8, 1.2)
            
            volumes.append(max(volume, 10.0))  # 最小成交量
        
        return volumes
    
    def _generate_order_book_data(self, prices: List[float], spreads: List[float]) -> Dict[str, List[float]]:
        """生成订单簿数据"""
        levels = 5
        
        # 生成5层买卖盘数据
        order_book = {}
        
        for level in range(1, levels + 1):
            # 买盘价格和数量
            bid_prices = [p - s/2 - level * 0.1 for p, s in zip(prices, spreads)]
            bid_sizes = [np.random.uniform(50, 200) * (1.0 / level) for _ in prices]
            
            # 卖盘价格和数量
            ask_prices = [p + s/2 + level * 0.1 for p, s in zip(prices, spreads)]
            ask_sizes = [np.random.uniform(50, 200) * (1.0 / level) for _ in prices]
            
            order_book[f'bid{level}_price'] = bid_prices
            order_book[f'bid{level}_size'] = bid_sizes
            order_book[f'ask{level}_price'] = ask_prices
            order_book[f'ask{level}_size'] = ask_sizes
        
        return order_book
    
    def calculate_realistic_ofi_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算写实化的OFI和CVD"""
        logger.info("计算写实化OFI和CVD...")
        
        ofi_values = []
        cvd_values = []
        
        for i in range(len(df)):
            # 基于订单簿不平衡计算OFI
            bid_pressure = sum(df[f'bid{j}_size'].iloc[i] for j in range(1, 6))
            ask_pressure = sum(df[f'ask{j}_size'].iloc[i] for j in range(1, 6))
            
            if bid_pressure + ask_pressure > 0:
                ofi = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
            else:
                ofi = 0.0
            
            ofi_values.append(ofi)
        
        # 计算CVD（累计成交量差）
        cvd = 0.0
        for i, ofi in enumerate(ofi_values):
            # 基于OFI和成交量计算CVD
            volume = df['volume'].iloc[i]
            cvd += ofi * volume * 0.01  # 调整因子
            cvd_values.append(cvd)
        
        # 计算Z分数
        ofi_z = []
        cvd_z = []
        
        window = 20
        for i in range(len(df)):
            if i < window:
                ofi_z.append(0.0)
                cvd_z.append(0.0)
            else:
                ofi_window = ofi_values[i-window:i]
                cvd_window = cvd_values[i-window:i]
                
                ofi_mean = np.mean(ofi_window)
                ofi_std = np.std(ofi_window)
                cvd_mean = np.mean(cvd_window)
                cvd_std = np.std(cvd_window)
                
                if ofi_std > 0:
                    ofi_z.append((ofi_values[i] - ofi_mean) / ofi_std)
                else:
                    ofi_z.append(0.0)
                
                if cvd_std > 0:
                    cvd_z.append((cvd_values[i] - cvd_mean) / cvd_std)
                else:
                    cvd_z.append(0.0)
        
        df['ofi'] = ofi_values
        df['cvd'] = cvd_values
        df['ofi_z'] = ofi_z
        df['cvd_z'] = cvd_z
        
        logger.info(f"OFI范围: {min(ofi_values):.4f} - {max(ofi_values):.4f}")
        logger.info(f"CVD范围: {min(cvd_values):.4f} - {max(cvd_values):.4f}")
        
        return df
    
    def add_realistic_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加写实化的交易成本"""
        logger.info("添加写实化交易成本...")
        
        # 手续费率
        maker_fee = 0.0002  # 0.02% maker fee
        taker_fee = 0.0004  # 0.04% taker fee
        
        # 滑点模型
        base_slippage = 0.0001  # 0.01% 基础滑点
        
        # 为每个数据点计算成本
        costs = []
        for i in range(len(df)):
            # 基于流动性计算滑点
            liquidity = sum(df[f'bid{j}_size'].iloc[i] for j in range(1, 6))
            liquidity += sum(df[f'ask{j}_size'].iloc[i] for j in range(1, 6))
            
            # 流动性越低，滑点越高
            slippage = base_slippage * (1000 / max(liquidity, 100))
            
            # 总成本 = 手续费 + 滑点
            total_cost = taker_fee + slippage
            costs.append(total_cost)
        
        df['trading_cost'] = costs
        df['maker_fee'] = maker_fee
        df['taker_fee'] = taker_fee
        
        logger.info(f"平均交易成本: {np.mean(costs):.6f}")
        
        return df
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """生成完整的写实化数据集"""
        logger.info("=" * 60)
        logger.info("V12写实化数据生成开始")
        logger.info("=" * 60)
        
        # 生成价格序列
        df = self.generate_realistic_price_series()
        
        # 计算OFI和CVD
        df = self.calculate_realistic_ofi_cvd(df)
        
        # 添加交易成本
        df = self.add_realistic_costs(df)
        
        # 添加技术指标
        df = self._add_technical_indicators(df)
        
        logger.info("=" * 60)
        logger.info("V12写实化数据生成完成")
        logger.info(f"数据集大小: {len(df)} 行 x {len(df.columns)} 列")
        logger.info(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}")
        logger.info(f"平均点差: {df['spread'].mean():.4f}")
        logger.info(f"平均交易成本: {df['trading_cost'].mean():.6f}")
        logger.info("=" * 60)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 简单移动平均
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 波动率
        df['volatility'] = df['price'].pct_change().rolling(window=20).std()
        
        return df


def test_v12_realistic_simulator():
    """测试V12写实化模拟器"""
    logger.info("测试V12写实化数据模拟器...")
    
    # 创建模拟器
    simulator = V12RealisticDataSimulator(seed=42)
    
    # 生成数据集
    df = simulator.generate_complete_dataset()
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"v12_realistic_data_seed_{simulator.seed}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"测试数据已保存到: {filename}")
    
    # 显示统计信息
    logger.info("数据统计:")
    logger.info(f"价格变化: {df['price'].iloc[0]:.2f} -> {df['price'].iloc[-1]:.2f}")
    logger.info(f"价格变化率: {(df['price'].iloc[-1]/df['price'].iloc[0]-1)*100:.2f}%")
    logger.info(f"OFI范围: {df['ofi'].min():.4f} - {df['ofi'].max():.4f}")
    logger.info(f"CVD范围: {df['cvd'].min():.4f} - {df['cvd'].max():.4f}")
    logger.info(f"平均交易成本: {df['trading_cost'].mean():.6f}")
    
    return df


if __name__ == "__main__":
    test_v12_realistic_simulator()
