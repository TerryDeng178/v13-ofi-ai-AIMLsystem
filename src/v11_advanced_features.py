#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11高级特征工程模块
扩展特征到100+个，包括技术指标、市场微观结构、情绪指标等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11AdvancedFeatureEngine:
    """V11高级特征工程引擎"""
    
    def __init__(self):
        self.feature_columns = []
        self.feature_importance = {}
        self.feature_correlation = None
        
        logger.info("V11高级特征工程引擎初始化完成")
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        logger.info("开始V11高级特征工程...")
        
        # 复制原始数据
        df_features = df.copy()
        
        # 1. 基础价格特征
        df_features = self._create_basic_price_features(df_features)
        
        # 2. 技术指标特征
        df_features = self._create_technical_indicators(df_features)
        
        # 3. 市场微观结构特征
        df_features = self._create_microstructure_features(df_features)
        
        # 4. 情绪指标特征
        df_features = self._create_sentiment_features(df_features)
        
        # 5. 宏观经济特征
        df_features = self._create_macro_features(df_features)
        
        # 6. 跨市场特征
        df_features = self._create_cross_market_features(df_features)
        
        # 7. 高级统计特征
        df_features = self._create_advanced_statistical_features(df_features)
        
        # 8. 机器学习特征
        df_features = self._create_ml_features(df_features)
        
        # 填充缺失值
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 更新特征列表
        self.feature_columns = [col for col in df_features.columns if col not in ['open_time', 'close_time', 'ignore']]
        
        logger.info(f"V11特征工程完成，总特征数: {len(self.feature_columns)}")
        
        return df_features
    
    def _create_basic_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础价格特征"""
        logger.info("创建基础价格特征...")
        
        # 价格变化
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['price_change_squared'] = df['price_change'] ** 2
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_position_5'] = df['price_position'].rolling(5).mean()
        df['price_position_20'] = df['price_position'].rolling(20).mean()
        
        # 价格波动
        df['price_volatility_5'] = df['price_change'].rolling(5).std()
        df['price_volatility_10'] = df['price_change'].rolling(10).std()
        df['price_volatility_20'] = df['price_change'].rolling(20).std()
        
        # 价格动量
        df['price_momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        logger.info("创建技术指标特征...")
        
        # 移动平均线
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # 移动平均线交叉
        df['sma_cross_5_20'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
        df['sma_cross_10_50'] = np.where(df['sma_10'] > df['sma_50'], 1, -1)
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_cross_12_26'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
        else:
            df['ema_cross_12_26'] = 0
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
        df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
        
        # MACD
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_cross'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
                                       np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0))
        else:
            # 使用SMA计算MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_cross'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
                                       np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5, 1, 0)
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_21'] = self._calculate_atr(df, 21)
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # 威廉指标
        df['williams_r'] = self._calculate_williams_r(df, 14)
        df['williams_oversold'] = np.where(df['williams_r'] < -80, 1, 0)
        df['williams_overbought'] = np.where(df['williams_r'] > -20, 1, 0)
        
        # 随机指标
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14)
        df['stoch_oversold'] = np.where(df['stoch_k'] < 20, 1, 0)
        df['stoch_overbought'] = np.where(df['stoch_k'] > 80, 1, 0)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建市场微观结构特征"""
        logger.info("创建市场微观结构特征...")
        
        # 成交量特征
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_spike'] = np.where(df['volume_ratio'] > 2, 1, 0)
        
        # 成交量价格关系
        df['volume_price_trend'] = df['volume'] * df['price_change']
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['price_change'])
        
        # 订单流特征（模拟）
        df['order_flow_imbalance'] = np.random.randn(len(df)) * 0.1
        df['order_flow_pressure'] = df['order_flow_imbalance'].rolling(10).mean()
        df['order_flow_momentum'] = df['order_flow_imbalance'].diff()
        
        # 价格冲击
        df['price_impact'] = df['price_change'] / (df['volume'] + 1e-8)
        df['price_impact_sma'] = df['price_impact'].rolling(10).mean()
        
        # 买卖压力
        df['buy_pressure'] = np.where(df['price_change'] > 0, df['volume'], 0)
        df['sell_pressure'] = np.where(df['price_change'] < 0, df['volume'], 0)
        df['buy_sell_ratio'] = df['buy_pressure'].rolling(20).sum() / (df['sell_pressure'].rolling(20).sum() + 1e-8)
        
        return df
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建情绪指标特征"""
        logger.info("创建情绪指标特征...")
        
        # 恐惧贪婪指数（模拟）
        df['fear_greed_index'] = np.random.uniform(0, 100, len(df))
        df['fear_greed_extreme'] = np.where((df['fear_greed_index'] < 20) | (df['fear_greed_index'] > 80), 1, 0)
        
        # 市场情绪
        df['market_sentiment'] = np.where(df['price_change'] > 0, 1, -1)
        df['sentiment_momentum'] = df['market_sentiment'].rolling(10).mean()
        df['sentiment_volatility'] = df['market_sentiment'].rolling(20).std()
        
        # 情绪指标
        df['sentiment_rsi'] = self._calculate_rsi(df['market_sentiment'], 14)
        df['sentiment_macd'] = df['sentiment_momentum'].ewm(span=12).mean() - df['sentiment_momentum'].ewm(span=26).mean()
        
        return df
    
    def _create_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建宏观经济特征"""
        logger.info("创建宏观经济特征...")
        
        # 利率特征（模拟）
        df['interest_rate'] = np.random.uniform(0.01, 0.05, len(df))
        df['interest_rate_change'] = df['interest_rate'].diff()
        
        # 通胀特征（模拟）
        df['inflation_rate'] = np.random.uniform(0.02, 0.04, len(df))
        df['inflation_change'] = df['inflation_rate'].diff()
        
        # 经济增长特征（模拟）
        df['gdp_growth'] = np.random.uniform(0.01, 0.03, len(df))
        df['gdp_change'] = df['gdp_growth'].diff()
        
        # 货币供应量（模拟）
        df['money_supply'] = np.random.uniform(1000, 2000, len(df))
        df['money_supply_growth'] = df['money_supply'].pct_change()
        
        return df
    
    def _create_cross_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建跨市场特征"""
        logger.info("创建跨市场特征...")
        
        # 股票市场特征（模拟）
        df['stock_market_index'] = np.random.randn(len(df)).cumsum() + 1000
        df['stock_market_return'] = df['stock_market_index'].pct_change()
        df['stock_crypto_correlation'] = df['stock_market_return'].rolling(20).corr(df['price_change'])
        
        # 债券市场特征（模拟）
        df['bond_yield'] = np.random.uniform(0.02, 0.04, len(df))
        df['bond_yield_change'] = df['bond_yield'].diff()
        df['bond_crypto_correlation'] = df['bond_yield_change'].rolling(20).corr(df['price_change'])
        
        # 商品市场特征（模拟）
        df['commodity_index'] = np.random.randn(len(df)).cumsum() + 100
        df['commodity_return'] = df['commodity_index'].pct_change()
        df['commodity_crypto_correlation'] = df['commodity_return'].rolling(20).corr(df['price_change'])
        
        # 外汇市场特征（模拟）
        df['usd_index'] = np.random.randn(len(df)).cumsum() + 100
        df['usd_change'] = df['usd_index'].pct_change()
        df['usd_crypto_correlation'] = df['usd_change'].rolling(20).corr(df['price_change'])
        
        return df
    
    def _create_advanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建高级统计特征"""
        logger.info("创建高级统计特征...")
        
        # 高阶矩
        df['skewness_5'] = df['price_change'].rolling(5).skew()
        df['skewness_20'] = df['price_change'].rolling(20).skew()
        df['kurtosis_5'] = df['price_change'].rolling(5).kurt()
        df['kurtosis_20'] = df['price_change'].rolling(20).kurt()
        
        # 分位数
        df['quantile_25'] = df['price_change'].rolling(20).quantile(0.25)
        df['quantile_75'] = df['price_change'].rolling(20).quantile(0.75)
        df['quantile_range'] = df['quantile_75'] - df['quantile_25']
        
        # 自相关
        df['autocorr_1'] = df['price_change'].rolling(20).apply(lambda x: x.autocorr(lag=1))
        df['autocorr_5'] = df['price_change'].rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # 协整关系
        df['cointegration'] = df['close'].rolling(20).apply(lambda x: self._calculate_cointegration(x))
        
        # 趋势强度
        df['trend_strength'] = df['close'].rolling(20).apply(lambda x: self._calculate_trend_strength(x))
        
        return df
    
    def _create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建机器学习特征"""
        logger.info("创建机器学习特征...")
        
        # 特征组合
        df['price_volume_interaction'] = df['price_change'] * df['volume_ratio']
        df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio']
        df['macd_volume_interaction'] = df['macd'] * df['volume_ratio']
        
        # 特征变换
        df['price_change_log'] = np.log1p(df['price_change'].abs())
        df['volume_log'] = np.log1p(df['volume'])
        df['price_sqrt'] = np.sqrt(df['close'])
        
        # 特征比率
        df['price_volume_ratio'] = df['close'] / (df['volume'] + 1e-8)
        df['rsi_macd_ratio'] = df['rsi_14'] / (df['macd'].abs() + 1e-8)
        
        # 特征差分
        df['price_change_diff'] = df['price_change'].diff()
        df['volume_diff'] = df['volume'].diff()
        df['rsi_diff'] = df['rsi_14'].diff()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算威廉指标"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """计算随机指标"""
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(3).mean()
        
        return k_percent, d_percent
    
    def _calculate_cointegration(self, series: pd.Series) -> float:
        """计算协整关系"""
        if len(series) < 2:
            return 0.0
        try:
            # 简化的协整检验
            return np.corrcoef(series[:-1], series[1:])[0, 1]
        except:
            return 0.0
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """计算趋势强度"""
        if len(series) < 2:
            return 0.0
        try:
            # 线性回归斜率
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            return slope
        except:
            return 0.0
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_column: str = 'price_change') -> Dict:
        """分析特征重要性"""
        logger.info("分析特征重要性...")
        
        # 计算特征与目标的相关性
        correlations = {}
        for col in self.feature_columns:
            if col in df.columns and col != target_column:
                try:
                    corr = df[col].corr(df[target_column])
                    correlations[col] = abs(corr) if not np.isnan(corr) else 0
                except:
                    correlations[col] = 0
        
        # 排序特征重要性
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        self.feature_importance = dict(sorted_features)
        
        logger.info("特征重要性分析完成")
        logger.info(f"前10个重要特征: {list(sorted_features[:10])}")
        
        return self.feature_importance
    
    def get_feature_summary(self) -> Dict:
        """获取特征摘要"""
        return {
            'total_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance
        }

def main():
    """主函数 - 演示V11高级特征工程"""
    logger.info("=" * 60)
    logger.info("V11高级特征工程演示")
    logger.info("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(100, 200, n_samples),
        'low': np.random.uniform(100, 200, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # 确保high >= low, high >= close, low <= close
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    df['high'] = np.maximum(df['high'], df['low'])
    
    logger.info(f"原始数据形状: {df.shape}")
    
    # 创建特征工程实例
    feature_engine = V11AdvancedFeatureEngine()
    
    # 创建所有特征
    df_features = feature_engine.create_all_features(df)
    
    logger.info(f"特征工程后数据形状: {df_features.shape}")
    logger.info(f"总特征数: {len(feature_engine.feature_columns)}")
    
    # 分析特征重要性
    feature_importance = feature_engine.analyze_feature_importance(df_features)
    
    # 获取特征摘要
    feature_summary = feature_engine.get_feature_summary()
    
    logger.info("=" * 60)
    logger.info("V11高级特征工程演示完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
