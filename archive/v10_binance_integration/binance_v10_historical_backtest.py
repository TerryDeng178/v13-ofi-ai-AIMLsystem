#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于币安真实历史数据的V10回测系统
使用币安API获取真实历史数据，进行V10算法回测
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceHistoricalData:
    """币安历史数据获取器"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        self.base_url = "https://fapi.binance.com"
        
    def get_klines(self, interval: str = "1m", limit: int = 1000, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """获取K线数据"""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"未获取到{self.symbol}的K线数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算中价
            df['mid_price'] = (df['high'] + df['low']) / 2
            
            logger.info(f"成功获取{len(df)}条K线数据，时间范围: {df['open_time'].min()} 到 {df['open_time'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, days: int = 7, interval: str = "1m") -> pd.DataFrame:
        """获取历史数据"""
        logger.info(f"开始获取{self.symbol}的{days}天历史数据...")
        
        # 计算时间范围
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # 每次获取1000条数据
            df_batch = self.get_klines(interval=interval, limit=1000, start_time=current_start)
            
            if df_batch.empty:
                break
                
            all_data.append(df_batch)
            
            # 更新起始时间
            current_start = int(df_batch['close_time'].max().timestamp() * 1000) + 1
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        if not all_data:
            logger.error("未获取到任何历史数据")
            return pd.DataFrame()
        
        # 合并所有数据
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
        
        logger.info(f"历史数据获取完成，共{len(df)}条记录")
        return df

class V10FeatureEngine:
    """V10特征工程"""
    
    def __init__(self):
        self.feature_columns = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        logger.info("计算技术指标...")
        
        # 价格变化
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # 移动平均线
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 价格动量
        df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # 波动率
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        logger.info(f"技术指标计算完成，特征数量: {len(df.columns)}")
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
    
    def calculate_ofi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算OFI相关特征"""
        logger.info("计算OFI特征...")
        
        # 模拟OFI计算（基于价格和成交量）
        df['ofi_raw'] = df['price_change'] * df['volume']
        df['ofi_sma'] = df['ofi_raw'].rolling(20).mean()
        df['ofi_std'] = df['ofi_raw'].rolling(20).std()
        df['ofi_z'] = (df['ofi_raw'] - df['ofi_sma']) / df['ofi_std']
        
        # 模拟CVD计算
        df['cvd_raw'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
        df['cvd_sma'] = df['cvd_raw'].rolling(20).mean()
        df['cvd_std'] = df['cvd_raw'].rolling(20).std()
        df['cvd_z'] = (df['cvd_raw'] - df['cvd_sma']) / df['cvd_std']
        
        # 订单流不平衡
        df['order_flow_imbalance'] = df['ofi_z'] + df['cvd_z']
        
        logger.info("OFI特征计算完成")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        logger.info("开始特征工程...")
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        # 计算OFI特征
        df = self.calculate_ofi_features(df)
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_columns = [col for col in df.columns if col not in ['open_time', 'close_time', 'ignore']]
        logger.info(f"特征工程完成，总特征数: {len(self.feature_columns)}")
        
        return df

class V10SignalGenerator:
    """V10信号生成器"""
    
    def __init__(self):
        self.signal_threshold = 1.5
        self.quality_threshold = 0.6
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        logger.info("生成V10交易信号...")
        
        # 初始化信号列
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_quality'] = 0.0
        
        # 基于OFI Z-score的信号
        ofi_signals = np.where(df['ofi_z'] > self.signal_threshold, 1,
                              np.where(df['ofi_z'] < -self.signal_threshold, -1, 0))
        
        # 基于价格动量的信号
        momentum_signals = np.where(df['momentum_5'] > 0.01, 1,
                                  np.where(df['momentum_5'] < -0.01, -1, 0))
        
        # 基于RSI的信号
        rsi_signals = np.where(df['rsi'] < 30, 1,
                              np.where(df['rsi'] > 70, -1, 0))
        
        # 基于MACD的信号
        macd_signals = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
                               np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0))
        
        # 综合信号
        combined_signals = ofi_signals + momentum_signals + rsi_signals + macd_signals
        
        # 信号强度
        signal_strength = np.abs(df['ofi_z']) + np.abs(df['momentum_5']) + np.abs(df['rsi'] - 50) / 50
        
        # 信号质量评分
        signal_quality = self._calculate_signal_quality(df)
        
        # 应用阈值过滤
        valid_signals = np.where(signal_strength > self.signal_threshold, combined_signals, 0)
        valid_signals = np.where(signal_quality > self.quality_threshold, valid_signals, 0)
        
        df['signal'] = valid_signals
        df['signal_strength'] = signal_strength
        df['signal_quality'] = signal_quality
        
        # 统计信号
        total_signals = (df['signal'] != 0).sum()
        long_signals = (df['signal'] == 1).sum()
        short_signals = (df['signal'] == -1).sum()
        
        logger.info(f"信号生成完成:")
        logger.info(f"  总信号数: {total_signals}")
        logger.info(f"  多头信号: {long_signals}")
        logger.info(f"  空头信号: {short_signals}")
        logger.info(f"  平均信号强度: {df['signal_strength'].mean():.4f}")
        logger.info(f"  平均信号质量: {df['signal_quality'].mean():.4f}")
        
        return df
    
    def _calculate_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """计算信号质量评分"""
        # 基于多个因子的质量评分
        ofi_quality = np.abs(df['ofi_z']) / 3.0  # 标准化到0-1
        momentum_quality = np.abs(df['momentum_5']) / 0.05  # 标准化到0-1
        volume_quality = df['volume_ratio'] / 3.0  # 标准化到0-1
        volatility_quality = df['volatility_5'] / 0.02  # 标准化到0-1
        
        # 综合质量评分
        quality = (ofi_quality * 0.3 + momentum_quality * 0.3 + 
                  volume_quality * 0.2 + volatility_quality * 0.2)
        
        return np.clip(quality, 0, 1)

class V10Backtester:
    """V10回测器"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """运行回测"""
        logger.info("开始V10回测...")
        
        self.reset()
        
        for i, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            signal_strength = row['signal_strength']
            signal_quality = row['signal_quality']
            
            # 更新权益曲线
            self.equity_curve.append(self.capital)
            
            # 计算回撤
            if self.capital > self.peak_equity:
                self.peak_equity = self.capital
            
            current_drawdown = (self.peak_equity - self.capital) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # 处理信号
            if signal != 0 and self.position == 0:
                # 开仓
                self._open_position(signal, current_price, signal_strength, signal_quality)
            elif self.position != 0:
                # 检查平仓条件
                self._check_exit_conditions(current_price, signal)
        
        # 计算回测结果
        results = self._calculate_results()
        
        logger.info("V10回测完成")
        return results
    
    def _open_position(self, signal: int, price: float, strength: float, quality: float):
        """开仓"""
        # 根据信号强度和质量调整仓位大小
        position_size = min(1.0, strength * quality)
        self.position = signal * position_size
        self.entry_price = price
        
        # 记录交易
        self.trades.append({
            'action': 'OPEN',
            'side': 'LONG' if signal > 0 else 'SHORT',
            'price': price,
            'size': position_size,
            'strength': strength,
            'quality': quality,
            'timestamp': len(self.equity_curve)
        })
    
    def _check_exit_conditions(self, current_price: float, signal: int):
        """检查平仓条件"""
        if self.position == 0:
            return
        
        # 计算当前盈亏
        if self.position > 0:  # 多头仓位
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空头仓位
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # 平仓条件
        should_close = False
        exit_reason = ""
        
        # 止损条件
        if pnl_pct < -0.02:  # 2%止损
            should_close = True
            exit_reason = "STOP_LOSS"
        
        # 止盈条件
        elif pnl_pct > 0.04:  # 4%止盈
            should_close = True
            exit_reason = "TAKE_PROFIT"
        
        # 信号反转
        elif signal != 0 and signal * self.position < 0:
            should_close = True
            exit_reason = "SIGNAL_REVERSAL"
        
        # 时间止损（持有时间过长）
        elif len(self.trades) > 0 and len(self.equity_curve) - self.trades[-1]['timestamp'] > 100:
            should_close = True
            exit_reason = "TIME_STOP"
        
        if should_close:
            self._close_position(current_price, exit_reason)
    
    def _close_position(self, price: float, reason: str):
        """平仓"""
        if self.position == 0:
            return
        
        # 计算盈亏
        if self.position > 0:  # 多头仓位
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # 空头仓位
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # 计算手续费
        commission_cost = abs(self.position) * self.commission
        
        # 更新资金
        pnl_amount = self.capital * pnl_pct * abs(self.position)
        self.capital += pnl_amount - commission_cost
        
        # 记录交易
        self.trades.append({
            'action': 'CLOSE',
            'side': 'LONG' if self.position > 0 else 'SHORT',
            'price': price,
            'size': abs(self.position),
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'commission': commission_cost,
            'reason': reason,
            'timestamp': len(self.equity_curve)
        })
        
        # 重置仓位
        self.position = 0.0
        self.entry_price = 0.0
    
    def _calculate_results(self) -> Dict:
        """计算回测结果"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0
            }
        
        # 计算交易统计
        closed_trades = [t for t in self.trades if t['action'] == 'CLOSE']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0
            }
        
        total_trades = len(closed_trades)
        total_pnl = sum(t['pnl_amount'] for t in closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl_amount'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # 计算夏普比率
        returns = [t['pnl_pct'] for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # 计算盈利因子
        gross_profit = sum(t['pnl_amount'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl_amount'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 计算最终收益
        final_return = (self.capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'final_capital': self.capital,
            'final_return': final_return,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'max_win': max([t['pnl_amount'] for t in closed_trades]) if closed_trades else 0,
            'max_loss': min([t['pnl_amount'] for t in closed_trades]) if closed_trades else 0
        }

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("币安V10历史数据回测系统")
    logger.info("=" * 60)
    
    # 配置参数
    symbol = "ETHUSDT"
    days = 7  # 获取7天历史数据
    initial_capital = 10000.0
    
    try:
        # 1. 获取历史数据
        logger.info("步骤1: 获取币安历史数据...")
        data_fetcher = BinanceHistoricalData(symbol)
        df = data_fetcher.get_historical_data(days=days, interval="1m")
        
        if df.empty:
            logger.error("未获取到历史数据，回测终止")
            return
        
        logger.info(f"历史数据获取完成: {len(df)}条记录")
        logger.info(f"时间范围: {df['open_time'].min()} 到 {df['open_time'].max()}")
        logger.info(f"价格范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
        
        # 2. 特征工程
        logger.info("步骤2: 特征工程...")
        feature_engine = V10FeatureEngine()
        df = feature_engine.create_features(df)
        
        # 3. 信号生成
        logger.info("步骤3: 生成交易信号...")
        signal_generator = V10SignalGenerator()
        df = signal_generator.generate_signals(df)
        
        # 4. 回测
        logger.info("步骤4: 运行回测...")
        backtester = V10Backtester(initial_capital=initial_capital)
        results = backtester.run_backtest(df)
        
        # 5. 输出结果
        logger.info("=" * 60)
        logger.info("V10回测结果")
        logger.info("=" * 60)
        logger.info(f"交易对: {symbol}")
        logger.info(f"数据期间: {days}天")
        logger.info(f"初始资金: ${initial_capital:,.2f}")
        logger.info(f"最终资金: ${results['final_capital']:,.2f}")
        logger.info(f"总收益: ${results['total_pnl']:,.2f}")
        logger.info(f"收益率: {results['final_return']:.2%}")
        logger.info(f"总交易数: {results['total_trades']}")
        logger.info(f"胜率: {results['win_rate']:.2%}")
        logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {results['sharpe_ratio']:.4f}")
        logger.info(f"盈利因子: {results['profit_factor']:.4f}")
        logger.info(f"平均每笔收益: ${results['avg_trade_pnl']:,.2f}")
        logger.info(f"最大单笔盈利: ${results['max_win']:,.2f}")
        logger.info(f"最大单笔亏损: ${results['max_loss']:,.2f}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"binance_v10_backtest_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"回测结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
