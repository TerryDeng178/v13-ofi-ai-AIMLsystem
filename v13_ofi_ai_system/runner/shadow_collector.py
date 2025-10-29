#!/usr/bin/env python3
"""
Task 1.5 核心算法v1 - 影子数据收集器
从现有数据源收集数据并运行影子算法
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.core_algo import CoreAlgorithm, SignalConfig
from runner.shadow_runner import ShadowTrader

# 导入成熟组件
from utils.strategy_mode_manager import StrategyModeManager, StrategyMode, MarketActivity
from utils.config_loader import load_config, get_config

class ShadowDataCollector:
    """影子数据收集器"""
    
    def __init__(self, symbols: List[str], config: Dict):
        self.symbols = symbols
        self.config = config
        
        # 加载统一配置
        self.system_config = load_config()
        
        # 初始化算法组件 - 使用统一配置
        signal_config = SignalConfig(
            w_ofi=config.get('w_ofi', 0.6),
            w_cvd=config.get('w_cvd', 0.4),
            z_hi=config.get('z_hi', 2.0),
            z_mid=config.get('z_mid', 1.0)
        )
        self.core_algo = CoreAlgorithm("BTCUSDT", signal_config)  # 使用默认symbol
        
        # 使用统一配置的策略模式管理器
        strategy_config = self.system_config.get('strategy', {})
        if not strategy_config:
            # 如果没有策略配置，使用默认配置
            strategy_config = {
                'strategy': {
                    'mode': 'auto',
                    'hysteresis': {
                        'window_secs': 60,
                        'min_active_windows': 3,
                        'min_quiet_windows': 6
                    },
                    'triggers': {
                        'schedule': {
                            'enabled': True,
                            'timezone': 'Asia/Hong_Kong',
                            'calendar': 'CRYPTO',
                            'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                            'holidays': [],
                            'active_windows': [],
                            'wrap_midnight': True
                        },
                        'market': {
                            'enabled': True,
                            'window_secs': 60,
                            'min_trades_per_min': 100.0,
                            'min_quote_updates_per_sec': 100,
                            'max_spread_bps': 15.0,
                            'min_volatility_bps': 200.0,  # 0.02 * 10000
                            'min_volume_usd': 1000000,
                            'use_median': True,
                            'winsorize_percentile': 95
                        }
                    }
                }
            }
        self.strategy_manager = StrategyModeManager(strategy_config)
        
        # 为每个交易对创建影子交易器
        self.traders = {}
        for symbol in symbols:
            self.traders[symbol] = ShadowTrader(symbol, config)
        
        # 数据缓存
        self.data_cache = {}
        self.last_update_time = {}
        
        # 日志
        self.logger = logging.getLogger("ShadowDataCollector")
        
    def load_historical_data(self, symbol: str, start_time: int, end_time: int) -> Dict:
        """加载实时数据收集器的真实历史数据"""
        data_dir = Path("artifacts/runtime/48h_collection/48h_collection_20251022_0655")
        symbol_data = {}
        
        # 查找最新的数据文件
        date_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('date=')])
        if not date_dirs:
            self.logger.warning(f"未找到数据目录 {data_dir}")
            return {}
        
        # 使用最新的数据目录
        latest_date_dir = date_dirs[-1]
        symbol_dir = latest_date_dir / f"symbol={symbol}"
        
        if not symbol_dir.exists():
            self.logger.warning(f"未找到 {symbol} 的数据目录 {symbol_dir}")
            return {}
        
        # 加载价格数据
        prices_path = symbol_dir / "kind=prices"
        if prices_path.exists():
            price_files = list(prices_path.glob("*.parquet"))
            if price_files:
                try:
                    # 修复pandas concat警告：过滤空DataFrame
                    price_dataframes = []
                    for f in price_files:
                        df = pd.read_parquet(f)
                        if not df.empty:
                            price_dataframes.append(df)
                    
                    if price_dataframes:
                        prices_df = pd.concat(price_dataframes, ignore_index=True)
                    else:
                        prices_df = pd.DataFrame()
                    # 按时间过滤
                    if 'event_ts_ms' in prices_df.columns:
                        prices_df = prices_df[
                            (prices_df['event_ts_ms'] >= start_time * 1000) & 
                            (prices_df['event_ts_ms'] <= end_time * 1000)
                        ]
                        # 统一时间列名
                        if 'ts_ms' not in prices_df.columns:
                            prices_df['ts_ms'] = prices_df['event_ts_ms']
                    symbol_data['prices'] = prices_df
                    self.logger.info(f"加载价格数据: {len(prices_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载价格数据失败: {e}")
        
        # 加载OFI数据
        ofi_path = symbol_dir / "kind=ofi"
        if ofi_path.exists():
            ofi_files = list(ofi_path.glob("*.parquet"))
            if ofi_files:
                try:
                    # 修复pandas concat警告：过滤空DataFrame
                    ofi_dataframes = []
                    for f in ofi_files:
                        df = pd.read_parquet(f)
                        if not df.empty:
                            ofi_dataframes.append(df)
                    
                    if ofi_dataframes:
                        ofi_df = pd.concat(ofi_dataframes, ignore_index=True)
                    else:
                        ofi_df = pd.DataFrame()
                    # 按时间过滤
                    if 'event_ts_ms' in ofi_df.columns:
                        ofi_df = ofi_df[
                            (ofi_df['event_ts_ms'] >= start_time * 1000) & 
                            (ofi_df['event_ts_ms'] <= end_time * 1000)
                        ]
                    symbol_data['ofi'] = ofi_df
                    self.logger.info(f"加载OFI数据: {len(ofi_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载OFI数据失败: {e}")
        
        # 加载CVD数据
        cvd_path = symbol_dir / "kind=cvd"
        if cvd_path.exists():
            cvd_files = list(cvd_path.glob("*.parquet"))
            if cvd_files:
                try:
                    # 修复pandas concat警告：过滤空DataFrame
                    cvd_dataframes = []
                    for f in cvd_files:
                        df = pd.read_parquet(f)
                        if not df.empty:
                            cvd_dataframes.append(df)
                    
                    if cvd_dataframes:
                        cvd_df = pd.concat(cvd_dataframes, ignore_index=True)
                    else:
                        cvd_df = pd.DataFrame()
                    # 按时间过滤
                    if 'event_ts_ms' in cvd_df.columns:
                        cvd_df = cvd_df[
                            (cvd_df['event_ts_ms'] >= start_time * 1000) & 
                            (cvd_df['event_ts_ms'] <= end_time * 1000)
                        ]
                    symbol_data['cvd'] = cvd_df
                    self.logger.info(f"加载CVD数据: {len(cvd_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载CVD数据失败: {e}")
        
        # 加载Fusion数据
        fusion_path = symbol_dir / "kind=fusion"
        if fusion_path.exists():
            fusion_files = list(fusion_path.glob("*.parquet"))
            if fusion_files:
                try:
                    # 修复pandas concat警告：过滤空DataFrame
                    fusion_dataframes = []
                    for f in fusion_files:
                        df = pd.read_parquet(f)
                        if not df.empty:
                            fusion_dataframes.append(df)
                    
                    if fusion_dataframes:
                        fusion_df = pd.concat(fusion_dataframes, ignore_index=True)
                    else:
                        fusion_df = pd.DataFrame()
                    # 按时间过滤
                    if 'event_ts_ms' in fusion_df.columns:
                        fusion_df = fusion_df[
                            (fusion_df['event_ts_ms'] >= start_time * 1000) & 
                            (fusion_df['event_ts_ms'] <= end_time * 1000)
                        ]
                    symbol_data['fusion'] = fusion_df
                    self.logger.info(f"加载Fusion数据: {len(fusion_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载Fusion数据失败: {e}")
        
        return symbol_data
    
    def calculate_z_scores(self, data: pd.DataFrame, window: int = 100) -> pd.Series:
        """计算Z分数"""
        if len(data) < window:
            return pd.Series([0.0] * len(data), index=data.index)
        
        z_scores = []
        for i in range(len(data)):
            if i < window:
                z_scores.append(0.0)
            else:
                window_data = data.iloc[i-window:i]
                median = window_data.median()
                mad = (window_data - median).abs().median()
                if mad > 0:
                    z_score = (data.iloc[i] - median) / (1.4826 * mad)
                else:
                    z_score = 0.0
                z_scores.append(z_score)
        
        return pd.Series(z_scores, index=data.index)
    
    def calculate_market_metrics(self, prices_df: pd.DataFrame, ofi_df: pd.DataFrame, 
                               cvd_df: pd.DataFrame) -> Dict:
        """计算市场指标"""
        if prices_df.empty:
            return {}
        
        # 计算交易频率
        trade_rate = len(prices_df) / 60.0  # 每分钟交易数
        
        # 计算已实现波动率
        if len(prices_df) > 1:
            returns = np.diff(np.log(prices_df['price'].values))
            realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # 年化波动率
        else:
            realized_vol = 0.0
        
        # 计算价差（如果有订单簿数据）
        if 'best_bid' in prices_df.columns and 'best_ask' in prices_df.columns:
            mid_prices = (prices_df['best_bid'] + prices_df['best_ask']) / 2
            spreads = (prices_df['best_ask'] - prices_df['best_bid']) / mid_prices
            avg_spread_bps = spreads.mean() * 10000
        else:
            avg_spread_bps = 5.0  # 默认5bps
        
        # 计算价格动量
        if len(prices_df) > 10:
            recent_prices = prices_df['price'].tail(10)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        else:
            price_momentum = 0.0
        
        return {
            'trade_rate': trade_rate,
            'realized_vol': realized_vol,
            'spread_bps': avg_spread_bps,
            'price_momentum': price_momentum,
            'volume_ratio': 1.0  # 默认值
        }
    
    def process_symbol_data(self, symbol: str) -> List[Dict]:
        """处理单个交易对的数据"""
        self.logger.info(f"处理 {symbol} 的影子数据...")
        
        # 加载数据
        symbol_data = self.load_historical_data(symbol, 0, int(time.time() * 1000))
        
        if not symbol_data:
            self.logger.warning(f"未找到 {symbol} 的数据")
            return []
        
        # 获取数据
        prices_df = symbol_data.get('prices', pd.DataFrame())
        ofi_df = symbol_data.get('ofi', pd.DataFrame())
        cvd_df = symbol_data.get('cvd', pd.DataFrame())
        fusion_df = symbol_data.get('fusion', pd.DataFrame())
        
        if prices_df.empty:
            self.logger.warning(f"{symbol} 价格数据为空")
            return []
        
        # 计算Z分数
        if not ofi_df.empty and 'ofi_value' in ofi_df.columns:
            ofi_z_scores = self.calculate_z_scores(ofi_df['ofi_value'])
        else:
            ofi_z_scores = pd.Series([0.0] * len(prices_df), index=prices_df.index)
        
        if not cvd_df.empty and 'cvd' in cvd_df.columns:
            cvd_z_scores = self.calculate_z_scores(cvd_df['cvd'])
        else:
            cvd_z_scores = pd.Series([0.0] * len(prices_df), index=prices_df.index)
        
        # 计算市场指标
        market_metrics = self.calculate_market_metrics(prices_df, ofi_df, cvd_df)
        
        # 处理每个数据点
        signals = []
        trader = self.traders[symbol]
        
        for i in range(len(prices_df)):
            try:
                row = prices_df.iloc[i]
                ts_ms = int(row['ts_ms'])
                price = float(row['price'])
                
                # 获取对应的Z分数
                ofi_z = ofi_z_scores.iloc[i] if i < len(ofi_z_scores) else 0.0
                cvd_z = cvd_z_scores.iloc[i] if i < len(cvd_z_scores) else 0.0
                
                # 处理信号
                signal_data = self.core_algo.process_signal(
                    ts_ms=ts_ms,
                    symbol=symbol,
                    z_ofi=ofi_z,
                    z_cvd=cvd_z,
                    price=price,
                    trade_rate=market_metrics.get('trade_rate', 0.0),
                    realized_vol=market_metrics.get('realized_vol', 0.0),
                    spread_bps=market_metrics.get('spread_bps', 5.0),
                    missing_msgs_rate=0.0001,  # 默认值
                    resync_detected=False,
                    reconnect_detected=False
                )
                
                # 检查退出条件
                exit_signal = trader.check_exit_conditions(price, ts_ms)
                if exit_signal:
                    fill_record = trader.execute_trade(exit_signal)
                    signals.append(fill_record)
                
                # 处理新信号
                if signal_data.confirm:
                    trade_signal = trader.process_signal(signal_data, {
                        'price': price,
                        'atr': market_metrics.get('realized_vol', 0.01)
                    })
                    
                    if trade_signal:
                        fill_record = trader.execute_trade(trade_signal)
                        signals.append(fill_record)
                
                # 记录信号
                signals.append({
                    'timestamp': ts_ms,
                    'symbol': symbol,
                    'signal_score': signal_data.score,
                    'z_ofi': signal_data.z_ofi,
                    'z_cvd': signal_data.z_cvd,
                    'regime': signal_data.regime,
                    'confirm': signal_data.confirm,
                    'gating': signal_data.gating,
                    'div_type': signal_data.div_type
                })
                
            except Exception as e:
                self.logger.error(f"处理 {symbol} 数据点 {i} 时出错: {e}")
                continue
        
        self.logger.info(f"{symbol} 处理完成，生成 {len(signals)} 个信号")
        return signals
    
    def run_shadow_collection(self, output_dir: str):
        """运行影子数据收集"""
        self.logger.info("开始影子数据收集...")
        
        all_signals = {}
        
        for symbol in self.symbols:
            try:
                signals = self.process_symbol_data(symbol)
                all_signals[symbol] = signals
                
                # 保存交易日志
                self.traders[symbol].save_trade_log(output_dir)
                
            except Exception as e:
                self.logger.error(f"处理 {symbol} 时出错: {e}")
                continue
        
        # 保存汇总报告
        self.save_summary_report(all_signals, output_dir)
        
        self.logger.info("影子数据收集完成")
    
    def save_summary_report(self, all_signals: Dict, output_dir: str):
        """保存汇总报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': list(all_signals.keys()),
            'total_signals': sum(len(signals) for signals in all_signals.values()),
            'performance_metrics': {}
        }
        
        # 收集性能指标
        for symbol, trader in self.traders.items():
            metrics = trader.get_performance_metrics()
            report['performance_metrics'][symbol] = metrics
        
        # 保存报告
        report_file = Path(output_dir) / "shadow_collection" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"汇总报告已保存: {report_file}")

def main():
    """测试主函数"""
    # 配置
    symbols = ["BTCUSDT", "ETHUSDT"]
    config = {
        'w_ofi': 0.6,
        'w_cvd': 0.4,
        'z_hi': 2.0,
        'z_mid': 1.0,
        'max_position_size': 1.0,
        'risk_budget': 1000.0,
        'max_trades_per_hour': 10,
        'cooldown_seconds': 60
    }
    
    # 创建收集器
    collector = ShadowDataCollector(symbols, config)
    
    # 运行影子收集
    output_dir = "artifacts/core_algo_v1"
    collector.run_shadow_collection(output_dir)

if __name__ == "__main__":
    main()
