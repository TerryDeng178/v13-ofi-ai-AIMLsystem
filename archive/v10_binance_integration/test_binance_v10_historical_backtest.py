#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基于币安真实历史数据的V10回测系统
"""

import time
import logging
from binance_v10_historical_backtest import BinanceHistoricalData, V10FeatureEngine, V10SignalGenerator, V10Backtester

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_binance_historical_data():
    """测试币安历史数据获取"""
    logger.info("测试币安历史数据获取...")
    
    # 测试不同时间范围的数据获取
    data_fetcher = BinanceHistoricalData("ETHUSDT")
    
    # 获取1天数据
    logger.info("获取1天数据...")
    df_1d = data_fetcher.get_historical_data(days=1, interval="1m")
    logger.info(f"1天数据: {len(df_1d)}条记录")
    
    if not df_1d.empty:
        logger.info(f"价格范围: {df_1d['close'].min():.2f} - {df_1d['close'].max():.2f}")
        logger.info(f"成交量范围: {df_1d['volume'].min():.2f} - {df_1d['volume'].max():.2f}")
    
    # 获取3天数据
    logger.info("获取3天数据...")
    df_3d = data_fetcher.get_historical_data(days=3, interval="1m")
    logger.info(f"3天数据: {len(df_3d)}条记录")
    
    return df_3d

def test_v10_features(df):
    """测试V10特征工程"""
    logger.info("测试V10特征工程...")
    
    feature_engine = V10FeatureEngine()
    df_with_features = feature_engine.create_features(df.copy())
    
    logger.info(f"特征工程完成，总特征数: {len(feature_engine.feature_columns)}")
    logger.info(f"数据形状: {df_with_features.shape}")
    
    # 检查关键特征
    key_features = ['ofi_z', 'cvd_z', 'rsi', 'macd', 'bb_position', 'atr']
    for feature in key_features:
        if feature in df_with_features.columns:
            logger.info(f"{feature}: 均值={df_with_features[feature].mean():.4f}, 标准差={df_with_features[feature].std():.4f}")
    
    return df_with_features

def test_v10_signals(df):
    """测试V10信号生成"""
    logger.info("测试V10信号生成...")
    
    signal_generator = V10SignalGenerator()
    df_with_signals = signal_generator.generate_signals(df.copy())
    
    # 统计信号
    total_signals = (df_with_signals['signal'] != 0).sum()
    long_signals = (df_with_signals['signal'] == 1).sum()
    short_signals = (df_with_signals['signal'] == -1).sum()
    
    logger.info(f"信号统计:")
    logger.info(f"  总信号数: {total_signals}")
    logger.info(f"  多头信号: {long_signals}")
    logger.info(f"  空头信号: {short_signals}")
    logger.info(f"  信号比例: {total_signals/len(df_with_signals)*100:.2f}%")
    
    return df_with_signals

def test_v10_backtest(df):
    """测试V10回测"""
    logger.info("测试V10回测...")
    
    backtester = V10Backtester(initial_capital=10000.0)
    results = backtester.run_backtest(df)
    
    logger.info("回测结果:")
    logger.info(f"  总交易数: {results['total_trades']}")
    logger.info(f"  最终资金: ${results['final_capital']:,.2f}")
    logger.info(f"  总收益: ${results['total_pnl']:,.2f}")
    logger.info(f"  收益率: {results['final_return']:.2%}")
    logger.info(f"  胜率: {results['win_rate']:.2%}")
    logger.info(f"  最大回撤: {results['max_drawdown']:.2%}")
    logger.info(f"  夏普比率: {results['sharpe_ratio']:.4f}")
    logger.info(f"  盈利因子: {results['profit_factor']:.4f}")
    
    return results

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("币安V10历史数据回测系统测试")
    logger.info("=" * 60)
    
    try:
        # 1. 测试数据获取
        logger.info("步骤1: 测试币安历史数据获取...")
        df = test_binance_historical_data()
        
        if df.empty:
            logger.error("未获取到历史数据，测试终止")
            return
        
        # 2. 测试特征工程
        logger.info("步骤2: 测试V10特征工程...")
        df_features = test_v10_features(df)
        
        # 3. 测试信号生成
        logger.info("步骤3: 测试V10信号生成...")
        df_signals = test_v10_signals(df_features)
        
        # 4. 测试回测
        logger.info("步骤4: 测试V10回测...")
        results = test_v10_backtest(df_signals)
        
        # 5. 总结
        logger.info("=" * 60)
        logger.info("测试总结")
        logger.info("=" * 60)
        logger.info("✅ 币安历史数据获取: 成功")
        logger.info("✅ V10特征工程: 成功")
        logger.info("✅ V10信号生成: 成功")
        logger.info("✅ V10回测: 成功")
        logger.info("")
        logger.info("币安V10历史数据回测系统测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
