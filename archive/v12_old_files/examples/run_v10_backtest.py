#!/usr/bin/env python3
"""
V10.0 深度学习集成回测脚本
使用深度学习模型进行策略回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入V10模块
try:
    from src.signals_v10_deep_learning import gen_signals_v10_deep_learning_enhanced, gen_signals_v10_real_time_optimized
    from src.strategy import run_strategy
    from src.backtest import run_backtest
    from src.data import load_data
    from src.features import add_feature_block
except ImportError as e:
    print(f"导入错误: {e}")
    # 尝试直接导入
    try:
        sys.path.append('src')
        from signals_v10_deep_learning import gen_signals_v10_deep_learning_enhanced, gen_signals_v10_real_time_optimized
        from strategy import run_strategy
        from backtest import run_backtest
        from data import load_data
        from features import add_feature_block
        print("使用直接导入方式成功")
    except ImportError as e2:
        print(f"直接导入也失败: {e2}")
        sys.exit(1)

def load_v10_config():
    """加载V10配置"""
    config_path = "config/params_v10_deep_learning.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def create_v10_test_data(n_samples: int = 5000) -> pd.DataFrame:
    """创建V10测试数据"""
    np.random.seed(42)
    
    # 生成价格数据
    price_base = 100
    price_changes = np.random.randn(n_samples) * 0.01
    prices = price_base + np.cumsum(price_changes)
    
    # 生成订单簿数据
    bid_prices = prices - np.random.uniform(0.01, 0.05, n_samples)
    ask_prices = prices + np.random.uniform(0.01, 0.05, n_samples)
    
    # 生成成交量数据
    volumes = np.random.randint(100, 1000, n_samples)
    
    # 生成订单簿深度
    bid_sizes = np.random.randint(100, 500, (n_samples, 3))
    ask_sizes = np.random.randint(100, 500, (n_samples, 3))
    
    # 生成技术指标
    ofi_z = np.random.randn(n_samples) * 2
    cvd_z = np.random.randn(n_samples) * 2
    ret_1s = np.random.randn(n_samples) * 0.001
    atr = np.random.uniform(0.01, 0.05, n_samples)
    vwap = prices + np.random.randn(n_samples) * 0.005
    
    # 生成信号质量标签
    signal_quality = np.random.uniform(0, 1, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': prices,
        'volume': volumes,
        'bid1': bid_prices,
        'ask1': ask_prices,
        'bid1_size': bid_sizes[:, 0],
        'ask1_size': ask_sizes[:, 0],
        'bid2': bid_prices - 0.01,
        'ask2': ask_prices + 0.01,
        'bid2_size': bid_sizes[:, 1],
        'ask2_size': ask_sizes[:, 1],
        'bid3': bid_prices - 0.02,
        'ask3': ask_prices + 0.02,
        'bid3_size': bid_sizes[:, 2],
        'ask3_size': ask_sizes[:, 2],
        'high': prices + np.random.uniform(0, 0.05, n_samples),
        'low': prices - np.random.uniform(0, 0.05, n_samples),
        'ofi_z': ofi_z,
        'cvd_z': cvd_z,
        'ret_1s': ret_1s,
        'atr': atr,
        'vwap': vwap,
        'signal_quality': signal_quality
    })
    
    return df

def run_v10_deep_learning_backtest():
    """运行V10深度学习回测"""
    print("=" * 60)
    print("V10.0 深度学习集成回测开始")
    print("=" * 60)
    
    # 加载配置
    config = load_v10_config()
    if config is None:
        print("无法加载V10配置，使用默认配置")
        config = {
            "risk": {
                "max_trade_risk_pct": 0.01,
                "daily_drawdown_stop_pct": 0.08,
                "atr_stop_lo": 0.06,
                "atr_stop_hi": 2.5,
                "min_tick_sl_mult": 2,
                "time_exit_seconds_min": 30,
                "time_exit_seconds_max": 300,
                "slip_bps_budget_frac": 0.15,
                "fee_bps": 0.2
            },
            "signals": {
                "deep_learning_enhanced": {
                    "ofi_z_min": 1.2,
                    "min_ret": 0.0,
                    "min_signal_strength": 1.6,
                    "min_ml_prediction": 0.8,
                    "min_uncertainty": 0.1,
                    "max_uncertainty": 0.9,
                    "thin_book_spread_bps_max": 4.0,
                    "adaptive_thresholds": True,
                    "quantile_window": 1000,
                    "momentum_quantile_hi": 0.70,
                    "momentum_quantile_lo": 0.30
                },
                "real_time_optimization": {
                    "enabled": True,
                    "update_frequency": 5,
                    "performance_window": 30,
                    "adaptation_rate": 0.15,
                    "min_performance_threshold": 0.7
                }
            },
            "sizing": {
                "k_ofi": 0.7,
                "size_max_usd": 300000
            },
            "execution": {
                "ioc": True,
                "fok": False,
                "slippage_budget_check": False,
                "max_slippage_bps": 8.0,
                "session_window_minutes": 60,
                "reject_on_budget_exceeded": False
            },
            "backtest": {
                "initial_equity_usd": 100000,
                "contract_multiplier": 1.0,
                "seed": 42
            }
        }
    
    # 创建测试数据
    print("创建V10测试数据...")
    df = create_v10_test_data(5000)
    print(f"数据创建完成: {df.shape}")
    
    # 添加基础特征
    print("添加基础特征...")
    df = add_feature_block(df, config)
    print(f"特征添加完成: {df.shape}")
    
    # 运行深度学习增强信号回测
    print("\n" + "=" * 40)
    print("运行V10深度学习增强信号回测")
    print("=" * 40)
    
    try:
        # 生成深度学习增强信号
        signals_df = gen_signals_v10_deep_learning_enhanced(df, config)
        print(f"深度学习增强信号生成完成: {signals_df.shape}")
        
        # 统计信号
        signal_count = signals_df['sig_side'].abs().sum()
        long_signals = (signals_df['sig_side'] == 1).sum()
        short_signals = (signals_df['sig_side'] == -1).sum()
        
        print(f"深度学习增强信号统计:")
        print(f"  总信号数: {signal_count}")
        print(f"  多头信号: {long_signals}")
        print(f"  空头信号: {short_signals}")
        
        if signal_count > 0:
            avg_quality = signals_df[signals_df['sig_side'] != 0]['quality_score'].mean()
            avg_ml_pred = signals_df[signals_df['sig_side'] != 0]['ml_prediction'].mean()
            print(f"  平均质量评分: {avg_quality:.4f}")
            print(f"  平均ML预测: {avg_ml_pred:.4f}")
        
        # 运行策略回测
        if signal_count > 0:
            print("\n运行深度学习增强策略回测...")
            trades_df = run_strategy(signals_df, config)
            
            if not trades_df.empty:
                print(f"深度学习增强策略回测完成: {len(trades_df)}笔交易")
                
                # 计算基础指标
                total_pnl = trades_df['net_pnl'].sum()
                gross_pnl = trades_df['pnl_gross'].sum()
                total_fees = trades_df['fee'].sum()
                total_slippage = trades_df['slippage'].sum()
                
                print(f"\n深度学习增强策略回测结果:")
                print(f"  总净收益: ${total_pnl:,.2f}")
                print(f"  总毛收益: ${gross_pnl:,.2f}")
                print(f"  总手续费: ${total_fees:,.2f}")
                print(f"  总滑点: ${total_slippage:,.2f}")
                
                # 计算胜率
                winning_trades = (trades_df['net_pnl'] > 0).sum()
                total_trades = len(trades_df)
                win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                
                print(f"  胜率: {win_rate:.2f}%")
                print(f"  平均每笔收益: ${trades_df['net_pnl'].mean():.2f}")
                print(f"  最大单笔收益: ${trades_df['net_pnl'].max():.2f}")
                print(f"  最大单笔亏损: ${trades_df['net_pnl'].min():.2f}")
                
                # 计算风险指标
                returns = trades_df['net_pnl'] / config['backtest']['initial_equity_usd']
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(trades_df)) if returns.std() != 0 else 0
                
                print(f"  夏普比率: {sharpe_ratio:.4f}")
                
                # 计算最大回撤
                cumulative_pnl = trades_df['net_pnl'].cumsum()
                peak = cumulative_pnl.expanding(min_periods=1).max()
                drawdown = (cumulative_pnl - peak) / peak
                max_drawdown = abs(drawdown.min()) * 100
                
                print(f"  最大回撤: {max_drawdown:.2f}%")
            else:
                print("深度学习增强策略未产生交易")
        else:
            print("深度学习增强信号数量为0，跳过策略回测")
            
    except Exception as e:
        print(f"深度学习增强信号回测失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 运行实时优化信号回测
    print("\n" + "=" * 40)
    print("运行V10实时优化信号回测")
    print("=" * 40)
    
    try:
        # 生成实时优化信号
        rt_signals_df = gen_signals_v10_real_time_optimized(df, config)
        print(f"实时优化信号生成完成: {rt_signals_df.shape}")
        
        # 统计信号
        rt_signal_count = rt_signals_df['sig_side'].abs().sum()
        rt_long_signals = (rt_signals_df['sig_side'] == 1).sum()
        rt_short_signals = (rt_signals_df['sig_side'] == -1).sum()
        
        print(f"实时优化信号统计:")
        print(f"  总信号数: {rt_signal_count}")
        print(f"  多头信号: {rt_long_signals}")
        print(f"  空头信号: {rt_short_signals}")
        
        if rt_signal_count > 0:
            avg_quality = rt_signals_df[rt_signals_df['sig_side'] != 0]['quality_score'].mean()
            avg_rt_score = rt_signals_df[rt_signals_df['sig_side'] != 0]['real_time_score'].mean()
            print(f"  平均质量评分: {avg_quality:.4f}")
            print(f"  平均实时评分: {avg_rt_score:.4f}")
        
        # 运行策略回测
        if rt_signal_count > 0:
            print("\n运行实时优化策略回测...")
            rt_trades_df = run_strategy(rt_signals_df, config)
            
            if not rt_trades_df.empty:
                print(f"实时优化策略回测完成: {len(rt_trades_df)}笔交易")
                
                # 计算基础指标
                rt_total_pnl = rt_trades_df['net_pnl'].sum()
                rt_gross_pnl = rt_trades_df['pnl_gross'].sum()
                rt_total_fees = rt_trades_df['fee'].sum()
                rt_total_slippage = rt_trades_df['slippage'].sum()
                
                print(f"\n实时优化策略回测结果:")
                print(f"  总净收益: ${rt_total_pnl:,.2f}")
                print(f"  总毛收益: ${rt_gross_pnl:,.2f}")
                print(f"  总手续费: ${rt_total_fees:,.2f}")
                print(f"  总滑点: ${rt_total_slippage:,.2f}")
                
                # 计算胜率
                rt_winning_trades = (rt_trades_df['net_pnl'] > 0).sum()
                rt_total_trades = len(rt_trades_df)
                rt_win_rate = rt_winning_trades / rt_total_trades * 100 if rt_total_trades > 0 else 0
                
                print(f"  胜率: {rt_win_rate:.2f}%")
                print(f"  平均每笔收益: ${rt_trades_df['net_pnl'].mean():.2f}")
                print(f"  最大单笔收益: ${rt_trades_df['net_pnl'].max():.2f}")
                print(f"  最大单笔亏损: ${rt_trades_df['net_pnl'].min():.2f}")
                
                # 计算风险指标
                rt_returns = rt_trades_df['net_pnl'] / config['backtest']['initial_equity_usd']
                rt_sharpe_ratio = rt_returns.mean() / rt_returns.std() * np.sqrt(len(rt_trades_df)) if rt_returns.std() != 0 else 0
                
                print(f"  夏普比率: {rt_sharpe_ratio:.4f}")
                
                # 计算最大回撤
                rt_cumulative_pnl = rt_trades_df['net_pnl'].cumsum()
                rt_peak = rt_cumulative_pnl.expanding(min_periods=1).max()
                rt_drawdown = (rt_cumulative_pnl - rt_peak) / rt_peak
                rt_max_drawdown = abs(rt_drawdown.min()) * 100
                
                print(f"  最大回撤: {rt_max_drawdown:.2f}%")
            else:
                print("实时优化策略未产生交易")
        else:
            print("实时优化信号数量为0，跳过策略回测")
            
    except Exception as e:
        print(f"实时优化信号回测失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("V10.0 深度学习集成回测完成")
    print("=" * 60)

def run_v10_comprehensive_backtest():
    """运行V10综合回测"""
    print("=" * 60)
    print("V10.0 综合回测开始")
    print("=" * 60)
    
    # 加载配置
    config = load_v10_config()
    if config is None:
        print("无法加载V10配置，使用默认配置")
        return
    
    # 创建测试数据
    print("创建V10测试数据...")
    df = create_v10_test_data(10000)  # 更大的数据集
    print(f"数据创建完成: {df.shape}")
    
    # 添加基础特征
    print("添加基础特征...")
    df = add_feature_block(df, config)
    print(f"特征添加完成: {df.shape}")
    
    # 运行综合回测
    print("\n运行V10综合回测...")
    try:
        # 使用深度学习增强信号
        signals_df = gen_signals_v10_deep_learning_enhanced(df, config)
        
        # 运行回测
        results = run_backtest(signals_df, config)
        
        print(f"\nV10综合回测结果:")
        print(f"  总交易数: {results.get('total_trades', 0)}")
        print(f"  总净收益: ${results.get('total_pnl', 0):,.2f}")
        print(f"  胜率: {results.get('win_rate', 0):.2f}%")
        print(f"  夏普比率: {results.get('sharpe_ratio', 0):.4f}")
        print(f"  最大回撤: {results.get('max_drawdown', 0):.2f}%")
        print(f"  信息比率: {results.get('information_ratio', 0):.4f}")
        
    except Exception as e:
        print(f"V10综合回测失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("V10.0 深度学习集成回测系统")
    print("=" * 60)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 运行深度学习回测
    run_v10_deep_learning_backtest()
    
    # 运行综合回测
    print("\n" + "=" * 60)
    print("运行V10综合回测")
    print("=" * 60)
    run_v10_comprehensive_backtest()

if __name__ == "__main__":
    main()
