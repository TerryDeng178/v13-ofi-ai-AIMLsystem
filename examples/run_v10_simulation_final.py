#!/usr/bin/env python3
"""
V10.0 最终版模拟器测试
使用现有V10回测系统进行模拟器数据测试
"""

import sys
import os
import numpy as np
import pandas as pd
import yaml
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def create_simulation_data(duration_seconds=300, seed=42):
    """创建模拟数据"""
    print("="*60)
    print("V10.0 模拟数据生成")
    print("="*60)
    
    # 生成时间序列
    start_time = pd.Timestamp.now()
    timestamps = pd.date_range(start=start_time, periods=duration_seconds*10, freq='100ms')
    
    # 生成价格数据
    np.random.seed(seed)
    price_base = 2500.0
    price_changes = np.random.normal(0, 0.01, len(timestamps))
    prices = price_base + np.cumsum(price_changes)
    prices = np.maximum(prices, 1.0)  # 确保价格为正
    
    # 生成买卖价
    spreads = np.random.uniform(0.1, 0.5, len(timestamps))
    bids = prices - spreads/2
    asks = prices + spreads/2
    
    # 生成成交量
    volumes = np.random.uniform(10, 50, len(timestamps))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ts': timestamps,
        'price': prices,
        'bid': bids,
        'ask': asks,
        'bid_sz': volumes,
        'ask_sz': volumes,
        'volume': volumes * 2
    })
    
    # 添加技术指标
    df['ret_1s'] = df['price'].pct_change()
    df['atr'] = df['ret_1s'].rolling(14).std() * np.sqrt(14)
    df['vwap'] = df['price'].rolling(20).mean()
    df['high'] = df['price'].rolling(20).max()
    df['low'] = df['price'].rolling(20).min()
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"模拟数据生成完成: {len(df)}条记录")
    print(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}")
    print(f"平均价格: {df['price'].mean():.2f}")
    print(f"价格波动: {df['ret_1s'].std():.4f}")
    
    return df

def run_v10_simulation_backtest_final(df, test_id):
    """运行V10模拟器回测"""
    print("\n" + "="*60)
    print(f"V10.0 模拟器回测 - 测试{test_id}")
    print("="*60)
    
    # 创建V10配置
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
    
    # 生成V10深度学习增强信号
    print("生成V10深度学习增强信号...")
    try:
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
            print("\n运行V10深度学习增强策略回测...")
            trades_df = run_strategy(signals_df, config)
            
            if not trades_df.empty:
                print(f"V10深度学习增强策略回测完成: {len(trades_df)}笔交易")
                
                # 计算关键指标
                total_pnl = trades_df['net_pnl'].sum()
                gross_pnl = trades_df['pnl_gross'].sum()
                total_fees = trades_df['fee'].sum()
                total_slippage = trades_df['slippage'].sum()
                
                # 计算胜率
                winning_trades = (trades_df['net_pnl'] > 0).sum()
                total_trades = len(trades_df)
                win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                
                # 计算风险指标
                returns = trades_df['net_pnl'] / config['backtest']['initial_equity_usd']
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(trades_df)) if returns.std() != 0 else 0
                
                # 计算最大回撤
                cumulative_pnl = trades_df['net_pnl'].cumsum()
                peak = cumulative_pnl.expanding(min_periods=1).max()
                drawdown = (cumulative_pnl - peak) / peak
                max_drawdown = abs(drawdown.min()) * 100
                
                # 计算盈亏比
                avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if (trades_df['net_pnl'] > 0).any() else 0
                avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean()) if (trades_df['net_pnl'] < 0).any() else 0
                profit_factor = avg_win / avg_loss if avg_loss != 0 else np.inf
                
                # 计算ROI
                initial_equity = config['backtest']['initial_equity_usd']
                roi = (total_pnl / initial_equity) * 100
                
                # 计算信息比率
                benchmark_return = 0.0
                excess_returns = returns - benchmark_return
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(len(trades_df)) if excess_returns.std() != 0 else 0
                
                # 计算平均持仓时间
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    holding_times = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds()
                    avg_holding_time = holding_times.mean()
                else:
                    avg_holding_time = 0
                
                # 保存结果
                results = {
                    "test_id": test_id,
                    "timestamp": datetime.now(),
                    "total_trades": total_trades,
                    "total_pnl": total_pnl,
                    "gross_pnl": gross_pnl,
                    "total_fees": total_fees,
                    "total_slippage": total_slippage,
                    "win_rate": win_rate,
                    "roi": roi,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "profit_factor": profit_factor,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "information_ratio": information_ratio,
                    "avg_holding_time": avg_holding_time,
                    "signal_count": signal_count,
                    "avg_quality": avg_quality if signal_count > 0 else 0,
                    "avg_ml_pred": avg_ml_pred if signal_count > 0 else 0
                }
                
                print(f"\nV10深度学习增强策略回测结果:")
                print(f"  总交易数: {total_trades}")
                print(f"  总净收益: ${total_pnl:,.2f}")
                print(f"  总毛收益: ${gross_pnl:,.2f}")
                print(f"  总手续费: ${total_fees:,.2f}")
                print(f"  总滑点: ${total_slippage:,.2f}")
                print(f"  胜率: {win_rate:.2f}%")
                print(f"  ROI: {roi:.2f}%")
                print(f"  夏普比率: {sharpe_ratio:.4f}")
                print(f"  最大回撤: {max_drawdown:.2f}%")
                print(f"  盈亏比: {profit_factor:.2f}")
                print(f"  信息比率: {information_ratio:.4f}")
                print(f"  平均持仓时间: {avg_holding_time:.1f}秒")
                
                return results, trades_df, signals_df
            else:
                print("V10深度学习增强策略未产生交易")
                return None, None, signals_df
        else:
            print("V10深度学习增强信号数量为0，跳过策略回测")
            return None, None, signals_df
            
    except Exception as e:
        print(f"V10深度学习增强信号回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_test_report_final(test_id, results, trades_df, signals_df, df):
    """创建最终版测试报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"test_reports_final/test_{test_id}_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n创建测试报告: {report_dir}")
    
    # 保存数据
    if trades_df is not None:
        trades_df.to_csv(f"{report_dir}/trades.csv", index=False)
    if signals_df is not None:
        signals_df.to_csv(f"{report_dir}/signals.csv", index=False)
    df.to_csv(f"{report_dir}/market_data.csv", index=False)
    
    # 创建报告
    report_content = f"""# V10.0 最终版模拟器回测报告 - 测试{test_id}

## 📊 测试概览

**测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**数据来源**: V10.0 模拟数据生成  
**回测状态**: {'成功' if results else '失败'}

## 🎯 关键指标

"""
    
    if results:
        report_content += f"""
### 盈利能力指标
- **总交易数**: {results['total_trades']}
- **总净收益**: ${results['total_pnl']:,.2f}
- **总毛收益**: ${results['gross_pnl']:,.2f}
- **ROI**: {results['roi']:.2f}%
- **胜率**: {results['win_rate']:.2f}%

### 风险指标
- **夏普比率**: {results['sharpe_ratio']:.4f}
- **最大回撤**: {results['max_drawdown']:.2f}%
- **信息比率**: {results['information_ratio']:.4f}
- **盈亏比**: {results['profit_factor']:.2f}

### 成本指标
- **总手续费**: ${results['total_fees']:,.2f}
- **总滑点**: ${results['total_slippage']:,.2f}
- **平均盈利**: ${results['avg_win']:,.2f}
- **平均亏损**: ${results['avg_loss']:,.2f}

### 信号质量
- **信号数量**: {results['signal_count']}
- **平均质量评分**: {results['avg_quality']:.4f}
- **平均ML预测**: {results['avg_ml_pred']:.4f}
- **平均持仓时间**: {results['avg_holding_time']:.1f}秒

## 📈 性能评估

### 盈利能力评估
"""
        
        if results['roi'] > 5:
            report_content += "- [SUCCESS] **优秀**: ROI > 5%，盈利能力强劲\n"
        elif results['roi'] > 0:
            report_content += "- [WARNING] **一般**: ROI > 0%，有盈利但需要优化\n"
        else:
            report_content += "- [FAIL] **不佳**: ROI < 0%，需要大幅改进\n"
        
        report_content += f"""
### 风险控制评估
"""
        
        if results['max_drawdown'] < 5:
            report_content += "- [SUCCESS] **优秀**: 最大回撤 < 5%，风险控制良好\n"
        elif results['max_drawdown'] < 10:
            report_content += "- [WARNING] **一般**: 最大回撤 < 10%，风险控制一般\n"
        else:
            report_content += "- [FAIL] **不佳**: 最大回撤 > 10%，风险控制需要改进\n"
        
        report_content += f"""
### 信号质量评估
"""
        
        if results['avg_quality'] > 0.8:
            report_content += "- [SUCCESS] **优秀**: 平均质量评分 > 0.8，信号质量高\n"
        elif results['avg_quality'] > 0.6:
            report_content += "- [WARNING] **一般**: 平均质量评分 > 0.6，信号质量一般\n"
        else:
            report_content += "- [FAIL] **不佳**: 平均质量评分 < 0.6，信号质量需要改进\n"
        
        report_content += f"""
## 🔧 优化建议

### 基于当前结果的优化方向
"""
        
        # 根据结果生成优化建议
        if results['roi'] < 0:
            report_content += """
1. **盈利能力优化**
   - 调整OFI阈值，提高信号质量
   - 优化止损止盈比例
   - 改进仓位管理策略
"""
        
        if results['max_drawdown'] > 10:
            report_content += """
2. **风险控制优化**
   - 降低单笔交易风险
   - 改进止损策略
   - 增加风险预算控制
"""
        
        if results['signal_count'] < 50:
            report_content += """
3. **信号频率优化**
   - 降低OFI阈值，增加信号数量
   - 优化信号筛选条件
   - 改进实时优化算法
"""
        
        if results['avg_quality'] < 0.6:
            report_content += """
4. **信号质量优化**
   - 改进深度学习模型
   - 优化特征工程
   - 调整信号筛选参数
"""
        
        report_content += f"""
### 下次测试参数建议

基于当前结果，建议下次测试调整以下参数：

```yaml
# 建议的优化参数
risk:
  max_trade_risk_pct: {max(0.005, 0.01 * 0.8)}  # 降低风险
  atr_stop_lo: {max(0.04, 0.06 * 0.8)}  # 收紧止损
  atr_stop_hi: {min(3.0, 2.5 * 1.2)}  # 提高止盈

signals:
  ofi_z_min: {max(1.0, 1.2 * 0.9)}  # 降低OFI阈值
  min_signal_strength: {max(1.2, 1.6 * 0.9)}  # 降低强度要求
  min_confidence: {max(0.6, 0.8 * 0.9)}  # 降低置信度要求

sizing:
  k_ofi: {min(1.0, 0.7 * 1.2)}  # 提高仓位倍数
  size_max_usd: {min(500000, 300000 * 1.2)}  # 提高最大仓位
```

## 📊 数据文件

- `trades.csv`: 交易记录
- `signals.csv`: 信号数据
- `market_data.csv`: 市场数据

## 🎯 下次测试计划

1. **参数优化**: 根据当前结果调整参数
2. **模型改进**: 优化深度学习模型
3. **特征工程**: 改进特征选择和工程
4. **风险控制**: 加强风险管理和控制
5. **性能监控**: 增加实时性能监控

---
**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**状态**: {'成功' if results else '失败'}
"""
    else:
        report_content += """
## [FAIL] 测试失败

本次测试未能成功完成，可能的原因：
1. 信号生成失败
2. 策略执行失败
3. 数据质量问题

## 🔧 故障排除

1. **检查数据质量**: 确保市场数据和OFI数据质量
2. **验证信号生成**: 检查信号生成逻辑
3. **调试策略执行**: 检查策略执行流程
4. **优化参数设置**: 调整配置参数

## 🎯 下次测试计划

1. **数据质量检查**: 确保数据完整性
2. **参数调整**: 优化配置参数
3. **模型验证**: 检查深度学习模型
4. **流程优化**: 改进测试流程

---
**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**状态**: 失败
"""
    
    # 保存报告
    with open(f"{report_dir}/report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"测试报告已保存到: {report_dir}/report.md")
    return report_dir

def main():
    """主函数"""
    print("V10.0 最终版模拟器数据生成和深度学习回测系统")
    print("="*60)
    
    # 创建测试报告目录
    os.makedirs("test_reports_final", exist_ok=True)
    
    # 运行多次测试
    for test_id in range(1, 6):  # 运行5次测试
        print(f"\n{'='*60}")
        print(f"开始测试 {test_id}/5")
        print(f"{'='*60}")
        
        try:
            # 1. 生成模拟数据
            print(f"\n步骤1: 生成V10模拟数据 (测试{test_id})")
            df = create_simulation_data(
                duration_seconds=300,  # 5分钟数据
                seed=42 + test_id
            )
            
            # 2. 运行V10回测
            print(f"\n步骤2: 运行V10深度学习回测 (测试{test_id})")
            results, trades_df, signals_df = run_v10_simulation_backtest_final(
                df, test_id
            )
            
            # 3. 创建测试报告
            print(f"\n步骤3: 创建测试报告 (测试{test_id})")
            report_dir = create_test_report_final(
                test_id, results, trades_df, signals_df, df
            )
            
            # 4. 评估结果
            if results:
                print(f"\n测试{test_id}结果评估:")
                print(f"  ROI: {results['roi']:.2f}%")
                print(f"  胜率: {results['win_rate']:.2f}%")
                print(f"  最大回撤: {results['max_drawdown']:.2f}%")
                print(f"  夏普比率: {results['sharpe_ratio']:.4f}")
                print(f"  交易数: {results['total_trades']}")
                
                # 判断是否达到满意结果
                if (results['roi'] > 5 and 
                    results['win_rate'] > 50 and 
                    results['max_drawdown'] < 10 and 
                    results['total_trades'] > 20):
                    print(f"\n[SUCCESS] 测试{test_id}达到满意结果！")
                    print(f"ROI: {results['roi']:.2f}% > 5%")
                    print(f"胜率: {results['win_rate']:.2f}% > 50%")
                    print(f"最大回撤: {results['max_drawdown']:.2f}% < 10%")
                    print(f"交易数: {results['total_trades']} > 20")
                    break
                else:
                    print(f"\n[WARNING] 测试{test_id}结果需要优化")
            else:
                print(f"\n[FAIL] 测试{test_id}失败")
            
        except Exception as e:
            print(f"\n[ERROR] 测试{test_id}出现异常: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("V10.0 最终版模拟器测试完成")
    print(f"{'='*60}")
    print("所有测试报告已保存到 test_reports_final/ 目录")
    print("请查看各测试报告了解详细结果和优化建议")

if __name__ == "__main__":
    main()
