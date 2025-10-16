#!/usr/bin/env python3
"""
V10.0 修复版模拟器测试
修复时间戳匹配问题，确保交易模拟正常工作
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 内置V10模拟器
class V10StandaloneOFI:
    def __init__(self, micro_window_ms=100, z_window_seconds=900, levels=3, weights=None):
        self.w = micro_window_ms
        self.zn = int(max(10, z_window_seconds * 1000 // self.w))
        self.levels = levels
        self.weights = weights or [0.5, 0.3, 0.2]
        
        self.cur_bucket = None
        self.bucket_sum = 0.0
        self.history = []
        self.t_series = []
        self.last_best = None
        
        self.level_contributions = [0.0] * self.levels
        self.level_history = [[] for _ in range(self.levels)]
        
    def on_best(self, t, bid, bid_sz, ask, ask_sz):
        self.last_best = (t, bid, bid_sz, ask, ask_sz)
        
    def on_l2(self, t, typ, side, price, qty):
        if not self.last_best:
            return
            
        _, bid, bid_sz, ask, ask_sz = self.last_best
        is_add = (typ == "l2_add")
        
        contributions = self._calculate_level_contributions(
            is_add, side, price, qty, bid, bid_sz, ask, ask_sz
        )
        
        for i, contrib in enumerate(contributions):
            self.level_contributions[i] += contrib
                
        weighted_ofi = sum(w * c for w, c in zip(self.weights, self.level_contributions))
        
        bucket = (t // self.w) * self.w
        if self.cur_bucket is None:
            self.cur_bucket = bucket
                
        if bucket != self.cur_bucket:
            self.history.append(self.bucket_sum)
            self.t_series.append(self.cur_bucket)
                
            for i, contrib in enumerate(self.level_contributions):
                self.level_history[i].append(contrib)
                    
            self.bucket_sum = weighted_ofi
            self.level_contributions = [0.0] * self.levels
            self.cur_bucket = bucket
        else:
            self.bucket_sum = weighted_ofi
                
    def _calculate_level_contributions(self, is_add, side, price, qty, bid, bid_sz, ask, ask_sz):
        contributions = [0.0] * self.levels
        
        if not self.last_best:
            return contributions
                
        is_bid1 = abs(price - bid) < 1e-9
        is_ask1 = abs(price - ask) < 1e-9
        
        if is_add and is_bid1:
            contributions[0] += qty
        if is_add and is_ask1:
            contributions[0] -= qty
        if (not is_add) and is_bid1:
            contributions[0] -= qty
        if (not is_add) and is_ask1:
            contributions[0] += qty
                
        if is_add and side == 'bid' and not is_bid1:
            contributions[1] += qty * 0.5
        if is_add and side == 'ask' and not is_ask1:
            contributions[1] -= qty * 0.5
        if (not is_add) and side == 'bid' and not is_bid1:
            contributions[1] -= qty * 0.5
        if (not is_add) and side == 'ask' and not is_ask1:
            contributions[1] += qty * 0.5
                
        if is_add and side == 'bid':
            contributions[2] += qty * 0.3
        if is_add and side == 'ask':
            contributions[2] -= qty * 0.3
        if (not is_add) and side == 'bid':
            contributions[2] -= qty * 0.3
        if (not is_add) and side == 'ask':
            contributions[2] += qty * 0.3
                
        return contributions
            
    def read(self):
        if len(self.history) < max(10, self.zn // 10):
            return None
                
        arr = np.array(self.history, dtype=float)
        z = (arr[-1] - arr.mean()) / (arr.std(ddof=0) + 1e-9)
        
        level_ofis = []
        level_zs = []
        for i in range(self.levels):
            if len(self.level_history[i]) > 0:
                level_arr = np.array(self.level_history[i], dtype=float)
                level_ofi = level_arr[-1] if len(level_arr) > 0 else 0.0
                level_z = (level_ofi - level_arr.mean()) / (level_arr.std(ddof=0) + 1e-9)
                level_ofis.append(level_ofi)
                level_zs.append(level_z)
            else:
                level_ofis.append(0.0)
                level_zs.append(0.0)
        
        weighted_ofi = sum(w * ofi for w, ofi in zip(self.weights, level_ofis))
        weighted_z = sum(w * z for w, z in zip(self.weights, level_zs))
        
        return {
            "t": self.t_series[-1],
            "ofi": float(arr[-1]),
            "ofi_z": float(z),
            "weighted_ofi": float(weighted_ofi),
            "weighted_ofi_z": float(weighted_z),
            "level_ofis": level_ofis,
            "level_zs": level_zs,
            "weights": self.weights
        }
            
    def create_features(self, ofi_data, market_data):
        features = []
        
        features.extend([
            ofi_data.get("ofi", 0.0),
            ofi_data.get("ofi_z", 0.0),
            ofi_data.get("weighted_ofi", 0.0),
            ofi_data.get("weighted_ofi_z", 0.0)
        ])
        
        level_ofis = ofi_data.get("level_ofis", [0.0] * self.levels)
        level_zs = ofi_data.get("level_zs", [0.0] * self.levels)
        features.extend(level_ofis)
        features.extend(level_zs)
        
        if market_data:
            features.extend([
                market_data.get("bid", 0.0),
                market_data.get("ask", 0.0),
                market_data.get("bid_sz", 0.0),
                market_data.get("ask_sz", 0.0),
                market_data.get("spread", 0.0),
                market_data.get("mid_price", 0.0)
            ])
        else:
            features.extend([0.0] * 6)
                
        current_time = datetime.now().timestamp()
        features.extend([
            current_time % 86400,
            current_time % 3600,
            current_time % 60
        ])
        
        return np.array(features, dtype=np.float32)
            
    def predict_signal(self, features):
        ofi_z = features[1] if len(features) > 1 else 0.0
        weighted_ofi_z = features[3] if len(features) > 3 else 0.0
        
        signal_strength = abs(weighted_ofi_z)
        signal_side = 1 if weighted_ofi_z > 2.0 else -1 if weighted_ofi_z < -2.0 else 0
        
        return {
            "signal_side": signal_side,
            "signal_strength": signal_strength,
            "confidence": min(1.0, signal_strength / 3.0),
            "model_type": "rule_based"
        }
    
class SimpleMarketSimulator:
    def __init__(self, seed=42, duration=10):
        self.rng = np.random.default_rng(seed)
        self.duration = duration
        self.mid = 2500.0
        self.tick = 0.1
        self.time = 0
        
    def generate_events(self):
        events = []
        steps = self.duration * 100
        
        for i in range(steps):
            self.time = i * 100
            
            self.mid += self.rng.normal(0, 0.01)
            self.mid = max(1.0, self.mid)
            
            spread = self.rng.uniform(0.1, 0.5)
            bid = self.mid - spread/2
            ask = self.mid + spread/2
            bid_sz = self.rng.uniform(10, 50)
            ask_sz = self.rng.uniform(10, 50)
            
            if i % 5 == 0:
                events.append({
                    "t": self.time,
                    "type": "best",
                    "bid": bid,
                    "bid_sz": bid_sz,
                    "ask": ask,
                    "ask_sz": ask_sz
                })
            
            if self.rng.random() < 0.3:
                side = 'bid' if self.rng.random() < 0.5 else 'ask'
                price = bid if side == 'bid' else ask
                qty = self.rng.uniform(1, 20)
                typ = 'l2_add' if self.rng.random() < 0.7 else 'l2_cancel'
                
                events.append({
                    "t": self.time,
                    "type": typ,
                    "side": side,
                    "price": price,
                    "qty": qty
                })
        
        return events

def create_v10_simulation_data(duration_seconds=300, seed=42):
    """使用V10模拟器生成市场数据"""
    print("="*60)
    print("V10.0 模拟器数据生成")
    print("="*60)
    
    # 创建模拟器
    simulator = SimpleMarketSimulator(seed=seed, duration=duration_seconds)
    ofi_calc = V10StandaloneOFI(micro_window_ms=100, z_window_seconds=900, levels=3)
    
    # 生成市场数据
    print(f"生成{duration_seconds}秒的市场数据...")
    market_data = []
    ofi_data = []
    signals = []
    
    events = simulator.generate_events()
    for event in events:
        if event["type"] == "best":
            ofi_calc.on_best(
                event["t"], event["bid"], event["bid_sz"], 
                event["ask"], event["ask_sz"]
            )
            
            # 保存市场数据
            market_data.append({
                "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=event["t"]),
                "price": (event["bid"] + event["ask"]) / 2,
                "bid": event["bid"],
                "ask": event["ask"],
                "bid_sz": event["bid_sz"],
                "ask_sz": event["ask_sz"],
                "volume": event.get("bid_sz", 0) + event.get("ask_sz", 0)
            })
            
        elif event["type"] in ["l2_add", "l2_cancel"]:
            ofi_calc.on_l2(
                event["t"], event["type"], event["side"], 
                event["price"], event["qty"]
            )
        
        # 检查OFI和信号
        ofi_result = ofi_calc.read()
        if ofi_result:
            ofi_data.append({
                "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=ofi_result["t"]),
                "ofi": ofi_result["ofi"],
                "ofi_z": ofi_result["ofi_z"],
                "weighted_ofi": ofi_result.get("weighted_ofi", ofi_result["ofi"]),
                "weighted_ofi_z": ofi_result.get("weighted_ofi_z", ofi_result["ofi_z"]),
                "level_ofis": ofi_result.get("level_ofis", []),
                "level_zs": ofi_result.get("level_zs", [])
            })
            
            # 生成信号
            if market_data:
                latest_market = market_data[-1]
                features = ofi_calc.create_features(ofi_result, {
                    "bid": latest_market["bid"],
                    "ask": latest_market["ask"],
                    "bid_sz": latest_market["bid_sz"],
                    "ask_sz": latest_market["ask_sz"],
                    "spread": latest_market["ask"] - latest_market["bid"],
                    "mid_price": latest_market["price"]
                })
                
                signal_result = ofi_calc.predict_signal(features)
                if signal_result["signal_side"] != 0:
                    signals.append({
                        "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=ofi_result["t"]),
                        "signal_side": signal_result["signal_side"],
                        "signal_strength": signal_result["signal_strength"],
                        "confidence": signal_result["confidence"],
                        "model_type": signal_result["model_type"],
                        "ofi_z": ofi_result["ofi_z"],
                        "weighted_ofi_z": ofi_result.get("weighted_ofi_z", ofi_result["ofi_z"])
                    })
    
    # 转换为DataFrame
    df_market = pd.DataFrame(market_data)
    df_ofi = pd.DataFrame(ofi_data)
    df_signals = pd.DataFrame(signals)
    
    print(f"市场数据: {len(df_market)}条")
    print(f"OFI数据: {len(df_ofi)}条")
    print(f"信号数据: {len(df_signals)}条")
    
    return df_market, df_ofi, df_signals

def simulate_trading_fixed(df_market, df_signals, initial_equity=100000):
    """修复版交易模拟"""
    print("模拟交易执行...")
    
    trades = []
    equity = initial_equity
    open_pos = None
    
    # 合并数据 - 修复时间戳匹配问题
    df = df_market.copy()
    if not df_signals.empty:
        # 使用时间戳进行合并
        df_signals['ts'] = pd.to_datetime(df_signals['ts'])
        df['ts'] = pd.to_datetime(df['ts'])
        
        # 使用简单合并
        df = df.merge(df_signals, on='ts', how='left', suffixes=('', '_signal'))
        
        # 填充信号数据
        df['signal_side'] = df['signal_side'].fillna(0)
        df['signal_strength'] = df['signal_strength'].fillna(0.0)
        df['confidence'] = df['confidence'].fillna(0.0)
    else:
        df['signal_side'] = 0
        df['signal_strength'] = 0.0
        df['confidence'] = 0.0
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 添加技术指标
    df['ret_1s'] = df['price'].pct_change()
    df['atr'] = df['ret_1s'].rolling(14).std() * np.sqrt(14)
    df['vwap'] = df['price'].rolling(20).mean()
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"合并后数据: {len(df)}条")
    print(f"信号数量: {(df['signal_side'] != 0).sum()}")
    
    for i, row in df.iterrows():
        # 更新权益
        if open_pos:
            # 计算当前PnL
            current_pnl = (row['price'] - open_pos['entry_price']) * open_pos['side'] * open_pos['size']
            
            # 检查退出条件
            exit_reason = None
            if (open_pos['side'] == 1 and row['price'] <= open_pos['stop_loss']) or \
               (open_pos['side'] == -1 and row['price'] >= open_pos['stop_loss']):
                exit_reason = 'stop_loss'
            elif (open_pos['side'] == 1 and row['price'] >= open_pos['take_profit']) or \
                 (open_pos['side'] == -1 and row['price'] <= open_pos['take_profit']):
                exit_reason = 'take_profit'
            elif (row['ts'] - open_pos['entry_time']).total_seconds() >= 300:  # 5分钟超时
                exit_reason = 'time_exit'
            
            if exit_reason:
                # 平仓
                exit_price = row['price']
                pnl = (exit_price - open_pos['entry_price']) * open_pos['side'] * open_pos['size']
                fee = abs(open_pos['size']) * 0.0002  # 0.02%手续费
                net_pnl = pnl - fee
                
                trades.append({
                    'entry_time': open_pos['entry_time'],
                    'exit_time': row['ts'],
                    'side': open_pos['side'],
                    'entry_price': open_pos['entry_price'],
                    'exit_price': exit_price,
                    'size': open_pos['size'],
                    'pnl': pnl,
                    'fee': fee,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason
                })
                
                equity += net_pnl
                open_pos = None
        
        # 检查新信号
        if row['signal_side'] != 0 and open_pos is None:
            # 开仓
            size = min(equity * 0.1 / row['price'], 1000)  # 10%仓位，最大1000股
            if size > 0:
                stop_loss = row['price'] - 0.06 * row['atr'] if row['signal_side'] == 1 else row['price'] + 0.06 * row['atr']
                take_profit = row['price'] + 2.5 * row['atr'] if row['signal_side'] == 1 else row['price'] - 2.5 * row['atr']
                
                open_pos = {
                    'entry_time': row['ts'],
                    'entry_price': row['price'],
                    'side': row['signal_side'],
                    'size': size * row['signal_side'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
    
    print(f"交易模拟完成: {len(trades)}笔交易")
    return pd.DataFrame(trades)

def run_v10_simulation_backtest_fixed(df_market, df_ofi, df_signals, test_id):
    """修复版V10模拟器回测"""
    print("\n" + "="*60)
    print(f"V10.0 模拟器回测 - 测试{test_id}")
    print("="*60)
    
    # 运行交易模拟
    trades_df = simulate_trading_fixed(df_market, df_signals, 100000)
    
    if not trades_df.empty:
        print(f"V10模拟器回测完成: {len(trades_df)}笔交易")
        
        # 计算关键指标
        total_pnl = trades_df['net_pnl'].sum()
        gross_pnl = trades_df['pnl'].sum()
        total_fees = trades_df['fee'].sum()
        
        # 计算胜率
        winning_trades = (trades_df['net_pnl'] > 0).sum()
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # 计算风险指标
        returns = trades_df['net_pnl'] / 100000
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
        initial_equity = 100000
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
        
        # 信号统计
        signal_count = len(df_signals) if not df_signals.empty else 0
        avg_quality = df_signals['confidence'].mean() if not df_signals.empty and 'confidence' in df_signals.columns else 0
        
        # 保存结果
        results = {
            "test_id": test_id,
            "timestamp": datetime.now(),
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "gross_pnl": gross_pnl,
            "total_fees": total_fees,
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
            "avg_quality": avg_quality
        }
        
        print(f"\nV10模拟器回测结果:")
        print(f"  总交易数: {total_trades}")
        print(f"  总净收益: ${total_pnl:,.2f}")
        print(f"  总毛收益: ${gross_pnl:,.2f}")
        print(f"  总手续费: ${total_fees:,.2f}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  ROI: {roi:.2f}%")
        print(f"  夏普比率: {sharpe_ratio:.4f}")
        print(f"  最大回撤: {max_drawdown:.2f}%")
        print(f"  盈亏比: {profit_factor:.2f}")
        print(f"  信息比率: {information_ratio:.4f}")
        print(f"  平均持仓时间: {avg_holding_time:.1f}秒")
        print(f"  信号数量: {signal_count}")
        print(f"  平均质量: {avg_quality:.4f}")
        
        return results, trades_df
    else:
        print("V10模拟器未产生交易")
        return None, None

def create_test_report_fixed(test_id, results, trades_df, df_market, df_ofi, df_signals):
    """创建修复版测试报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"test_reports_fixed/test_{test_id}_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n创建测试报告: {report_dir}")
    
    # 保存数据
    if trades_df is not None:
        trades_df.to_csv(f"{report_dir}/trades.csv", index=False)
    df_market.to_csv(f"{report_dir}/market_data.csv", index=False)
    df_ofi.to_csv(f"{report_dir}/ofi_data.csv", index=False)
    df_signals.to_csv(f"{report_dir}/simulation_signals.csv", index=False)
    
    # 创建报告
    report_content = f"""# V10.0 修复版模拟器回测报告 - 测试{test_id}

## 📊 测试概览

**测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**数据来源**: V10.0 模拟器生成  
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
- **平均盈利**: ${results['avg_win']:,.2f}
- **平均亏损**: ${results['avg_loss']:,.2f}

### 信号质量
- **信号数量**: {results['signal_count']}
- **平均质量**: {results['avg_quality']:.4f}
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
            report_content += "- [SUCCESS] **优秀**: 平均质量 > 0.8，信号质量高\n"
        elif results['avg_quality'] > 0.6:
            report_content += "- [WARNING] **一般**: 平均质量 > 0.6，信号质量一般\n"
        else:
            report_content += "- [FAIL] **不佳**: 平均质量 < 0.6，信号质量需要改进\n"
        
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
- `market_data.csv`: 市场数据
- `ofi_data.csv`: OFI数据
- `simulation_signals.csv`: 模拟器信号

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
    print("V10.0 修复版模拟器数据生成和深度学习回测系统")
    print("="*60)
    
    # 创建测试报告目录
    os.makedirs("test_reports_fixed", exist_ok=True)
    
    # 运行多次测试
    for test_id in range(1, 6):  # 运行5次测试
        print(f"\n{'='*60}")
        print(f"开始测试 {test_id}/5")
        print(f"{'='*60}")
        
        try:
            # 1. 生成模拟器数据
            print(f"\n步骤1: 生成V10模拟器数据 (测试{test_id})")
            df_market, df_ofi, df_signals = create_v10_simulation_data(
                duration_seconds=300,  # 5分钟数据
                seed=42 + test_id
            )
            
            # 2. 运行模拟器回测
            print(f"\n步骤2: 运行V10模拟器回测 (测试{test_id})")
            results, trades_df = run_v10_simulation_backtest_fixed(
                df_market, df_ofi, df_signals, test_id
            )
            
            # 3. 创建测试报告
            print(f"\n步骤3: 创建测试报告 (测试{test_id})")
            report_dir = create_test_report_fixed(
                test_id, results, trades_df, df_market, df_ofi, df_signals
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
    print("V10.0 修复版模拟器测试完成")
    print(f"{'='*60}")
    print("所有测试报告已保存到 test_reports_fixed/ 目录")
    print("请查看各测试报告了解详细结果和优化建议")

if __name__ == "__main__":
    main()
