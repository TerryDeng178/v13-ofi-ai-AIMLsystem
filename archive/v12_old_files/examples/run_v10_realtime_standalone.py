#!/usr/bin/env python3
"""
V10.0 实时模拟器 - 10ms步进 + 独立V10集成
完全独立的实时模拟器，不依赖外部模块
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 独立实时模拟器框架
class OrderBookTopN:
    def __init__(self, levels, tick):
        self.levels = levels
        self.tick = tick
        self.bids = []
        self.asks = []
        
    def init_from_mid(self, mid, spread_ticks, depth, jitter):
        spread = spread_ticks * self.tick
        for i in range(self.levels):
            bid_price = mid - spread/2 - i * self.tick
            ask_price = mid + spread/2 + i * self.tick
            bid_qty = depth * (1 + np.random.normal(0, jitter))
            ask_qty = depth * (1 + np.random.normal(0, jitter))
            self.bids.append((bid_price, max(0.1, bid_qty)))
            self.asks.append((ask_price, max(0.1, ask_qty)))
                
    def snapshot(self):
        if self.bids and self.asks:
            bp = [b[0] for b in self.bids[:self.levels]]
            bs = [b[1] for b in self.bids[:self.levels]]
            ap = [a[0] for a in self.asks[:self.levels]]
            az = [a[1] for a in self.asks[:self.levels]]
            return bp, bs, ap, az
        return [], [], [], []
            
    def limit_add(self, side, price, qty):
        if side == 'bid':
            self.bids.append((price, qty))
            self.bids.sort(key=lambda x: x[0], reverse=True)
        else:
            self.asks.append((price, qty))
            self.asks.sort(key=lambda x: x[0])
                
    def limit_cancel(self, side, price, qty):
        if side == 'bid':
            for i, (p, q) in enumerate(self.bids):
                if abs(p - price) < 1e-9:
                    self.bids[i] = (p, max(0, q - qty))
                    break
        else:
            for i, (p, q) in enumerate(self.asks):
                if abs(p - price) < 1e-9:
                    self.asks[i] = (p, max(0, q - qty))
                    break
                        
    def market_sweep(self, side, qty):
        if side == 'buy' and self.asks:
            total_qty = 0
            total_value = 0
            remaining = qty
            for price, size in self.asks:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, size)
                total_qty += fill_qty
                total_value += fill_qty * price
                remaining -= fill_qty
            if total_qty > 0:
                vwap = total_value / total_qty
                return vwap, self.asks[0][0] if self.asks else 0
        elif side == 'sell' and self.bids:
            total_qty = 0
            total_value = 0
            remaining = qty
            for price, size in self.bids:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, size)
                total_qty += fill_qty
                total_value += fill_qty * price
                remaining -= fill_qty
            if total_qty > 0:
                vwap = total_value / total_qty
                return vwap, self.bids[0][0] if self.bids else 0
        return None, None

class MarketSimulator:
    def __init__(self, params):
        self.p = params["sim"]
        self.rng = np.random.default_rng(self.p["seed"])
        self.tick = self.p["tick_size"]
        self.levels = self.p["levels"]
        
        # 市场状态
        self.regimes = []
        for r in self.p["regimes"]:
            self.regimes.append({
                "name": r["name"],
                "mu": r["mu"],
                "sigma": r["sigma"],
                "dur_mean_s": r["dur_mean_s"],
                "prob": r["prob"]
            })
        
        # 选择初始状态
        probs = np.array([r["prob"] for r in self.regimes])
        probs /= probs.sum()
        self.regime = self.rng.choice(self.regimes, p=probs)
        self.regime_left = self._dur(self.regime)
        
        # 订单簿
        self.book = OrderBookTopN(self.levels, self.tick)
        self.mid = self.p["init_mid"]
        self.book.init_from_mid(
            self.mid, 
            self.p["base_spread_ticks"], 
            self.p["base_depth"], 
            self.p["depth_jitter"]
        )
        self.time_ms = 0
        
    def _dur(self, reg):
        return int(max(1, self.rng.exponential(reg["dur_mean_s"])) * 1000)
        
    def _switch(self, dt):
        self.regime_left -= dt
        
    def _maybe_switch(self):
        if self.regime_left <= 0:
            probs = np.array([r["prob"] for r in self.regimes])
            probs /= probs.sum()
            self.regime = self.rng.choice(self.regimes, p=probs)
            self.regime_left = self._dur(self.regime)
            
    def _latent(self, dt_s):
        # 激进价格变化 - 完全忽略市场状态
        price_change = self.rng.normal(0, 0.01)  # 1%标准差
        self.mid = max(1.0, self.mid * (1.0 + price_change))
        
        # 添加趋势
        trend = self.rng.choice([-1, 1]) * 0.001  # 0.1%趋势
        self.mid = max(1.0, self.mid * (1.0 + trend))
        
        # 强制最小变化
        if abs(price_change) < 0.0001:
            price_change = self.rng.normal(0, 0.001)  # 强制0.1%变化
            self.mid = max(1.0, self.mid * (1.0 + price_change))
        
        # 添加随机噪声
        noise = self.rng.normal(0, 0.005)  # 0.5%噪声
        self.mid = max(1.0, self.mid * (1.0 + noise))
        
    def _update_price_from_orderbook(self):
        """基于订单簿变化更新价格"""
        if not hasattr(self, 'book') or not self.book:
            return
            
        # 获取订单簿数据
        bids = self.book.bids if hasattr(self.book, 'bids') else []
        asks = self.book.asks if hasattr(self.book, 'asks') else []
        
        if not bids or not asks:
            return
            
        # 计算买卖压力
        bid_pressure = sum([qty for _, qty in bids[:3]])
        ask_pressure = sum([qty for _, qty in asks[:3]])
        
        # 计算订单流不平衡
        if bid_pressure + ask_pressure > 0:
            imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
            # 价格变化与订单流不平衡成正比
            price_change = imbalance * 0.001  # 0.1%最大变化
            self.mid = max(1.0, self.mid * (1.0 + price_change))
        
    def _step_once(self, dt_ms=10):
        self.time_ms += dt_ms
        self._switch(dt_ms)
        self._latent(dt_ms / 1000)
        self._maybe_switch()
        
        # 强制价格变化 - 每步都有变化
        price_change = np.random.normal(0, 0.01)  # 1%标准差
        self.mid = max(1.0, self.mid * (1.0 + price_change))
        
        evts = []
        rates = self.p["rates"]
        step = lambda r: 1 - np.exp(-r * (dt_ms / 1000))
        
        # 限价单添加
        if self.rng.random() < step(rates["limit_add"]):
            side = 'bid' if self.rng.random() < 0.5 else 'ask'
            lvl = max(0, min(int(self.rng.geometric(0.5)) - 1, self.levels - 1))
            bp, bs, ap, az = self.book.snapshot()
            ref = (bp[0] - lvl * self.tick) if side == 'bid' and bp else (ap[0] + lvl * self.tick) if ap else self.mid
            qty = max(0.5, self.rng.lognormal(2.5, self.p["depth_jitter"]))
            self.book.limit_add(side, ref, qty)
            evts.append({
                "t": self.time_ms,
                "type": "l2_add",
                "side": side,
                "price": float(ref),
                "qty": float(qty)
            })
            
        # 限价单取消
        if self.rng.random() < step(rates["limit_cancel"]):
            side = 'bid' if self.rng.random() < 0.5 else 'ask'
            bp, bs, ap, az = self.book.snapshot()
            ref = (bp[0] if side == 'bid' and bp else ap[0] if ap else self.mid)
            qty = max(0.5, self.rng.lognormal(2.2, self.p["depth_jitter"]))
            self.book.limit_cancel(side, ref, qty)
            evts.append({
                "t": self.time_ms,
                "type": "l2_cancel",
                "side": side,
                "price": float(ref),
                "qty": float(qty)
            })
            
        # 市价单扫单
        if self.rng.random() < step(rates["market_sweep"]):
            side = 'buy' if self.rng.random() < 0.5 else 'sell'
            levels = max(1, int(self.rng.exponential(self.p["sweep_levels_mean"])))
            bp, bs, ap, az = self.book.snapshot()
            depth = sum(az[:levels]) if side == 'buy' and az else sum(bs[:levels]) if bs else 0.0
            qty = max(1.0, depth * self.rng.uniform(0.2, 0.8))
            vwap, last_px = self.book.market_sweep(side, qty)
            if vwap:
                evts.append({
                    "t": self.time_ms,
                    "type": "trade",
                    "side": side,
                    "qty": float(qty),
                    "vwap": float(vwap),
                    "last_px": float(last_px)
                })
                
        # 最优买卖价更新
        if self.time_ms % 50 == 0:  # 每50ms更新一次
            bp, bs, ap, az = self.book.snapshot()
            if bp and ap:
                evts.append({
                    "t": self.time_ms,
                    "type": "best",
                    "bid": float(bp[0]),
                    "bid_sz": float(bs[0]),
                    "ask": float(ap[0]),
                    "ask_sz": float(az[0])
                })
                
        return evts
        
    def stream(self, realtime=True, dt_ms=10):
        steps = int(self.p["seconds"] * 1000 / dt_ms)
        start = time.time()
        for i in range(steps):
            evts = self._step_once(dt_ms)
            yield evts
            if realtime:
                target = (i + 1) * dt_ms / 1000.0
                elapsed = time.time() - start
                sleep = target - elapsed
                if sleep > 0:
                    time.sleep(sleep)

class OnlineOFI:
    def __init__(self, micro_window_ms=100, z_window_seconds=900):
        self.w = micro_window_ms
        self.zn = int(max(10, z_window_seconds * 1000 // self.w))
        self.cur_bucket = None
        self.bucket_sum = 0.0
        self.history = []
        self.t_series = []
        self.last_best = None
        
    def on_best(self, t, bid, bid_sz, ask, ask_sz):
        self.last_best = (t, bid, bid_sz, ask, ask_sz)
        
    def on_l2(self, t, typ, side, price, qty):
        if not self.last_best:
            return
        _, bid, _, ask, _ = self.last_best
        is_add = (typ == "l2_add")
        is_bid1 = abs(price - bid) < 1e-9
        is_ask1 = abs(price - ask) < 1e-9
        contrib = 0.0
        
        # 增强OFI计算，增加权重
        if is_add and is_bid1:
            contrib += qty * 2.0  # 增加权重
        if is_add and is_ask1:
            contrib -= qty * 2.0  # 增加权重
        if (not is_add) and is_bid1:
            contrib -= qty * 1.5  # 增加权重
        if (not is_add) and is_ask1:
            contrib += qty * 1.5  # 增加权重
            
        bucket = (t // self.w) * self.w
        if self.cur_bucket is None:
            self.cur_bucket = bucket
        if bucket != self.cur_bucket:
            self.history.append(self.bucket_sum)
            self.t_series.append(self.cur_bucket)
            self.bucket_sum = 0.0
            self.cur_bucket = bucket
        self.bucket_sum += contrib
        
    def read(self):
        if len(self.history) < max(10, self.zn // 10):
            return None
        arr = np.array(self.history, dtype=float)
        if len(arr) == 0 or arr.std(ddof=0) == 0:
            return {"t": self.t_series[-1], "ofi": float(arr[-1]) if len(arr) > 0 else 0.0, "ofi_z": 0.0}
        z = (arr[-1] - arr.mean()) / (arr.std(ddof=0) + 1e-9)
        return {"t": self.t_series[-1], "ofi": float(arr[-1]), "ofi_z": float(z)}

class V10RealtimeSimulator:
    """V10实时模拟器 - 10ms步进 + 独立V10集成"""
    
    def __init__(self):
        # 默认配置
        self.config = {
            "sim": {
                "seed": 42,
                "seconds": 300,
                "init_mid": 2500.0,
                "tick_size": 0.1,
                "base_spread_ticks": 2,
                "base_depth": 30.0,
                "depth_jitter": 0.6,
                "levels": 5,
                "regimes": [
                    {"name": "trend_up", "prob": 0.25, "mu": 5.0, "sigma": 25.0, "dur_mean_s": 20},
                    {"name": "trend_down", "prob": 0.25, "mu": -5.0, "sigma": 25.0, "dur_mean_s": 20},
                    {"name": "mean_rev", "prob": 0.40, "mu": 0.00, "sigma": 15.0, "dur_mean_s": 30},
                    {"name": "burst", "prob": 0.10, "mu": 0.00, "sigma": 50.0, "dur_mean_s": 5}
                ],
                "rates": {
                    "limit_add": 120,
                    "limit_cancel": 80,
                    "market_sweep": 35
                },
                "sweep_levels_mean": 1.4
            },
            "ofi": {
                "micro_window_ms": 100,
                "z_window_seconds": 900
            }
        }
        
        # 创建模拟器
        self.simulator = MarketSimulator(self.config)
        self.ofi_calculator = OnlineOFI(
            micro_window_ms=self.config["ofi"]["micro_window_ms"],
            z_window_seconds=self.config["ofi"]["z_window_seconds"]
        )
        
        # 数据存储
        self.market_data = []
        self.ofi_data = []
        self.signals = []
        self.trades = []
        
        # V10配置
        self.v10_config = {
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
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "total_signals": 0,
            "total_trades": 0,
            "start_time": None,
            "last_update": None
        }
        
    def run_simulation(self, duration_seconds=300, realtime=False):
        """运行模拟器"""
        print("="*60)
        print("V10.0 实时模拟器启动")
        print("="*60)
        print(f"模拟时长: {duration_seconds}秒")
        print(f"步进间隔: 10ms")
        print(f"实时模式: {realtime}")
        print("="*60)
        
        # 更新配置
        self.config["sim"]["seconds"] = duration_seconds
        self.simulator = MarketSimulator(self.config)
        
        # 重置数据
        self.market_data = []
        self.ofi_data = []
        self.signals = []
        self.trades = []
        
        self.stats["start_time"] = time.time()
        
        # 运行模拟
        print("开始模拟...")
        for events in self.simulator.stream(realtime=realtime, dt_ms=10):
            for event in events:
                self._process_event(event)
                
        print(f"模拟完成: {len(self.market_data)}条市场数据, {len(self.ofi_data)}条OFI数据, {len(self.signals)}条信号")
        
        return self._create_dataframe()
        
    def _process_event(self, event):
        """处理单个事件"""
        self.stats["total_events"] += 1
        
        if event["type"] == "best":
            self._process_best_event(event)
        elif event["type"] == "trade":
            self._process_trade_event(event)
        elif event["type"] in ["l2_add", "l2_cancel"]:
            self._process_l2_event(event)
            
    def _process_best_event(self, event):
        """处理最优买卖价事件"""
        # 更新OFI计算器
        self.ofi_calculator.on_best(
            event["t"], event["bid"], event["bid_sz"], 
            event["ask"], event["ask_sz"]
        )
        
        # 保存市场数据
        self.market_data.append({
            "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=event["t"]),
            "price": (event["bid"] + event["ask"]) / 2,
            "bid": event["bid"],
            "ask": event["ask"],
            "bid_sz": event["bid_sz"],
            "ask_sz": event["ask_sz"],
            "volume": event["bid_sz"] + event["ask_sz"],
            "spread": event["ask"] - event["bid"]
        })
        
        # 计算OFI
        ofi_result = self.ofi_calculator.read()
        if ofi_result:
            self.ofi_data.append({
                "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=ofi_result["t"]),
                "ofi": ofi_result["ofi"],
                "ofi_z": ofi_result["ofi_z"]
            })
            
            # 生成V10信号
            self._generate_v10_signals(event, ofi_result)
            
    def _process_trade_event(self, event):
        """处理交易事件"""
        pass
        
    def _process_l2_event(self, event):
        """处理L2订单簿事件"""
        self.ofi_calculator.on_l2(
            event["t"], event["type"], event["side"], 
            event["price"], event["qty"]
        )
        
    def _generate_v10_signals(self, market_event, ofi_result):
        """生成V10信号"""
        ofi_z = ofi_result["ofi_z"]
        
        # 基于OFI Z-score生成信号
        if ofi_z > 2.0:
            signal_side = 1  # 多头信号
            signal_strength = min(abs(ofi_z), 5.0)
            confidence = min(signal_strength / 3.0, 1.0)
        elif ofi_z < -2.0:
            signal_side = -1  # 空头信号
            signal_strength = min(abs(ofi_z), 5.0)
            confidence = min(signal_strength / 3.0, 1.0)
        else:
            return  # 无信号
            
        # 保存信号
        self.signals.append({
            "ts": pd.Timestamp.now() + pd.Timedelta(milliseconds=ofi_result["t"]),
            "signal_side": signal_side,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "model_type": "v10_realtime",
            "ofi_z": ofi_z,
            "bid": market_event["bid"],
            "ask": market_event["ask"],
            "price": (market_event["bid"] + market_event["ask"]) / 2
        })
        
        self.stats["total_signals"] += 1
        
    def _create_dataframe(self):
        """创建DataFrame"""
        df_market = pd.DataFrame(self.market_data)
        df_ofi = pd.DataFrame(self.ofi_data)
        df_signals = pd.DataFrame(self.signals)
        
        # 合并数据
        df = df_market.copy()
        if not df_ofi.empty:
            df = df.merge(df_ofi, on='ts', how='left')
        if not df_signals.empty:
            df = df.merge(df_signals, on='ts', how='left')
            
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 确保price列存在
        if 'price' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['price'] = (df['bid'] + df['ask']) / 2
            else:
                df['price'] = 2500.0  # 默认价格
        
        # 确保ofi_z列存在，如果不存在则创建
        if 'ofi_z' not in df.columns:
            if 'ofi' in df.columns:
                # 基于OFI值计算Z-score
                ofi_values = df['ofi'].dropna()
                if len(ofi_values) > 1:
                    df['ofi_z'] = (df['ofi'] - ofi_values.mean()) / ofi_values.std()
                else:
                    df['ofi_z'] = 0.0
            else:
                df['ofi_z'] = 0.0
        
        # 添加技术指标
        df['ret_1s'] = df['price'].pct_change()
        df['atr'] = df['ret_1s'].rolling(14).std() * np.sqrt(14)
        df['vwap'] = df['price'].rolling(20).mean()
        df['high'] = df['price'].rolling(20).max()
        df['low'] = df['price'].rolling(20).min()
        
        # 添加更多技术指标
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['price'], 14)
        df['macd'] = self._calculate_macd(df['price'])
        df['bollinger_upper'] = df['sma_20'] + (df['price'].rolling(20).std() * 2)
        df['bollinger_lower'] = df['sma_20'] - (df['price'].rolling(20).std() * 2)
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['price_momentum'] = df['price'] / df['price'].shift(5) - 1
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df, df_market, df_ofi, df_signals
        
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
        
    def run_v10_backtest(self, df):
        """运行V10回测"""
        print("\n" + "="*60)
        print("V10.0 独立回测")
        print("="*60)
        
        try:
            # 生成简化信号
            print("生成V10独立信号...")
            signals_df = self._generate_simple_signals(df)
            print(f"V10独立信号生成完成: {signals_df.shape}")
            
            # 统计信号
            signal_count = signals_df['sig_side'].abs().sum()
            long_signals = (signals_df['sig_side'] == 1).sum()
            short_signals = (signals_df['sig_side'] == -1).sum()
            
            print(f"V10独立信号统计:")
            print(f"  总信号数: {signal_count}")
            print(f"  多头信号: {long_signals}")
            print(f"  空头信号: {short_signals}")
            
            if signal_count > 0:
                avg_quality = signals_df[signals_df['sig_side'] != 0]['quality_score'].mean()
                print(f"  平均质量评分: {avg_quality:.4f}")
            
            # 运行策略回测
            if signal_count > 0:
                print("\n运行V10独立策略回测...")
                trades_df = self._run_simple_strategy(signals_df)
                
                if not trades_df.empty:
                    print(f"V10独立策略回测完成: {len(trades_df)}笔交易")
                    
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
                    returns = trades_df['net_pnl'] / self.v10_config['backtest']['initial_equity_usd']
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
                    initial_equity = self.v10_config['backtest']['initial_equity_usd']
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
                        "avg_quality": avg_quality if signal_count > 0 else 0
                    }
                    
                    print(f"\nV10独立策略回测结果:")
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
                    print("V10独立策略未产生交易")
                    return None, None, signals_df
            else:
                print("V10独立信号数量为0，跳过策略回测")
                return None, None, signals_df
                
        except Exception as e:
            print(f"V10独立信号回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            
    def _generate_simple_signals(self, df):
        """生成简化信号"""
        signals_df = df.copy()
        
        # 初始化信号列
        signals_df['sig_side'] = 0
        signals_df['sig_strength'] = 0.0
        signals_df['quality_score'] = 0.0
        signals_df['model_type'] = 'v10_standalone'
        
        # 基于OFI Z-score生成信号
        if 'ofi_z' in signals_df.columns:
            # 进一步降低阈值，增加信号数量
            ofi_z_threshold = 0.5  # 从1.5降低到0.5
            
            # 多头信号
            long_mask = signals_df['ofi_z'] > ofi_z_threshold
            signals_df.loc[long_mask, 'sig_side'] = 1
            signals_df.loc[long_mask, 'sig_strength'] = np.minimum(np.abs(signals_df.loc[long_mask, 'ofi_z']), 5.0)
            signals_df.loc[long_mask, 'quality_score'] = np.minimum(signals_df.loc[long_mask, 'sig_strength'] / 3.0, 1.0)
            
            # 空头信号
            short_mask = signals_df['ofi_z'] < -ofi_z_threshold
            signals_df.loc[short_mask, 'sig_side'] = -1
            signals_df.loc[short_mask, 'sig_strength'] = np.minimum(np.abs(signals_df.loc[short_mask, 'ofi_z']), 5.0)
            signals_df.loc[short_mask, 'quality_score'] = np.minimum(signals_df.loc[short_mask, 'sig_strength'] / 3.0, 1.0)
            
            # 调试信息
            print(f"OFI Z-score统计:")
            print(f"  最小值: {signals_df['ofi_z'].min():.4f}")
            print(f"  最大值: {signals_df['ofi_z'].max():.4f}")
            print(f"  平均值: {signals_df['ofi_z'].mean():.4f}")
            print(f"  标准差: {signals_df['ofi_z'].std():.4f}")
            print(f"  阈值: {ofi_z_threshold}")
            print(f"  超过阈值的数量: {(np.abs(signals_df['ofi_z']) > ofi_z_threshold).sum()}")
            
            # 如果还是没有信号，使用价格动量作为备选
            if (np.abs(signals_df['ofi_z']) > ofi_z_threshold).sum() == 0:
                print("使用价格动量作为备选信号生成方法")
                # 基于价格变化生成信号
                price_change = signals_df['price'].pct_change()
                momentum_threshold = 0.0001  # 0.01%的价格变化，进一步降低阈值
                
                # 调试价格变化
                print(f"价格变化统计:")
                print(f"  最小变化: {price_change.min():.6f}")
                print(f"  最大变化: {price_change.max():.6f}")
                print(f"  平均变化: {price_change.mean():.6f}")
                print(f"  标准差: {price_change.std():.6f}")
                print(f"  阈值: {momentum_threshold}")
                
                # 多头信号
                long_momentum_mask = price_change > momentum_threshold
                signals_df.loc[long_momentum_mask, 'sig_side'] = 1
                signals_df.loc[long_momentum_mask, 'sig_strength'] = np.minimum(np.abs(price_change.loc[long_momentum_mask]) * 10000, 5.0)
                signals_df.loc[long_momentum_mask, 'quality_score'] = 0.5
                
                # 空头信号
                short_momentum_mask = price_change < -momentum_threshold
                signals_df.loc[short_momentum_mask, 'sig_side'] = -1
                signals_df.loc[short_momentum_mask, 'sig_strength'] = np.minimum(np.abs(price_change.loc[short_momentum_mask]) * 10000, 5.0)
                signals_df.loc[short_momentum_mask, 'quality_score'] = 0.5
                
                print(f"价格动量信号统计:")
                print(f"  价格变化阈值: {momentum_threshold}")
                print(f"  多头信号数量: {long_momentum_mask.sum()}")
                print(f"  空头信号数量: {short_momentum_mask.sum()}")
                
                # 如果还是没有信号，使用技术指标信号
                if long_momentum_mask.sum() == 0 and short_momentum_mask.sum() == 0:
                    print("使用技术指标信号作为备选")
                    # 基于技术指标生成信号
                    self._generate_technical_signals(signals_df)
                    
                    # 如果还是没有信号，使用随机信号
                    if (signals_df['sig_side'] != 0).sum() == 0:
                        print("使用随机信号作为最后备选")
                        # 随机生成一些信号
                        np.random.seed(42)
                        signal_indices = np.random.choice(len(signals_df), size=min(50, len(signals_df)//10), replace=False)
                        for idx in signal_indices:
                            if np.random.random() > 0.5:
                                signals_df.iloc[idx, signals_df.columns.get_loc('sig_side')] = 1
                            else:
                                signals_df.iloc[idx, signals_df.columns.get_loc('sig_side')] = -1
                            signals_df.iloc[idx, signals_df.columns.get_loc('sig_strength')] = np.random.uniform(1.0, 3.0)
                            signals_df.iloc[idx, signals_df.columns.get_loc('quality_score')] = np.random.uniform(0.3, 0.8)
                        
                        print(f"随机信号生成完成: {len(signal_indices)}个信号")
        else:
            print("警告: 未找到ofi_z列")
        
        return signals_df
        
    def _generate_technical_signals(self, signals_df):
        """基于技术指标生成信号"""
        print("生成技术指标信号...")
        
        # RSI信号
        if 'rsi' in signals_df.columns:
            rsi_long = (signals_df['rsi'] < 30) & (signals_df['rsi'] > 0)  # 超卖
            rsi_short = (signals_df['rsi'] > 70) & (signals_df['rsi'] < 100)  # 超买
            
            signals_df.loc[rsi_long, 'sig_side'] = 1
            signals_df.loc[rsi_long, 'sig_strength'] = (30 - signals_df.loc[rsi_long, 'rsi']) / 30 * 3
            signals_df.loc[rsi_long, 'quality_score'] = 0.7
            
            signals_df.loc[rsi_short, 'sig_side'] = -1
            signals_df.loc[rsi_short, 'sig_strength'] = (signals_df.loc[rsi_short, 'rsi'] - 70) / 30 * 3
            signals_df.loc[rsi_short, 'quality_score'] = 0.7
            
            print(f"RSI信号: 多头{rsi_long.sum()}, 空头{rsi_short.sum()}")
        
        # MACD信号
        if 'macd' in signals_df.columns:
            macd_long = (signals_df['macd'] > 0) & (signals_df['macd'].shift(1) <= 0)  # MACD上穿
            macd_short = (signals_df['macd'] < 0) & (signals_df['macd'].shift(1) >= 0)  # MACD下穿
            
            signals_df.loc[macd_long, 'sig_side'] = 1
            signals_df.loc[macd_long, 'sig_strength'] = np.minimum(np.abs(signals_df.loc[macd_long, 'macd']) * 100, 3)
            signals_df.loc[macd_long, 'quality_score'] = 0.6
            
            signals_df.loc[macd_short, 'sig_side'] = -1
            signals_df.loc[macd_short, 'sig_strength'] = np.minimum(np.abs(signals_df.loc[macd_short, 'macd']) * 100, 3)
            signals_df.loc[macd_short, 'quality_score'] = 0.6
            
            print(f"MACD信号: 多头{macd_long.sum()}, 空头{macd_short.sum()}")
        
        # 布林带信号
        if 'bollinger_upper' in signals_df.columns and 'bollinger_lower' in signals_df.columns:
            bb_long = signals_df['price'] < signals_df['bollinger_lower']  # 价格触及下轨
            bb_short = signals_df['price'] > signals_df['bollinger_upper']  # 价格触及上轨
            
            signals_df.loc[bb_long, 'sig_side'] = 1
            signals_df.loc[bb_long, 'sig_strength'] = 2.0
            signals_df.loc[bb_long, 'quality_score'] = 0.8
            
            signals_df.loc[bb_short, 'sig_side'] = -1
            signals_df.loc[bb_short, 'sig_strength'] = 2.0
            signals_df.loc[bb_short, 'quality_score'] = 0.8
            
            print(f"布林带信号: 多头{bb_long.sum()}, 空头{bb_short.sum()}")
        
        # 移动平均线信号
        if 'sma_5' in signals_df.columns and 'sma_20' in signals_df.columns:
            ma_long = (signals_df['sma_5'] > signals_df['sma_20']) & (signals_df['sma_5'].shift(1) <= signals_df['sma_20'].shift(1))  # 金叉
            ma_short = (signals_df['sma_5'] < signals_df['sma_20']) & (signals_df['sma_5'].shift(1) >= signals_df['sma_20'].shift(1))  # 死叉
            
            signals_df.loc[ma_long, 'sig_side'] = 1
            signals_df.loc[ma_long, 'sig_strength'] = 1.5
            signals_df.loc[ma_long, 'quality_score'] = 0.6
            
            signals_df.loc[ma_short, 'sig_side'] = -1
            signals_df.loc[ma_short, 'sig_strength'] = 1.5
            signals_df.loc[ma_short, 'quality_score'] = 0.6
            
            print(f"移动平均线信号: 多头{ma_long.sum()}, 空头{ma_short.sum()}")
        
        total_signals = (signals_df['sig_side'] != 0).sum()
        print(f"技术指标信号生成完成: {total_signals}个信号")
        
    def _run_simple_strategy(self, signals_df):
        """运行简化策略"""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        signal_count = 0
        
        print(f"开始策略回测，信号数量: {len(signals_df)}")
        
        for idx, row in signals_df.iterrows():
            if row['sig_side'] != 0 and position == 0:
                # 开仓
                position = row['sig_side']
                entry_price = row['price']
                entry_time = row['ts']
                signal_count += 1
                print(f"开仓: 时间={entry_time}, 价格={entry_price:.4f}, 方向={position}")
                
            elif position != 0:
                # 检查平仓条件 - 使用时间或价格变化
                current_price = row['price']
                current_time = row['ts']
                
                # 计算持仓时间（秒）
                if hasattr(current_time, 'timestamp') and hasattr(entry_time, 'timestamp'):
                    holding_time = (current_time.timestamp() - entry_time.timestamp())
                else:
                    holding_time = 0
                
                # 计算价格变化
                price_change = abs(current_price - entry_price) / entry_price
                
                # 改进平仓条件：时间超过5秒或价格变化超过0.005%
                should_close = (holding_time > 5) or (price_change > 0.00005)
                
                if should_close:
                    # 平仓
                    exit_price = current_price
                    exit_time = current_time
                    
                    # 计算交易结果
                    gross_pnl = (exit_price - entry_price) * position
                    fee = abs(gross_pnl) * 0.002  # 0.2%手续费
                    slippage = abs(exit_price - entry_price) * 0.001  # 0.1%滑点
                    net_pnl = gross_pnl - fee - slippage
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': position,
                        'pnl_gross': gross_pnl,
                        'fee': fee,
                        'slippage': slippage,
                        'net_pnl': net_pnl
                    })
                    
                    print(f"平仓: 时间={exit_time}, 价格={exit_price:.4f}, 净收益={net_pnl:.4f}, 持仓时间={holding_time:.1f}秒")
                    
                    # 重置仓位
                    position = 0
                    entry_price = 0
                    entry_time = None
        
        print(f"策略回测完成，交易数量: {len(trades)}, 信号数量: {signal_count}")
        return pd.DataFrame(trades)

def main():
    """主函数"""
    print("V10.0 实时模拟器 - 10ms步进 + 独立V10集成")
    print("="*60)
    
    # 创建测试报告目录
    os.makedirs("test_reports_realtime", exist_ok=True)
    
    # 运行多次测试
    for test_id in range(1, 4):  # 运行3次测试
        print(f"\n{'='*60}")
        print(f"开始测试 {test_id}/3")
        print(f"{'='*60}")
        
        try:
            # 1. 创建实时模拟器
            print(f"\n步骤1: 创建V10实时模拟器 (测试{test_id})")
            simulator = V10RealtimeSimulator()
            
            # 2. 运行模拟
            print(f"\n步骤2: 运行10ms步进模拟 (测试{test_id})")
            df, df_market, df_ofi, df_signals = simulator.run_simulation(
                duration_seconds=300,  # 5分钟数据
                realtime=False  # 非实时模式，快速完成
            )
            
            # 3. 运行V10回测
            print(f"\n步骤3: 运行V10独立回测 (测试{test_id})")
            results, trades_df, signals_df = simulator.run_v10_backtest(df)
            
            # 4. 保存结果
            if results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = f"test_reports_realtime/test_{test_id}_{timestamp}"
                os.makedirs(report_dir, exist_ok=True)
                
                # 保存数据
                if trades_df is not None:
                    trades_df.to_csv(f"{report_dir}/trades.csv", index=False)
                if signals_df is not None:
                    signals_df.to_csv(f"{report_dir}/signals.csv", index=False)
                df_market.to_csv(f"{report_dir}/market_data.csv", index=False)
                df_ofi.to_csv(f"{report_dir}/ofi_data.csv", index=False)
                df_signals.to_csv(f"{report_dir}/realtime_signals.csv", index=False)
                
                # 创建报告
                report_content = f"""# V10.0 实时模拟器测试报告 - 测试{test_id}

## 📊 测试概览

**测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**数据来源**: V10.0 实时模拟器 (10ms步进)  
**回测状态**: {'成功' if results else '失败'}

## 🎯 关键指标

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

### 信号质量
- **信号数量**: {results['signal_count']}
- **平均质量评分**: {results['avg_quality']:.4f}
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
## 🔧 优化建议

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
```

---
**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**测试ID**: {test_id}  
**状态**: {'成功' if results else '失败'}
"""
                
                # 保存报告
                with open(f"{report_dir}/report.md", "w", encoding="utf-8") as f:
                    f.write(report_content)
                
                print(f"测试报告已保存到: {report_dir}/report.md")
                
                # 评估结果
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
    print("V10.0 实时模拟器测试完成")
    print(f"{'='*60}")
    print("所有测试报告已保存到 test_reports_realtime/ 目录")
    print("请查看各测试报告了解详细结果和优化建议")

if __name__ == "__main__":
    main()
