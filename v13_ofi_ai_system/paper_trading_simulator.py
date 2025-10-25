#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纸上交易模拟器：验证2×2场景化参数的实际效果
"""

import os
import sys
import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

# 快速自检：确保stderr未被提前关闭
assert not getattr(sys.stderr, "closed", False), "stderr 已被提前关闭！"

# 添加项目路径 - 使用绝对路径避免相对路径漂移
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))           # 项目根
sys.path.insert(0, str(PROJECT_ROOT / "core"))  # core 包
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # src 包

from src.utils.strategy_mode_manager import StrategyModeManager
from src.utils.config_loader import load_config
from core_algo import CoreAlgorithm, SignalConfig

class PaperTradingSimulator:
    """纸上交易模拟器"""
    
    def __init__(self, config_path: str = None, symbol: str = "BTCUSDT"):
        """初始化模拟器"""
        self.config_path = config_path or str(PROJECT_ROOT / "reports/scenario_opt/strategy_params_fusion_clean.yaml")
        self.symbol = symbol
        self.manager = None
        self.core_algo = None
        self.trades = []
        self.positions = {}
        self.kpis = {
            'Q_L': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'A_L': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'A_H': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'Q_H': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'Active_High': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},  # 添加新场景
            'Active_Low': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'Quiet_High': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0},
            'Quiet_Low': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0}
        }
        self.log_entries = []  # 结构化日志
        self.cost_bps = 3.0  # 3bps交易成本
        
        # 场景门控参数 - 提频优化版（再下调20-30%）
        self.SCENE_GATE = {
            "Q_H": {"enter": 0.8, "exit": 0.4, "cooldown_s": 96},   # 1.0→0.8, 120→96s
            "A_H": {"enter": 0.6, "exit": 0.3, "cooldown_s": 72},   # 0.8→0.6, 90→72s
            "A_L": {"enter": 0.55, "exit": 0.28, "cooldown_s": 72}, # 0.7→0.55, 90→72s
            "Q_L": {"enter": 0.4, "exit": 0.2, "cooldown_s": 96},   # 0.5→0.4, 120→96s
        }
        
        # 信号稳态判定
        self.signal_history = {}  # {symbol: deque(maxlen=3)}
        self.last_flip_time = {}  # {symbol: timestamp}
        self.min_hold_time = {
            "Q_H": 120, "A_H": 90, "A_L": 60, "Q_L": 30
        }
        
        # 翻转计数
        self.flip_count = {}  # {symbol: {hour: count}}
        self.last_calibration = None
        
        # 第二步优化：自适应场景标签
        self.scene_cache = {}  # {symbol: {'price_history': [], 'trade_history': []}}
        self.adaptive_weights = {}  # {symbol: {'w_ofi': 0.6, 'w_cvd': 0.4}}
        self.ic_history = {}  # {symbol: {'ofi_ic': [], 'cvd_ic': []}}
        
        # 升级版反转闸门参数
        self.last_mid_price = {}  # {symbol: mid_price}
        self.min_move_ticks = 2  # 最小移动tick数
        self.tick_size = 0.01  # BTC tick size
        self.max_spread_bps = {  # 最大点差限制
            "Q_H": 5.0, "A_H": 3.0, "A_L": 2.0, "Q_L": 1.5
        }
        self.reverse_count_30m = {}  # {symbol: deque(timestamps)}
        
        # 场景标签强制均衡参数
        self.scenario_coverage = {}  # {symbol: {scenario: count}}
        self.coverage_window = 4 * 60  # 4小时窗口
        self.min_coverage_percent = 0.15  # 15%最小覆盖
        self.scenario_smoothing = {}  # {symbol: deque(maxlen=3)}
        self.weak_signal_threshold = 0.3  # 弱信号区域阈值
        
        # A/B测试参数
        self.enable_weak_signal_throttle = False  # B方案启用
        self.weak_signal_volatility_threshold = 0.12  # 0.12%/h
        self.weak_signal_activity_threshold = 20  # 20分位
        
        # 信号计数（用于A/B测试）
        self.signal_count = 0
        self.confirmed_count = 0
        
    def initialize(self):
        """初始化模拟器"""
        print("[初始化] 初始化纸上交易模拟器...")
        
        try:
            # 加载配置
            config = load_config()
            print("成功 配置加载成功")
            
            # 初始化核心算法
            signal_config = SignalConfig()
            self.core_algo = CoreAlgorithm(self.symbol, signal_config, config_loader=config)
            print("成功 核心算法初始化成功")
            
            # 初始化StrategyModeManager
            self.manager = StrategyModeManager(config_loader=None)
            print("成功 StrategyModeManager初始化成功")
            
            # 加载场景参数
            success = self.manager.load_scenario_params(self.config_path)
            if not success:
                raise Exception("场景参数加载失败")
            print("成功 场景参数加载成功")
            
            print("成功 纸上交易模拟器初始化完成")
            print(f"   交易对: {self.symbol}")
            print(f"   场景参数配置: {self.config_path}")
            
        except Exception as e:
            print(f"[失败] 初始化失败: {e}")
            raise
        
    def simulate_trade(self, symbol: str, price: float, fusion_score: float, 
                      scenario_2x2: str, timestamp: datetime) -> Dict[str, Any]:
        """模拟单笔交易"""
        
        # 获取场景参数（修复参数键名）
        try:
            if fusion_score > 0:  # 多头信号
                params = self.manager.get_params_for_scenario(scenario_2x2, 'long')
                side = 'long'
                entry_threshold = params.get('Z_HI_LONG', params.get('Z_HI', 2.0))  # 兼容不同键名
            else:  # 空头信号
                params = self.manager.get_params_for_scenario(scenario_2x2, 'short')
                side = 'short'
                entry_threshold = params.get('Z_HI_SHORT', params.get('Z_HI', 2.0))  # 兼容不同键名
        except Exception as e:
            print(f"[警告] 获取场景参数失败: {e}")
            # 使用默认阈值
            entry_threshold = 2.0
            side = 'long' if fusion_score > 0 else 'short'
            params = {'Z_HI_LONG': 2.0, 'Z_HI_SHORT': 2.0, 'TP_BPS': 12, 'SL_BPS': 9}
        
        # 信号计数（用于A/B测试）
        self.signal_count += 1
        
        # 检查是否触发入场
        if abs(fusion_score) >= entry_threshold:
            self.confirmed_count += 1
            # 检查是否已有持仓
            if symbol in self.positions:
                current_position = self.positions[symbol]
                # 若已有持仓，先判断是否需要反向平仓
                if current_position['side'] != side:
                    # 检查反转闸门和翻转次数限制（使用price作为mid_price）
                    if not self.can_reverse(symbol, fusion_score, timestamp, scenario_2x2, 
                                          best_bid=price*0.999, best_ask=price*1.001, mid_price=price):
                        print(f"[BLOCK] 反转闸门阻止: {symbol}, 冷却中或信号不足")
                        return None
                    
                    if self._get_flip_count(symbol, timestamp) >= 3:
                        print(f"[BLOCK] 翻转次数超限: {symbol}, 当前小时翻转次数已达上限")
                        return None
                    
                    # 反向仓位，先平仓
                    self.close_position(symbol, price, timestamp, "reverse_open")
                    self._increment_flip_count(symbol, timestamp)
                    self.last_flip_time[symbol] = timestamp
                    self._record_reverse(symbol, timestamp)  # 记录反转事件
                    # 再考虑是否开仓
                else:
                    # 同方向仓位，忽略重复信号
                    return None
            
            # 模拟入场
            trade = {
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'fusion_score': fusion_score,
                'scenario_2x2': scenario_2x2,
                'params': params,
                'status': 'open'
            }
            
            # 计算止损止盈价格
            if side == 'long':
                trade['stop_loss'] = price * (1 - params['SL_BPS'] / 10000)
                trade['take_profit'] = price * (1 + params['TP_BPS'] / 10000)
            else:
                trade['stop_loss'] = price * (1 + params['SL_BPS'] / 10000)
                trade['take_profit'] = price * (1 - params['TP_BPS'] / 10000)
            
            self.trades.append(trade)
            self.positions[symbol] = trade
            
            print(f"[趋势] {scenario_2x2} {side} 入场: {symbol} @ {price:.4f}, fusion={fusion_score:.3f}")
            
            # 记录结构化日志
            self.log_entries.append({
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'action': 'open',
                'side': side,
                'price': price,
                'fusion_score': fusion_score,
                'scenario_2x2': scenario_2x2,
                'config_version': self.manager.get_scenario_stats().get('version', 'unknown') if self.manager else 'unknown'
            })
            
            return trade
        
        return None
    
    def _calibrate_thresholds(self, merged_df: pd.DataFrame, timestamp: datetime):
        """升级版动态阈值校准：Q90阈值 + 确认率目标微调"""
        try:
            # 获取最近30分钟的数据
            cutoff_time = timestamp - timedelta(minutes=30)
            recent_data = merged_df[merged_df['timestamp'] >= cutoff_time]
            
            if len(recent_data) < 10:
                return
            
            # 计算滚动分位数校准（升级到Q90）
            fusion_scores = recent_data['z_ofi'] * 0.6 + recent_data['z_cvd'] * 0.4
            q90 = np.percentile(np.abs(fusion_scores), 90)  # 升级到Q90
            
            # 计算当前确认率
            current_confirm_rate = self._calculate_confirm_rate(recent_data)
            
            # 更新场景门控参数
            for scenario in self.SCENE_GATE:
                if q90 > 0:
                    # 基于确认率目标微调（20%-35%）
                    if current_confirm_rate < 0.20:  # 确认率过低，降低阈值
                        adjustment = -0.1
                    elif current_confirm_rate > 0.35:  # 确认率过高，提高阈值
                        adjustment = 0.1
                    else:
                        adjustment = 0
                    
                    # 动态调整进入阈值
                    new_enter = max(self.SCENE_GATE[scenario]["enter"] + adjustment, q90 * 0.8)
                    self.SCENE_GATE[scenario]["enter"] = min(new_enter, 3.0)  # 上限3.0
                    
            print(f"[初始化] 动态阈值校准完成，Q90={q90:.3f}, 确认率={current_confirm_rate:.1%}")
            
        except Exception as e:
            print(f"[警告] 动态阈值校准失败: {e}")
    
    def _calculate_confirm_rate(self, recent_data: pd.DataFrame):
        """计算当前确认率"""
        if len(recent_data) == 0:
            return 0.0
        
        # 计算融合分数
        fusion_scores = recent_data['z_ofi'] * 0.6 + recent_data['z_cvd'] * 0.4
        
        # 计算确认率（通过阈值的信号比例）
        confirmed_signals = 0
        total_signals = len(fusion_scores)
        
        for score in fusion_scores:
            # 检查是否通过任何场景的进入阈值
            for scenario in self.SCENE_GATE:
                if abs(score) >= self.SCENE_GATE[scenario]["enter"]:
                    confirmed_signals += 1
                    break
        
        return confirmed_signals / total_signals if total_signals > 0 else 0.0
    
    def _check_weak_signal_region(self, symbol: str, current_price: float, timestamp: datetime):
        """检查弱信号区域：波动<0.12%/h 或 活跃<20分位"""
        try:
            # 如果未启用弱信号节流，直接返回False
            if not self.enable_weak_signal_throttle:
                return False
            
            # 获取最近1小时的价格数据
            cutoff_time = timestamp - timedelta(hours=1)
            
            # 这里需要价格历史数据，简化实现
            # 实际应该从数据源获取历史价格
            hourly_volatility = 0.05  # 简化：假设5%小时波动率
            activity_percentile = 30   # 简化：假设30分位活跃度
            
            # 弱信号条件：波动<阈值 或 活跃<阈值
            is_weak_vol = hourly_volatility < self.weak_signal_volatility_threshold
            is_weak_activity = activity_percentile < self.weak_signal_activity_threshold
            
            is_weak = is_weak_vol or is_weak_activity
            
            if is_weak:
                print(f"🔍 弱信号区域检测: 波动={hourly_volatility:.3f}, 活跃={activity_percentile}, 弱信号={'是' if is_weak else '否'}")
            
            return is_weak
            
        except Exception as e:
            print(f"[警告] 弱信号区域检测失败: {e}")
            return False
    
    def calculate_adaptive_scenario_labels(self, symbol: str, current_time: datetime, 
                                        price_data: pd.DataFrame, trade_data: pd.DataFrame = None):
        """升级版场景标签：强制均衡 + 时间平滑"""
        try:
            # 获取历史数据窗口
            window_minutes = 60
            cutoff_time = current_time - timedelta(minutes=window_minutes)
            
            # 过滤最近数据
            recent_prices = price_data[price_data['timestamp'] >= cutoff_time]
            
            if len(recent_prices) < 10:
                return 'Q_L'  # 默认场景
            
            # 计算价格变化率（波动度）
            price_changes = recent_prices['price'].pct_change().abs().dropna()
            
            # 计算活跃度（成交频率）
            if trade_data is not None:
                recent_trades = trade_data[trade_data['timestamp'] >= cutoff_time]
                trade_frequency = len(recent_trades) / window_minutes
            else:
                # 使用价格数据估算活跃度
                trade_frequency = len(recent_prices) / window_minutes
            
            # 维护最近1h的 trade_frequency 历史
            if symbol not in self.scene_cache:
                self.scene_cache[symbol] = {}
            hist = self.scene_cache[symbol].setdefault("trade_freq_hist", deque(maxlen=60))
            hist.append(trade_frequency)
            
            # 1. 双变量自适应分位计算
            p_vol, p_act = self._calculate_adaptive_percentiles(price_changes, trade_frequency)
            
            # 2. 强制均衡：检查覆盖约束
            p_vol, p_act = self._rebalance_cuts(symbol, p_vol, p_act, current_time)
            
            # 3. 当前波动度判定
            current_vol = price_changes.iloc[-1] if len(price_changes) > 0 else 0
            vol_percentile = np.percentile(price_changes, p_vol) if len(price_changes) > 0 else 0.001
            vol_bucket = 'High' if current_vol >= vol_percentile else 'Low'
            
            # 4. 当前活跃度判定 - 用历史分布的分位做门槛
            act_threshold = np.percentile(list(hist), p_act) if len(hist) >= 10 else trade_frequency
            regime = 'Active' if trade_frequency >= act_threshold else 'Quiet'
            
            # 5. 弱信号区域检测
            is_weak_signal = (current_vol < np.percentile(price_changes, 30) and 
                             trade_frequency < act_threshold * 0.5)
            
            scenario = f"{regime}_{vol_bucket}"
            
            # 6. 时间平滑：3点中值滤波
            if symbol not in self.scenario_smoothing:
                self.scenario_smoothing[symbol] = deque(maxlen=3)
            
            self.scenario_smoothing[symbol].append(scenario)
            if len(self.scenario_smoothing[symbol]) >= 3:
                # 使用中值滤波
                scenarios = list(self.scenario_smoothing[symbol])
                scenario = max(set(scenarios), key=scenarios.count)
            
            # 7. 记录场景分布统计
            self._update_scenario_coverage(symbol, scenario, current_time)
            
            return scenario
            
        except Exception as e:
            print(f"[警告] 场景标签计算失败: {e}")
            return 'Q_L'  # 默认场景
    
    def calculate_adaptive_weights(self, symbol: str, ofi_data: pd.DataFrame, 
                                 cvd_data: pd.DataFrame, lookback_hours: int = 24):
        """升级版自适应权重：防抖机制 + 日间波动控制"""
        try:
            if symbol not in self.adaptive_weights:
                self.adaptive_weights[symbol] = {'w_ofi': 0.6, 'w_cvd': 0.4}
            
            # 获取当前权重
            current_weights = self.adaptive_weights[symbol]
            old_w_ofi = current_weights['w_ofi']
            old_w_cvd = current_weights['w_cvd']
            
            # 计算OFI和CVD对未来收益的IC
            ofi_ic = self._calculate_ic(ofi_data, 'z_ofi', lookback_hours) if 'z_ofi' in ofi_data.columns else 0.0
            cvd_ic = self._calculate_ic(cvd_data, 'z_cvd', lookback_hours) if 'z_cvd' in cvd_data.columns else 0.0
            
            # 1. 滚动IC→权重时加入收缩/裁剪
            lambda_shrink = 0.1  # 收缩参数
            ofi_ic_shrunk = ofi_ic * (1 - lambda_shrink) + 0.5 * lambda_shrink
            cvd_ic_shrunk = cvd_ic * (1 - lambda_shrink) + 0.5 * lambda_shrink
            
            # 权重归一化
            total_ic = max(ofi_ic_shrunk, 0) + max(cvd_ic_shrunk, 0)
            if total_ic > 0:
                new_w_ofi = max(ofi_ic_shrunk, 0) / total_ic
                new_w_cvd = max(cvd_ic_shrunk, 0) / total_ic
            else:
                new_w_ofi, new_w_cvd = 0.6, 0.4  # 默认权重
            
            # 2. 防抖换向：仅当 |Δw| > 0.15 才应用新权重
            delta_w_ofi = abs(new_w_ofi - old_w_ofi)
            delta_w_cvd = abs(new_w_cvd - old_w_cvd)
            
            if delta_w_ofi > 0.15 or delta_w_cvd > 0.15:
                # 应用新权重
                w_ofi = new_w_ofi
                w_cvd = new_w_cvd
                print(f"[初始化] {symbol} 权重更新: OFI={w_ofi:.3f}, CVD={w_cvd:.3f}")
            else:
                # 延续旧权重
                w_ofi = old_w_ofi
                w_cvd = old_w_cvd
                print(f"[初始化] {symbol} 权重保持: OFI={w_ofi:.3f}, CVD={w_cvd:.3f}")
            
            # 3. 防过拟合裁剪
            w_ofi = np.clip(w_ofi, 0.2, 0.8)
            w_cvd = np.clip(w_cvd, 0.2, 0.8)
            
            # 4. 日间波动控制：检查权重变化是否过大
            total_delta = abs(w_ofi - old_w_ofi) + abs(w_cvd - old_w_cvd)
            if total_delta > 0.2:  # 日间波动限制
                print(f"[警告] {symbol} 权重变化过大，限制调整")
                w_ofi = old_w_ofi + (w_ofi - old_w_ofi) * 0.5
                w_cvd = old_w_cvd + (w_cvd - old_w_cvd) * 0.5
            
            # 更新权重
            self.adaptive_weights[symbol] = {'w_ofi': w_ofi, 'w_cvd': w_cvd}
            
            return w_ofi, w_cvd
            
        except Exception as e:
            print(f"[警告] 自适应权重计算失败: {e}")
            return 0.6, 0.4  # 默认权重
    
    def _calculate_ic(self, data: pd.DataFrame, signal_column: str, lookback_hours: int):
        """计算信息系数(IC)"""
        try:
            if len(data) < 10:
                return 0.0
            
            # 计算未来收益
            data_sorted = data.sort_values('timestamp')
            data_sorted['future_return'] = data_sorted['price'].pct_change().shift(-1)
            
            # 计算IC：信号与未来收益的相关性
            valid_mask = ~(data_sorted['future_return'].isna() | data_sorted[signal_column].isna())
            
            if valid_mask.sum() < 5:
                return 0.0
            
            ic = np.corrcoef(
                data_sorted.loc[valid_mask, signal_column],
                data_sorted.loc[valid_mask, 'future_return']
            )[0, 1]
            
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            print(f"[警告] IC计算失败: {e}")
            return 0.0
    
    def can_reverse(self, symbol: str, new_score: float, timestamp: datetime, scenario: str, 
                   best_bid: float = None, best_ask: float = None, mid_price: float = None):
        """升级版反转闸门：防止频繁反向开仓"""
        if symbol not in self.last_flip_time:
            return True
        
        gate = self.SCENE_GATE.get(scenario, self.SCENE_GATE["Q_L"])
        time_since_flip = (timestamp - self.last_flip_time[symbol]).total_seconds()
        
        # 1. 分场景冷却时间检查
        cooldown_ok = time_since_flip >= gate["cooldown_s"]
        if not cooldown_ok:
            return False
        
        # 2. 翻转额外裕度：信号强度要求（优化版）
        strength_ok = abs(new_score) >= (gate["enter"] + 0.3)  # 0.5 → 0.3 (降低额外裕度)
        if not strength_ok:
            return False
        
        # 3. 信号稳定性：连续稳定信号≥2/3
        stability_ok = self._check_signal_stability_for_reverse(symbol, new_score)
        if not stability_ok:
            return False
        
        # 4. 最小位移/点差过滤
        if best_bid is not None and best_ask is not None and mid_price is not None:
            move_ok = self._check_minimum_movement(symbol, mid_price)
            spread_ok = (best_ask - best_bid) <= self._get_max_spread_price(scenario, mid_price)
            if not (move_ok and spread_ok):
                return False
        
        # 5. 翻转频率上限：30分钟内最多2次
        freq_ok = self._check_reverse_frequency(symbol, timestamp)
        if not freq_ok:
            return False
        
        return True
    
    def _check_signal_stability_for_reverse(self, symbol: str, new_score: float):
        """检查信号稳定性用于反转判定"""
        if symbol not in self.signal_history:
            return True
        
        history = list(self.signal_history[symbol])
        if len(history) < 3:
            return True
        
        # 检查最近3个信号中是否有≥2个同方向
        target_side = -1 if new_score > 0 else 1  # 反转方向
        same_side_count = sum(1 for s in history[-3:] if s == target_side)
        return same_side_count >= 2
    
    def _check_minimum_movement(self, symbol: str, current_mid: float):
        """检查最小价格位移"""
        if symbol not in self.last_mid_price:
            self.last_mid_price[symbol] = current_mid
            return True
        
        last_mid = self.last_mid_price[symbol]
        move_ticks = abs(current_mid - last_mid) / self.tick_size
        return move_ticks >= self.min_move_ticks
    
    def _get_max_spread(self, scenario: str):
        """获取最大点差限制（bps）"""
        return self.max_spread_bps.get(scenario, 3.0)
    
    def _get_max_spread_price(self, scenario: str, mid: float) -> float:
        """获取最大点差限制（价格单位）"""
        bps = self.max_spread_bps.get(scenario, 3.0)
        return mid * bps / 10000.0
    
    def _check_reverse_frequency(self, symbol: str, timestamp: datetime):
        """检查翻转频率限制"""
        if symbol not in self.reverse_count_30m:
            self.reverse_count_30m[symbol] = deque(maxlen=10)
        
        # 清理30分钟前的记录
        cutoff = timestamp - timedelta(minutes=30)
        while (self.reverse_count_30m[symbol] and 
               self.reverse_count_30m[symbol][0] < cutoff):
            self.reverse_count_30m[symbol].popleft()
        
        # 检查是否超过限制
        return len(self.reverse_count_30m[symbol]) < 2
    
    def _record_reverse(self, symbol: str, timestamp: datetime):
        """记录反转事件"""
        if symbol not in self.reverse_count_30m:
            self.reverse_count_30m[symbol] = deque(maxlen=10)
        self.reverse_count_30m[symbol].append(timestamp)
    
    def _calculate_adaptive_percentiles(self, price_changes, trade_frequency):
        """计算自适应分位阈值"""
        # 初始分位：50/50
        p_vol = 50
        p_act = 50
        
        # 基于数据质量调整
        if len(price_changes) > 0:
            vol_std = np.std(price_changes)
            if vol_std > 0.01:  # 高波动市场，提高阈值
                p_vol = 60
            elif vol_std < 0.005:  # 低波动市场，降低阈值
                p_vol = 40
        
        if trade_frequency > 10:  # 高活跃市场
            p_act = 60
        elif trade_frequency < 2:  # 低活跃市场
            p_act = 40
            
        return p_vol, p_act
    
    def _rebalance_cuts(self, symbol: str, p_vol: int, p_act: int, current_time: datetime):
        """强制均衡：检查覆盖约束并调整分位阈值"""
        if symbol not in self.scenario_coverage:
            self.scenario_coverage[symbol] = {}
        
        # 计算4小时窗口内的场景覆盖
        cutoff_time = current_time - timedelta(minutes=self.coverage_window)
        total_count = 0
        coverage = {}
        
        for scenario in ["Q_L", "A_L", "A_H", "Q_H"]:
            if scenario in self.scenario_coverage[symbol]:
                # 清理过期数据
                recent_counts = [count for timestamp, count in self.scenario_coverage[symbol][scenario] 
                               if timestamp >= cutoff_time]
                coverage[scenario] = sum(recent_counts)
                total_count += coverage[scenario]
            else:
                coverage[scenario] = 0
        
        if total_count > 0:
            # 检查每个象限是否达到15%最小覆盖
            for scenario in ["Q_L", "A_L", "A_H", "Q_H"]:
                coverage_pct = coverage[scenario] / total_count
                if coverage_pct < self.min_coverage_percent:
                    # 朝中位数回调：哪边缺，就把对应分位往50%挪5pp
                    if scenario in ["Q_L", "A_L"]:  # Low volatility
                        p_vol = p_vol + (50 - p_vol) * 0.1
                    if scenario in ["Q_L", "Q_H"]:  # Quiet regime
                        p_act = p_act + (50 - p_act) * 0.1
        
        return round(p_vol), round(p_act)
    
    def _update_scenario_coverage(self, symbol: str, scenario: str, timestamp: datetime):
        """更新场景覆盖统计"""
        if symbol not in self.scenario_coverage:
            self.scenario_coverage[symbol] = {}
        
        if scenario not in self.scenario_coverage[symbol]:
            self.scenario_coverage[symbol][scenario] = []
        
        self.scenario_coverage[symbol][scenario].append((timestamp, 1))
        
        # 清理过期数据
        cutoff_time = timestamp - timedelta(minutes=self.coverage_window)
        self.scenario_coverage[symbol][scenario] = [
            (ts, count) for ts, count in self.scenario_coverage[symbol][scenario]
            if ts >= cutoff_time
        ]
    
    def check_signal_stability(self, symbol: str, fusion_score: float):
        """信号稳态判定：多数表决"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = deque(maxlen=3)
        
        self.signal_history[symbol].append(1 if fusion_score > 0 else -1)
        
        # 多数表决：近3个bar中≥2个同号才开仓
        if len(self.signal_history[symbol]) < 3:
            return False
        
        recent_signals = list(self.signal_history[symbol])
        return abs(sum(recent_signals)) >= 2
    
    def _get_flip_count(self, symbol: str, timestamp: datetime):
        """获取当前小时的翻转次数"""
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        if symbol not in self.flip_count:
            self.flip_count[symbol] = {}
        return self.flip_count[symbol].get(hour_key, 0)
    
    def _increment_flip_count(self, symbol: str, timestamp: datetime):
        """增加翻转次数"""
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        if symbol not in self.flip_count:
            self.flip_count[symbol] = {}
        self.flip_count[symbol][hour_key] = self.flip_count[symbol].get(hour_key, 0) + 1
    
    def check_risk_management(self, symbol: str, price: float, timestamp: datetime):
        """检查风险管理规则"""
        if symbol not in self.positions:
            return None
            
        trade = self.positions[symbol]
        entry_price = trade['entry_price']
        entry_time = trade['entry_time']
        
        # 计算当前PnL
        if trade['side'] == 'long':
            current_pnl_bps = (price - entry_price) / entry_price * 10000
        else:
            current_pnl_bps = (entry_price - price) / entry_price * 10000
        
        # 1. 对称止损检查
        stop_loss_bps = 20  # 从配置读取
        if current_pnl_bps <= -stop_loss_bps:
            return self.close_position(symbol, price, timestamp, "stop_loss")
        
        # 2. 分级止盈检查
        if current_pnl_bps >= 10:  # 第一级止盈
            if 'level1_closed' not in trade:
                trade['level1_closed'] = True
                # 部分平仓30%
                return self.close_position(symbol, price, timestamp, "take_profit_level1")
        elif current_pnl_bps >= 20:  # 第二级止盈
            if 'level2_closed' not in trade:
                trade['level2_closed'] = True
                # 部分平仓50%
                return self.close_position(symbol, price, timestamp, "take_profit_level2")
        elif current_pnl_bps >= 40:  # 第三级止盈
            if 'level3_closed' not in trade:
                trade['level3_closed'] = True
                # 全部平仓
                return self.close_position(symbol, price, timestamp, "take_profit_level3")
        
        # 3. 时间止损检查
        time_elapsed = (timestamp - entry_time).total_seconds()
        if time_elapsed >= 300:  # 5分钟时间止损
            return self.close_position(symbol, price, timestamp, "time_stop_loss")
        
        return None

    def close_position(self, symbol: str, price: float, timestamp: datetime, reason: str = "manual"):
        """平仓"""
        if symbol not in self.positions:
            return None
            
        trade = self.positions[symbol]
        trade['exit_price'] = price
        trade['exit_time'] = timestamp
        trade['exit_reason'] = reason
        trade['status'] = 'closed'
        
        # 计算PnL
        if trade['side'] == 'long':
            net_pnl_bps = (price - trade['entry_price']) / trade['entry_price'] * 10000 - self.cost_bps
        else:
            net_pnl_bps = (trade['entry_price'] - price) / trade['entry_price'] * 10000 - self.cost_bps
        
        trade['net_pnl_bps'] = net_pnl_bps
        
        # 更新KPI
        scenario = trade['scenario_2x2']
        if scenario not in self.kpis:
            self.kpis[scenario] = {'trades': 0, 'pnl': 0, 'win_rate': 0, 'sharpe': 0}
        
        self.kpis[scenario]['trades'] += 1
        self.kpis[scenario]['pnl'] += net_pnl_bps
        
        # 移除持仓
        del self.positions[symbol]
        
        print(f"[离场] {scenario} {trade['side']} 离场: {symbol} @ {price:.4f}, "
              f"PnL={net_pnl_bps:.2f}bps, 原因={reason}")
        
        # 记录结构化日志
        self.log_entries.append({
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'action': 'close',
            'side': trade['side'],
            'price': price,
            'exit_reason': reason,
            'scenario_2x2': scenario,
            'net_pnl_bps': net_pnl_bps,
            'config_version': self.manager.get_scenario_stats().get('version', 'unknown') if self.manager else 'unknown'
        })
        
        return trade
    
    def check_exit_conditions(self, symbol: str, current_price: float, 
                            current_fusion_score: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """检查离场条件"""
        
        if symbol not in self.positions:
            return None
        
        trade = self.positions[symbol]
        scenario = trade['scenario_2x2']
        
        # 获取场景参数
        try:
            params = self.manager.get_params_for_scenario(scenario, trade['side'])
        except Exception as e:
            print(f"[警告] 获取场景参数失败: {e}")
            return None
        
        # 检查止损止盈
        exit_reason = None
        exit_price = current_price
        
        if trade['side'] == 'long':
            if current_price <= trade['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = trade['stop_loss']
            elif current_price >= trade['take_profit']:
                exit_reason = 'take_profit'
                exit_price = trade['take_profit']
        else:  # short
            if current_price >= trade['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = trade['stop_loss']
            elif current_price <= trade['take_profit']:
                exit_reason = 'take_profit'
                exit_price = trade['take_profit']
        
        # 使用场景化退出阈值
        gate = self.SCENE_GATE.get(scenario, self.SCENE_GATE["Q_L"])
        if not exit_reason and abs(current_fusion_score) <= gate["exit"]:
            exit_reason = 'scenario_exit'
        
        # 升级版ATR移动止损（立改清单优化）
        if not exit_reason:
            atr_multiplier = {
                "Q_H": 2.2, "A_H": 2.0, "A_L": 1.6, "Q_L": 1.6
            }
            
            # 计算ATR（简化版，实际应使用真实ATR）
            price_change = abs(current_price - trade['entry_price']) / trade['entry_price']
            atr_stop = price_change * atr_multiplier.get(scenario, 2.0)
            
            if abs(current_fusion_score) <= atr_stop:
                exit_reason = 'atr_stop'
        
        # 升级版时间止盈/止损（立改清单优化）
        if not exit_reason:
            time_limits = {
                "Q_H": 120, "A_H": 90, "A_L": 90, "Q_L": 120  # A场景90s、Q场景120s
            }
            
            hold_time = (timestamp - trade['entry_time']).total_seconds()
            if hold_time >= time_limits.get(scenario, 90):
                exit_reason = 'time_stop'
        
        # 交易节流：弱信号区域检测
        if not exit_reason:
            is_weak_signal = self._check_weak_signal_region(symbol, current_price, timestamp)
            if is_weak_signal:
                # 弱信号区域，降低确认率目标并禁止反向开仓
                exit_reason = 'weak_signal_throttle'
        
        # 检查最大持仓时间
        if not exit_reason:
            max_hold_time = timedelta(seconds=params.get('MAX_HOLD_S', 600))
            if timestamp - trade['entry_time'] >= max_hold_time:
                exit_reason = 'timeout'
        
        if exit_reason:
            # 计算PnL
            if trade['side'] == 'long':
                pnl_bps = (exit_price - trade['entry_price']) / trade['entry_price'] * 10000
            else:
                pnl_bps = (trade['entry_price'] - exit_price) / trade['entry_price'] * 10000
            
            # 扣除交易成本
            cost_bps = 3.0  # 3bps交易成本
            net_pnl_bps = pnl_bps - cost_bps
            
            # 更新交易记录
            trade.update({
                'exit_price': exit_price,
                'exit_time': timestamp,
                'exit_reason': exit_reason,
                'pnl_bps': pnl_bps,
                'net_pnl_bps': net_pnl_bps,
                'status': 'closed'
            })
            
            # 更新KPI
            self.kpis[scenario]['trades'] += 1
            self.kpis[scenario]['pnl'] += net_pnl_bps
            
            # 移除持仓
            del self.positions[symbol]
            
            print(f"[离场] {scenario} {trade['side']} 离场: {symbol} @ {exit_price:.4f}, "
                  f"PnL={net_pnl_bps:.2f}bps, 原因={exit_reason}")
            
            return trade
        
        return None
    
    def simulate_from_data(self, symbol: str = None, duration_minutes: int = 60):
        """从数据文件模拟交易 - 使用核心算法处理信号"""
        
        if symbol is None:
            symbol = self.symbol
            
        print(f"[统计] 模拟交易对: {symbol}")
        print(f"[时间] 模拟时长: {duration_minutes}分钟")
        
        try:
            # 读取数据 - 扫描所有可用日期的数据
            data_base_dir = Path("C:/Users/user/Desktop/ofi_cvd_framework/ofi_cvd_framework/v13_ofi_ai_system/data/ofi_cvd")
            
            # 扫描所有日期目录
            date_dirs = [d for d in data_base_dir.iterdir() if d.is_dir() and d.name.startswith("date=")]
            print(f"[统计] 发现日期目录: {len(date_dirs)}个")
            
            # 收集所有数据文件
            prices_files = []
            ofi_files = []
            cvd_files = []
            
            for date_dir in date_dirs:
                prices_dir = date_dir / f"symbol={symbol}/kind=prices"
                ofi_dir = date_dir / f"symbol={symbol}/kind=ofi"
                cvd_dir = date_dir / f"symbol={symbol}/kind=cvd"
                
                if prices_dir.exists():
                    prices_files.extend(list(prices_dir.glob("*.parquet")))
                if ofi_dir.exists():
                    ofi_files.extend(list(ofi_dir.glob("*.parquet")))
                if cvd_dir.exists():
                    cvd_files.extend(list(cvd_dir.glob("*.parquet")))
            
            print(f"[统计] 扫描完成，发现:")
            print(f"   价格文件: {len(prices_files)}个")
            print(f"   OFI文件: {len(ofi_files)}个")
            print(f"   CVD文件: {len(cvd_files)}个")
            
            if not prices_files:
                print(f"[失败] 数据文件不存在: {prices_dir}")
                return
            
            # 读取所有价格数据文件
            print(f"[统计] 开始加载价格数据...")
            prices_dfs = []
            total_records = 0
            
            for i, file in enumerate(prices_files):
                try:
                    df = pd.read_parquet(file)
                    prices_dfs.append(df)
                    total_records += len(df)
                    print(f"[统计] 加载文件 {i+1}/{len(prices_files)}: {file.name} - {len(df)}条记录")
                except Exception as e:
                    print(f"[警告] 跳过文件 {file.name}: {e}")
            
            if not prices_dfs:
                print(f"[失败] 无法加载任何价格数据")
                return
                
            # 合并所有价格数据
            print(f"[统计] 合并价格数据...")
            prices_df = pd.concat(prices_dfs, ignore_index=True)
            prices_df['timestamp'] = pd.to_datetime(prices_df['ts_ms'], unit='ms')
            prices_df = prices_df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"[统计] 价格数据合并完成，总记录数: {len(prices_df)}")
            print(f"[趋势] 数据范围: {prices_df['timestamp'].min()} 到 {prices_df['timestamp'].max()}")
            
            # 读取OFI数据
            ofi_dfs = []
            for file in ofi_files:
                try:
                    df = pd.read_parquet(file)
                    ofi_dfs.append(df)
                except Exception as e:
                    print(f"[警告] 跳过OFI文件 {file.name}: {e}")
            ofi_df = pd.concat(ofi_dfs, ignore_index=True) if ofi_dfs else pd.DataFrame()
            if not ofi_df.empty:
                ofi_df['timestamp'] = pd.to_datetime(ofi_df['ts_ms'], unit='ms')
            
            # 读取CVD数据
            cvd_dfs = []
            for file in cvd_files:
                try:
                    df = pd.read_parquet(file)
                    cvd_dfs.append(df)
                except Exception as e:
                    print(f"[警告] 跳过CVD文件 {file.name}: {e}")
            cvd_df = pd.concat(cvd_dfs, ignore_index=True) if cvd_dfs else pd.DataFrame()
            if not cvd_df.empty:
                cvd_df['timestamp'] = pd.to_datetime(cvd_df['ts_ms'], unit='ms')
            
            print(f"[统计] OFI数据: {len(ofi_df)}条记录")
            print(f"[统计] CVD数据: {len(cvd_df)}条记录")
            
            # 限制模拟时长 - 24小时测试
            start_time = prices_df['timestamp'].iloc[0]
            if duration_minutes >= 1440:  # 24小时或更长
                # 使用所有可用数据
                end_time = prices_df['timestamp'].iloc[-1]
                print(f"[趋势] 使用全部数据: {start_time} 到 {end_time}")
            else:
                # 限制到指定时长
                end_time = start_time + timedelta(minutes=duration_minutes)
                prices_df = prices_df[prices_df['timestamp'] <= end_time]
                print(f"[趋势] 数据范围: {start_time} 到 {end_time}")
            
            print(f"[统计] 价格记录数: {len(prices_df)}")
            print(f"[统计] 实际模拟时长: {(end_time - start_time).total_seconds() / 3600:.1f}小时")
            
            # 使用 merge_asof 优化数据查询性能
            merged_df = prices_df.copy()
            
            if not ofi_df.empty:
                ofi_df_sorted = ofi_df.sort_values('timestamp')
                merged_df = pd.merge_asof(
                    merged_df.sort_values('timestamp'),
                    ofi_df_sorted[['timestamp', 'ofi_z']].rename(columns={'ofi_z': 'z_ofi'}),
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta(seconds=5)
                )
            else:
                merged_df['z_ofi'] = 0.0
            
            if not cvd_df.empty:
                cvd_df_sorted = cvd_df.sort_values('timestamp')
                merged_df = pd.merge_asof(
                    merged_df.sort_values('timestamp'),
                    cvd_df_sorted[['timestamp', 'z_cvd']],
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta(seconds=5)
                )
            else:
                merged_df['z_cvd'] = 0.0
            
            # 填充缺失值
            merged_df['z_ofi'] = merged_df['z_ofi'].fillna(0.0)
            merged_df['z_cvd'] = merged_df['z_cvd'].fillna(0.0)
            
            print(f"成功 数据合并完成，记录数: {len(merged_df)}")
            
            # 模拟交易 - 使用核心算法处理信号
            for _, row in merged_df.iterrows():
                timestamp = row['timestamp']
                ts_ms = int(timestamp.timestamp() * 1000)
                price = row['price']
                
                # 获取OFI和CVD数据（已合并）
                z_ofi = row.get('z_ofi', 0.0)
                z_cvd = row.get('z_cvd', 0.0)
                
                # 第二步：使用自适应场景标签
                scenario_2x2 = self.calculate_adaptive_scenario_labels(
                    symbol, timestamp, merged_df
                )
                
                # 使用核心算法一站式处理信号
                ts_ms = int(timestamp.timestamp() * 1000)
                
                # 风险管理检查（在信号处理前）
                if self.positions:
                    for symbol in list(self.positions.keys()):
                        risk_result = self.check_risk_management(symbol, price, timestamp)
                        if risk_result:
                            print(f"[风险管理] {symbol} 触发风险管理: {risk_result['exit_reason']}")
                
                # 关键修复：确保OFI/CVD计算器正确更新
                # 从数据中提取订单簿信息用于OFI计算
                if 'best_bid' in row and 'best_ask' in row:
                    # 构建完整5档订单簿快照用于OFI计算
                    # 基于最优价构建L1-L5档深度数据
                    best_bid = row['best_bid']
                    best_ask = row['best_ask']
                    spread = best_ask - best_bid
                    
                    # 构建5档买盘（价格递减，数量递减）
                    bids = [
                        [best_bid, 1.0],                    # L1: 最优买价
                        [best_bid - spread * 0.1, 0.8],     # L2: 下1档
                        [best_bid - spread * 0.2, 0.6],     # L3: 下2档
                        [best_bid - spread * 0.3, 0.4],     # L4: 下3档
                        [best_bid - spread * 0.4, 0.2]      # L5: 下4档
                    ]
                    
                    # 构建5档卖盘（价格递增，数量递减）
                    asks = [
                        [best_ask, 1.0],                    # L1: 最优卖价
                        [best_ask + spread * 0.1, 0.8],     # L2: 上1档
                        [best_ask + spread * 0.2, 0.6],     # L3: 上2档
                        [best_ask + spread * 0.3, 0.4],     # L4: 上3档
                        [best_ask + spread * 0.4, 0.2]      # L5: 上4档
                    ]
                    
                    # 更新OFI计算器（通过核心算法方法，确保统计正确）
                    # 强制打通OFI链路：确保OFI更新数>0
                    self.core_algo.update_ofi(bids, asks, ts_ms)
                    
                    # 记录OFI更新状态用于诊断
                    if not hasattr(self, '_ofi_update_count'):
                        self._ofi_update_count = 0
                    self._ofi_update_count += 1
                
                # 从数据中提取交易信息用于CVD计算（通过核心算法方法，确保统计正确）
                if 'is_buy' in row:
                    # 更新CVD计算器
                    self.core_algo.update_cvd(
                        price=price, qty=1.0, is_buy=row['is_buy'], event_time_ms=ts_ms
                    )
                else:
                    # 使用Tick Rule
                    self.core_algo.update_cvd(
                        price=price, qty=1.0, is_buy=None, event_time_ms=ts_ms
                    )
                
                # 从数据帧估算质量指标
                mid = (merged_df["best_bid"].iloc[-1] + merged_df["best_ask"].iloc[-1]) / 2 if "best_bid" in merged_df.columns else price
                spread_bps = ((merged_df["best_ask"].iloc[-1] - merged_df["best_bid"].iloc[-1]) / mid * 10000) if "best_bid" in merged_df.columns else 5.0
                realized_vol = float(merged_df["ret"].rolling(60).std().iloc[-1]) if "ret" in merged_df.columns else 0.01
                trade_rate = float(merged_df["trades_1m"].iloc[-1]) if "trades_1m" in merged_df.columns else 60.0
                missing_msgs_rate = 0.0
                
                # 调用核心算法处理信号
                sig = self.core_algo.process_signal(
                    ts_ms=ts_ms, symbol=symbol, z_ofi=z_ofi, z_cvd=z_cvd, price=price,
                    trade_rate=trade_rate, realized_vol=realized_vol,
                    spread_bps=spread_bps, missing_msgs_rate=missing_msgs_rate
                )
                
                # 护栏/确认 → 统一用成熟组件结果
                if sig.gating:
                    print(f"[BLOCK] 护栏触发: {self.core_algo.guard_reason}")
                    continue
                
                if not sig.confirm:
                    print("[警告] 信号未确认，跳过交易")
                    continue
                
                # 使用核心算法计算的融合分数（统一口径）
                fusion_score = sig.score
                
                # 结构化信号日志（使用SafeJsonlWriter版）
                self.core_algo.log_signal(sig, output_dir=os.getenv("V13_OUTPUT_DIR", "./runtime"))
                
                # 检查离场条件
                self.check_exit_conditions(symbol, price, fusion_score, timestamp)
                
                # 检查入场条件（使用统一信号）
                self.simulate_trade(symbol, price, fusion_score, scenario_2x2, timestamp)
            
            # 强制平仓所有持仓
            for symbol, trade in list(self.positions.items()):
                # 使用最后一次价格
                last_price = prices_df['price'].iloc[-1] if len(prices_df) > 0 else 3000.0
                self.check_exit_conditions(symbol, last_price, 0.0, 
                                         trade['entry_time'] + timedelta(minutes=duration_minutes))
            
            print("成功 交易模拟完成")
            
        except Exception as e:
            print(f"[失败] 交易模拟失败: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_kpis(self):
        """计算KPI指标"""
        print("\n[统计] 计算KPI指标...")
        
        for scenario, kpi in self.kpis.items():
            if kpi['trades'] > 0:
                # 计算胜率
                winning_trades = sum(1 for trade in self.trades 
                                   if trade['scenario_2x2'] == scenario and trade.get('net_pnl_bps', 0) > 0)
                kpi['win_rate'] = winning_trades / kpi['trades']
                
                # 计算夏普比率（简化版）
                if kpi['trades'] > 1:
                    scenario_trades = [trade['net_pnl_bps'] for trade in self.trades 
                                     if trade['scenario_2x2'] == scenario and 'net_pnl_bps' in trade]
                    if scenario_trades:
                        kpi['sharpe'] = np.mean(scenario_trades) / (np.std(scenario_trades) + 1e-6)
        
        return self.kpis
    
    def _calculate_extended_kpis(self):
        """计算扩展KPI指标"""
        extended_kpis = {
            'max_drawdown': 0.0,
            'expected_return': 0.0,
            'payoff_ratio': 0.0,
            'avg_hold_time': 0.0,
            'scenario_drawdowns': {}
        }
        
        if not self.trades:
            return extended_kpis
        
        # 计算最大回撤
        cumulative_pnl = 0
        peak_pnl = 0
        max_dd = 0
        
        for trade in self.trades:
            if 'net_pnl_bps' in trade:
                cumulative_pnl += trade['net_pnl_bps']
                peak_pnl = max(peak_pnl, cumulative_pnl)
                drawdown = peak_pnl - cumulative_pnl
                max_dd = max(max_dd, drawdown)
        
        extended_kpis['max_drawdown'] = max_dd
        
        # 计算期望收益
        winning_trades = [t['net_pnl_bps'] for t in self.trades if t.get('net_pnl_bps', 0) > 0]
        losing_trades = [t['net_pnl_bps'] for t in self.trades if t.get('net_pnl_bps', 0) < 0]
        
        if winning_trades and losing_trades:
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean(winning_trades)
            avg_loss = abs(np.mean(losing_trades))
            extended_kpis['expected_return'] = win_rate * avg_win - (1 - win_rate) * avg_loss
            extended_kpis['payoff_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 计算平均持仓时长
        hold_times = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                hold_time = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
                hold_times.append(hold_time)
        
        if hold_times:
            extended_kpis['avg_hold_time'] = np.mean(hold_times)
        
        # 计算分场景回撤
        for scenario in self.kpis.keys():
            scenario_trades = [t for t in self.trades if t.get('scenario_2x2') == scenario and 'net_pnl_bps' in t]
            if scenario_trades:
                scenario_cumulative = 0
                scenario_peak = 0
                scenario_dd = 0
                for trade in scenario_trades:
                    scenario_cumulative += trade['net_pnl_bps']
                    scenario_peak = max(scenario_peak, scenario_cumulative)
                    drawdown = scenario_peak - scenario_cumulative
                    scenario_dd = max(scenario_dd, drawdown)
                extended_kpis['scenario_drawdowns'][scenario] = scenario_dd
        
        return extended_kpis
    
    def _check_stage_targets(self, extended_kpis):
        """检查阶段目标对齐"""
        print(f"\n[目标] 阶段目标对齐检查:")
        
        # 阶段2目标：胜率>55%、日交易5-20、PnL正、回撤<10%
        total_trades = sum(kpi['trades'] for kpi in self.kpis.values())
        total_pnl = sum(kpi['pnl'] for kpi in self.kpis.values())
        
        # 计算总体胜率
        winning_trades = sum(1 for trade in self.trades if trade.get('net_pnl_bps', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 检查目标
        targets = {
            "胜率>55%": win_rate > 0.55,
            "PnL>0": total_pnl > 0,
            "回撤<10%": extended_kpis['max_drawdown'] < 10.0,
            "交易数5-20": 5 <= total_trades <= 20
        }
        
        for target, passed in targets.items():
            status = "[通过]" if passed else "[未通过]"
            print(f"   {status} {target}")
        
        # 综合评估
        passed_targets = sum(targets.values())
        total_targets = len(targets)
        print(f"\n[评估] 目标达成率: {passed_targets}/{total_targets} ({passed_targets/total_targets*100:.1f}%)")
        
        if passed_targets == total_targets:
            print("[成功] 所有阶段目标达成！")
        else:
            print("[警告] 部分目标未达成，需要进一步优化")
    
    def print_results(self):
        """打印结果 - 扩展版KPI"""
        print("\n[结果] 纸上交易模拟结果:")
        print("=" * 80)
        
        total_trades = sum(kpi['trades'] for kpi in self.kpis.values())
        total_pnl = sum(kpi['pnl'] for kpi in self.kpis.values())
        
        # 计算扩展KPI
        extended_kpis = self._calculate_extended_kpis()
        
        print(f"[统计] 总体统计:")
        print(f"   总交易数: {total_trades}")
        print(f"   总PnL: {total_pnl:.2f}bps")
        print(f"   最大回撤: {extended_kpis['max_drawdown']:.2f}bps")
        print(f"   期望收益: {extended_kpis['expected_return']:.2f}bps")
        print(f"   盈亏比: {extended_kpis['payoff_ratio']:.2f}")
        print(f"   平均持仓时长: {extended_kpis['avg_hold_time']:.1f}分钟")
        
        print(f"\n[趋势] 分场景表现:")
        for scenario, kpi in self.kpis.items():
            if kpi['trades'] > 0:
                print(f"   {scenario}:")
                print(f"     交易数: {kpi['trades']}")
                print(f"     PnL: {kpi['pnl']:.2f}bps")
                print(f"     胜率: {kpi['win_rate']:.2%}")
                print(f"     夏普: {kpi['sharpe']:.3f}")
                print(f"     回撤: {extended_kpis['scenario_drawdowns'].get(scenario, 0):.2f}bps")
        
        # 反向开仓分析
        reverse_trades = sum(1 for trade in self.trades if trade.get('exit_reason') == 'reverse_open')
        reverse_ratio = reverse_trades / total_trades if total_trades > 0 else 0
        print(f"\n[风险] 反向开仓分析:")
        print(f"   反向开仓数: {reverse_trades}")
        print(f"   反向开仓占比: {reverse_ratio:.1%}")
        
        # 阶段目标对齐检查
        self._check_stage_targets(extended_kpis)
        
        # 闸门原因统计诊断
        self._print_gate_reason_diagnostics()
        
        print("\n[结果] 金丝雀重点场景:")
        ql_kpi = self.kpis['Q_L']
        al_kpi = self.kpis['A_L']
        
        print(f"   Q_L (Sharpe=0.717预期):")
        print(f"     交易数: {ql_kpi['trades']}")
        print(f"     PnL: {ql_kpi['pnl']:.2f}bps")
        print(f"     胜率: {ql_kpi['win_rate']:.2%}")
        print(f"     夏普: {ql_kpi['sharpe']:.3f}")
        
        print(f"   A_L (Sharpe=0.301预期):")
        print(f"     交易数: {al_kpi['trades']}")
        print(f"     PnL: {al_kpi['pnl']:.2f}bps")
        print(f"     胜率: {al_kpi['win_rate']:.2%}")
        print(f"     夏普: {al_kpi['sharpe']:.3f}")
        
        # 第二步优化：场景覆盖分析
        print(f"\n[统计] 场景覆盖分析:")
        active_scenarios = [scenario for scenario, kpi in self.kpis.items() if kpi['trades'] > 0]
        scenario_coverage = len(active_scenarios) / 4 * 100
        
        print(f"   活跃场景数: {len(active_scenarios)}/4 ({scenario_coverage:.1f}%)")
        print(f"   活跃场景: {active_scenarios}")
        
        # 反向开仓占比分析
        reverse_trades = sum(1 for trade in self.trades if trade.get('exit_reason') == 'reverse_open')
        reverse_ratio = reverse_trades / total_trades if total_trades > 0 else 0
        print(f"   反向开仓占比: {reverse_ratio:.1%}")
        
        # 自适应权重报告
        if hasattr(self, 'adaptive_weights') and self.adaptive_weights:
            print(f"\n[初始化] 自适应权重:")
            for symbol, weights in self.adaptive_weights.items():
                print(f"   {symbol}: OFI={weights['w_ofi']:.3f}, CVD={weights['w_cvd']:.3f}")
        
        # 场景分布统计
        if hasattr(self, 'scene_cache') and self.scene_cache:
            print(f"\n[趋势] 场景分布统计:")
            for symbol, cache in self.scene_cache.items():
                if 'scenario_counts' in cache:
                    print(f"   {symbol}: {cache['scenario_counts']}")
        
        # 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'scenario_kpis': self.kpis,
            'trades': [trade for trade in self.trades if 'net_pnl_bps' in trade]
        }
    
    def _print_gate_reason_diagnostics(self):
        """打印闸门原因统计诊断"""
        if not hasattr(self.core_algo, 'get_gate_reason_stats'):
            return
        
        print("\n[诊断] 闸门原因统计:")
        gate_stats = self.core_algo.get_gate_reason_stats()
        total_signals = self.core_algo.stats.get('total_updates', 0)
        
        if total_signals == 0:
            print("   无信号处理记录")
            return
        
        print(f"   总信号数: {total_signals}")
        print(f"   闸门原因分布:")
        
        for reason, count in gate_stats.items():
            if count > 0:
                percentage = (count / total_signals) * 100
                print(f"     {reason}: {count}次 ({percentage:.1f}%)")
        
        # 自动诊断建议
        suggestions = self.core_algo.check_gate_reason_thresholds()
        if suggestions:
            print(f"\n[建议] 参数优化建议:")
            for suggestion in suggestions:
                print(f"   - {suggestion}")
        else:
            print(f"\n[状态] 闸门原因分布正常，无需调整")
        
        # 保存结果到文件（移除这部分，因为results变量不在这个方法中定义）
        pass

def main():
    """主函数"""
    # 设置统一日志（仅在入口调用，不在import顶层）
    from logging_setup import setup_logging
    logger = setup_logging(os.getenv("V13_OUTPUT_DIR", "./runtime") + "/logs", "INFO")
    
    print("启动纸上交易模拟器（集成核心算法+2×2场景化）")
    
    try:
        # 创建模拟器
        print("创建模拟器...")
        simulator = PaperTradingSimulator(symbol="BTCUSDT")
        print("成功 模拟器创建成功")
        
        # 初始化
        print("初始化模拟器...")
        simulator.initialize()
        print("成功 模拟器初始化成功")
        
        # 运行模拟 - 24小时测试
        print("运行模拟...")
        simulator.simulate_from_data(duration_minutes=1440)  # 24小时 = 1440分钟
        print("成功 模拟运行成功")
        
        # 计算KPI
        print("计算KPI...")
        simulator.calculate_kpis()
        print("成功 KPI计算成功")
        
        # 打印结果
        print("打印结果...")
        simulator.print_results()
        print("成功 结果打印成功")
        
        # 打印核心算法统计信息
        if simulator.core_algo:
            print(f"\n[统计] 核心算法统计信息:")
            stats = simulator.core_algo.get_component_stats()
            for component, stat in stats.items():
                print(f"   {component}: {stat}")
        
        print("\n[完成] 纸上交易模拟完成！")
        
    except Exception as e:
        print(f"[失败] 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # 统一日志初始化（在程序入口调用）
    from logging_setup import setup_logging
    import os
    logger = setup_logging(os.path.join(os.getenv("V13_OUTPUT_DIR", "./runtime"), "logs"), "INFO")
    
    print("开始测试纸上交易模拟器...")
    try:
        success = main()
        print(f"测试结果: {'成功' if success else '失败'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
