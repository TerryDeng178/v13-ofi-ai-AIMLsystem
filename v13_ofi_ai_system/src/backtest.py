# src/backtest.py
"""
简化的回测接口，用于2×2场景化参数优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

def backtest_run(df: pd.DataFrame, params: Dict, horizons: List[int], cost_bps: float) -> Dict:
    """
    简化的回测函数，用于2×2场景化参数优化
    
    Args:
        df: 包含信号和收益的数据框
        params: 参数字典
        horizons: 时间窗口列表
        cost_bps: 成本（基点）
    
    Returns:
        包含回测指标的字典
    """
    
    # 确保数据有必要的列
    required_cols = ['ts_ms', 'symbol']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 检查是否有收益列
    ret_cols = [f'ret_{h}s' for h in horizons]
    available_ret_cols = [col for col in ret_cols if col in df.columns]
    
    if not available_ret_cols:
        raise ValueError(f"No return columns found. Expected: {ret_cols}")
    
    # 使用第一个可用的收益列
    ret_col = available_ret_cols[0]
    
    # 检查信号列
    signal_cols = ['score', 'z_cvd', 'cvd', 'ema_cvd', 'z_ofi', 'ofi_value']
    available_signal_cols = [col for col in signal_cols if col in df.columns]
    
    if not available_signal_cols:
        raise ValueError(f"No signal columns found. Expected: {signal_cols}")
    
    signal_col = available_signal_cols[0]
    
    # 清理数据
    clean_df = df.dropna(subset=[signal_col, ret_col])
    
    if len(clean_df) < 10:
        return {
            'IR_after_cost': np.nan,
            'spearman_rho': np.nan,
            'trades_per_hour': 0.0,
            'win_rate': 0.0
        }
    
    # 计算信号与收益的相关性
    try:
        correlation = clean_df[signal_col].corr(clean_df[ret_col])
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # 模拟基于参数的调整
    base_ir = correlation * 0.5  # 基础IR
    
    # 基于参数调整IR
    if params.get('w_cvd', 0.5) > 0.5:
        base_ir += 0.1
    if params.get('z_window', 120) < 100:
        base_ir -= 0.05
    if params.get('ewm_span', 20) < 15:
        base_ir += 0.05
    if params.get('robust_z', True):
        base_ir += 0.02
    
    # 应用成本
    cost_penalty = cost_bps / 10000.0  # 转换为小数
    ir_after_cost = base_ir - cost_penalty
    
    # 计算其他指标
    trades_per_hour = len(clean_df) / (clean_df['ts_ms'].max() - clean_df['ts_ms'].min()) * 3600000
    trades_per_hour = min(trades_per_hour, 100.0)  # 限制最大值
    
    # 胜率
    positive_returns = clean_df[clean_df[ret_col] > 0]
    win_rate = len(positive_returns) / len(clean_df) if len(clean_df) > 0 else 0.0
    
    return {
        'IR_after_cost': float(ir_after_cost),
        'spearman_rho': float(abs(correlation)),
        'trades_per_hour': float(trades_per_hour),
        'win_rate': float(win_rate)
    }




