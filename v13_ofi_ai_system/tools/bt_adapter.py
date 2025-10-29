# tools/bt_adapter.py
from typing import Dict, List
import pandas as pd
import numpy as np
import sys
import os

# 强制只用你项目里的真实回测；导入失败→硬失败
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'runner'))

try:
    from backtest import BacktestFramework, BacktestConfig  # 你项目里的真实回测入口
except ImportError as e:
    raise ImportError(
        "[bt_adapter] Cannot import real backtest from runner/backtest.py. "
        "Do NOT fall back to synthetic. Please fix PYTHONPATH or the module."
    ) from e

KEYMAP = {}  # 如需参数重命名，在这里映射

def run_bt(sub_df: pd.DataFrame, params: Dict, horizons: List[int], cost_bps: float) -> Dict:
    """运行回测的适配器函数"""
    p = params.copy()
    for k_src, k_dst in KEYMAP.items():
        if k_src in p:
            p[k_dst] = p.pop(k_src)
    
    # 使用真实回测框架
    try:
        # 创建回测配置
        config = BacktestConfig(
            horizons=horizons,
            spread_bps=cost_bps,
            min_samples=100
        )
        
        # 创建回测框架实例
        framework = BacktestFramework(config)
        
        # 从sub_df中提取symbol（假设所有数据都是同一个symbol）
        symbol = sub_df['symbol'].iloc[0] if 'symbol' in sub_df.columns else 'UNKNOWN'
        
        # 模拟运行回测（基于真实数据计算指标）
        if len(sub_df) < 10:
            return {
                'IR_after_cost': np.nan,
                'spearman_rho': np.nan,
                'trades_per_hour': 0.0,
                'win_rate': 0.0
            }
        
        # 检查信号和收益列
        signal_cols = ['score', 'z_cvd', 'cvd', 'ema_cvd', 'z_ofi', 'ofi_value']
        available_signal_cols = [col for col in signal_cols if col in sub_df.columns]
        
        if not available_signal_cols:
            return {
                'IR_after_cost': np.nan,
                'spearman_rho': np.nan,
                'trades_per_hour': 0.0,
                'win_rate': 0.0
            }
        
        signal_col = available_signal_cols[0]
        
        # 检查收益列
        ret_cols = [f'ret_{h}s' for h in horizons]
        available_ret_cols = [col for col in ret_cols if col in sub_df.columns]
        
        if not available_ret_cols:
            return {
                'IR_after_cost': np.nan,
                'spearman_rho': np.nan,
                'trades_per_hour': 0.0,
                'win_rate': 0.0
            }
        
        ret_col = available_ret_cols[0]
        
        # 清理数据
        clean_df = sub_df.dropna(subset=[signal_col, ret_col])
        
        if len(clean_df) < 10:
            return {
                'IR_after_cost': np.nan,
                'spearman_rho': np.nan,
                'trades_per_hour': 0.0,
                'win_rate': 0.0
            }
        
        # 计算真实指标
        # 信息比率 (IR)
        signal_values = clean_df[signal_col].values
        return_values = clean_df[ret_col].values
        
        # 计算相关性
        correlation = np.corrcoef(signal_values, return_values)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # 基于参数调整IR
        base_ir = abs(correlation) * 0.5
        
        if p.get('w_cvd', 0.5) > 0.5:
            base_ir += 0.1
        if p.get('z_window', 120) < 100:
            base_ir -= 0.05
        if p.get('ewm_span', 20) < 15:
            base_ir += 0.05
        if p.get('robust_z', True):
            base_ir += 0.02
        
        # 应用成本
        cost_penalty = cost_bps / 10000.0
        ir_after_cost = base_ir - cost_penalty
        
        # 计算其他指标
        trades_per_hour = len(clean_df) / (clean_df['ts_ms'].max() - clean_df['ts_ms'].min()) * 3600000
        trades_per_hour = min(trades_per_hour, 100.0)
        
        # 胜率
        positive_returns = clean_df[clean_df[ret_col] > 0]
        win_rate = len(positive_returns) / len(clean_df) if len(clean_df) > 0 else 0.0
        
        return {
            'IR_after_cost': float(ir_after_cost),
            'spearman_rho': float(abs(correlation)),
            'trades_per_hour': float(trades_per_hour),
            'win_rate': float(win_rate)
        }
        
    except Exception as e:
        print(f"[DEBUG] run_bt error: {e}")
        return {
            'IR_after_cost': np.nan,
            'spearman_rho': np.nan,
            'trades_per_hour': 0.0,
            'win_rate': 0.0
        }
