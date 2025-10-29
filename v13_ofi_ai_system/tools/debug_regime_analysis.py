#!/usr/bin/env python3
"""
调试2x2场景分析，查看为什么只有TL场景被优化
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2X2'))
from regime2x2_pipeline import label_regime_2x2, compute_features, fit_2x2_thresholds

def debug_regime_analysis():
    """调试场景分析"""
    print("=== 调试2x2场景分析 ===")
    
    # 加载BTCUSDT的数据作为示例
    data_dir = 'C:/Users/user/Desktop/ofi_cvd_framework/ofi_cvd_framework/v13_ofi_ai_system/artifacts/runtime/48h_collection/48h_collection_20251022_0655/date=2025-10-22/symbol=BTCUSDT'
    
    # 加载价格数据
    prices_dir = os.path.join(data_dir, 'kind=prices')
    price_files = []
    for f in os.listdir(prices_dir):
        if f.endswith('.parquet'):
            price_files.append(os.path.join(prices_dir, f))
    
    prices_list = []
    for f in price_files[:5]:  # 只加载前5个文件
        df = pd.read_parquet(f)
        prices_list.append(df)
    prices_df = pd.concat(prices_list, ignore_index=True)
    prices_df = prices_df.sort_values('ts_ms')
    
    # 加载信号数据
    fusion_dir = os.path.join(data_dir, 'kind=fusion')
    signal_files = []
    for f in os.listdir(fusion_dir):
        if f.endswith('.parquet'):
            signal_files.append(os.path.join(fusion_dir, f))
    
    signals_list = []
    for f in signal_files[:5]:  # 只加载前5个文件
        df = pd.read_parquet(f)
        signals_list.append(df)
    signals_df = pd.concat(signals_list, ignore_index=True)
    signals_df = signals_df.sort_values('ts_ms')
    
    print(f"价格数据形状: {prices_df.shape}")
    print(f"信号数据形状: {signals_df.shape}")
    
    # 计算特征
    print("\n=== 计算特征 ===")
    features = compute_features(prices_df, signals_df)
    print(f"特征数据形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")
    
    # 拟合阈值
    print("\n=== 拟合阈值 ===")
    thresholds = fit_2x2_thresholds(features)
    print(f"阈值: {thresholds}")
    
    # 标签场景
    print("\n=== 标签场景 ===")
    labeled = label_regime_2x2(features, thresholds)
    print(f"标签数据形状: {labeled.shape}")
    print(f"场景分布: {labeled['regime_2x2'].value_counts().to_dict()}")
    
    # 检查每个场景的样本数
    print("\n=== 场景样本数检查 ===")
    for regime in ['TL', 'TH', 'WL', 'WH']:
        count = len(labeled[labeled['regime_2x2'] == regime])
        print(f"{regime}: {count} 样本")
        if count > 0:
            print(f"  -> 样本数 > 10: {count > 10}")
            print(f"  -> 样本数 > 500: {count > 500}")
    
    # 检查数据合并
    print("\n=== 数据合并检查 ===")
    # 构建未来收益
    horizons = [15, 60, 300]
    fwd = build_forward_returns_from_prices(prices_df, horizons=horizons, price_col="price")
    
    merged = (signals_df
              .merge(labeled[["ts_ms","symbol","regime_2x2"]], on=["ts_ms","symbol"], how="left")
              .merge(fwd, on=["ts_ms","symbol"], how="left"))
    
    merged['regime_2x2'] = merged['regime_2x2'].fillna("TL")
    merged = merged.sort_values(['symbol','ts_ms'])
    
    print(f"合并后数据形状: {merged.shape}")
    print(f"合并后场景分布: {merged['regime_2x2'].value_counts().to_dict()}")
    
    # 检查每个场景在合并后数据中的样本数
    print("\n=== 合并后场景样本数检查 ===")
    for regime in ['TL', 'TH', 'WL', 'WH']:
        count = len(merged[merged['regime_2x2'] == regime])
        print(f"{regime}: {count} 样本")
        if count > 0:
            print(f"  -> 样本数 > 10: {count > 10}")
            print(f"  -> 样本数 > 500: {count > 500}")
    
    return merged, thresholds

def build_forward_returns_from_prices(prices_df: pd.DataFrame, horizons=(60, 300), price_col="price"):
    """基于价格构造未来收益"""
    if price_col not in prices_df.columns:
        if {"bid","ask"}.issubset(prices_df.columns):
            prices_df = prices_df.copy()
            prices_df[price_col] = (prices_df["bid"] + prices_df["ask"]) / 2.0
        elif "price" in prices_df.columns:
            price_col = "price"
        else:
            raise ValueError("No price column: need mid or bid+ask or price.")
    
    base = prices_df[["ts_ms","symbol",price_col]].dropna().sort_values(["symbol","ts_ms"]).copy()
    out = base[["ts_ms","symbol"]].copy()
    
    for h in horizons:
        fut = base.copy()
        fut["ts_ms"] = fut["ts_ms"] + h*1000
        # forward 对齐未来时刻的价格
        aligned = pd.merge_asof(
            base.sort_values("ts_ms"),
            fut.sort_values("ts_ms"),
            on="ts_ms", by="symbol", direction="forward", suffixes=("","_fut")
        )
        out[f"ret_{h}s"] = (aligned[f"{price_col}_fut"] / aligned[price_col] - 1.0)
    
    return out

if __name__ == '__main__':
    debug_regime_analysis()
