#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从历史价格数据计算 CVD

功能：
- 读取 parquet 历史数据（kind=prices）
- 使用 RealCVDCalculator 计算 CVD
- 保存结果为 parquet 供分析使用
"""

import sys
import io
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# 设置UTF-8输出
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# 添加项目路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(THIS_DIR, "..", "src")),
    os.path.abspath(os.path.join(THIS_DIR, "..", "..", "src")),
    THIS_DIR,
]
for p in CANDIDATE_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

from real_cvd_calculator import RealCVDCalculator, CVDConfig


def read_all_price_files(data_dir, symbol):
    """读取指定交易对的所有价格文件"""
    symbol_dir = data_dir / f"date=2025-10-27" / f"symbol={symbol}" / "kind=prices"
    
    if not symbol_dir.exists():
        print(f"目录不存在: {symbol_dir}")
        return None
    
    # 获取所有 parquet 文件
    files = sorted(symbol_dir.glob("*.parquet"))
    
    if not files:
        print(f"没有找到文件: {symbol_dir}")
        return None
    
    print(f"找到 {len(files)} 个文件")
    
    # 读取所有文件
    dfs = []
    for file in tqdm(files, desc=f"读取 {symbol}"):
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"读取失败: {file} - {e}")
    
    if not dfs:
        return None
    
    # 合并数据
    df_all = pd.concat(dfs, ignore_index=True)
    
    # 按时间戳排序
    df_all = df_all.sort_values(['ts_ms', 'agg_trade_id'], kind='mergesort').reset_index(drop=True)
    
    print(f"{symbol}: 总计 {len(df_all):,} 笔交易")
    return df_all


def calculate_cvd(df, symbol):
    """计算 CVD"""
    print(f"\n开始计算 {symbol} 的 CVD...")
    
    # 初始化 CVD 计算器
    config = CVDConfig()
    calc = RealCVDCalculator(symbol, config)
    
    # 准备结果列表
    results = []
    
    # 遍历每笔交易
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"计算 {symbol}"):
        # 提取数据
        price = float(row['price']) if pd.notna(row['price']) else None
        qty = float(row['qty']) if pd.notna(row['qty']) else 0.0
        is_buyer_maker = row.get('is_buyer_maker', None)
        ts_ms = int(row['ts_ms']) if pd.notna(row['ts_ms']) else None
        
        # 转换为 is_buy (is_buyer_maker=True表示买方是maker，即卖方是taker，主动卖出)
        is_buy = not is_buyer_maker if is_buyer_maker is not None else None
        
        # 计算 CVD
        result = calc.update_with_trade(
            price=price,
            qty=qty,
            is_buy=is_buy,
            event_time_ms=ts_ms
        )
        
        # 收集结果
        output_row = {
            'agg_trade_id': row.get('agg_trade_id', idx),
            'ts_ms': ts_ms,
            'event_time_ms': ts_ms,
            'timestamp': ts_ms / 1000.0 if ts_ms else None,
            'symbol': symbol,
            'price': price,
            'qty': qty,
            'is_buy': is_buy,
            'is_buyer_maker': is_buyer_maker,
            'cvd': result['cvd'],
            'z_cvd': result.get('z_cvd'),
            'ema_cvd': result.get('ema_cvd'),
            'warmup': result.get('meta', {}).get('warmup', True),
            'std_zero': result.get('meta', {}).get('std_zero', False),
        }
        
        results.append(output_row)
    
    # 转换为 DataFrame
    df_result = pd.DataFrame(results)
    
    print(f"{symbol} 计算完成: {len(df_result):,} 笔")
    
    # 统计信息
    warmup_pct = df_result['warmup'].mean() * 100
    std_zero_count = df_result['std_zero'].sum()
    z_valid = df_result['z_cvd'].notna().sum()
    
    print(f"  - Warmup: {warmup_pct:.2f}%")
    print(f"  - Std Zero: {std_zero_count} 笔")
    print(f"  - Z-score 有效: {z_valid:,} 笔")
    
    return df_result


def main():
    """主函数"""
    # 数据目录
    data_dir = Path("data/ofi_cvd")
    
    # 输出目录
    output_dir = Path("cvd_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 交易对列表
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    
    print("=" * 80)
    print("CVD 历史数据计算")
    print("=" * 80)
    
    # 处理每个交易对
    for symbol in symbols:
        print(f"\n处理交易对: {symbol}")
        print("-" * 80)
        
        # 读取数据
        df = read_all_price_files(data_dir, symbol)
        
        if df is None or len(df) == 0:
            print(f"跳过 {symbol}（无数据）")
            continue
        
        # 计算 CVD
        df_result = calculate_cvd(df, symbol)
        
        # 保存结果
        output_file = output_dir / f"{symbol}_cvd.parquet"
        df_result.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"[OK] 结果已保存: {output_file}")
    
    print("\n" + "=" * 80)
    print("全部完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

