#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 is_buyer_maker 字段是否正确写入 prices Parquet 文件
"""
import glob
import pandas as pd
from pathlib import Path

def verify_is_buyer_maker():
    """验证 is_buyer_maker 字段"""
    
    # 查找最新的 prices Parquet 文件
    base_dir = Path(__file__).parent / "data" / "ofi_cvd"
    
    # 查找所有日期目录
    date_dirs = sorted([d for d in base_dir.glob("date=*") if d.is_dir()])
    
    if not date_dirs:
        print("未找到数据文件")
        return
    
    # 使用最新的日期目录
    latest_date = date_dirs[-1]
    print(f"检查日期: {latest_date.name}")
    
    # 查找所有 symbol 目录
    symbol_dirs = sorted([d for d in latest_date.glob("symbol=*") if d.is_dir()])
    
    if not symbol_dirs:
        print("未找到 symbol 目录")
        return
    
    # 检查第一个 symbol
    first_symbol = symbol_dirs[0]
    prices_dir = first_symbol / "kind=prices"
    
    if not prices_dir.exists():
        print(f"未找到 kind=prices 目录: {prices_dir}")
        return
    
    # 查找 Parquet 文件
    parquet_files = sorted(prices_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"未找到 Parquet 文件: {prices_dir}")
        return
    
    # 读取最新的文件
    latest_file = parquet_files[-1]
    print(f"\n读取文件: {latest_file.name}")
    
    # 读取数据
    df = pd.read_parquet(latest_file)
    
    # 检查列
    print(f"\n列数: {len(df.columns)}")
    print(f"\n前10列: {df.columns[:10].tolist()}")
    
    # 检查 is_buyer_maker 是否存在
    has_is_buyer_maker = 'is_buyer_maker' in df.columns
    print(f"\nis_buyer_maker 是否存在: {has_is_buyer_maker}")
    
    if has_is_buyer_maker:
        print("\nis_buyer_maker 数据类型:", df['is_buyer_maker'].dtype)
        print("is_buyer_maker 值统计:")
        print(df['is_buyer_maker'].value_counts())
        
        # 显示前5行数据
        print("\n前5行数据:")
        print(df[['ts_ms','symbol','price','qty','is_buyer_maker']].head())
        
        # 验证数据有效性
        print("\n数据有效性检查:")
        print(f"- 总行数: {len(df)}")
        print(f"- is_buyer_maker 非空行数: {df['is_buyer_maker'].notna().sum()}")
        print(f"- is_buyer_maker True 数量: {(df['is_buyer_maker'] == True).sum()}")
        print(f"- is_buyer_maker False 数量: {(df['is_buyer_maker'] == False).sum()}")
    else:
        print("\n[警告] is_buyer_maker 字段不存在于文件中")
        print("可能需要重新运行数据采集进程")
    
    # 打印所有列（如果字段很多，只打印部分）
    if len(df.columns) > 20:
        print(f"\n所有列（共{len(df.columns)}列）:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
    else:
        print(f"\n所有列: {df.columns.tolist()}")
    
    print("\n[完成] 验证脚本执行完毕")

if __name__ == "__main__":
    verify_is_buyer_maker()

