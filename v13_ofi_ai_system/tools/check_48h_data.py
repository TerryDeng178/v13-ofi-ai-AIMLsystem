#!/usr/bin/env python3
"""
检查48小时数据收集情况
"""
import pandas as pd
import glob
import os

def check_48h_data():
    """检查48小时数据样本统计"""
    data_dir = 'C:/Users/user/Desktop/ofi_cvd_framework/ofi_cvd_framework/v13_ofi_ai_system/artifacts/runtime/48h_collection/48h_collection_20251022_0655/date=2025-10-22'
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    print('=== 48小时数据样本统计 ===')
    
    for symbol in symbols:
        symbol_dir = os.path.join(data_dir, f'symbol={symbol}')
        fusion_dir = os.path.join(symbol_dir, 'kind=fusion')
        
        if os.path.exists(fusion_dir):
            files = glob.glob(os.path.join(fusion_dir, '*.parquet'))
            if files:
                try:
                    # 加载第一个文件查看样本数
                    df = pd.read_parquet(files[0])
                    print(f'{symbol}: {len(df)} 样本, 列: {list(df.columns)}')
                    
                    # 检查是否有更多文件
                    if len(files) > 1:
                        total_samples = 0
                        for f in files:
                            df_temp = pd.read_parquet(f)
                            total_samples += len(df_temp)
                        print(f'  -> 总文件数: {len(files)}, 总样本数: {total_samples}')
                    
                except Exception as e:
                    print(f'{symbol}: 加载失败 - {e}')
            else:
                print(f'{symbol}: 无fusion数据文件')
        else:
            print(f'{symbol}: 无fusion目录')
        
        # 也检查价格数据
        prices_dir = os.path.join(symbol_dir, 'kind=prices')
        if os.path.exists(prices_dir):
            price_files = glob.glob(os.path.join(prices_dir, '*.parquet'))
            if price_files:
                try:
                    df_prices = pd.read_parquet(price_files[0])
                    print(f'  -> 价格数据: {len(df_prices)} 样本')
                except Exception as e:
                    print(f'  -> 价格数据加载失败: {e}')

if __name__ == '__main__':
    check_48h_data()
