"""
生成测试用的 OFI/CVD 数据文件（模拟真实数据格式）

Author: V13 QA Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def generate_test_data():
    """生成 6 个交易对的测试数据"""
    
    project_root = Path(__file__).parent.parent
    preview_dir = project_root / "deploy" / "preview" / "ofi_cvd"
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    date_str = "2025-10-27"
    
    # 生成时间戳
    start_ts = int(time.mktime(time.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    n_samples = 10000
    
    rng = np.random.default_rng(42)
    
    for symbol in symbols:
        print(f"生成 {symbol} 数据...")
        
        # 生成时间戳序列（每秒 100 个样本）
        ts_list = [start_ts + i * 10 for i in range(n_samples)]
        
        # 生成 OFI 数据
        ofi_df = pd.DataFrame({
            'ts_ms': ts_list,
            'symbol': symbol,
            'ofi_z': rng.normal(0, 1.5, n_samples),  # OFI Z-score
            'ofi_value': rng.normal(0, 1000000, n_samples),
            'price': rng.uniform(50000, 52000, n_samples),  # 价格
            'regime': rng.choice(['calm', 'active'], n_samples),
            'vol_bucket': rng.choice(['low', 'high'], n_samples)
        })
        
        # 生成 CVD 数据
        cvd_df = pd.DataFrame({
            'ts_ms': ts_list,
            'symbol': symbol,
            'z_cvd': rng.normal(0, 1.5, n_samples),  # CVD Z-score
            'cvd': rng.normal(0, 1000000, n_samples),
            'price': rng.uniform(50000, 52000, n_samples),
            'regime': rng.choice(['calm', 'active'], n_samples),
            'vol_bucket': rng.choice(['low', 'high'], n_samples)
        })
        
        # 生成 fusion 数据（包含 future return）
        fusion_df = pd.DataFrame({
            'ts_ms': ts_list,
            'symbol': symbol,
            'score': rng.normal(0, 1.5, n_samples),
            'signal': rng.choice(['buy', 'sell', 'strong_buy', 'strong_sell', 'neutral'], n_samples, p=[0.1, 0.1, 0.05, 0.05, 0.7]),
            'consistency': rng.uniform(0, 1, n_samples),
            'price': rng.uniform(50000, 52000, n_samples),
            'return_5s': rng.normal(0, 0.001, n_samples),  # 5秒后收益
            'return_30s': rng.normal(0, 0.002, n_samples),  # 30秒后收益
            'regime': rng.choice(['calm', 'active'], n_samples),
            'scenario_2x2': rng.choice(['calm-low', 'calm-high', 'active-low', 'active-high'], n_samples)
        })
        
        # 保存文件
        for kind, df in [('ofi', ofi_df), ('cvd', cvd_df), ('fusion', fusion_df)]:
            filepath = preview_dir / f"date={date_str}" / f"symbol={symbol}" / f"kind={kind}"
            filepath.mkdir(parents=True, exist_ok=True)
            
            filename = f"test_data_{symbol}_{kind}.parquet"
            df.to_parquet(filepath / filename, compression='snappy', index=False)
            
            print(f"  已保存 {kind} 数据: {len(df)} 行 -> {filepath / filename}")
    
    print(f"\n[OK] 所有测试数据已生成到: {preview_dir}")

if __name__ == "__main__":
    generate_test_data()



