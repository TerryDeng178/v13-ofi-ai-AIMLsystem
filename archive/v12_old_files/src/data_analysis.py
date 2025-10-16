import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    深度分析数据质量，检查OFI/CVD与未来收益的相关性
    """
    print("=== 数据质量诊断分析 ===")
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息:")
    print(f"数据长度: {len(df)} 行")
    print(f"时间范围: {df['ts'].min()} 到 {df['ts'].max()}")
    
    # 2. OFI/CVD分布特征
    print("\n2. OFI/CVD分布特征:")
    print("OFI_z分布:")
    print(df["ofi_z"].describe())
    print("\nCVD_z分布:")
    print(df["cvd_z"].describe())
    
    # 3. 价格收益分布
    print("\n3. 价格收益分布:")
    print("ret_1s分布:")
    print(df["ret_1s"].describe())
    
    # 4. 信号与未来收益的相关性分析
    print("\n4. 信号与未来收益相关性分析:")
    
    # 计算不同时间窗口的未来收益
    for window in [1, 3, 5, 10, 30, 60]:
        future_ret = df["ret_1s"].shift(-window)
        ofi_corr = df["ofi_z"].corr(future_ret)
        cvd_corr = df["cvd_z"].corr(future_ret)
        print(f"  {window}秒后收益: OFI相关性={ofi_corr:.4f}, CVD相关性={cvd_corr:.4f}")
    
    # 5. 动量一致性分析
    print("\n5. 动量一致性分析:")
    momentum_consistency = (df["ofi_z"] * df["ret_1s"]).mean()
    print(f"OFI与当前收益一致性: {momentum_consistency:.4f}")
    
    # 6. 信号强度分布
    print("\n6. 信号强度分布:")
    signal_strength = (abs(df["ofi_z"]) + abs(df["cvd_z"])) / 2
    print("信号强度分布:")
    print(signal_strength.describe())
    
    # 7. 分位数分析
    print("\n7. 分位数分析:")
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        ofi_q = df["ofi_z"].quantile(q)
        cvd_q = df["cvd_z"].quantile(q)
        print(f"  {int(q*100)}%分位数: OFI={ofi_q:.3f}, CVD={cvd_q:.3f}")
    
    # 8. 条件收益分析
    print("\n8. 条件收益分析:")
    
    # 高OFI情况下的未来收益
    high_ofi = df["ofi_z"] > df["ofi_z"].quantile(0.8)
    low_ofi = df["ofi_z"] < df["ofi_z"].quantile(0.2)
    
    for window in [1, 3, 5, 10]:
        future_ret = df["ret_1s"].shift(-window)
        high_ofi_ret = future_ret[high_ofi].mean()
        low_ofi_ret = future_ret[low_ofi].mean()
        print(f"  {window}秒后: 高OFI平均收益={high_ofi_ret:.6f}, 低OFI平均收益={low_ofi_ret:.6f}")
    
    # 9. 信号有效性测试
    print("\n9. 信号有效性测试:")
    test_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for threshold in test_thresholds:
        # 测试OFI信号
        ofi_signal = abs(df["ofi_z"]) >= threshold
        if ofi_signal.sum() > 0:
            future_ret = df["ret_1s"].shift(-1)
            signal_ret = future_ret[ofi_signal].mean()
            signal_count = ofi_signal.sum()
            print(f"  OFI阈值{threshold}: 信号数={signal_count}, 平均1秒后收益={signal_ret:.6f}")
    
    # 10. 成本效益分析
    print("\n10. 成本效益分析:")
    fee_bps = 2.0  # 2bps手续费
    for threshold in test_thresholds:
        ofi_signal = abs(df["ofi_z"]) >= threshold
        if ofi_signal.sum() > 0:
            future_ret = df["ret_1s"].shift(-1)
            signal_ret = future_ret[ofi_signal].mean()
            breakeven_needed = fee_bps / 10000  # 转换为收益率
            profit_margin = signal_ret - breakeven_needed
            print(f"  OFI阈值{threshold}: 需要收益={breakeven_needed:.6f}, 实际收益={signal_ret:.6f}, 利润边际={profit_margin:.6f}")
    
    return {
        "data_length": len(df),
        "ofi_corr_1s": df["ofi_z"].corr(df["ret_1s"].shift(-1)),
        "cvd_corr_1s": df["cvd_z"].corr(df["ret_1s"].shift(-1)),
        "momentum_consistency": momentum_consistency,
        "signal_strength_mean": signal_strength.mean(),
        "high_ofi_ret_1s": future_ret[high_ofi].mean() if high_ofi.sum() > 0 else 0
    }

def create_signal_effectiveness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建信号有效性报告
    """
    results = []
    
    for ofi_threshold in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for cvd_threshold in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            # 测试不同组合的信号
            signal = (abs(df["ofi_z"]) >= ofi_threshold) & (abs(df["cvd_z"]) >= cvd_threshold)
            
            if signal.sum() > 10:  # 至少10个信号
                # 计算未来1-10秒的收益
                future_rets = {}
                for window in [1, 3, 5, 10]:
                    future_ret = df["ret_1s"].shift(-window)
                    future_rets[f"ret_{window}s"] = future_ret[signal].mean()
                
                results.append({
                    "ofi_threshold": ofi_threshold,
                    "cvd_threshold": cvd_threshold,
                    "signal_count": signal.sum(),
                    "signal_rate": signal.sum() / len(df),
                    **future_rets
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # 加载数据进行测试
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.data import load_csv
    from src.features import add_feature_block
    
    # 加载数据
    df = load_csv("examples/sample_data.csv")
    
    # 创建默认参数
    params = {
        "features": {
            "ofi_window_seconds": 2,
            "ofi_levels": 5,
            "vwap_window_seconds": 1800,
            "atr_window": 14,
            "z_window": 1200
        }
    }
    
    df = add_feature_block(df, params)
    
    # 运行分析
    analysis_result = analyze_data_quality(df)
    
    # 创建信号有效性报告
    effectiveness_df = create_signal_effectiveness_report(df)
    print("\n=== 信号有效性报告 ===")
    print(effectiveness_df.head(10))
    
    # 保存报告
    effectiveness_df.to_csv("examples/out/signal_effectiveness_report.csv", index=False)
    print(f"\n信号有效性报告已保存到: examples/out/signal_effectiveness_report.csv")
