import numpy as np
import pandas as pd

def gen_signals_v7_quality_filtered(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v7 信号质量筛选版 - 只选择高质量信号
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    
    # 1. 基础OFI信号
    ofi_threshold = m.get("ofi_z_min", 1.8)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 2. 信号质量评分
    quality_score = calculate_signal_quality(out, params)
    min_quality_score = m.get("min_quality_score", 0.7)
    high_quality = quality_score >= min_quality_score
    
    # 3. 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = m.get("min_signal_strength", 2.5)
    strong_signal = signal_strength >= min_signal_strength
    
    # 4. 价格动量确认
    price_momentum_threshold = m.get("price_momentum_threshold", 0.00002)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 5. 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 6. 组合高质量信号
    long_mask = ofi_signal & strong_signal & high_quality & direction_consistent_long
    short_mask = ofi_signal & strong_signal & high_quality & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = quality_score[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = quality_score[short_mask]
    
    return out

def calculate_signal_quality(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    计算信号质量评分
    """
    quality_score = pd.Series(0.0, index=df.index)
    
    # 1. OFI强度评分 (0-0.4)
    ofi_strength = abs(df["ofi_z"])
    ofi_score = np.clip(ofi_strength / 3.0, 0, 1) * 0.4  # 归一化到0-0.4
    
    # 2. 价格动量评分 (0-0.3)
    price_momentum = abs(df["ret_1s"])
    momentum_score = np.clip(price_momentum * 10000, 0, 1) * 0.3  # 归一化到0-0.3
    
    # 3. 流动性评分 (0-0.2)
    spread_bps = (df["ask1"] - df["bid1"]) / df["price"] * 1e4
    depth_score = (df["bid1_size"] + df["ask1_size"]) / (df["bid1_size"] + df["ask1_size"]).rolling(100).quantile(0.8)
    liquidity_score = np.clip(1 - spread_bps / 10.0, 0, 1) * 0.1 + np.clip(depth_score, 0, 1) * 0.1
    
    # 4. 时间评分 (0-0.1)
    # 简单的时间评分：避免开盘和收盘时段
    hour = df.index.hour if hasattr(df.index, 'hour') else 0
    time_score = np.where((hour >= 9) & (hour <= 15), 0.1, 0.05)  # 交易时段评分更高
    
    quality_score = ofi_score + momentum_score + liquidity_score + time_score
    
    return quality_score

def gen_signals_v7_dynamic_threshold(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v7 动态阈值版 - 根据市场条件动态调整阈值
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["dynamic_threshold"] = 0.0
    
    # 1. 市场状态识别
    market_volatility = df["ret_1s"].rolling(100).std()
    ofi_volatility = df["ofi_z"].rolling(100).std()
    
    # 2. 动态阈值计算
    base_threshold = m.get("ofi_z_min", 1.8)
    
    # 高波动率时提高阈值
    high_vol = (market_volatility > market_volatility.quantile(0.7)) | \
               (ofi_volatility > ofi_volatility.quantile(0.7))
    
    dynamic_threshold = base_threshold * np.where(high_vol, 1.3, 1.0)
    
    # 3. 信号生成
    ofi_signal = abs(out["ofi_z"]) >= dynamic_threshold
    strong_long = (out["ofi_z"] >= dynamic_threshold) & ofi_signal
    strong_short = (out["ofi_z"] <= -dynamic_threshold) & ofi_signal
    
    # 4. 信号强度
    signal_strength = abs(out["ofi_z"])
    
    # 应用信号
    out.loc[strong_long, "sig_type"] = "momentum"
    out.loc[strong_long, "sig_side"] = 1
    out.loc[strong_long, "signal_strength"] = signal_strength[strong_long]
    out.loc[strong_long, "dynamic_threshold"] = dynamic_threshold[strong_long]
    
    out.loc[strong_short, "sig_type"] = "momentum"
    out.loc[strong_short, "sig_side"] = -1
    out.loc[strong_short, "signal_strength"] = signal_strength[strong_short]
    out.loc[strong_short, "dynamic_threshold"] = dynamic_threshold[strong_short]
    
    return out

def gen_signals_v7_momentum_enhanced(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v7 动量增强版 - 结合价格动量和OFI信号
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["momentum_score"] = 0.0
    
    # 1. 价格动量计算
    price_momentum = df["ret_1s"]
    momentum_ma = price_momentum.rolling(5).mean()  # 5秒动量均值
    momentum_std = price_momentum.rolling(20).std()  # 20秒动量标准差
    
    # 2. 动量评分
    momentum_score = abs(momentum_ma) / (momentum_std + 1e-8)
    momentum_score = np.clip(momentum_score, 0, 3)  # 限制在0-3范围内
    
    # 3. OFI信号
    ofi_threshold = m.get("ofi_z_min", 1.8)
    ofi_signal = abs(df["ofi_z"]) >= ofi_threshold
    
    # 4. 组合信号
    long_momentum = (price_momentum > 0) & (momentum_score >= 1.0)
    short_momentum = (price_momentum < 0) & (momentum_score >= 1.0)
    
    long_mask = (df["ofi_z"] >= ofi_threshold) & ofi_signal & long_momentum
    short_mask = (df["ofi_z"] <= -ofi_threshold) & ofi_signal & short_momentum
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = abs(df["ofi_z"][long_mask])
    out.loc[long_mask, "momentum_score"] = momentum_score[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = abs(df["ofi_z"][short_mask])
    out.loc[short_mask, "momentum_score"] = momentum_score[short_mask]
    
    return out
