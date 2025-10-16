import numpy as np
import pandas as pd

def gen_signals_v8_intelligent_filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v8 智能信号筛选版 - 基于历史表现动态调整筛选标准
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    out["intelligence_score"] = 0.0
    
    # 1. 基础OFI信号
    ofi_threshold = m.get("ofi_z_min", 1.6)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 2. 智能信号质量评分
    intelligence_score = calculate_intelligent_quality(out, params)
    min_intelligence_score = m.get("min_intelligence_score", 0.6)
    high_intelligence = intelligence_score >= min_intelligence_score
    
    # 3. 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = m.get("min_signal_strength", 2.0)
    strong_signal = signal_strength >= min_signal_strength
    
    # 4. 价格动量确认
    price_momentum_threshold = m.get("price_momentum_threshold", 0.000015)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 5. 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 6. 市场状态适应性
    market_condition = detect_market_condition(out)
    market_adaptive = apply_market_adaptation(out, market_condition, params)
    
    # 7. 组合智能信号
    long_mask = ofi_signal & strong_signal & high_intelligence & direction_consistent_long & market_adaptive
    short_mask = ofi_signal & strong_signal & high_intelligence & direction_consistent_short & market_adaptive
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = intelligence_score[long_mask]
    out.loc[long_mask, "intelligence_score"] = intelligence_score[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = intelligence_score[short_mask]
    out.loc[short_mask, "intelligence_score"] = intelligence_score[short_mask]
    
    return out

def calculate_intelligent_quality(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    计算智能信号质量评分 - 基于多维度分析和历史表现
    """
    quality_score = pd.Series(0.0, index=df.index)
    
    # 1. OFI强度评分 (0-0.3)
    ofi_strength = abs(df["ofi_z"])
    ofi_score = np.clip(ofi_strength / 2.5, 0, 1) * 0.3  # 调整归一化参数
    
    # 2. 价格动量评分 (0-0.25)
    price_momentum = abs(df["ret_1s"])
    momentum_score = np.clip(price_momentum * 15000, 0, 1) * 0.25  # 调整归一化参数
    
    # 3. 流动性评分 (0-0.2)
    spread_bps = (df["ask1"] - df["bid1"]) / df["price"] * 1e4
    depth_score = (df["bid1_size"] + df["ask1_size"]) / (df["bid1_size"] + df["ask1_size"]).rolling(100).quantile(0.8)
    liquidity_score = np.clip(1 - spread_bps / 8.0, 0, 1) * 0.1 + np.clip(depth_score, 0, 1) * 0.1
    
    # 4. 时间评分 (0-0.1)
    hour = df.index.hour if hasattr(df.index, 'hour') else 0
    time_score = np.where((hour >= 9) & (hour <= 15), 0.1, 0.05)
    
    # 5. 市场状态评分 (0-0.15)
    market_volatility = df["ret_1s"].rolling(50).std()
    vol_percentile = market_volatility.rolling(200).rank(pct=True)
    market_score = np.clip(1 - abs(vol_percentile - 0.5) * 2, 0, 1) * 0.15  # 偏好中等波动率
    
    quality_score = ofi_score + momentum_score + liquidity_score + time_score + market_score
    
    return quality_score

def detect_market_condition(df: pd.DataFrame) -> pd.Series:
    """
    检测市场状态：趋势、震荡、高波动、低波动
    """
    # 价格趋势检测
    price_trend = df["price"].rolling(50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    trend_strength = abs(price_trend)
    
    # 波动率检测
    volatility = df["ret_1s"].rolling(50).std()
    vol_percentile = volatility.rolling(200).rank(pct=True)
    
    # 市场状态分类
    condition = pd.Series("normal", index=df.index)
    condition[trend_strength > trend_strength.quantile(0.7)] = "trending"
    condition[trend_strength < trend_strength.quantile(0.3)] = "ranging"
    condition[vol_percentile > 0.8] = "high_volatility"
    condition[vol_percentile < 0.2] = "low_volatility"
    
    return condition

def apply_market_adaptation(df: pd.DataFrame, market_condition: pd.Series, params: dict) -> pd.Series:
    """
    根据市场状态应用适应性调整
    """
    adaptation = pd.Series(True, index=df.index)
    
    # 趋势市场：提高OFI要求
    trending_mask = market_condition == "trending"
    adaptation[trending_mask] = abs(df["ofi_z"][trending_mask]) >= params["signals"]["momentum"]["ofi_z_min"] * 1.2
    
    # 震荡市场：降低OFI要求
    ranging_mask = market_condition == "ranging"
    adaptation[ranging_mask] = abs(df["ofi_z"][ranging_mask]) >= params["signals"]["momentum"]["ofi_z_min"] * 0.8
    
    # 高波动市场：提高信号强度要求
    high_vol_mask = market_condition == "high_volatility"
    adaptation[high_vol_mask] = (abs(df["ofi_z"][high_vol_mask]) >= params["signals"]["momentum"]["ofi_z_min"]) & \
                                (abs(df["signal_strength"][high_vol_mask]) >= params["signals"]["momentum"]["min_signal_strength"] * 1.3)
    
    # 低波动市场：降低信号强度要求
    low_vol_mask = market_condition == "low_volatility"
    adaptation[low_vol_mask] = (abs(df["ofi_z"][low_vol_mask]) >= params["signals"]["momentum"]["ofi_z_min"]) & \
                               (abs(df["signal_strength"][low_vol_mask]) >= params["signals"]["momentum"]["min_signal_strength"] * 0.7)
    
    return adaptation

def gen_signals_v8_frequency_optimized(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v8 频率优化版 - 在保持质量前提下增加交易频率
    """
    p = params["signals"]
    m = p["momentum"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["frequency_score"] = 0.0
    
    # 1. 基础OFI信号（降低阈值）
    ofi_threshold = m.get("ofi_z_min", 1.6)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 2. 频率优化评分
    frequency_score = calculate_frequency_score(out, params)
    min_frequency_score = m.get("min_frequency_score", 0.5)
    high_frequency = frequency_score >= min_frequency_score
    
    # 3. 信号强度筛选（适度降低）
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = m.get("min_signal_strength", 2.0)
    strong_signal = signal_strength >= min_signal_strength
    
    # 4. 价格动量确认（适度降低）
    price_momentum_threshold = m.get("price_momentum_threshold", 0.000015)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 5. 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 6. 组合频率优化信号
    long_mask = ofi_signal & strong_signal & high_frequency & direction_consistent_long
    short_mask = ofi_signal & strong_signal & high_frequency & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "momentum"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "frequency_score"] = frequency_score[long_mask]
    
    out.loc[short_mask, "sig_type"] = "momentum"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "frequency_score"] = frequency_score[short_mask]
    
    return out

def calculate_frequency_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    计算频率优化评分 - 平衡质量和频率
    """
    frequency_score = pd.Series(0.0, index=df.index)
    
    # 1. OFI强度评分 (0-0.4)
    ofi_strength = abs(df["ofi_z"])
    ofi_score = np.clip(ofi_strength / 2.0, 0, 1) * 0.4  # 降低阈值要求
    
    # 2. 价格动量评分 (0-0.3)
    price_momentum = abs(df["ret_1s"])
    momentum_score = np.clip(price_momentum * 12000, 0, 1) * 0.3  # 降低阈值要求
    
    # 3. 流动性评分 (0-0.2)
    spread_bps = (df["ask1"] - df["bid1"]) / df["price"] * 1e4
    depth_score = (df["bid1_size"] + df["ask1_size"]) / (df["bid1_size"] + df["ask1_size"]).rolling(100).quantile(0.8)
    liquidity_score = np.clip(1 - spread_bps / 10.0, 0, 1) * 0.1 + np.clip(depth_score, 0, 1) * 0.1
    
    # 4. 时间评分 (0-0.1)
    hour = df.index.hour if hasattr(df.index, 'hour') else 0
    time_score = np.where((hour >= 9) & (hour <= 15), 0.1, 0.05)
    
    frequency_score = ofi_score + momentum_score + liquidity_score + time_score
    
    return frequency_score
