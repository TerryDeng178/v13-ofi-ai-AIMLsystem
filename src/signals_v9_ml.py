import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime

class MLSignalPredictor:
    """
    v9 机器学习信号预测器
    """
    
    def __init__(self, model_type: str = "ensemble", model_path: str = "models/"):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.training_data = []
        self.performance_history = []
        
        # 确保模型目录存在
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备机器学习特征
        """
        features_df = df.copy()
        
        # 基础特征
        features_df["spread_bps"] = (features_df["ask1"] - features_df["bid1"]) / features_df["price"] * 1e4
        features_df["depth_ratio"] = (features_df["bid1_size"] + features_df["ask1_size"]) / \
                                   (features_df["bid1_size"] + features_df["ask1_size"]).rolling(100).quantile(0.8)
        features_df["price_volatility"] = features_df["ret_1s"].rolling(50).std()
        features_df["ofi_volatility"] = features_df["ofi_z"].rolling(50).std()
        
        # 技术指标特征
        features_df["rsi"] = self._calculate_rsi(features_df["ret_1s"], 14)
        features_df["macd"] = self._calculate_macd(features_df["price"], 12, 26, 9)
        features_df["bollinger_upper"] = self._calculate_bollinger_bands(features_df["price"], 20, 2)[0]
        features_df["bollinger_lower"] = self._calculate_bollinger_bands(features_df["price"], 20, 2)[1]
        
        # 市场状态特征
        features_df["trend_strength"] = self._calculate_trend_strength(features_df["price"], 50)
        features_df["volatility_regime"] = self._calculate_volatility_regime(features_df["ret_1s"], 100)
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """计算趋势强度"""
        trend = prices.rolling(window=window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        return trend.fillna(0)
    
    def _calculate_volatility_regime(self, returns: pd.Series, window: int = 100) -> pd.Series:
        """计算波动率状态"""
        vol = returns.rolling(window=window).std()
        vol_percentile = vol.rolling(window*2).rank(pct=True)
        return vol_percentile.fillna(0.5)
    
    def create_training_data(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        创建训练数据，标记信号质量
        """
        # 准备特征
        features_df = self.prepare_features(df)
        
        # 生成基础信号
        try:
            from .signals_v8 import gen_signals_v8_intelligent_filter
        except ImportError:
            from signals_v8 import gen_signals_v8_intelligent_filter
        signals_df = gen_signals_v8_intelligent_filter(features_df, params)
        
        # 创建目标变量（信号质量）
        signals_df["signal_quality"] = 0.0
        
        # 标记高质量信号
        quality_mask = (signals_df["sig_side"] != 0) & (signals_df["quality_score"] >= 0.7)
        signals_df.loc[quality_mask, "signal_quality"] = 1.0
        
        # 标记中等质量信号
        medium_mask = (signals_df["sig_side"] != 0) & (signals_df["quality_score"] >= 0.5) & (signals_df["quality_score"] < 0.7)
        signals_df.loc[medium_mask, "signal_quality"] = 0.5
        
        return signals_df
    
    def train_model(self, df: pd.DataFrame, params: dict):
        """
        训练机器学习模型
        """
        # 创建训练数据
        training_df = self.create_training_data(df, params)
        
        # 选择特征列
        feature_columns = [
            "ofi_z", "cvd_z", "ret_1s", "atr", "vwap", 
            "bid1_size", "ask1_size", "spread_bps", "depth_ratio",
            "price_volatility", "ofi_volatility", "rsi", "macd",
            "bollinger_upper", "bollinger_lower", "trend_strength", "volatility_regime"
        ]
        
        # 过滤有效数据
        valid_data = training_df[feature_columns + ["signal_quality"]].dropna()
        
        if len(valid_data) < 100:
            print("警告: 训练数据不足，跳过模型训练")
            return
        
        X = valid_data[feature_columns]
        y = valid_data["signal_quality"]
        
        # 根据模型类型选择算法
        if self.model_type == "ensemble":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            except ImportError:
                print("XGBoost未安装，使用随机森林")
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
            except ImportError:
                print("LightGBM未安装，使用随机森林")
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 训练模型
        self.model.fit(X, y)
        self.feature_columns = feature_columns
        self.is_trained = True
        
        # 保存模型
        model_file = os.path.join(self.model_path, f"signal_predictor_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        joblib.dump(self.model, model_file)
        print(f"模型已保存到: {model_file}")
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("特征重要性:")
            print(feature_importance.head(10))
    
    def predict_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        预测信号质量
        """
        if not self.is_trained or self.model is None:
            # 如果模型未训练，返回默认评分
            return pd.Series(0.5, index=df.index)
        
        # 准备特征
        features_df = self.prepare_features(df)
        
        # 选择特征列
        X = features_df[self.feature_columns].fillna(0)
        
        # 预测
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)
    
    def update_model_performance(self, actual_quality: float, predicted_quality: float):
        """
        更新模型性能
        """
        self.performance_history.append({
            'actual': actual_quality,
            'predicted': predicted_quality,
            'error': abs(actual_quality - predicted_quality),
            'timestamp': datetime.now()
        })
        
        # 保持最近1000条记录
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

def gen_signals_v9_ml_enhanced(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v9 机器学习增强信号生成
    """
    p = params["signals"]["ml_enhanced"]
    ml_params = params["ml_integration"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    out["ml_prediction"] = 0.0
    out["ml_confidence"] = 0.0
    
    # 初始化ML预测器
    if not hasattr(gen_signals_v9_ml_enhanced, 'ml_predictor'):
        gen_signals_v9_ml_enhanced.ml_predictor = MLSignalPredictor(
            model_type=ml_params["model_type"],
            model_path=ml_params["model_save_path"]
        )
    
    ml_predictor = gen_signals_v9_ml_enhanced.ml_predictor
    
    # 如果模型未训练，先训练
    if not ml_predictor.is_trained:
        print("训练ML模型...")
        ml_predictor.train_model(df, params)
    
    # 获取ML预测
    ml_predictions = ml_predictor.predict_signal_quality(df)
    out["ml_prediction"] = ml_predictions
    
    # 基础信号生成
    ofi_threshold = p.get("ofi_z_min", 1.4)
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # ML增强筛选
    ml_threshold = p.get("min_ml_prediction", 0.7)
    ml_enhanced = ml_predictions >= ml_threshold
    
    # 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = p.get("min_signal_strength", 1.8)
    strong_signal = signal_strength >= min_signal_strength
    
    # 价格动量确认
    price_momentum_threshold = p.get("price_momentum_threshold", 0.00001)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 组合ML增强信号
    long_mask = ofi_signal & strong_signal & ml_enhanced & direction_consistent_long
    short_mask = ofi_signal & strong_signal & ml_enhanced & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "ml_enhanced"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = ml_predictions[long_mask]
    out.loc[long_mask, "ml_confidence"] = ml_predictions[long_mask]
    
    out.loc[short_mask, "sig_type"] = "ml_enhanced"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = ml_predictions[short_mask]
    out.loc[short_mask, "ml_confidence"] = ml_predictions[short_mask]
    
    return out

def gen_signals_v9_real_time_optimized(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    v9 实时优化信号生成
    """
    p = params["signals"]["ml_enhanced"]
    rt_params = params["signals"]["real_time_optimization"]
    
    out = df.copy()
    out["sig_type"] = None
    out["sig_side"] = 0
    out["signal_strength"] = 0.0
    out["quality_score"] = 0.0
    out["real_time_score"] = 0.0
    
    # 实时优化参数
    if not hasattr(gen_signals_v9_real_time_optimized, 'optimization_state'):
        gen_signals_v9_real_time_optimized.optimization_state = {
            'performance_history': [],
            'current_threshold': p.get("ofi_z_min", 1.4),
            'adaptation_rate': rt_params.get("adaptation_rate", 0.1),
            'update_counter': 0
        }
    
    state = gen_signals_v9_real_time_optimized.optimization_state
    
    # 基础信号生成
    ofi_threshold = state['current_threshold']
    ofi_signal = abs(out["ofi_z"]) >= ofi_threshold
    
    # 信号强度筛选
    signal_strength = abs(out["ofi_z"])
    min_signal_strength = p.get("min_signal_strength", 1.8)
    strong_signal = signal_strength >= min_signal_strength
    
    # 价格动量确认
    price_momentum_threshold = p.get("price_momentum_threshold", 0.00001)
    price_momentum_long = out["ret_1s"] > price_momentum_threshold
    price_momentum_short = out["ret_1s"] < -price_momentum_threshold
    
    # 方向一致性检查
    direction_consistent_long = (out["ofi_z"] > 0) & price_momentum_long
    direction_consistent_short = (out["ofi_z"] < 0) & price_momentum_short
    
    # 组合实时优化信号
    long_mask = ofi_signal & strong_signal & direction_consistent_long
    short_mask = ofi_signal & strong_signal & direction_consistent_short
    
    # 应用信号
    out.loc[long_mask, "sig_type"] = "real_time_optimized"
    out.loc[long_mask, "sig_side"] = 1
    out.loc[long_mask, "signal_strength"] = signal_strength[long_mask]
    out.loc[long_mask, "quality_score"] = signal_strength[long_mask] / 3.0  # 归一化
    out.loc[long_mask, "real_time_score"] = signal_strength[long_mask] / 3.0
    
    out.loc[short_mask, "sig_type"] = "real_time_optimized"
    out.loc[short_mask, "sig_side"] = -1
    out.loc[short_mask, "signal_strength"] = signal_strength[short_mask]
    out.loc[short_mask, "quality_score"] = signal_strength[short_mask] / 3.0
    out.loc[short_mask, "real_time_score"] = signal_strength[short_mask] / 3.0
    
    # 实时优化更新
    state['update_counter'] += 1
    if state['update_counter'] >= rt_params.get("update_frequency", 10):
        # 更新优化状态
        state['update_counter'] = 0
        # 这里可以添加实时性能评估和阈值调整逻辑
    
    return out
