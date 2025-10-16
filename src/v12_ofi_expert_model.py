"""
V12 OFI专家模型
基于V9机器学习集成 + 真实OFI数据的专业AI模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12OFIExpertModel:
    """
    V12 OFI专家模型
    基于V9机器学习集成 + 真实OFI数据
    """
    
    def __init__(self, model_type="ensemble", model_path="models/v12/"):
        """
        初始化OFI专家模型
        
        Args:
            model_type: 模型类型 (ensemble, xgboost, lightgbm, neural_network)
            model_path: 模型保存路径
        """
        self.model_type = model_type
        self.model_path = model_path
        
        # 确保模型目录存在
        os.makedirs(model_path, exist_ok=True)
        
        # 模型组件
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # V9特征配置 (继承)
        self.v9_features = [
            "ofi_z", "cvd_z", "ret_1s", "atr", "vwap", 
            "bid1_size", "ask1_size", "spread_bps", "depth_ratio",
            "price_volatility", "ofi_volatility", "rsi", "macd",
            "bollinger_upper", "bollinger_lower", "trend_strength", "volatility_regime"
        ]
        
        # V12新增特征
        self.v12_features = [
            "real_ofi_z", "real_cvd_z", "ofi_momentum_1s", "ofi_momentum_5s", 
            "ofi_momentum_30s", "cvd_momentum_1s", "cvd_momentum_5s",
            "order_book_imbalance", "depth_pressure", "spread_trend",
            "volume_profile", "time_of_day", "market_regime"
        ]
        
        # 完整特征列表
        self.all_features = self.v9_features + self.v12_features
        
        # 训练数据存储
        self.training_data = []
        self.validation_data = []
        
        # 性能历史
        self.performance_history = []
        
        # 统计信息
        self.stats = {
            'training_samples': 0,
            'validation_samples': 0,
            'model_accuracy': 0.0,
            'last_training': None,
            'feature_importance': {}
        }
        
        logger.info(f"V12 OFI专家模型初始化完成 - 模型类型: {model_type}, 特征数: {len(self.all_features)}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备机器学习特征 (V9 + V12)
        
        Args:
            df: 输入数据框
            
        Returns:
            特征数据框
        """
        try:
            features_df = df.copy()
            
            # V9基础特征
            features_df["spread_bps"] = (features_df["ask1"] - features_df["bid1"]) / features_df["price"] * 1e4
            features_df["depth_ratio"] = (features_df["bid1_size"] + features_df["ask1_size"]) / \
                                       (features_df["bid1_size"] + features_df["ask1_size"]).rolling(100).quantile(0.8)
            features_df["price_volatility"] = features_df["ret_1s"].rolling(50).std()
            features_df["ofi_volatility"] = features_df["ofi_z"].rolling(50).std()
            
            # V9技术指标特征
            features_df["rsi"] = self._calculate_rsi(features_df["ret_1s"], 14)
            features_df["macd"] = self._calculate_macd(features_df["price"], 12, 26, 9)
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(features_df["price"], 20, 2)
            features_df["bollinger_upper"] = bollinger_upper
            features_df["bollinger_lower"] = bollinger_lower
            
            # V9市场状态特征
            features_df["trend_strength"] = self._calculate_trend_strength(features_df["price"], 50)
            features_df["volatility_regime"] = self._calculate_volatility_regime(features_df["ret_1s"], 100)
            
            # V12新增特征 (真实OFI相关)
            if 'real_ofi_z' in df.columns:
                features_df["real_ofi_z"] = df["real_ofi_z"]
            else:
                features_df["real_ofi_z"] = df.get("ofi_z", 0.0)
            
            if 'real_cvd_z' in df.columns:
                features_df["real_cvd_z"] = df["real_cvd_z"]
            else:
                features_df["real_cvd_z"] = df.get("cvd_z", 0.0)
            
            # V12 OFI动量特征
            features_df["ofi_momentum_1s"] = features_df["real_ofi_z"].diff(1)
            features_df["ofi_momentum_5s"] = features_df["real_ofi_z"].diff(5)
            features_df["ofi_momentum_30s"] = features_df["real_ofi_z"].diff(30)
            
            # V12 CVD动量特征
            features_df["cvd_momentum_1s"] = features_df["real_cvd_z"].diff(1)
            features_df["cvd_momentum_5s"] = features_df["real_cvd_z"].diff(5)
            
            # V12订单簿不平衡特征
            features_df["order_book_imbalance"] = (features_df["bid1_size"] - features_df["ask1_size"]) / \
                                                 (features_df["bid1_size"] + features_df["ask1_size"] + 1e-9)
            
            # V12深度压力特征
            total_bid_size = sum([features_df.get(f"bid{i+1}_size", 0) for i in range(5)])
            total_ask_size = sum([features_df.get(f"ask{i+1}_size", 0) for i in range(5)])
            features_df["depth_pressure"] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-9)
            
            # V12价差趋势特征
            features_df["spread_trend"] = features_df["spread_bps"].rolling(10).mean()
            
            # V12成交量分布特征
            features_df["volume_profile"] = features_df["size"].rolling(20).std() / features_df["size"].rolling(20).mean()
            
            # V12时间特征
            if 'timestamp' in df.columns:
                features_df["time_of_day"] = pd.to_datetime(df['timestamp']).dt.hour / 24.0
            else:
                features_df["time_of_day"] = 0.5
            
            # V12市场状态特征
            features_df["market_regime"] = self._calculate_market_regime(features_df)
            
            return features_df
            
        except Exception as e:
            logger.error(f"准备特征失败: {e}")
            return df
    
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
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """计算市场状态"""
        # 基于多个指标综合判断市场状态
        trend = df.get("trend_strength", pd.Series(0, index=df.index))
        volatility = df.get("volatility_regime", pd.Series(0.5, index=df.index))
        volume = df.get("volume_profile", pd.Series(0.5, index=df.index))
        
        # 综合评分 (0-1)
        regime_score = (trend.abs() * 0.4 + volatility * 0.3 + volume * 0.3).fillna(0.5)
        return regime_score
    
    def create_training_data(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        创建训练数据，标记信号质量
        
        Args:
            df: 输入数据
            params: 策略参数
            
        Returns:
            训练数据框
        """
        try:
            # 准备特征
            features_df = self.prepare_features(df)
            
            # 创建目标变量（信号质量）
            features_df["signal_quality"] = 0.0
            
            # 基于V9逻辑标记高质量信号
            ofi_z_min = params.get("signals", {}).get("momentum", {}).get("ofi_z_min", 1.4)
            cvd_z_min = params.get("signals", {}).get("momentum", {}).get("cvd_z_min", 0.6)
            min_signal_strength = params.get("signals", {}).get("momentum", {}).get("min_signal_strength", 1.8)
            
            # 计算信号强度
            signal_strength = (abs(features_df["real_ofi_z"]) + abs(features_df["real_cvd_z"])) / 2
            
            # 标记高质量信号
            high_quality_mask = (
                (abs(features_df["real_ofi_z"]) >= ofi_z_min) &
                (abs(features_df["real_cvd_z"]) >= cvd_z_min) &
                (signal_strength >= min_signal_strength)
            )
            features_df.loc[high_quality_mask, "signal_quality"] = 1.0
            
            # 标记中等质量信号
            medium_quality_mask = (
                (abs(features_df["real_ofi_z"]) >= ofi_z_min * 0.7) &
                (abs(features_df["real_cvd_z"]) >= cvd_z_min * 0.7) &
                (signal_strength >= min_signal_strength * 0.7) &
                (features_df["signal_quality"] == 0.0)
            )
            features_df.loc[medium_quality_mask, "signal_quality"] = 0.5
            
            return features_df
            
        except Exception as e:
            logger.error(f"创建训练数据失败: {e}")
            return df
    
    def train_model(self, df: pd.DataFrame, params: dict):
        """
        训练OFI专家模型
        
        Args:
            df: 训练数据
            params: 策略参数
        """
        try:
            logger.info("开始训练V12 OFI专家模型...")
            
            # 创建训练数据
            training_df = self.create_training_data(df, params)
            
            # 选择特征列
            available_features = [col for col in self.all_features if col in training_df.columns]
            logger.info(f"可用特征: {len(available_features)}/{len(self.all_features)}")
            
            # 过滤有效数据
            valid_data = training_df[available_features + ["signal_quality"]].dropna()
            
            if len(valid_data) < 100:
                logger.warning("训练数据不足，跳过模型训练")
                return
            
            X = valid_data[available_features]
            y = valid_data["signal_quality"]
            
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和验证集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 根据模型类型选择算法
            if self.model_type == "ensemble":
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            
            # 训练模型
            self.model.fit(X_train, y_train)
            self.feature_columns = available_features
            self.is_trained = True
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 更新统计信息
            self.stats['training_samples'] = len(X_train)
            self.stats['validation_samples'] = len(X_test)
            self.stats['model_accuracy'] = r2
            self.stats['last_training'] = datetime.now()
            
            # 计算特征重要性
            if hasattr(self.model, 'feature_importances_'):
                self.stats['feature_importance'] = dict(zip(
                    available_features, 
                    self.model.feature_importances_
                ))
            
            # 保存模型
            self._save_model()
            
            logger.info(f"V12 OFI专家模型训练完成 - 准确率: {r2:.4f}, MSE: {mse:.4f}")
            logger.info(f"训练样本: {len(X_train)}, 验证样本: {len(X_test)}")
            
            # 显示特征重要性
            if self.stats['feature_importance']:
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in self.stats['feature_importance'].items()
                ]).sort_values('importance', ascending=False)
                logger.info("特征重要性 (Top 10):")
                logger.info(importance_df.head(10).to_string(index=False))
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
    
    def predict_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        预测信号质量
        
        Args:
            df: 输入数据
            
        Returns:
            信号质量预测
        """
        try:
            if not self.is_trained or self.model is None:
                # 如果模型未训练，返回默认评分
                return pd.Series(0.5, index=df.index)
            
            # 准备特征
            features_df = self.prepare_features(df)
            
            # 选择特征列
            available_features = [col for col in self.feature_columns if col in features_df.columns]
            
            if not available_features:
                logger.warning("没有可用的特征列")
                return pd.Series(0.5, index=df.index)
            
            # 数据预处理
            X = features_df[available_features].fillna(0)
            
            # 标准化
            X_scaled = self.scaler.transform(X)
            
            # 预测
            predictions = self.model.predict(X_scaled)
            
            # 确保预测值在合理范围内
            predictions = np.clip(predictions, 0.0, 1.0)
            
            return pd.Series(predictions, index=df.index)
            
        except Exception as e:
            logger.error(f"预测信号质量失败: {e}")
            return pd.Series(0.5, index=df.index)
    
    def update_model_performance(self, actual_quality: float, predicted_quality: float):
        """
        更新模型性能
        
        Args:
            actual_quality: 实际质量
            predicted_quality: 预测质量
        """
        try:
            self.performance_history.append({
                'actual': actual_quality,
                'predicted': predicted_quality,
                'error': abs(actual_quality - predicted_quality),
                'timestamp': datetime.now()
            })
            
            # 保持最近1000条记录
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"更新模型性能失败: {e}")
    
    def _save_model(self):
        """保存模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存模型
            model_file = os.path.join(self.model_path, f"v12_ofi_expert_{self.model_type}_{timestamp}.joblib")
            joblib.dump(self.model, model_file)
            
            # 保存标准化器
            scaler_file = os.path.join(self.model_path, f"v12_ofi_expert_scaler_{timestamp}.joblib")
            joblib.dump(self.scaler, scaler_file)
            
            # 保存特征列
            features_file = os.path.join(self.model_path, f"v12_ofi_expert_features_{timestamp}.joblib")
            joblib.dump(self.feature_columns, features_file)
            
            # 保存统计信息
            stats_file = os.path.join(self.model_path, f"v12_ofi_expert_stats_{timestamp}.joblib")
            joblib.dump(self.stats, stats_file)
            
            logger.info(f"V12 OFI专家模型已保存到: {model_file}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_model(self, model_file: str):
        """
        加载模型
        
        Args:
            model_file: 模型文件路径
        """
        try:
            # 加载模型
            self.model = joblib.load(model_file)
            
            # 加载标准化器
            scaler_file = model_file.replace("model_", "scaler_")
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
            
            # 加载特征列
            features_file = model_file.replace("model_", "features_")
            if os.path.exists(features_file):
                self.feature_columns = joblib.load(features_file)
            
            # 加载统计信息
            stats_file = model_file.replace("model_", "stats_")
            if os.path.exists(stats_file):
                self.stats = joblib.load(stats_file)
            
            self.is_trained = True
            logger.info(f"V12 OFI专家模型已加载: {model_file}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'training_samples': self.stats['training_samples'],
            'validation_samples': self.stats['validation_samples'],
            'model_accuracy': self.stats['model_accuracy'],
            'last_training': self.stats['last_training'],
            'performance_history_count': len(self.performance_history)
        }


def test_v12_ofi_expert_model():
    """测试V12 OFI专家模型"""
    logger.info("开始测试V12 OFI专家模型...")
    
    # 创建专家模型
    expert_model = V12OFIExpertModel(model_type="ensemble")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟特征数据
    mock_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': 3000 + np.random.randn(n_samples).cumsum() * 0.1,
        'bid1': 3000 + np.random.randn(n_samples).cumsum() * 0.1 - 0.5,
        'ask1': 3000 + np.random.randn(n_samples).cumsum() * 0.1 + 0.5,
        'bid1_size': 100 + np.random.randn(n_samples) * 10,
        'ask1_size': 100 + np.random.randn(n_samples) * 10,
        'size': 50 + np.random.randn(n_samples) * 5,
        'ofi_z': np.random.randn(n_samples) * 2,
        'cvd_z': np.random.randn(n_samples) * 2,
        'ret_1s': np.random.randn(n_samples) * 0.001,
        'atr': 1.0 + np.random.randn(n_samples) * 0.1,
        'vwap': 3000 + np.random.randn(n_samples) * 0.1
    }
    
    df = pd.DataFrame(mock_data)
    
    # V12参数配置
    v12_params = {
        'signals': {
            'momentum': {
                'ofi_z_min': 1.4,
                'cvd_z_min': 0.6,
                'min_signal_strength': 1.8
            }
        }
    }
    
    # 训练模型
    expert_model.train_model(df, v12_params)
    
    # 预测信号质量
    predictions = expert_model.predict_signal_quality(df)
    logger.info(f"预测结果统计: 均值={predictions.mean():.4f}, 标准差={predictions.std():.4f}")
    
    # 获取统计信息
    stats = expert_model.get_statistics()
    logger.info(f"模型统计信息: {stats}")
    
    logger.info("V12 OFI专家模型测试完成")


if __name__ == "__main__":
    test_v12_ofi_expert_model()
