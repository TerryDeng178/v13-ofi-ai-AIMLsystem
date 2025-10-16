import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    """
    V10 高级特征工程 - 50+维度特征提取
    包含: 基础特征、技术指标、市场微观结构、时间序列特征、机器学习特征
    """
    
    def __init__(self, feature_dim: int = 50):
        self.feature_dim = feature_dim
        self.scalers = {}
        self.feature_importance = {}
        self.selected_features = []
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建基础特征 (20个)
        Args:
            df: 输入数据
        Returns:
            包含基础特征的DataFrame
        """
        features_df = df.copy()
        
        # 价格特征
        features_df['price_change'] = features_df['price'].pct_change()
        features_df['price_change_abs'] = features_df['price_change'].abs()
        features_df['price_volatility'] = features_df['price_change'].rolling(20).std()
        features_df['price_skewness'] = features_df['price_change'].rolling(20).skew()
        features_df['price_kurtosis'] = features_df['price_change'].rolling(20).kurt()
        
        # 成交量特征
        features_df['volume_change'] = features_df['volume'].pct_change()
        features_df['volume_ma_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
        features_df['volume_volatility'] = features_df['volume'].rolling(20).std()
        
        # 价差特征
        features_df['spread'] = features_df['ask1'] - features_df['bid1']
        features_df['spread_bps'] = features_df['spread'] / features_df['price'] * 10000
        features_df['spread_volatility'] = features_df['spread_bps'].rolling(20).std()
        
        # 深度特征
        features_df['bid_depth'] = features_df['bid1_size'] + features_df['bid2_size'] + features_df['bid3_size']
        features_df['ask_depth'] = features_df['ask1_size'] + features_df['ask2_size'] + features_df['ask3_size']
        features_df['depth_imbalance'] = (features_df['bid_depth'] - features_df['ask_depth']) / (features_df['bid_depth'] + features_df['ask_depth'])
        features_df['depth_ratio'] = features_df['bid_depth'] / features_df['ask_depth']
        
        # 订单流特征
        features_df['order_flow_imbalance'] = (features_df['bid1_size'] - features_df['ask1_size']) / (features_df['bid1_size'] + features_df['ask1_size'])
        features_df['order_flow_volatility'] = features_df['order_flow_imbalance'].rolling(20).std()
        
        # 时间特征
        features_df['hour'] = features_df['ts'].dt.hour
        features_df['minute'] = features_df['ts'].dt.minute
        features_df['day_of_week'] = features_df['ts'].dt.dayofweek
        features_df['is_market_open'] = ((features_df['hour'] >= 9) & (features_df['hour'] < 16)).astype(int)
        
        return features_df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建技术指标特征 (15个)
        Args:
            df: 输入数据
        Returns:
            包含技术指标的DataFrame
        """
        features_df = df.copy()
        
        # 移动平均线
        features_df['sma_5'] = talib.SMA(features_df['price'], timeperiod=5)
        features_df['sma_20'] = talib.SMA(features_df['price'], timeperiod=20)
        features_df['sma_50'] = talib.SMA(features_df['price'], timeperiod=50)
        features_df['ema_12'] = talib.EMA(features_df['price'], timeperiod=12)
        features_df['ema_26'] = talib.EMA(features_df['price'], timeperiod=26)
        
        # 技术指标
        features_df['rsi'] = talib.RSI(features_df['price'], timeperiod=14)
        features_df['macd'], features_df['macd_signal'], features_df['macd_hist'] = talib.MACD(features_df['price'])
        features_df['bb_upper'], features_df['bb_middle'], features_df['bb_lower'] = talib.BBANDS(features_df['price'])
        features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
        features_df['bb_position'] = (features_df['price'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # 波动率指标
        features_df['atr'] = talib.ATR(features_df['high'], features_df['low'], features_df['price'], timeperiod=14)
        features_df['natr'] = talib.NATR(features_df['high'], features_df['low'], features_df['price'], timeperiod=14)
        features_df['trange'] = talib.TRANGE(features_df['high'], features_df['low'], features_df['price'])
        
        return features_df
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建市场微观结构特征 (10个)
        Args:
            df: 输入数据
        Returns:
            包含微观结构特征的DataFrame
        """
        features_df = df.copy()
        
        # 订单流不平衡
        features_df['ofi_level1'] = (features_df['bid1_size'] - features_df['ask1_size']) / (features_df['bid1_size'] + features_df['ask1_size'])
        features_df['ofi_level2'] = ((features_df['bid1_size'] + features_df['bid2_size']) - (features_df['ask1_size'] + features_df['ask2_size'])) / ((features_df['bid1_size'] + features_df['bid2_size']) + (features_df['ask1_size'] + features_df['ask2_size']))
        features_df['ofi_level3'] = ((features_df['bid1_size'] + features_df['bid2_size'] + features_df['bid3_size']) - (features_df['ask1_size'] + features_df['ask2_size'] + features_df['ask3_size'])) / ((features_df['bid1_size'] + features_df['bid2_size'] + features_df['bid3_size']) + (features_df['ask1_size'] + features_df['ask2_size'] + features_df['ask3_size']))
        
        # 价格冲击
        features_df['price_impact_bid'] = features_df['price'].diff() / features_df['bid1_size'].shift(1)
        features_df['price_impact_ask'] = features_df['price'].diff() / features_df['ask1_size'].shift(1)
        
        # 流动性指标
        features_df['liquidity_ratio'] = features_df['volume'] / features_df['spread']
        features_df['liquidity_volatility'] = features_df['liquidity_ratio'].rolling(20).std()
        
        # 市场深度
        features_df['market_depth'] = features_df['bid_depth'] + features_df['ask_depth']
        features_df['depth_volatility'] = features_df['market_depth'].rolling(20).std()
        
        # 订单流强度
        features_df['order_flow_strength'] = features_df['ofi_level1'].rolling(10).mean()
        features_df['order_flow_acceleration'] = features_df['ofi_level1'].diff()
        
        return features_df
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间序列特征 (8个)
        Args:
            df: 输入数据
        Returns:
            包含时间序列特征的DataFrame
        """
        features_df = df.copy()
        
        # 自相关特征
        features_df['price_autocorr_1'] = features_df['price'].autocorr(lag=1)
        features_df['price_autocorr_5'] = features_df['price'].autocorr(lag=5)
        features_df['price_autocorr_10'] = features_df['price'].autocorr(lag=10)
        
        # 偏度和峰度
        features_df['price_skewness_rolling'] = features_df['price'].rolling(20).skew()
        features_df['price_kurtosis_rolling'] = features_df['price'].rolling(20).kurt()
        
        # 趋势强度
        features_df['trend_strength'] = features_df['price'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features_df['trend_consistency'] = features_df['price'].rolling(20).apply(lambda x: np.corrcoef(range(len(x)), x)[0, 1])
        
        # 周期性特征
        features_df['price_fft_1'] = np.abs(np.fft.fft(features_df['price'].values))[1]
        features_df['price_fft_5'] = np.abs(np.fft.fft(features_df['price'].values))[5]
        
        return features_df
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建机器学习特征 (7个)
        Args:
            df: 输入数据
        Returns:
            包含ML特征的DataFrame
        """
        features_df = df.copy()
        
        # PCA特征
        price_features = features_df[['price', 'volume', 'spread_bps', 'depth_imbalance']].fillna(0)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(price_features)
        features_df['pca_1'] = pca_features[:, 0]
        features_df['pca_2'] = pca_features[:, 1]
        
        # 聚类特征
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_features = features_df[['price', 'volume', 'spread_bps']].fillna(0)
        features_df['cluster_label'] = kmeans.fit_predict(cluster_features)
        
        # 异常检测特征
        features_df['price_zscore'] = np.abs((features_df['price'] - features_df['price'].rolling(20).mean()) / features_df['price'].rolling(20).std())
        features_df['volume_zscore'] = np.abs((features_df['volume'] - features_df['volume'].rolling(20).mean()) / features_df['volume'].rolling(20).std())
        features_df['spread_zscore'] = np.abs((features_df['spread_bps'] - features_df['spread_bps'].rolling(20).mean()) / features_df['spread_bps'].rolling(20).std())
        
        return features_df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        Args:
            df: 输入数据
        Returns:
            包含所有特征的DataFrame
        """
        print("开始创建高级特征...")
        
        # 基础特征
        features_df = self.create_basic_features(df)
        print(f"基础特征创建完成: {len([col for col in features_df.columns if col not in df.columns])}个特征")
        
        # 技术指标
        features_df = self.create_technical_indicators(features_df)
        print(f"技术指标创建完成: {len([col for col in features_df.columns if col not in df.columns])}个特征")
        
        # 市场微观结构特征
        features_df = self.create_market_microstructure_features(features_df)
        print(f"微观结构特征创建完成: {len([col for col in features_df.columns if col not in df.columns])}个特征")
        
        # 时间序列特征
        features_df = self.create_time_series_features(features_df)
        print(f"时间序列特征创建完成: {len([col for col in features_df.columns if col not in df.columns])}个特征")
        
        # 机器学习特征
        features_df = self.create_ml_features(features_df)
        print(f"机器学习特征创建完成: {len([col for col in features_df.columns if col not in df.columns])}个特征")
        
        # 填充缺失值
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"所有特征创建完成，总共{len(features_df.columns)}个特征")
        
        return features_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """
        特征选择
        Args:
            X: 特征数据
            y: 目标变量
            k: 选择的特征数量
        Returns:
            选择的特征列表
        """
        # 移除缺失值
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # 特征选择
        selector = SelectKBest(score_func=f_regression, k=min(k, X_clean.shape[1]))
        selector.fit(X_clean, y_clean)
        
        # 获取选择的特征
        selected_features = X_clean.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        
        # 保存特征重要性
        self.feature_importance = dict(zip(X_clean.columns, selector.scores_))
        
        print(f"特征选择完成，选择了{len(selected_features)}个特征")
        print(f"前10个重要特征: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        return selected_features
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        特征缩放
        Args:
            X: 特征数据
            method: 缩放方法 ('standard', 'minmax', 'robust')
        Returns:
            缩放后的特征数据
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")
        
        # 拟合和转换
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 保存缩放器
        self.scalers[method] = scaler
        
        print(f"特征缩放完成，使用方法: {method}")
        
        return X_scaled_df
    
    def create_sequence_data(self, df: pd.DataFrame, sequence_length: int = 60, 
                           target_col: str = 'signal_quality') -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        Args:
            df: 输入数据
            sequence_length: 序列长度
            target_col: 目标列名
        Returns:
            X: 序列特征 (samples, seq_len, features)
            y: 目标值 (samples,)
        """
        # 选择特征列
        feature_cols = [col for col in df.columns if col != target_col]
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_data)):
            X_sequences.append(X_data[i-sequence_length:i])
            y_sequences.append(y_data[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"序列数据创建完成:")
        print(f"  序列数量: {len(X_sequences)}")
        print(f"  序列长度: {sequence_length}")
        print(f"  特征维度: {X_sequences.shape[2]}")
        
        return X_sequences, y_sequences
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        Returns:
            特征重要性字典
        """
        return self.feature_importance
    
    def get_selected_features(self) -> List[str]:
        """
        获取选择的特征
        Returns:
            选择的特征列表
        """
        return self.selected_features

def create_advanced_features(df: pd.DataFrame, target_col: str = 'signal_quality',
                           feature_dim: int = 50, sequence_length: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    """
    创建高级特征的便捷函数
    Args:
        df: 输入数据
        target_col: 目标列名
        feature_dim: 特征维度
        sequence_length: 序列长度
    Returns:
        特征DataFrame和选择的特征列表
    """
    # 创建特征工程器
    feature_engineer = AdvancedFeatureEngineering(feature_dim)
    
    # 创建所有特征
    features_df = feature_engineer.create_all_features(df)
    
    # 选择特征
    feature_cols = [col for col in features_df.columns if col != target_col]
    selected_features = feature_engineer.select_features(
        features_df[feature_cols], 
        features_df[target_col], 
        k=feature_dim
    )
    
    # 缩放特征
    features_df[selected_features] = feature_engineer.scale_features(
        features_df[selected_features], 
        method='standard'
    )
    
    return features_df, selected_features

if __name__ == "__main__":
    # 测试高级特征工程
    print("测试高级特征工程...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(100, 1000, n_samples),
        'bid1': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.01,
        'ask1': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.01,
        'bid1_size': np.random.randint(100, 500, n_samples),
        'ask1_size': np.random.randint(100, 500, n_samples),
        'bid2': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.02,
        'ask2': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.02,
        'bid2_size': np.random.randint(50, 300, n_samples),
        'ask2_size': np.random.randint(50, 300, n_samples),
        'bid3': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.03,
        'ask3': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.03,
        'bid3_size': np.random.randint(20, 200, n_samples),
        'ask3_size': np.random.randint(20, 200, n_samples),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + 0.05,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - 0.05,
        'signal_quality': np.random.uniform(0, 1, n_samples)
    })
    
    # 创建高级特征
    features_df, selected_features = create_advanced_features(test_data, feature_dim=50)
    
    print(f"测试完成:")
    print(f"  原始特征数: {len(test_data.columns)}")
    print(f"  总特征数: {len(features_df.columns)}")
    print(f"  选择特征数: {len(selected_features)}")
    print(f"  特征形状: {features_df[selected_features].shape}")
    
    print("高级特征工程测试完成!")
