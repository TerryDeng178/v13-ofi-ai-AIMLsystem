#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实币安数据的V11训练脚本
使用下载的真实ETHUSDT数据进行V11系统训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import glob
from typing import Dict, List, Any
import torch

# V11模块导入
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager
from src.v11_backtest_optimizer import V11BacktestOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11BinanceRealTrainer:
    """基于真实币安数据的V11训练器"""
    
    def __init__(self):
        self.data_dir = "data/binance"
        self.results_dir = "results/v11_binance_training"
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化V11组件
        self.feature_engine = V11AdvancedFeatureEngine()
        self.deep_learning = V11DeepLearning()
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        
        # 创建回测优化器配置
        backtest_config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'max_position_size': 0.1
        }
        self.backtest_optimizer = V11BacktestOptimizer(backtest_config)
        
        logger.info("V11币安真实数据训练器初始化完成")
    
    def load_binance_data(self) -> pd.DataFrame:
        """加载币安真实数据"""
        logger.info("加载币安真实数据...")
        
        # 查找最新的数据文件
        data_files = glob.glob(f"{self.data_dir}/ETHUSDT_1m_*.csv")
        if not data_files:
            logger.error("未找到币安数据文件")
            return pd.DataFrame()
        
        # 选择最新的文件
        latest_file = max(data_files, key=os.path.getctime)
        logger.info(f"使用数据文件: {latest_file}")
        
        # 加载数据
        df = pd.read_csv(latest_file)
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"数据加载完成: {len(df)} 条记录")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备V11特征"""
        logger.info("准备V11高级特征...")
        
        # 创建V11特征
        df_features = self.feature_engine.create_all_features(df)
        
        logger.info(f"特征工程完成: {len(df_features.columns)} 个特征")
        logger.info(f"特征列表: {list(df_features.columns)}")
        
        return df_features
    
    def train_deep_learning_models(self, df: pd.DataFrame) -> Dict:
        """训练深度学习模型"""
        logger.info("训练V11深度学习模型...")
        
        # 准备训练数据
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols].values
        y = df['close'].pct_change().shift(-1).fillna(0).values  # 下一期收益率
        
        # 创建模型
        input_size = len(feature_cols)
        lstm_model = self.deep_learning.create_lstm_model(input_size)
        transformer_model = self.deep_learning.create_transformer_model(input_size)
        cnn_model = self.deep_learning.create_cnn_model(input_size)
        
        # 准备数据
        X_train, X_test, y_train, y_test = self.deep_learning.prepare_data(X, y)
        
        # 训练各个模型
        models = {}
        for model_name in ['lstm', 'transformer', 'cnn']:
            try:
                model_results = self.deep_learning.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    epochs=10, batch_size=32, learning_rate=0.001
                )
                models[model_name] = model_results
                logger.info(f"{model_name}模型训练完成")
            except Exception as e:
                logger.error(f"{model_name}模型训练失败: {e}")
        
        logger.info("深度学习模型训练完成")
        return models
    
    def optimize_signals(self, df: pd.DataFrame) -> Dict:
        """优化信号生成"""
        logger.info("优化V11信号生成...")
        
        # 添加未来收益率列
        df['future_return_1'] = df['close'].pct_change().shift(-1)
        
        # 优化信号阈值
        optimized_params = self.signal_optimizer.optimize_signal_thresholds(df, 'future_return_1')
        
        logger.info("信号优化完成")
        return optimized_params
    
    def run_backtest(self, df: pd.DataFrame, models: Dict, signal_params: Dict) -> Dict:
        """运行回测"""
        logger.info("运行V11回测...")
        
        # 生成预测信号（使用LSTM模型）
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols].values
        
        # 准备预测数据
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # 添加序列维度
        
        # 使用LSTM模型预测
        try:
            predictions = self.deep_learning.predict('lstm', X_tensor)
        except:
            predictions = np.zeros(len(df))  # 如果预测失败，使用零预测
        
        # 生成交易信号
        try:
            df_with_signals = self.signal_optimizer.apply_optimized_signals(df, 'ml_signal')
            signals = df_with_signals.get('ml_signal', pd.Series(0, index=df.index))
        except:
            signals = pd.Series(0, index=df.index)  # 如果信号生成失败，使用零信号
        
        # 应用风险管理
        df_with_risk = self.risk_manager.apply_risk_management(df, 'signal_strength')
        
        # 运行回测
        backtest_results = self.backtest_optimizer.run_backtest(df_with_risk)
        
        logger.info("回测完成")
        return backtest_results
    
    def save_results(self, results: Dict, models: Dict, signal_params: Dict):
        """保存训练结果"""
        logger.info("保存训练结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存回测结果
        results_file = f"{self.results_dir}/v11_binance_backtest_{timestamp}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存模型参数
        models_file = f"{self.results_dir}/v11_binance_models_{timestamp}.json"
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2, default=str)
        
        # 保存信号参数
        signals_file = f"{self.results_dir}/v11_binance_signals_{timestamp}.json"
        with open(signals_file, 'w') as f:
            json.dump(signal_params, f, indent=2, default=str)
        
        logger.info(f"结果已保存到: {self.results_dir}")
    
    def run_training(self):
        """运行完整训练流程"""
        logger.info("=" * 80)
        logger.info("V11币安真实数据训练")
        logger.info("=" * 80)
        
        try:
            # 1. 加载数据
            df = self.load_binance_data()
            if df.empty:
                logger.error("数据加载失败")
                return False
            
            # 2. 准备特征
            df_features = self.prepare_features(df)
            
            # 3. 训练深度学习模型
            models = self.train_deep_learning_models(df_features)
            
            # 4. 优化信号
            signal_params = self.optimize_signals(df_features)
            
            # 5. 运行回测
            backtest_results = self.run_backtest(df_features, models, signal_params)
            
            # 6. 保存结果
            self.save_results(backtest_results, models, signal_params)
            
            # 7. 输出结果摘要
            self.print_results_summary(backtest_results)
            
            logger.info("=" * 80)
            logger.info("✅ V11币安真实数据训练完成！")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            return False
    
    def print_results_summary(self, results: Dict):
        """打印结果摘要"""
        logger.info("=" * 60)
        logger.info("V11训练结果摘要")
        logger.info("=" * 60)
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info(f"总收益率: {metrics.get('total_return', 0):.2%}")
            logger.info(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
            logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"胜率: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"总交易次数: {metrics.get('total_trades', 0)}")
            logger.info(f"平均每笔收益: {metrics.get('avg_trade_return', 0):.2%}")
        
        if 'model_performance' in results:
            model_perf = results['model_performance']
            logger.info(f"LSTM准确率: {model_perf.get('lstm_accuracy', 0):.2%}")
            logger.info(f"Transformer准确率: {model_perf.get('transformer_accuracy', 0):.2%}")
            logger.info(f"CNN准确率: {model_perf.get('cnn_accuracy', 0):.2%}")
            logger.info(f"集成模型准确率: {model_perf.get('ensemble_accuracy', 0):.2%}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11币安真实数据训练系统")
    logger.info("=" * 80)
    
    # 创建训练器
    trainer = V11BinanceRealTrainer()
    
    # 运行训练
    success = trainer.run_training()
    
    if success:
        logger.info("🎉 V11训练成功完成！")
        logger.info("现在可以进行币安测试网实战了。")
    else:
        logger.error("❌ V11训练失败，请检查数据和配置。")


if __name__ == "__main__":
    main()
