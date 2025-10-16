#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11币安真实数据回测训练器
使用币安真实历史数据训练V11系统
"""

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入V11模块
from src.v11_advanced_features import V11AdvancedFeatureEngine
from src.v11_deep_learning import V11DeepLearning
from src.v11_signal_optimizer import V11SignalOptimizer
from src.v11_risk_manager import V11RiskManager
from src.v11_advanced_backtest_optimizer import V11AdvancedBacktestOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11BinanceBacktestTrainer:
    """V11币安真实数据回测训练器"""
    
    def __init__(self, data_dir: str = "data/binance"):
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化V11组件
        self.feature_engineer = V11AdvancedFeatureEngine()
        self.ml_models = V11DeepLearning(device=self.device)
        self.signal_optimizer = V11SignalOptimizer()
        self.risk_manager = V11RiskManager()
        
        # 训练配置
        self.config = {
            'feature_dim': 128,
            'sequence_length': 60,
            'optimization_strategy': 'adaptive',
            'training_split': 0.8,  # 80%训练，20%测试
            'validation_split': 0.2,  # 20%验证
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001
        }
        
        # 训练结果存储
        self.training_results = {}
        self.model_performance = {}
        
        logger.info(f"V11币安回测训练器初始化完成，设备: {self.device}")
    
    def load_binance_data(self, symbol: str = "ETHUSDT", interval: str = "1m") -> pd.DataFrame:
        """
        加载币安数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
        
        Returns:
            数据DataFrame
        """
        logger.info(f"加载币安数据: {symbol} {interval}")
        
        # 查找数据文件
        data_files = [f for f in os.listdir(self.data_dir) 
                     if f.startswith(f"{symbol}_{interval}") and f.endswith('.csv')]
        
        if not data_files:
            raise FileNotFoundError(f"未找到 {symbol} {interval} 数据文件")
        
        # 使用最新的数据文件
        latest_file = sorted(data_files)[-1]
        file_path = os.path.join(self.data_dir, latest_file)
        
        logger.info(f"加载数据文件: {file_path}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 数据预处理
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 移除缺失值
        df = df.dropna()
        
        logger.info(f"数据加载完成: {len(df)} 条记录")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        准备训练数据
        
        Args:
            df: 原始数据
        
        Returns:
            训练集、验证集、测试集
        """
        logger.info("准备训练数据...")
        
        # 特征工程
        logger.info("执行特征工程...")
        df_features = self.feature_engineer.create_all_features(df)
        
        # 创建标签 (未来收益率)
        df_features['future_return'] = df_features['close'].pct_change(periods=5).shift(-5)
        
        # 创建交易信号标签
        df_features['signal_label'] = 0  # 0: 持有
        df_features.loc[df_features['future_return'] > 0.001, 'signal_label'] = 1  # 1: 买入
        df_features.loc[df_features['future_return'] < -0.001, 'signal_label'] = -1  # -1: 卖出
        
        # 移除缺失值
        df_features = df_features.dropna()
        
        # 数据分割
        total_len = len(df_features)
        train_len = int(total_len * self.config['training_split'])
        val_len = int(total_len * self.config['validation_split'])
        
        train_data = df_features[:train_len].copy()
        val_data = df_features[train_len:train_len + val_len].copy()
        test_data = df_features[train_len + val_len:].copy()
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(train_data)} 条记录")
        logger.info(f"  验证集: {len(val_data)} 条记录")
        logger.info(f"  测试集: {len(test_data)} 条记录")
        
        return train_data, val_data, test_data
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, any]:
        """
        训练V11模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
        
        Returns:
            训练结果
        """
        logger.info("开始训练V11模型...")
        
        # 准备训练数据
        X_train, y_train = self._prepare_model_data(train_data)
        X_val, y_val = self._prepare_model_data(val_data)
        
        training_results = {}
        
        # 训练各个模型
        models = ['LSTM', 'Transformer', 'CNN', 'Ensemble']
        
        for model_name in models:
            logger.info(f"训练 {model_name} 模型...")
            
            try:
                # 训练模型
                model_result = self.ml_models.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    learning_rate=self.config['learning_rate']
                )
                
                training_results[model_name] = model_result
                
                logger.info(f"{model_name} 模型训练完成")
                logger.info(f"  训练损失: {model_result.get('train_loss', 'N/A'):.4f}")
                logger.info(f"  验证损失: {model_result.get('val_loss', 'N/A'):.4f}")
                logger.info(f"  训练准确率: {model_result.get('train_acc', 'N/A'):.4f}")
                logger.info(f"  验证准确率: {model_result.get('val_acc', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"训练 {model_name} 模型失败: {e}")
                training_results[model_name] = {"error": str(e)}
        
        return training_results
    
    def _prepare_model_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备模型训练数据
        
        Args:
            data: 数据DataFrame
        
        Returns:
            特征和标签数组
        """
        # 选择数值特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['signal_label', 'future_return']]
        
        # 创建序列数据
        sequence_length = self.config['sequence_length']
        features = []
        labels = []
        
        for i in range(sequence_length, len(data)):
            # 特征序列
            feature_seq = data[feature_cols].iloc[i-sequence_length:i].values
            features.append(feature_seq)
            
            # 标签
            label = data['signal_label'].iloc[i]
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def run_backtest(self, test_data: pd.DataFrame) -> Dict[str, any]:
        """
        运行回测
        
        Args:
            test_data: 测试数据
        
        Returns:
            回测结果
        """
        logger.info("运行V11回测...")
        
        # 创建回测优化器
        optimizer = V11AdvancedBacktestOptimizer(self.config)
        
        # 运行回测
        backtest_result = optimizer.run_advanced_optimization_cycle(
            test_data, 
            max_iterations=10  # 减少迭代次数以加快速度
        )
        
        return backtest_result
    
    def evaluate_performance(self, backtest_result: Dict[str, any]) -> Dict[str, any]:
        """
        评估性能
        
        Args:
            backtest_result: 回测结果
        
        Returns:
            性能评估结果
        """
        logger.info("评估V11性能...")
        
        summary = backtest_result.get('optimization_summary', {})
        
        performance = {
            'overall_score': summary.get('best_overall_score', 0),
            'total_return': summary.get('best_total_return', 0),
            'sharpe_ratio': summary.get('best_sharpe_ratio', 0),
            'max_drawdown': summary.get('best_max_drawdown', 0),
            'win_rate': summary.get('best_win_rate', 0),
            'profit_factor': summary.get('best_profit_factor', 0),
            'trading_ready': summary.get('trading_ready', False)
        }
        
        # 计算改进指标
        baseline_return = 0.1  # 基准收益率10%
        improvement = {
            'return_improvement': (performance['total_return'] - baseline_return) / baseline_return,
            'sharpe_improvement': performance['sharpe_ratio'] - 1.0,
            'risk_adjusted_return': performance['total_return'] / (1 + abs(performance['max_drawdown']))
        }
        
        performance.update(improvement)
        
        return performance
    
    def train_and_evaluate(self, symbol: str = "ETHUSDT", interval: str = "1m") -> Dict[str, any]:
        """
        完整的训练和评估流程
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
        
        Returns:
            完整的训练和评估结果
        """
        logger.info("=" * 80)
        logger.info(f"V11币安真实数据训练 - {symbol} {interval}")
        logger.info("=" * 80)
        
        try:
            # 1. 加载数据
            df = self.load_binance_data(symbol, interval)
            
            # 2. 准备训练数据
            train_data, val_data, test_data = self.prepare_training_data(df)
            
            # 3. 训练模型
            training_results = self.train_models(train_data, val_data)
            
            # 4. 运行回测
            backtest_result = self.run_backtest(test_data)
            
            # 5. 评估性能
            performance = self.evaluate_performance(backtest_result)
            
            # 6. 整合结果
            final_results = {
                'symbol': symbol,
                'interval': interval,
                'training_time': datetime.now().isoformat(),
                'data_info': {
                    'total_records': len(df),
                    'train_records': len(train_data),
                    'val_records': len(val_data),
                    'test_records': len(test_data),
                    'time_range': {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    }
                },
                'training_results': training_results,
                'backtest_results': backtest_result,
                'performance': performance,
                'config': self.config
            }
            
            # 7. 保存结果
            self.save_results(final_results, symbol, interval)
            
            # 8. 输出总结
            self.print_summary(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            return {"error": str(e)}
    
    def save_results(self, results: Dict[str, any], symbol: str, interval: str):
        """
        保存训练结果
        
        Args:
            results: 训练结果
            symbol: 交易对符号
            interval: 时间间隔
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"v11_binance_training_results_{symbol}_{interval}_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"训练结果已保存到: {results_file}")
    
    def print_summary(self, results: Dict[str, any]):
        """
        打印训练总结
        
        Args:
            results: 训练结果
        """
        logger.info("=" * 80)
        logger.info("V11币安真实数据训练总结")
        logger.info("=" * 80)
        
        if 'error' in results:
            logger.error(f"训练失败: {results['error']}")
            return
        
        # 数据信息
        data_info = results['data_info']
        logger.info(f"数据信息:")
        logger.info(f"  交易对: {results['symbol']} {results['interval']}")
        logger.info(f"  总记录数: {data_info['total_records']:,}")
        logger.info(f"  训练集: {data_info['train_records']:,}")
        logger.info(f"  验证集: {data_info['val_records']:,}")
        logger.info(f"  测试集: {data_info['test_records']:,}")
        logger.info(f"  时间范围: {data_info['time_range']['start']} 到 {data_info['time_range']['end']}")
        
        # 训练结果
        training_results = results['training_results']
        logger.info(f"\n模型训练结果:")
        for model_name, result in training_results.items():
            if 'error' in result:
                logger.error(f"  {model_name}: 训练失败 - {result['error']}")
            else:
                logger.info(f"  {model_name}:")
                logger.info(f"    训练损失: {result.get('train_loss', 'N/A'):.4f}")
                logger.info(f"    验证损失: {result.get('val_loss', 'N/A'):.4f}")
                logger.info(f"    训练准确率: {result.get('train_acc', 'N/A'):.4f}")
                logger.info(f"    验证准确率: {result.get('val_acc', 'N/A'):.4f}")
        
        # 性能结果
        performance = results['performance']
        logger.info(f"\n最终性能:")
        logger.info(f"  综合评分: {performance['overall_score']:.2f}")
        logger.info(f"  年化收益率: {performance['total_return']:.1%}")
        logger.info(f"  夏普比率: {performance['sharpe_ratio']:.2f}")
        logger.info(f"  最大回撤: {performance['max_drawdown']:.1%}")
        logger.info(f"  胜率: {performance['win_rate']:.1%}")
        logger.info(f"  盈利因子: {performance['profit_factor']:.2f}")
        logger.info(f"  交易准备度: {'✅ 已准备' if performance['trading_ready'] else '❌ 未准备'}")
        
        # 改进指标
        logger.info(f"\n改进指标:")
        logger.info(f"  收益改进: {performance['return_improvement']:.1%}")
        logger.info(f"  夏普改进: {performance['sharpe_improvement']:.2f}")
        logger.info(f"  风险调整收益: {performance['risk_adjusted_return']:.2f}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11币安真实数据训练系统")
    logger.info("=" * 80)
    
    # 创建训练器
    trainer = V11BinanceBacktestTrainer()
    
    # 训练V11系统
    results = trainer.train_and_evaluate("ETHUSDT", "1m")
    
    if 'error' not in results:
        logger.info("V11币安真实数据训练完成！")
        logger.info("系统已准备好进行币安测试网实战！")
    else:
        logger.error("训练失败，请检查数据和配置")


if __name__ == "__main__":
    main()
