#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11信号生成优化模块
实现信号阈值优化、权重优化、时机优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11SignalOptimizer:
    """V11信号生成优化器"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_parameters = {}
        self.performance_metrics = {}
        
        logger.info("V11信号生成优化器初始化完成")
    
    def optimize_signal_thresholds(self, df: pd.DataFrame, target_column: str = 'future_return_1',
                                 feature_columns: List[str] = None, 
                                 optimization_method: str = 'genetic') -> Dict:
        """优化信号阈值"""
        logger.info("开始信号阈值优化...")
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in ['open_time', 'close_time', 'ignore', target_column]]
        
        # 准备数据
        X = df[feature_columns].values
        y = df[target_column].values
        
        # 创建信号标签
        y_labels = np.where(y > 0.001, 1, np.where(y < -0.001, -1, 0))
        
        if optimization_method == 'genetic':
            best_params = self._genetic_algorithm_optimization(X, y_labels)
        elif optimization_method == 'grid_search':
            best_params = self._grid_search_optimization(X, y_labels)
        elif optimization_method == 'bayesian':
            best_params = self._bayesian_optimization(X, y_labels)
        else:
            best_params = self._gradient_optimization(X, y_labels)
        
        self.best_parameters['thresholds'] = best_params
        
        logger.info(f"信号阈值优化完成: {best_params}")
        return best_params
    
    def optimize_signal_weights(self, df: pd.DataFrame, target_column: str = 'future_return_1',
                              signal_columns: List[str] = None) -> Dict:
        """优化信号权重"""
        logger.info("开始信号权重优化...")
        
        if signal_columns is None:
            signal_columns = [col for col in df.columns if 'signal' in col.lower() or 'prediction' in col.lower()]
        
        # 准备信号数据
        signals = df[signal_columns].values
        targets = df[target_column].values
        
        # 创建目标标签
        y_labels = np.where(targets > 0.001, 1, np.where(targets < -0.001, -1, 0))
        
        # 权重优化
        best_weights = self._optimize_signal_weights(signals, y_labels)
        
        self.best_parameters['weights'] = best_weights
        
        logger.info(f"信号权重优化完成: {best_weights}")
        return best_weights
    
    def optimize_signal_timing(self, df: pd.DataFrame, target_column: str = 'future_return_1',
                             signal_column: str = 'ml_signal') -> Dict:
        """优化信号时机"""
        logger.info("开始信号时机优化...")
        
        # 准备数据
        signals = df[signal_column].values
        targets = df[target_column].values
        
        # 创建目标标签
        y_labels = np.where(targets > 0.001, 1, np.where(targets < -0.001, -1, 0))
        
        # 时机优化
        best_timing = self._optimize_signal_timing(signals, y_labels)
        
        self.best_parameters['timing'] = best_timing
        
        logger.info(f"信号时机优化完成: {best_timing}")
        return best_timing
    
    def _genetic_algorithm_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """遗传算法优化"""
        logger.info("使用遗传算法优化信号阈值...")
        
        def objective_function(params):
            # 解包参数
            threshold = params[0]
            confidence_threshold = params[1]
            
            # 生成信号
            signals = self._generate_signals_from_features(X, threshold, confidence_threshold)
            
            # 计算性能指标
            if len(np.unique(signals)) > 1:
                accuracy = accuracy_score(y, signals)
                precision = precision_score(y, signals, average='weighted', zero_division=0)
                recall = recall_score(y, signals, average='weighted', zero_division=0)
                f1 = f1_score(y, signals, average='weighted', zero_division=0)
                
                # 综合评分
                score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
            else:
                score = 0.0
            
            return -score  # 最小化负分数
        
        # 参数边界
        bounds = [
            (0.0001, 0.01),  # threshold
            (0.1, 0.9)        # confidence_threshold
        ]
        
        # 遗传算法优化
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        best_params = {
            'threshold': result.x[0],
            'confidence_threshold': result.x[1],
            'score': -result.fun
        }
        
        return best_params
    
    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """网格搜索优化"""
        logger.info("使用网格搜索优化信号阈值...")
        
        # 参数网格
        thresholds = np.linspace(0.0001, 0.01, 20)
        confidence_thresholds = np.linspace(0.1, 0.9, 10)
        
        best_score = 0.0
        best_params = {}
        
        for threshold in thresholds:
            for confidence_threshold in confidence_thresholds:
                # 生成信号
                signals = self._generate_signals_from_features(X, threshold, confidence_threshold)
                
                # 计算性能指标
                if len(np.unique(signals)) > 1:
                    accuracy = accuracy_score(y, signals)
                    precision = precision_score(y, signals, average='weighted', zero_division=0)
                    recall = recall_score(y, signals, average='weighted', zero_division=0)
                    f1 = f1_score(y, signals, average='weighted', zero_division=0)
                    
                    # 综合评分
                    score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'threshold': threshold,
                            'confidence_threshold': confidence_threshold,
                            'score': score
                        }
        
        return best_params
    
    def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """贝叶斯优化"""
        logger.info("使用贝叶斯优化信号阈值...")
        
        def objective_function(params):
            threshold = params[0]
            confidence_threshold = params[1]
            
            # 生成信号
            signals = self._generate_signals_from_features(X, threshold, confidence_threshold)
            
            # 计算性能指标
            if len(np.unique(signals)) > 1:
                accuracy = accuracy_score(y, signals)
                precision = precision_score(y, signals, average='weighted', zero_division=0)
                recall = recall_score(y, signals, average='weighted', zero_division=0)
                f1 = f1_score(y, signals, average='weighted', zero_division=0)
                
                # 综合评分
                score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
            else:
                score = 0.0
            
            return score
        
        # 使用scipy.optimize.minimize进行优化
        from scipy.optimize import minimize
        
        result = minimize(
            lambda x: -objective_function(x),
            x0=[0.001, 0.5],
            bounds=[(0.0001, 0.01), (0.1, 0.9)],
            method='L-BFGS-B'
        )
        
        best_params = {
            'threshold': result.x[0],
            'confidence_threshold': result.x[1],
            'score': -result.fun
        }
        
        return best_params
    
    def _gradient_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """梯度优化"""
        logger.info("使用梯度优化信号阈值...")
        
        def objective_function(params):
            threshold = params[0]
            confidence_threshold = params[1]
            
            # 生成信号
            signals = self._generate_signals_from_features(X, threshold, confidence_threshold)
            
            # 计算性能指标
            if len(np.unique(signals)) > 1:
                accuracy = accuracy_score(y, signals)
                precision = precision_score(y, signals, average='weighted', zero_division=0)
                recall = recall_score(y, signals, average='weighted', zero_division=0)
                f1 = f1_score(y, signals, average='weighted', zero_division=0)
                
                # 综合评分
                score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
            else:
                score = 0.0
            
            return -score
        
        # 使用scipy.optimize.minimize进行优化
        from scipy.optimize import minimize
        
        result = minimize(
            objective_function,
            x0=[0.001, 0.5],
            bounds=[(0.0001, 0.01), (0.1, 0.9)],
            method='L-BFGS-B'
        )
        
        best_params = {
            'threshold': result.x[0],
            'confidence_threshold': result.x[1],
            'score': -result.fun
        }
        
        return best_params
    
    def _generate_signals_from_features(self, X: np.ndarray, threshold: float, confidence_threshold: float) -> np.ndarray:
        """从特征生成信号"""
        # 简化的信号生成逻辑
        # 基于特征的平均值和标准差生成信号
        
        signals = np.zeros(len(X))
        
        for i in range(len(X)):
            # 计算特征统计
            feature_mean = np.mean(X[i])
            feature_std = np.std(X[i])
            feature_confidence = min(1.0, feature_std / (feature_mean + 1e-8))
            
            # 生成信号
            if feature_confidence > confidence_threshold:
                if feature_mean > threshold:
                    signals[i] = 1
                elif feature_mean < -threshold:
                    signals[i] = -1
                else:
                    signals[i] = 0
            else:
                signals[i] = 0
        
        return signals
    
    def _optimize_signal_weights(self, signals: np.ndarray, y: np.ndarray) -> Dict:
        """优化信号权重"""
        logger.info("优化信号权重...")
        
        def objective_function(weights):
            # 归一化权重
            weights = weights / np.sum(weights)
            
            # 加权信号
            weighted_signals = np.dot(signals, weights)
            
            # 生成最终信号
            final_signals = np.where(weighted_signals > 0.5, 1, 
                                   np.where(weighted_signals < -0.5, -1, 0))
            
            # 计算性能指标
            if len(np.unique(final_signals)) > 1:
                accuracy = accuracy_score(y, final_signals)
                precision = precision_score(y, final_signals, average='weighted', zero_division=0)
                recall = recall_score(y, final_signals, average='weighted', zero_division=0)
                f1 = f1_score(y, final_signals, average='weighted', zero_division=0)
                
                # 综合评分
                score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
            else:
                score = 0.0
            
            return -score
        
        # 优化权重
        from scipy.optimize import minimize
        
        n_signals = signals.shape[1]
        result = minimize(
            objective_function,
            x0=np.ones(n_signals) / n_signals,
            bounds=[(0, 1) for _ in range(n_signals)],
            method='L-BFGS-B'
        )
        
        best_weights = result.x / np.sum(result.x)
        
        return {
            'weights': best_weights.tolist(),
            'score': -result.fun
        }
    
    def _optimize_signal_timing(self, signals: np.ndarray, y: np.ndarray) -> Dict:
        """优化信号时机"""
        logger.info("优化信号时机...")
        
        # 测试不同的延迟
        delays = range(0, 6)  # 0-5个时间步延迟
        
        best_score = 0.0
        best_delay = 0
        
        for delay in delays:
            if delay == 0:
                delayed_signals = signals
            else:
                delayed_signals = np.roll(signals, delay)
                delayed_signals[:delay] = 0  # 前面的信号设为0
            
            # 计算性能指标
            if len(np.unique(delayed_signals)) > 1:
                accuracy = accuracy_score(y, delayed_signals)
                precision = precision_score(y, delayed_signals, average='weighted', zero_division=0)
                recall = recall_score(y, delayed_signals, average='weighted', zero_division=0)
                f1 = f1_score(y, delayed_signals, average='weighted', zero_division=0)
                
                # 综合评分
                score = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
                
                if score > best_score:
                    best_score = score
                    best_delay = delay
        
        return {
            'delay': best_delay,
            'score': best_score
        }
    
    def apply_optimized_signals(self, df: pd.DataFrame, signal_column: str = 'ml_signal') -> pd.DataFrame:
        """应用优化后的信号"""
        logger.info("应用优化后的信号...")
        
        df_optimized = df.copy()
        
        # 应用阈值优化
        if 'thresholds' in self.best_parameters:
            threshold = self.best_parameters['thresholds']['threshold']
            confidence_threshold = self.best_parameters['thresholds']['confidence_threshold']
            
            # 重新生成信号
            if signal_column in df_optimized.columns:
                # 基于优化参数重新生成信号
                signals = df_optimized[signal_column].values
                confidence = np.abs(signals)
                
                # 应用阈值过滤
                optimized_signals = np.where(
                    confidence > confidence_threshold,
                    np.where(signals > threshold, 1, np.where(signals < -threshold, -1, 0)),
                    0
                )
                
                df_optimized[f'{signal_column}_optimized'] = optimized_signals
        
        # 应用权重优化
        if 'weights' in self.best_parameters:
            weights = np.array(self.best_parameters['weights']['weights'])
            
            # 找到信号列
            signal_columns = [col for col in df_optimized.columns if 'signal' in col.lower()]
            
            if len(signal_columns) > 0:
                # 加权信号
                weighted_signals = np.zeros(len(df_optimized))
                for i, col in enumerate(signal_columns[:len(weights)]):
                    weighted_signals += df_optimized[col].values * weights[i]
                
                df_optimized['weighted_signal'] = weighted_signals
        
        # 应用时机优化
        if 'timing' in self.best_parameters:
            delay = self.best_parameters['timing']['delay']
            
            if delay > 0:
                # 延迟信号
                if 'weighted_signal' in df_optimized.columns:
                    df_optimized['timed_signal'] = df_optimized['weighted_signal'].shift(delay)
                else:
                    df_optimized['timed_signal'] = df_optimized[signal_column].shift(delay)
        
        logger.info("优化后的信号应用完成")
        return df_optimized
    
    def evaluate_optimization(self, df: pd.DataFrame, target_column: str = 'future_return_1') -> Dict:
        """评估优化效果"""
        logger.info("评估优化效果...")
        
        # 原始信号性能
        if 'ml_signal' in df.columns:
            original_signals = df['ml_signal'].values
            targets = df[target_column].values
            y_labels = np.where(targets > 0.001, 1, np.where(targets < -0.001, -1, 0))
            
            if len(np.unique(original_signals)) > 1:
                original_accuracy = accuracy_score(y_labels, original_signals)
                original_precision = precision_score(y_labels, original_signals, average='weighted', zero_division=0)
                original_recall = recall_score(y_labels, original_signals, average='weighted', zero_division=0)
                original_f1 = f1_score(y_labels, original_signals, average='weighted', zero_division=0)
            else:
                original_accuracy = original_precision = original_recall = original_f1 = 0.0
        
        # 优化后信号性能
        if 'timed_signal' in df.columns:
            optimized_signals = df['timed_signal'].values
        elif 'weighted_signal' in df.columns:
            optimized_signals = df['weighted_signal'].values
        elif 'ml_signal_optimized' in df.columns:
            optimized_signals = df['ml_signal_optimized'].values
        else:
            optimized_signals = original_signals
        
        if len(np.unique(optimized_signals)) > 1:
            optimized_accuracy = accuracy_score(y_labels, optimized_signals)
            optimized_precision = precision_score(y_labels, optimized_signals, average='weighted', zero_division=0)
            optimized_recall = recall_score(y_labels, optimized_signals, average='weighted', zero_division=0)
            optimized_f1 = f1_score(y_labels, optimized_signals, average='weighted', zero_division=0)
        else:
            optimized_accuracy = optimized_precision = optimized_recall = optimized_f1 = 0.0
        
        # 计算改进
        accuracy_improvement = optimized_accuracy - original_accuracy
        precision_improvement = optimized_precision - original_precision
        recall_improvement = optimized_recall - original_recall
        f1_improvement = optimized_f1 - original_f1
        
        evaluation = {
            'original': {
                'accuracy': original_accuracy,
                'precision': original_precision,
                'recall': original_recall,
                'f1': original_f1
            },
            'optimized': {
                'accuracy': optimized_accuracy,
                'precision': optimized_precision,
                'recall': optimized_recall,
                'f1': optimized_f1
            },
            'improvement': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement
            }
        }
        
        logger.info("优化效果评估完成")
        logger.info(f"准确率改进: {accuracy_improvement:.4f}")
        logger.info(f"精确率改进: {precision_improvement:.4f}")
        logger.info(f"召回率改进: {recall_improvement:.4f}")
        logger.info(f"F1分数改进: {f1_improvement:.4f}")
        
        return evaluation

def main():
    """主函数 - 演示V11信号优化"""
    logger.info("=" * 60)
    logger.info("V11信号生成优化演示")
    logger.info("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'ml_signal': np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3]),
        'future_return_1': np.random.randn(n_samples) * 0.01
    })
    
    logger.info(f"示例数据创建完成: {len(df)} 条记录")
    
    # 创建信号优化器
    optimizer = V11SignalOptimizer()
    
    # 优化信号阈值
    logger.info("优化信号阈值...")
    threshold_params = optimizer.optimize_signal_thresholds(df, optimization_method='grid_search')
    
    # 优化信号权重
    logger.info("优化信号权重...")
    weight_params = optimizer.optimize_signal_weights(df)
    
    # 优化信号时机
    logger.info("优化信号时机...")
    timing_params = optimizer.optimize_signal_timing(df)
    
    # 应用优化后的信号
    logger.info("应用优化后的信号...")
    df_optimized = optimizer.apply_optimized_signals(df)
    
    # 评估优化效果
    logger.info("评估优化效果...")
    evaluation = optimizer.evaluate_optimization(df_optimized)
    
    logger.info("=" * 60)
    logger.info("V11信号生成优化演示完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
