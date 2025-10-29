#!/usr/bin/env python3
"""
标签构造和切片分析工具
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class LabelConstructor:
    """前瞻标签构造器"""
    
    def __init__(self, horizons: List[int], price_type: str = "trade"):
        self.horizons = horizons  # 前瞻窗口(秒)
        self.price_type = price_type  # 价格类型: "trade", "mid", "micro"
    
    def construct_labels(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """构造多窗口前瞻标签（基于时间戳asof对齐，支持多种价格类型）"""
        if prices_df.empty:
            return pd.DataFrame()
        
        # 根据价格类型选择价格列
        if self.price_type == "mid" and 'best_bid' in prices_df.columns and 'best_ask' in prices_df.columns:
            # 中间价标签
            prices_df['price'] = (prices_df['best_bid'] + prices_df['best_ask']) / 2
            print("    使用中间价标签: (best_bid + best_ask) / 2")
        elif self.price_type == "micro" and 'best_bid' in prices_df.columns and 'best_ask' in prices_df.columns:
            # 微价格标签（成交量加权）
            if 'best_bid_qty' in prices_df.columns and 'best_ask_qty' in prices_df.columns:
                # 真正的微价格 = (best_bid_qty * best_ask + best_ask_qty * best_bid) / (best_bid_qty + best_ask_qty)
                prices_df['price'] = (
                    prices_df['best_bid_qty'] * prices_df['best_ask'] + 
                    prices_df['best_ask_qty'] * prices_df['best_bid']
                ) / (prices_df['best_bid_qty'] + prices_df['best_ask_qty'])
                print("    使用微价格标签: 真实成交量加权")
            else:
                # 缺少队列量数据，回退到中间价
                prices_df['price'] = (prices_df['best_bid'] + prices_df['best_ask']) / 2
                print("    微价格权重缺失，已回退中间价")
        elif 'price' not in prices_df.columns:
            print("    错误: 未找到价格列")
            return pd.DataFrame()
        else:
            print("    使用成交价标签")
        
        # 确保数据按时间排序
        if 'ts_ms' in prices_df.columns:
            prices_df = prices_df.sort_values('ts_ms').reset_index(drop=True)
        
        # 为每个前瞻窗口构造标签
        for horizon in self.horizons:
            horizon_ms = horizon * 1000  # 转换为毫秒
            
            # 创建前瞻时间戳
            prices_df[f'ts_ms_fwd_{horizon}s'] = prices_df['ts_ms'] + horizon_ms
            
            # 使用merge_asof进行时间对齐（修复关键问题）
            # 左表：当前价格，右表：前瞻价格
            current_prices = prices_df[['ts_ms', 'price']].copy()
            future_prices = prices_df[['ts_ms', 'price']].copy()
            future_prices['ts_ms'] = future_prices['ts_ms'] + horizon_ms
            future_prices = future_prices.rename(columns={'price': 'price_fwd'})
            
            # 使用merge_asof进行前瞻对齐
            merged = pd.merge_asof(
                current_prices,
                future_prices,
                on='ts_ms',
                direction='forward',  # ✅ 改为 forward，确保拿 t+h
                tolerance=horizon_ms * 2,  # 允许2倍时间窗口的容差
                suffixes=('_curr', '_fwd')
            )
            
            # 计算前瞻收益率
            valid_mask = merged['price_fwd'].notna() & merged['price'].notna()
            if valid_mask.sum() > 0:
                forward_returns = (merged['price_fwd'] - merged['price']) / merged['price']
                
                # 分类标签 (1: 上涨, 0: 下跌)
                labels = (forward_returns > 0).astype(int)
                
                # 存储标签（对齐到原始数据）
                prices_df[f'label_{horizon}s'] = np.nan
                prices_df[f'return_{horizon}s'] = np.nan
                
                # 只对有效数据赋值
                valid_indices = merged.index[valid_mask]
                prices_df.loc[valid_indices, f'label_{horizon}s'] = labels[valid_mask]
                prices_df.loc[valid_indices, f'return_{horizon}s'] = forward_returns[valid_mask]
                
                print(f"    {horizon}s标签: {valid_mask.sum()}/{len(prices_df)} 有效 ({valid_mask.sum()/len(prices_df):.1%})")
            else:
                # 如果没有有效数据，填充NaN
                prices_df[f'label_{horizon}s'] = np.nan
                prices_df[f'return_{horizon}s'] = np.nan
                print(f"    {horizon}s标签: 无有效数据")
        
        # 移除无法计算前瞻标签的行
        prices_df = prices_df.dropna(subset=[f'label_{h}s' for h in self.horizons], how='all')
        
        return prices_df
    
    def validate_labels(self, labeled_df: pd.DataFrame) -> Dict[str, float]:
        """验证标签质量"""
        validation_results = {}
        
        for horizon in self.horizons:
            label_col = f'label_{horizon}s'
            return_col = f'return_{horizon}s'
            
            if label_col in labeled_df.columns:
                # 计算标签分布
                label_dist = labeled_df[label_col].value_counts(normalize=True)
                validation_results[f'{horizon}s_label_dist'] = label_dist.to_dict()
                
                # 计算收益率统计
                if return_col in labeled_df.columns:
                    returns = labeled_df[return_col].dropna()
                    validation_results[f'{horizon}s_return_stats'] = {
                        'mean': returns.mean(),
                        'std': returns.std(),
                        'min': returns.min(),
                        'max': returns.max()
                    }
        
        return validation_results


class SliceAnalyzer:
    """切片分析器"""
    
    def __init__(self, slice_config: Dict[str, List[str]]):
        self.slice_config = slice_config
    
    def create_regime_slices(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建市场活跃度切片"""
        if 'regime' not in df.columns:
            # 基于波动率创建活跃度切片
            if 'price' in df.columns:
                returns = df['price'].pct_change().dropna()
                volatility = returns.rolling(window=60).std()  # 1分钟滚动波动率
                
                # 分位数切分
                vol_quantiles = volatility.quantile([0.33, 0.67])
                
                df['regime'] = 'Quiet'
                df.loc[volatility > vol_quantiles[0.67], 'regime'] = 'Active'
                df.loc[(volatility > vol_quantiles[0.33]) & 
                       (volatility <= vol_quantiles[0.67]), 'regime'] = 'Medium'
        
        return df
    
    def create_time_slices(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时段切片"""
        if 'ts_ms' in df.columns:
            # 转换为UTC时间
            df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
            
            # 提取小时
            df['hour_utc'] = df['datetime'].dt.hour
            
            # 定义时段
            df['tod'] = 'Other'
            df.loc[(df['hour_utc'] >= 0) & (df['hour_utc'] < 8), 'tod'] = 'Tokyo'
            df.loc[(df['hour_utc'] >= 8) & (df['hour_utc'] < 16), 'tod'] = 'London'
            df.loc[(df['hour_utc'] >= 16) & (df['hour_utc'] < 24), 'tod'] = 'NY'
        
        return df
    
    def create_volatility_slices(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建波动率切片"""
        if 'price' in df.columns:
            returns = df['price'].pct_change().dropna()
            volatility = returns.rolling(window=60).std()
            
            # 分位数切分
            vol_quantiles = volatility.quantile([0.33, 0.67])
            
            df['vol_regime'] = 'low'
            df.loc[volatility > vol_quantiles[0.67], 'vol_regime'] = 'high'
            df.loc[(volatility > vol_quantiles[0.33]) & 
                   (volatility <= vol_quantiles[0.67]), 'vol_regime'] = 'mid'
        
        return df
    
    def analyze_slices(self, df: pd.DataFrame, metrics: Dict) -> Dict:
        """分析各切片的指标（输出完整性能指标）"""
        slice_results = {}
        
        # 活跃度切片分析
        if 'regime' in df.columns:
            regime_metrics = self._analyze_slice_performance(df, 'regime', metrics)
            slice_results['regime'] = regime_metrics
        
        # 时段切片分析
        if 'tod' in df.columns:
            tod_metrics = self._analyze_slice_performance(df, 'tod', metrics)
            slice_results['tod'] = tod_metrics
        
        # 波动率切片分析
        if 'vol_regime' in df.columns:
            vol_metrics = self._analyze_slice_performance(df, 'vol_regime', metrics)
            slice_results['volatility'] = vol_metrics
        
        return slice_results
    
    def _analyze_slice_performance(self, df: pd.DataFrame, slice_col: str, metrics: Dict) -> Dict:
        """分析单个切片的完整性能指标"""
        slice_metrics = {}
        
        for slice_value in df[slice_col].unique():
            if pd.isna(slice_value):
                continue
            
            slice_data = df[df[slice_col] == slice_value]
            
            # 计算切片统计
            slice_stats = {
                'count': len(slice_data),
                'percentage': len(slice_data) / len(df) * 100
            }
            
            # 计算切片内的性能指标（AUC/PR-AUC/IC等）
            for horizon in [60, 180, 300, 900]:
                label_col = f'label_{horizon}s'
                if label_col in slice_data.columns:
                    labels = slice_data[label_col].dropna()
                    if len(labels) > 0:
                        slice_stats[f'{horizon}s_positive_rate'] = labels.mean()
                        
                        # 如果有信号数据，计算性能指标
                        for signal_type in ['ofi', 'cvd', 'fusion']:
                            signal_col = f'{signal_type}_z' if signal_type != 'fusion' else 'score'
                            if signal_col in slice_data.columns:
                                signal_data = slice_data[signal_col].dropna()
                                if len(signal_data) > 10 and len(labels) > 10:
                                    # 计算AUC等指标
                                    try:
                                        from sklearn.metrics import roc_auc_score
                                        auc = roc_auc_score(labels, signal_data)
                                        slice_stats[f'{horizon}s_{signal_type}_auc'] = auc
                                    except:
                                        slice_stats[f'{horizon}s_{signal_type}_auc'] = np.nan
            
            slice_metrics[slice_value] = slice_stats
        
        return slice_metrics
    
    def _analyze_slice(self, df: pd.DataFrame, slice_col: str, metrics: Dict) -> Dict:
        """分析单个切片的指标"""
        slice_metrics = {}
        
        for slice_value in df[slice_col].unique():
            if pd.isna(slice_value):
                continue
            
            slice_data = df[df[slice_col] == slice_value]
            
            # 计算切片统计
            slice_stats = {
                'count': len(slice_data),
                'percentage': len(slice_data) / len(df) * 100
            }
            
            # 如果有标签数据，计算切片内的指标
            for horizon in [60, 180, 300, 900]:
                label_col = f'label_{horizon}s'
                if label_col in slice_data.columns:
                    labels = slice_data[label_col].dropna()
                    if len(labels) > 0:
                        slice_stats[f'{horizon}s_positive_rate'] = labels.mean()
            
            slice_metrics[slice_value] = slice_stats
        
        return slice_metrics


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, data_type: str) -> Dict[str, any]:
        """检查数据质量"""
        quality_report = {
            'data_type': data_type,
            'total_rows': len(df),
            'missing_data': {},
            'data_quality_score': 0.0
        }
        
        if df.empty:
            quality_report['data_quality_score'] = 0.0
            return quality_report
        
        # 检查缺失值
        missing_data = df.isnull().sum()
        quality_report['missing_data'] = missing_data.to_dict()
        
        # 计算数据质量分数
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_data.sum()
        quality_score = 1.0 - (missing_cells / total_cells)
        quality_report['data_quality_score'] = quality_score
        
        return quality_report
    
    @staticmethod
    def check_temporal_consistency(df: pd.DataFrame) -> Dict[str, any]:
        """检查时间一致性"""
        if 'ts_ms' not in df.columns:
            return {'temporal_consistency': 'no_timestamp'}
        
        # 检查时间戳顺序
        timestamps = df['ts_ms'].dropna()
        is_sorted = timestamps.is_monotonic_increasing
        
        # 检查时间间隔
        time_diffs = timestamps.diff().dropna()
        time_stats = {
            'mean_interval_ms': time_diffs.mean(),
            'median_interval_ms': time_diffs.median(),
            'std_interval_ms': time_diffs.std()
        }
        
        return {
            'temporal_consistency': is_sorted,
            'time_stats': time_stats,
            'gaps_detected': len(time_diffs[time_diffs > time_diffs.quantile(0.95)])
        }
    
    @staticmethod
    def check_signal_quality(signal_df: pd.DataFrame, signal_type: str) -> Dict[str, any]:
        """检查信号质量"""
        quality_report = {
            'signal_type': signal_type,
            'total_signals': len(signal_df),
            'valid_signals': 0,
            'signal_quality_score': 0.0
        }
        
        if signal_df.empty:
            return quality_report
        
        # 根据信号类型检查质量
        if signal_type == 'ofi' and 'ofi_z' in signal_df.columns:
            valid_signals = signal_df['ofi_z'].notna().sum()
            quality_report['valid_signals'] = valid_signals
            quality_report['signal_quality_score'] = valid_signals / len(signal_df)
            
        elif signal_type == 'cvd' and 'z_cvd' in signal_df.columns:
            valid_signals = signal_df['z_cvd'].notna().sum()
            quality_report['valid_signals'] = valid_signals
            quality_report['signal_quality_score'] = valid_signals / len(signal_df)
            
        elif signal_type == 'fusion' and 'score' in signal_df.columns:
            valid_signals = signal_df['score'].notna().sum()
            quality_report['valid_signals'] = valid_signals
            quality_report['signal_quality_score'] = valid_signals / len(signal_df)
        
        return quality_report
