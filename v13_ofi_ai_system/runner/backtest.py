#!/usr/bin/env python3
"""
Task 1.5 核心算法v1 - 回测框架
完成Parquet数据的事件重放、交易成本模型、评估指标与网格扫描
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# 添加src路径以导入成熟组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入统一配置系统
from utils.config_loader import load_config, get_config

@dataclass
class BacktestConfig:
    """回测配置"""
    # 数据配置
    data_dir: str = "data/ofi_cvd"
    warehouse_dir: str = "data/warehouse"
    
    # 时间配置
    start_date: str = "2025-10-21"
    end_date: str = "2025-10-22"
    
    # 标签配置
    horizons: List[int] = None  # [15, 30, 60] 秒
    
    # 评估配置
    min_samples: int = 1000
    confidence_level: float = 0.95
    
    # 成本模型
    spread_bps: float = 5.0
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 2.0
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [15, 30, 60]

@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    horizon: int
    auc: float
    pr_auc: float
    ic: float
    ic_pvalue: float
    hit_rate: float
    total_samples: int
    positive_samples: int
    negative_samples: int
    avg_return: float
    std_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

class BacktestFramework:
    """回测框架"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 加载统一配置
        self.system_config = load_config()
        
    def load_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """加载实时数据收集器的真实数据"""
        data = {}
        data_dir = Path(self.config.data_dir)
        
        # 查找最新的数据文件
        date_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('date=')])
        if not date_dirs:
            self.logger.warning(f"未找到数据目录 {data_dir}")
            return {}
        
        # 使用最新的数据目录
        latest_date_dir = date_dirs[-1]
        symbol_dir = latest_date_dir / f"symbol={symbol}"
        
        if not symbol_dir.exists():
            self.logger.warning(f"未找到 {symbol} 的数据目录 {symbol_dir}")
            return {}
        
        # 加载价格数据
        prices_path = symbol_dir / "kind=prices"
        if prices_path.exists():
            price_files = list(prices_path.glob("*.parquet"))
            if price_files:
                try:
                    prices_df = pd.concat([pd.read_parquet(f) for f in price_files])
                    data['prices'] = prices_df.sort_values('ts_ms')
                    self.logger.info(f"加载价格数据: {len(prices_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载价格数据失败: {e}")
        
        # 加载OFI数据
        ofi_path = symbol_dir / "kind=ofi"
        if ofi_path.exists():
            ofi_files = list(ofi_path.glob("*.parquet"))
            if ofi_files:
                try:
                    ofi_df = pd.concat([pd.read_parquet(f) for f in ofi_files])
                    data['ofi'] = ofi_df.sort_values('ts_ms')
                    self.logger.info(f"加载OFI数据: {len(ofi_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载OFI数据失败: {e}")
        
        # 加载CVD数据
        cvd_path = symbol_dir / "kind=cvd"
        if cvd_path.exists():
            cvd_files = list(cvd_path.glob("*.parquet"))
            if cvd_files:
                try:
                    cvd_df = pd.concat([pd.read_parquet(f) for f in cvd_files])
                    data['cvd'] = cvd_df.sort_values('ts_ms')
                    self.logger.info(f"加载CVD数据: {len(cvd_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载CVD数据失败: {e}")
        
        # 加载Fusion数据
        fusion_path = symbol_dir / "kind=fusion"
        if fusion_path.exists():
            fusion_files = list(fusion_path.glob("*.parquet"))
            if fusion_files:
                try:
                    fusion_df = pd.concat([pd.read_parquet(f) for f in fusion_files])
                    data['fusion'] = fusion_df.sort_values('ts_ms')
                    self.logger.info(f"加载Fusion数据: {len(fusion_df)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载Fusion数据失败: {e}")
        
        return data
    
    def calculate_forward_returns(self, prices_df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """计算前瞻收益 - 基于时间对齐而非行偏移"""
        returns_df = prices_df[['ts_ms', 'price']].copy()
        
        for horizon in horizons:
            # 计算未来时间戳（毫秒）
            future_ts = prices_df['ts_ms'] + horizon * 1000
            
            # 使用merge_asof找到未来价格
            future_prices_df = pd.DataFrame({
                'ts_ms': future_ts,
                'future_price': prices_df['price']
            })
            
            # 合并当前价格和未来价格
            merged_df = pd.merge_asof(
                prices_df[['ts_ms', 'price']].sort_values('ts_ms'),
                future_prices_df.sort_values('ts_ms'),
                on='ts_ms',
                direction='forward',
                tolerance=horizon * 1000 + 5000  # 允许5秒容差
            )
            
            # 计算收益率
            returns = (merged_df['future_price'] / merged_df['price'] - 1) * 100
            returns_df[f'return_{horizon}s'] = returns
            
            # 计算二分类标签（正收益为1，负收益为0）
            returns_df[f'label_{horizon}s'] = (returns > 0).astype(int)
        
        return returns_df
    
    def merge_signals_with_labels(self, signals_df: pd.DataFrame, labels_df: pd.DataFrame, 
                                tolerance_ms: int = 1500) -> pd.DataFrame:
        """合并信号与标签"""
        # 使用merge_asof进行时间对齐
        merged_df = pd.merge_asof(
            signals_df.sort_values('ts_ms'),
            labels_df.sort_values('ts_ms'),
            on='ts_ms',
            direction='backward',
            tolerance=tolerance_ms
        )
        
        return merged_df
    
    def calculate_metrics(self, y_true: np.ndarray, y_score: np.ndarray, 
                        returns: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # 移除NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_score) | np.isnan(returns))
        y_true = y_true[mask]
        y_score = y_score[mask]
        returns = returns[mask]
        
        if len(y_true) < 10:
            return {}
        
        # AUC
        try:
            auc_score = roc_auc_score(y_true, y_score)
        except:
            auc_score = 0.5
        
        # PR-AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.5
        
        # IC (Information Coefficient)
        try:
            ic, ic_pvalue = stats.spearmanr(y_score, returns)
        except:
            ic, ic_pvalue = 0.0, 1.0
        
        # 命中率
        hit_rate = np.mean(y_true)
        
        # 收益统计
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns / 100)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar比率
        calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        return {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'hit_rate': hit_rate,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def run_backtest(self, symbol: str) -> List[BacktestResult]:
        """运行回测"""
        self.logger.info(f"开始回测 {symbol}")
        
        # 加载数据
        data = self.load_data(symbol)
        if not data:
            self.logger.warning(f"未找到 {symbol} 的数据")
            return []
        
        # 计算前瞻收益
        prices_df = data['prices']
        labels_df = self.calculate_forward_returns(prices_df, self.config.horizons)
        
        results = []
        
        # 对每个时间窗口进行回测
        for horizon in self.config.horizons:
            self.logger.info(f"回测 {symbol} - {horizon}秒窗口")
            
            # 准备信号数据
            if 'fusion' in data:
                signals_df = data['fusion'][['ts_ms', 'score']].copy()
            else:
                # 如果没有fusion数据，使用OFI和CVD组合
                ofi_df = data.get('ofi', pd.DataFrame())
                cvd_df = data.get('cvd', pd.DataFrame())
                
                if not ofi_df.empty and not cvd_df.empty:
                    # 合并OFI和CVD
                    merged_df = pd.merge_asof(
                        ofi_df[['ts_ms', 'z_ofi']].sort_values('ts_ms'),
                        cvd_df[['ts_ms', 'z_cvd']].sort_values('ts_ms'),
                        on='ts_ms',
                        direction='backward',
                        tolerance=1000
                    )
                    merged_df['score'] = 0.6 * merged_df['z_ofi'] + 0.4 * merged_df['z_cvd']
                    signals_df = merged_df[['ts_ms', 'score']].copy()
                else:
                    self.logger.warning(f"{symbol} 缺少信号数据")
                    continue
            
            # 合并信号与标签
            merged_df = self.merge_signals_with_labels(
                signals_df, 
                labels_df[['ts_ms', f'label_{horizon}s', f'return_{horizon}s']].rename(
                    columns={f'label_{horizon}s': 'label', f'return_{horizon}s': 'return'}
                )
            )
            
            # 移除NaN值
            merged_df = merged_df.dropna(subset=['score', 'label', 'return'])
            
            if len(merged_df) < self.config.min_samples:
                self.logger.warning(f"{symbol} - {horizon}秒窗口样本不足: {len(merged_df)}")
                continue
            
            # 计算成本调整后的收益
            cost_bps = self.config.spread_bps + self.config.slippage_bps + self.config.commission_rate * 10000
            trade_flag = abs(merged_df['score']) > self.config.z_mid
            net_returns = merged_df['return'] - (cost_bps * trade_flag)
            
            # 计算指标（使用成本调整后的收益）
            metrics = self.calculate_metrics(
                merged_df['label'].values,
                merged_df['score'].values,
                net_returns.values
            )
            
            if not metrics:
                continue
            
            # 创建结果
            result = BacktestResult(
                symbol=symbol,
                horizon=horizon,
                auc=metrics['auc'],
                pr_auc=metrics['pr_auc'],
                ic=metrics['ic'],
                ic_pvalue=metrics['ic_pvalue'],
                hit_rate=metrics['hit_rate'],
                total_samples=len(merged_df),
                positive_samples=int(np.sum(merged_df['label'])),
                negative_samples=int(len(merged_df) - np.sum(merged_df['label'])),
                avg_return=metrics['avg_return'],
                std_return=metrics['std_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                calmar_ratio=metrics['calmar_ratio']
            )
            
            results.append(result)
            
            self.logger.info(f"{symbol} - {horizon}秒: AUC={metrics['auc']:.3f}, IC={metrics['ic']:.3f}")
        
        return results
    
    def run_grid_search(self, symbols: List[str], param_grid: Dict) -> Dict:
        """运行网格搜索"""
        self.logger.info("开始网格搜索")
        
        best_score = -np.inf
        best_params = {}
        
        # 实现真正的网格搜索
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            # 更新配置
            for key, value in params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # 运行回测
            try:
                results_list = self.run_backtest(symbols[0])
                if results_list:
                    avg_auc = np.mean([r.metrics.get('auc', 0.5) for r in results_list])
                    avg_ic = np.mean([r.metrics.get('ic', 0.0) for r in results_list])
                    score = avg_auc + abs(avg_ic) * 0.1  # 综合评分
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
            except Exception as e:
                self.logger.warning(f"参数组合 {params} 失败: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }
    
    def save_results(self, results: List[BacktestResult], output_dir: str):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_df = pd.DataFrame([{
            'symbol': r.symbol,
            'horizon': r.horizon,
            'auc': r.auc,
            'pr_auc': r.pr_auc,
            'ic': r.ic,
            'ic_pvalue': r.ic_pvalue,
            'hit_rate': r.hit_rate,
            'total_samples': r.total_samples,
            'avg_return': r.avg_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'calmar_ratio': r.calmar_ratio
        } for r in results])
        
        results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        
        # 保存汇总报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'symbols': list(set(r.symbol for r in results)),
            'horizons': list(set(r.horizon for r in results)),
            'avg_auc': np.mean([r.auc for r in results]),
            'avg_ic': np.mean([r.ic for r in results]),
            'best_auc': max([r.auc for r in results]) if results else 0.0,
            'best_ic': max([r.ic for r in results]) if results else 0.0
        }
        
        summary_file = output_path / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"回测结果已保存: {results_file}")
        self.logger.info(f"汇总报告已保存: {summary_file}")

def main():
    """测试主函数"""
    # 配置
    config = BacktestConfig(
        data_dir="data/ofi_cvd",
        start_date="2025-10-21",
        end_date="2025-10-22",
        horizons=[15, 30, 60]
    )
    
    # 创建回测框架
    framework = BacktestFramework(config)
    
    # 运行回测
    symbols = ["BTCUSDT", "ETHUSDT"]
    all_results = []
    
    for symbol in symbols:
        results = framework.run_backtest(symbol)
        all_results.extend(results)
    
    # 保存结果
    framework.save_results(all_results, "artifacts/core_algo_v1/backtest")
    
    print(f"回测完成，共生成 {len(all_results)} 个结果")

if __name__ == "__main__":
    main()
