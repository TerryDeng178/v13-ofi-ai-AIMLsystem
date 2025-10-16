"""
V12策略优化系统
实现策略分桶与门控机制
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v12_realistic_data_simulator import V12RealisticDataSimulator
from src.v12_strict_validation_framework import V12StrictValidationFramework
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem
from src.v12_high_frequency_execution_engine import V12HighFrequencyExecutionEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V12StrategyOptimizer:
    """V12策略优化器 - 实现策略分桶与门控"""
    
    def __init__(self):
        """初始化策略优化器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 策略分桶配置
        self.strategy_buckets = {
            'high_volatility': {
                'volatility_threshold': 0.02,
                'signal_strength_threshold': 0.6,
                'position_size_multiplier': 1.2,
                'stop_loss_bps': 15,
                'take_profit_bps': 30,
                'enabled': True
            },
            'trending': {
                'trend_strength_threshold': 0.7,
                'signal_strength_threshold': 0.5,
                'position_size_multiplier': 1.0,
                'stop_loss_bps': 20,
                'take_profit_bps': 40,
                'enabled': True
            },
            'ranging': {
                'range_threshold': 0.01,
                'signal_strength_threshold': 0.7,
                'position_size_multiplier': 0.8,
                'stop_loss_bps': 10,
                'take_profit_bps': 20,
                'enabled': True
            },
            'low_volatility': {
                'volatility_threshold': 0.005,
                'signal_strength_threshold': 0.8,
                'position_size_multiplier': 0.6,
                'stop_loss_bps': 8,
                'take_profit_bps': 15,
                'enabled': False  # 默认关闭低波动策略
            }
        }
        
        # 门控配置
        self.gating_config = {
            'max_daily_trades': 50,
            'max_daily_loss_bps': 100,
            'min_signal_quality': 0.5,
            'max_position_size_bps': 200,
            'risk_budget_bps': 500
        }
        
        logger.info(f"V12策略优化器初始化完成 - 设备: {self.device}")
    
    def analyze_market_state(self, data: pd.DataFrame) -> str:
        """分析市场状态"""
        try:
            # 计算市场状态指标
            price_returns = data['close'].pct_change().dropna()
            volatility = price_returns.std()
            
            # 计算趋势强度
            sma_20 = data['close'].rolling(20).mean()
            trend_strength = abs((data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
            
            # 计算价格范围
            price_range = (data['close'].max() - data['close'].min()) / data['close'].mean()
            
            # 判断市场状态
            if volatility > self.strategy_buckets['high_volatility']['volatility_threshold']:
                return 'high_volatility'
            elif trend_strength > self.strategy_buckets['trending']['trend_strength_threshold']:
                return 'trending'
            elif price_range < self.strategy_buckets['ranging']['range_threshold']:
                return 'ranging'
            else:
                return 'low_volatility'
                
        except Exception as e:
            logger.error(f"市场状态分析失败: {e}")
            return 'trending'  # 默认状态
    
    def calculate_signal_quality(self, data: pd.DataFrame, signal_strength: float, 
                               ofi_confidence: float, ai_confidence: float) -> float:
        """计算信号质量评分"""
        try:
            # 基础信号质量
            base_quality = (signal_strength + ofi_confidence + ai_confidence) / 3
            
            # 市场状态调整
            market_state = self.analyze_market_state(data)
            state_multiplier = {
                'high_volatility': 0.9,
                'trending': 1.1,
                'ranging': 1.0,
                'low_volatility': 0.8
            }.get(market_state, 1.0)
            
            # 流动性调整
            liquidity_score = min(data['bid1_size'].iloc[-1] / 1000, 1.0)
            
            # 时间调整（避免开盘收盘时段）
            current_hour = datetime.now().hour
            time_multiplier = 1.0 if 9 <= current_hour <= 15 else 0.8
            
            # 综合评分
            quality_score = base_quality * state_multiplier * liquidity_score * time_multiplier
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"信号质量计算失败: {e}")
            return 0.5
    
    def apply_strategy_bucketing(self, market_state: str, signal_quality: float) -> Dict:
        """应用策略分桶"""
        try:
            bucket_config = self.strategy_buckets.get(market_state, self.strategy_buckets['trending'])
            
            # 检查策略是否启用
            if not bucket_config['enabled']:
                return None
            
            # 检查信号质量阈值
            if signal_quality < bucket_config['signal_strength_threshold']:
                return None
            
            # 返回策略配置
            return {
                'strategy_type': market_state,
                'position_size_multiplier': bucket_config['position_size_multiplier'],
                'stop_loss_bps': bucket_config['stop_loss_bps'],
                'take_profit_bps': bucket_config['take_profit_bps'],
                'signal_quality': signal_quality,
                'enabled': True
            }
            
        except Exception as e:
            logger.error(f"策略分桶失败: {e}")
            return None
    
    def apply_gating_mechanism(self, strategy_config: Dict, current_risk: Dict) -> bool:
        """应用门控机制"""
        try:
            if not strategy_config:
                return False
            
            # 检查日交易次数限制
            if current_risk.get('daily_trades', 0) >= self.gating_config['max_daily_trades']:
                logger.info("达到日交易次数限制，拒绝交易")
                return False
            
            # 检查日损失限制
            if current_risk.get('daily_loss_bps', 0) >= self.gating_config['max_daily_loss_bps']:
                logger.info("达到日损失限制，拒绝交易")
                return False
            
            # 检查信号质量
            if strategy_config['signal_quality'] < self.gating_config['min_signal_quality']:
                logger.info("信号质量不足，拒绝交易")
                return False
            
            # 检查风险预算
            if current_risk.get('risk_budget_bps', 0) <= strategy_config['stop_loss_bps']:
                logger.info("风险预算不足，拒绝交易")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"门控机制失败: {e}")
            return False
    
    def optimize_position_sizing(self, strategy_config: Dict, base_size: float, 
                               volatility: float) -> float:
        """优化仓位大小"""
        try:
            # 基础仓位大小
            position_size = base_size * strategy_config['position_size_multiplier']
            
            # 波动率调整
            volatility_adjustment = max(0.5, min(2.0, 1.0 / volatility))
            position_size *= volatility_adjustment
            
            # 信号质量调整
            quality_adjustment = strategy_config['signal_quality']
            position_size *= quality_adjustment
            
            # 确保在合理范围内
            max_size = self.gating_config['max_position_size_bps'] / 10000
            position_size = min(position_size, max_size)
            
            return max(position_size, 0.01)  # 最小仓位
            
        except Exception as e:
            logger.error(f"仓位优化失败: {e}")
            return base_size
    
    def run_strategy_optimization(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """运行策略优化"""
        try:
            logger.info("开始V12策略优化...")
            
            # 初始化风险状态
            risk_state = {
                'daily_trades': 0,
                'daily_loss_bps': 0,
                'risk_budget_bps': self.gating_config['risk_budget_bps'],
                'current_positions': 0
            }
            
            # 策略执行结果
            optimization_results = {
                'total_signals': len(signals),
                'processed_signals': 0,
                'approved_trades': 0,
                'rejected_trades': 0,
                'strategy_bucket_counts': {bucket: 0 for bucket in self.strategy_buckets.keys()},
                'gating_rejections': {
                    'daily_trades_limit': 0,
                    'daily_loss_limit': 0,
                    'signal_quality': 0,
                    'risk_budget': 0
                },
                'performance_metrics': {
                    'avg_signal_quality': 0,
                    'avg_position_size': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            }
            
            signal_qualities = []
            position_sizes = []
            trade_results = []
            
            for idx, signal in signals.iterrows():
                try:
                    # 获取当前数据切片
                    current_data = data.iloc[max(0, idx-100):idx+1].copy()
                    
                    if len(current_data) < 20:
                        continue
                    
                    # 计算信号质量
                    signal_quality = self.calculate_signal_quality(
                        current_data, 
                        signal.get('signal_strength', 0.5),
                        signal.get('ofi_confidence', 0.5),
                        signal.get('ai_confidence', 0.5)
                    )
                    signal_qualities.append(signal_quality)
                    
                    # 分析市场状态
                    market_state = self.analyze_market_state(current_data)
                    
                    # 应用策略分桶
                    strategy_config = self.apply_strategy_bucketing(market_state, signal_quality)
                    
                    if strategy_config:
                        optimization_results['strategy_bucket_counts'][market_state] += 1
                        
                        # 应用门控机制
                        if self.apply_gating_mechanism(strategy_config, risk_state):
                            # 优化仓位大小
                            volatility = current_data['close'].pct_change().std()
                            position_size = self.optimize_position_sizing(
                                strategy_config, 0.01, volatility
                            )
                            position_sizes.append(position_size)
                            
                            # 模拟交易执行
                            trade_pnl = self.simulate_trade_execution(
                                strategy_config, position_size, current_data
                            )
                            trade_results.append(trade_pnl)
                            
                            # 更新风险状态
                            risk_state['daily_trades'] += 1
                            risk_state['risk_budget_bps'] -= strategy_config['stop_loss_bps']
                            
                            if trade_pnl < 0:
                                risk_state['daily_loss_bps'] += abs(trade_pnl * 10000)
                            
                            optimization_results['approved_trades'] += 1
                        else:
                            optimization_results['rejected_trades'] += 1
                    else:
                        optimization_results['rejected_trades'] += 1
                    
                    optimization_results['processed_signals'] += 1
                    
                except Exception as e:
                    logger.error(f"信号处理失败 {idx}: {e}")
                    continue
            
            # 计算性能指标
            if signal_qualities:
                optimization_results['performance_metrics']['avg_signal_quality'] = np.mean(signal_qualities)
            if position_sizes:
                optimization_results['performance_metrics']['avg_position_size'] = np.mean(position_sizes)
            if trade_results:
                optimization_results['performance_metrics']['total_pnl'] = sum(trade_results)
                optimization_results['performance_metrics']['win_rate'] = sum(1 for pnl in trade_results if pnl > 0) / len(trade_results)
            
            logger.info("V12策略优化完成")
            return optimization_results
            
        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {}
    
    def simulate_trade_execution(self, strategy_config: Dict, position_size: float, 
                               data: pd.DataFrame) -> float:
        """模拟交易执行"""
        try:
            # 简化的交易执行模拟
            entry_price = data['close'].iloc[-1]
            stop_loss_price = entry_price * (1 - strategy_config['stop_loss_bps'] / 10000)
            take_profit_price = entry_price * (1 + strategy_config['take_profit_bps'] / 10000)
            
            # 模拟价格变动
            price_change = np.random.normal(0, 0.001)
            exit_price = entry_price * (1 + price_change)
            
            # 计算PnL
            if exit_price <= stop_loss_price:
                pnl = (stop_loss_price - entry_price) * position_size
            elif exit_price >= take_profit_price:
                pnl = (take_profit_price - entry_price) * position_size
            else:
                pnl = (exit_price - entry_price) * position_size
            
            return pnl
            
        except Exception as e:
            logger.error(f"交易执行模拟失败: {e}")
            return 0.0

def main():
    """主函数"""
    try:
        logger.info("V12策略优化系统启动")
        
        # 初始化组件
        optimizer = V12StrategyOptimizer()
        data_simulator = V12RealisticDataSimulator()
        validation_framework = V12StrictValidationFramework()
        
        # 生成测试数据
        logger.info("生成优化测试数据...")
        test_data = data_simulator.generate_complete_dataset()
        
        # 生成模拟信号
        logger.info("生成模拟交易信号...")
        signals = []
        for i in range(100, len(test_data)):
            signal = {
                'timestamp': test_data.iloc[i]['timestamp'],
                'signal_strength': np.random.uniform(0.3, 0.9),
                'ofi_confidence': np.random.uniform(0.4, 0.8),
                'ai_confidence': np.random.uniform(0.5, 0.9),
                'signal_type': np.random.choice(['momentum', 'divergence', 'breakout'])
            }
            signals.append(signal)
        
        signals_df = pd.DataFrame(signals)
        
        # 运行策略优化
        logger.info("开始策略优化...")
        optimization_results = optimizer.run_strategy_optimization(test_data, signals_df)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results/v12_strategy_optimization_{timestamp}.json"
        
        os.makedirs("backtest_results", exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, ensure_ascii=False, indent=2)
        
        # 生成报告
        logger.info("生成策略优化报告...")
        generate_optimization_report(optimization_results, timestamp)
        
        logger.info("V12策略优化完成")
        
    except Exception as e:
        logger.error(f"策略优化系统失败: {e}")
        raise

def generate_optimization_report(results: Dict, timestamp: str):
    """生成优化报告"""
    try:
        report_content = f"""# V12策略优化报告

## 优化概述
**优化时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**数据量**: {results.get('total_signals', 0)}条信号

## 信号处理结果
- **处理信号数**: {results.get('processed_signals', 0)}
- **批准交易数**: {results.get('approved_trades', 0)}
- **拒绝交易数**: {results.get('rejected_trades', 0)}

## 策略分桶统计
"""
        
        for bucket, count in results.get('strategy_bucket_counts', {}).items():
            report_content += f"- **{bucket}**: {count}笔交易\n"
        
        metrics = results.get('performance_metrics', {})
        report_content += f"""
## 性能指标
- **平均信号质量**: {metrics.get('avg_signal_quality', 0):.4f}
- **平均仓位大小**: {metrics.get('avg_position_size', 0):.4f}
- **总PnL**: {metrics.get('total_pnl', 0):.2f}
- **胜率**: {metrics.get('win_rate', 0):.2%}

## 门控统计
"""
        
        for gate, count in results.get('gating_rejections', {}).items():
            report_content += f"- **{gate}**: {count}次拒绝\n"
        
        report_content += """
## 优化建议

### 策略分桶优化
1. **高波动策略**: 适合趋势明显的市场
2. **趋势策略**: 适合有明确方向的市场
3. **震荡策略**: 适合区间波动的市场
4. **低波动策略**: 建议谨慎使用

### 门控机制优化
1. **交易频率控制**: 避免过度交易
2. **风险预算管理**: 控制最大损失
3. **信号质量过滤**: 提升交易质量
4. **仓位大小优化**: 基于波动率调整

## 结论
V12策略优化系统成功实现了策略分桶与门控机制，为下一步的验证测试奠定了基础。
"""
        
        report_file = f"V12_STRATEGY_OPTIMIZATION_REPORT_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"策略优化报告已保存: {report_file}")
        
    except Exception as e:
        logger.error(f"报告生成失败: {e}")

if __name__ == "__main__":
    main()
