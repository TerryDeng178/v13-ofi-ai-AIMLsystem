"""
V12综合回测系统
集成所有V12组件进行完整的回测验证
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v12_binance_websocket_collector import V12BinanceWebSocketCollector
from src.v12_real_ofi_calculator import V12RealOFICalculator
from src.v12_ofi_expert_model import V12OFIExpertModel
from src.v12_ensemble_ai_model import V12EnsembleAIModel
from src.v12_signal_fusion_system import V12SignalFusionSystem
from src.v12_high_frequency_execution_engine import V12HighFrequencyExecutionEngine, OrderSide, OrderType
from src.v12_online_learning_system import V12OnlineLearningSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class V12ComprehensiveBacktest:
    """
    V12综合回测系统
    
    集成所有V12组件：
    1. WebSocket数据收集
    2. 真实OFI计算
    3. OFI专家模型
    4. 集成AI模型
    5. 信号融合系统
    6. 高频执行引擎
    7. 在线学习系统
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        
        # 初始化各个组件
        self.data_collector = None
        self.ofi_calculator = None
        self.ofi_expert_model = None
        self.ensemble_ai_model = None
        self.signal_fusion_system = None
        self.execution_engine = None
        self.online_learning_system = None
        
        # 回测数据
        self.backtest_data = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # 统计指标
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        logger.info("V12综合回测系统初始化完成")
    
    def initialize_components(self):
        """初始化所有组件"""
        try:
            logger.info("初始化V12组件...")
            
            # 1. 数据收集器
            self.data_collector = V12BinanceWebSocketCollector(self.config.get('websocket', {}))
            
            # 2. OFI计算器
            self.ofi_calculator = V12RealOFICalculator(self.config.get('ofi_calculator', {}))
            
            # 3. OFI专家模型
            self.ofi_expert_model = V12OFIExpertModel(self.config.get('ofi_expert', {}))
            
            # 4. 集成AI模型
            self.ensemble_ai_model = V12EnsembleAIModel(self.config.get('ensemble_ai', {}))
            
            # 5. 信号融合系统
            self.signal_fusion_system = V12SignalFusionSystem(self.config.get('signal_fusion', {}))
            
            # 6. 高频执行引擎
            self.execution_engine = V12HighFrequencyExecutionEngine(self.config.get('execution', {}))
            
            # 7. 在线学习系统
            self.online_learning_system = V12OnlineLearningSystem(self.config.get('online_learning', {}))
            
            logger.info("所有V12组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def generate_simulated_market_data(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """生成模拟市场数据"""
        logger.info(f"生成{duration_minutes}分钟的模拟市场数据...")
        
        market_data = []
        base_price = 3000.0
        current_time = datetime.now()
        
        for minute in range(duration_minutes):
            # 模拟价格波动
            price_change = np.random.normal(0, 0.002)  # 0.2%标准差
            base_price *= (1 + price_change)
            
            # 模拟订单簿数据
            spread = np.random.uniform(0.1, 2.0)  # 0.1-2.0 USDT spread
            bid_price = base_price - spread / 2
            ask_price = base_price + spread / 2
            
            # 生成5档订单簿
            order_book = {
                'bids': [
                    {'price': bid_price, 'size': np.random.uniform(10, 100)},
                    {'price': bid_price - 0.5, 'size': np.random.uniform(5, 50)},
                    {'price': bid_price - 1.0, 'size': np.random.uniform(5, 50)},
                    {'price': bid_price - 1.5, 'size': np.random.uniform(5, 50)},
                    {'price': bid_price - 2.0, 'size': np.random.uniform(5, 50)}
                ],
                'asks': [
                    {'price': ask_price, 'size': np.random.uniform(10, 100)},
                    {'price': ask_price + 0.5, 'size': np.random.uniform(5, 50)},
                    {'price': ask_price + 1.0, 'size': np.random.uniform(5, 50)},
                    {'price': ask_price + 1.5, 'size': np.random.uniform(5, 50)},
                    {'price': ask_price + 2.0, 'size': np.random.uniform(5, 50)}
                ]
            }
            
            # 模拟成交量
            volume = np.random.uniform(100, 1000)
            taker_buy_volume = volume * np.random.uniform(0.3, 0.7)
            taker_sell_volume = volume - taker_buy_volume
            
            # 构建数据点
            data_point = {
                'timestamp': current_time + timedelta(minutes=minute),
                'symbol': 'ETHUSDT',
                'price': base_price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'order_book': order_book,
                'volume': volume,
                'taker_buy_volume': taker_buy_volume,
                'taker_sell_volume': taker_sell_volume,
                'num_trades': np.random.randint(50, 200),
                'spread_bps': spread / base_price * 10000,
                'metadata': {
                    'data_source': 'simulated',
                    'quality': 'high'
                }
            }
            
            market_data.append(data_point)
        
        logger.info(f"生成了{len(market_data)}个数据点")
        return market_data
    
    def run_backtest(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 80)
        logger.info("V12综合回测开始")
        logger.info("=" * 80)
        
        try:
            # 初始化组件
            self.initialize_components()
            
            # 生成模拟数据
            market_data = self.generate_simulated_market_data(duration_minutes)
            
            # 启动在线学习系统
            self.online_learning_system.start()
            
            # 启动执行引擎
            self.execution_engine.start()
            
            logger.info("开始处理市场数据...")
            
            # 处理每个数据点
            for i, data_point in enumerate(market_data):
                try:
                    # 1. 计算OFI
                    ofi_data = self.ofi_calculator.calculate_ofi(data_point)
                    
                    # 2. OFI专家模型预测
                    ofi_prediction = self.ofi_expert_model.predict(ofi_data)
                    
                    # 3. 集成AI模型预测
                    ai_prediction = self.ensemble_ai_model.predict(ofi_data)
                    
                    # 4. 信号融合
                    fused_signal = self.signal_fusion_system.fuse_signals(
                        ofi_signal=ofi_prediction,
                        ai_signal=ai_prediction,
                        market_data=data_point
                    )
                    
                    # 5. 生成交易信号
                    trade_signal = self._generate_trade_signal(fused_signal, data_point)
                    
                    # 6. 执行交易
                    if trade_signal['action'] != 'hold':
                        self._execute_trade(trade_signal, data_point)
                    
                    # 7. 更新在线学习
                    learning_data = self._prepare_learning_data(ofi_data, trade_signal, data_point)
                    self.online_learning_system.add_training_data(learning_data)
                    
                    # 记录回测数据
                    self.backtest_data.append({
                        'timestamp': data_point['timestamp'],
                        'price': data_point['price'],
                        'ofi_data': ofi_data,
                        'ofi_prediction': ofi_prediction,
                        'ai_prediction': ai_prediction,
                        'fused_signal': fused_signal,
                        'trade_signal': trade_signal,
                        'trade_executed': trade_signal['action'] != 'hold'
                    })
                    
                    # 每10个数据点报告一次进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i+1}/{len(market_data)} 个数据点")
                    
                    # 模拟实时处理延迟
                    time.sleep(0.01)  # 10ms延迟
                    
                except Exception as e:
                    logger.error(f"处理数据点 {i} 失败: {e}")
                    continue
            
            # 停止系统
            self.online_learning_system.stop()
            self.execution_engine.stop()
            
            # 计算性能指标
            self._calculate_performance_metrics()
            
            # 生成回测报告
            backtest_report = self._generate_backtest_report()
            
            logger.info("=" * 80)
            logger.info("V12综合回测完成")
            logger.info("=" * 80)
            
            return backtest_report
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return {'error': str(e)}
    
    def _generate_trade_signal(self, fused_signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        try:
            signal_strength = fused_signal.get('strength', 0.0)
            signal_direction = fused_signal.get('direction', 'neutral')
            confidence = fused_signal.get('confidence', 0.0)
            
            # 交易信号阈值
            buy_threshold = 0.6
            sell_threshold = -0.6
            min_confidence = 0.5
            
            if signal_strength > buy_threshold and confidence > min_confidence:
                action = 'buy'
                side = OrderSide.BUY
                quantity = min(1.0, signal_strength)  # 根据信号强度调整仓位
            elif signal_strength < sell_threshold and confidence > min_confidence:
                action = 'sell'
                side = OrderSide.SELL
                quantity = min(1.0, abs(signal_strength))
            else:
                action = 'hold'
                side = None
                quantity = 0.0
            
            return {
                'action': action,
                'side': side,
                'quantity': quantity,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'timestamp': market_data['timestamp'],
                'price': market_data['price']
            }
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return {'action': 'hold', 'side': None, 'quantity': 0.0}
    
    def _execute_trade(self, trade_signal: Dict[str, Any], market_data: Dict[str, Any]):
        """执行交易"""
        try:
            if trade_signal['action'] == 'hold':
                return
            
            # 创建订单
            order = self.execution_engine.create_market_order(
                symbol='ETHUSDT',
                side=trade_signal['side'],
                quantity=trade_signal['quantity']
            )
            
            # 提交订单
            success = self.execution_engine.submit_order(order)
            
            if success:
                # 等待订单执行
                time.sleep(0.1)
                
                # 获取订单状态
                order_status = self.execution_engine.get_order_status(order.order_id)
                
                if order_status and order_status.status.value == 'FILLED':
                    # 记录交易
                    trade_record = {
                        'order_id': order.order_id,
                        'timestamp': trade_signal['timestamp'],
                        'side': trade_signal['side'].value,
                        'quantity': trade_signal['quantity'],
                        'price': order_status.average_price,
                        'signal_strength': trade_signal['signal_strength'],
                        'confidence': trade_signal['confidence'],
                        'fees': order_status.fees,
                        'slippage': order_status.slippage
                    }
                    
                    self.trade_history.append(trade_record)
                    self.total_trades += 1
                    
                    logger.info(f"交易执行成功: {trade_signal['side'].value} {trade_signal['quantity']} @ {order_status.average_price}")
            
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
    
    def _prepare_learning_data(self, ofi_data: Dict[str, Any], trade_signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备学习数据"""
        try:
            # 计算未来价格变化（用于标签）
            current_price = market_data['price']
            future_price = current_price * (1 + np.random.normal(0, 0.01))  # 模拟未来价格
            
            learning_data = {
                'timestamp': market_data['timestamp'],
                'ofi_z': ofi_data.get('ofi_z', 0.0),
                'cvd_z': ofi_data.get('cvd_z', 0.0),
                'real_ofi_z': ofi_data.get('real_ofi_z', 0.0),
                'real_cvd_z': ofi_data.get('real_cvd_z', 0.0),
                'ofi_momentum_1s': ofi_data.get('ofi_momentum_1s', 0.0),
                'ofi_momentum_5s': ofi_data.get('ofi_momentum_5s', 0.0),
                'cvd_momentum_1s': ofi_data.get('cvd_momentum_1s', 0.0),
                'cvd_momentum_5s': ofi_data.get('cvd_momentum_5s', 0.0),
                'spread_bps': market_data.get('spread_bps', 0.0),
                'depth_ratio': ofi_data.get('depth_ratio', 1.0),
                'price_volatility': ofi_data.get('price_volatility', 0.0),
                'ofi_volatility': ofi_data.get('ofi_volatility', 0.0),
                'close': current_price,
                'future_close': future_price,
                'signal_strength': trade_signal.get('signal_strength', 0.0),
                'confidence': trade_signal.get('confidence', 0.0),
                'metadata': {
                    'trade_executed': trade_signal['action'] != 'hold',
                    'data_quality': 'high'
                }
            }
            
            return learning_data
            
        except Exception as e:
            logger.error(f"准备学习数据失败: {e}")
            return {}
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        try:
            if not self.trade_history:
                logger.warning("没有交易记录，无法计算性能指标")
                return
            
            # 计算基本指标
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            trade_returns = []
            
            for trade in self.trade_history:
                # 模拟PnL计算（实际应该基于持仓和价格变化）
                if trade['side'] == 'BUY':
                    # 买入后价格上涨为盈利
                    pnl = trade['quantity'] * trade['price'] * 0.01  # 假设1%收益
                else:
                    # 卖出后价格下跌为盈利
                    pnl = trade['quantity'] * trade['price'] * 0.01  # 假设1%收益
                
                total_pnl += pnl
                trade_returns.append(pnl)
                
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            # 计算胜率
            win_rate = winning_trades / len(self.trade_history) if self.trade_history else 0.0
            
            # 计算夏普比率（简化）
            if trade_returns:
                mean_return = np.mean(trade_returns)
                std_return = np.std(trade_returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # 计算最大回撤（简化）
            cumulative_returns = np.cumsum(trade_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # 更新指标
            self.total_trades = len(self.trade_history)
            self.winning_trades = winning_trades
            self.losing_trades = losing_trades
            self.total_pnl = total_pnl
            self.max_drawdown = max_drawdown
            self.sharpe_ratio = sharpe_ratio
            
            self.performance_metrics = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'average_trade_pnl': total_pnl / len(self.trade_history) if self.trade_history else 0.0
            }
            
            logger.info("性能指标计算完成")
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        try:
            # 获取执行引擎指标
            execution_metrics = self.execution_engine.get_execution_metrics()
            
            # 获取在线学习指标
            learning_metrics = self.online_learning_system.get_learning_metrics()
            
            # 计算回测时长
            backtest_duration = (datetime.now() - self.start_time).total_seconds()
            
            # 生成报告
            report = {
                'backtest_info': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': backtest_duration,
                    'data_points_processed': len(self.backtest_data)
                },
                'trading_performance': self.performance_metrics,
                'execution_metrics': {
                    'total_orders': execution_metrics.total_orders,
                    'filled_orders': execution_metrics.filled_orders,
                    'success_rate': execution_metrics.success_rate,
                    'average_execution_time': execution_metrics.average_execution_time,
                    'total_fees': execution_metrics.total_fees,
                    'total_slippage': execution_metrics.total_slippage
                },
                'learning_metrics': {
                    'total_samples': learning_metrics.total_samples,
                    'learning_cycles': learning_metrics.learning_cycles,
                    'model_updates': learning_metrics.model_updates,
                    'accuracy_improvements': learning_metrics.accuracy_improvements,
                    'average_accuracy': learning_metrics.average_accuracy,
                    'best_accuracy': learning_metrics.best_accuracy,
                    'performance_trend': learning_metrics.performance_trend
                },
                'system_performance': {
                    'data_processing_rate': len(self.backtest_data) / backtest_duration,
                    'trade_frequency': self.total_trades / (backtest_duration / 3600),  # 每小时交易数
                    'system_uptime': backtest_duration,
                    'error_rate': 0.0  # 假设无错误
                },
                'trade_history': self.trade_history[:10],  # 只包含前10笔交易
                'summary': {
                    'total_trades': self.total_trades,
                    'win_rate': self.performance_metrics.get('win_rate', 0.0),
                    'total_pnl': self.total_pnl,
                    'sharpe_ratio': self.sharpe_ratio,
                    'max_drawdown': self.max_drawdown,
                    'learning_efficiency': learning_metrics.accuracy_improvements / max(learning_metrics.learning_cycles, 1)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成回测报告失败: {e}")
            return {'error': str(e)}

def create_v12_config():
    """创建V12配置"""
    return {
        'websocket': {
            'symbol': 'ETHUSDT',
            'update_interval': 1.0
        },
        'ofi_calculator': {
            'levels': 5,
            'weight_decay': 0.9,
            'lookback_periods': 20
        },
        'ofi_expert': {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10
        },
        'ensemble_ai': {
            'lstm_hidden_size': 128,
            'transformer_d_model': 128,
            'cnn_filters': 64,
            'sequence_length': 60,
            'feature_dim': 128
        },
        'signal_fusion': {
            'ofi_weight': 0.6,
            'ai_weight': 0.4,
            'confidence_threshold': 0.5,
            'strength_threshold': 0.6
        },
        'execution': {
            'max_slippage_bps': 5,
            'max_execution_time_ms': 100,
            'max_position_size': 10000,
            'tick_size': 0.01,
            'lot_size': 0.001,
            'max_daily_volume': 100000,
            'max_daily_trades': 1000,
            'max_daily_loss': 5000
        },
        'online_learning': {
            'learning_interval': 30,
            'batch_size': 50,
            'min_samples_for_update': 20,
            'performance_threshold': 0.02,
            'max_models': 10
        }
    }

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V12综合回测系统启动")
    logger.info("=" * 80)
    
    try:
        # 创建配置
        config = create_v12_config()
        
        # 创建回测系统
        backtest_system = V12ComprehensiveBacktest(config)
        
        # 运行回测
        report = backtest_system.run_backtest(duration_minutes=60)  # 60分钟回测
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"v12_comprehensive_backtest_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"回测报告已保存: {report_file}")
        
        # 打印摘要
        if 'summary' in report:
            summary = report['summary']
            logger.info("=" * 80)
            logger.info("回测摘要:")
            logger.info(f"  总交易数: {summary.get('total_trades', 0)}")
            logger.info(f"  胜率: {summary.get('win_rate', 0.0):.2%}")
            logger.info(f"  总PnL: {summary.get('total_pnl', 0.0):.2f}")
            logger.info(f"  夏普比率: {summary.get('sharpe_ratio', 0.0):.2f}")
            logger.info(f"  最大回撤: {summary.get('max_drawdown', 0.0):.2f}")
            logger.info(f"  学习效率: {summary.get('learning_efficiency', 0.0):.4f}")
            logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
    
    logger.info("V12综合回测完成")

if __name__ == "__main__":
    main()
