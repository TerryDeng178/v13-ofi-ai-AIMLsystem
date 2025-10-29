#!/usr/bin/env python3
"""
Task 1.5 核心算法v1 - 影子运行器
实现影子/纸上撮合，将信号映射为交易动作意图
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.core_algo import CoreAlgorithm, SignalConfig, SignalData

# 导入成熟组件
from utils.strategy_mode_manager import StrategyModeManager, StrategyMode, MarketActivity
from utils.config_loader import load_config, get_config

class TradeAction(Enum):
    """交易动作"""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    CANCEL_ORDER = "cancel_order"
    HOLD = "hold"

class TradeState(Enum):
    """交易状态"""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"
    COOLING = "cooling"

@dataclass
class TradeSignal:
    """交易信号"""
    timestamp: int
    symbol: str
    action: TradeAction
    price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""

@dataclass
class Position:
    """仓位信息"""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_time: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0

@dataclass
class Order:
    """订单信息"""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    order_type: str  # "market", "limit", "stop"
    status: str  # "pending", "filled", "cancelled"
    timestamp: int
    ttl_seconds: int = 30

class ShadowTrader:
    """影子交易器"""
    
    def __init__(self, symbol: str, config: Dict):
        self.symbol = symbol
        self.config = config
        
        # 加载统一配置
        self.system_config = load_config()
        
        # 初始化算法组件 - 使用统一配置
        signal_config = SignalConfig(
            w_ofi=config.get('w_ofi', 0.6),
            w_cvd=config.get('w_cvd', 0.4),
            z_hi=config.get('z_hi', 2.0),
            z_mid=config.get('z_mid', 1.0)
        )
        self.core_algo = CoreAlgorithm(symbol, signal_config)
        
        # 使用统一配置的策略模式管理器
        strategy_config = self.system_config.get('strategy', {})
        if not strategy_config:
            # 如果没有策略配置，使用默认配置
            strategy_config = {
                'strategy': {
                    'mode': 'auto',
                    'hysteresis': {
                        'window_secs': 60,
                        'min_active_windows': 3,
                        'min_quiet_windows': 6
                    },
                    'triggers': {
                        'schedule': {
                            'enabled': True,
                            'timezone': 'Asia/Hong_Kong',
                            'calendar': 'CRYPTO',
                            'enabled_weekdays': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                            'holidays': [],
                            'active_windows': [],
                            'wrap_midnight': True
                        },
                        'market': {
                            'enabled': True,
                            'window_secs': 60,
                            'min_trades_per_min': 100.0,
                            'min_quote_updates_per_sec': 100,
                            'max_spread_bps': 15.0,
                            'min_volatility_bps': 200.0,  # 0.02 * 10000
                            'min_volume_usd': 1000000,
                            'use_median': True,
                            'winsorize_percentile': 95
                        }
                    }
                }
            }
        self.strategy_manager = StrategyModeManager(strategy_config)
        
        # 交易状态
        self.state = TradeState.FLAT
        self.position: Optional[Position] = None
        self.pending_orders: List[Order] = []
        self.trade_history: List[TradeSignal] = []
        
        # 风控参数
        self.max_position_size = config.get('max_position_size', 1.0)
        self.risk_budget = config.get('risk_budget', 1000.0)
        self.max_trades_per_hour = config.get('max_trades_per_hour', 10)
        self.cooldown_seconds = config.get('cooldown_seconds', 60)
        
        # 统计信息
        self.trades_count = 0
        self.last_trade_time = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        # 日志
        self.logger = logging.getLogger(f"ShadowTrader_{symbol}")
        
    def calculate_position_size(self, signal_strength: float, stop_loss_distance: float) -> float:
        """计算仓位大小"""
        if stop_loss_distance <= 0:
            return 0.0
        
        # 基于风险预算计算仓位大小
        risk_amount = self.risk_budget * min(abs(signal_strength) / 3.0, 1.0)  # 信号强度限制风险
        position_size = risk_amount / stop_loss_distance
        
        # 限制最大仓位
        return min(position_size, self.max_position_size)
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """计算止损价格"""
        if side == "long":
            # 多头止损：入场价 - 1.5倍价差或0.8倍ATR
            spread_stop = entry_price * (1 - 1.5 * 0.0001)  # 假设价差15bps
            atr_stop = entry_price * (1 - 0.8 * atr)
            return max(spread_stop, atr_stop)
        else:
            # 空头止损：入场价 + 1.5倍价差或0.8倍ATR
            spread_stop = entry_price * (1 + 1.5 * 0.0001)
            atr_stop = entry_price * (1 + 0.8 * atr)
            return min(spread_stop, atr_stop)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, side: str, k: float = 1.5) -> float:
        """计算止盈价格"""
        if side == "long":
            return entry_price + k * (entry_price - stop_loss)
        else:
            return entry_price - k * (stop_loss - entry_price)
    
    def check_trading_conditions(self, signal_data: SignalData) -> Tuple[bool, str]:
        """检查交易条件"""
        current_time = int(time.time() * 1000)
        
        # 检查冷却期
        if current_time - self.last_trade_time < self.cooldown_seconds * 1000:
            return False, "cooldown_period"
        
        # 检查交易频率限制
        if self.trades_count >= self.max_trades_per_hour:
            return False, "max_trades_exceeded"
        
        # 检查信号确认
        if not signal_data.confirm:
            return False, "signal_not_confirmed"
        
        # 检查护栏
        if signal_data.gating:
            return False, f"guarded_{signal_data.gating}"
        
        # 检查状态
        if self.state != TradeState.FLAT:
            return False, f"not_flat_state_{self.state.value}"
        
        return True, ""
    
    def process_signal(self, signal_data: SignalData, market_data: Dict) -> Optional[TradeSignal]:
        """处理信号"""
        # 检查交易条件
        can_trade, reason = self.check_trading_conditions(signal_data)
        if not can_trade:
            self.logger.debug(f"交易条件不满足: {reason}")
            return None
        
        # 确定交易方向
        if signal_data.score > self.config.get('z_hi', 2.0):
            action = TradeAction.ENTER_LONG
            side = "long"
        elif signal_data.score < -self.config.get('z_hi', 2.0):
            action = TradeAction.ENTER_SHORT
            side = "short"
        else:
            return None
        
        # 获取市场数据
        current_price = market_data.get('price', 0.0)
        atr = market_data.get('atr', 0.01)
        
        if current_price <= 0:
            return None
        
        # 计算仓位大小
        stop_loss_distance = current_price * atr * 0.8  # 0.8倍ATR作为止损距离
        position_size = self.calculate_position_size(signal_data.score, stop_loss_distance)
        
        if position_size <= 0:
            return None
        
        # 计算止损和止盈
        stop_loss = self.calculate_stop_loss(current_price, side, atr)
        take_profit = self.calculate_take_profit(current_price, stop_loss, side)
        
        # 创建交易信号
        trade_signal = TradeSignal(
            timestamp=signal_data.ts_ms,
            symbol=signal_data.symbol,
            action=action,
            price=current_price,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"signal_score_{signal_data.score:.3f}_regime_{signal_data.regime}"
        )
        
        # 更新状态
        self.state = TradeState.LONG if side == "long" else TradeState.SHORT
        self.position = Position(
            symbol=signal_data.symbol,
            side=side,
            size=position_size,
            entry_price=current_price,
            entry_time=signal_data.ts_ms,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # 更新统计
        self.trades_count += 1
        self.last_trade_time = signal_data.ts_ms
        
        # 记录交易历史
        self.trade_history.append(trade_signal)
        
        self.logger.info(f"生成交易信号: {trade_signal}")
        
        return trade_signal
    
    def check_exit_conditions(self, current_price: float, current_time: int) -> Optional[TradeSignal]:
        """检查退出条件"""
        if not self.position:
            return None
        
        # 检查止损
        if self.position.side == "long" and current_price <= self.position.stop_loss:
            return TradeSignal(
                timestamp=current_time,
                symbol=self.position.symbol,
                action=TradeAction.EXIT_LONG,
                price=current_price,
                size=self.position.size,
                reason="stop_loss"
            )
        elif self.position.side == "short" and current_price >= self.position.stop_loss:
            return TradeSignal(
                timestamp=current_time,
                symbol=self.position.symbol,
                action=TradeAction.EXIT_SHORT,
                price=current_price,
                size=self.position.size,
                reason="stop_loss"
            )
        
        # 检查止盈
        if self.position.side == "long" and current_price >= self.position.take_profit:
            return TradeSignal(
                timestamp=current_time,
                symbol=self.position.symbol,
                action=TradeAction.EXIT_LONG,
                price=current_price,
                size=self.position.size,
                reason="take_profit"
            )
        elif self.position.side == "short" and current_price <= self.position.take_profit:
            return TradeSignal(
                timestamp=current_time,
                symbol=self.position.symbol,
                action=TradeAction.EXIT_SHORT,
                price=current_price,
                size=self.position.size,
                reason="take_profit"
            )
        
        return None
    
    def execute_trade(self, trade_signal: TradeSignal) -> Dict:
        """执行交易（模拟）"""
        # 模拟成交
        fill_price = trade_signal.price
        fill_size = trade_signal.size
        
        # 计算PnL
        if self.position and trade_signal.action in [TradeAction.EXIT_LONG, TradeAction.EXIT_SHORT]:
            if self.position.side == "long":
                pnl = (fill_price - self.position.entry_price) * fill_size
            else:
                pnl = (self.position.entry_price - fill_price) * fill_size
            
            self.total_pnl += pnl
            
            # 更新最大回撤
            if pnl < 0:
                self.max_drawdown = min(self.max_drawdown, pnl)
        
        # 更新状态
        if trade_signal.action in [TradeAction.EXIT_LONG, TradeAction.EXIT_SHORT]:
            self.state = TradeState.COOLING
            self.position = None
        
        # 记录成交
        fill_record = {
            "timestamp": trade_signal.timestamp,
            "symbol": trade_signal.symbol,
            "action": trade_signal.action.value,
            "price": fill_price,
            "size": fill_size,
            "pnl": self.total_pnl,
            "reason": trade_signal.reason
        }
        
        self.logger.info(f"交易执行: {fill_record}")
        
        return fill_record
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_trade_pnl": 0.0
            }
        
        # 计算胜率
        winning_trades = len([t for t in self.trade_history if "profit" in t.reason])
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 计算平均交易PnL
        avg_trade_pnl = self.total_pnl / total_trades if total_trades > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "win_rate": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "current_state": self.state.value,
            "has_position": self.position is not None
        }
    
    def save_trade_log(self, output_dir: str):
        """保存交易日志"""
        log_file = Path(output_dir) / "shadow_trading" / self.symbol / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            for trade in self.trade_history:
                trade_dict = asdict(trade)
                # 转换Enum为字符串
                trade_dict['action'] = trade_dict['action'].value if hasattr(trade_dict['action'], 'value') else str(trade_dict['action'])
                f.write(json.dumps(trade_dict, ensure_ascii=False) + '\n')
        
        # 保存性能指标
        metrics_file = Path(output_dir) / "shadow_trading" / self.symbol / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_performance_metrics(), f, indent=2, ensure_ascii=False)

def load_real_data(symbol: str, data_dir: str = "artifacts/runtime/48h_collection/48h_collection_20251022_0655") -> Dict:
    """加载实时数据收集器的真实数据"""
    data_path = Path(data_dir)
    symbol_data = {}
    
    # 查找最新的数据文件
    date_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('date=')])
    if not date_dirs:
        print(f"警告: 未找到数据目录 {data_path}")
        return {}
    
    latest_date_dir = date_dirs[-1]
    symbol_dir = latest_date_dir / f"symbol={symbol}"
    
    if not symbol_dir.exists():
        print(f"警告: 未找到 {symbol} 的数据目录 {symbol_dir}")
        return {}
    
    # 加载价格数据
    prices_path = symbol_dir / "kind=prices"
    if prices_path.exists():
        price_files = list(prices_path.glob("*.parquet"))
        if price_files:
            try:
                prices_df = pd.concat([pd.read_parquet(f) for f in price_files])
                symbol_data['prices'] = prices_df
                print(f"加载价格数据: {len(prices_df)} 条记录")
            except Exception as e:
                print(f"加载价格数据失败: {e}")
    
    # 加载OFI数据
    ofi_path = symbol_dir / "kind=ofi"
    if ofi_path.exists():
        ofi_files = list(ofi_path.glob("*.parquet"))
        if ofi_files:
            try:
                ofi_df = pd.concat([pd.read_parquet(f) for f in ofi_files])
                symbol_data['ofi'] = ofi_df
                print(f"加载OFI数据: {len(ofi_df)} 条记录")
            except Exception as e:
                print(f"加载OFI数据失败: {e}")
    
    # 加载CVD数据
    cvd_path = symbol_dir / "kind=cvd"
    if cvd_path.exists():
        cvd_files = list(cvd_path.glob("*.parquet"))
        if cvd_files:
            try:
                cvd_df = pd.concat([pd.read_parquet(f) for f in cvd_files])
                symbol_data['cvd'] = cvd_df
                print(f"加载CVD数据: {len(cvd_df)} 条记录")
            except Exception as e:
                print(f"加载CVD数据失败: {e}")
    
    # 加载Fusion数据
    fusion_path = symbol_dir / "kind=fusion"
    if fusion_path.exists():
        fusion_files = list(fusion_path.glob("*.parquet"))
        if fusion_files:
            try:
                fusion_df = pd.concat([pd.read_parquet(f) for f in fusion_files])
                symbol_data['fusion'] = fusion_df
                print(f"加载Fusion数据: {len(fusion_df)} 条记录")
            except Exception as e:
                print(f"加载Fusion数据失败: {e}")
    
    return symbol_data

def main():
    """测试主函数 - 使用真实数据"""
    # 创建影子交易器
    config = {
        'w_ofi': 0.6,
        'w_cvd': 0.4,
        'z_hi': 2.0,
        'z_mid': 1.0,
        'max_position_size': 1.0,
        'risk_budget': 1000.0,
        'max_trades_per_hour': 10,
        'cooldown_seconds': 60
    }
    
    trader = ShadowTrader("BTCUSDT", config)
    
    # 加载真实数据
    print("=== 加载实时数据收集器的真实数据 ===")
    real_data = load_real_data("BTCUSDT")
    
    if not real_data:
        print("❌ 未找到真实数据，请先运行数据收集器")
        print("运行命令: python examples/run_working_harvest.py")
        return
    
    # 使用真实数据进行测试
    if 'prices' in real_data and not real_data['prices'].empty:
        prices_df = real_data['prices']
        print(f"✅ 使用真实价格数据: {len(prices_df)} 条记录")
        
        # 取最后一条价格数据作为测试
        latest_price = prices_df.iloc[-1]
        market_data = {
            'price': float(latest_price.get('price', 50000.0)),
            'atr': 0.01  # 可以从真实数据计算
        }
        
        # 如果有OFI和CVD数据，使用真实信号
        if 'ofi' in real_data and 'cvd' in real_data:
            ofi_df = real_data['ofi']
            cvd_df = real_data['cvd']
            
            if not ofi_df.empty and not cvd_df.empty:
                # 取最新的OFI和CVD数据
                latest_ofi = ofi_df.iloc[-1]
                latest_cvd = cvd_df.iloc[-1]
                
                signal_data = SignalData(
                    ts_ms=int(time.time() * 1000),
                    symbol="BTCUSDT",
                    score=float(latest_ofi.get('z_ofi', 0.0)) * 0.6 + float(latest_cvd.get('z_cvd', 0.0)) * 0.4,
                    z_ofi=float(latest_ofi.get('z_ofi', 0.0)),
                    z_cvd=float(latest_cvd.get('z_cvd', 0.0)),
                    regime="active",
                    confirm=True,
                    gating=False
                )
                
                print(f"✅ 使用真实信号数据: score={signal_data.score:.3f}, z_ofi={signal_data.z_ofi:.3f}, z_cvd={signal_data.z_cvd:.3f}")
            else:
                print("⚠️ OFI或CVD数据为空，使用默认信号")
                signal_data = SignalData(
                    ts_ms=int(time.time() * 1000),
                    symbol="BTCUSDT",
                    score=0.0,
                    z_ofi=0.0,
                    z_cvd=0.0,
                    regime="active",
                    confirm=True,
                    gating=False
                )
        else:
            print("⚠️ 未找到OFI或CVD数据，使用默认信号")
            signal_data = SignalData(
                ts_ms=int(time.time() * 1000),
                symbol="BTCUSDT",
                score=0.0,
                z_ofi=0.0,
                z_cvd=0.0,
                regime="active",
                confirm=True,
                gating=False
            )
        
        # 处理信号
        trade_signal = trader.process_signal(signal_data, market_data)
        
        if trade_signal:
            print(f"✅ 生成交易信号: {trade_signal}")
            
            # 执行交易
            fill_record = trader.execute_trade(trade_signal)
            print(f"✅ 交易执行: {fill_record}")
            
            # 获取性能指标
            metrics = trader.get_performance_metrics()
            print(f"✅ 性能指标: {metrics}")
        else:
            print("ℹ️ 未生成交易信号")
    else:
        print("❌ 未找到价格数据")

if __name__ == "__main__":
    main()
