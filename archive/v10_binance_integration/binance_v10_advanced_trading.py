#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10高级交易系统
集成完整的订单管理、风险控制、仓位管理功能
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

from binance_v10_integration import BinanceV10Integration

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    created_time: float = 0.0
    updated_time: float = 0.0
    client_order_id: str = ""

@dataclass
class Position:
    """仓位数据结构"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float
    leverage: float

@dataclass
class RiskLimits:
    """风险限制配置"""
    max_position_size: float = 1000.0  # 最大仓位大小
    max_daily_loss: float = 500.0      # 最大日亏损
    max_drawdown: float = 0.1          # 最大回撤比例
    max_leverage: float = 10.0         # 最大杠杆
    stop_loss_pct: float = 0.02        # 止损百分比
    take_profit_pct: float = 0.04      # 止盈百分比
    max_orders_per_minute: int = 10     # 每分钟最大订单数

class AdvancedOrderManager:
    """高级订单管理器"""
    
    def __init__(self, integration: BinanceV10Integration):
        self.integration = integration
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: deque = deque(maxlen=10000)
        self.risk_limits = RiskLimits()
        self.daily_pnl = 0.0
        self.max_daily_pnl = 0.0
        self.order_count = 0
        self.last_order_time = 0.0
        self.lock = threading.Lock()
        
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                    quantity: float, price: Optional[float] = None, 
                    stop_price: Optional[float] = None) -> Optional[Order]:
        """创建订单"""
        with self.lock:
            # 风险检查
            if not self._check_risk_limits(symbol, side, quantity):
                logger.warning(f"风险检查失败，拒绝订单: {symbol} {side.value} {quantity}")
                return None
            
            # 生成订单ID
            order_id = f"{symbol}_{side.value}_{int(time.time() * 1000)}"
            client_order_id = f"V10_{order_id}"
            
            # 创建订单对象
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                remaining_quantity=quantity,
                created_time=time.time(),
                client_order_id=client_order_id
            )
            
            # 提交到币安
            try:
                result = self.integration.place_order(
                    symbol=symbol,
                    side=side.value,
                    type=order_type.value,
                    quantity=quantity,
                    price=price,
                    clientOrderId=client_order_id
                )
                
                if result and result.get('orderId'):
                    order.order_id = str(result['orderId'])
                    self.orders[order_id] = order
                    self.order_count += 1
                    self.last_order_time = time.time()
                    logger.info(f"订单创建成功: {order_id}")
                    return order
                else:
                    logger.error(f"订单创建失败: {result}")
                    return None
                    
            except Exception as e:
                logger.error(f"订单创建异常: {e}")
                return None
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        with self.lock:
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"订单已完成，无法取消: {order_id}")
                return False
            
            try:
                result = self.integration.cancel_order(order_id=order_id)
                if result and result.get('status') == 'CANCELED':
                    order.status = OrderStatus.CANCELLED
                    order.updated_time = time.time()
                    logger.info(f"订单取消成功: {order_id}")
                    return True
                else:
                    logger.error(f"订单取消失败: {result}")
                    return False
                    
            except Exception as e:
                logger.error(f"订单取消异常: {e}")
                return False
    
    def update_order_status(self, order_id: str) -> bool:
        """更新订单状态"""
        with self.lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            try:
                # 获取订单状态
                open_orders = self.integration.get_open_orders()
                order_found = False
                
                for o in open_orders:
                    if str(o.get('orderId')) == order_id:
                        order_found = True
                        order.status = OrderStatus(o.get('status', 'PENDING'))
                        order.filled_quantity = float(o.get('executedQty', 0))
                        order.remaining_quantity = float(o.get('origQty', 0)) - order.filled_quantity
                        order.average_price = float(o.get('avgPrice', 0))
                        order.updated_time = time.time()
                        break
                
                if not order_found and order.status == OrderStatus.PENDING:
                    # 订单不在挂单列表中，可能已成交
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = 0.0
                    order.updated_time = time.time()
                
                return True
                
            except Exception as e:
                logger.error(f"更新订单状态异常: {e}")
                return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取仓位信息"""
        try:
            positions = self.integration.get_position_info(symbol)
            if positions and len(positions) > 0:
                pos = positions[0]
                return Position(
                    symbol=symbol,
                    side="LONG" if float(pos.get('positionAmt', 0)) > 0 else "SHORT",
                    size=abs(float(pos.get('positionAmt', 0))),
                    entry_price=float(pos.get('entryPrice', 0)),
                    mark_price=float(pos.get('markPrice', 0)),
                    unrealized_pnl=float(pos.get('unRealizedProfit', 0)),
                    realized_pnl=float(pos.get('realizedPnl', 0)),
                    margin=float(pos.get('isolatedMargin', 0)),
                    leverage=float(pos.get('leverage', 1))
                )
            return None
        except Exception as e:
            logger.error(f"获取仓位信息异常: {e}")
            return None
    
    def _check_risk_limits(self, symbol: str, side: OrderSide, quantity: float) -> bool:
        """风险限制检查"""
        current_time = time.time()
        
        # 检查订单频率限制
        if current_time - self.last_order_time < 6.0:  # 每分钟最多10个订单
            logger.warning("订单频率过高，拒绝订单")
            return False
        
        # 检查仓位大小限制
        position = self.get_position(symbol)
        if position:
            new_size = position.size + quantity if side == OrderSide.BUY else position.size - quantity
            if new_size > self.risk_limits.max_position_size:
                logger.warning(f"仓位大小超限: {new_size} > {self.risk_limits.max_position_size}")
                return False
        
        # 检查日亏损限制
        if self.daily_pnl < -self.risk_limits.max_daily_loss:
            logger.warning(f"日亏损超限: {self.daily_pnl} < -{self.risk_limits.max_daily_loss}")
            return False
        
        return True
    
    def update_daily_pnl(self):
        """更新日盈亏"""
        try:
            account_info = self.integration.get_account_info()
            if account_info and 'totalWalletBalance' in account_info:
                current_balance = float(account_info['totalWalletBalance'])
                # 这里需要与初始余额比较来计算日盈亏
                # 简化处理，使用未实现盈亏
                self.daily_pnl = current_balance - 10000.0  # 假设初始余额10000
                self.max_daily_pnl = max(self.max_daily_pnl, self.daily_pnl)
        except Exception as e:
            logger.error(f"更新日盈亏异常: {e}")

class AdvancedRiskManager:
    """高级风险管理器"""
    
    def __init__(self, order_manager: AdvancedOrderManager):
        self.order_manager = order_manager
        self.stop_loss_orders: Dict[str, str] = {}  # symbol -> stop_loss_order_id
        self.take_profit_orders: Dict[str, str] = {}  # symbol -> take_profit_order_id
        
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """检查止损条件"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        # 计算止损价格
        if position.side == "LONG":
            stop_loss_price = position.entry_price * (1 - self.order_manager.risk_limits.stop_loss_pct)
            if current_price <= stop_loss_price:
                logger.warning(f"触发止损: {symbol} 当前价格: {current_price} 止损价格: {stop_loss_price}")
                return True
        else:  # SHORT
            stop_loss_price = position.entry_price * (1 + self.order_manager.risk_limits.stop_loss_pct)
            if current_price >= stop_loss_price:
                logger.warning(f"触发止损: {symbol} 当前价格: {current_price} 止损价格: {stop_loss_price}")
                return True
        
        return False
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """检查止盈条件"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        # 计算止盈价格
        if position.side == "LONG":
            take_profit_price = position.entry_price * (1 + self.order_manager.risk_limits.take_profit_pct)
            if current_price >= take_profit_price:
                logger.info(f"触发止盈: {symbol} 当前价格: {current_price} 止盈价格: {take_profit_price}")
                return True
        else:  # SHORT
            take_profit_price = position.entry_price * (1 - self.order_manager.risk_limits.take_profit_pct)
            if current_price <= take_profit_price:
                logger.info(f"触发止盈: {symbol} 当前价格: {current_price} 止盈价格: {take_profit_price}")
                return True
        
        return False
    
    def execute_stop_loss(self, symbol: str) -> bool:
        """执行止损"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        try:
            # 平仓订单
            side = OrderSide.SELL if position.side == "LONG" else OrderSide.BUY
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position.size
            )
            
            if order:
                logger.info(f"止损订单已提交: {symbol} {side.value} {position.size}")
                return True
            else:
                logger.error(f"止损订单提交失败: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"执行止损异常: {e}")
            return False
    
    def execute_take_profit(self, symbol: str) -> bool:
        """执行止盈"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        try:
            # 平仓订单
            side = OrderSide.SELL if position.side == "LONG" else OrderSide.BUY
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position.size
            )
            
            if order:
                logger.info(f"止盈订单已提交: {symbol} {side.value} {position.size}")
                return True
            else:
                logger.error(f"止盈订单提交失败: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"执行止盈异常: {e}")
            return False

class BinanceV10AdvancedTrading:
    """币安V10高级交易系统"""
    
    def __init__(self, api_key: str, secret_key: str, symbol: str = "ETHUSDT"):
        self.integration = BinanceV10Integration(api_key, secret_key)
        self.order_manager = AdvancedOrderManager(self.integration)
        self.risk_manager = AdvancedRiskManager(self.order_manager)
        self.symbol = symbol
        self.is_running = False
        self.trading_thread = None
        
    def start(self):
        """启动高级交易系统"""
        logger.info("启动币安V10高级交易系统...")
        
        # 启动数据流
        self.integration.start()
        
        # 启动交易线程
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("币安V10高级交易系统启动成功")
    
    def stop(self):
        """停止高级交易系统"""
        logger.info("停止币安V10高级交易系统...")
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        self.integration.stop()
        logger.info("币安V10高级交易系统已停止")
    
    def _trading_loop(self):
        """交易主循环"""
        logger.info("交易循环开始...")
        
        while self.is_running:
            try:
                # 获取最新特征
                features = self.integration.get_latest_features()
                current_price = features.get('mid_price', 0)
                
                if current_price == 0:
                    time.sleep(0.1)
                    continue
                
                # 更新订单状态
                self._update_all_orders()
                
                # 更新日盈亏
                self.order_manager.update_daily_pnl()
                
                # 风险检查
                self._risk_management(current_price)
                
                # 信号生成和交易
                self._process_trading_signals(features, current_price)
                
                time.sleep(0.1)  # 100ms循环
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                time.sleep(1)
    
    def _update_all_orders(self):
        """更新所有订单状态"""
        for order_id in list(self.order_manager.orders.keys()):
            self.order_manager.update_order_status(order_id)
    
    def _risk_management(self, current_price: float):
        """风险管理"""
        # 检查止损
        if self.risk_manager.check_stop_loss(self.symbol, current_price):
            self.risk_manager.execute_stop_loss(self.symbol)
        
        # 检查止盈
        if self.risk_manager.check_take_profit(self.symbol, current_price):
            self.risk_manager.execute_take_profit(self.symbol)
    
    def _process_trading_signals(self, features: Dict, current_price: float):
        """处理交易信号"""
        ofi_zscore = features.get('ofi_zscore', 0)
        signal_strength = abs(ofi_zscore)
        
        # 信号阈值
        if signal_strength < 1.5:
            return
        
        # 获取当前仓位
        position = self.order_manager.get_position(self.symbol)
        
        # 生成交易信号
        if ofi_zscore > 1.5 and (not position or position.size == 0):
            # 买入信号
            quantity = min(0.01, self.order_manager.risk_limits.max_position_size * 0.1)
            self.order_manager.create_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            logger.info(f"买入信号: {self.symbol} 数量: {quantity} OFI: {ofi_zscore:.3f}")
            
        elif ofi_zscore < -1.5 and (not position or position.size == 0):
            # 卖出信号
            quantity = min(0.01, self.order_manager.risk_limits.max_position_size * 0.1)
            self.order_manager.create_order(
                symbol=self.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            logger.info(f"卖出信号: {self.symbol} 数量: {quantity} OFI: {ofi_zscore:.3f}")
    
    def get_trading_status(self) -> Dict:
        """获取交易状态"""
        position = self.order_manager.get_position(self.symbol)
        features = self.integration.get_latest_features()
        
        return {
            "symbol": self.symbol,
            "current_price": features.get('mid_price', 0),
            "ofi_zscore": features.get('ofi_zscore', 0),
            "position": {
                "side": position.side if position else "NONE",
                "size": position.size if position else 0,
                "entry_price": position.entry_price if position else 0,
                "unrealized_pnl": position.unrealized_pnl if position else 0
            } if position else None,
            "daily_pnl": self.order_manager.daily_pnl,
            "order_count": self.order_manager.order_count,
            "active_orders": len([o for o in self.order_manager.orders.values() if o.status == OrderStatus.PENDING])
        }

if __name__ == "__main__":
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建高级交易系统
    trading_system = BinanceV10AdvancedTrading(API_KEY, SECRET_KEY)
    
    try:
        # 启动系统
        trading_system.start()
        
        # 运行测试
        logger.info("高级交易系统运行中...")
        time.sleep(60)  # 运行60秒
        
        # 显示状态
        status = trading_system.get_trading_status()
        logger.info(f"交易状态: {status}")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"系统异常: {e}")
    finally:
        # 停止系统
        trading_system.stop()
        logger.info("高级交易系统测试完成")
