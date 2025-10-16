#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安V10简化高级交易系统
基于现有的币安V10集成系统，添加简化的订单管理和风险控制
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

from binance_v10_integration import BinanceV10Integration

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimpleOrder:
    """简化订单数据结构"""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    status: str  # "PENDING", "FILLED", "CANCELLED"
    created_time: float
    filled_quantity: float = 0.0

@dataclass
class SimplePosition:
    """简化仓位数据结构"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float

class SimpleOrderManager:
    """简化订单管理器"""
    
    def __init__(self, integration: BinanceV10Integration):
        self.integration = integration
        self.orders: Dict[str, SimpleOrder] = {}
        self.positions: Dict[str, SimplePosition] = {}
        self.order_count = 0
        self.daily_pnl = 0.0
        self.lock = threading.Lock()
        
    def create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[SimpleOrder]:
        """创建市价单"""
        with self.lock:
            try:
                # 生成订单ID
                order_id = f"{symbol}_{side}_{int(time.time() * 1000)}"
                
                # 创建订单对象
                order = SimpleOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=0.0,  # 市价单
                    status="PENDING",
                    created_time=time.time()
                )
                
                # 这里只是模拟，实际应该调用币安API
                logger.info(f"模拟创建订单: {symbol} {side} {quantity}")
                
                # 模拟订单立即成交
                order.status = "FILLED"
                order.filled_quantity = quantity
                
                self.orders[order_id] = order
                self.order_count += 1
                
                # 更新仓位
                self._update_position(symbol, side, quantity, 3990.0)  # 模拟价格
                
                logger.info(f"订单创建成功: {order_id}")
                return order
                
            except Exception as e:
                logger.error(f"订单创建异常: {e}")
                return None
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """更新仓位"""
        if symbol not in self.positions:
            self.positions[symbol] = SimplePosition(
                symbol=symbol,
                side="LONG" if side == "BUY" else "SHORT",
                size=quantity,
                entry_price=price,
                mark_price=price,
                unrealized_pnl=0.0
            )
        else:
            pos = self.positions[symbol]
            if pos.side == "LONG" and side == "BUY":
                # 加仓
                new_size = pos.size + quantity
                pos.entry_price = (pos.entry_price * pos.size + price * quantity) / new_size
                pos.size = new_size
            elif pos.side == "SHORT" and side == "SELL":
                # 加仓
                new_size = pos.size + quantity
                pos.entry_price = (pos.entry_price * pos.size + price * quantity) / new_size
                pos.size = new_size
            elif pos.side == "LONG" and side == "SELL":
                # 减仓或平仓
                if quantity >= pos.size:
                    # 平仓
                    self.positions[symbol] = SimplePosition(
                        symbol=symbol,
                        side="NONE",
                        size=0.0,
                        entry_price=0.0,
                        mark_price=price,
                        unrealized_pnl=0.0
                    )
                else:
                    # 减仓
                    pos.size -= quantity
            elif pos.side == "SHORT" and side == "BUY":
                # 减仓或平仓
                if quantity >= pos.size:
                    # 平仓
                    self.positions[symbol] = SimplePosition(
                        symbol=symbol,
                        side="NONE",
                        size=0.0,
                        entry_price=0.0,
                        mark_price=price,
                        unrealized_pnl=0.0
                    )
                else:
                    # 减仓
                    pos.size -= quantity
    
    def get_position(self, symbol: str) -> Optional[SimplePosition]:
        """获取仓位信息"""
        return self.positions.get(symbol)
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """更新仓位盈亏"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.mark_price = current_price
            
            if pos.side == "LONG" and pos.size > 0:
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            elif pos.side == "SHORT" and pos.size > 0:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
            else:
                pos.unrealized_pnl = 0.0

class SimpleRiskManager:
    """简化风险管理器"""
    
    def __init__(self, order_manager: SimpleOrderManager):
        self.order_manager = order_manager
        self.max_position_size = 1.0  # 最大仓位
        self.stop_loss_pct = 0.02    # 止损2%
        self.take_profit_pct = 0.04   # 止盈4%
        
    def check_risk_limits(self, symbol: str, side: str, quantity: float) -> bool:
        """检查风险限制"""
        position = self.order_manager.get_position(symbol)
        
        if position and position.size > 0:
            # 检查仓位大小
            if side == "BUY" and position.side == "LONG":
                new_size = position.size + quantity
            elif side == "SELL" and position.side == "SHORT":
                new_size = position.size + quantity
            else:
                new_size = abs(position.size - quantity)
            
            if new_size > self.max_position_size:
                logger.warning(f"仓位大小超限: {new_size} > {self.max_position_size}")
                return False
        
        return True
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """检查止损"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        if position.side == "LONG":
            stop_loss_price = position.entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                logger.warning(f"触发止损: {symbol} 当前价格: {current_price:.2f} 止损价格: {stop_loss_price:.2f}")
                return True
        elif position.side == "SHORT":
            stop_loss_price = position.entry_price * (1 + self.stop_loss_pct)
            if current_price >= stop_loss_price:
                logger.warning(f"触发止损: {symbol} 当前价格: {current_price:.2f} 止损价格: {stop_loss_price:.2f}")
                return True
        
        return False
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """检查止盈"""
        position = self.order_manager.get_position(symbol)
        if not position or position.size == 0:
            return False
        
        if position.side == "LONG":
            take_profit_price = position.entry_price * (1 + self.take_profit_pct)
            if current_price >= take_profit_price:
                logger.info(f"触发止盈: {symbol} 当前价格: {current_price:.2f} 止盈价格: {take_profit_price:.2f}")
                return True
        elif position.side == "SHORT":
            take_profit_price = position.entry_price * (1 - self.take_profit_pct)
            if current_price <= take_profit_price:
                logger.info(f"触发止盈: {symbol} 当前价格: {current_price:.2f} 止盈价格: {take_profit_price:.2f}")
                return True
        
        return False

class BinanceV10SimpleAdvanced:
    """币安V10简化高级交易系统"""
    
    def __init__(self, api_key: str, secret_key: str, symbol: str = "ETHUSDT"):
        self.integration = BinanceV10Integration(api_key, secret_key)
        self.order_manager = SimpleOrderManager(self.integration)
        self.risk_manager = SimpleRiskManager(self.order_manager)
        self.symbol = symbol
        self.is_running = False
        self.trading_thread = None
        
    def start(self):
        """启动简化高级交易系统"""
        logger.info("启动币安V10简化高级交易系统...")
        
        # 启动数据流
        self.integration.start()
        
        # 启动交易线程
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("币安V10简化高级交易系统启动成功")
    
    def stop(self):
        """停止简化高级交易系统"""
        logger.info("停止币安V10简化高级交易系统...")
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        self.integration.stop()
        logger.info("币安V10简化高级交易系统已停止")
    
    def _trading_loop(self):
        """交易主循环"""
        logger.info("交易循环开始...")
        
        while self.is_running:
            try:
                # 获取最新数据
                data = self.integration.get_latest_data()
                current_price = data.get('mid_price', 3990.0)  # 默认价格
                ofi_zscore = data.get('ofi_zscore', 0.0)
                
                # 更新仓位盈亏
                self.order_manager.update_position_pnl(self.symbol, current_price)
                
                # 风险检查
                self._risk_management(current_price)
                
                # 信号生成和交易
                self._process_trading_signals(current_price, ofi_zscore)
                
                time.sleep(1)  # 1秒循环
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                time.sleep(1)
    
    def _risk_management(self, current_price: float):
        """风险管理"""
        # 检查止损
        if self.risk_manager.check_stop_loss(self.symbol, current_price):
            self._execute_stop_loss()
        
        # 检查止盈
        if self.risk_manager.check_take_profit(self.symbol, current_price):
            self._execute_take_profit()
    
    def _execute_stop_loss(self):
        """执行止损"""
        position = self.order_manager.get_position(self.symbol)
        if position and position.size > 0:
            side = "SELL" if position.side == "LONG" else "BUY"
            self.order_manager.create_market_order(self.symbol, side, position.size)
            logger.info(f"执行止损: {self.symbol} {side} {position.size}")
    
    def _execute_take_profit(self):
        """执行止盈"""
        position = self.order_manager.get_position(self.symbol)
        if position and position.size > 0:
            side = "SELL" if position.side == "LONG" else "BUY"
            self.order_manager.create_market_order(self.symbol, side, position.size)
            logger.info(f"执行止盈: {self.symbol} {side} {position.size}")
    
    def _process_trading_signals(self, current_price: float, ofi_zscore: float):
        """处理交易信号"""
        signal_strength = abs(ofi_zscore)
        
        # 信号阈值
        if signal_strength < 1.5:
            return
        
        # 获取当前仓位
        position = self.order_manager.get_position(self.symbol)
        
        # 生成交易信号
        if ofi_zscore > 1.5 and (not position or position.size == 0):
            # 买入信号
            quantity = 0.01
            if self.risk_manager.check_risk_limits(self.symbol, "BUY", quantity):
                self.order_manager.create_market_order(self.symbol, "BUY", quantity)
                logger.info(f"买入信号: {self.symbol} 数量: {quantity} OFI: {ofi_zscore:.3f}")
            
        elif ofi_zscore < -1.5 and (not position or position.size == 0):
            # 卖出信号
            quantity = 0.01
            if self.risk_manager.check_risk_limits(self.symbol, "SELL", quantity):
                self.order_manager.create_market_order(self.symbol, "SELL", quantity)
                logger.info(f"卖出信号: {self.symbol} 数量: {quantity} OFI: {ofi_zscore:.3f}")
    
    def get_trading_status(self) -> Dict:
        """获取交易状态"""
        position = self.order_manager.get_position(self.symbol)
        data = self.integration.get_latest_data()
        
        return {
            "symbol": self.symbol,
            "current_price": data.get('mid_price', 3990.0),
            "ofi_zscore": data.get('ofi_zscore', 0.0),
            "position": {
                "side": position.side if position else "NONE",
                "size": position.size if position else 0,
                "entry_price": position.entry_price if position else 0,
                "unrealized_pnl": position.unrealized_pnl if position else 0
            } if position else None,
            "order_count": self.order_manager.order_count,
            "active_orders": len([o for o in self.order_manager.orders.values() if o.status == "PENDING"])
        }

if __name__ == "__main__":
    # 币安测试网API配置
    API_KEY = "tKs2nQW3t3rcHnUCGOPWLIYsDotF6ot1GRZzo5n3QOsNRqbPTxO06TC8QglxtUYy"
    SECRET_KEY = "I5SqluL8yqvkcSSb0hBBMr80cHmNndxrl6DAufw9t6myjlRBQDvPxUmabHNJpkZJ"
    
    # 创建简化高级交易系统
    trading_system = BinanceV10SimpleAdvanced(API_KEY, SECRET_KEY)
    
    try:
        # 启动系统
        trading_system.start()
        
        # 运行测试
        logger.info("简化高级交易系统运行中...")
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
        logger.info("简化高级交易系统测试完成")
