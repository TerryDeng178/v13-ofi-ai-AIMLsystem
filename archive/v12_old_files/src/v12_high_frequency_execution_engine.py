"""
V12 高频交易执行引擎
实现毫秒级交易执行、智能订单路由、滑点控制和执行成本优化
"""

import asyncio
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import threading
from collections import deque
import queue
import uuid

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "POST_ONLY"

class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    timestamp: datetime = None
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    cancel_time: Optional[datetime] = None
    fees: float = 0.0
    slippage: float = 0.0
    execution_cost: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def __lt__(self, other):
        """支持PriorityQueue的比较操作"""
        if not isinstance(other, Order):
            return NotImplemented
        return self.submit_time < other.submit_time if self.submit_time and other.submit_time else False

@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    average_slippage: float = 0.0
    execution_cost: float = 0.0

class V12HighFrequencyExecutionEngine:
    """
    V12 高频交易执行引擎
    
    核心功能:
    1. 毫秒级订单执行
    2. 智能订单路由
    3. 滑点控制
    4. 执行成本优化
    5. 风险管理
    6. 实时监控
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.execution_queue = queue.PriorityQueue()
        self.execution_thread = None
        self.running = False
        self.metrics = ExecutionMetrics()
        
        # 执行参数
        self.max_slippage_bps = config.get('max_slippage_bps', 5)  # 最大滑点5bps
        self.max_execution_time_ms = config.get('max_execution_time_ms', 100)  # 最大执行时间100ms
        self.max_position_size = config.get('max_position_size', 10000)  # 最大仓位
        self.tick_size = config.get('tick_size', 0.01)  # 最小价格变动
        self.lot_size = config.get('lot_size', 0.001)  # 最小数量变动
        
        # 订单簿模拟
        self.order_book = {
            'bids': deque(maxlen=100),
            'asks': deque(maxlen=100),
            'last_update': datetime.now()
        }
        
        # 执行统计
        self.execution_times = deque(maxlen=1000)
        self.slippage_history = deque(maxlen=1000)
        self.fees_history = deque(maxlen=1000)
        
        # 风险管理
        self.daily_volume = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.max_daily_volume = config.get('max_daily_volume', 100000)
        self.max_daily_trades = config.get('max_daily_trades', 1000)
        self.max_daily_loss = config.get('max_daily_loss', 5000)
        
        # 性能监控
        self.start_time = datetime.now()
        self.last_metrics_update = datetime.now()
        
        logger.info("V12高频交易执行引擎初始化完成")
        logger.info(f"最大滑点: {self.max_slippage_bps}bps")
        logger.info(f"最大执行时间: {self.max_execution_time_ms}ms")
        logger.info(f"最大仓位: {self.max_position_size}")
    
    def start(self):
        """启动执行引擎"""
        if self.running:
            logger.warning("执行引擎已在运行")
            return
        
        self.running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        logger.info("V12高频交易执行引擎已启动")
    
    def stop(self):
        """停止执行引擎"""
        if not self.running:
            logger.warning("执行引擎未在运行")
            return
        
        self.running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        logger.info("V12高频交易执行引擎已停止")
    
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        try:
            # 风险检查
            if not self._risk_check(order):
                order.status = OrderStatus.REJECTED
                self.metrics.rejected_orders += 1
                logger.warning(f"订单被拒绝: {order.order_id}, 风险检查失败")
                return False
            
            # 订单验证
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                self.metrics.rejected_orders += 1
                logger.warning(f"订单被拒绝: {order.order_id}, 订单验证失败")
                return False
            
            # 添加到订单簿
            self.orders[order.order_id] = order
            order.status = OrderStatus.SUBMITTED
            order.submit_time = datetime.now()
            
            # 添加到执行队列
            self.execution_queue.put((order.submit_time, order))
            self.metrics.total_orders += 1
            
            logger.info(f"订单已提交: {order.order_id}, 类型: {order.order_type.value}, "
                       f"方向: {order.side.value}, 数量: {order.quantity}")
            return True
            
        except Exception as e:
            logger.error(f"提交订单失败: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"订单状态不允许取消: {order.status.value}")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.cancel_time = datetime.now()
            self.metrics.cancelled_orders += 1
            
            logger.info(f"订单已取消: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """获取执行指标"""
        # 计算成功率
        if self.metrics.total_orders > 0:
            self.metrics.success_rate = self.metrics.filled_orders / self.metrics.total_orders
        
        # 计算平均滑点
        if len(self.slippage_history) > 0:
            self.metrics.average_slippage = np.mean(list(self.slippage_history))
        
        # 计算平均执行时间
        if len(self.execution_times) > 0:
            self.metrics.average_execution_time = np.mean(list(self.execution_times))
        
        return self.metrics
    
    def _execution_loop(self):
        """执行循环"""
        logger.info("执行循环已启动")
        
        while self.running:
            try:
                # 获取下一个订单
                try:
                    _, order = self.execution_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 执行订单
                self._execute_order(order)
                
                # 更新订单簿
                self._update_order_book(order)
                
            except Exception as e:
                logger.error(f"执行循环错误: {e}")
                time.sleep(0.001)  # 1ms延迟
    
    def _execute_order(self, order: Order):
        """执行单个订单"""
        start_time = time.time()
        
        try:
            # 获取当前市场价格
            market_price = self._get_market_price(order.symbol)
            if market_price is None:
                order.status = OrderStatus.REJECTED
                self.metrics.rejected_orders += 1
                return
            
            # 计算执行价格
            execution_price = self._calculate_execution_price(order, market_price)
            if execution_price is None:
                order.status = OrderStatus.REJECTED
                self.metrics.rejected_orders += 1
                return
            
            # 计算滑点
            slippage = self._calculate_slippage(order, market_price, execution_price)
            
            # 滑点检查
            if abs(slippage) > self.max_slippage_bps / 10000:
                order.status = OrderStatus.REJECTED
                self.metrics.rejected_orders += 1
                logger.warning(f"订单滑点过大: {slippage:.4f}, 订单: {order.order_id}")
                return
            
            # 计算手续费
            fees = self._calculate_fees(order, execution_price)
            
            # 执行订单
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            order.fill_time = datetime.now()
            order.fees = fees
            order.slippage = slippage
            order.execution_cost = abs(slippage * order.quantity * execution_price) + fees
            
            # 更新指标
            self.metrics.filled_orders += 1
            self.metrics.total_volume += order.quantity
            self.metrics.total_fees += fees
            self.metrics.total_slippage += abs(slippage)
            self.metrics.execution_cost += order.execution_cost
            
            # 记录执行时间
            execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.execution_times.append(execution_time)
            self.slippage_history.append(slippage)
            self.fees_history.append(fees)
            
            # 更新日统计
            self.daily_volume += order.quantity
            self.daily_trades += 1
            
            logger.info(f"订单执行完成: {order.order_id}, 价格: {execution_price:.4f}, "
                       f"滑点: {slippage:.4f}, 执行时间: {execution_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"订单执行失败: {order.order_id}, 错误: {e}")
            order.status = OrderStatus.REJECTED
            self.metrics.rejected_orders += 1
    
    def _risk_check(self, order: Order) -> bool:
        """风险检查"""
        try:
            # 检查日交易量限制
            if self.daily_volume + order.quantity > self.max_daily_volume:
                logger.warning(f"超过日交易量限制: {self.daily_volume + order.quantity}")
                return False
            
            # 检查日交易次数限制
            if self.daily_trades >= self.max_daily_trades:
                logger.warning(f"超过日交易次数限制: {self.daily_trades}")
                return False
            
            # 检查日损失限制
            if self.daily_pnl < -self.max_daily_loss:
                logger.warning(f"超过日损失限制: {self.daily_pnl}")
                return False
            
            # 检查仓位大小
            if order.quantity > self.max_position_size:
                logger.warning(f"超过最大仓位限制: {order.quantity}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return False
    
    def _validate_order(self, order: Order) -> bool:
        """订单验证"""
        try:
            # 检查基本参数
            if order.quantity <= 0:
                logger.warning(f"订单数量无效: {order.quantity}")
                return False
            
            if order.side not in [OrderSide.BUY, OrderSide.SELL]:
                logger.warning(f"订单方向无效: {order.side}")
                return False
            
            if order.order_type not in [OrderType.MARKET, OrderType.LIMIT, OrderType.IOC, OrderType.FOK, OrderType.POST_ONLY]:
                logger.warning(f"订单类型无效: {order.order_type}")
                return False
            
            # 检查价格
            if order.order_type in [OrderType.LIMIT, OrderType.IOC, OrderType.FOK, OrderType.POST_ONLY]:
                if order.price is None or order.price <= 0:
                    logger.warning(f"限价单价格无效: {order.price}")
                    return False
                
                # 检查价格精度 - 放宽精度要求
                if abs(order.price % self.tick_size) > 1e-6:
                    # 自动调整到最近的tick
                    order.price = round(order.price / self.tick_size) * self.tick_size
            
            # 检查数量精度 - 放宽精度要求
            if abs(order.quantity % self.lot_size) > 1e-6:
                # 自动调整到最近的lot
                order.quantity = round(order.quantity / self.lot_size) * self.lot_size
            
            return True
            
        except Exception as e:
            logger.error(f"订单验证失败: {e}")
            return False
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """获取市场价格"""
        try:
            # 模拟获取市场价格
            # 实际实现中应该从订单簿或市场数据源获取
            base_price = 3000.0  # ETH价格基准
            volatility = 0.001  # 1%波动
            price_change = np.random.normal(0, volatility)
            market_price = base_price * (1 + price_change)
            
            return round(market_price, 2)
            
        except Exception as e:
            logger.error(f"获取市场价格失败: {e}")
            return None
    
    def _calculate_execution_price(self, order: Order, market_price: float) -> Optional[float]:
        """计算执行价格"""
        try:
            if order.order_type == OrderType.MARKET:
                # 市价单直接使用市场价格
                return market_price
            
            elif order.order_type in [OrderType.LIMIT, OrderType.IOC, OrderType.FOK]:
                # 限价单使用订单价格
                return order.price
            
            elif order.order_type == OrderType.POST_ONLY:
                # Post Only订单需要检查是否能立即成交
                if order.side == OrderSide.BUY and order.price < market_price:
                    return order.price
                elif order.side == OrderSide.SELL and order.price > market_price:
                    return order.price
                else:
                    return None  # 无法成交
            
            return market_price
            
        except Exception as e:
            logger.error(f"计算执行价格失败: {e}")
            return None
    
    def _calculate_slippage(self, order: Order, market_price: float, execution_price: float) -> float:
        """计算滑点"""
        try:
            if order.side == OrderSide.BUY:
                slippage = (execution_price - market_price) / market_price
            else:
                slippage = (market_price - execution_price) / market_price
            
            return slippage
            
        except Exception as e:
            logger.error(f"计算滑点失败: {e}")
            return 0.0
    
    def _calculate_fees(self, order: Order, execution_price: float) -> float:
        """计算手续费"""
        try:
            # 币安期货手续费: 0.02%
            fee_rate = 0.0002
            fees = order.quantity * execution_price * fee_rate
            return round(fees, 4)
            
        except Exception as e:
            logger.error(f"计算手续费失败: {e}")
            return 0.0
    
    def _update_order_book(self, order: Order):
        """更新订单簿"""
        try:
            if order.status == OrderStatus.FILLED:
                if order.side == OrderSide.BUY:
                    self.order_book['bids'].append({
                        'price': order.average_price,
                        'quantity': order.filled_quantity,
                        'timestamp': order.fill_time
                    })
                else:
                    self.order_book['asks'].append({
                        'price': order.average_price,
                        'quantity': order.filled_quantity,
                        'timestamp': order.fill_time
                    })
                
                self.order_book['last_update'] = datetime.now()
                
        except Exception as e:
            logger.error(f"更新订单簿失败: {e}")
    
    def generate_order_id(self) -> str:
        """生成订单ID"""
        return f"V12_{uuid.uuid4().hex[:8]}"
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float) -> Order:
        """创建市价单"""
        order_id = self.generate_order_id()
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, price: float) -> Order:
        """创建限价单"""
        order_id = self.generate_order_id()
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
    
    def create_ioc_order(self, symbol: str, side: OrderSide, quantity: float, price: float) -> Order:
        """创建IOC订单"""
        order_id = self.generate_order_id()
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.IOC,
            quantity=quantity,
            price=price
        )
    
    def create_fok_order(self, symbol: str, side: OrderSide, quantity: float, price: float) -> Order:
        """创建FOK订单"""
        order_id = self.generate_order_id()
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.FOK,
            quantity=quantity,
            price=price
        )
    
    def reset_daily_metrics(self):
        """重置日指标"""
        self.daily_volume = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        logger.info("日指标已重置")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_orders': self.metrics.total_orders,
            'filled_orders': self.metrics.filled_orders,
            'success_rate': self.metrics.success_rate,
            'average_execution_time_ms': self.metrics.average_execution_time,
            'average_slippage': self.metrics.average_slippage,
            'total_fees': self.metrics.total_fees,
            'total_execution_cost': self.metrics.execution_cost,
            'daily_volume': self.daily_volume,
            'daily_trades': self.daily_trades,
            'orders_per_second': self.metrics.total_orders / uptime if uptime > 0 else 0,
            'last_update': datetime.now().isoformat()
        }
