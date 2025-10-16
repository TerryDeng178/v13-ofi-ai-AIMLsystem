
import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, Any

class OrderType(Enum):
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    LMT = "LMT"  # Limit Order

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class OrderResult:
    def __init__(self, status: OrderStatus, fill_price: Optional[float] = None, 
                 fill_qty: float = 0.0, fee: float = 0.0, slippage_bps: float = 0.0,
                 reject_reason: Optional[str] = None):
        self.status = status
        self.fill_price = fill_price
        self.fill_qty = fill_qty
        self.fee = fee
        self.slippage_bps = slippage_bps
        self.reject_reason = reject_reason

class SimBroker:
    def __init__(self, fee_bps: float = 2.0, slip_bps_budget_frac: float = 0.33,
                 params: Optional[Dict[str, Any]] = None):
        self.fee_bps = fee_bps
        self.slip_bps_budget_frac = slip_bps_budget_frac
        self.params = params or {}
        
        # Execution parameters
        self.exec_params = self.params.get("execution", {})
        self.slippage_budget_check = self.exec_params.get("slippage_budget_check", True)
        self.max_slippage_bps = self.exec_params.get("max_slippage_bps", 10.0)
        self.reject_on_budget_exceeded = self.exec_params.get("reject_on_budget_exceeded", True)
        self.ioc_enabled = self.exec_params.get("ioc", True)
        self.fok_enabled = self.exec_params.get("fok", False)

    def check_slippage_budget(self, expected_slippage_bps: float, expected_reward: float) -> Tuple[bool, str]:
        """
        Check if slippage is within budget constraints.
        Returns (is_valid, reason)
        """
        if not self.slippage_budget_check:
            return True, ""
        
        # Check absolute slippage limit
        if expected_slippage_bps > self.max_slippage_bps:
            return False, f"Slippage {expected_slippage_bps:.2f}bps exceeds max {self.max_slippage_bps}bps"
        
        # Check slippage vs reward ratio
        if expected_reward > 0:
            slippage_fraction = expected_slippage_bps / (expected_reward * 10000)  # Convert reward to bps
            if slippage_fraction > self.slip_bps_budget_frac:
                return False, f"Slippage fraction {slippage_fraction:.3f} exceeds budget {self.slip_bps_budget_frac:.3f}"
        
        return True, ""

    def simulate_fill(self, side: int, qty_usd: float, price: float, atr: float, 
                     expected_reward: float, order_type: OrderType = OrderType.LMT,
                     limit_price: Optional[float] = None) -> OrderResult:
        """
        Simulate order fill with IOC/FOK logic and slippage budget checking.
        """
        # Estimate slippage
        slip_bps = min(10.0, 0.1 * (qty_usd / 100000.0) * max(1.0, 10000*atr/price))
        
        # Check slippage budget
        is_valid, reason = self.check_slippage_budget(slip_bps, expected_reward)
        if not is_valid and self.reject_on_budget_exceeded:
            return OrderResult(OrderStatus.REJECTED, reject_reason=reason)
        
        # Handle different order types
        if order_type == OrderType.IOC:
            if not self.ioc_enabled:
                return OrderResult(OrderStatus.REJECTED, reject_reason="IOC orders disabled")
            # IOC: Fill immediately or cancel
            fill_price = price * (1 + slip_bps/1e4 * (1 if side > 0 else -1))
            fill_qty = qty_usd  # Assume full fill for IOC
            
        elif order_type == OrderType.FOK:
            if not self.fok_enabled:
                return OrderResult(OrderStatus.REJECTED, reject_reason="FOK orders disabled")
            # FOK: Fill completely or reject
            if slip_bps > self.max_slippage_bps:
                return OrderResult(OrderStatus.REJECTED, 
                                 reject_reason=f"FOK: Insufficient liquidity, slippage {slip_bps:.2f}bps")
            fill_price = price * (1 + slip_bps/1e4 * (1 if side > 0 else -1))
            fill_qty = qty_usd  # Full fill for FOK
            
        else:  # LMT
            # Limit order: use limit price if provided, otherwise market
            if limit_price is not None:
                # Check if limit price is achievable
                market_price = price * (1 + slip_bps/1e4 * (1 if side > 0 else -1))
                if side > 0 and limit_price < market_price:
                    return OrderResult(OrderStatus.REJECTED, 
                                     reject_reason=f"Limit price {limit_price:.4f} below market {market_price:.4f}")
                elif side < 0 and limit_price > market_price:
                    return OrderResult(OrderStatus.REJECTED, 
                                     reject_reason=f"Limit price {limit_price:.4f} above market {market_price:.4f}")
                fill_price = limit_price
            else:
                fill_price = price * (1 + slip_bps/1e4 * (1 if side > 0 else -1))
            fill_qty = qty_usd
        
        # Calculate fee
        fee = abs(qty_usd) * self.fee_bps/1e4
        
        return OrderResult(
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee=fee,
            slippage_bps=slip_bps
        )

