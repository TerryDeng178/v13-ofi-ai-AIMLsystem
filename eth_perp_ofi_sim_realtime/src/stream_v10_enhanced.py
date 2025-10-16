#!/usr/bin/env python3
"""
V10.0 增强实时流处理器
集成深度学习模型和3级加权OFI的实时数据处理
"""

import asyncio
import json
import websockets
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .ofi_v10_enhanced import V10EnhancedOFI
from .sim import MarketSimulator

class V10EnhancedWSHub:
    """V10增强WebSocket Hub"""
    
    def __init__(self, host='127.0.0.1', port=8765, config: Dict = None):
        self.host = host
        self.port = port
        self.clients = set()
        self.config = config or {}
        
        # V10增强OFI计算器
        self.ofi_calculator = V10EnhancedOFI(
            micro_window_ms=self.config.get("ofi", {}).get("micro_window_ms", 100),
            z_window_seconds=self.config.get("ofi", {}).get("z_window_seconds", 900),
            levels=3,
            weights=[0.5, 0.3, 0.2]
        )
        
        # 市场模拟器
        self.simulator = None
        self.is_running = False
        
        # 实时数据缓存
        self.data_buffer = []
        self.signal_buffer = []
        
        # 性能统计
        self.stats = {
            "total_events": 0,
            "total_signals": 0,
            "start_time": None,
            "last_update": None
        }
        
    async def handler(self, websocket, path):
        """WebSocket连接处理器"""
        self.clients.add(websocket)
        print(f"客户端连接: {websocket.remote_address}")
        
        try:
            # 发送欢迎消息
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "V10.0 增强实时流处理器已连接",
                "timestamp": time.time()
            }))
            
            # 发送初始配置
            await websocket.send(json.dumps({
                "type": "config",
                "config": self.config,
                "timestamp": time.time()
            }))
            
            # 保持连接
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "无效的JSON格式",
                        "timestamp": time.time()
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"处理消息失败: {str(e)}",
                        "timestamp": time.time()
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"客户端断开连接: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
            
    async def _handle_client_message(self, websocket, data: Dict):
        """处理客户端消息"""
        message_type = data.get("type")
        
        if message_type == "start_simulation":
            await self._start_simulation(websocket, data)
        elif message_type == "stop_simulation":
            await self._stop_simulation(websocket, data)
        elif message_type == "get_stats":
            await self._send_stats(websocket)
        elif message_type == "get_signals":
            await self._send_signals(websocket, data)
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"未知消息类型: {message_type}",
                "timestamp": time.time()
            }))
            
    async def _start_simulation(self, websocket, data: Dict):
        """启动市场模拟"""
        if self.is_running:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "模拟已在运行中",
                "timestamp": time.time()
            }))
            return
            
        try:
            # 创建市场模拟器
            self.simulator = MarketSimulator(self.config)
            self.is_running = True
            self.stats["start_time"] = time.time()
            
            await websocket.send(json.dumps({
                "type": "simulation_started",
                "message": "市场模拟已启动",
                "timestamp": time.time()
            }))
            
            # 启动模拟循环
            asyncio.create_task(self._simulation_loop())
            
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"启动模拟失败: {str(e)}",
                "timestamp": time.time()
            }))
            
    async def _stop_simulation(self, websocket, data: Dict):
        """停止市场模拟"""
        if not self.is_running:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "模拟未在运行",
                "timestamp": time.time()
            }))
            return
            
        self.is_running = False
        self.simulator = None
        
        await websocket.send(json.dumps({
            "type": "simulation_stopped",
            "message": "市场模拟已停止",
            "timestamp": time.time()
        }))
        
    async def _simulation_loop(self):
        """模拟循环"""
        if not self.simulator:
            return
            
        try:
            for events in self.simulator.stream(realtime=True, dt_ms=10):
                if not self.is_running:
                    break
                    
                # 处理事件
                for event in events:
                    await self._process_event(event)
                    
                # 发送数据给所有客户端
                if self.clients:
                    await self._broadcast_data()
                    
        except Exception as e:
            print(f"模拟循环错误: {e}")
        finally:
            self.is_running = False
            
    async def _process_event(self, event: Dict):
        """处理单个事件"""
        self.stats["total_events"] += 1
        
        # 处理不同类型的事件
        if event["type"] == "best":
            await self._process_best_event(event)
        elif event["type"] == "trade":
            await self._process_trade_event(event)
        elif event["type"] in ["l2_add", "l2_cancel"]:
            await self._process_l2_event(event)
            
    async def _process_best_event(self, event: Dict):
        """处理最优买卖价事件"""
        self.ofi_calculator.on_best(
            event["t"], event["bid"], event["bid_sz"], 
            event["ask"], event["ask_sz"]
        )
        
        # 计算OFI
        ofi_data = self.ofi_calculator.read()
        if ofi_data:
            # 创建特征
            market_data = {
                "bid": event["bid"],
                "ask": event["ask"],
                "bid_sz": event["bid_sz"],
                "ask_sz": event["ask_sz"],
                "spread": event["ask"] - event["bid"],
                "mid_price": (event["bid"] + event["ask"]) / 2
            }
            
            features = self.ofi_calculator.create_features(ofi_data, market_data)
            
            # 预测信号
            signal_result = self.ofi_calculator.predict_signal(features)
            
            # 保存数据
            data_point = {
                "timestamp": event["t"],
                "type": "best",
                "bid": event["bid"],
                "ask": event["ask"],
                "bid_sz": event["bid_sz"],
                "ask_sz": event["ask_sz"],
                "ofi_data": ofi_data,
                "signal_result": signal_result,
                "features": features.tolist()
            }
            
            self.data_buffer.append(data_point)
            
            # 如果有信号，保存到信号缓冲区
            if signal_result["signal_side"] != 0:
                self.signal_buffer.append({
                    "timestamp": event["t"],
                    "signal_side": signal_result["signal_side"],
                    "signal_strength": signal_result["signal_strength"],
                    "confidence": signal_result["confidence"],
                    "model_type": signal_result["model_type"],
                    "ofi_z": ofi_data["ofi_z"],
                    "weighted_ofi_z": ofi_data["weighted_ofi_z"]
                })
                self.stats["total_signals"] += 1
                
    async def _process_trade_event(self, event: Dict):
        """处理交易事件"""
        # 这里可以添加交易相关的处理逻辑
        pass
        
    async def _process_l2_event(self, event: Dict):
        """处理L2订单簿事件"""
        self.ofi_calculator.on_l2(
            event["t"], event["type"], event["side"], 
            event["price"], event["qty"]
        )
        
    async def _broadcast_data(self):
        """广播数据给所有客户端"""
        if not self.data_buffer:
            return
            
        # 获取最新的数据点
        latest_data = self.data_buffer[-1]
        
        # 准备广播数据
        broadcast_data = {
            "type": "market_data",
            "timestamp": latest_data["timestamp"],
            "data": latest_data,
            "stats": self.stats
        }
        
        # 广播给所有客户端
        await self.broadcast(broadcast_data)
        
        # 清空缓冲区（保留最新的一些数据）
        if len(self.data_buffer) > 100:
            self.data_buffer = self.data_buffer[-50:]
            
    async def _send_stats(self, websocket):
        """发送统计信息"""
        ofi_stats = self.ofi_calculator.get_statistics()
        
        stats_data = {
            "type": "stats",
            "timestamp": time.time(),
            "simulation_stats": self.stats,
            "ofi_stats": ofi_stats,
            "data_buffer_size": len(self.data_buffer),
            "signal_buffer_size": len(self.signal_buffer)
        }
        
        await websocket.send(json.dumps(stats_data))
        
    async def _send_signals(self, websocket, data: Dict):
        """发送信号数据"""
        limit = data.get("limit", 100)
        signals = list(self.signal_buffer)[-limit:]
        
        signals_data = {
            "type": "signals",
            "timestamp": time.time(),
            "signals": signals,
            "total_signals": len(self.signal_buffer)
        }
        
        await websocket.send(json.dumps(signals_data))
        
    async def broadcast(self, msg: dict):
        """广播消息给所有客户端"""
        if not self.clients:
            return
            
        data = json.dumps(msg)
        await asyncio.gather(
            *[c.send(data) for c in list(self.clients)], 
            return_exceptions=True
        )
        
    async def serve(self):
        """启动WebSocket服务器"""
        async with websockets.serve(self.handler, self.host, self.port):
            print(f'V10.0 增强WebSocket服务器启动: ws://{self.host}:{self.port}')
            print(f'支持的功能: 3级加权OFI, 深度学习信号生成, 实时优化')
            await asyncio.Future()
