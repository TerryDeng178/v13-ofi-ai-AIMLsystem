#!/usr/bin/env python3
"""
V10.0 增强实时客户端
支持3级加权OFI和深度学习信号的实时监控
"""

import asyncio
import json
import websockets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
import time

class V10EnhancedClient:
    """V10增强实时客户端"""
    
    def __init__(self, uri='ws://127.0.0.1:8765'):
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        
        # 数据存储
        self.market_data = []
        self.signals = []
        self.ofi_data = []
        
        # 统计信息
        self.stats = {
            "total_messages": 0,
            "total_signals": 0,
            "start_time": None,
            "last_update": None
        }
        
        # 实时监控
        self.monitoring = True
        self.update_interval = 1.0  # 1秒更新一次
        
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            self.stats["start_time"] = time.time()
            print(f"已连接到V10.0增强服务器: {self.uri}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
            
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("已断开连接")
            
    async def start_simulation(self):
        """启动市场模拟"""
        if not self.is_connected:
            print("未连接到服务器")
            return
            
        message = {
            "type": "start_simulation",
            "timestamp": time.time()
        }
        
        await self.websocket.send(json.dumps(message))
        print("已发送启动模拟请求")
        
    async def stop_simulation(self):
        """停止市场模拟"""
        if not self.is_connected:
            print("未连接到服务器")
            return
            
        message = {
            "type": "stop_simulation",
            "timestamp": time.time()
        }
        
        await self.websocket.send(json.dumps(message))
        print("已发送停止模拟请求")
        
    async def get_stats(self):
        """获取统计信息"""
        if not self.is_connected:
            print("未连接到服务器")
            return
            
        message = {
            "type": "get_stats",
            "timestamp": time.time()
        }
        
        await self.websocket.send(json.dumps(message))
        
    async def get_signals(self, limit: int = 100):
        """获取信号数据"""
        if not self.is_connected:
            print("未连接到服务器")
            return
            
        message = {
            "type": "get_signals",
            "limit": limit,
            "timestamp": time.time()
        }
        
        await self.websocket.send(json.dumps(message))
        
    async def listen(self):
        """监听服务器消息"""
        if not self.is_connected:
            print("未连接到服务器")
            return
            
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("服务器连接已关闭")
            self.is_connected = False
        except Exception as e:
            print(f"监听消息时出错: {e}")
            
    async def _handle_message(self, message: str):
        """处理服务器消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "welcome":
                print(f"服务器欢迎消息: {data.get('message')}")
                
            elif message_type == "config":
                print("已接收服务器配置")
                
            elif message_type == "simulation_started":
                print(f"模拟已启动: {data.get('message')}")
                
            elif message_type == "simulation_stopped":
                print(f"模拟已停止: {data.get('message')}")
                
            elif message_type == "market_data":
                await self._handle_market_data(data)
                
            elif message_type == "stats":
                await self._handle_stats(data)
                
            elif message_type == "signals":
                await self._handle_signals(data)
                
            elif message_type == "error":
                print(f"服务器错误: {data.get('message')}")
                
            else:
                print(f"未知消息类型: {message_type}")
                
        except json.JSONDecodeError:
            print("无效的JSON消息")
        except Exception as e:
            print(f"处理消息时出错: {e}")
            
    async def _handle_market_data(self, data: Dict):
        """处理市场数据"""
        market_data = data.get("data", {})
        timestamp = market_data.get("timestamp", 0)
        
        # 存储市场数据
        self.market_data.append({
            "timestamp": timestamp,
            "bid": market_data.get("bid", 0.0),
            "ask": market_data.get("ask", 0.0),
            "bid_sz": market_data.get("bid_sz", 0.0),
            "ask_sz": market_data.get("ask_sz", 0.0)
        })
        
        # 处理OFI数据
        ofi_data = market_data.get("ofi_data", {})
        if ofi_data:
            self.ofi_data.append({
                "timestamp": timestamp,
                "ofi": ofi_data.get("ofi", 0.0),
                "ofi_z": ofi_data.get("ofi_z", 0.0),
                "weighted_ofi": ofi_data.get("weighted_ofi", 0.0),
                "weighted_ofi_z": ofi_data.get("weighted_ofi_z", 0.0),
                "level_ofis": ofi_data.get("level_ofis", []),
                "level_zs": ofi_data.get("level_zs", [])
            })
            
        # 处理信号数据
        signal_result = market_data.get("signal_result", {})
        if signal_result and signal_result.get("signal_side") != 0:
            self.signals.append({
                "timestamp": timestamp,
                "signal_side": signal_result.get("signal_side", 0),
                "signal_strength": signal_result.get("signal_strength", 0.0),
                "confidence": signal_result.get("confidence", 0.0),
                "model_type": signal_result.get("model_type", "unknown"),
                "ofi_z": ofi_data.get("ofi_z", 0.0),
                "weighted_ofi_z": ofi_data.get("weighted_ofi_z", 0.0)
            })
            self.stats["total_signals"] += 1
            
        self.stats["total_messages"] += 1
        self.stats["last_update"] = time.time()
        
        # 实时显示
        if self.monitoring and len(self.market_data) % 10 == 0:
            await self._display_realtime_info()
            
    async def _handle_stats(self, data: Dict):
        """处理统计信息"""
        print("\n" + "="*50)
        print("V10.0 增强实时统计信息")
        print("="*50)
        
        simulation_stats = data.get("simulation_stats", {})
        ofi_stats = data.get("ofi_stats", {})
        
        print(f"模拟统计:")
        print(f"  总事件数: {simulation_stats.get('total_events', 0)}")
        print(f"  总信号数: {simulation_stats.get('total_signals', 0)}")
        print(f"  运行时间: {simulation_stats.get('start_time', 0)}")
        
        print(f"OFI统计:")
        print(f"  总信号数: {ofi_stats.get('total_signals', 0)}")
        print(f"  胜率: {ofi_stats.get('win_rate', 0.0):.2%}")
        print(f"  平均置信度: {ofi_stats.get('avg_confidence', 0.0):.3f}")
        print(f"  当前权重: {ofi_stats.get('current_weights', [])}")
        
        print(f"数据缓冲区:")
        print(f"  市场数据: {data.get('data_buffer_size', 0)}")
        print(f"  信号数据: {data.get('signal_buffer_size', 0)}")
        print("="*50)
        
    async def _handle_signals(self, data: Dict):
        """处理信号数据"""
        signals = data.get("signals", [])
        total_signals = data.get("total_signals", 0)
        
        print(f"\n信号数据 (共{total_signals}个信号):")
        for signal in signals[-10:]:  # 显示最近10个信号
            timestamp = signal.get("timestamp", 0)
            side = "多头" if signal.get("signal_side") > 0 else "空头"
            strength = signal.get("signal_strength", 0.0)
            confidence = signal.get("confidence", 0.0)
            model_type = signal.get("model_type", "unknown")
            
            print(f"  时间: {timestamp}, 方向: {side}, 强度: {strength:.3f}, "
                  f"置信度: {confidence:.3f}, 模型: {model_type}")
                  
    async def _display_realtime_info(self):
        """显示实时信息"""
        if not self.market_data:
            return
            
        latest_market = self.market_data[-1]
        latest_ofi = self.ofi_data[-1] if self.ofi_data else None
        latest_signal = self.signals[-1] if self.signals else None
        
        print(f"\n实时数据 (时间: {datetime.now().strftime('%H:%M:%S')}):")
        print(f"  买卖价: {latest_market['bid']:.2f} / {latest_market['ask']:.2f}")
        print(f"  买卖量: {latest_market['bid_sz']:.1f} / {latest_market['ask_sz']:.1f}")
        
        if latest_ofi:
            print(f"  OFI: {latest_ofi['ofi']:.2f}, OFI_Z: {latest_ofi['ofi_z']:.3f}")
            print(f"  加权OFI: {latest_ofi['weighted_ofi']:.2f}, 加权OFI_Z: {latest_ofi['weighted_ofi_z']:.3f}")
            
        if latest_signal:
            side = "多头" if latest_signal['signal_side'] > 0 else "空头"
            print(f"  最新信号: {side}, 强度: {latest_signal['signal_strength']:.3f}, "
                  f"置信度: {latest_signal['confidence']:.3f}")
                  
    def save_data(self, filename: str = None):
        """保存数据到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v10_realtime_data_{timestamp}.csv"
            
        # 合并数据
        df_market = pd.DataFrame(self.market_data)
        df_ofi = pd.DataFrame(self.ofi_data)
        df_signals = pd.DataFrame(self.signals)
        
        # 保存到CSV
        with pd.ExcelWriter(filename.replace('.csv', '.xlsx')) as writer:
            if not df_market.empty:
                df_market.to_excel(writer, sheet_name='market_data', index=False)
            if not df_ofi.empty:
                df_ofi.to_excel(writer, sheet_name='ofi_data', index=False)
            if not df_signals.empty:
                df_signals.to_excel(writer, sheet_name='signals', index=False)
                
        print(f"数据已保存到: {filename}")
        
    def plot_data(self):
        """绘制数据图表"""
        if not self.market_data or not self.ofi_data:
            print("没有足够的数据进行绘图")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('V10.0 增强实时数据分析', fontsize=16)
        
        # 价格数据
        df_market = pd.DataFrame(self.market_data)
        axes[0, 0].plot(df_market['timestamp'], df_market['bid'], label='Bid', alpha=0.7)
        axes[0, 0].plot(df_market['timestamp'], df_market['ask'], label='Ask', alpha=0.7)
        axes[0, 0].set_title('买卖价格')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # OFI数据
        df_ofi = pd.DataFrame(self.ofi_data)
        axes[0, 1].plot(df_ofi['timestamp'], df_ofi['ofi_z'], label='OFI_Z', alpha=0.7)
        axes[0, 1].plot(df_ofi['timestamp'], df_ofi['weighted_ofi_z'], label='Weighted OFI_Z', alpha=0.7)
        axes[0, 1].set_title('OFI指标')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 信号数据
        if self.signals:
            df_signals = pd.DataFrame(self.signals)
            signal_colors = ['red' if s > 0 else 'blue' for s in df_signals['signal_side']]
            axes[1, 0].scatter(df_signals['timestamp'], df_signals['signal_strength'], 
                             c=signal_colors, alpha=0.7)
            axes[1, 0].set_title('信号强度')
            axes[1, 0].grid(True)
            
        # 置信度数据
        if self.signals:
            axes[1, 1].plot(df_signals['timestamp'], df_signals['confidence'], 
                           marker='o', markersize=3, alpha=0.7)
            axes[1, 1].set_title('信号置信度')
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        plt.show()
        
    async def run_interactive(self):
        """运行交互式客户端"""
        print("V10.0 增强实时客户端启动")
        print("可用命令:")
        print("  start - 启动市场模拟")
        print("  stop - 停止市场模拟")
        print("  stats - 获取统计信息")
        print("  signals - 获取信号数据")
        print("  save - 保存数据")
        print("  plot - 绘制图表")
        print("  quit - 退出")
        
        # 启动监听任务
        listen_task = asyncio.create_task(self.listen())
        
        try:
            while True:
                command = input("\n请输入命令: ").strip().lower()
                
                if command == "start":
                    await self.start_simulation()
                elif command == "stop":
                    await self.stop_simulation()
                elif command == "stats":
                    await self.get_stats()
                elif command == "signals":
                    await self.get_signals()
                elif command == "save":
                    self.save_data()
                elif command == "plot":
                    self.plot_data()
                elif command == "quit":
                    break
                else:
                    print("未知命令")
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            listen_task.cancel()
            await self.disconnect()
            
    async def run_automated(self, duration: int = 60):
        """运行自动化客户端"""
        print(f"V10.0 增强自动化客户端启动 (运行{duration}秒)")
        
        # 启动模拟
        await self.start_simulation()
        
        # 启动监听任务
        listen_task = asyncio.create_task(self.listen())
        
        try:
            # 运行指定时间
            await asyncio.sleep(duration)
            
            # 停止模拟
            await self.stop_simulation()
            
            # 获取最终统计
            await self.get_stats()
            
            # 保存数据
            self.save_data()
            
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            listen_task.cancel()
            await self.disconnect()
            
            # 显示最终统计
            print(f"\n最终统计:")
            print(f"  总消息数: {self.stats['total_messages']}")
            print(f"  总信号数: {self.stats['total_signals']}")
            print(f"  市场数据: {len(self.market_data)}")
            print(f"  OFI数据: {len(self.ofi_data)}")
            print(f"  信号数据: {len(self.signals)}")
