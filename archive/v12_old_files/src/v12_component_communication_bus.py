"""
V12 组件通信总线
实现组件间消息传递机制，解决组件间通信问题
"""

import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V12ComponentCommunicationBus:
    """V12组件通信总线 - 实现组件间消息传递机制"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.subscribers = {}
        self.running = False
        self.processor_thread = None
        self.lock = threading.Lock()
        self.message_count = 0
        self.start_time = datetime.now()
        
        logger.info("V12组件通信总线初始化完成")
    
    def start(self):
        """启动消息处理器"""
        if not self.running:
            self.running = True
            self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
            self.processor_thread.start()
            logger.info("消息处理器已启动")
    
    def stop(self):
        """停止消息处理器"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        logger.info("消息处理器已停止")
    
    def publish(self, topic: str, message: Dict[str, Any], priority: int = 0) -> bool:
        """
        发布消息到指定主题
        
        Args:
            topic: 主题名称
            message: 消息内容
            priority: 优先级 (0=普通, 1=高, 2=紧急)
            
        Returns:
            是否发布成功
        """
        try:
            message_data = {
                'topic': topic,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'priority': priority,
                'message_id': self.message_count
            }
            
            self.message_queue.put(message_data, timeout=1.0)
            self.message_count += 1
            
            logger.debug(f"消息发布成功 - 主题: {topic}, 优先级: {priority}")
            return True
            
        except queue.Full:
            logger.warning(f"消息队列已满，丢弃消息 - 主题: {topic}")
            return False
        except Exception as e:
            logger.error(f"消息发布失败: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable[[str, Dict], None]) -> bool:
        """
        订阅主题消息
        
        Args:
            topic: 主题名称
            callback: 回调函数
            
        Returns:
            是否订阅成功
        """
        try:
            with self.lock:
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                
                self.subscribers[topic].append(callback)
                logger.info(f"订阅成功 - 主题: {topic}")
                return True
                
        except Exception as e:
            logger.error(f"订阅失败: {e}")
            return False
    
    def unsubscribe(self, topic: str, callback: Callable[[str, Dict], None]) -> bool:
        """
        取消订阅
        
        Args:
            topic: 主题名称
            callback: 回调函数
            
        Returns:
            是否取消订阅成功
        """
        try:
            with self.lock:
                if topic in self.subscribers and callback in self.subscribers[topic]:
                    self.subscribers[topic].remove(callback)
                    logger.info(f"取消订阅成功 - 主题: {topic}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False
    
    def _process_messages(self):
        """处理消息队列"""
        logger.info("消息处理器开始运行")
        
        while self.running:
            try:
                # 获取消息，设置超时避免阻塞
                message_data = self.message_queue.get(timeout=1.0)
                
                # 处理消息
                self._handle_message(message_data)
                
                # 标记任务完成
                self.message_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"消息处理失败: {e}")
                continue
        
        logger.info("消息处理器已停止")
    
    def _handle_message(self, message_data: Dict[str, Any]):
        """处理单个消息"""
        try:
            topic = message_data['topic']
            message = message_data['message']
            timestamp = message_data['timestamp']
            priority = message_data['priority']
            message_id = message_data['message_id']
            
            # 记录消息处理
            logger.debug(f"处理消息 - ID: {message_id}, 主题: {topic}, 优先级: {priority}")
            
            # 通知订阅者
            with self.lock:
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        try:
                            callback(topic, message)
                        except Exception as e:
                            logger.error(f"回调函数执行失败: {e}")
                            
        except Exception as e:
            logger.error(f"消息处理失败: {e}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.message_queue.qsize()
    
    def get_subscriber_count(self, topic: str = None) -> int:
        """获取订阅者数量"""
        with self.lock:
            if topic is None:
                return sum(len(subscribers) for subscribers in self.subscribers.values())
            else:
                return len(self.subscribers.get(topic, []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'running': self.running,
            'message_count': self.message_count,
            'queue_size': self.get_queue_size(),
            'subscriber_count': self.get_subscriber_count(),
            'uptime_seconds': uptime,
            'topics': list(self.subscribers.keys()),
            'start_time': self.start_time.isoformat()
        }
    
    def clear_queue(self):
        """清空消息队列"""
        try:
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
            logger.info("消息队列已清空")
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"清空队列失败: {e}")


def test_v12_component_communication_bus():
    """测试V12组件通信总线"""
    logger.info("开始测试V12组件通信总线...")
    
    # 创建通信总线
    bus = V12ComponentCommunicationBus()
    
    # 启动消息处理器
    bus.start()
    
    # 测试消息回调
    def ai_model_callback(topic: str, message: Dict):
        logger.info(f"AI模型收到消息 - 主题: {topic}, 内容: {message}")
    
    def signal_processing_callback(topic: str, message: Dict):
        logger.info(f"信号处理收到消息 - 主题: {topic}, 内容: {message}")
    
    # 订阅消息
    bus.subscribe("ai_model_update", ai_model_callback)
    bus.subscribe("signal_processing", signal_processing_callback)
    
    # 发布测试消息
    bus.publish("ai_model_update", {
        "model_type": "lstm",
        "accuracy": 0.85,
        "status": "trained"
    }, priority=1)
    
    bus.publish("signal_processing", {
        "signal_count": 100,
        "quality_score": 0.75,
        "timestamp": datetime.now().isoformat()
    })
    
    # 等待消息处理
    time.sleep(0.5)
    
    # 获取统计信息
    stats = bus.get_statistics()
    logger.info(f"通信总线统计信息: {stats}")
    
    # 停止消息处理器
    bus.stop()
    
    logger.info("V12组件通信总线测试完成")


if __name__ == "__main__":
    test_v12_component_communication_bus()
