"""
V12 同步系统综合测试
测试统一配置中心、组件通信总线、优化同步系统的集成功能
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v12_unified_config_center import V12UnifiedConfigCenter
from src.v12_component_communication_bus import V12ComponentCommunicationBus
from src.v12_optimization_synchronization_system import V12OptimizationSynchronizationSystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V12SynchronizationSystemTest:
    """V12同步系统综合测试"""
    
    def __init__(self):
        self.config_center = None
        self.communication_bus = None
        self.sync_system = None
        self.test_results = {}
        
    def setup_systems(self):
        """设置所有系统"""
        logger.info("设置V12同步系统...")
        
        try:
            # 创建统一配置中心
            self.config_center = V12UnifiedConfigCenter("config/v12_test_config.json")
            logger.info("✅ 统一配置中心创建成功")
            
            # 创建组件通信总线
            self.communication_bus = V12ComponentCommunicationBus()
            self.communication_bus.start()
            logger.info("✅ 组件通信总线启动成功")
            
            # 创建优化同步系统
            self.sync_system = V12OptimizationSynchronizationSystem()
            self.sync_system.start()
            logger.info("✅ 优化同步系统启动成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统设置失败: {e}")
            return False
    
    def test_config_management(self):
        """测试配置管理功能"""
        logger.info("测试配置管理功能...")
        
        try:
            # 测试配置更新
            ai_config = {
                "lstm": {
                    "input_size": 31,
                    "hidden_size": 256,
                    "learning_rate": 0.002
                },
                "transformer": {
                    "input_size": 31,
                    "d_model": 256,
                    "learning_rate": 0.001
                }
            }
            
            success = self.config_center.update_config("ai_models", ai_config)
            self.test_results['config_update'] = success
            
            # 测试配置获取
            config = self.config_center.get_config("ai_models")
            self.test_results['config_retrieval'] = len(config) > 0
            
            # 测试配置版本
            version = self.config_center.get_version()
            self.test_results['config_version'] = version > 0
            
            logger.info(f"✅ 配置管理测试完成 - 更新: {success}, 获取: {len(config)} 项, 版本: {version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置管理测试失败: {e}")
            return False
    
    def test_communication_system(self):
        """测试通信系统功能"""
        logger.info("测试通信系统功能...")
        
        try:
            # 设置消息回调
            received_messages = []
            
            def message_callback(topic: str, message: dict):
                received_messages.append((topic, message))
                logger.info(f"收到消息 - 主题: {topic}, 内容: {message}")
            
            # 订阅消息
            self.communication_bus.subscribe("test_topic", message_callback)
            self.communication_bus.subscribe("ai_update", message_callback)
            
            # 发布测试消息
            self.communication_bus.publish("test_topic", {
                "message": "测试消息",
                "timestamp": datetime.now().isoformat()
            })
            
            self.communication_bus.publish("ai_update", {
                "model_type": "lstm",
                "accuracy": 0.85,
                "status": "trained"
            }, priority=1)
            
            # 等待消息处理
            time.sleep(1.0)
            
            # 验证消息接收
            self.test_results['message_received'] = len(received_messages) >= 2
            self.test_results['message_count'] = len(received_messages)
            
            # 获取统计信息
            stats = self.communication_bus.get_statistics()
            self.test_results['bus_stats'] = stats
            
            logger.info(f"✅ 通信系统测试完成 - 收到消息: {len(received_messages)} 条")
            return True
            
        except Exception as e:
            logger.error(f"❌ 通信系统测试失败: {e}")
            return False
    
    def test_optimization_sync(self):
        """测试优化同步功能"""
        logger.info("测试优化同步功能...")
        
        try:
            # 模拟组件
            class MockComponent:
                def __init__(self, name):
                    self.name = name
                    self.parameters = {}
                    self.optimization_count = 0
                
                def update_parameters(self, params):
                    self.parameters.update(params)
                    self.optimization_count += 1
                    logger.info(f"组件 {self.name} 参数已更新: {params}")
                    return True
                
                def optimize(self, data):
                    self.parameters.update(data)
                    self.optimization_count += 1
                    logger.info(f"组件 {self.name} 优化完成: {data}")
                    return True
            
            # 注册组件
            ai_component = MockComponent("AI模型")
            signal_component = MockComponent("信号处理")
            
            self.sync_system.register_component("ai_model", ai_component)
            self.sync_system.register_component("signal_processing", signal_component)
            
            # 请求优化
            self.sync_system.request_optimization("ai_model", {
                "learning_rate": 0.003,
                "batch_size": 128,
                "priority": 1
            })
            
            self.sync_system.request_optimization("signal_processing", {
                "threshold": 0.5,
                "window_size": 30,
                "priority": 0
            })
            
            # 等待优化处理
            time.sleep(2.0)
            
            # 验证优化结果
            status = self.sync_system.get_component_status()
            self.test_results['component_status'] = status
            
            # 验证优化历史
            history = self.sync_system.get_optimization_history()
            self.test_results['optimization_history'] = len(history) >= 2
            
            # 获取统计信息
            stats = self.sync_system.get_statistics()
            self.test_results['sync_stats'] = stats
            
            logger.info(f"✅ 优化同步测试完成 - 组件状态: {len(status)} 个, 优化历史: {len(history)} 条")
            return True
            
        except Exception as e:
            logger.error(f"❌ 优化同步测试失败: {e}")
            return False
    
    def test_integrated_workflow(self):
        """测试集成工作流程"""
        logger.info("测试集成工作流程...")
        
        try:
            # 模拟完整的优化工作流程
            workflow_messages = []
            
            def workflow_callback(topic: str, message: dict):
                workflow_messages.append((topic, message))
                logger.info(f"工作流程消息 - 主题: {topic}, 内容: {message}")
            
            # 订阅工作流程消息
            self.communication_bus.subscribe("workflow", workflow_callback)
            self.communication_bus.subscribe("optimization", workflow_callback)
            
            # 1. 配置更新触发优化
            new_config = {
                "signal_processing": {
                    "quality_threshold": 0.4,
                    "confidence_threshold": 0.6
                }
            }
            
            self.config_center.update_config("signal_processing", new_config)
            self.communication_bus.publish("workflow", {
                "step": "config_updated",
                "component": "signal_processing",
                "timestamp": datetime.now().isoformat()
            })
            
            # 2. 请求优化
            self.sync_system.request_optimization("signal_processing", {
                "threshold": 0.4,
                "confidence": 0.6,
                "priority": 1
            })
            
            self.communication_bus.publish("optimization", {
                "step": "optimization_requested",
                "component": "signal_processing",
                "timestamp": datetime.now().isoformat()
            })
            
            # 3. 等待处理完成
            time.sleep(2.0)
            
            # 4. 发布完成消息
            self.communication_bus.publish("workflow", {
                "step": "optimization_completed",
                "component": "signal_processing",
                "timestamp": datetime.now().isoformat()
            })
            
            # 验证工作流程
            self.test_results['workflow_messages'] = len(workflow_messages) >= 3
            self.test_results['workflow_complete'] = any(
                msg[1].get('step') == 'optimization_completed' 
                for msg in workflow_messages
            )
            
            logger.info(f"✅ 集成工作流程测试完成 - 消息: {len(workflow_messages)} 条")
            return True
            
        except Exception as e:
            logger.error(f"❌ 集成工作流程测试失败: {e}")
            return False
    
    def cleanup_systems(self):
        """清理系统"""
        logger.info("清理V12同步系统...")
        
        try:
            if self.communication_bus:
                self.communication_bus.stop()
                logger.info("✅ 组件通信总线已停止")
            
            if self.sync_system:
                self.sync_system.stop()
                logger.info("✅ 优化同步系统已停止")
            
            logger.info("✅ 系统清理完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统清理失败: {e}")
            return False
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        try:
            report = {
                "test_timestamp": datetime.now().isoformat(),
                "test_results": self.test_results,
                "summary": {
                    "total_tests": len(self.test_results),
                    "passed_tests": sum(1 for result in self.test_results.values() if result is True),
                    "failed_tests": sum(1 for result in self.test_results.values() if result is False)
                }
            }
            
            # 保存报告
            os.makedirs("backtest_results", exist_ok=True)
            report_file = f"backtest_results/v12_synchronization_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 测试报告已保存: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成测试报告失败: {e}")
            return None


def main():
    """主测试函数"""
    logger.info("开始V12同步系统综合测试...")
    
    # 创建测试实例
    test = V12SynchronizationSystemTest()
    
    try:
        # 设置系统
        if not test.setup_systems():
            logger.error("❌ 系统设置失败，测试终止")
            return False
        
        # 执行测试
        test.test_config_management()
        test.test_communication_system()
        test.test_optimization_sync()
        test.test_integrated_workflow()
        
        # 生成报告
        report = test.generate_report()
        
        # 清理系统
        test.cleanup_systems()
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("V12同步系统综合测试完成")
        logger.info(f"测试结果: {test.test_results}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 V12同步系统综合测试成功完成！")
    else:
        logger.error("💥 V12同步系统综合测试失败！")
