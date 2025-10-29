#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景配置管理服务
提供HTTP接口和定时任务，用于管理场景配置的热加载

功能：
1. HTTP API接口
2. 定时配置检查
3. 配置健康检查
4. 配置版本管理
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import schedule

logger = logging.getLogger(__name__)

class ScenarioConfigService:
    """场景配置管理服务"""
    
    def __init__(self, strategy_manager, config_loader, 
                 host: str = '0.0.0.0', port: int = 8080,
                 enable_http: bool = True, enable_scheduler: bool = True):
        """
        初始化配置服务
        
        Args:
            strategy_manager: StrategyModeManager实例
            config_loader: ScenarioConfigLoader实例
            host: HTTP服务主机
            port: HTTP服务端口
            enable_http: 是否启用HTTP接口
            enable_scheduler: 是否启用定时任务
        """
        self.strategy_manager = strategy_manager
        self.config_loader = config_loader
        self.host = host
        self.port = port
        self.enable_http = enable_http
        self.enable_scheduler = enable_scheduler
        
        # HTTP应用
        self.app = None
        self.http_thread = None
        
        # 定时任务
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # 健康检查
        self.health_status = {
            'status': 'healthy',
            'last_check': datetime.now().isoformat(),
            'config_version': 'unknown',
            'scenarios_count': 0
        }
        
        # 初始化
        if self.enable_http:
            self._setup_http_server()
        
        if self.enable_scheduler:
            self._setup_scheduler()
    
    def _setup_http_server(self):
        """设置HTTP服务器"""
        self.app = Flask(__name__)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify(self.health_status)
        
        @self.app.route('/config/status', methods=['GET'])
        def get_config_status():
            """获取配置状态"""
            status = self.config_loader.get_config_status()
            return jsonify(status)
        
        @self.app.route('/config/reload', methods=['POST'])
        def reload_config():
            """手动重载配置"""
            try:
                config_path = request.json.get('config_path') if request.json else None
                success = self.config_loader.reload_config(config_path)
                
                return jsonify({
                    'success': success,
                    'message': '配置重载成功' if success else '配置重载失败',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'配置重载失败: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/config/rollback', methods=['POST'])
        def rollback_config():
            """回滚配置"""
            try:
                steps = request.json.get('steps', 1) if request.json else 1
                success = self.config_loader.rollback_config(steps)
                
                return jsonify({
                    'success': success,
                    'message': '配置回滚成功' if success else '配置回滚失败',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'配置回滚失败: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/scenarios', methods=['GET'])
        def get_scenarios():
            """获取场景参数"""
            try:
                scenario = request.args.get('scenario')
                side = request.args.get('side', 'long')
                
                if scenario:
                    params = self.strategy_manager.get_params_for_scenario(scenario, side)
                    return jsonify(params)
                else:
                    # 返回所有场景
                    scenarios = {}
                    for sc in ['A_H', 'A_L', 'Q_H', 'Q_L']:
                        scenarios[sc] = {
                            'long': self.strategy_manager.get_params_for_scenario(sc, 'long'),
                            'short': self.strategy_manager.get_params_for_scenario(sc, 'short')
                        }
                    return jsonify(scenarios)
            except Exception as e:
                return jsonify({
                    'error': f'获取场景参数失败: {str(e)}'
                }), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """获取Prometheus指标"""
            try:
                metrics = self.strategy_manager.get_metrics()
                return jsonify(metrics)
            except Exception as e:
                return jsonify({
                    'error': f'获取指标失败: {str(e)}'
                }), 500
        
        logger.info(f"✅ HTTP服务器已设置: http://{self.host}:{self.port}")
    
    def _setup_scheduler(self):
        """设置定时任务"""
        # 每分钟检查配置健康状态
        schedule.every(1).minutes.do(self._health_check)
        
        # 每小时检查配置文件变化
        schedule.every(1).hours.do(self._periodic_config_check)
        
        logger.info("✅ 定时任务已设置")
    
    def _health_check(self):
        """健康检查"""
        try:
            self.health_status.update({
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'config_version': self.strategy_manager.scenario_config_version,
                'scenarios_count': len(self.strategy_manager.current_params_by_scenario)
            })
            
            logger.debug("✅ 健康检查通过")
            
        except Exception as e:
            self.health_status.update({
                'status': 'unhealthy',
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            })
            
            logger.error(f"❌ 健康检查失败: {e}")
    
    def _periodic_config_check(self):
        """定时配置检查"""
        try:
            # 检查配置文件是否需要重载
            if self.config_loader.config_path.exists():
                success = self.config_loader.reload_config()
                if success:
                    logger.info("✅ 定时配置检查完成")
                else:
                    logger.warning("⚠️ 定时配置检查失败")
            else:
                logger.warning(f"⚠️ 配置文件不存在: {self.config_loader.config_path}")
                
        except Exception as e:
            logger.error(f"❌ 定时配置检查失败: {e}")
    
    def _run_scheduler(self):
        """运行定时任务"""
        self.scheduler_running = True
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"定时任务执行失败: {e}")
                time.sleep(5)
    
    def _run_http_server(self):
        """运行HTTP服务器"""
        try:
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"HTTP服务器运行失败: {e}")
    
    def start(self):
        """启动服务"""
        logger.info("🚀 启动场景配置管理服务...")
        
        # 启动HTTP服务器
        if self.enable_http:
            self.http_thread = threading.Thread(
                target=self._run_http_server,
                daemon=True
            )
            self.http_thread.start()
            logger.info(f"✅ HTTP服务器已启动: http://{self.host}:{self.port}")
        
        # 启动定时任务
        if self.enable_scheduler:
            self.scheduler_thread = threading.Thread(
                target=self._run_scheduler,
                daemon=True
            )
            self.scheduler_thread.start()
            logger.info("✅ 定时任务已启动")
        
        logger.info("🎉 场景配置管理服务启动完成")
    
    def stop(self):
        """停止服务"""
        logger.info("🛑 停止场景配置管理服务...")
        
        # 停止定时任务
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # 停止HTTP服务器
        if self.http_thread:
            self.http_thread.join(timeout=5)
        
        logger.info("✅ 场景配置管理服务已停止")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'http_enabled': self.enable_http,
            'scheduler_enabled': self.enable_scheduler,
            'scheduler_running': self.scheduler_running,
            'health_status': self.health_status,
            'endpoints': {
                'health': f'http://{self.host}:{self.port}/health',
                'config_status': f'http://{self.host}:{self.port}/config/status',
                'config_reload': f'http://{self.host}:{self.port}/config/reload',
                'config_rollback': f'http://{self.host}:{self.port}/config/rollback',
                'scenarios': f'http://{self.host}:{self.port}/scenarios',
                'metrics': f'http://{self.host}:{self.port}/metrics'
            }
        }

# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 模拟策略管理器和配置加载器
    class MockStrategyManager:
        def __init__(self):
            self.current_params_by_scenario = {}
            self.scenario_config_version = 'unknown'
        
        def load_scenario_params(self, config_data):
            self.current_params_by_scenario = config_data.get('scenarios', {})
            self.scenario_config_version = config_data.get('version', 'unknown')
            return True
        
        def get_params_for_scenario(self, scenario, side):
            return {'scenario': scenario, 'side': side, 'Z_HI': 2.5}
        
        def get_metrics(self):
            return {'test': 'metrics'}
    
    class MockConfigLoader:
        def __init__(self):
            self.config_path = Path("test_config.yaml")
        
        def get_config_status(self):
            return {'status': 'ok'}
        
        def reload_config(self, config_path=None):
            return True
        
        def rollback_config(self, steps=1):
            return True
    
    # 创建服务
    strategy_manager = MockStrategyManager()
    config_loader = MockConfigLoader()
    
    service = ScenarioConfigService(
        strategy_manager=strategy_manager,
        config_loader=config_loader,
        enable_http=False,  # 测试时不启动HTTP服务
        enable_scheduler=False  # 测试时不启动定时任务
    )
    
    # 获取服务状态
    status = service.get_service_status()
    print(f"服务状态: {json.dumps(status, indent=2, ensure_ascii=False)}")



