"""
Grafana仪表盘动态生成器

基于统一配置系统动态生成Grafana仪表盘JSON配置
支持从配置模板和统一配置生成完整的仪表盘配置
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from src.grafana_config import GrafanaConfigLoader

logger = logging.getLogger(__name__)

class GrafanaDashboardGenerator:
    """Grafana仪表盘生成器"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.grafana_loader = GrafanaConfigLoader(config_loader)
    
    def generate_strategy_mode_dashboard(self) -> Dict[str, Any]:
        """
        生成策略模式仪表盘配置
        
        Returns:
            Dict[str, Any]: 仪表盘JSON配置
        """
        try:
            # 获取仪表盘配置
            dashboard_config = self.grafana_loader.get_dashboard_config('strategy_mode')
            if not dashboard_config:
                logger.warning("Strategy mode dashboard config not found, using defaults")
                from src.grafana_config import GrafanaDashboardConfig
                dashboard_config = GrafanaDashboardConfig(
                    uid="strategy-mode-overview",
                    title="Strategy Mode Overview",
                    description="V1核心面板：策略模式切换监控仪表盘",
                    tags=["strategy", "mode", "monitoring"],
                    timezone="Asia/Hong_Kong",
                    refresh="30s",
                    time_range="6h"
                )
            
            # 获取变量配置
            variables = self.grafana_loader.generate_variables_json()
            
            # 生成仪表盘配置
            dashboard = {
                "dashboard": {
                    "id": None,
                    "uid": dashboard_config.uid,
                    "title": dashboard_config.title,
                    "description": dashboard_config.description,
                    "tags": dashboard_config.tags,
                    "timezone": dashboard_config.timezone,
                    "time": {
                        "from": f"now-{dashboard_config.time_range}",
                        "to": "now"
                    },
                    "refresh": dashboard_config.refresh,
                    "templating": {
                        "list": variables
                    },
                    "annotations": {
                        "list": [
                            {
                                "name": "Mode Switches",
                                "datasource": "Prometheus",
                                "enable": True,
                                "expr": "strategy_mode_transitions_total{env=\"$env\",symbol=~\"$symbol\"}",
                                "iconColor": "rgba(0, 211, 255, 1)",
                                "titleFormat": "Mode Switch",
                                "textFormat": "{{ .Labels.reason }}"
                            }
                        ]
                    },
                    "panels": self._generate_strategy_mode_panels()
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating strategy mode dashboard: {e}")
            return {}
    
    def generate_performance_dashboard(self) -> Dict[str, Any]:
        """
        生成性能仪表盘配置
        
        Returns:
            Dict[str, Any]: 仪表盘JSON配置
        """
        try:
            # 获取仪表盘配置
            dashboard_config = self.grafana_loader.get_dashboard_config('performance')
            if not dashboard_config:
                logger.warning("Performance dashboard config not found, using defaults")
                dashboard_config = self.grafana_loader.GrafanaDashboardConfig(
                    uid="strategy-performance",
                    title="Strategy Performance",
                    description="策略性能监控仪表盘",
                    tags=["strategy", "performance", "monitoring"],
                    timezone="Asia/Hong_Kong",
                    refresh="30s",
                    time_range="6h"
                )
            
            # 获取变量配置
            variables = self.grafana_loader.generate_variables_json()
            
            # 生成仪表盘配置
            dashboard = {
                "dashboard": {
                    "id": None,
                    "uid": dashboard_config.uid,
                    "title": dashboard_config.title,
                    "description": dashboard_config.description,
                    "tags": dashboard_config.tags,
                    "timezone": dashboard_config.timezone,
                    "time": {
                        "from": f"now-{dashboard_config.time_range}",
                        "to": "now"
                    },
                    "refresh": dashboard_config.refresh,
                    "templating": {
                        "list": variables
                    },
                    "panels": self._generate_performance_panels()
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return {}
    
    def generate_alerts_dashboard(self) -> Dict[str, Any]:
        """
        生成告警仪表盘配置
        
        Returns:
            Dict[str, Any]: 仪表盘JSON配置
        """
        try:
            # 获取仪表盘配置
            dashboard_config = self.grafana_loader.get_dashboard_config('alerts')
            if not dashboard_config:
                logger.warning("Alerts dashboard config not found, using defaults")
                dashboard_config = self.grafana_loader.GrafanaDashboardConfig(
                    uid="strategy-alerts",
                    title="Strategy Alerts",
                    description="策略告警监控仪表盘",
                    tags=["strategy", "alerts", "monitoring"],
                    timezone="Asia/Hong_Kong",
                    refresh="30s",
                    time_range="6h"
                )
            
            # 获取变量配置
            variables = self.grafana_loader.generate_variables_json()
            
            # 生成仪表盘配置
            dashboard = {
                "dashboard": {
                    "id": None,
                    "uid": dashboard_config.uid,
                    "title": dashboard_config.title,
                    "description": dashboard_config.description,
                    "tags": dashboard_config.tags,
                    "timezone": dashboard_config.timezone,
                    "time": {
                        "from": f"now-{dashboard_config.time_range}",
                        "to": "now"
                    },
                    "refresh": dashboard_config.refresh,
                    "templating": {
                        "list": variables
                    },
                    "panels": self._generate_alerts_panels()
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating alerts dashboard: {e}")
            return {}
    
    def _generate_strategy_mode_panels(self) -> List[Dict[str, Any]]:
        """生成策略模式面板配置"""
        return [
            # Panel 1: 当前模式状态
            {
                "id": 1,
                "title": "Current Mode",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "avg without(instance,pod) (strategy_mode_active{env=\"$env\",symbol=~\"$symbol\"})",
                        "legendFormat": "Mode",
                        "datasource": "Prometheus"
                    }
                ],
                "options": {
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"]
                    },
                    "text": {
                        "valueSize": 72
                    },
                    "colorMode": "value",
                    "graphMode": "none"
                },
                "fieldConfig": {
                    "overrides": [
                        {
                            "matcher": {"id": "byValue", "options": "0"},
                            "properties": [
                                {"id": "displayName", "value": "Quiet"},
                                {"id": "color", "value": {"mode": "fixed", "fixedColor": "blue"}}
                            ]
                        },
                        {
                            "matcher": {"id": "byValue", "options": "1"},
                            "properties": [
                                {"id": "displayName", "value": "Active"},
                                {"id": "color", "value": {"mode": "fixed", "fixedColor": "green"}}
                            ]
                        }
                    ]
                }
            },
            # Panel 2: 最后切换距今
            {
                "id": 2,
                "title": "Last Switch Duration",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                "targets": [
                    {
                        "expr": "time() - max without(instance,pod) (strategy_mode_last_change_timestamp{env=\"$env\",symbol=~\"$symbol\"})",
                        "legendFormat": "Duration",
                        "datasource": "Prometheus"
                    }
                ],
                "options": {
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"]
                    },
                    "text": {
                        "valueSize": 48
                    },
                    "colorMode": "value",
                    "graphMode": "none"
                },
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "custom": {
                            "displayMode": "basic"
                        }
                    }
                }
            },
            # Panel 3: 今日切换次数
            {
                "id": 3,
                "title": "Today's Switches",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "increase(strategy_mode_transitions_total{env=\"$env\",symbol=~\"$symbol\"}[24h])",
                        "legendFormat": "Switches",
                        "datasource": "Prometheus"
                    }
                ],
                "options": {
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"]
                    },
                    "text": {
                        "valueSize": 48
                    },
                    "colorMode": "value",
                    "graphMode": "none"
                }
            },
            # Panel 4: 模式切换时间线
            {
                "id": 4,
                "title": "Mode Switch Timeline",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "strategy_mode_active{env=\"$env\",symbol=~\"$symbol\"}",
                        "legendFormat": "{{symbol}} - {{instance}}",
                        "datasource": "Prometheus"
                    }
                ],
                "options": {
                    "legend": {
                        "displayMode": "table",
                        "placement": "bottom"
                    },
                    "tooltip": {
                        "mode": "single"
                    }
                },
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "drawStyle": "line",
                            "lineInterpolation": "linear",
                            "barAlignment": 0,
                            "lineWidth": 1,
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "spanNulls": False,
                            "insertNulls": False,
                            "showPoints": "never",
                            "pointSize": 5,
                            "stacking": {
                                "mode": "none",
                                "group": "A"
                            },
                            "axisPlacement": "auto",
                            "axisLabel": "",
                            "scaleDistribution": {
                                "type": "linear"
                            },
                            "hideFrom": {
                                "legend": False,
                                "tooltip": False,
                                "vis": False
                            },
                            "thresholdsStyle": {
                                "mode": "off"
                            }
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 80}
                            ]
                        },
                        "unit": "short",
                        "min": 0,
                        "max": 1
                    }
                }
            }
        ]
    
    def _generate_performance_panels(self) -> List[Dict[str, Any]]:
        """生成性能面板配置"""
        return [
            # Panel 1: 参数更新耗时分布
            {
                "id": 1,
                "title": "Parameter Update Duration Distribution",
                "type": "histogram",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env=\"$env\"}[$__rate_interval])))",
                        "legendFormat": "P95",
                        "datasource": "Prometheus"
                    },
                    {
                        "expr": "histogram_quantile(0.50, sum by(le) (rate(strategy_params_update_duration_ms_bucket{env=\"$env\"}[$__rate_interval])))",
                        "legendFormat": "P50",
                        "datasource": "Prometheus"
                    }
                ]
            },
            # Panel 2: 更新失败次数
            {
                "id": 2,
                "title": "Update Failures",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "increase(strategy_params_update_failures_total{env=\"$env\"}[$__range]) by (module)",
                        "legendFormat": "{{module}}",
                        "datasource": "Prometheus"
                    }
                ]
            }
        ]
    
    def _generate_alerts_panels(self) -> List[Dict[str, Any]]:
        """生成告警面板配置"""
        return [
            # Panel 1: 当前告警列表
            {
                "id": 1,
                "title": "Current Alerts",
                "type": "table",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "ALERTS{alertstate=\"firing\"}",
                        "legendFormat": "{{alertname}}",
                        "datasource": "Prometheus",
                        "format": "table"
                    }
                ]
            },
            # Panel 2: 告警触发历史
            {
                "id": 2,
                "title": "Alert History",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "changes(ALERTS{alertstate=\"firing\"}[1m])",
                        "legendFormat": "{{alertname}}",
                        "datasource": "Prometheus"
                    }
                ]
            }
        ]
    
    def save_dashboard(self, dashboard_name: str, output_path: str) -> bool:
        """
        保存仪表盘配置到文件
        
        Args:
            dashboard_name: 仪表盘名称
            output_path: 输出文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 生成仪表盘配置
            if dashboard_name == "strategy_mode":
                dashboard_config = self.generate_strategy_mode_dashboard()
            elif dashboard_name == "performance":
                dashboard_config = self.generate_performance_dashboard()
            elif dashboard_name == "alerts":
                dashboard_config = self.generate_alerts_dashboard()
            else:
                logger.error(f"Unknown dashboard name: {dashboard_name}")
                return False
            
            # 确保输出目录存在
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dashboard {dashboard_name} saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving dashboard {dashboard_name}: {e}")
            return False
    
    def generate_all_dashboards(self, output_dir: str) -> bool:
        """
        生成所有仪表盘配置
        
        Args:
            output_dir: 输出目录
            
        Returns:
            bool: 是否全部生成成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            dashboards = [
                ("strategy_mode", "strategy_mode_overview.json"),
                ("performance", "strategy_performance.json"),
                ("alerts", "strategy_alerts.json")
            ]
            
            success = True
            for dashboard_name, filename in dashboards:
                file_path = output_path / filename
                if not self.save_dashboard(dashboard_name, str(file_path)):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error generating all dashboards: {e}")
            return False
