"""
Grafana配置管理模块

支持从统一配置系统加载Grafana相关配置，包括：
- 仪表盘配置
- 数据源配置
- 变量配置
- 告警规则配置
- 通知渠道配置
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

@dataclass
class GrafanaServerConfig:
    """Grafana服务器配置"""
    host: str = "localhost"
    port: int = 3000
    protocol: str = "http"
    admin_user: str = "admin"
    admin_password: str = "admin"
    timeout: int = 30

@dataclass
class GrafanaDashboardConfig:
    """Grafana仪表盘配置"""
    uid: str = ""
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    timezone: str = "Asia/Hong_Kong"
    refresh: str = "30s"
    time_range: str = "6h"

@dataclass
class GrafanaDatasourceConfig:
    """Grafana数据源配置"""
    name: str = ""
    type: str = "prometheus"
    url: str = ""
    access: str = "proxy"
    is_default: bool = False
    editable: bool = True

@dataclass
class GrafanaVariableConfig:
    """Grafana变量配置"""
    name: str = ""
    type: str = "query"
    query: str = ""
    refresh: int = 1
    include_all: bool = False
    multi: bool = False
    default_value: Union[str, List[str]] = ""

@dataclass
class GrafanaAlertRuleConfig:
    """Grafana告警规则配置"""
    alert: str = ""
    expr: str = ""
    for_duration: str = "0m"
    severity: str = "warning"
    summary: str = ""
    description: str = ""

@dataclass
class GrafanaNotificationConfig:
    """Grafana通知渠道配置"""
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    email_enabled: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""
    email_from_address: str = ""
    email_to_addresses: List[str] = field(default_factory=list)

@dataclass
class GrafanaConfig:
    """Grafana完整配置"""
    # 基础配置
    dashboard_uid: str = "divergence-monitoring"
    refresh_interval: str = "5s"
    
    # 服务器配置
    server: GrafanaServerConfig = field(default_factory=GrafanaServerConfig)
    
    # 仪表盘配置
    dashboards: Dict[str, GrafanaDashboardConfig] = field(default_factory=dict)
    
    # 数据源配置
    datasources: Dict[str, GrafanaDatasourceConfig] = field(default_factory=dict)
    
    # 变量配置
    variables: Dict[str, GrafanaVariableConfig] = field(default_factory=dict)
    
    # 告警配置
    alerting: Dict[str, Any] = field(default_factory=dict)
    
    # 通知配置
    notifications: Dict[str, Any] = field(default_factory=dict)

class GrafanaConfigLoader:
    """Grafana配置加载器"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def load_config(self) -> GrafanaConfig:
        """
        从统一配置系统加载Grafana配置
        
        Returns:
            GrafanaConfig: 完整的Grafana配置对象
        """
        try:
            # 获取Grafana配置
            grafana_config_raw = self.config_loader.get('monitoring.grafana', {})
            
            # 基础配置
            dashboard_uid = grafana_config_raw.get('dashboard_uid', 'divergence-monitoring')
            refresh_interval = grafana_config_raw.get('refresh_interval', '5s')
            
            # 服务器配置
            server_config_raw = grafana_config_raw.get('server', {})
            server_config = GrafanaServerConfig(
                host=server_config_raw.get('host', 'localhost'),
                port=server_config_raw.get('port', 3000),
                protocol=server_config_raw.get('protocol', 'http'),
                admin_user=server_config_raw.get('admin_user', 'admin'),
                admin_password=server_config_raw.get('admin_password', 'admin'),
                timeout=server_config_raw.get('timeout', 30)
            )
            
            # 仪表盘配置
            dashboards = {}
            dashboards_config = grafana_config_raw.get('dashboards', {})
            for name, config in dashboards_config.items():
                dashboards[name] = GrafanaDashboardConfig(
                    uid=config.get('uid', ''),
                    title=config.get('title', ''),
                    description=config.get('description', ''),
                    tags=config.get('tags', []),
                    timezone=config.get('timezone', 'Asia/Hong_Kong'),
                    refresh=config.get('refresh', '30s'),
                    time_range=config.get('time_range', '6h')
                )
            
            # 数据源配置
            datasources = {}
            datasources_config = grafana_config_raw.get('datasources', {})
            for name, config in datasources_config.items():
                datasources[name] = GrafanaDatasourceConfig(
                    name=config.get('name', ''),
                    type=config.get('type', 'prometheus'),
                    url=config.get('url', ''),
                    access=config.get('access', 'proxy'),
                    is_default=config.get('is_default', False),
                    editable=config.get('editable', True)
                )
            
            # 变量配置
            variables = {}
            variables_config = grafana_config_raw.get('variables', {})
            for name, config in variables_config.items():
                variables[name] = GrafanaVariableConfig(
                    name=config.get('name', ''),
                    type=config.get('type', 'query'),
                    query=config.get('query', ''),
                    refresh=config.get('refresh', 1),
                    include_all=config.get('include_all', False),
                    multi=config.get('multi', False),
                    default_value=config.get('default_value', '')
                )
            
            # 告警配置
            alerting = grafana_config_raw.get('alerting', {})
            
            # 通知配置
            notifications = grafana_config_raw.get('notifications', {})
            
            return GrafanaConfig(
                dashboard_uid=dashboard_uid,
                refresh_interval=refresh_interval,
                server=server_config,
                dashboards=dashboards,
                datasources=datasources,
                variables=variables,
                alerting=alerting,
                notifications=notifications
            )
            
        except Exception as e:
            logger.error(f"Error loading Grafana config: {e}. Using default config.")
            return GrafanaConfig()
    
    def get_dashboard_config(self, dashboard_name: str) -> Optional[GrafanaDashboardConfig]:
        """
        获取指定仪表盘配置
        
        Args:
            dashboard_name: 仪表盘名称
            
        Returns:
            GrafanaDashboardConfig: 仪表盘配置对象
        """
        config = self.load_config()
        return config.dashboards.get(dashboard_name)
    
    def get_datasource_config(self, datasource_name: str) -> Optional[GrafanaDatasourceConfig]:
        """
        获取指定数据源配置
        
        Args:
            datasource_name: 数据源名称
            
        Returns:
            GrafanaDatasourceConfig: 数据源配置对象
        """
        config = self.load_config()
        return config.datasources.get(datasource_name)
    
    def get_variable_config(self, variable_name: str) -> Optional[GrafanaVariableConfig]:
        """
        获取指定变量配置
        
        Args:
            variable_name: 变量名称
            
        Returns:
            GrafanaVariableConfig: 变量配置对象
        """
        config = self.load_config()
        return config.variables.get(variable_name)
    
    def get_alert_rules(self) -> List[GrafanaAlertRuleConfig]:
        """
        获取告警规则配置列表
        
        Returns:
            List[GrafanaAlertRuleConfig]: 告警规则配置列表
        """
        config = self.load_config()
        alerting = config.alerting
        
        if not alerting.get('enabled', False):
            return []
        
        rules = []
        rules_config = alerting.get('rules', {})
        
        for rule_name, rule_config in rules_config.items():
            rules.append(GrafanaAlertRuleConfig(
                alert=rule_config.get('alert', ''),
                expr=rule_config.get('expr', ''),
                for_duration=rule_config.get('for', '0m'),
                severity=rule_config.get('severity', 'warning'),
                summary=rule_config.get('summary', ''),
                description=rule_config.get('description', '')
            ))
        
        return rules
    
    def generate_dashboard_json(self, dashboard_name: str, template_path: str) -> Dict[str, Any]:
        """
        基于配置和模板生成仪表盘JSON
        
        Args:
            dashboard_name: 仪表盘名称
            template_path: 模板文件路径
            
        Returns:
            Dict[str, Any]: 生成的仪表盘JSON配置
        """
        try:
            import json
            
            # 加载模板
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            # 获取仪表盘配置
            dashboard_config = self.get_dashboard_config(dashboard_name)
            if not dashboard_config:
                logger.warning(f"Dashboard config not found: {dashboard_name}")
                return template
            
            # 更新模板配置
            if 'dashboard' in template:
                dashboard = template['dashboard']
                
                # 更新基础信息
                dashboard['uid'] = dashboard_config.uid
                dashboard['title'] = dashboard_config.title
                dashboard['description'] = dashboard_config.description
                dashboard['tags'] = dashboard_config.tags
                dashboard['timezone'] = dashboard_config.timezone
                dashboard['refresh'] = dashboard_config.refresh
                
                # 更新时间范围
                if 'time' in dashboard:
                    dashboard['time']['from'] = f"now-{dashboard_config.time_range}"
                    dashboard['time']['to'] = "now"
            
            return template
            
        except Exception as e:
            logger.error(f"Error generating dashboard JSON for {dashboard_name}: {e}")
            return {}
    
    def generate_datasource_json(self, datasource_name: str) -> Dict[str, Any]:
        """
        生成数据源JSON配置
        
        Args:
            datasource_name: 数据源名称
            
        Returns:
            Dict[str, Any]: 数据源JSON配置
        """
        datasource_config = self.get_datasource_config(datasource_name)
        if not datasource_config:
            return {}
        
        return {
            "name": datasource_config.name,
            "type": datasource_config.type,
            "url": datasource_config.url,
            "access": datasource_config.access,
            "isDefault": datasource_config.is_default,
            "editable": datasource_config.editable
        }
    
    def generate_variables_json(self) -> List[Dict[str, Any]]:
        """
        生成变量JSON配置列表
        
        Returns:
            List[Dict[str, Any]]: 变量配置列表
        """
        config = self.load_config()
        variables = []
        
        for var_name, var_config in config.variables.items():
            variables.append({
                "name": var_config.name,
                "type": var_config.type,
                "query": var_config.query,
                "refresh": var_config.refresh,
                "includeAll": var_config.include_all,
                "multi": var_config.multi,
                "allValue": ".*" if var_config.include_all else "",
                "datasource": "Prometheus",
                "definition": var_config.query,
                "current": {
                    "selected": True,
                    "text": var_config.default_value,
                    "value": var_config.default_value
                }
            })
        
        return variables
