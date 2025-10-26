#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI监控和告警系统 - 支持灰度/实盘监控
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ofi_config_parser import OFIConfigParser, Guardrails

@dataclass
class MonitoringMetrics:
    """监控指标"""
    timestamp: datetime
    symbol: str
    profile: str
    regime: str
    p_gt2: float
    p_gt3: float
    iqr: float
    median: float
    z_clip: Optional[float]
    is_violation: bool = False
    violation_type: Optional[str] = None

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str  # "p_gt2_out", "p_gt3_out", "iqr_out", "median_out"
    threshold: float
    duration_minutes: int
    action: str  # "adjust_z_clip", "rollback", "notify"

@dataclass
class ViolationRecord:
    """越界记录"""
    symbol: str
    profile: str
    regime: str
    violation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: int = 0
    action_taken: Optional[str] = None
    metrics_before: Optional[Dict] = None
    metrics_after: Optional[Dict] = None

class OFIMonitor:
    """OFI监控系统"""
    
    def __init__(self, config_path: str = "../config/defaults.yaml"):
        self.config_parser = OFIConfigParser(config_path)
        self.guardrails = self.config_parser.get_guardrails()
        
        # 监控数据存储
        self.metrics_history: Dict[str, deque] = {}  # symbol -> deque of MonitoringMetrics
        self.violations: List[ViolationRecord] = []
        self.active_violations: Dict[str, ViolationRecord] = {}  # symbol -> ViolationRecord
        
        # 告警规则
        self.alert_rules = self._setup_alert_rules()
        
        # 日志设置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _setup_alert_rules(self) -> List[AlertRule]:
        """设置告警规则"""
        return [
            AlertRule(
                name="p_gt2_too_low",
                condition="p_gt2_out",
                threshold=self.guardrails.p_gt2_range[0],
                duration_minutes=self.guardrails.rollback_minutes,
                action="adjust_z_clip"
            ),
            AlertRule(
                name="p_gt2_too_high",
                condition="p_gt2_out",
                threshold=self.guardrails.p_gt2_range[1],
                duration_minutes=self.guardrails.rollback_minutes,
                action="adjust_z_clip"
            ),
            AlertRule(
                name="p_gt3_too_high",
                condition="p_gt3_out",
                threshold=self.guardrails.p_gt3_max,
                duration_minutes=self.guardrails.rollback_minutes,
                action="adjust_z_clip"
            ),
        ]
    
    def add_metrics(self, metrics: MonitoringMetrics) -> None:
        """添加监控指标"""
        symbol_key = f"{metrics.symbol}_{metrics.profile}_{metrics.regime}"
        
        if symbol_key not in self.metrics_history:
            self.metrics_history[symbol_key] = deque(maxlen=1000)  # 保留最近1000个数据点
        
        self.metrics_history[symbol_key].append(metrics)
        
        # 检查越界
        self._check_violations(metrics)
        
        # 检查是否需要触发告警
        self._check_alerts(metrics)
    
    def _check_violations(self, metrics: MonitoringMetrics) -> None:
        """检查越界情况"""
        symbol_key = f"{metrics.symbol}_{metrics.profile}_{metrics.regime}"
        
        # 检查P(|z|>2)越界
        p_gt2_min, p_gt2_max = self.guardrails.p_gt2_range
        is_p_gt2_out = metrics.p_gt2 < p_gt2_min or metrics.p_gt2 > p_gt2_max
        
        # 检查P(|z|>3)越界
        is_p_gt3_out = metrics.p_gt3 > self.guardrails.p_gt3_max
        
        # 检查IQR越界（假设正常范围0.8-1.6）
        is_iqr_out = metrics.iqr < 0.8 or metrics.iqr > 1.6
        
        # 检查中位数越界（假设正常范围±0.1）
        is_median_out = abs(metrics.median) > 0.1
        
        # 确定越界类型
        violation_types = []
        if is_p_gt2_out:
            if metrics.p_gt2 < p_gt2_min:
                violation_types.append("p_gt2_too_low")
            else:
                violation_types.append("p_gt2_too_high")
        
        if is_p_gt3_out:
            violation_types.append("p_gt3_too_high")
        
        if is_iqr_out:
            violation_types.append("iqr_out")
        
        if is_median_out:
            violation_types.append("median_out")
        
        # 更新越界状态
        if violation_types:
            metrics.is_violation = True
            metrics.violation_type = ",".join(violation_types)
            
            # 检查是否是新越界
            if symbol_key not in self.active_violations:
                violation = ViolationRecord(
                    symbol=metrics.symbol,
                    profile=metrics.profile,
                    regime=metrics.regime,
                    violation_type=metrics.violation_type,
                    start_time=metrics.timestamp,
                    metrics_before={
                        'p_gt2': metrics.p_gt2,
                        'p_gt3': metrics.p_gt3,
                        'iqr': metrics.iqr,
                        'median': metrics.median,
                        'z_clip': metrics.z_clip
                    }
                )
                self.active_violations[symbol_key] = violation
                self.logger.warning(f"New violation detected for {symbol_key}: {metrics.violation_type}")
        else:
            # 检查是否越界结束
            if symbol_key in self.active_violations:
                violation = self.active_violations[symbol_key]
                violation.end_time = metrics.timestamp
                violation.duration_minutes = int((violation.end_time - violation.start_time).total_seconds() / 60)
                violation.metrics_after = {
                    'p_gt2': metrics.p_gt2,
                    'p_gt3': metrics.p_gt3,
                    'iqr': metrics.iqr,
                    'median': metrics.median,
                    'z_clip': metrics.z_clip
                }
                
                self.violations.append(violation)
                del self.active_violations[symbol_key]
                self.logger.info(f"Violation ended for {symbol_key}: duration={violation.duration_minutes}min")
    
    def _check_alerts(self, metrics: MonitoringMetrics) -> None:
        """检查告警条件"""
        if not metrics.is_violation:
            return
        
        symbol_key = f"{metrics.symbol}_{metrics.profile}_{metrics.regime}"
        
        # 检查持续越界时间
        if symbol_key in self.active_violations:
            violation = self.active_violations[symbol_key]
            duration_minutes = int((metrics.timestamp - violation.start_time).total_seconds() / 60)
            
            # 检查是否达到告警阈值
            for rule in self.alert_rules:
                if rule.condition in metrics.violation_type and duration_minutes >= rule.duration_minutes:
                    self._trigger_alert(rule, metrics, violation)
    
    def _trigger_alert(self, rule: AlertRule, metrics: MonitoringMetrics, violation: ViolationRecord) -> None:
        """触发告警"""
        self.logger.error(f"ALERT: {rule.name} triggered for {metrics.symbol}")
        self.logger.error(f"  Condition: {rule.condition}")
        self.logger.error(f"  Duration: {int((metrics.timestamp - violation.start_time).total_seconds() / 60)}min")
        self.logger.error(f"  Metrics: P(|z|>2)={metrics.p_gt2:.3f}, P(|z|>3)={metrics.p_gt3:.3f}")
        
        if rule.action == "adjust_z_clip":
            self._adjust_z_clip(metrics, violation)
        elif rule.action == "rollback":
            self._rollback_config(metrics)
        elif rule.action == "notify":
            self._send_notification(rule, metrics)
    
    def _adjust_z_clip(self, metrics: MonitoringMetrics, violation: ViolationRecord) -> None:
        """调整z_clip参数"""
        current_z_clip = metrics.z_clip
        adjustment_step = self.guardrails.clip_adjust_step
        
        if "p_gt2_too_low" in metrics.violation_type:
            # P(|z|>2)过低，需要放宽z_clip
            if current_z_clip is None:
                new_z_clip = 2.5  # 从禁用状态启用
            else:
                new_z_clip = min(current_z_clip + adjustment_step, 5.0)  # 上限5.0
        elif "p_gt2_too_high" in metrics.violation_type or "p_gt3_too_high" in metrics.violation_type:
            # P(|z|>2)或P(|z|>3)过高，需要收紧z_clip
            if current_z_clip is None:
                new_z_clip = 3.0  # 从禁用状态启用
            else:
                new_z_clip = max(current_z_clip - adjustment_step, self.guardrails.min_z_clip)
        else:
            return
        
        # 应用调整
        override_config = {'z_clip': new_z_clip}
        self.config_parser.add_symbol_override(metrics.symbol, metrics.profile, override_config)
        
        violation.action_taken = f"adjusted_z_clip_{current_z_clip}_to_{new_z_clip}"
        self.logger.info(f"Adjusted z_clip for {metrics.symbol}: {current_z_clip} -> {new_z_clip}")
    
    def _rollback_config(self, metrics: MonitoringMetrics) -> None:
        """回滚配置"""
        # 移除symbol override，回到全局基线
        self.config_parser.remove_symbol_override(metrics.symbol, metrics.profile)
        
        if f"{metrics.symbol}_{metrics.profile}_{metrics.regime}" in self.active_violations:
            violation = self.active_violations[f"{metrics.symbol}_{metrics.profile}_{metrics.regime}"]
            violation.action_taken = "rolled_back_to_global_baseline"
        
        self.logger.warning(f"Rolled back config for {metrics.symbol} to global baseline")
    
    def _send_notification(self, rule: AlertRule, metrics: MonitoringMetrics) -> None:
        """发送通知"""
        # 这里可以集成邮件、短信、Slack等通知方式
        self.logger.info(f"Notification sent for {rule.name}: {metrics.symbol}")
    
    def get_status_report(self) -> Dict:
        """获取状态报告"""
        now = datetime.now()
        
        # 统计活跃越界
        active_violations = {}
        for symbol_key, violation in self.active_violations.items():
            duration_minutes = int((now - violation.start_time).total_seconds() / 60)
            active_violations[symbol_key] = {
                'violation_type': violation.violation_type,
                'duration_minutes': duration_minutes,
                'action_taken': violation.action_taken
            }
        
        # 统计历史越界
        recent_violations = [
            v for v in self.violations 
            if v.end_time and (now - v.end_time).total_seconds() < 3600  # 最近1小时
        ]
        
        return {
            'timestamp': now.isoformat(),
            'active_violations': active_violations,
            'recent_violations_count': len(recent_violations),
            'total_violations_count': len(self.violations),
            'monitored_symbols': len(self.metrics_history),
            'guardrails': {
                'p_gt2_range': self.guardrails.p_gt2_range,
                'p_gt3_max': self.guardrails.p_gt3_max,
                'rollback_minutes': self.guardrails.rollback_minutes
            }
        }
    
    def save_report(self, filename: str = None) -> None:
        """保存监控报告"""
        if filename is None:
            filename = f"ofi_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'status': self.get_status_report(),
            'violations': [
                {
                    'symbol': v.symbol,
                    'profile': v.profile,
                    'regime': v.regime,
                    'violation_type': v.violation_type,
                    'start_time': v.start_time.isoformat(),
                    'end_time': v.end_time.isoformat() if v.end_time else None,
                    'duration_minutes': v.duration_minutes,
                    'action_taken': v.action_taken,
                    'metrics_before': v.metrics_before,
                    'metrics_after': v.metrics_after
                }
                for v in self.violations
            ],
            'active_violations': [
                {
                    'symbol': v.symbol,
                    'profile': v.profile,
                    'regime': v.regime,
                    'violation_type': v.violation_type,
                    'start_time': v.start_time.isoformat(),
                    'duration_minutes': int((datetime.now() - v.start_time).total_seconds() / 60),
                    'action_taken': v.action_taken,
                    'metrics_before': v.metrics_before
                }
                for v in self.active_violations.values()
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Monitoring report saved to {filename}")

def test_monitoring_system():
    """测试监控系统"""
    monitor = OFIMonitor()
    
    print("=== OFI监控系统测试 ===")
    
    # 模拟正常指标
    normal_metrics = MonitoringMetrics(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        profile="offline_eval",
        regime="active",
        p_gt2=0.05,
        p_gt3=0.01,
        iqr=1.2,
        median=0.0,
        z_clip=None
    )
    
    print("1. 测试正常指标:")
    monitor.add_metrics(normal_metrics)
    print("OK 正常指标处理完成")
    
    # 模拟越界指标
    violation_metrics = MonitoringMetrics(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        profile="offline_eval",
        regime="active",
        p_gt2=0.005,  # 过低
        p_gt3=0.001,
        iqr=1.2,
        median=0.0,
        z_clip=None
    )
    
    print("\n2. 测试越界指标:")
    monitor.add_metrics(violation_metrics)
    print("OK 越界指标处理完成")
    
    # 获取状态报告
    print("\n3. 状态报告:")
    status = monitor.get_status_report()
    print(f"OK 活跃越界数量: {len(status['active_violations'])}")
    print(f"OK 历史越界数量: {status['total_violations_count']}")
    print(f"OK 监控交易对数量: {status['monitored_symbols']}")
    
    # 保存报告
    print("\n4. 保存报告:")
    monitor.save_report("test_monitoring_report.json")
    print("OK 监控报告已保存")

if __name__ == "__main__":
    test_monitoring_system()
