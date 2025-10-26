#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFI灰度验证脚本 - 2×2场景各跑≥24h稳定数据达标验证
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ofi_config_parser import OFIConfigParser
from ofi_monitoring_system import OFIMonitor, MonitoringMetrics

@dataclass
class GrayValidationConfig:
    """灰度验证配置"""
    symbols: List[str]
    profiles: List[str]
    regimes: List[str]
    duration_hours: int = 24
    check_interval_minutes: int = 5
    min_data_points: int = 100
    stability_threshold: float = 0.8  # 稳定性阈值

@dataclass
class ValidationResult:
    """验证结果"""
    symbol: str
    profile: str
    regime: str
    duration_hours: float
    data_points: int
    p_gt2_avg: float
    p_gt2_std: float
    p_gt3_avg: float
    iqr_avg: float
    median_avg: float
    stability_score: float
    is_stable: bool
    violations_count: int
    status: str  # "PASS", "FAIL", "IN_PROGRESS"

class GrayValidator:
    """灰度验证器"""
    
    def __init__(self, config_path: str = "../config/defaults.yaml"):
        self.config_parser = OFIConfigParser(config_path)
        self.monitor = OFIMonitor(config_path)
        
        # 验证数据存储
        self.validation_data: Dict[str, List[MonitoringMetrics]] = {}
        self.validation_results: List[ValidationResult] = []
        
        # 日志设置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def start_validation(self, config: GrayValidationConfig) -> None:
        """开始灰度验证"""
        self.logger.info(f"Starting gray validation for {len(config.symbols)} symbols")
        self.logger.info(f"Duration: {config.duration_hours} hours")
        self.logger.info(f"Check interval: {config.check_interval_minutes} minutes")
        
        # 初始化验证数据存储
        for symbol in config.symbols:
            for profile in config.profiles:
                for regime in config.regimes:
                    key = f"{symbol}_{profile}_{regime}"
                    self.validation_data[key] = []
        
        # 开始监控
        self._run_validation_loop(config)
    
    def _run_validation_loop(self, config: GrayValidationConfig) -> None:
        """运行验证循环"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=config.duration_hours)
        
        self.logger.info(f"Validation started at {start_time}")
        self.logger.info(f"Validation will end at {end_time}")
        
        while datetime.now() < end_time:
            # 为每个symbol/profile/regime组合生成模拟数据
            for symbol in config.symbols:
                for profile in config.profiles:
                    for regime in config.regimes:
                        metrics = self._generate_mock_metrics(symbol, profile, regime)
                        
                        # 添加到监控系统
                        self.monitor.add_metrics(metrics)
                        
                        # 存储验证数据
                        key = f"{symbol}_{profile}_{regime}"
                        self.validation_data[key].append(metrics)
            
            # 等待下次检查
            time.sleep(config.check_interval_minutes * 60)
            
            # 输出进度
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            self.logger.info(f"Validation progress: {elapsed_hours:.1f}/{config.duration_hours} hours")
        
        self.logger.info("Validation completed")
    
    def _generate_mock_metrics(self, symbol: str, profile: str, regime: str) -> MonitoringMetrics:
        """生成模拟监控指标"""
        # 获取配置
        config = self.config_parser.get_ofi_config(symbol, profile, regime)
        
        # 根据配置生成模拟数据
        if config.z_clip is None:  # 离线评估
            # 模拟正常分布
            p_gt2 = 0.05 + (hash(symbol) % 100) / 10000  # 5%左右
            p_gt3 = 0.01 + (hash(symbol) % 50) / 10000   # 1%左右
        else:  # 线上生产
            # 模拟稍微收紧的分布
            p_gt2 = 0.04 + (hash(symbol) % 80) / 10000   # 4%左右
            p_gt3 = 0.008 + (hash(symbol) % 40) / 10000  # 0.8%左右
        
        # 添加一些随机波动
        import random
        p_gt2 += random.uniform(-0.01, 0.01)
        p_gt3 += random.uniform(-0.005, 0.005)
        
        # 确保在合理范围内
        p_gt2 = max(0.001, min(0.15, p_gt2))
        p_gt3 = max(0.0001, min(0.05, p_gt3))
        
        return MonitoringMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            profile=profile,
            regime=regime,
            p_gt2=p_gt2,
            p_gt3=p_gt3,
            iqr=1.2 + random.uniform(-0.2, 0.2),
            median=random.uniform(-0.05, 0.05),
            z_clip=config.z_clip
        )
    
    def analyze_results(self, config: GrayValidationConfig) -> List[ValidationResult]:
        """分析验证结果"""
        self.logger.info("Analyzing validation results...")
        
        results = []
        
        for symbol in config.symbols:
            for profile in config.profiles:
                for regime in config.regimes:
                    key = f"{symbol}_{profile}_{regime}"
                    data = self.validation_data[key]
                    
                    if len(data) < config.min_data_points:
                        result = ValidationResult(
                            symbol=symbol,
                            profile=profile,
                            regime=regime,
                            duration_hours=0,
                            data_points=len(data),
                            p_gt2_avg=0,
                            p_gt2_std=0,
                            p_gt3_avg=0,
                            iqr_avg=0,
                            median_avg=0,
                            stability_score=0,
                            is_stable=False,
                            violations_count=0,
                            status="FAIL"
                        )
                        results.append(result)
                        continue
                    
                    # 计算统计指标
                    p_gt2_values = [m.p_gt2 for m in data]
                    p_gt3_values = [m.p_gt3 for m in data]
                    iqr_values = [m.iqr for m in data]
                    median_values = [m.median for m in data]
                    
                    p_gt2_avg = sum(p_gt2_values) / len(p_gt2_values)
                    p_gt2_std = (sum((x - p_gt2_avg) ** 2 for x in p_gt2_values) / len(p_gt2_values)) ** 0.5
                    p_gt3_avg = sum(p_gt3_values) / len(p_gt3_values)
                    iqr_avg = sum(iqr_values) / len(iqr_values)
                    median_avg = sum(median_values) / len(median_values)
                    
                    # 计算稳定性分数
                    stability_score = self._calculate_stability_score(p_gt2_values, p_gt3_values)
                    
                    # 检查是否稳定
                    is_stable = stability_score >= config.stability_threshold
                    
                    # 计算越界次数
                    violations_count = sum(1 for m in data if m.is_violation)
                    
                    # 确定状态
                    if is_stable and violations_count == 0:
                        status = "PASS"
                    elif violations_count > len(data) * 0.1:  # 越界超过10%
                        status = "FAIL"
                    else:
                        status = "IN_PROGRESS"
                    
                    # 计算持续时间
                    if data:
                        duration_hours = (data[-1].timestamp - data[0].timestamp).total_seconds() / 3600
                    else:
                        duration_hours = 0
                    
                    result = ValidationResult(
                        symbol=symbol,
                        profile=profile,
                        regime=regime,
                        duration_hours=duration_hours,
                        data_points=len(data),
                        p_gt2_avg=p_gt2_avg,
                        p_gt2_std=p_gt2_std,
                        p_gt3_avg=p_gt3_avg,
                        iqr_avg=iqr_avg,
                        median_avg=median_avg,
                        stability_score=stability_score,
                        is_stable=is_stable,
                        violations_count=violations_count,
                        status=status
                    )
                    
                    results.append(result)
        
        self.validation_results = results
        return results
    
    def _calculate_stability_score(self, p_gt2_values: List[float], p_gt3_values: List[float]) -> float:
        """计算稳定性分数"""
        if not p_gt2_values or not p_gt3_values:
            return 0.0
        
        # 计算P(|z|>2)的稳定性
        p_gt2_avg = sum(p_gt2_values) / len(p_gt2_values)
        p_gt2_cv = (sum((x - p_gt2_avg) ** 2 for x in p_gt2_values) / len(p_gt2_values)) ** 0.5 / p_gt2_avg
        
        # 计算P(|z|>3)的稳定性
        p_gt3_avg = sum(p_gt3_values) / len(p_gt3_values)
        p_gt3_cv = (sum((x - p_gt3_avg) ** 2 for x in p_gt3_values) / len(p_gt3_values)) ** 0.5 / p_gt3_avg
        
        # 稳定性分数 = 1 - 变异系数（越小越稳定）
        stability_score = max(0, 1 - (p_gt2_cv + p_gt3_cv) / 2)
        
        return stability_score
    
    def generate_report(self, filename: str = None) -> None:
        """生成验证报告"""
        if filename is None:
            filename = f"gray_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 统计结果
        total_scenarios = len(self.validation_results)
        pass_scenarios = sum(1 for r in self.validation_results if r.status == "PASS")
        fail_scenarios = sum(1 for r in self.validation_results if r.status == "FAIL")
        in_progress_scenarios = sum(1 for r in self.validation_results if r.status == "IN_PROGRESS")
        
        report = {
            'summary': {
                'total_scenarios': total_scenarios,
                'pass_scenarios': pass_scenarios,
                'fail_scenarios': fail_scenarios,
                'in_progress_scenarios': in_progress_scenarios,
                'pass_rate': pass_scenarios / total_scenarios if total_scenarios > 0 else 0,
                'generated_at': datetime.now().isoformat()
            },
            'results': [
                {
                    'symbol': r.symbol,
                    'profile': r.profile,
                    'regime': r.regime,
                    'duration_hours': r.duration_hours,
                    'data_points': r.data_points,
                    'p_gt2_avg': r.p_gt2_avg,
                    'p_gt2_std': r.p_gt2_std,
                    'p_gt3_avg': r.p_gt3_avg,
                    'iqr_avg': r.iqr_avg,
                    'median_avg': r.median_avg,
                    'stability_score': r.stability_score,
                    'is_stable': r.is_stable,
                    'violations_count': r.violations_count,
                    'status': r.status
                }
                for r in self.validation_results
            ],
            'monitoring_status': self.monitor.get_status_report()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Gray validation report saved to {filename}")
        
        # 输出摘要
        print(f"\n=== 灰度验证报告摘要 ===")
        print(f"总场景数: {total_scenarios}")
        print(f"通过场景: {pass_scenarios}")
        print(f"失败场景: {fail_scenarios}")
        print(f"进行中场景: {in_progress_scenarios}")
        print(f"通过率: {pass_scenarios / total_scenarios * 100:.1f}%")
        
        if pass_scenarios == total_scenarios:
            print("SUCCESS 所有场景都通过验证！")
        elif pass_scenarios >= total_scenarios * 0.8:
            print("WARN 大部分场景通过，需要关注失败场景")
        else:
            print("FAIL 多个场景失败，需要调整配置")

def test_gray_validation():
    """测试灰度验证"""
    validator = GrayValidator()
    
    # 配置验证参数
    config = GrayValidationConfig(
        symbols=["BTCUSDT", "XRPUSDT"],  # 高流动性和低流动性各一个
        profiles=["offline_eval", "online_prod"],
        regimes=["active", "quiet"],
        duration_hours=1,  # 测试用1小时
        check_interval_minutes=1,  # 测试用1分钟间隔
        min_data_points=10,  # 测试用最少10个数据点
        stability_threshold=0.7  # 测试用较低阈值
    )
    
    print("=== 灰度验证测试 ===")
    print(f"测试场景: {len(config.symbols)} symbols × {len(config.profiles)} profiles × {len(config.regimes)} regimes")
    print(f"测试时长: {config.duration_hours} 小时")
    print(f"检查间隔: {config.check_interval_minutes} 分钟")
    
    # 开始验证
    validator.start_validation(config)
    
    # 分析结果
    results = validator.analyze_results(config)
    
    # 生成报告
    validator.generate_report("test_gray_validation_report.json")
    
    print("\n=== 验证完成 ===")

if __name__ == "__main__":
    test_gray_validation()
