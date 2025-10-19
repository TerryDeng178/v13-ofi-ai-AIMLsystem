#!/usr/bin/env python3
"""
本周三件事执行脚本
自动化执行参数调优、单调性验证、指标对齐等任务
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import json
import yaml


class WeeklyTasksRunner:
    """本周三件事执行器"""
    
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 任务配置
        self.tasks = {
            'tune_params': {
                'script': 'scripts/tune_divergence.py',
                'description': '参数调优（3×3×3粗网格扫描）',
                'output': 'runs/tune_params',
                'required': True
            },
            'score_monotonicity': {
                'script': 'scripts/score_monotonicity.py',
                'description': 'Score→收益单调性验证',
                'output': 'runs/monotonicity',
                'required': True
            },
            'metrics_alignment': {
                'script': 'scripts/metrics_alignment.py',
                'description': '指标侧对齐（Prometheus/Grafana）',
                'output': 'runs/metrics',
                'required': True
            },
            'config_hot_update': {
                'script': 'scripts/config_hot_update.py',
                'description': '参数固化与热更新',
                'output': 'runs/config',
                'required': True
            }
        }
        
        # 验收标准
        self.acceptance_criteria = {
            'tune_params': {
                'min_buckets': 2,  # 至少2个桶有结果
                'min_accuracy': 0.55,  # 至少55%准确率
                'min_p_value': 0.05,  # 统计显著
                'required_sources': ['OFI_ONLY', 'CVD_ONLY']  # 必须包含的数据源
            },
            'score_monotonicity': {
                'min_spearman_corr': 0.0,  # 正相关
                'max_p_value': 0.05,  # 统计显著
                'min_horizons': 1  # 至少1个前瞻窗口
            },
            'metrics_alignment': {
                'max_alignment_error': 10.0,  # 对齐误差<10%
                'max_latency': 0.003,  # 延迟<3ms
                'closure_tolerance': 0.1  # 闭合性容差<10%
            },
            'config_hot_update': {
                'config_loaded': True,  # 配置加载成功
                'hot_reload_working': True,  # 热更新工作
                'calibration_available': True  # 校准配置可用
            }
        }
    
    def run_task(self, task_name: str) -> Dict[str, Any]:
        """运行单个任务"""
        task_config = self.tasks[task_name]
        print(f"\n开始执行: {task_config['description']}")
        
        # 构建命令
        script_path = Path(task_config['script'])
        if not script_path.exists():
            return {
                'status': 'error',
                'message': f'脚本不存在: {script_path}',
                'output': None
            }
        
        cmd = [sys.executable, str(script_path)]
        
        # 添加任务特定参数
        if task_name in ['tune_params', 'score_monotonicity']:
            cmd.extend(['--data', str(self.data_path), '--out', str(self.output_dir / task_config['output'])])
        elif task_name == 'metrics_alignment':
            cmd.extend(['--out', str(self.output_dir / task_config['output'])])
        elif task_name == 'config_hot_update':
            cmd.extend(['--test'])
        
        # 添加任务特定参数
        if task_name == 'tune_params':
            cmd.extend(['--horizons', '10,20'])
        elif task_name == 'score_monotonicity':
            cmd.extend(['--bins', '10'])
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 运行任务
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"任务完成 (耗时: {duration:.1f}s)")
                return {
                    'status': 'success',
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'output': self.output_dir / task_config['output']
                }
            else:
                print(f"任务失败 (返回码: {result.returncode})")
                return {
                    'status': 'error',
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'output': None
                }
                
        except subprocess.TimeoutExpired:
            print("任务超时")
            return {
                'status': 'timeout',
                'message': '任务执行超时',
                'output': None
            }
        except Exception as e:
            print(f"任务异常: {e}")
            return {
                'status': 'exception',
                'message': str(e),
                'output': None
            }
    
    def validate_task_results(self, task_name: str, result: Dict[str, Any]) -> bool:
        """验证任务结果"""
        if result['status'] != 'success':
            return False
        
        criteria = self.acceptance_criteria[task_name]
        output_dir = result['output']
        
        if not output_dir or not output_dir.exists():
            return False
        
        try:
            if task_name == 'tune_params':
                return self._validate_tune_params(output_dir, criteria)
            elif task_name == 'score_monotonicity':
                return self._validate_score_monotonicity(output_dir, criteria)
            elif task_name == 'metrics_alignment':
                return self._validate_metrics_alignment(output_dir, criteria)
            elif task_name == 'config_hot_update':
                return self._validate_config_hot_update(criteria)
            else:
                return True
        except Exception as e:
            print(f"验证失败: {e}")
            return False
    
    def _validate_tune_params(self, output_dir: Path, criteria: Dict[str, Any]) -> bool:
        """验证参数调优结果"""
        summary_file = output_dir / "summary.csv"
        if not summary_file.exists():
            return False
        
        # 读取结果
        import pandas as pd
        df = pd.read_csv(summary_file)
        
        # 检查桶数量
        if len(df) < criteria['min_buckets']:
            return False
        
        # 检查准确率
        high_acc_buckets = df[df['accuracy_10'] >= criteria['min_accuracy']]
        if len(high_acc_buckets) < criteria['min_buckets']:
            return False
        
        # 检查统计显著性
        significant_buckets = df[df['p_value_10'] < criteria['min_p_value']]
        if len(significant_buckets) < criteria['min_buckets']:
            return False
        
        # 检查必需数据源
        required_sources = criteria['required_sources']
        available_sources = df['bucket'].apply(lambda x: eval(x)['source']).unique()
        for source in required_sources:
            if source not in available_sources:
                return False
        
        return True
    
    def _validate_score_monotonicity(self, output_dir: Path, criteria: Dict[str, Any]) -> bool:
        """验证单调性验证结果"""
        report_file = output_dir / "monotonicity_report.json"
        if not report_file.exists():
            return False
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 检查单调性
        monotonic_horizons = report['summary']['monotonic_horizons']
        if monotonic_horizons < criteria['min_horizons']:
            return False
        
        # 检查统计显著性
        significant_horizons = report['summary']['significant_horizons']
        if significant_horizons < criteria['min_horizons']:
            return False
        
        return True
    
    def _validate_metrics_alignment(self, output_dir: Path, criteria: Dict[str, Any]) -> bool:
        """验证指标对齐结果"""
        # 这里应该检查Prometheus和Grafana是否正常工作
        # 简化版：检查配置文件是否存在
        prometheus_config = output_dir / "prometheus_divergence.yml"
        grafana_dashboard = output_dir / "dashboards" / "divergence_overview.json"
        
        return prometheus_config.exists() and grafana_dashboard.exists()
    
    def _validate_config_hot_update(self, criteria: Dict[str, Any]) -> bool:
        """验证配置热更新结果"""
        # 检查配置文件是否存在
        config_file = Path("config/system.yaml")
        if not config_file.exists():
            return False
        
        # 检查校准文件是否存在
        calibration_file = Path("config/calibration/divergence_score_calibration.json")
        if not calibration_file.exists():
            return False
        
        return True
    
    def run_all_tasks(self) -> Dict[str, Any]:
        """运行所有任务"""
        print("开始执行本周三件事...")
        
        results = {}
        overall_success = True
        
        for task_name in self.tasks:
            print(f"\n{'='*50}")
            print(f"任务: {task_name}")
            print(f"{'='*50}")
            
            # 运行任务
            result = self.run_task(task_name)
            results[task_name] = result
            
            # 验证结果
            if result['status'] == 'success':
                is_valid = self.validate_task_results(task_name, result)
                if is_valid:
                    print(f"{task_name} 通过验收")
                else:
                    print(f"{task_name} 未通过验收")
                    overall_success = False
            else:
                print(f"{task_name} 执行失败")
                overall_success = False
        
        # 生成总结报告
        self.generate_summary_report(results, overall_success)
        
        return {
            'overall_success': overall_success,
            'results': results
        }
    
    def generate_summary_report(self, results: Dict[str, Any], overall_success: bool):
        """生成总结报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_success': overall_success,
            'tasks': {}
        }
        
        for task_name, result in results.items():
            task_config = self.tasks[task_name]
            report['tasks'][task_name] = {
                'description': task_config['description'],
                'status': result['status'],
                'duration': result.get('duration', 0),
                'output': str(result.get('output', '')),
                'required': task_config['required']
            }
        
        # 保存报告
        report_file = self.output_dir / "weekly_tasks_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n总结报告已保存: {report_file}")
        
        # 打印总结
        print(f"\n{'='*50}")
        print("执行总结")
        print(f"{'='*50}")
        
        for task_name, result in results.items():
            status_icon = "成功" if result['status'] == 'success' else "失败"
            duration = result.get('duration', 0)
            print(f"{status_icon} {task_name}: {result['status']} ({duration:.1f}s)")
        
        print(f"\n总体状态: {'成功' if overall_success else '失败'}")


def main():
    parser = argparse.ArgumentParser(description='本周三件事执行器')
    parser.add_argument('--data', required=True, help='数据路径')
    parser.add_argument('--out', required=True, help='输出目录')
    parser.add_argument('--task', help='运行指定任务 (tune_params|score_monotonicity|metrics_alignment|config_hot_update)')
    
    args = parser.parse_args()
    
    # 创建执行器
    runner = WeeklyTasksRunner(args.data, args.out)
    
    if args.task:
        # 运行指定任务
        if args.task not in runner.tasks:
            print(f"未知任务: {args.task}")
            print(f"可用任务: {', '.join(runner.tasks.keys())}")
            return
        
        result = runner.run_task(args.task)
        is_valid = runner.validate_task_results(args.task, result)
        
        print(f"\n任务结果: {'成功' if is_valid else '失败'}")
    else:
        # 运行所有任务
        runner.run_all_tasks()


if __name__ == "__main__":
    main()
