import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime, timedelta
import threading
import time

class RealTimeMonitor:
    """
    v9 实时监控系统
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.monitoring_data = {
            'trades': [],
            'performance_metrics': {},
            'alerts': [],
            'system_status': 'running'
        }
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.update_frequency = config.get('dashboard_update_frequency', 5)
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """启动实时监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("实时监控系统已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("实时监控系统已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._update_performance_metrics()
                self._check_alerts()
                self._update_dashboard()
                time.sleep(self.update_frequency)
            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(1)
    
    def add_trade(self, trade_data: dict):
        """添加交易记录"""
        if isinstance(trade_data, dict):
            trade_data['timestamp'] = datetime.now()
            self.monitoring_data['trades'].append(trade_data)
        else:
            # 如果是DataFrame行，转换为字典
            trade_dict = trade_data.to_dict()
            trade_dict['timestamp'] = datetime.now()
            self.monitoring_data['trades'].append(trade_dict)
        
        # 保持最近1000条交易记录
        if len(self.monitoring_data['trades']) > 1000:
            self.monitoring_data['trades'] = self.monitoring_data['trades'][-1000:]
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        if not self.monitoring_data['trades']:
            return
        
        trades_df = pd.DataFrame(self.monitoring_data['trades'])
        
        # 计算基本指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df.get('pnl', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 计算收益指标
        total_pnl = trades_df.get('pnl', 0).sum()
        avg_pnl = trades_df.get('pnl', 0).mean()
        
        # 计算风险指标
        if total_trades > 1:
            pnl_std = trades_df.get('pnl', 0).std()
            sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        cumulative_pnl = trades_df.get('pnl', 0).cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # 计算成本效率
        total_fees = trades_df.get('fee', 0).sum()
        cost_efficiency = total_pnl / total_fees if total_fees > 0 else float('inf')
        
        # 更新性能指标
        self.monitoring_data['performance_metrics'] = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cost_efficiency': cost_efficiency,
            'last_update': datetime.now().isoformat()
        }
    
    def _check_alerts(self):
        """检查告警条件"""
        metrics = self.monitoring_data['performance_metrics']
        
        # 检查最大回撤告警
        if metrics.get('max_drawdown', 0) < -self.alert_thresholds.get('max_drawdown', 0.05):
            self._add_alert('max_drawdown', f"最大回撤超过阈值: {metrics.get('max_drawdown', 0):.4f}")
        
        # 检查胜率告警
        if metrics.get('win_rate', 0) < self.alert_thresholds.get('min_win_rate', 0.8):
            self._add_alert('low_win_rate', f"胜率低于阈值: {metrics.get('win_rate', 0):.4f}")
        
        # 检查滑点告警
        recent_trades = self.monitoring_data['trades'][-10:]  # 最近10笔交易
        if recent_trades:
            avg_slippage = np.mean([trade.get('slippage', 0) for trade in recent_trades])
            if avg_slippage > self.alert_thresholds.get('max_slippage', 15.0):
                self._add_alert('high_slippage', f"平均滑点过高: {avg_slippage:.2f}")
    
    def _add_alert(self, alert_type: str, message: str):
        """添加告警"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning'
        }
        self.monitoring_data['alerts'].append(alert)
        
        # 保持最近100条告警
        if len(self.monitoring_data['alerts']) > 100:
            self.monitoring_data['alerts'] = self.monitoring_data['alerts'][-100:]
        
        print(f"告警: {message}")
    
    def _update_dashboard(self):
        """更新仪表板数据"""
        dashboard_data = {
            'system_status': self.monitoring_data['system_status'],
            'performance_metrics': self.monitoring_data['performance_metrics'],
            'recent_alerts': self.monitoring_data['alerts'][-5:],  # 最近5条告警
            'last_update': datetime.now().isoformat()
        }
        
        # 保存到文件
        dashboard_file = "examples/out/real_time_dashboard.json"
        os.makedirs(os.path.dirname(dashboard_file), exist_ok=True)
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    def get_dashboard_data(self) -> dict:
        """获取仪表板数据"""
        return {
            'system_status': self.monitoring_data['system_status'],
            'performance_metrics': self.monitoring_data['performance_metrics'],
            'recent_trades': self.monitoring_data['trades'][-10:],  # 最近10笔交易
            'recent_alerts': self.monitoring_data['alerts'][-5:],  # 最近5条告警
            'last_update': datetime.now().isoformat()
        }
    
    def generate_performance_report(self) -> dict:
        """生成性能报告"""
        if not self.monitoring_data['trades']:
            return {'error': 'No trades available'}
        
        trades_df = pd.DataFrame(self.monitoring_data['trades'])
        
        # 计算详细指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df.get('pnl', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # 收益分析
        total_pnl = trades_df.get('pnl', 0).sum()
        avg_pnl = trades_df.get('pnl', 0).mean()
        max_win = trades_df.get('pnl', 0).max()
        max_loss = trades_df.get('pnl', 0).min()
        
        # 风险分析
        if total_trades > 1:
            pnl_std = trades_df.get('pnl', 0).std()
            sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        cumulative_pnl = trades_df.get('pnl', 0).cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # 成本分析
        total_fees = trades_df.get('fee', 0).sum()
        total_slippage = trades_df.get('slippage', 0).sum()
        cost_efficiency = total_pnl / (total_fees + total_slippage) if (total_fees + total_slippage) > 0 else float('inf')
        
        # 信号质量分析
        avg_signal_strength = trades_df.get('signal_strength', 0).mean()
        avg_quality_score = trades_df.get('quality_score', 0).mean()
        
        return {
            'summary': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'max_win': max_win,
                'max_loss': max_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'cost_efficiency': cost_efficiency
            },
            'signal_quality': {
                'avg_signal_strength': avg_signal_strength,
                'avg_quality_score': avg_quality_score
            },
            'cost_analysis': {
                'total_fees': total_fees,
                'total_slippage': total_slippage,
                'cost_efficiency': cost_efficiency
            },
            'generated_at': datetime.now().isoformat()
        }

class PerformanceTracker:
    """
    v9 性能跟踪器
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = []
        self.metrics_history = []
    
    def add_performance_data(self, metrics: dict):
        """添加性能数据"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
        
        # 保持窗口大小
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
    
    def get_performance_trend(self) -> dict:
        """获取性能趋势"""
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_metrics = self.performance_history[-10:]  # 最近10个数据点
        older_metrics = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else self.performance_history[:-10]
        
        if not older_metrics:
            return {'trend': 'insufficient_data'}
        
        # 计算趋势
        recent_win_rate = np.mean([p['metrics'].get('win_rate', 0) for p in recent_metrics])
        older_win_rate = np.mean([p['metrics'].get('win_rate', 0) for p in older_metrics])
        
        recent_pnl = np.mean([p['metrics'].get('total_pnl', 0) for p in recent_metrics])
        older_pnl = np.mean([p['metrics'].get('total_pnl', 0) for p in older_metrics])
        
        return {
            'win_rate_trend': 'improving' if recent_win_rate > older_win_rate else 'declining',
            'pnl_trend': 'improving' if recent_pnl > older_pnl else 'declining',
            'recent_win_rate': recent_win_rate,
            'recent_pnl': recent_pnl
        }
    
    def get_optimization_suggestions(self) -> list:
        """获取优化建议"""
        suggestions = []
        
        if not self.performance_history:
            return suggestions
        
        recent_metrics = self.performance_history[-1]['metrics']
        
        # 基于胜率的建议
        if recent_metrics.get('win_rate', 0) < 0.8:
            suggestions.append("胜率较低，建议提高信号筛选标准")
        
        # 基于收益的建议
        if recent_metrics.get('total_pnl', 0) < 0:
            suggestions.append("总收益为负，建议优化止盈止损设置")
        
        # 基于回撤的建议
        if recent_metrics.get('max_drawdown', 0) < -0.05:
            suggestions.append("回撤过大，建议加强风险控制")
        
        # 基于成本效率的建议
        if recent_metrics.get('cost_efficiency', 0) < 1.0:
            suggestions.append("成本效率较低，建议优化交易成本")
        
        return suggestions
