#!/usr/bin/env python3
"""
统一OFI+CVD数据采集脚本 (Task 1.3.1 v2)

连续采集48-72小时BTCUSDT、ETHUSDT实时数据，产出5类分区化数据集：
- prices: 价格数据
- ofi: OFI指标数据  
- cvd: CVD指标数据
- fusion: 融合指标数据
- events: 背离/枢轴/异常事件数据

支持环境变量配置、Prometheus监控、自动恢复、数据质量报告。
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
import hashlib

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import pyarrow as pa
import pyarrow.parquet as pq

# 导入项目模块
from src.binance_websocket_adapter import BinanceWebSocketAdapter
from src.real_ofi_calculator import RealOFICalculator
from src.real_cvd_calculator import RealCVDCalculator
from src.ofi_cvd_fusion import RegimeClassifier
from src.ofi_cvd_divergence import DivergenceDetector
from src.utils.config_loader import ConfigLoader

# ============================================================================
# 配置和常量
# ============================================================================

# 环境变量默认值
DEFAULT_CONFIG = {
    'SYMBOLS': 'BTCUSDT,ETHUSDT',
    'RUN_HOURS': '72',
    'PARQUET_ROTATE_SEC': '60',
    'WSS_PING_INTERVAL': '20',
    'DEDUP_LRU': '8192',
    'Z_MODE': 'delta',
    'SCALE_MODE': 'hybrid',
    'MAD_MULTIPLIER': '1.8',
    'SCALE_FAST_WEIGHT': '0.20',
    'HALF_LIFE_SEC': '600',
    'WINSOR_LIMIT': '8',
    'PROMETHEUS_PORT': '8009',
    'LOG_LEVEL': 'INFO',
    'OUTPUT_DIR': 'data/ofi_cvd',
    'ARTIFACTS_DIR': 'artifacts'
}

# Prometheus指标
METRICS = {
    'recv_rate_tps': Gauge('recv_rate_tps', '接收速率 (tps)', ['symbol']),
    'ws_reconnects_total': Counter('ws_reconnects_total', 'WebSocket重连次数', ['symbol']),
    'dedup_hits_total': Counter('dedup_hits_total', '去重命中次数', ['symbol']),
    'latency_ms': Histogram('latency_ms', '延迟分布 (ms)', ['symbol'], buckets=[10, 30, 60, 120, 300, 600]),
    'cvd_scale_median': Gauge('cvd_scale_median', 'CVD Scale中位数', ['symbol']),
    'cvd_floor_hit_rate': Gauge('cvd_floor_hit_rate', 'CVD Floor命中率', ['symbol']),
    'write_errors_total': Counter('write_errors_total', '写入错误次数', ['kind']),
    'parquet_flush_sec': Histogram('parquet_flush_sec', 'Parquet刷新耗时 (s)', ['kind']),
    'data_rows_total': Counter('data_rows_total', '数据行数', ['symbol', 'kind']),
    'empty_buckets_total': Counter('empty_buckets_total', '空桶数量'),
    'duplicate_rate': Gauge('duplicate_rate', '重复率', ['symbol'])
}

# ============================================================================
# 数据采集器类
# ============================================================================

class OFICVDHarvester:
    """OFI+CVD数据采集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config['SYMBOLS'].split(',')
        self.run_hours = int(config['RUN_HOURS'])
        self.rotate_sec = int(config['PARQUET_ROTATE_SEC'])
        self.dedup_lru_size = int(config['DEDUP_LRU'])
        
        # 创建输出目录
        self.output_dir = Path(config['OUTPUT_DIR'])
        self.artifacts_dir = Path(config['ARTIFACTS_DIR'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        self.log_dir = self.artifacts_dir / 'run_logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.setup_logging()
        self.setup_components()
        self.setup_dedup()
        self.setup_checkpoint()
        
        # 数据缓冲区
        self.data_buffers = {kind: {symbol: [] for symbol in self.symbols} 
                           for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']}
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_rows': {symbol: {kind: 0 for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']} 
                          for symbol in self.symbols},
            'reconnects': {symbol: 0 for symbol in self.symbols},
            'duplicates': {symbol: 0 for symbol in self.symbols},
            'errors': {kind: 0 for kind in ['prices', 'ofi', 'cvd', 'fusion', 'events']}
        }
        
        self.logger.info(f"OFI+CVD采集器初始化完成，目标运行{self.run_hours}小时")
        self.logger.info(f"交易对: {self.symbols}")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        # 打印配置指纹
        fingerprint = {
            'Z_MODE': self.config['Z_MODE'],
            'SCALE_MODE': self.config['SCALE_MODE'],
            'MAD_MULTIPLIER': self.config['MAD_MULTIPLIER'],
            'SCALE_FAST_WEIGHT': self.config['SCALE_FAST_WEIGHT'],
            'HALF_LIFE_SEC': self.config['HALF_LIFE_SEC'],
            'WINSOR_LIMIT': self.config['WINSOR_LIMIT']
        }
        self.logger.info(f"配置指纹: {fingerprint}")
    
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = self.log_dir / f'harvest_{timestamp}.log'
        
        logging.basicConfig(
            level=getattr(logging, self.config['LOG_LEVEL']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OFICVDHarvester')
    
    def setup_components(self):
        """初始化计算组件"""
        # 加载配置
        config_loader = ConfigLoader()
        
        # 初始化OFI计算器
        self.ofi_calculator = RealOFICalculator(
            levels=5,
            weights=[0.4, 0.3, 0.2, 0.08, 0.02],
            z_mode=self.config['Z_MODE'],
            scale_mode=self.config['SCALE_MODE'],
            mad_multiplier=float(self.config['MAD_MULTIPLIER']),
            scale_fast_weight=float(self.config['SCALE_FAST_WEIGHT']),
            half_life_trades=int(float(self.config['HALF_LIFE_SEC']) * 2),  # 转换为交易数
            winsor_limit=float(self.config['WINSOR_LIMIT'])
        )
        
        # 初始化CVD计算器
        self.cvd_calculator = RealCVDCalculator(
            z_mode=self.config['Z_MODE'],
            scale_mode=self.config['SCALE_MODE'],
            mad_multiplier=float(self.config['MAD_MULTIPLIER']),
            scale_fast_weight=float(self.config['SCALE_FAST_WEIGHT']),
            half_life_trades=int(float(self.config['HALF_LIFE_SEC']) * 2),
            winsor_limit=float(self.config['WINSOR_LIMIT'])
        )
        
        # 初始化其他组件
        self.regime_classifier = RegimeClassifier()
        self.divergence_detector = DivergenceDetector()
        
        self.logger.info("计算组件初始化完成")
    
    def setup_dedup(self):
        """设置去重缓存"""
        self.dedup_cache = {symbol: deque(maxlen=self.dedup_lru_size) 
                          for symbol in self.symbols}
    
    def setup_checkpoint(self):
        """设置检查点"""
        self.checkpoint_file = self.artifacts_dir / 'state' / 'checkpoint.json'
        self.checkpoint_file.parent.mkdir(exist_ok=True)
        
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoint = json.load(f)
            self.logger.info(f"从检查点恢复: {self.checkpoint}")
        else:
            self.checkpoint = {
                'last_offset': {},
                'last_timestamp': {},
                'run_tag': datetime.now().strftime('%Y%m%d_%H%M')
            }
    
    def save_checkpoint(self, symbol: str, offset: int, timestamp: int):
        """保存检查点"""
        self.checkpoint['last_offset'][symbol] = offset
        self.checkpoint['last_timestamp'][symbol] = timestamp
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def update_checkpoint_from_trade(self, symbol: str, trade_data: Dict[str, Any]):
        """从交易数据更新检查点"""
        event_ts = trade_data.get('event_ts_ms', 0)
        trade_id = trade_data.get('agg_trade_id', '')
        
        if event_ts and trade_id:
            self.save_checkpoint(symbol, trade_id, event_ts)
    
    def is_duplicate(self, symbol: str, trade_id: str) -> bool:
        """检查是否重复"""
        if trade_id in self.dedup_cache[symbol]:
            return True
        
        self.dedup_cache[symbol].append(trade_id)
        return False
    
    def calculate_metrics(self, symbol: str, data: Dict[str, Any]):
        """计算指标"""
        try:
            # 计算OFI
            ofi_result = self.ofi_calculator.update(data)
            
            # 计算CVD
            cvd_result = self.cvd_calculator.update(data)
            
            # 计算市场状态
            regime = self.regime_classifier.classify(data)
            
            # 计算融合指标
            if ofi_result and cvd_result:
                fusion_score = 0.6 * ofi_result.get('ofi_z', 0) + 0.4 * cvd_result.get('z_cvd', 0)
                
                # 维护融合指标窗口用于z标准化
                if not hasattr(self, 'fusion_window'):
                    from collections import deque
                    self.fusion_window = {s: deque() for s in self.symbols}
                
                self.fusion_window[symbol].append(fusion_score)
                if len(self.fusion_window[symbol]) > 3600:  # 约60秒窗口
                    self.fusion_window[symbol].popleft()
                
                # 计算z标准化
                if len(self.fusion_window[symbol]) > 10:  # 至少10个样本
                    import numpy as np
                    scores = list(self.fusion_window[symbol])
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    fusion_score_z = (fusion_score - mean_score) / max(std_score, 1e-8)
                else:
                    fusion_score_z = fusion_score
            else:
                fusion_score = 0
                fusion_score_z = 0
            
            # 检测背离信号
            events = []
            if ofi_result and cvd_result:
                divergence_events = self.divergence_detector.detect(data, ofi_result, cvd_result)
                events.extend(divergence_events)
            
            return {
                'ofi': ofi_result,
                'cvd': cvd_result,
                'regime': regime,
                'fusion': {
                    'score': fusion_score,
                    'score_z': fusion_score_z
                },
                'events': events
            }
        except Exception as e:
            self.logger.error(f"计算指标失败 {symbol}: {e}")
            return None
    
    def create_dataframe(self, kind: str, symbol: str, data: List[Dict]) -> pd.DataFrame:
        """创建DataFrame"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 添加分区列
        df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.date.astype(str)
        df['symbol'] = symbol
        df['kind'] = kind
        
        return df
    
    def write_parquet(self, kind: str, symbol: str, df: pd.DataFrame):
        """写入Parquet文件"""
        if df.empty:
            return
        
        try:
            # 创建分区目录
            date_str = df['date'].iloc[0]
            output_path = self.output_dir / f'date={date_str}' / f'symbol={symbol}' / f'kind={kind}'
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = int(time.time())
            filename = f'part-{timestamp}.parquet'
            file_path = output_path / filename
            
            # 写入Parquet（原子写）
            with METRICS['parquet_flush_sec'].labels(kind=kind).time():
                tmp_file = file_path.with_suffix('.parquet.tmp')
                df.to_parquet(tmp_file, compression='snappy', index=False)
                tmp_file.replace(file_path)
            
            # 更新统计
            self.stats['total_rows'][symbol][kind] += len(df)
            METRICS['data_rows_total'].labels(symbol=symbol, kind=kind).inc(len(df))
            
            self.logger.debug(f"写入 {kind} 数据: {len(df)} 行 -> {file_path}")
            
        except Exception as e:
            self.logger.error(f"写入Parquet失败 {kind}/{symbol}: {e}")
            METRICS['write_errors_total'].labels(kind=kind).inc()
            self.stats['errors'][kind] += 1
    
    def flush_buffers(self):
        """刷新所有缓冲区"""
        for kind in self.data_buffers:
            for symbol in self.symbols:
                if self.data_buffers[kind][symbol]:
                    df = self.create_dataframe(kind, symbol, self.data_buffers[kind][symbol])
                    if not df.empty:
                        self.write_parquet(kind, symbol, df)
                    self.data_buffers[kind][symbol].clear()
        
        # 更新窗口统计到Prometheus
        if hasattr(self, 'cvd_diag'):
            for s in self.symbols:
                sc = list(self.cvd_diag[s]['scale'])
                fr = list(self.cvd_diag[s]['floor'])
                if sc:
                    import numpy as np
                    METRICS['cvd_scale_median'].labels(symbol=s).set(float(np.median(sc)))
                if fr:
                    METRICS['cvd_floor_hit_rate'].labels(symbol=s).set(float(np.mean(fr)))
        
        # 更新重复率
        for s in self.symbols:
            total = sum(self.stats['total_rows'][s].values())
            dups = self.stats['duplicates'][s]
            if total > 0:
                METRICS['duplicate_rate'].labels(symbol=s).set(dups / max(1, total))
    
    def process_trade(self, symbol: str, trade_data: Dict[str, Any]):
        """处理单笔交易"""
        try:
            # 去重检查
            trade_id = trade_data.get('agg_trade_id', '')
            if trade_id and self.is_duplicate(symbol, trade_id):
                METRICS['dedup_hits_total'].labels(symbol=symbol).inc()
                self.stats['duplicates'][symbol] += 1
                return
            
            # 计算延迟
            event_ts = trade_data.get('event_ts_ms', 0)
            recv_ts = trade_data.get('recv_ts_ms', 0)
            latency_ms = recv_ts - event_ts if recv_ts and event_ts else 0
            
            # 记录延迟指标
            if latency_ms > 0:
                METRICS['latency_ms'].labels(symbol=symbol).observe(latency_ms)
            
            # 60s 滑窗 TPS
            if not hasattr(self, 'tps_windows'):
                from collections import deque
                self.tps_windows = {s: deque() for s in self.symbols}
            
            now_sec = trade_data.get('recv_ts_ms', int(time.time() * 1000)) / 1000.0
            win = self.tps_windows[symbol]
            win.append(now_sec)
            
            # 清理超过60秒的数据
            while win and (now_sec - win[0]) > 60.0:
                win.popleft()
            
            tps = len(win) / 60.0
            METRICS['recv_rate_tps'].labels(symbol=symbol).set(tps)
            
            # 准备价格数据
            price_data = {
                'ts_ms': trade_data.get('event_ts_ms', trade_data.get('ts_ms', 0)),
                'event_ts_ms': event_ts,
                'symbol': symbol,
                'price': trade_data.get('price', 0),
                'qty': trade_data.get('qty', 0),
                'agg_trade_id': trade_id,
                'latency_ms': latency_ms,
                'recv_rate_tps': tps
            }
            self.data_buffers['prices'][symbol].append(price_data)
            
            # 更新检查点
            self.update_checkpoint_from_trade(symbol, trade_data)
            
            # 计算指标
            metrics_result = self.calculate_metrics(symbol, trade_data)
            if metrics_result:
                # OFI数据
                if metrics_result['ofi']:
                    ofi_data = {
                        'ts_ms': trade_data.get('ts_ms', 0),
                        'symbol': symbol,
                        'ofi_value': metrics_result['ofi'].get('ofi_value', 0),
                        'ofi_z': metrics_result['ofi'].get('ofi_z', 0),
                        'scale': metrics_result['ofi'].get('scale', 0),
                        'regime': metrics_result['regime']
                    }
                    self.data_buffers['ofi'][symbol].append(ofi_data)
                
                # CVD数据
                if metrics_result['cvd']:
                    cvd_data = {
                        'ts_ms': trade_data.get('ts_ms', 0),
                        'symbol': symbol,
                        'cvd': metrics_result['cvd'].get('cvd', 0),
                        'delta': metrics_result['cvd'].get('delta', 0),
                        'z_raw': metrics_result['cvd'].get('z_raw', 0),
                        'z_cvd': metrics_result['cvd'].get('z_cvd', 0),
                        'scale': metrics_result['cvd'].get('scale', 0),
                        'sigma_floor': metrics_result['cvd'].get('sigma_floor', 0),
                        'floor_used': metrics_result['cvd'].get('floor_used', 0),
                        'regime': metrics_result['regime']
                    }
                    self.data_buffers['cvd'][symbol].append(cvd_data)
                    
                    # 维护CVD诊断窗口
                    if not hasattr(self, 'cvd_diag'):
                        from collections import deque
                        self.cvd_diag = {s: {'scale': deque(), 'floor': deque()} for s in self.symbols}
                    
                    self.cvd_diag[symbol]['scale'].append(metrics_result['cvd'].get('scale', 0.0))
                    self.cvd_diag[symbol]['floor'].append(1.0 if metrics_result['cvd'].get('floor_used', False) else 0.0)
                    
                    # 截窗（保持约60秒的数据）
                    if len(self.cvd_diag[symbol]['scale']) > 3600:  # 约= 60s * ~TPS(60)
                        self.cvd_diag[symbol]['scale'].popleft()
                        self.cvd_diag[symbol]['floor'].popleft()
                
                # 融合指标数据
                fusion_data = {
                    'ts_ms': trade_data.get('ts_ms', 0),
                    'symbol': symbol,
                    'score': metrics_result['fusion']['score'],
                    'score_z': metrics_result['fusion']['score_z'],
                    'regime': metrics_result['regime']
                }
                self.data_buffers['fusion'][symbol].append(fusion_data)
                
                # 事件数据
                for event in metrics_result['events']:
                    event_data = {
                        'ts_ms': trade_data.get('ts_ms', 0),
                        'symbol': symbol,
                        'event_type': event.get('type', 'unknown'),
                        'meta_json': json.dumps(event.get('meta', {}))
                    }
                    self.data_buffers['events'][symbol].append(event_data)
            
        except Exception as e:
            self.logger.error(f"处理交易失败 {symbol}: {e}")
    
    async def run_harvest(self):
        """运行数据采集"""
        self.logger.info("开始数据采集...")
        self.stats['start_time'] = datetime.now()
        
        # 启动Prometheus服务器
        prometheus_port = int(self.config['PROMETHEUS_PORT'])
        start_http_server(prometheus_port)
        self.logger.info(f"Prometheus监控启动: http://localhost:{prometheus_port}/metrics")
        
        # 创建WebSocket适配器
        ws_adapter = BinanceWebSocketAdapter()
        
        # 设置回调函数
        def on_trade(symbol: str, trade_data: Dict[str, Any]):
            self.process_trade(symbol, trade_data)
        
        def on_reconnect(symbol: str):
            METRICS['ws_reconnects_total'].labels(symbol=symbol).inc()
            self.stats['reconnects'][symbol] += 1
            self.logger.warning(f"WebSocket重连: {symbol}")
        
        # 启动WebSocket连接（传递参数）
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(
                ws_adapter.subscribe_trades(
                    symbol, 
                    on_trade, 
                    on_reconnect,
                    ping_interval=int(self.config['WSS_PING_INTERVAL']),
                    heartbeat_timeout=30,
                    reconnect_delay=1.0,
                    max_reconnect_attempts=10
                )
            )
            tasks.append(task)
        
        # 定期刷新缓冲区
        async def flush_worker():
            while True:
                await asyncio.sleep(self.rotate_sec)
                self.flush_buffers()
                self.logger.info(f"缓冲区已刷新，当前统计: {self.stats['total_rows']}")
        
        flush_task = asyncio.create_task(flush_worker())
        tasks.append(flush_task)
        
        # 运行指定时间
        try:
            await asyncio.sleep(self.run_hours * 3600)
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，正在停止...")
        finally:
            # 停止所有任务
            for task in tasks:
                task.cancel()
            
            # 最后刷新缓冲区
            self.flush_buffers()
            
            # 先记录结束时间，再生成报告
            self.stats['end_time'] = datetime.now()
            self.generate_final_report()
            
            self.logger.info("数据采集完成")
    
    def generate_final_report(self):
        """生成最终报告"""
        report = {
            'run_info': {
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
                'duration_hours': (self.stats['end_time'] - self.stats['start_time']).total_seconds() / 3600 
                                if self.stats['start_time'] and self.stats['end_time'] else 0,
                'symbols': self.symbols,
                'run_tag': self.checkpoint.get('run_tag', 'unknown')
            },
            'data_quality': {
                'total_rows': self.stats['total_rows'],
                'reconnects': self.stats['reconnects'],
                'duplicates': self.stats['duplicates'],
                'errors': self.stats['errors']
            },
            'config_fingerprint': {
                'z_mode': self.config['Z_MODE'],
                'scale_mode': self.config['SCALE_MODE'],
                'mad_multiplier': self.config['MAD_MULTIPLIER'],
                'scale_fast_weight': self.config['SCALE_FAST_WEIGHT'],
                'half_life_sec': self.config['HALF_LIFE_SEC'],
                'winsor_limit': self.config['WINSOR_LIMIT']
            }
        }
        
        # 保存报告
        report_file = self.artifacts_dir / 'dq_reports' / f'dq_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"最终报告已保存: {report_file}")
        self.logger.info(f"数据采集统计: {json.dumps(report['data_quality'], indent=2)}")

# ============================================================================
# 主程序
# ============================================================================

def load_config() -> Dict[str, Any]:
    """加载配置"""
    config = DEFAULT_CONFIG.copy()
    
    # 从环境变量加载
    for key in config:
        if key in os.environ:
            config[key] = os.environ[key]
    
    return config

def print_config_fingerprint(config: Dict[str, Any]):
    """打印配置指纹"""
    fingerprint = {
        'Z_MODE': config['Z_MODE'],
        'SCALE_MODE': config['SCALE_MODE'],
        'MAD_MULTIPLIER': config['MAD_MULTIPLIER'],
        'SCALE_FAST_WEIGHT': config['SCALE_FAST_WEIGHT'],
        'HALF_LIFE_SEC': config['HALF_LIFE_SEC'],
        'WINSOR_LIMIT': config['WINSOR_LIMIT']
    }
    
    print("=" * 60)
    print("配置指纹:")
    for key, value in fingerprint.items():
        print(f"  {key}: {value}")
    print("=" * 60)

async def run_precheck(harvester: OFICVDHarvester) -> bool:
    """运行10分钟预检"""
    print("开始10分钟预检...")
    start_time = time.time()
    
    # 重置计数
    for s in harvester.symbols:
        METRICS['ws_reconnects_total'].labels(symbol=s)._value.set(0)
        METRICS['dedup_hits_total'].labels(symbol=s)._value.set(0)
    
    # 运行短采集（复用同一适配器）
    async def _short_run():
        await asyncio.sleep(600)  # 10分钟
    
    try:
        await _short_run()
    finally:
        harvester.flush_buffers()
    
    # 评估：重连、重复、延迟、是否有落盘
    violations = []
    
    # 重连 < 3 次/10m
    for s in harvester.symbols:
        reconnects = METRICS['ws_reconnects_total'].labels(symbol=s)._value.get()
        if reconnects > 3:
            violations.append(f"reconnects[{s}]={reconnects} > 3")
    
    # 重复率 < 0.2%
    for s in harvester.symbols:
        total = sum(harvester.stats['total_rows'][s].values())
        dups = harvester.stats['duplicates'][s]
        if total > 0 and dups / total > 0.002:
            violations.append(f"duplicate_rate[{s}]={dups/total:.4f} > 0.002")
    
    # 必须有 prices/cvd 文件
    base = Path(harvester.output_dir)
    found_prices = list(base.glob("date=*/symbol=*/kind=prices/*.parquet"))
    found_cvd = list(base.glob("date=*/symbol=*/kind=cvd/*.parquet"))
    if not found_prices or not found_cvd:
        violations.append("no parquet flushed for prices/cvd in precheck")
    
    # 延迟检查（如果有数据）
    if found_prices:
        try:
            # 读取最新的prices文件检查延迟
            latest_file = max(found_prices, key=os.path.getctime)
            df = pd.read_parquet(latest_file)
            if 'latency_ms' in df.columns:
                latency_p99 = df['latency_ms'].quantile(0.99)
                if latency_p99 > 120:
                    violations.append(f"latency_p99={latency_p99:.1f}ms > 120ms")
        except Exception as e:
            violations.append(f"latency_check_failed: {e}")
    
    ok = (len(violations) == 0)
    print("预检完成：", "✅" if ok else f"❌ {violations}")
    return ok

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OFI+CVD数据采集器')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--precheck-only', action='store_true', help='仅运行预检')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 打印配置指纹
    print_config_fingerprint(config)
    
    # 创建采集器
    harvester = OFICVDHarvester(config)
    
    # 运行预检
    if not await run_precheck(harvester):
        print("预检失败，退出")
        return
    
    if args.precheck_only:
        print("预检完成，退出")
        return
    
    # 运行数据采集
    await harvester.run_harvest()

if __name__ == '__main__':
    asyncio.run(main())
