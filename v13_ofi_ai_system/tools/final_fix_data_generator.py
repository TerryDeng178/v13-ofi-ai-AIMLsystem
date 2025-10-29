#!/usr/bin/env python3
"""
最终修复测试数据生成器
应用final_fix_patch.yaml配置，生成修复后的信号数据
"""

import os
import sys
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加core路径以导入CoreAlgorithm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from core_algo import CoreAlgorithm, SignalConfig, SignalData
except ImportError as e:
    print(f"ERROR: Failed to import CoreAlgorithm: {e}")
    sys.exit(1)

class FinalFixDataGenerator:
    """最终修复数据生成器"""
    
    def __init__(self, config_file: str = "config/final_fix_patch.yaml"):
        self.config_file = Path(config_file)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
        self.algorithms = {}
        self.fix_config = {}
        
    def load_fix_config(self):
        """加载修复配置"""
        if not self.config_file.exists():
            print(f"ERROR: Config file not found: {self.config_file}")
            sys.exit(1)
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.fix_config = yaml.safe_load(f)
            
        print("=== 最终修复配置 ===")
        print(f"强信号阈值: ±{self.fix_config['fusion']['thresholds']['fuse_strong_buy']}")
        print(f"弱一致性门槛: {self.fix_config['fusion']['consistency']['min_consistency']}")
        print(f"强一致性门槛: {self.fix_config['fusion']['consistency']['strong_min_consistency']}")
        print(f"背离强度: {self.fix_config['divergence']['min_strength']}")
        
    def init_algorithms_with_fix(self):
        """使用修复配置初始化算法实例"""
        for symbol in self.symbols:
            try:
                # 创建基础配置
                config = SignalConfig()
                
                # 应用修复配置
                fusion_config = self.fix_config.get('fusion', {})
                divergence_config = self.fix_config.get('divergence', {})
                
                # 更新融合阈值
                thresholds = fusion_config.get('thresholds', {})
                config.fusion_thresholds = {
                    'fuse_buy': thresholds.get('fuse_buy', 1.0),
                    'fuse_sell': thresholds.get('fuse_sell', -1.0),
                    'fuse_strong_buy': thresholds.get('fuse_strong_buy', 2.3),
                    'fuse_strong_sell': thresholds.get('fuse_strong_sell', -2.3)
                }
                
                # 更新一致性门槛
                consistency = fusion_config.get('consistency', {})
                config.fusion_consistency = {
                    'min_consistency': consistency.get('min_consistency', 0.20),
                    'strong_min_consistency': consistency.get('strong_min_consistency', 0.65)
                }
                
                # 更新背离配置
                config.divergence_config = {
                    'min_strength': divergence_config.get('min_strength', 0.90),
                    'min_separation_secs': divergence_config.get('min_separation_secs', 120),
                    'count_conflict_only_when_fusion_ge': divergence_config.get('count_conflict_only_when_fusion_ge', 1.0)
                }
                
                # 创建算法实例
                algo = CoreAlgorithm(symbol, config)
                self.algorithms[symbol] = algo
                print(f"Initialized algorithm for {symbol} with fix config")
                
            except Exception as e:
                print(f"Failed to initialize algorithm for {symbol}: {e}")
                
    def generate_realistic_test_data(self, symbol: str, duration_minutes: int = 15) -> List[SignalData]:
        """生成基于真实数据特征的测试数据"""
        if symbol not in self.algorithms:
            print(f"Algorithm not initialized for {symbol}")
            return []
            
        algo = self.algorithms[symbol]
        signals = []
        
        # 基础价格（基于真实市场）
        base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 300.0,
            'SOLUSDT': 100.0,
            'XRPUSDT': 0.5,
            'DOGEUSDT': 0.08
        }
        
        base_price = base_prices.get(symbol, 100.0)
        base_time = int(time.time() * 1000) - (duration_minutes * 60 * 1000)
        
        # 生成更真实的数据模式
        for i in range(duration_minutes * 60):  # 每分钟60个数据点
            ts_ms = base_time + i * 1000
            
            # 模拟真实的价格波动和订单簿
            price_offset = (i % 300 - 150) * 0.0003  # ±0.045%波动
            price = base_price * (1 + price_offset)
            
            # 生成5档订单簿
            bids = []
            asks = []
            for level in range(5):
                bid_price = price - (level + 1) * 0.05
                ask_price = price + (level + 1) * 0.05
                bid_size = 15.0 - level * 2.0
                ask_size = 15.0 - level * 2.0
                
                bids.append([round(bid_price, 2), round(bid_size, 2)])
                asks.append([round(ask_price, 2), round(ask_size, 2)])
            
            # 模拟交易
            qty = 0.2 + (i % 30) * 0.05  # 0.2-1.7数量
            is_buy = (i % 4) != 0  # 3/4概率为买入
            
            try:
                # 更新OFI计算器
                ofi_result = algo.update_ofi(bids, asks, ts_ms)
                z_ofi = ofi_result.get('z_ofi', 0.0)
                
                # 更新CVD计算器
                cvd_result = algo.update_cvd(price=price, qty=qty, is_buy=is_buy, event_time_ms=ts_ms)
                z_cvd = cvd_result.get('z_cvd', 0.0)
                
                # 计算市场指标
                trade_rate = 80.0  # 每分钟80笔交易
                realized_vol = 0.008  # 0.8%波动率
                spread_bps = 4.0  # 4个基点价差
                missing_msgs_rate = 0.00005  # 0.005%缺失率
                
                # 处理信号
                signal_data = algo.process_signal(
                    ts_ms=ts_ms,
                    symbol=symbol,
                    z_ofi=z_ofi,
                    z_cvd=z_cvd,
                    price=price,
                    trade_rate=trade_rate,
                    realized_vol=realized_vol,
                    spread_bps=spread_bps,
                    missing_msgs_rate=missing_msgs_rate
                )
                
                if signal_data:
                    signals.append(signal_data)
                    
            except Exception as e:
                print(f"Error processing data for {symbol} at index {i}: {e}")
                continue
                
        print(f"Generated {len(signals)} signals for {symbol}")
        return signals
        
    def save_signals_to_jsonl(self, signals: List[SignalData], symbol: str):
        """保存信号到JSONL文件"""
        output_dir = Path("runtime/ready/signal") / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按分钟分组保存
        current_minute = None
        current_file = None
        current_count = 0
        
        for signal in signals:
            signal_time = datetime.fromtimestamp(signal.ts_ms / 1000)
            minute_key = signal_time.strftime("%Y%m%d_%H%M")
            
            if minute_key != current_minute:
                if current_file:
                    current_file.close()
                    
                file_path = output_dir / f"signals_{minute_key}.jsonl"
                current_file = open(file_path, 'w', encoding='utf-8')
                current_minute = minute_key
                current_count = 0
                
            # 写入信号数据
            signal_dict = {
                "timestamp": signal_time.isoformat(),
                "ts_ms": signal.ts_ms,
                "symbol": symbol,
                "score": signal.score,
                "z_ofi": signal.z_ofi,
                "z_cvd": signal.z_cvd,
                "regime": signal.regime,
                "div_type": signal.div_type,
                "confirm": signal.confirm,
                "gating": signal.gating,
                "guard_reason": None
            }
            
            current_file.write(json.dumps(signal_dict, ensure_ascii=False) + "\n")
            current_count += 1
            
        if current_file:
            current_file.close()
            
        print(f"Saved {current_count} signals for {symbol} to {output_dir}")
        
    def generate_gate_stats(self, symbols: List[str]):
        """生成gate_stats.jsonl文件"""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        gate_stats_file = artifacts_dir / "gate_stats.jsonl"
        
        with open(gate_stats_file, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                if symbol in self.algorithms:
                    algo = self.algorithms[symbol]
                    
                    stats_record = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "gate_stats",
                        "symbol": symbol,
                        "total_signals": algo.stats.get('total_updates', 0),
                        "gate_reasons": algo.gate_reason_stats.copy(),
                        "current_regime": getattr(algo, '_current_regime', 'unknown'),
                        "guard_active": algo.guard_active,
                        "guard_reason": algo.guard_reason
                    }
                    
                    f.write(json.dumps(stats_record, ensure_ascii=False) + "\n")
                    
        print(f"Generated gate_stats.jsonl with {len(symbols)} symbols")

def main():
    """主函数"""
    print("=== 最终修复测试数据生成器 ===")
    print("目标：将Strong ratio从8.75%压回1-3%区间")
    
    # 初始化生成器
    generator = FinalFixDataGenerator()
    generator.load_fix_config()
    generator.init_algorithms_with_fix()
    
    # 生成所有交易对的测试数据
    all_signals = []
    
    for symbol in generator.symbols:
        print(f"\n--- 处理 {symbol} ---")
        try:
            signals = generator.generate_realistic_test_data(symbol, duration_minutes=15)
            all_signals.extend(signals)
            
            # 保存信号到JSONL
            generator.save_signals_to_jsonl(signals, symbol)
            
        except Exception as e:
            print(f"ERROR: Failed to process {symbol}: {e}")
            continue
            
    # 生成gate_stats
    generator.generate_gate_stats(generator.symbols)
    
    print(f"\n=== 最终修复测试数据生成完成 ===")
    print(f"总共生成 {len(all_signals)} 个信号")
    print(f"涉及 {len(generator.symbols)} 个交易对")
    print(f"数据已保存到 runtime/ready/signal/ 和 artifacts/")
    print(f"\n下一步：运行复测脚本验证修复效果")

if __name__ == "__main__":
    main()
