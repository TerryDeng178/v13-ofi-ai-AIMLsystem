#!/usr/bin/env python3
"""
P0-B 60分钟测试进度检查脚本
"""
import sys
import io
from pathlib import Path
from datetime import datetime

# Windows兼容性
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_progress():
    data_dir = Path(__file__).parent.parent / "data" / "cvd_p0b_60min_ethusdt"
    
    print("=" * 60)
    print("P0-B 60分钟正式验收 - 进度检查")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not data_dir.exists():
        print("❌ 测试尚未开始或数据目录不存在")
        print(f"   期望目录: {data_dir}")
        return
    
    # 查找parquet文件
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("⏳ 测试运行中，尚未生成数据文件")
        print("   （前5分钟可能无输出，属正常现象）")
        return
    
    # 读取最新的parquet文件
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    
    try:
        import pandas as pd
        df = pd.read_parquet(latest_file)
        
        n_records = len(df)
        if n_records == 0:
            print("⏳ 数据文件已创建但尚无记录")
            return
        
        # 计算时间跨度
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'ts' in df.columns:
            time_col = 'ts'
        else:
            print("⚠️ 找不到时间戳列")
            return
        
        elapsed_seconds = df[time_col].max() - df[time_col].min()
        elapsed_minutes = elapsed_seconds / 60
        progress_pct = (elapsed_seconds / 3600) * 100
        
        print(f"✅ 测试进行中")
        print(f"   数据文件: {latest_file.name}")
        print(f"   记录数: {n_records:,} 条")
        print(f"   已运行: {elapsed_minutes:.1f} 分钟 ({elapsed_seconds:.0f}秒)")
        print(f"   进度: {progress_pct:.1f}% / 100%")
        print(f"   速率: {n_records/elapsed_seconds:.1f} 条/秒")
        
        # 估算剩余时间
        if progress_pct > 0:
            total_estimated = elapsed_seconds / (progress_pct / 100)
            remaining_seconds = total_estimated - elapsed_seconds
            remaining_minutes = remaining_seconds / 60
            eta = datetime.now().timestamp() + remaining_seconds
            eta_str = datetime.fromtimestamp(eta).strftime('%H:%M:%S')
            print(f"   预计完成: {eta_str} (剩余 {remaining_minutes:.0f} 分钟)")
        
        print()
        
        # 关键指标快速检查
        if 'agg_dup_count' in df.columns or 'agg_backward_count' in df.columns:
            print("🔍 关键指标快照（最新值）:")
            if 'cvd' in df.columns:
                print(f"   CVD: {df['cvd'].iloc[-1]:.2f}")
            if 'z_cvd' in df.columns:
                z_val = df['z_cvd'].iloc[-1]
                if z_val is not None and not pd.isna(z_val):
                    print(f"   Z-score: {z_val:.2f}")
        
    except ImportError:
        print("⚠️ 需要pandas来分析数据，跳过详细统计")
        print(f"   数据文件: {latest_file.name}")
        print(f"   文件大小: {latest_file.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ 读取数据时出错: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_progress()

