"""
Bucket Attribution Analysis Module
分桶归因分析：sig_type × OFI_z分位 × spread分位 × depth分位 × session(是/否)
"""
import pandas as pd
import numpy as np
import json
import os

def analyze_buckets(trades_df, signals_df, params):
    """
    分桶归因分析
    维度：sig_type × OFI_z分位 × spread分位 × depth分位 × session(是/否)
    """
    if len(trades_df) == 0:
        return {"buckets": [], "summary": {"total_buckets": 0, "positive_ir_buckets": 0}}
    
    # 合并交易数据和信号数据
    merged = pd.merge(trades_df, signals_df[['ts', 'sig_type', 'ofi_z', 'spread_bps', 'depth_ratio', 'in_session']], 
                     left_on='entry_ts', right_on='ts', how='left')
    
    # 计算分位数
    merged['ofi_z_quartile'] = pd.qcut(merged['ofi_z'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    merged['spread_quartile'] = pd.qcut(merged['spread_bps'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    merged['depth_quartile'] = pd.qcut(merged['depth_ratio'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # 创建分桶
    buckets = []
    bucket_groups = merged.groupby(['sig_type', 'ofi_z_quartile', 'spread_quartile', 'depth_quartile', 'in_session'])
    
    for (sig_type, ofi_q, spread_q, depth_q, in_session), group in bucket_groups:
        if len(group) < 2:  # 至少需要2笔交易才能计算IR
            continue
            
        # 计算桶指标
        total_trades = len(group)
        winning_trades = len(group[group['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        avg_pnl = group['pnl'].mean()
        avg_holding = group['holding_sec'].mean()
        
        # 计算IR (简化版)
        returns = group['pnl'] / group['qty_usd'].abs()
        returns = returns.dropna()
        if len(returns) > 1:
            ir = returns.mean() / (returns.std() + 1e-9) * np.sqrt(365*24*60*60)
        else:
            ir = 0.0
            
        bucket = {
            'bucket_id': f"{sig_type}_{ofi_q}_{spread_q}_{depth_q}_{in_session}",
            'sig_type': sig_type,
            'ofi_z_quartile': ofi_q,
            'spread_quartile': spread_q,
            'depth_quartile': depth_q,
            'in_session': in_session,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_holding_sec': avg_holding,
            'ir': ir,
            'positive_ir': ir > 0
        }
        buckets.append(bucket)
    
    # 计算汇总
    total_buckets = len(buckets)
    positive_ir_buckets = len([b for b in buckets if b['positive_ir']])
    
    summary = {
        'total_buckets': total_buckets,
        'positive_ir_buckets': positive_ir_buckets,
        'positive_ir_ratio': positive_ir_buckets / total_buckets if total_buckets > 0 else 0
    }
    
    return {
        'buckets': buckets,
        'summary': summary
    }

def generate_bucket_report(bucket_results, config_name):
    """生成分桶报告"""
    output_dir = "examples/out"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细分桶数据
    with open(f"{output_dir}/bucket_analysis_{config_name}.json", "w", encoding="utf-8") as f:
        json.dump(bucket_results, f, indent=2, ensure_ascii=False)
    
    # 生成正IR桶列表
    positive_buckets = [b for b in bucket_results['buckets'] if b['positive_ir']]
    
    report = {
        'config_name': config_name,
        'total_buckets': bucket_results['summary']['total_buckets'],
        'positive_ir_buckets': bucket_results['summary']['positive_ir_buckets'],
        'positive_ir_ratio': bucket_results['summary']['positive_ir_ratio'],
        'positive_buckets': positive_buckets
    }
    
    with open(f"{output_dir}/bucket_report_{config_name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report
