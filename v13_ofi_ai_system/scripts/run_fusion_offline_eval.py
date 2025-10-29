"""
OFI_CVD_Fusion 离线数据评测脚本

基于本地采集数据评估融合组件的信号质量

Author: V13 QA Team
Date: 2025-10-28
"""

import sys
import os
import importlib.util
from pathlib import Path
import time
import csv

# 动态导入依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[ERROR] numpy 未安装，请安装: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[ERROR] pandas 未安装，请安装: pip install pandas")
    sys.exit(1)

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy 未安装，将跳过 IC 计算")


def load_fusion_module():
    """动态加载 ofi_cvd_fusion 模块"""
    project_root = Path(__file__).parent.parent
    fusion_path = project_root / "src" / "ofi_cvd_fusion.py"
    
    if not fusion_path.exists():
        raise FileNotFoundError(f"找不到 ofi_cvd_fusion.py: {fusion_path}")
    
    spec = importlib.util.spec_from_file_location("ofi_cvd_fusion", str(fusion_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def find_data_files(base_dir):
    """搜索数据文件"""
    data_files = []
    
    if not base_dir.exists():
        return data_files
    
    # 搜索所有 Parquet 文件
    try:
        for file_path in base_dir.rglob("*.parquet"):
            if file_path.is_file():
                # 过滤掉死信队列文件
                if 'deadletter' not in str(file_path):
                    data_files.append(file_path)
    except Exception as e:
        print(f"  [WARN] 搜索文件时出错: {e}")
    
    return data_files


def detect_column_names(df):
    """自动检测列名"""
    column_map = {}
    
    # OFI Z-score
    for col in ['ofi_z', 'z_ofi', 'ofi_zscore']:
        if col in df.columns:
            column_map['ofi_z'] = col
            break
    
    # CVD Z-score
    for col in ['z_cvd', 'cvd_z', 'cvd_zscore']:
        if col in df.columns:
            column_map['cvd_z'] = col
            break
    
    # 时间戳
    for col in ['ts_ms', 'second_ts', 'timestamp', 'ts']:
        if col in df.columns:
            column_map['ts'] = col
            break
    
    # 滞后
    for col in ['lag_ms_to_trade', 'lag_sec', 'lag']:
        if col in df.columns:
            column_map['lag'] = col
            break
    
    # 收益列
    column_map['returns'] = []
    for col in df.columns:
        if 'return' in col.lower() or 'ret' in col.lower():
            column_map['returns'].append(col)
    
    # 场景
    if 'scenario_2x2' in df.columns:
        column_map['scenario'] = 'scenario_2x2'
    
    # 价格
    for col in ['mid', 'price', 'last_price']:
        if col in df.columns:
            column_map['price'] = col
            break
    
    return column_map


def load_data_file(file_path):
    """加载数据文件"""
    print(f"  [加载] {file_path.relative_to(file_path.parents[4])}")
    
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return None
        
        if df.empty:
            print(f"  [WARN] 文件为空，跳过")
            return None
        
        print(f"  [OK] 已加载 {len(df)} 行数据")
        return df
    except Exception as e:
        print(f"  [ERROR] 加载失败: {e}")
        return None


def calculate_signal_quality(df_results, df_data, column_map):
    """计算信号质量指标"""
    metrics = {}
    
    # 合并结果与数据
    df_merged = pd.concat([df_results, df_data], axis=1)
    
    # 仅分析非中性信号
    df_non_neutral = df_merged[df_merged['signal'] != 'neutral'].copy()
    
    if len(df_non_neutral) == 0:
        print("  [WARN] 没有非中性信号，跳过质量评估")
        return metrics
    
    # 计算 Hit 率（对于每个收益列）
    for ret_col in column_map.get('returns', []):
        if ret_col not in df_data.columns:
            continue
        
        # 信号方向（1=buy, -1=sell, 0=neutral）
        signal_side = df_non_neutral['signal'].map({
            'buy': 1,
            'strong_buy': 1,
            'sell': -1,
            'strong_sell': -1,
            'neutral': 0
        })
        
        # 实际收益方向
        actual_side = np.sign(df_non_neutral[ret_col])
        
        # Hit 率
        hit_rate = (signal_side == actual_side).mean()
        metrics[f'hit_rate_{ret_col}'] = hit_rate
        
        # IC（相关系数）
        if SCIPY_AVAILABLE:
            ic, ic_pvalue = spearmanr(df_non_neutral['fusion_score'], df_non_neutral[ret_col])
            metrics[f'ic_{ret_col}'] = ic
            metrics[f'ic_pvalue_{ret_col}'] = ic_pvalue
    
    # Top-10% 收益提升（Lift）
    for ret_col in column_map.get('returns', []):
        if ret_col not in df_data.columns:
            continue
        
        # 按 fusion_score 排序
        df_sorted = df_merged.sort_values('fusion_score', ascending=False)
        
        # Top-10%
        n_top = max(1, int(len(df_sorted) * 0.1))
        df_top = df_sorted.head(n_top)
        avg_return_top = df_top[ret_col].mean()
        
        # 全样本平均
        avg_return_all = df_merged[ret_col].mean()
        
        # Lift
        lift = avg_return_top / avg_return_all if avg_return_all != 0 else 0.0
        metrics[f'lift_{ret_col}'] = lift
    
    return metrics


def run_offline_evaluation(fusion, df, column_map, file_name, all_data_files):
    """运行离线评估"""
    results = []
    
    # 获取列名
    ofi_col = column_map.get('ofi_z')
    cvd_col = column_map.get('cvd_z')
    ts_col = column_map.get('ts')
    lag_col = column_map.get('lag')
    
    # 如果当前文件是 fusion 类型，尝试从其他文件合并 OFI/CVD 数据
    if not ofi_col or not cvd_col:
        print(f"  [INFO] 当前文件缺少 OFI/CVD 数据，尝试从其他文件合并...")
        
        # 从文件名提取 symbol
        file_stem = Path(file_name).stem
        symbol = file_stem.split('_')[1]  # 获取 symbol
        
        # 查找对应的 OFI 和 CVD 文件
        ofi_df = None
        cvd_df = None
        for other_file in all_data_files:
            other_stem = Path(other_file).stem
            if symbol in other_stem and 'ofi' in other_stem and ofi_col is None:
                ofi_df = pd.read_parquet(other_file)
                ofi_col = 'ofi_z' if 'ofi_z' in ofi_df.columns else None
            elif symbol in other_stem and 'cvd' in other_stem and cvd_col is None:
                cvd_df = pd.read_parquet(other_file)
                cvd_col = 'z_cvd' if 'z_cvd' in cvd_df.columns else None
        
        # 合并数据
        if ofi_df is not None and cvd_df is not None and ts_col:
            print(f"  [INFO] 正在合并 OFI 和 CVD 数据...")
            merged = pd.merge_asof(
                df[[ts_col]].sort_values(ts_col),
                ofi_df[[ts_col, ofi_col]].rename(columns={ofi_col: 'ofi_z_temp'}),
                on=ts_col,
                direction='nearest',
                tolerance=100  # 允许 100ms 误差
            )
            merged = pd.merge_asof(
                merged,
                cvd_df[[ts_col, cvd_col]].rename(columns={cvd_col: 'cvd_z_temp'}),
                on=ts_col,
                direction='nearest',
                tolerance=100
            )
            df = df.merge(merged, on=ts_col, how='left')
            ofi_col = 'ofi_z_temp'
            cvd_col = 'cvd_z_temp'
    
    if not all([ofi_col, cvd_col, ts_col]):
        print(f"  [ERROR] 缺少必要的列: ofi_z={ofi_col}, cvd_z={cvd_col}, ts={ts_col}")
        return None
    
    # 处理时间戳
    if ts_col in df.columns:
        if df[ts_col].dtype == 'datetime64[ns]':
            df['_ts_epoch'] = df[ts_col].astype('int64') / 1e9  # 转为秒
        elif df[ts_col].max() > 1e12:  # 假设是毫秒
            df['_ts_epoch'] = df[ts_col] / 1000.0
        else:  # 假设是秒
            df['_ts_epoch'] = df[ts_col]
    
    # 处理滞后
    if lag_col and lag_col in df.columns:
        if df[lag_col].max() > 10.0:  # 假设是毫秒
            df['_lag_sec'] = df[lag_col] / 1000.0
        else:  # 假设是秒
            df['_lag_sec'] = df[lag_col]
    else:
        df['_lag_sec'] = 0.0
    
    # 清理数据
    df_clean = df[[ofi_col, cvd_col, '_ts_epoch', '_lag_sec']].copy()
    df_clean = df_clean.dropna()
    
    print(f"  [处理] {len(df_clean)} 行有效数据")
    
    # 顺序调用 fusion.update
    for idx, row in df_clean.iterrows():
        result = fusion.update(
            z_ofi=row[ofi_col],
            z_cvd=row[cvd_col],
            ts=row['_ts_epoch'],
            lag_sec=row['_lag_sec']
        )
        
        results.append({
            'signal': result.get('signal', 'neutral'),
            'fusion_score': result.get('fusion_score', 0.0),
            'consistency': result.get('consistency', 0.0),
            'reason_codes': ','.join(result.get('reason_codes', []))
        })
        
        if (idx + 1) % 1000 == 0 and idx > 0:
            print(f"  [进度] {idx+1}/{len(df_clean)} ({100*(idx+1)/len(df_clean):.1f}%)")
    
    # 转为 DataFrame
    df_results = pd.DataFrame(results)
    
    # 计算质量指标
    print("  [计算] 信号质量指标...")
    quality_metrics = calculate_signal_quality(df_results, df_clean, column_map)
    
    # 合并指标
    metrics = {
        'file': file_name,
        'updates': len(results),
        'non_neutral_count': (df_results['signal'] != 'neutral').sum(),
        'non_neutral_rate': (df_results['signal'] != 'neutral').mean()
    }
    metrics.update(quality_metrics)
    
    return metrics


def append_to_summary(metrics, summary_file):
    """追加到汇总文件"""
    fieldnames = list(metrics.keys())
    file_exists = summary_file.exists()
    
    with open(summary_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)


def generate_offline_report_section(metrics_list, report_file):
    """生成离线评测报告章节"""
    if not metrics_list:
        return
    
    section = "\n## 离线数据评测\n\n"
    section += "### 概览\n\n"
    section += "| 文件 | 更新次数 | 非中性率 |\n"
    section += "--- | --- | --- |\n"
    
    for m in metrics_list:
        section += f"| {Path(m['file']).name} | {m['updates']} | {m.get('non_neutral_rate', 0):.2%} |\n"
    
    section += "\n### 信号质量指标\n\n"
    
    # 找出所有收益列
    ret_cols = set()
    for m in metrics_list:
        for k in m.keys():
            if k.startswith('hit_rate_') or k.startswith('ic_') or k.startswith('lift_'):
                ret_col = k.split('_', 2)[-1]
                ret_cols.add(ret_col)
    
    for ret_col in sorted(ret_cols):
        section += f"#### {ret_col}\n\n"
        section += "| 文件 | Hit率 | IC | Lift |\n"
        section += "--- | --- | --- | --- |\n"
        
        for m in metrics_list:
            file_name = Path(m['file']).name
            hit_key = f'hit_rate_{ret_col}'
            ic_key = f'ic_{ret_col}'
            lift_key = f'lift_{ret_col}'
            
            hit_val = m.get(hit_key, 0.0)
            ic_val = m.get(ic_key, 0.0)
            lift_val = m.get(lift_key, 0.0)
            
            section += f"| {file_name} | {hit_val:.3f} | {ic_val:.3f} | {lift_val:.2f} |\n"
        
        section += "\n"
    
    # 追加到报告文件
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(section)
    
    print(f"[OK] 报告章节已追加到: {report_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("OFI_CVD_Fusion 离线数据评测")
    print("=" * 60)
    
    # 加载融合模块
    try:
        fusion_mod = load_fusion_module()
        OFI_CVD_Fusion = fusion_mod.OFI_CVD_Fusion
        OFICVDFusionConfig = fusion_mod.OFICVDFusionConfig
    except Exception as e:
        print(f"[ERROR] 无法加载融合模块: {e}")
        sys.exit(1)
    
    # 搜索数据目录
    project_root = Path(__file__).parent.parent
    deploy_dir = project_root / "deploy" / "data"
    preview_dir = project_root / "deploy" / "preview" / "ofi_cvd"  # 直接定位到 ofi_cvd 目录
    
    # 合并搜索
    search_dirs = [deploy_dir, preview_dir]
    
    all_data_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            files = find_data_files(search_dir)
            all_data_files.extend(files)
            print(f"[发现] 在 {search_dir} 找到 {len(files)} 个文件")
    
    if not all_data_files:
        print("[SKIP] 未找到数据文件，跳过离线评测")
        return
    
    print(f"\n[总计] 找到 {len(all_data_files)} 个数据文件")
    
    # 输出路径
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_csv = results_dir / "fusion_metrics_summary.csv"
    report_file = results_dir / "fusion_test_report.md"
    
    # 评估每个文件（跳过 fusion 类型，只处理 OFI/CVD）
    all_metrics = []
    
    # 按交易对分组处理
    symbol_files = {}
    print(f"  [DEBUG] 找到 {len(all_data_files)} 个文件")
    
    for file_path in all_data_files:
        file_name = Path(file_path).name
        # 测试文件名格式: test_data_BTCUSDT_ofi.parquet
        if 'test_data_' in file_name:
            file_stem = file_name.replace('.parquet', '')
            parts = file_stem.split('_')
            # parts = ['test', 'data', 'BTCUSDT', 'ofi']
            if len(parts) == 4:
                symbol = parts[2]  # BTCUSDT
                kind = parts[3]    # ofi
                if symbol not in symbol_files:
                    symbol_files[symbol] = {}
                symbol_files[symbol][kind] = file_path
                print(f"    - {file_name} -> symbol={symbol}, kind={kind}")
    
    # 处理每个交易对
    for symbol, files in symbol_files.items():
        if 'ofi' not in files or 'cvd' not in files:
            print(f"\n[跳过] {symbol}: 缺少 OFI 或 CVD 数据")
            continue
        
        print(f"\n[处理] {symbol}")
        print("-" * 60)
        
        # 加载 OFI 和 CVD 数据
        ofi_df = load_data_file(files['ofi'])
        cvd_df = load_data_file(files['cvd'])
        
        if ofi_df is None or cvd_df is None:
            continue
        
        # 按时间戳合并
        if 'ts_ms' in ofi_df.columns and 'ts_ms' in cvd_df.columns:
            merged_df = pd.merge_asof(
                ofi_df[[col for col in ofi_df.columns if col.startswith(('ts_ms', 'ofi_'))]].sort_values('ts_ms'),
                cvd_df[[col for col in cvd_df.columns if col.startswith(('ts_ms', 'cvd_', 'z_'))]].sort_values('ts_ms'),
                on='ts_ms',
                direction='nearest',
                tolerance=100
            )
            
            # 检测列名
            column_map = detect_column_names(merged_df)
            print(f"  [列映射] {column_map}")
            
            # 运行评估
            fusion = OFI_CVD_Fusion(cfg=OFICVDFusionConfig())
            metrics = run_offline_evaluation(fusion, merged_df, column_map, f"{symbol}_merged", all_data_files)
            
            if metrics:
                all_metrics.append(metrics)
                append_to_summary(metrics, summary_csv)
                print(f"  [OK] 评估完成")
    
    # 不再逐个文件处理，只处理已合并的 symbol_files
    print(f"\n[完成] 成功处理 {len(all_metrics)} 个交易对")
    
    # 生成报告
    if all_metrics:
        print(f"\n[报告] 生成离线评测报告...")
        generate_offline_report_section(all_metrics, report_file)
        
        print(f"\n[SUCCESS] 离线评测完成！")
        print(f"  - 处理文件数: {len(all_metrics)}")
        print(f"  - 指标汇总: {summary_csv}")
    else:
        print("\n[WARN] 没有成功处理的文件")


if __name__ == "__main__":
    main()

