# tools/run_regime2x2_opt.py
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2X2'))

from regime2x2_pipeline import (
    compute_features, fit_2x2_thresholds, label_regime_2x2,
    grid_search_cvd_by_regime, export_best_params_yaml
)
from bt_adapter import run_bt

def build_forward_returns_from_prices(prices_df: pd.DataFrame, horizons=(60, 300), price_col="mid"):
    """
    基于价格构造未来收益：对每个 symbol，使用 merge_asof 时间对齐
    返回包含 ret_{h}s 列的 DataFrame（ts_ms, symbol 对齐）
    """
    if price_col not in prices_df.columns:
        # 若没有 mid，尝试用 (bid+ask)/2 或 price
        if {"bid","ask"}.issubset(prices_df.columns):
            prices_df = prices_df.copy()
            prices_df[price_col] = (prices_df["bid"] + prices_df["ask"]) / 2.0
        elif "price" in prices_df.columns:
            price_col = "price"
        else:
            raise ValueError("No price column: need mid or bid+ask or price.")
    base = prices_df[["ts_ms","symbol",price_col]].dropna().sort_values(["symbol","ts_ms"]).copy()
    out = base[["ts_ms","symbol"]].copy()
    for h in horizons:
        fut = base.copy()
        fut["ts_ms"] = fut["ts_ms"] + h*1000
        # forward 对齐未来时刻的价格
        aligned = pd.merge_asof(
            base.sort_values("ts_ms"),
            fut.sort_values("ts_ms"),
            on="ts_ms", by="symbol", direction="forward", suffixes=("","_fut")
        )
        out[f"ret_{h}s"] = (aligned[f"{price_col}_fut"] / aligned[price_col] - 1.0)
    return out

def load_prices(path_prices: str) -> pd.DataFrame:
    """加载价格数据"""
    if path_prices.endswith(".parquet"):
        return pd.read_parquet(path_prices)
    else:
        return pd.read_csv(path_prices)

def load_signals(path_signals: str) -> pd.DataFrame:
    """加载信号数据"""
    if path_signals.endswith(".parquet"):
        return pd.read_parquet(path_signals)
    else:
        return pd.read_csv(path_signals)

def find_real_data_files() -> tuple:
    """查找真实数据文件"""
    print("=== 查找真实数据文件 ===")
    
    # 查找价格数据
    price_patterns = [
        "**/data/**/prices_*.parquet",
        "**/data/**/*prices*.parquet",
        "**/data/**/*tick*.parquet"
    ]
    
    price_files = []
    for pattern in price_patterns:
        price_files.extend(glob.glob(pattern, recursive=True))
    
    # 查找信号数据
    signal_patterns = [
        "**/data/**/fusion_*.parquet",
        "**/data/**/cvd_*.parquet",
        "**/data/**/ofi_*.parquet"
    ]
    
    signal_files = []
    for pattern in signal_patterns:
        signal_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"找到价格文件: {len(price_files)}")
    for f in price_files[:5]:  # 显示前5个
        print(f"  - {f}")
    
    print(f"找到信号文件: {len(signal_files)}")
    for f in signal_files[:5]:  # 显示前5个
        print(f"  - {f}")
    
    return price_files, signal_files

def load_real_data_for_symbol(symbol: str) -> tuple:
    """为特定交易对加载真实数据"""
    print(f"\n=== 加载 {symbol} 的真实数据 ===")
    
    # 定义数据目录 - 优先使用48小时收集的数据
    data_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'runtime', '48h_collection', '48h_collection_20251022_0655'),
        os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'runtime', '20251022_T+1_batch_b1'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'ofi_cvd'),
        os.path.join(os.path.dirname(__file__), '..', 'data'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    ]
    
    prices_df = None
    signals_df = None
    
    # 使用新的数据结构加载数据
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            # 查找最新的日期目录
            date_dirs = []
            for item in os.listdir(data_dir):
                if item.startswith('date='):
                    date_dirs.append(os.path.join(data_dir, item))
            
            if not date_dirs:
                continue
                
            # 使用最新的日期目录
            latest_date_dir = sorted(date_dirs)[-1]
            symbol_dir = os.path.join(latest_date_dir, f'symbol={symbol}')
            
            if not os.path.exists(symbol_dir):
                continue
            
            print(f"  使用数据目录: {symbol_dir}")
            
            # 加载价格数据
            prices_dir = os.path.join(symbol_dir, 'kind=prices')
            if os.path.exists(prices_dir):
                price_files = glob.glob(os.path.join(prices_dir, '*.parquet'))
                if price_files:
                    try:
                        prices_list = []
                        for f in price_files:  # 加载所有文件
                            df = pd.read_parquet(f)
                            prices_list.append(df)
                        prices_df = pd.concat(prices_list, ignore_index=True)
                        prices_df = prices_df.sort_values('ts_ms')
                        print(f"  加载价格数据: {prices_df.shape}")
                        print(f"  价格数据列: {list(prices_df.columns)}")
                    except Exception as e:
                        print(f"  价格数据加载失败: {e}")
            
            # 加载信号数据
            signal_dirs = ['kind=fusion', 'kind=cvd', 'kind=ofi']
            for signal_dir_name in signal_dirs:
                signal_dir = os.path.join(symbol_dir, signal_dir_name)
                if os.path.exists(signal_dir):
                    signal_files = glob.glob(os.path.join(signal_dir, '*.parquet'))
                    if signal_files:
                        try:
                            signals_list = []
                            for f in signal_files:  # 加载所有文件
                                df = pd.read_parquet(f)
                                signals_list.append(df)
                            signals_df = pd.concat(signals_list, ignore_index=True)
                            signals_df = signals_df.sort_values('ts_ms')
                            print(f"  加载信号数据 ({signal_dir_name}): {signals_df.shape}")
                            print(f"  信号数据列: {list(signals_df.columns)}")
                            break  # 找到第一个可用的信号数据就停止
                        except Exception as e:
                            print(f"  信号数据加载失败 ({signal_dir_name}): {e}")
                            continue
            
            if prices_df is not None and signals_df is not None:
                break
    
    # 硬失败：不允许用价格凑信号
    if signals_df is None:
        raise RuntimeError(f"[{symbol}] 未找到真实信号文件（包含 score/z_ofi/z_cvd 的表）。请提供 --signals 或放置到 data/。")
    
    # 确保价格数据有必要的列
    if prices_df is not None:
        if 'event_ts_ms' in prices_df.columns:
            prices_df['ts_ms'] = prices_df['event_ts_ms']
        elif 'timestamp' in prices_df.columns:
            prices_df['ts_ms'] = prices_df['timestamp']
        
        if 'symbol' not in prices_df.columns:
            prices_df['symbol'] = symbol
    
    return prices_df, signals_df

def main():
    """主函数"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", help="path to prices ticks with bid/ask or mid")
    ap.add_argument("--signals", help="path to signals with forward returns")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--cost_bps", type=float, default=15.0)
    ap.add_argument("--horizons", default="15,60,300")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT")
    ap.add_argument("--use_real_data", action="store_true", default=True, help="使用真实数据")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"grid_search/{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== 2×2场景化参数优化 ===")
    print(f"输出目录: {args.outdir}")
    print(f"运行目录: {run_dir}")
    print(f"使用真实数据: {args.use_real_data}")

    if args.use_real_data:
        # 使用真实数据
        symbols = [s.strip() for s in args.symbols.split(",")]
        all_results = {}
        
        # 进入 for symbol 之前定义horizons_list，全流程复用
        horizons_list = [int(x) for x in args.horizons.split(",") if x]
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"处理交易对: {symbol}")
            print(f"{'='*50}")
            
            # 加载该交易对的真实数据
            prices_df, signals_df = load_real_data_for_symbol(symbol)
            
            if prices_df is None or signals_df is None:
                print(f"  {symbol}: 数据加载失败，跳过")
                continue
            
            try:
                # A1: 特征计算
                print("A1: 计算2×2特征...")
                feats = compute_features(prices_df)
                print(f"  特征数据形状: {feats.shape}")

                # A2: 阈值拟合 + 打标
                print("A2: 阈值拟合 + 打标...")
                thr = fit_2x2_thresholds(feats)
                labeled = label_regime_2x2(feats, thr)
                print(f"  标签数据形状: {labeled.shape}")
                print(f"  场景分布: {labeled['regime_2x2'].value_counts().to_dict()}")

                # A3: 合并到信号&收益（基于价格构造未来收益）
                print("A3: 合并信号与基于价格的未来收益...")
                if 'ts_ms' not in signals_df.columns and 'event_time_ms' in signals_df.columns:
                    signals_df = signals_df.copy()
                    signals_df['ts_ms'] = signals_df['event_time_ms']

                # 基于价格生成 forward returns（建议与 horizons 保持一致）
                fwd = build_forward_returns_from_prices(prices_df, horizons=horizons_list, price_col="mid")

                merged = (signals_df
                          .merge(labeled[["ts_ms","symbol","regime_2x2"]], on=["ts_ms","symbol"], how="left")
                          .merge(fwd, on=["ts_ms","symbol"], how="left"))

                merged['regime_2x2'] = merged['regime_2x2'].fillna("TL")
                merged = merged.sort_values(['symbol','ts_ms'])
                print(f"  合并后数据形状: {merged.shape}")

                # （可选）最低样本门槛：避免小样本场景误判
                MIN_SAMPLES = 10  # 进一步降低样本数要求
                if merged.shape[0] < MIN_SAMPLES:
                    print(f"  样本不足（{merged.shape[0]} < {MIN_SAMPLES}），跳过 {symbol}")
                    continue

                # A4: 网格搜索
                print("A4: 网格搜索...")
                param_grid = {
                    "ewm_span": [5, 10, 20, 40],
                    "z_window": [60, 120, 300],
                    "robust_z": [True, False],
                    "winsorize_pct": [95, 98, 99],
                    "w_cvd": [0.3, 0.4, 0.5],
                }
                global_anchor = {"ewm_span": 20, "z_window": 120, "robust_z": True, "winsorize_pct": 98, "w_cvd": 0.4}
                
                best_params, results_df = grid_search_cvd_by_regime(
                    merged_df=merged,
                    param_grid=param_grid,
                    horizons=horizons_list,
                    cost_bps=args.cost_bps,
                    global_anchor=global_anchor,
                    keys_for_reg=["ewm_span","z_window","w_cvd"],
                    lambda_reg=0.2,
                    backtest_fn=run_bt,
                    min_samples=10  # 降低最小样本数要求
                )
                
                print(f"  网格搜索结果形状: {results_df.shape}")
                print(f"  最优参数: {best_params}")

                # 保存结果
                symbol_results_path = os.path.join(run_dir, f"results_{symbol.lower()}.csv")
                results_df.to_csv(symbol_results_path, index=False)
                
                all_results[symbol] = {
                    'best_params': best_params,
                    'results_df': results_df,
                    'thresholds': thr,
                    'labeled': labeled
                }
                
                print(f"  [OK] {symbol} 结果保存到: {symbol_results_path}")
                
            except Exception as e:
                print(f"  {symbol} 处理失败: {e}")
                continue
        
        # A5: 导出所有交易对的最优参数
        if all_results:
            print(f"\nA5: 导出最优参数...")
            combined_best_params = {}
            for symbol, result in all_results.items():
                combined_best_params[symbol] = result['best_params']
            
            best_yaml = os.path.join(args.outdir, "best_params.yaml")
            export_best_params_yaml(combined_best_params, global_anchor, best_yaml)
            print(f"[OK] 最优参数: {best_yaml}")
            
            # A6: 保存阈值（便于复现）
            import yaml
            thresholds_yaml = os.path.join(run_dir, "thresholds.yaml")
            with open(thresholds_yaml, "w") as f:
                yaml.safe_dump({symbol: result['thresholds'] for symbol, result in all_results.items()}, 
                              f, allow_unicode=True, sort_keys=False)
            print(f"[OK] 阈值配置: {thresholds_yaml}")
            
            # 保存标签预览
            for symbol, result in all_results.items():
                preview_path = os.path.join(run_dir, f"labeled_preview_{symbol.lower()}.parquet")
                result['labeled'].head(1000).to_parquet(preview_path)
            print(f"[OK] 标签预览已保存")
            
        else:
            print("错误: 没有成功处理任何交易对")
            return 1
    
    else:
        # 使用指定文件
        if not args.prices or not args.signals:
            print("错误: 必须指定 --prices 和 --signals 参数")
            return 1
        
        prices = load_prices(args.prices)
        signals = load_signals(args.signals)

        # A1: 特征
        feats = compute_features(prices)

        # A2: 阈值拟合 + 打标
        thr = fit_2x2_thresholds(feats)
        labeled = label_regime_2x2(feats, thr)

        # A3: 合并到信号&收益
        merged = signals.merge(labeled[["ts_ms","symbol","regime_2x2"]], on=["ts_ms","symbol"], how="left").fillna({"regime_2x2":"TL"})

        # A4: 网格
        horizons = [int(x) for x in args.horizons.split(",") if x]
        param_grid = {
            "ewm_span": [5, 10, 20, 40],
            "z_window": [60, 120, 300],
            "robust_z": [True, False],
            "winsorize_pct": [95, 98, 99],
            "w_cvd": [0.3, 0.4, 0.5],
        }
        global_anchor = {"ewm_span": 20, "z_window": 120, "robust_z": True, "winsorize_pct": 98, "w_cvd": 0.4}
        best_params, results_df = grid_search_cvd_by_regime(
            merged_df=merged,
            param_grid=param_grid,
            horizons=horizons,
            cost_bps=args.cost_bps,
            global_anchor=global_anchor,
            keys_for_reg=["ewm_span","z_window","w_cvd"],
            lambda_reg=0.2,
            backtest_fn=run_bt
        )

        results_path = os.path.join(run_dir, "results_grid.csv")
        results_df.to_csv(results_path, index=False)

        # A5: 导出最优参数
        best_yaml = os.path.join(args.outdir, "best_params.yaml")
        export_best_params_yaml(best_params, global_anchor, best_yaml)

        # A6: 保存阈值（便于复现）
        import yaml
        with open(os.path.join(run_dir, "thresholds.yaml"), "w") as f:
            yaml.safe_dump(thr, f, allow_unicode=True, sort_keys=False)
        labeled.head(1000).to_parquet(os.path.join(run_dir, "labeled_preview.parquet"))

        print(f"[OK] results: {results_path}")
        print(f"[OK] best params: {best_yaml}")
        print(f"[OK] thresholds: {os.path.join(run_dir,'thresholds.yaml')}")

    print(f"\n=== 分析完成 ===")
    return 0

if __name__ == "__main__":
    exit(main())