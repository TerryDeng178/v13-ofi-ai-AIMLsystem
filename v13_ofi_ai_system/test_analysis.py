#!/usr/bin/env python3
"""
测试OFI+CVD信号分析工具
"""

import sys
sys.path.append('.')

from analysis.ofi_cvd_signal_eval import OFICVDSignalEvaluator

def test_analysis():
    print("Starting OFI+CVD signal analysis test...")
    
    # 创建评估器
    evaluator = OFICVDSignalEvaluator(
        data_root='data/ofi_cvd',
        symbols=['ETHUSDT'],
        date_from='2025-10-21',
        date_to='2025-10-21',
        horizons=[60, 180, 300],
        fusion_weights={'w_ofi': 0.6, 'w_cvd': 0.4},
        slices={'regime': ['Active', 'Quiet']},
        output_dir='artifacts/analysis/ofi_cvd',
        run_tag='eth_analysis_test'
    )
    
    print("Evaluator created successfully")
    print("Starting data loading...")
    
    # 加载数据
    data = evaluator.load_data()
    print("Data loaded successfully")
    
    # 检查数据
    for symbol in data:
        print(f"{symbol} data types:")
        for data_type, df in data[symbol].items():
            print(f"  {data_type}: {len(df)} rows")
    
    print("Starting schema validation...")
    schema_checks = evaluator.validate_schema()
    
    for symbol in schema_checks:
        print(f"{symbol} schema checks:")
        for data_type, is_valid in schema_checks[symbol].items():
            status = "OK" if is_valid else "FAIL"
            print(f"  {data_type}: {status}")
    
    print("Starting label construction...")
    labeled_data = evaluator.construct_labels()
    
    for symbol in labeled_data:
        print(f"{symbol} labeled data: {len(labeled_data[symbol])} rows")
        for horizon in [60, 180, 300]:
            label_col = f'label_{horizon}s'
            if label_col in labeled_data[symbol].columns:
                valid_labels = labeled_data[symbol][label_col].notna().sum()
                print(f"  {horizon}s labels: {valid_labels} valid")
    
    print("Starting signal extraction...")
    signals = evaluator.extract_signals()
    
    for symbol in signals:
        print(f"{symbol} signal data:")
        for signal_type, df in signals[symbol].items():
            print(f"  {signal_type}: {len(df)} rows")
    
    print("Starting metrics calculation...")
    metrics = evaluator.calculate_metrics(labeled_data, signals)
    
    for symbol in metrics:
        print(f"{symbol} metrics:")
        for signal_type, windows in metrics[symbol].items():
            print(f"  {signal_type}: {len(windows)} windows")
    
    print("Generating reports...")
    results = evaluator.generate_reports()
    
    print("Analysis completed!")
    print(f"Results saved to: artifacts/analysis/ofi_cvd")
    print(f"Run tag: {results.get('run_tag', 'N/A')}")
    
    return results

if __name__ == "__main__":
    test_analysis()
