# BTCUSDT 40分钟金测
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BTCUSDT 40分钟金测" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format 'yyyyMMdd_HHmm'
$output_dir = "data\cvd_gold_btc_$timestamp"

Write-Host "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host "输出目录: $output_dir" -ForegroundColor Yellow
Write-Host "预计完成: $(Get-Date).AddMinutes(40)" -ForegroundColor Yellow
Write-Host ""

cd examples
python run_realtime_cvd.py --symbol BTCUSDT --duration 2400 --output-dir "..\$output_dir"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "测试完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "结束时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    
    # 自动分析
    Write-Host ""
    Write-Host "开始分析结果..." -ForegroundColor Cyan
    
    $parquet_file = Get-ChildItem -Path "..\$output_dir" -Filter "*.parquet" | Select-Object -First 1
    if ($parquet_file) {
        $report_dir = "..\docs\reports\cvd_gold_btc_$timestamp"
        python analysis_cvd.py --data "..\$output_dir\$($parquet_file.Name)" --out $report_dir --report "$report_dir\REPORT.md"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "报告生成完成: $report_dir\REPORT.md" -ForegroundColor Green
        }
    }
} else {
    Write-Host ""
    Write-Host "测试异常退出！" -ForegroundColor Red
}

Write-Host ""
Write-Host "按任意键关闭..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

