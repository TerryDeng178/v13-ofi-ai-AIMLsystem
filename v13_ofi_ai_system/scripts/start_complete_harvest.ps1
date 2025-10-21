# 完整版OFI+CVD数据采集启动脚本
Write-Host "启动完整版OFI+CVD数据采集..." -ForegroundColor Green
Write-Host ""

# 设置环境变量
$env:SYMBOLS = "BTCUSDT,ETHUSDT"
$env:RUN_HOURS = "2"
$env:OUTPUT_DIR = "data/ofi_cvd"

Write-Host "配置参数:" -ForegroundColor Yellow
Write-Host "- 交易对: $env:SYMBOLS"
Write-Host "- 运行时间: $env:RUN_HOURS 小时"
Write-Host "- 输出目录: $env:OUTPUT_DIR"
Write-Host ""

# 检查Python环境
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "错误: 未找到Python环境" -ForegroundColor Red
    exit 1
}

# 检查依赖包
Write-Host "检查依赖包..." -ForegroundColor Yellow
$packages = @("pandas", "numpy", "pyarrow", "websockets")
foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        Write-Host "✓ $package" -ForegroundColor Green
    } catch {
        Write-Host "✗ $package (需要安装)" -ForegroundColor Red
        Write-Host "正在安装 $package..." -ForegroundColor Yellow
        pip install $package
    }
}

Write-Host ""
Write-Host "启动完整版数据采集脚本..." -ForegroundColor Green

# 启动完整版采集脚本
try {
    python examples/run_complete_harvest.py
    Write-Host ""
    Write-Host "完整版数据采集完成！" -ForegroundColor Green
} catch {
    Write-Host "错误: 数据采集失败" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

