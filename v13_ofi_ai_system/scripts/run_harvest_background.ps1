# OFI+CVD 数据采集后台运行脚本

param(
    [int]$RunHours = 2
)

Write-Host "启动OFI+CVD数据采集系统..." -ForegroundColor Green

# 设置工作目录
$RepoRoot = Split-Path -Parent $PSCommandPath
Set-Location $RepoRoot

# 检查Python
try {
    $py = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Python: $py" -ForegroundColor Green
} catch {
    Write-Host "错误：未找到 python" -ForegroundColor Red
    exit 1
}

# 检查必要文件
$PyHarvest = "examples\run_realtime_harvest.py"
if (!(Test-Path $PyHarvest)) {
    Write-Host "错误：缺少 $PyHarvest" -ForegroundColor Red
    exit 1
}

# 创建必要目录
$dirs = @("data\ofi_cvd", "artifacts\run_logs", "artifacts\dq_reports")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Yellow
    }
}

# 设置环境变量
$env:SYMBOLS = "BTCUSDT,ETHUSDT"
$env:RUN_HOURS = "$RunHours"
$env:PARQUET_ROTATE_SEC = "60"
$env:WSS_PING_INTERVAL = "20"
$env:DEDUP_LRU = "8192"
$env:Z_MODE = "delta"
$env:SCALE_MODE = "hybrid"
$env:MAD_MULTIPLIER = "1.8"
$env:SCALE_FAST_WEIGHT = "0.20"
$env:HALF_LIFE_SEC = "600"
$env:WINSOR_LIMIT = "8.0"
$env:OUTPUT_DIR = "data\ofi_cvd"
$env:ARTIFACTS_DIR = "artifacts"

# 生成日志文件名
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = "artifacts\run_logs\harvest_$stamp.log"

Write-Host "配置: 运行 $RunHours 小时" -ForegroundColor Cyan
Write-Host "日志: $logPath" -ForegroundColor Cyan

# 启动数据采集进程
Write-Host "启动数据采集进程..." -ForegroundColor Green
$proc = Start-Process -FilePath $py -ArgumentList $PyHarvest -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath

Write-Host "数据采集进程已启动" -ForegroundColor Green
Write-Host "进程ID: $($proc.Id)" -ForegroundColor Green
Write-Host "日志文件: $logPath" -ForegroundColor Cyan
Write-Host "进程将在后台运行 $RunHours 小时" -ForegroundColor Yellow

# 等待一段时间让进程启动
Start-Sleep -Seconds 5

# 检查进程是否还在运行
if (!$proc.HasExited) {
    Write-Host "数据采集进程正在运行中..." -ForegroundColor Green
} else {
    Write-Host "数据采集进程已退出，请检查日志文件" -ForegroundColor Red
}

Write-Host "脚本执行完成，进程继续在后台运行" -ForegroundColor Green

