# OFI+CVD 数据采集启动脚本 (简化版)
# PowerShell 5+/pwsh 兼容

param(
    [int]$RunHours = 2,
    [switch]$SkipPrecheck
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Green
Write-Host "OFI+CVD 数据采集系统 (Task 1.3.1 v2)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 设置工作目录
$RepoRoot = Split-Path -Parent $PSCommandPath
Set-Location $RepoRoot

# 检查Python
try {
    $py = (Get-Command python -ErrorAction Stop).Source
    Write-Host "[Python] 找到: $py" -ForegroundColor Green
} catch {
    Write-Host "错误：未找到 python" -ForegroundColor Red
    exit 1
}

# 检查必要文件
$PyHarvest = "examples\run_realtime_harvest.py"
$PyValidate = "scripts\validate_ofi_cvd_harvest.py"

if (!(Test-Path $PyHarvest)) {
    Write-Host "错误：缺少 $PyHarvest" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $PyValidate)) {
    Write-Host "错误：缺少 $PyValidate" -ForegroundColor Red
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

Write-Host "配置: 运行 $RunHours 小时，日志: $logPath" -ForegroundColor Cyan

# 预检（如果未跳过）
if (-not $SkipPrecheck) {
    Write-Host "`n开始 10 分钟预检..." -ForegroundColor Yellow
    try {
        $preArgs = @($PyHarvest, "--precheck-only")
        $pre = Start-Process -FilePath $py -ArgumentList $preArgs -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath
        Wait-Process $pre -Timeout 900
        Write-Host "预检完成" -ForegroundColor Green
    } catch {
        Write-Host "预检失败，继续执行..." -ForegroundColor Yellow
    }
}

# 正式采集
Write-Host "`n开始正式采集（$RunHours 小时）..." -ForegroundColor Green
$args = @($PyHarvest)
$proc = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath

Write-Host "数据采集进程已启动，PID: $($proc.Id)" -ForegroundColor Green
Write-Host "日志文件: $logPath" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止监控，进程将继续在后台运行" -ForegroundColor Yellow

# 监控进程
try {
    while (!$proc.HasExited) {
        Start-Sleep -Seconds 10
        if (Test-Path $logPath) {
            $logSize = (Get-Item $logPath).Length
            Write-Host "进程运行中... 日志大小: $logSize 字节" -ForegroundColor DarkGray
        }
    }
    Write-Host "数据采集完成！" -ForegroundColor Green
} catch {
    Write-Host "监控中断，但进程可能仍在后台运行" -ForegroundColor Yellow
}

# 质量验证
Write-Host "`n运行数据质量验证..." -ForegroundColor Yellow
try {
    $validateArgs = @($PyValidate, "--base-dir", "data\ofi_cvd", "--output-dir", "artifacts\dq_reports")
    & $py $validateArgs 2>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
    Write-Host "质量验证完成" -ForegroundColor Green
} catch {
    Write-Host "质量验证失败" -ForegroundColor Red
}

Write-Host "`n任务完成！日志：$logPath" -ForegroundColor Green
