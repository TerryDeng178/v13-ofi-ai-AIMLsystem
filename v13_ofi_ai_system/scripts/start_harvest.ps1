# OFI+CVD 数据采集启动脚本 (Task 1.3.1 v2)
# PowerShell 5+/pwsh 兼容；一键：预检 -> 采集 -> 验证 -> 输出日志

param(
  [string]$Symbols = $(if ($env:SYMBOLS) { $env:SYMBOLS } else { "BTCUSDT,ETHUSDT" }),
  [int]   $RunHours = $(if ($env:RUN_HOURS) { [int]$env:RUN_HOURS } else { 72 }),
  [int]   $ParquetRotateSec = $(if ($env:PARQUET_ROTATE_SEC) { [int]$env:PARQUET_ROTATE_SEC } else { 60 }),
  [int]   $WssPingInterval = $(if ($env:WSS_PING_INTERVAL) { [int]$env:WSS_PING_INTERVAL } else { 20 }),
  [int]   $DedupLRU = $(if ($env:DEDUP_LRU) { [int]$env:DEDUP_LRU } else { 8192 }),
  [string]$ZMode = $(if ($env:Z_MODE) { $env:Z_MODE } else { "delta" }),
  [string]$ScaleMode = $(if ($env:SCALE_MODE) { $env:SCALE_MODE } else { "hybrid" }),
  [double]$MadMultiplier = $(if ($env:MAD_MULTIPLIER) { [double]$env:MAD_MULTIPLIER } else { 1.8 }),
  [double]$ScaleFastWeight = $(if ($env:SCALE_FAST_WEIGHT) { [double]$env:SCALE_FAST_WEIGHT } else { 0.20 }),
  [int]   $HalfLifeSec = $(if ($env:HALF_LIFE_SEC) { [int]$env:HALF_LIFE_SEC } else { 600 }),
  [double]$WinsorLimit = $(if ($env:WINSOR_LIMIT) { [double]$env:WINSOR_LIMIT } else { 8.0 }),
  [string]$OutputDir = $(if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "data\ofi_cvd" }),
  [string]$ArtifactsDir = $(if ($env:ARTIFACTS_DIR) { $env:ARTIFACTS_DIR } else { "artifacts" }),
  [switch]$SkipPrecheck
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Green
Write-Host "OFI+CVD 数据采集系统 (Task 1.3.1 v2)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 1) 解析工程根目录与脚本路径
$RepoRoot = Split-Path -Parent $PSCommandPath
Set-Location $RepoRoot
$PyHarvest  = Join-Path $RepoRoot "examples\run_realtime_harvest.py"
$PyValidate = Join-Path $RepoRoot "scripts\validate_ofi_cvd_harvest.py"

if (!(Test-Path $PyHarvest))  { throw "缺少 $PyHarvest" }
if (!(Test-Path $PyValidate)) { throw "缺少 $PyValidate" }

# 2) 打印配置指纹
$cfg = "SYMBOLS=$Symbols | RUN_HOURS=$RunHours | ROTATE=${ParquetRotateSec}s | Z=$ZMode | SCALE=$ScaleMode | MAD=$MadMultiplier | FAST=$ScaleFastWeight | HL=$HalfLifeSec | WINSOR=$WinsorLimit"
Write-Host "[CONFIG] $cfg" -ForegroundColor Cyan

# 3) Python 与依赖自检
try {
  $py = (Get-Command python -ErrorAction Stop).Source
} catch {
  throw "未找到 python，可在 PowerShell 输入 'py -3 -m pip install ...' 后再试"
}
$ver = & $py -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "[Python] $py ($ver)" -ForegroundColor DarkGray

try {
  & $py -c "import pandas, pyarrow, prometheus_client, websockets; print('deps ok')" | Out-Null
  Write-Host "依赖包检查通过" -ForegroundColor Green
} catch {
  Write-Host "错误：缺少依赖，将尝试安装 pandas pyarrow prometheus_client websockets" -ForegroundColor Yellow
  & $py -m pip install -q pandas pyarrow prometheus_client websockets
}

# 4) 目录就绪
$newDirs = @($OutputDir, (Join-Path $ArtifactsDir "run_logs"), (Join-Path $ArtifactsDir "dq_reports"))
$newDirs | % { if (!(Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $ArtifactsDir ("run_logs\harvest_"+$stamp+".log")

# 5) 注入环境变量（供 Python 读取）
$env:SYMBOLS = $Symbols
$env:RUN_HOURS = "$RunHours"
$env:PARQUET_ROTATE_SEC = "$ParquetRotateSec"
$env:WSS_PING_INTERVAL = "$WssPingInterval"
$env:DEDUP_LRU = "$DedupLRU"
$env:Z_MODE = $ZMode
$env:SCALE_MODE = $ScaleMode
$env:MAD_MULTIPLIER = "$MadMultiplier"
$env:SCALE_FAST_WEIGHT = "$ScaleFastWeight"
$env:HALF_LIFE_SEC = "$HalfLifeSec"
$env:WINSOR_LIMIT = "$WinsorLimit"
$env:OUTPUT_DIR = $OutputDir
$env:ARTIFACTS_DIR = $ArtifactsDir

# 6) 预检（10 分钟），脚本未支持 --precheck-only 时自动跳过
if (-not $SkipPrecheck) {
  Write-Host "`n开始 10 分钟预检..." -ForegroundColor Yellow
  try {
    # 优先尝试脚本自带预检参数
    $pre = Start-Process -FilePath $py -ArgumentList @($PyHarvest, "--precheck-only") -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath
    Wait-Process $pre -Timeout 900
  } catch {
    Write-Host "未检测到 --precheck-only，执行简易预检（直连 10 分钟后退出）" -ForegroundColor DarkYellow
    $pre = Start-Process -FilePath $py -ArgumentList @($PyHarvest, "--run-secs","600","--exit-after") -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath
    Wait-Process $pre -Timeout 900
  }
  Write-Host "预检完成，日志：$logPath" -ForegroundColor Green
}

# 7) 正式采集（后台输出到日志）
Write-Host "`n开始正式采集（$RunHours 小时）..." -ForegroundColor Green
$args = @($PyHarvest)
$proc = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $logPath -RedirectStandardError $logPath

Wait-Process $proc

# 8) 质量验证（输出 JSON 到 artifacts/dq_reports）
Write-Host "`n运行数据质量验证..." -ForegroundColor Yellow
& $py $PyValidate --base-dir $OutputDir --output-dir (Join-Path $ArtifactsDir "dq_reports") 2>&1 | Tee-Object -FilePath $logPath -Append | Out-Null

Write-Host "`n任务完成！日志：$logPath" -ForegroundColor Green
