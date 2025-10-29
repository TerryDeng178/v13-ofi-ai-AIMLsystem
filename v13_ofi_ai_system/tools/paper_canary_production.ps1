# 生产环境纸上交易金丝雀测试脚本（60分钟完整版）
# 用于合并后主线验证

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "============================================================"
Write-Host "生产环境纸上交易金丝雀测试（60分钟完整版）"
Write-Host "============================================================"
Write-Host ""
Write-Host "目标："
Write-Host "  - error_rate=0"
Write-Host "  - latency.p99 < 500ms"
Write-Host "  - 指纹无漂移"
Write-Host "  - 4类信号在活跃时段触发率 > 0"
Write-Host ""
Write-Host "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

python tools/paper_canary.py --mins 60 --p99-limit-ms 500

Write-Host ""
Write-Host "完成时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

# 检查结果
$reportPath = "reports\paper_canary_report.json"
if (Test-Path $reportPath) {
    $result = python -c "import json; f=open('reports/paper_canary_report.json','r',encoding='utf-8'); data=json.load(f); f.close(); print('[SUCCESS] 生产金丝雀测试通过' if data.get('overall_pass') else '[FAIL] 生产金丝雀测试失败'); exit(0 if data.get('overall_pass') else 1)"
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "[FAIL] 生产金丝雀测试失败"
        exit 1
    }
} else {
    Write-Host "[ERROR] 报告文件不存在"
    exit 1
}
