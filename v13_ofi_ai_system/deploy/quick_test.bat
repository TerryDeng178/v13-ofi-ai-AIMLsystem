@echo off
REM harvestd - 快速测试脚本
REM 快速检查守护进程是否可访问

setlocal

set TEST_PORT=8088

echo ========================================
echo harvestd Quick Test
echo ========================================
echo.

echo [1] 检查端口状态...
netstat -ano | findstr ":%TEST_PORT%" | findstr "LISTENING"
if errorlevel 1 (
    echo [FAIL] 端口 %TEST_PORT% 未在监听
    echo.
    echo 可能的原因:
    echo   - 守护进程未启动
    echo   - 守护进程启动失败
    echo   - 端口被其他程序占用
    echo.
    echo 解决方案:
    echo   1. 运行 start_harvestd.bat
    echo   2. 查看是否有错误输出
    echo   3. 运行 diagnose.bat 进行诊断
) else (
    echo [OK] 端口 %TEST_PORT% 正在监听
    echo.
    echo [2] 测试HTTP连接...
    echo.
    where curl >nul 2>&1
    if errorlevel 1 (
        echo 未安装curl，使用PowerShell测试...
        powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:%TEST_PORT%/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '[OK] HTTP连接成功'; Write-Host $response.Content } catch { Write-Host '[FAIL] HTTP连接失败'; Write-Host $_.Exception.Message }"
    ) else (
        echo 使用curl测试...
        curl -s -m 5 http://localhost:%TEST_PORT%/health
        if errorlevel 1 (
            echo [FAIL] HTTP连接失败
            echo 可能的原因:
            echo   - 守护进程未正确启动
            echo   - 防火墙阻止连接
            echo   - 服务内部错误
        ) else (
            echo [OK] HTTP连接成功
            echo.
            echo ========================================
            echo 测试成功！
            echo ========================================
            echo.
            echo 可访问的URL:
            echo   - 主页: http://localhost:%TEST_PORT%/
            echo   - 健康检查: http://localhost:%TEST_PORT%/health
            echo   - 日志: http://localhost:%TEST_PORT%/logs
            echo   - DQ报告: http://localhost:%TEST_PORT%/dq
            echo   - 指标: http://localhost:%TEST_PORT%/metrics
            echo.
        )
    )
)
echo.

pause
