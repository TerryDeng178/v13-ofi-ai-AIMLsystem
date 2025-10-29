@echo off
REM harvestd - 诊断脚本
REM 诊断守护进程无法访问的问题

setlocal

set HARVESTD_PORT=8088

echo ========================================
echo harvestd Diagnostic Tool
echo ========================================
echo.

echo [1/6] 检查Python环境...
python --version 2>nul
if errorlevel 1 (
    echo [FAIL] Python未安装或不在PATH中
    echo 请安装Python 3.11+
    goto :end
) else (
    echo [OK] Python已安装
    python --version
)
echo.

echo [2/6] 检查必要文件...
if not exist "tools\harvestd.py" (
    echo [FAIL] 找不到 tools\harvestd.py
    echo 当前目录: %CD%
    echo 请确保在 v13_ofi_ai_system 目录下运行此脚本
    goto :end
) else (
    echo [OK] tools\harvestd.py 存在
)
echo.

echo [3/6] 检查端口占用...
netstat -ano | findstr ":%HARVESTD_PORT%" >nul 2>&1
if errorlevel 1 (
    echo [OK] 端口 %HARVESTD_PORT% 未被占用
) else (
    echo [WARN] 端口 %HARVESTD_PORT% 已被占用
    echo 占用此端口的进程:
    netstat -ano | findstr ":%HARVESTD_PORT%"
    echo.
    echo 解决方案:
    echo   1. 停止占用端口的进程
    echo   2. 或修改 HARVESTD_PORT 环境变量
)
echo.

echo [4/6] 检查网络连接...
ping -n 1 127.0.0.1 >nul 2>&1
if errorlevel 1 (
    echo [FAIL] 无法连接到本地回环地址
    goto :end
) else (
    echo [OK] 网络连接正常
)
echo.

echo [5/6] 尝试启动守护进程（测试模式）...
echo 注意: 这将启动守护进程，请在新窗口查看输出
echo.
echo 在新窗口运行以下命令:
echo   cd %CD%
echo   python tools\harvestd.py
echo.
echo 然后在新窗口按 Ctrl+C 停止
echo.

echo [6/6] 检查防火墙设置...
netsh advfirewall show allprofiles | findstr "State" | findstr "ON" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Windows防火墙未启用或未检测到
) else (
    echo [WARN] Windows防火墙已启用
    echo 如果无法访问，可能需要添加端口例外
    echo 端口: %HARVESTD_PORT%
)
echo.

echo ========================================
echo 诊断完成
echo ========================================
echo.
echo 快速测试步骤:
echo   1. 在新窗口运行: python tools\harvestd.py
echo   2. 等待显示 "[ui] HTTP listening on :8088"
echo   3. 浏览器访问: http://localhost:8088/
echo   4. 或命令行测试: curl http://localhost:8088/health
echo.
echo 常见问题:
echo   - 如果提示 "Address already in use": 端口被占用
echo   - 如果无任何输出: 检查Python环境
echo   - 如果无法连接: 检查防火墙设置
echo.

:end
pause
