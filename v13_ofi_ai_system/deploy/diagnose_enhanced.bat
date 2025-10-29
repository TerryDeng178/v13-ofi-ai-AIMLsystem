@echo off
REM harvestd - 增强版诊断脚本
REM 提供全面的系统状态检查和故障诊断

setlocal enabledelayedexpansion

echo ========================================
echo harvestd - 系统诊断工具 (增强版)
echo ========================================
echo.

REM 设置脚本所在目录为deploy目录
set DEPLOY_DIR=%~dp0
set SCRIPT_DIR=%DEPLOY_DIR%..
cd /d "%SCRIPT_DIR%"

echo [INFO] 当前目录: %CD%
echo [INFO] 诊断时间: %DATE% %TIME%
echo.

REM 1. 检查Python环境
echo ========================================
echo 1. Python环境检查
echo ========================================
python --version
if errorlevel 1 (
    echo [ERROR] Python未安装或不在PATH中
    goto :end
) else (
    echo [OK] Python环境正常
)
echo.

REM 2. 检查必要文件
echo ========================================
echo 2. 必要文件检查
echo ========================================
if exist "tools\harvestd.py" (
    echo [OK] tools\harvestd.py
) else (
    echo [ERROR] 缺少 tools\harvestd.py
)

if exist "deploy\run_success_harvest.py" (
    echo [OK] deploy\run_success_harvest.py
) else (
    echo [ERROR] 缺少 deploy\run_success_harvest.py
)

if exist "scripts\validate_ofi_cvd_harvest.py" (
    echo [OK] scripts\validate_ofi_cvd_harvest.py
) else (
    echo [ERROR] 缺少 scripts\validate_ofi_cvd_harvest.py
)
echo.

REM 3. 检查进程状态
echo ========================================
echo 3. 进程状态检查
echo ========================================
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
echo.

REM 4. 检查端口使用
echo ========================================
echo 4. 端口使用检查
echo ========================================
netstat -an | findstr ":8088"
echo.

REM 5. 检查数据文件
echo ========================================
echo 5. 数据文件检查
echo ========================================
if exist "deploy\data\ofi_cvd" (
    echo [OK] 数据目录存在
    dir "deploy\data\ofi_cvd" /B
) else (
    echo [WARN] 数据目录不存在
)
echo.

REM 6. 检查日志文件
echo ========================================
echo 6. 日志文件检查
echo ========================================
if exist "deploy\artifacts\run_logs" (
    echo [OK] 日志目录存在
    echo 最新日志文件:
    dir "deploy\artifacts\run_logs" /O-D /B | findstr /R "harvester_.*\.log" | head -3
) else (
    echo [WARN] 日志目录不存在
)
echo.

REM 7. 检查网络连接
echo ========================================
echo 7. 网络连接检查
echo ========================================
ping -n 1 fstream.binance.com >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 无法连接到 fstream.binance.com
) else (
    echo [OK] 网络连接正常
)
echo.

REM 8. 检查磁盘空间
echo ========================================
echo 8. 磁盘空间检查
echo ========================================
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set FREE_SPACE=%%a
echo 可用磁盘空间: %FREE_SPACE% bytes
echo.

REM 9. 检查环境变量
echo ========================================
echo 9. 环境变量检查
echo ========================================
echo PYTHONPATH: %PYTHONPATH%
echo HARVESTD_PORT: %HARVESTD_PORT%
echo VALIDATE_INTERVAL_MIN: %VALIDATE_INTERVAL_MIN%
echo.

REM 10. 提供修复建议
echo ========================================
echo 10. 修复建议
echo ========================================
if not exist "tools\harvestd.py" (
    echo [建议] 请确保在正确的项目根目录运行此脚本
    echo [建议] 检查文件路径是否正确
)

tasklist /FI "IMAGENAME eq python.exe" | findstr "python.exe" >nul
if errorlevel 1 (
    echo [建议] 没有发现Python进程，可能需要启动harvestd
    echo [建议] 运行: deploy\start_harvestd_fixed.bat
) else (
    echo [建议] 发现Python进程，检查是否为harvestd进程
)

echo.
echo ========================================
echo 诊断完成
echo ========================================
echo [建议] 如果问题持续，请检查日志文件获取详细错误信息
echo [建议] 日志位置: deploy\artifacts\run_logs\
echo.

:end
pause
