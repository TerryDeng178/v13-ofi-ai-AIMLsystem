@echo off
echo ========================================
echo V13 OFI+CVD 快速推送脚本
echo ========================================

echo 正在检查Git状态...
git status

echo.
echo 正在添加所有更改...
git add .

echo.
echo 请输入提交信息（或按回车使用默认信息）:
set /p commit_msg="提交信息: "

if "%commit_msg%"=="" (
    set commit_msg=feat: Update V13 OFI+CVD system
)

echo.
echo 正在提交更改...
git commit -m "%commit_msg%"

echo.
echo 正在推送到GitHub...
git push origin main

echo.
echo ========================================
echo 推送完成！
echo 仓库地址: https://github.com/TerryDeng178/v13-ofi-ai-AIMLsystem
echo ========================================
pause
