@echo off
chcp 65001 >nul
title 🚀 专业套利量化系统 - 收益拉满版

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                              ║
echo ║    🚀 专业套利量化系统 - 收益拉满版                                            ║
echo ║    Professional Arbitrage Quantitative System - Maximum Profit Edition      ║
echo ║                                                                              ║
echo ║    💰 多交易所套利 ^| 🔄 复利增长 ^| 📊 实时监控 ^| 🛡️ 智能风控                  ║
echo ║                                                                              ║
echo ║    🎯 目标日收益: 1.2%% ^| 📈 年化收益: 5,493%% ^| 💎 3年增长: 7,333倍           ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo 🚀 正在启动专业套利量化系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.8+
    echo 📥 下载地址: https://python.org
    pause
    exit /b 1
)

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 启动专业套利系统
python start_arbitrage_system.py

echo.
echo 👋 感谢使用专业套利量化系统！
pause
