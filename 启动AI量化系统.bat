@echo off
chcp 65001 >nul
title AI量化交易系统 - 一键启动

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║    🤖 AI量化交易系统 - 专业级合约交易平台                      ║
echo ║    AI Quantitative Trading System - Professional Platform    ║
echo ║                                                              ║
echo ║    🚀 一键启动 ^| 💰 实时数据 ^| 🧠 AI决策 ^| 🛡️ 风险控制        ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 正在启动AI量化交易系统...
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

REM 启动系统
python start_with_api_config.py

echo.
echo 👋 感谢使用AI量化交易系统！
pause
