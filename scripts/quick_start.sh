#!/bin/bash

# 🚀 AI量化交易系统快速启动脚本

set -e

echo "🚀 AI量化交易系统快速启动"
echo "================================"

# 检查Python版本
echo "📋 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查是否安装了Poetry
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry未安装，正在安装..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# 安装依赖
echo "📦 安装Python依赖..."
poetry install

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到.env文件，复制示例配置..."
    cp .env.example .env
    echo "✅ 请编辑.env文件并填入真实的API密钥"
fi

# 检查Docker
if command -v docker &> /dev/null; then
    echo "🐳 检测到Docker，启动基础服务..."
    docker-compose up -d redis clickhouse postgres kafka
    echo "⏳ 等待服务启动..."
    sleep 10
else
    echo "⚠️  未检测到Docker，请手动启动数据库服务"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p logs models data

# 启动系统
echo "🚀 启动AI量化交易系统..."
poetry run python start.py

echo "✅ 系统启动完成！"
echo "🌐 Web界面: http://localhost:8000"
echo "📊 监控面板: http://localhost:3001"
