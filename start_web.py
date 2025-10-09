#!/usr/bin/env python3
"""
🚀 启动专业级Web界面
Launch Professional Web Interface

快速启动命令：
python start_web.py
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def setup_environment():
    """设置环境变量"""
    # 设置Bitget API密钥
    os.environ['BITGET_API_KEY'] = 'bg_361f925c6f2139ad15bff1e662995fdd'
    os.environ['BITGET_SECRET_KEY'] = '6b9f6868b5c6e90b4a866d1a626c3722a169e557dfcfd2175fbeb5fa84085c43'
    os.environ['BITGET_PASSPHRASE'] = 'Ganfeng321'
    
    print("✅ 环境变量配置完成")

def check_dependencies():
    """检查依赖"""
    try:
        import flask
        import flask_socketio
        import eventlet
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("正在安装依赖...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-socketio', 'eventlet'])
        return True

def start_web_server():
    """启动Web服务器"""
    print("🚀 启动AI量化交易系统Web界面...")
    print("📊 功能特性:")
    print("   - 实时合约账户数据")
    print("   - 专业级交易界面")
    print("   - 实时终端日志")
    print("   - WebSocket数据推送")
    print("   - 响应式设计")
    print()
    
    try:
        # 启动Web服务器
        subprocess.run([sys.executable, 'web_server.py'])
    except KeyboardInterrupt:
        print("\n🛑 Web服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 AI量化交易系统 - 专业级Web界面")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 启动服务器
    start_web_server()

if __name__ == '__main__':
    main()
