#!/usr/bin/env python3
"""
🚀 专业套利量化系统启动器 - 收益拉满版
Professional Arbitrage System Launcher - Maximum Profit Edition

功能：
- 🔧 一键启动套利系统
- 🌐 自动启动Web界面
- 📊 实时监控和控制
- 💰 复利增长追踪
- 🛡️ 智能风险管理
"""

import os
import sys
import time
import asyncio
import threading
import subprocess
from datetime import datetime

from api_config_manager import APIConfigManager
from arbitrage_system_core import arbitrage_system

def display_startup_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🚀 专业套利量化系统 - 收益拉满版                                            ║
║    Professional Arbitrage Quantitative System - Maximum Profit Edition      ║
║                                                                              ║
║    💰 多交易所套利 | 🔄 复利增长 | 📊 实时监控 | 🛡️ 智能风控                  ║
║                                                                              ║
║    🎯 目标日收益: 1.2% | 📈 年化收益: 5,493% | 💎 3年增长: 7,333倍           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查必要的包
    required_packages = [
        'flask', 'flask-socketio', 'eventlet', 'asyncio',
        'numpy', 'pandas', 'loguru', 'cryptography'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 安装缺失的包: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
        print("✅ 依赖包安装完成")
    
    print("✅ 系统要求检查通过")
    return True

def check_api_configurations():
    """检查API配置"""
    print("🔧 检查交易所API配置...")
    
    config_manager = APIConfigManager()
    has_configs = config_manager.load_configs()
    
    if not has_configs or config_manager.get_exchange_count() == 0:
        print("⚠️ 未找到交易所API配置")
        print("🔧 请先配置至少一个交易所API")
        
        setup_choice = input("\n是否现在配置交易所API? (Y/n): ").lower()
        if setup_choice in ['', 'y', 'yes']:
            config_manager.interactive_setup()
            
            if config_manager.get_exchange_count() == 0:
                print("❌ 未配置任何交易所，无法启动系统")
                return False
        else:
            print("❌ 需要配置交易所API才能启动套利系统")
            return False
    
    print(f"✅ 找到 {config_manager.get_exchange_count()} 个交易所配置")
    
    # 显示配置的交易所
    configs = config_manager.get_all_configs()
    for exchange_key, config in configs.items():
        exchange_info = config_manager.supported_exchanges.get(exchange_key, {"name": exchange_key})
        status_icon = "✅" if config.get("status") == "active" else "⚠️"
        print(f"   {status_icon} {exchange_info['name']}")
    
    return True

def setup_environment_variables():
    """设置环境变量"""
    print("🔧 设置环境变量...")
    
    config_manager = APIConfigManager()
    config_manager.load_configs()
    configs = config_manager.get_all_configs()
    
    # 设置交易所API环境变量
    for exchange_key, config in configs.items():
        prefix = exchange_key.upper()
        os.environ[f'{prefix}_API_KEY'] = config['api_key']
        os.environ[f'{prefix}_SECRET_KEY'] = config['secret_key']
        if 'passphrase' in config:
            os.environ[f'{prefix}_PASSPHRASE'] = config['passphrase']
        print(f"✅ {exchange_key.title()} API环境变量已设置")

async def initialize_arbitrage_system():
    """初始化套利系统"""
    print("🚀 初始化专业套利系统...")
    
    # 初始化系统
    success = await arbitrage_system.initialize_system()
    
    if not success:
        print("❌ 套利系统初始化失败")
        return False
    
    print("✅ 套利系统初始化成功")
    return True

def start_web_interface():
    """启动Web界面"""
    print("🌐 启动Web控制界面...")
    
    def run_web_server():
        try:
            from arbitrage_web_interface import app, socketio
            print("📊 Web界面启动成功!")
            print("🌐 访问地址: http://localhost:5000")
            print("📱 支持手机和电脑访问")
            socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        except Exception as e:
            print(f"❌ Web界面启动失败: {e}")
    
    # 在后台线程中启动Web服务器
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # 等待Web服务器启动
    time.sleep(3)
    return True

def display_system_info():
    """显示系统信息"""
    print("\n" + "="*80)
    print("📊 专业套利量化系统信息")
    print("="*80)
    
    status = arbitrage_system.get_system_status()
    
    print(f"💰 当前资金: {status['current_capital']:.2f} USDT")
    print(f"📈 总利润: {status['total_profit']:.2f} USDT")
    print(f"📊 增长率: {status['growth_rate']*100:.2f}%")
    print(f"🏦 已连接交易所: {status['connected_exchanges']} 个")
    print(f"🎯 日收益目标: {status['daily_target_rate']*100:.2f}%")
    print(f"📋 总交易次数: {status['stats']['total_trades']}")
    print(f"🏆 胜率: {status['stats']['win_rate']*100:.2f}%")
    
    print("\n🎯 复利增长预测:")
    current_capital = status['current_capital']
    daily_rate = status['daily_target_rate']
    
    projections = [
        (30, "1个月"),
        (90, "3个月"),
        (180, "6个月"),
        (365, "1年"),
        (730, "2年"),
        (1095, "3年")
    ]
    
    for days, period in projections:
        projected_capital = current_capital * (1 + daily_rate) ** days
        growth_rate = (projected_capital - current_capital) / current_capital * 100
        print(f"   {period:6}: {projected_capital:10.2f} USDT (+{growth_rate:8.1f}%)")
    
    print("="*80)

def display_control_menu():
    """显示控制菜单"""
    print("\n📋 系统控制菜单:")
    print("1. 🚀 启动套利引擎")
    print("2. 🛑 停止套利引擎")
    print("3. 📊 查看系统状态")
    print("4. 🔧 管理API配置")
    print("5. 🌐 打开Web界面")
    print("6. 📈 查看复利预测")
    print("7. 🔄 重启系统")
    print("0. 👋 退出系统")

async def handle_user_input():
    """处理用户输入"""
    while True:
        try:
            display_control_menu()
            choice = input("\n请选择操作 (0-7): ").strip()
            
            if choice == "1":
                print("🚀 启动套利引擎...")
                # 在新任务中启动套利引擎
                asyncio.create_task(arbitrage_system.start_arbitrage_engine())
                print("✅ 套利引擎已启动")
                
            elif choice == "2":
                print("🛑 停止套利引擎...")
                arbitrage_system.stop_system()
                print("✅ 套利引擎已停止")
                
            elif choice == "3":
                display_system_info()
                
            elif choice == "4":
                print("🔧 启动API配置管理...")
                config_manager = APIConfigManager()
                config_manager.interactive_setup()
                setup_environment_variables()
                print("✅ API配置已更新")
                
            elif choice == "5":
                try:
                    import webbrowser
                    webbrowser.open("http://localhost:5000")
                    print("✅ 已在浏览器中打开Web界面")
                except Exception as e:
                    print(f"❌ 无法自动打开浏览器: {e}")
                    print("请手动访问: http://localhost:5000")
                    
            elif choice == "6":
                display_compound_projection()
                
            elif choice == "7":
                print("🔄 重启系统...")
                arbitrage_system.stop_system()
                await asyncio.sleep(2)
                await initialize_arbitrage_system()
                print("✅ 系统重启完成")
                
            elif choice == "0":
                print("👋 感谢使用专业套利量化系统，再见!")
                arbitrage_system.stop_system()
                break
                
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，系统退出")
            arbitrage_system.stop_system()
            break
        except Exception as e:
            print(f"❌ 操作错误: {e}")

def display_compound_projection():
    """显示复利预测"""
    print("\n" + "="*80)
    print("📈 复利增长详细预测")
    print("="*80)
    
    current_capital = arbitrage_system.current_capital
    daily_rate = arbitrage_system.daily_target_rate
    
    print(f"起始资金: {current_capital:.2f} USDT")
    print(f"日收益率: {daily_rate*100:.2f}%")
    print()
    
    # 详细预测
    periods = [
        (1, "第1天"), (7, "第1周"), (14, "第2周"), (30, "第1月"),
        (60, "第2月"), (90, "第3月"), (180, "第6月"), (365, "第1年"),
        (547, "第1.5年"), (730, "第2年"), (912, "第2.5年"), (1095, "第3年")
    ]
    
    print("时间节点        资金规模        利润增长        增长倍数")
    print("-" * 60)
    
    for days, period in periods:
        projected_capital = current_capital * (1 + daily_rate) ** days
        profit = projected_capital - current_capital
        multiplier = projected_capital / current_capital
        
        print(f"{period:12} {projected_capital:12.2f} USDT {profit:12.2f} USDT {multiplier:8.1f}x")
    
    print("="*80)
    print("💡 提示: 以上预测基于理想情况，实际收益可能因市场条件而有所不同")

async def main():
    """主函数"""
    try:
        # 显示启动横幅
        display_startup_banner()
        
        # 检查系统要求
        if not check_system_requirements():
            return
        
        # 检查API配置
        if not check_api_configurations():
            return
        
        # 设置环境变量
        setup_environment_variables()
        
        # 初始化套利系统
        if not await initialize_arbitrage_system():
            return
        
        # 启动Web界面
        start_web_interface()
        
        # 显示系统信息
        display_system_info()
        
        print("\n🎉 专业套利量化系统启动完成!")
        print("🌐 Web控制界面: http://localhost:5000")
        print("📊 实时监控和控制您的套利系统")
        
        # 处理用户输入
        await handle_user_input()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，系统退出")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        arbitrage_system.stop_system()

if __name__ == "__main__":
    asyncio.run(main())
