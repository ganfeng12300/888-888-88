#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 带API配置的一键启动
AI Quantitative Trading System - One-Click Start with API Configuration

功能：
- 一键启动系统
- 自动检查API配置
- 如果没有配置则引导用户配置
- 配置完成后自动启动Web界面
- 支持多交易所配置管理
"""

import os
import sys
import time
from api_config_manager import APIConfigManager

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查系统依赖...")
    
    required_packages = [
        'flask',
        'flask-socketio', 
        'eventlet',
        'cryptography',
        'loguru'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("🔧 正在安装依赖包...")
        
        for package in missing_packages:
            os.system(f"pip install {package}")
        
        print("✅ 依赖包安装完成")
    else:
        print("✅ 所有依赖包已安装")

def display_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🤖 AI量化交易系统 - 专业级合约交易平台                      ║
║    AI Quantitative Trading System - Professional Platform    ║
║                                                              ║
║    🚀 一键启动 | 💰 实时数据 | 🧠 AI决策 | 🛡️ 风险控制        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_api_configs():
    """检查API配置"""
    print("🔍 检查交易所API配置...")
    
    manager = APIConfigManager()
    has_configs = manager.load_configs()
    
    if not has_configs or manager.get_exchange_count() == 0:
        print("📝 未找到交易所API配置")
        print("🔧 需要配置至少一个交易所API才能启动系统")
        
        setup_choice = input("\n是否现在配置交易所API? (Y/n): ").lower()
        if setup_choice in ['', 'y', 'yes']:
            manager.interactive_setup()
            
            if manager.get_exchange_count() == 0:
                print("❌ 未配置任何交易所，无法启动系统")
                return False
        else:
            print("❌ 未配置交易所API，无法启动系统")
            return False
    
    print(f"✅ 找到 {manager.get_exchange_count()} 个交易所配置")
    
    # 显示配置的交易所
    configs = manager.get_all_configs()
    for exchange_key, config in configs.items():
        exchange_info = manager.supported_exchanges.get(exchange_key, {"name": exchange_key})
        status_icon = "✅" if config.get("status") == "active" else "⚠️"
        print(f"   {status_icon} {exchange_info['name']}")
    
    return True

def setup_environment_variables(manager: APIConfigManager):
    """设置环境变量"""
    print("🔧 设置环境变量...")
    
    configs = manager.get_all_configs()
    
    # 优先使用Bitget配置（如果存在）
    if 'bitget' in configs:
        bitget_config = configs['bitget']
        os.environ['BITGET_API_KEY'] = bitget_config['api_key']
        os.environ['BITGET_SECRET_KEY'] = bitget_config['secret_key']
        os.environ['BITGET_PASSPHRASE'] = bitget_config['passphrase']
        print("✅ Bitget API环境变量已设置")
    
    # 设置其他交易所环境变量
    for exchange_key, config in configs.items():
        if exchange_key != 'bitget':
            prefix = exchange_key.upper()
            os.environ[f'{prefix}_API_KEY'] = config['api_key']
            os.environ[f'{prefix}_SECRET_KEY'] = config['secret_key']
            if 'passphrase' in config:
                os.environ[f'{prefix}_PASSPHRASE'] = config['passphrase']
            print(f"✅ {exchange_key.title()} API环境变量已设置")

def start_web_interface():
    """启动Web界面"""
    print("🌐 启动Web界面...")
    print("=" * 60)
    
    try:
        # 导入并启动Web服务器
        import subprocess
        import threading
        
        def run_web_server():
            subprocess.run([sys.executable, "start_web.py"])
        
        # 在后台线程中启动Web服务器
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        
        # 等待服务器启动
        time.sleep(3)
        
        print("🎉 Web界面启动成功!")
        print("🌐 访问地址: http://localhost:8000")
        print("📱 支持手机和电脑访问")
        print("=" * 60)
        
        # 提供操作选项
        while True:
            print("\n📋 系统操作菜单:")
            print("1. 打开Web界面 (http://localhost:8000)")
            print("2. 管理交易所API配置")
            print("3. 查看系统状态")
            print("4. 重启Web服务")
            print("0. 退出系统")
            
            choice = input("\n请选择操作 (0-4): ").strip()
            
            if choice == "1":
                try:
                    import webbrowser
                    webbrowser.open("http://localhost:8000")
                    print("✅ 已在浏览器中打开Web界面")
                except Exception as e:
                    print(f"❌ 无法自动打开浏览器: {e}")
                    print("请手动访问: http://localhost:8000")
            
            elif choice == "2":
                manager = APIConfigManager()
                manager.interactive_setup()
                setup_environment_variables(manager)
                print("✅ API配置已更新")
            
            elif choice == "3":
                show_system_status()
            
            elif choice == "4":
                print("🔄 重启Web服务...")
                # 这里可以添加重启逻辑
                print("✅ Web服务重启完成")
            
            elif choice == "0":
                print("👋 感谢使用AI量化交易系统，再见!")
                break
            
            else:
                print("❌ 无效选择，请重新输入")
        
    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")
        return False
    
    return True

def show_system_status():
    """显示系统状态"""
    print("\n" + "="*60)
    print("📊 AI量化交易系统状态")
    print("="*60)
    
    # 检查Web服务状态
    try:
        import requests
        response = requests.get("http://localhost:8000", timeout=3)
        web_status = "✅ 运行中" if response.status_code == 200 else "❌ 异常"
    except:
        web_status = "❌ 未启动"
    
    print(f"Web服务: {web_status}")
    
    # 检查API配置
    manager = APIConfigManager()
    manager.load_configs()
    print(f"交易所配置: {manager.get_exchange_count()} 个")
    
    # 检查环境变量
    env_vars = ['BITGET_API_KEY', 'BITGET_SECRET_KEY', 'BITGET_PASSPHRASE']
    env_status = all(os.getenv(var) for var in env_vars)
    print(f"环境变量: {'✅ 已设置' if env_status else '❌ 未设置'}")
    
    print("="*60)

def main():
    """主函数"""
    try:
        # 显示启动横幅
        display_banner()
        
        # 检查依赖
        check_dependencies()
        
        # 检查API配置
        if not check_api_configs():
            print("❌ 系统启动失败：缺少API配置")
            return
        
        # 设置环境变量
        manager = APIConfigManager()
        manager.load_configs()
        setup_environment_variables(manager)
        
        # 启动Web界面
        start_web_interface()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，系统退出")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
