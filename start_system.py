#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 一键启动脚本
自动检测环境、配置API、启动所有系统模块
专为交易所带单设计，支持多AI融合决策，目标周收益20%+
"""
import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
import colorama
from colorama import Fore, Back, Style

# 初始化颜色输出
colorama.init()

class SystemLauncher:
    """系统启动器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.system_status = {}
        self.required_apis = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'OPENAI_API_KEY'
        ]
        
    def print_banner(self):
        """打印启动横幅"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          🚀 AI量化交易系统启动器                              ║
║                     专业级量化交易 • 多AI融合决策 • 实时监控                    ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}🎯 系统目标: 周收益20%+ | 🤖 AI驱动: 6大AI模型融合 | 📊 实时监控: Web界面{Style.RESET_ALL}
{Fore.GREEN}启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}
"""
        print(banner)
        
    def check_environment(self):
        """检查运行环境"""
        logger.info("🔍 检查运行环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            logger.error(f"❌ Python版本过低: {python_version.major}.{python_version.minor}, 需要3.8+")
            return False
            
        logger.success(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查必要的包
        required_packages = [
            'numpy', 'pandas', 'loguru', 'flask', 'flask_socketio',
            'ccxt', 'tensorflow', 'torch', 'transformers'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.success(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"⚠️ 缺少包: {package}")
        
        if missing_packages:
            logger.warning(f"⚠️ 缺少以下包: {', '.join(missing_packages)}")
            logger.info("💡 请运行: pip install -r requirements.txt")
            
        return True
        
    def check_api_configuration(self):
        """检查API配置"""
        logger.info("🔑 检查API配置...")
        
        missing_apis = []
        for api_key in self.required_apis:
            if not os.getenv(api_key):
                missing_apis.append(api_key)
                logger.warning(f"⚠️ 缺少API密钥: {api_key}")
            else:
                # 隐藏密钥显示
                key_preview = os.getenv(api_key)[:8] + "..." if os.getenv(api_key) else ""
                logger.success(f"✅ {api_key}: {key_preview}")
        
        if missing_apis:
            logger.warning("⚠️ 部分API密钥未配置，系统将使用模拟模式")
            self.prompt_api_configuration(missing_apis)
        else:
            logger.success("✅ 所有API密钥已配置")
            
        return True
        
    def prompt_api_configuration(self, missing_apis: List[str]):
        """提示用户配置API"""
        print(f"\n{Fore.YELLOW}🔧 API配置向导{Style.RESET_ALL}")
        print("为了获得最佳交易体验，请配置以下API密钥:")
        
        for api_key in missing_apis:
            print(f"\n{Fore.CYAN}📝 {api_key}:{Style.RESET_ALL}")
            if api_key.startswith('BINANCE'):
                print("  • 用于连接币安交易所")
                print("  • 获取地址: https://www.binance.com/cn/my/settings/api-management")
            elif api_key.startswith('OPENAI'):
                print("  • 用于AI分析和决策")
                print("  • 获取地址: https://platform.openai.com/api-keys")
                
        print(f"\n{Fore.GREEN}💡 配置方法:{Style.RESET_ALL}")
        print("1. 创建 .env 文件")
        print("2. 添加: API_KEY=your_key_here")
        print("3. 重新启动系统")
        
    def start_core_system(self):
        """启动核心系统"""
        logger.info("🚀 启动核心系统...")
        
        try:
            # 导入并启动主系统
            from main import QuantTradingSystem
            
            logger.info("🔧 初始化量化交易系统...")
            self.trading_system = QuantTradingSystem()
            
            # 在后台线程中启动系统
            system_thread = threading.Thread(
                target=self.trading_system.start_system,
                daemon=True
            )
            system_thread.start()
            
            logger.success("✅ 核心系统启动成功")
            self.system_status['core'] = True
            
        except Exception as e:
            logger.error(f"❌ 核心系统启动失败: {e}")
            self.system_status['core'] = False
            return False
            
        return True
        
    def start_web_interface(self):
        """启动Web界面"""
        logger.info("🌐 启动Web界面...")
        
        try:
            # 在后台线程中启动Web服务器
            web_thread = threading.Thread(
                target=self._run_web_server,
                daemon=True
            )
            web_thread.start()
            
            # 等待Web服务器启动
            time.sleep(3)
            
            logger.success("✅ Web界面启动成功")
            logger.info("🌐 访问地址: http://localhost:5000")
            self.system_status['web'] = True
            
        except Exception as e:
            logger.error(f"❌ Web界面启动失败: {e}")
            self.system_status['web'] = False
            return False
            
        return True
        
    def _run_web_server(self):
        """运行Web服务器"""
        try:
            from web.app import run_web_server
            run_web_server()
        except Exception as e:
            logger.error(f"Web服务器运行错误: {e}")
            
    def monitor_system_health(self):
        """监控系统健康状态"""
        logger.info("📊 启动系统健康监控...")
        
        while True:
            try:
                # 检查各个模块状态
                current_time = datetime.now()
                uptime = current_time - self.start_time
                
                # 每5分钟输出一次状态
                if uptime.total_seconds() % 300 == 0:
                    logger.info(f"💓 系统运行时间: {uptime}")
                    logger.info(f"📈 系统状态: {self.system_status}")
                
                time.sleep(60)  # 每分钟检查一次
                
            except KeyboardInterrupt:
                logger.info("🛑 收到停止信号，正在关闭系统...")
                break
            except Exception as e:
                logger.error(f"系统监控错误: {e}")
                time.sleep(60)
                
    def print_startup_summary(self):
        """打印启动总结"""
        print(f"\n{Fore.GREEN}🎉 系统启动完成!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                              🚀 系统状态总览                                  ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        for module, status in self.system_status.items():
            status_icon = "✅" if status else "❌"
            status_text = "运行中" if status else "失败"
            print(f"║ {status_icon} {module.upper():15} : {status_text:10}                                    ║")
            
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        print(f"║ 🌐 Web界面: http://localhost:5000                                            ║")
        print(f"║ 📊 实时监控: 交易数据、AI状态、系统性能                                        ║")
        print(f"║ 🤖 AI模型: 6大AI融合决策系统                                                  ║")
        print(f"║ 🎯 目标收益: 周收益20%+                                                       ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}💡 使用提示:{Style.RESET_ALL}")
        print("• 访问 http://localhost:5000 查看实时监控面板")
        print("• 按 Ctrl+C 安全停止系统")
        print("• 查看日志了解系统运行状态")
        
    def run(self):
        """运行启动器"""
        try:
            # 打印启动横幅
            self.print_banner()
            
            # 检查环境
            if not self.check_environment():
                return False
                
            # 检查API配置
            self.check_api_configuration()
            
            # 启动核心系统
            if not self.start_core_system():
                return False
                
            # 启动Web界面
            self.start_web_interface()
            
            # 打印启动总结
            self.print_startup_summary()
            
            # 开始系统健康监控
            self.monitor_system_health()
            
        except KeyboardInterrupt:
            logger.info("🛑 用户中断，正在安全关闭系统...")
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            return False
        finally:
            logger.info("👋 系统已关闭")
            
        return True

def main():
    """主函数"""
    launcher = SystemLauncher()
    return launcher.run()

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
