#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 一键启动脚本
支持多交易所API配置，生产级实盘交易
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
        self.exchange_configs = {}
        self.required_apis = [
            'OPENAI_API_KEY'
        ]
        
        # 支持的交易所配置
        self.supported_exchanges = {
            'binance': {
                'name': 'Binance',
                'api_key_env': 'BINANCE_API_KEY',
                'secret_env': 'BINANCE_SECRET_KEY',
                'testnet_env': 'BINANCE_TESTNET'
            },
            'okex': {
                'name': 'OKEx',
                'api_key_env': 'OKEX_API_KEY',
                'secret_env': 'OKEX_SECRET_KEY',
                'passphrase_env': 'OKEX_PASSPHRASE'
            },
            'huobi': {
                'name': 'Huobi',
                'api_key_env': 'HUOBI_API_KEY',
                'secret_env': 'HUOBI_SECRET_KEY'
            },
            'bybit': {
                'name': 'Bybit',
                'api_key_env': 'BYBIT_API_KEY',
                'secret_env': 'BYBIT_SECRET_KEY'
            },
            'gate': {
                'name': 'Gate.io',
                'api_key_env': 'GATE_API_KEY',
                'secret_env': 'GATE_SECRET_KEY'
            },
            'kucoin': {
                'name': 'KuCoin',
                'api_key_env': 'KUCOIN_API_KEY',
                'secret_env': 'KUCOIN_SECRET_KEY',
                'passphrase_env': 'KUCOIN_PASSPHRASE'
            },
            'bitget': {
                'name': 'Bitget',
                'api_key_env': 'BITGET_API_KEY',
                'secret_env': 'BITGET_SECRET_KEY',
                'passphrase_env': 'BITGET_PASSPHRASE',
                'testnet_env': 'BITGET_TESTNET'
            }
        }
        
    def print_banner(self):
        """打印启动横幅"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          🚀 AI量化交易系统启动器                              ║
║                     专业级量化交易 • 多AI融合决策 • 实时监控                    ║
║                     多交易所支持 • 生产级实盘交易 • 统一信号分发                ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}🎯 系统目标: 周收益20%+ | 🤖 AI驱动: 6大AI模型融合 | 📊 实时监控: Web界面{Style.RESET_ALL}
{Fore.GREEN}🏦 多交易所: 统一信号分发 | 🔒 风险控制: 多层安全保障 | ⚡ 实盘交易: 生产级代码{Style.RESET_ALL}
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
            'ccxt', 'talib', 'colorama'
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
        
    def check_exchange_apis(self):
        """检查交易所API配置"""
        logger.info("🏦 检查交易所API配置...")
        
        configured_exchanges = []
        
        for exchange_id, config in self.supported_exchanges.items():
            api_key = os.getenv(config['api_key_env'])
            secret = os.getenv(config['secret_env'])
            
            if api_key and secret:
                # 检查passphrase（如果需要）
                passphrase = None
                if 'passphrase_env' in config:
                    passphrase = os.getenv(config['passphrase_env'])
                    if not passphrase:
                        logger.warning(f"⚠️ {config['name']} 缺少Passphrase")
                        continue
                
                # 检查测试网配置
                testnet = False
                if 'testnet_env' in config:
                    testnet = os.getenv(config['testnet_env'], '').lower() in ['true', '1', 'yes']
                
                self.exchange_configs[exchange_id] = {
                    'name': config['name'],
                    'api_key': api_key,
                    'secret': secret,
                    'passphrase': passphrase,
                    'testnet': testnet
                }
                
                # 隐藏密钥显示
                key_preview = api_key[:8] + "..." if api_key else ""
                testnet_info = " (测试网)" if testnet else ""
                logger.success(f"✅ {config['name']}: {key_preview}{testnet_info}")
                configured_exchanges.append(config['name'])
            else:
                logger.info(f"⚪ {config['name']}: 未配置")
        
        if not configured_exchanges:
            logger.warning("⚠️ 未配置任何交易所API，系统将使用模拟模式")
            self.prompt_exchange_configuration()
        else:
            logger.success(f"✅ 已配置交易所: {', '.join(configured_exchanges)}")
            
        return True
        
    def check_api_configuration(self):
        """检查其他API配置"""
        logger.info("🔑 检查其他API配置...")
        
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
            logger.warning("⚠️ 部分API密钥未配置，相关功能可能受限")
            self.prompt_api_configuration(missing_apis)
        else:
            logger.success("✅ 所有API密钥已配置")
            
        return True
        
    def prompt_exchange_configuration(self):
        """提示用户配置交易所API"""
        print(f"\n{Fore.YELLOW}🏦 交易所API配置向导{Style.RESET_ALL}")
        print("为了进行实盘交易，请配置至少一个交易所的API密钥:")
        
        for exchange_id, config in self.supported_exchanges.items():
            print(f"\n{Fore.CYAN}📝 {config['name']}:{Style.RESET_ALL}")
            print(f"  • API Key: {config['api_key_env']}")
            print(f"  • Secret: {config['secret_env']}")
            if 'passphrase_env' in config:
                print(f"  • Passphrase: {config['passphrase_env']}")
            if 'testnet_env' in config:
                print(f"  • 测试网: {config['testnet_env']} (可选)")
                
        print(f"\n{Fore.GREEN}💡 配置方法:{Style.RESET_ALL}")
        print("1. 创建 .env 文件")
        print("2. 添加交易所API配置:")
        print("   BINANCE_API_KEY=your_binance_api_key")
        print("   BINANCE_SECRET_KEY=your_binance_secret_key")
        print("   BINANCE_TESTNET=false")
        print("3. 重新启动系统")
        
        print(f"\n{Fore.RED}⚠️ 重要提示:{Style.RESET_ALL}")
        print("• 请确保API密钥具有交易权限")
        print("• 建议先在测试网环境测试")
        print("• 妥善保管API密钥，避免泄露")
        
    def prompt_api_configuration(self, missing_apis: List[str]):
        """提示用户配置API"""
        print(f"\n{Fore.YELLOW}🔧 API配置向导{Style.RESET_ALL}")
        print("为了获得最佳交易体验，请配置以下API密钥:")
        
        for api_key in missing_apis:
            print(f"\n{Fore.CYAN}📝 {api_key}:{Style.RESET_ALL}")
            if api_key.startswith('OPENAI'):
                print("  • 用于AI分析和决策")
                print("  • 获取地址: https://platform.openai.com/api-keys")
                
        print(f"\n{Fore.GREEN}💡 配置方法:{Style.RESET_ALL}")
        print("1. 创建 .env 文件")
        print("2. 添加: API_KEY=your_key_here")
        print("3. 重新启动系统")
        
    def configure_exchanges(self):
        """配置交易所连接"""
        if not self.exchange_configs:
            logger.info("📊 未配置交易所API，系统将以监控模式运行")
            return True
            
        logger.info("🏦 配置交易所连接...")
        
        try:
            from src.exchanges.multi_exchange_manager import multi_exchange_manager, ExchangeConfig
            
            for exchange_id, config in self.exchange_configs.items():
                try:
                    exchange_config = ExchangeConfig(
                        name=exchange_id,
                        api_key=config['api_key'],
                        secret=config['secret'],
                        passphrase=config.get('passphrase'),
                        testnet=config.get('testnet', False)
                    )
                    
                    success = multi_exchange_manager.add_exchange(exchange_config)
                    if success:
                        logger.success(f"✅ {config['name']} 连接成功")
                    else:
                        logger.error(f"❌ {config['name']} 连接失败")
                        
                except Exception as e:
                    logger.error(f"❌ {config['name']} 配置失败: {e}")
                    
            active_exchanges = multi_exchange_manager.get_active_exchanges()
            if active_exchanges:
                logger.success(f"🎉 成功连接 {len(active_exchanges)} 个交易所: {', '.join(active_exchanges)}")
                self.system_status['exchanges'] = True
            else:
                logger.warning("⚠️ 未成功连接任何交易所")
                self.system_status['exchanges'] = False
                
        except Exception as e:
            logger.error(f"❌ 交易所配置失败: {e}")
            self.system_status['exchanges'] = False
            
        return True
        
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
        
    def start_signal_generator(self):
        """启动信号生成器"""
        logger.info("📡 启动信号生成器...")
        
        try:
            from src.strategies.production_signal_generator import production_signal_generator
            
            production_signal_generator.start_generation()
            
            logger.success("✅ 信号生成器启动成功")
            self.system_status['signal_generator'] = True
            
        except Exception as e:
            logger.error(f"❌ 信号生成器启动失败: {e}")
            self.system_status['signal_generator'] = False
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
            
    def start_trading_loop(self):
        """启动交易循环"""
        if not self.exchange_configs:
            logger.info("📊 未配置交易所，跳过交易循环启动")
            return True
            
        logger.info("🔄 启动交易循环...")
        
        try:
            # 启动交易循环线程
            trading_thread = threading.Thread(
                target=self._trading_loop,
                daemon=True
            )
            trading_thread.start()
            
            logger.success("✅ 交易循环启动成功")
            self.system_status['trading'] = True
            
        except Exception as e:
            logger.error(f"❌ 交易循环启动失败: {e}")
            self.system_status['trading'] = False
            return False
            
        return True
        
    def _trading_loop(self):
        """交易循环"""
        from src.exchanges.multi_exchange_manager import multi_exchange_manager
        from src.strategies.production_signal_generator import production_signal_generator
        
        logger.info("🔄 交易循环开始运行...")
        
        # 支持的交易对
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        while True:
            try:
                for symbol in symbols:
                    # 生成交易信号
                    signal = production_signal_generator.generate_signal(symbol)
                    
                    if signal:
                        logger.info(f"🎯 收到交易信号: {signal.symbol} {signal.action}")
                        
                        # 转换为交易信号
                        trading_signal = production_signal_generator.convert_to_trading_signal(signal)
                        
                        # 广播到所有交易所
                        results = multi_exchange_manager.broadcast_signal(trading_signal)
                        
                        # 记录结果
                        success_count = sum(1 for r in results if r.status not in ['failed', 'timeout'])
                        logger.info(f"📊 信号执行结果: {success_count}/{len(results)} 成功")
                
                # 等待下一轮
                time.sleep(60)  # 每分钟检查一次
                
            except KeyboardInterrupt:
                logger.info("🛑 交易循环收到停止信号")
                break
            except Exception as e:
                logger.error(f"❌ 交易循环错误: {e}")
                time.sleep(30)  # 错误后等待30秒
                
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
                    
                    # 检查交易所健康状态
                    if self.exchange_configs:
                        try:
                            from src.exchanges.multi_exchange_manager import multi_exchange_manager
                            health = multi_exchange_manager.health_check()
                            logger.info(f"🏦 交易所状态: {health['overall_status']}")
                        except:
                            pass
                
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
            module_name = {
                'core': '核心系统',
                'exchanges': '交易所连接',
                'signal_generator': '信号生成器',
                'trading': '交易循环',
                'web': 'Web界面'
            }.get(module, module.upper())
            print(f"║ {status_icon} {module_name:15} : {status_text:10}                                    ║")
            
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # 显示配置的交易所
        if self.exchange_configs:
            exchange_names = [config['name'] for config in self.exchange_configs.values()]
            print(f"║ 🏦 已配置交易所: {', '.join(exchange_names)[:50]:50}                    ║")
        else:
            print(f"║ 🏦 交易所: 未配置 (监控模式)                                               ║")
            
        print(f"║ 🌐 Web界面: http://localhost:5000                                            ║")
        print(f"║ 📊 实时监控: 交易数据、AI状态、系统性能                                        ║")
        print(f"║ 🤖 AI模型: 6大AI融合决策系统                                                  ║")
        print(f"║ 🎯 目标收益: 周收益20%+                                                       ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}💡 使用提示:{Style.RESET_ALL}")
        print("• 访问 http://localhost:5000 查看实时监控面板")
        print("• 按 Ctrl+C 安全停止系统")
        print("• 查看日志了解系统运行状态")
        
        if self.exchange_configs:
            print(f"\n{Fore.GREEN}🚀 实盘交易模式:{Style.RESET_ALL}")
            print("• 系统将自动生成交易信号")
            print("• 信号将同时发送到所有配置的交易所")
            print("• 请确保账户有足够余额进行交易")
        else:
            print(f"\n{Fore.BLUE}📊 监控模式:{Style.RESET_ALL}")
            print("• 系统以监控模式运行，不会执行实际交易")
            print("• 配置交易所API后可启用实盘交易")
        
    def run(self):
        """运行启动器"""
        try:
            # 打印启动横幅
            self.print_banner()
            
            # 检查环境
            if not self.check_environment():
                return False
                
            # 检查交易所API配置
            self.check_exchange_apis()
            
            # 检查其他API配置
            self.check_api_configuration()
            
            # 配置交易所连接
            self.configure_exchanges()
            
            # 启动核心系统
            if not self.start_core_system():
                return False
                
            # 启动信号生成器
            self.start_signal_generator()
                
            # 启动Web界面
            self.start_web_interface()
            
            # 启动交易循环
            self.start_trading_loop()
            
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
