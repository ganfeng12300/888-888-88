#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 实盘交易启动器
专为实盘交易设计，支持7大主流交易所
无测试网，无模拟，纯实盘API，专业级量化交易
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
from dotenv import load_dotenv

# 初始化颜色输出
colorama.init()

# 加载.env文件
load_dotenv()

class ProductionSystemLauncher:
    """生产级系统启动器 - 实盘交易专用"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.system_status = {}
        self.exchange_configs = {}
        self.required_apis = [
            'OPENAI_API_KEY'
        ]
        
        # 支持的交易所配置 - 实盘交易专用
        self.supported_exchanges = {
            'binance': {
                'name': 'Binance',
                'api_key_env': 'BINANCE_API_KEY',
                'secret_env': 'BINANCE_SECRET_KEY'
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
                'passphrase_env': 'BITGET_PASSPHRASE'
            }
        }
        
    def print_banner(self):
        """打印启动横幅"""
        banner = f"""
{Fore.RED}╔══════════════════════════════════════════════════════════════════════════════╗
║                      🔥 AI量化交易系统 - 实盘交易启动器                        ║
║                   专业级实盘交易 • 7大交易所 • 真实资金操作                     ║
║                   ⚠️  WARNING: LIVE TRADING WITH REAL MONEY ⚠️               ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}🎯 实盘目标: 周收益20%+ | 🤖 AI驱动: 6大AI模型融合 | 📊 实时监控: Web界面{Style.RESET_ALL}
{Fore.RED}💰 真实资金: 实盘交易模式 | 🔒 风险控制: 多层安全保障 | ⚡ 7大交易所: 统一执行{Style.RESET_ALL}
{Fore.GREEN}启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}

{Fore.RED}⚠️  重要提醒: 本系统将使用真实资金进行交易，请确保您已充分了解风险！{Style.RESET_ALL}
"""
        print(banner)
        
    def check_environment(self):
        """检查运行环境"""
        logger.info("🔍 检查生产环境...")
        
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
        
    def interactive_api_configuration(self):
        """交互式API配置"""
        print(f"\n{Fore.CYAN}🏦 交易所API配置向导{Style.RESET_ALL}")
        print("请为您要使用的交易所配置实盘API密钥")
        print("⚠️ 注意：这些API将用于真实资金交易！")
        
        configured_count = 0
        
        for exchange_id, config in self.supported_exchanges.items():
            print(f"\n{Fore.YELLOW}━━━ {config['name']} 配置 ━━━{Style.RESET_ALL}")
            
            # 询问是否配置此交易所
            while True:
                choice = input(f"是否配置 {config['name']} 交易所？(y/n/skip): ").lower().strip()
                if choice in ['y', 'yes', '是']:
                    break
                elif choice in ['n', 'no', '否', 'skip', 's']:
                    print(f"⚪ 跳过 {config['name']} 配置")
                    break
                else:
                    print("请输入 y(是) 或 n(否)")
            
            if choice in ['n', 'no', '否', 'skip', 's']:
                continue
                
            # 配置API密钥
            print(f"\n📝 请输入 {config['name']} 的实盘API信息:")
            
            # API Key
            while True:
                api_key = input(f"API Key: ").strip()
                if api_key:
                    break
                print("❌ API Key不能为空，请重新输入")
            
            # Secret Key
            while True:
                secret = input(f"Secret Key: ").strip()
                if secret:
                    break
                print("❌ Secret Key不能为空，请重新输入")
            
            # Passphrase (如果需要)
            passphrase = None
            if 'passphrase_env' in config:
                while True:
                    passphrase = input(f"Passphrase: ").strip()
                    if passphrase:
                        break
                    print("❌ Passphrase不能为空，请重新输入")
            
            # 保存配置
            self.exchange_configs[exchange_id] = {
                'name': config['name'],
                'api_key': api_key,
                'secret': secret,
                'passphrase': passphrase
            }
            
            configured_count += 1
            key_preview = api_key[:8] + "..." if len(api_key) > 8 else api_key
            print(f"✅ {config['name']} 配置完成: {key_preview}")
        
        if configured_count == 0:
            print(f"\n{Fore.RED}❌ 未配置任何交易所API！{Style.RESET_ALL}")
            print("实盘交易系统需要至少配置一个交易所API才能启动")
            return False
        
        print(f"\n{Fore.GREEN}🎉 成功配置 {configured_count} 个交易所API{Style.RESET_ALL}")
        
        # 显示配置摘要
        print(f"\n{Fore.CYAN}📋 配置摘要:{Style.RESET_ALL}")
        for exchange_id, config in self.exchange_configs.items():
            key_preview = config['api_key'][:8] + "..." if len(config['api_key']) > 8 else config['api_key']
            print(f"  ✅ {config['name']}: {key_preview}")
        
        # 最终确认
        print(f"\n{Fore.RED}⚠️ 最终确认 ⚠️{Style.RESET_ALL}")
        print("以上配置将用于真实资金交易，请确认无误！")
        
        while True:
            confirm = input("确认使用以上配置进行实盘交易？(yes/no): ").lower().strip()
            if confirm in ['yes', 'y', '是']:
                return True
            elif confirm in ['no', 'n', '否']:
                print("❌ 用户取消配置")
                return False
            else:
                print("请输入 yes 或 no")
    
    def check_exchange_apis(self):
        """检查交易所API配置"""
        logger.info("🏦 检查实盘交易所API配置...")
        
        # 首先检查环境变量中是否有配置
        env_configured_exchanges = []
        
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
                
                self.exchange_configs[exchange_id] = {
                    'name': config['name'],
                    'api_key': api_key,
                    'secret': secret,
                    'passphrase': passphrase
                }
                
                # 隐藏密钥显示
                key_preview = api_key[:8] + "..." if api_key else ""
                logger.success(f"✅ {config['name']}: {key_preview} (来自环境变量)")
                env_configured_exchanges.append(config['name'])
        
        if env_configured_exchanges:
            logger.success(f"✅ 从环境变量加载: {', '.join(env_configured_exchanges)}")
            return True
        
        # 如果环境变量中没有配置，提示用户配置并退出
        logger.error("❌ 未配置任何交易所API，系统无法启动")
        print(f"\n{Fore.RED}❌ 错误：未配置任何交易所API{Style.RESET_ALL}")
        print("实盘交易系统需要至少配置一个交易所API才能启动。")
        self.prompt_exchange_configuration()
        return False
        
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
        print(f"\n{Fore.RED}🚨 实盘交易API配置向导{Style.RESET_ALL}")
        print("⚠️ 警告：以下配置将用于真实资金交易，请谨慎操作！")
        
        for exchange_id, config in self.supported_exchanges.items():
            print(f"\n{Fore.CYAN}📝 {config['name']} (实盘):{Style.RESET_ALL}")
            print(f"  • API Key: {config['api_key_env']}")
            print(f"  • Secret: {config['secret_env']}")
            if 'passphrase_env' in config:
                print(f"  • Passphrase: {config['passphrase_env']}")
                
        print(f"\n{Fore.GREEN}💡 配置方法:{Style.RESET_ALL}")
        print("1. 创建 .env 文件")
        print("2. 添加实盘交易所API配置:")
        print("   BINANCE_API_KEY=your_real_binance_api_key")
        print("   BINANCE_SECRET_KEY=your_real_binance_secret_key")
        print("3. 重新启动系统")
        
        print(f"\n{Fore.RED}⚠️ 重要安全提示:{Style.RESET_ALL}")
        print("• 确保API密钥具有交易权限")
        print("• 这些是实盘API，将操作真实资金")
        print("• 建议设置合理的API权限限制")
        print("• 妥善保管API密钥，避免泄露")
        print("• 建议先小资金测试系统稳定性")

        # 询问是否现在配置
        print(f"\n{Fore.CYAN}🤔 是否现在配置交易所API？{Style.RESET_ALL}")
        configure_now = input("输入 'y' 现在配置，或按回车跳过: ").lower().strip()
        
        if configure_now == 'y':
            self.interactive_exchange_setup()
        
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
        
        # 询问用户是否要现在配置
        print(f"\n{Fore.CYAN}🤔 是否现在配置API密钥？{Style.RESET_ALL}")
        configure_now = input("输入 'y' 现在配置，或按回车跳过: ").lower().strip()
        
        if configure_now == 'y':
            self.interactive_api_setup(missing_apis)
    
    def interactive_api_setup(self, missing_apis: List[str]):
        """交互式API设置"""
        import os
        
        env_content = []
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                env_content = f.readlines()
        
        print(f"\n{Fore.GREEN}🔧 交互式API配置{Style.RESET_ALL}")
        
        for api_key in missing_apis:
            if api_key.startswith('OPENAI'):
                print(f"\n{Fore.CYAN}配置 {api_key}:{Style.RESET_ALL}")
                print("• 用于AI分析和决策增强")
                print("• 获取地址: https://platform.openai.com/api-keys")
                
                api_value = input(f"请输入 {api_key} (或按回车跳过): ").strip()
                
                if api_value:
                    # 添加到环境变量内容
                    env_line = f"{api_key}={api_value}\n"
                    env_content.append(env_line)
                    print(f"✅ {api_key} 已添加到配置")
        
        # 写入.env文件
        if env_content:
            with open('.env', 'w', encoding='utf-8') as f:
                f.writelines(env_content)
            print(f"\n{Fore.GREEN}✅ API配置已保存到 .env 文件{Style.RESET_ALL}")
            print("重新启动系统后配置将生效")
        else:
            print(f"\n{Fore.YELLOW}⚠️ 未配置任何API密钥{Style.RESET_ALL}")

    def interactive_exchange_setup(self):
        """交互式交易所API设置"""
        import os
        
        env_content = []
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                env_content = f.readlines()
        
        print(f"\n{Fore.GREEN}🏦 交互式交易所API配置{Style.RESET_ALL}")
        print("⚠️ 系统需要至少配置一个交易所API才能启动！")
        
        configured_count = 0
        
        for exchange_id, config in self.supported_exchanges.items():
            print(f"\n{Fore.CYAN}配置 {config['name']}:{Style.RESET_ALL}")
            print(f"• 全球知名交易所，支持现货和合约交易")
            print(f"• API获取地址: https://{exchange_id}.com")
            
            # 配置API Key
            api_key = input(f"请输入 {config['api_key_env']} (或按回车跳过): ").strip()
            if api_key:
                secret_key = input(f"请输入 {config['secret_env']}: ").strip()
                if secret_key:
                    env_content.append(f"{config['api_key_env']}={api_key}\n")
                    env_content.append(f"{config['secret_env']}={secret_key}\n")
                    
                    # 如果需要passphrase
                    if 'passphrase_env' in config:
                        passphrase = input(f"请输入 {config['passphrase_env']}: ").strip()
                        if passphrase:
                            env_content.append(f"{config['passphrase_env']}={passphrase}\n")
                    
                    configured_count += 1
                    print(f"✅ {config['name']} 配置完成")
        
        # 检查是否至少配置了一个交易所
        if configured_count == 0:
            print(f"\n{Fore.RED}❌ 错误：必须至少配置一个交易所API！{Style.RESET_ALL}")
            print("系统无法在没有交易所API的情况下启动。")
            print("请重新运行系统并配置至少一个交易所API。")
            sys.exit(1)
        
        # 写入.env文件
        if env_content:
            with open('.env', 'w', encoding='utf-8') as f:
                f.writelines(env_content)
            print(f"\n{Fore.GREEN}✅ 交易所API配置已保存到 .env 文件{Style.RESET_ALL}")
            print(f"已配置 {configured_count} 个交易所")
            print("重新启动系统后配置将生效")
        else:
            print(f"\n{Fore.YELLOW}⚠️ 未配置任何交易所API{Style.RESET_ALL}")
        
    def configure_exchanges(self):
        """配置交易所连接"""
        if not self.exchange_configs:
            logger.error("❌ 未配置交易所API，无法启动实盘交易系统")
            return False
            
        logger.info("🏦 配置实盘交易所连接...")
        
        try:
            from src.exchanges.multi_exchange_manager import multi_exchange_manager, ExchangeConfig
            
            for exchange_id, config in self.exchange_configs.items():
                try:
                    exchange_config = ExchangeConfig(
                        name=exchange_id,
                        api_key=config['api_key'],
                        secret=config['secret'],
                        passphrase=config.get('passphrase')
                    )
                    
                    success = multi_exchange_manager.add_exchange(exchange_config)
                    if success:
                        logger.success(f"✅ {config['name']} 实盘连接成功")
                    else:
                        logger.error(f"❌ {config['name']} 实盘连接失败")
                        
                except Exception as e:
                    logger.error(f"❌ {config['name']} 配置失败: {e}")
                    
            active_exchanges = multi_exchange_manager.get_active_exchanges()
            if active_exchanges:
                logger.success(f"🎉 成功连接 {len(active_exchanges)} 个实盘交易所: {', '.join(active_exchanges)}")
                self.system_status['exchanges'] = True
            else:
                logger.error("❌ 未成功连接任何实盘交易所")
                self.system_status['exchanges'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ 实盘交易所配置失败: {e}")
            self.system_status['exchanges'] = False
            return False
            
        return True
        
    def start_core_system(self):
        """启动核心系统"""
        logger.info("🚀 启动实盘交易核心系统...")
        
        try:
            # 导入并启动主系统
            from main import QuantTradingSystem
            
            logger.info("🔧 初始化实盘量化交易系统...")
            self.trading_system = QuantTradingSystem()
            
            # 在后台线程中启动系统
            system_thread = threading.Thread(
                target=self.trading_system.start_system,
                daemon=True
            )
            system_thread.start()
            
            logger.success("✅ 实盘交易核心系统启动成功")
            self.system_status['core'] = True
            
        except Exception as e:
            logger.error(f"❌ 实盘交易核心系统启动失败: {e}")
            self.system_status['core'] = False
            return False
            
        return True
        
    def start_signal_generator(self):
        """启动信号生成器"""
        logger.info("📡 启动实盘交易信号生成器...")
        
        try:
            from src.strategies.production_signal_generator import production_signal_generator
            
            production_signal_generator.start_generation()
            
            logger.success("✅ 实盘交易信号生成器启动成功")
            self.system_status['signal_generator'] = True
            
        except Exception as e:
            logger.error(f"❌ 实盘交易信号生成器启动失败: {e}")
            self.system_status['signal_generator'] = False
            return False
            
        return True
        
    def start_web_interface(self):
        """启动Web界面"""
        logger.info("🌐 启动实盘交易监控界面...")
        
        try:
            # 在后台线程中启动Web服务器
            web_thread = threading.Thread(
                target=self._run_web_server,
                daemon=True
            )
            web_thread.start()
            
            # 等待Web服务器启动
            time.sleep(3)
            
            logger.success("✅ 实盘交易监控界面启动成功")
            logger.info("🌐 访问地址: http://localhost:8080")
            self.system_status['web'] = True
            
        except Exception as e:
            logger.error(f"❌ 实盘交易监控界面启动失败: {e}")
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
        """启动实盘交易循环"""
        logger.info("🔄 启动实盘交易循环...")
        
        try:
            # 启动交易循环线程
            trading_thread = threading.Thread(
                target=self._trading_loop,
                daemon=True
            )
            trading_thread.start()
            
            logger.success("✅ 实盘交易循环启动成功")
            self.system_status['trading'] = True
            
        except Exception as e:
            logger.error(f"❌ 实盘交易循环启动失败: {e}")
            self.system_status['trading'] = False
            return False
            
        return True
        
    def _trading_loop(self):
        """实盘交易循环"""
        from src.exchanges.multi_exchange_manager import multi_exchange_manager
        from src.strategies.production_signal_generator import production_signal_generator
        
        logger.info("🔄 实盘交易循环开始运行...")
        logger.warning("⚠️ 注意：系统现在将使用真实资金进行交易！")
        
        # 支持的交易对
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        while True:
            try:
                for symbol in symbols:
                    # 生成交易信号
                    signal = production_signal_generator.generate_signal(symbol)
                    
                    if signal:
                        logger.warning(f"🎯 实盘交易信号: {signal.symbol} {signal.action} (真实资金)")
                        
                        # 转换为交易信号
                        trading_signal = production_signal_generator.convert_to_trading_signal(signal)
                        
                        # 广播到所有实盘交易所
                        results = multi_exchange_manager.broadcast_signal(trading_signal)
                        
                        # 记录结果
                        success_count = sum(1 for r in results if r.status not in ['failed', 'timeout'])
                        logger.info(f"📊 实盘交易执行结果: {success_count}/{len(results)} 成功")
                
                # 等待下一轮
                time.sleep(60)  # 每分钟检查一次
                
            except KeyboardInterrupt:
                logger.info("🛑 实盘交易循环收到停止信号")
                break
            except Exception as e:
                logger.error(f"❌ 实盘交易循环错误: {e}")
                time.sleep(30)  # 错误后等待30秒
                
    def monitor_system_health(self):
        """监控系统健康状态"""
        logger.info("📊 启动实盘交易系统健康监控...")
        
        while True:
            try:
                # 检查各个模块状态
                current_time = datetime.now()
                uptime = current_time - self.start_time
                
                # 每5分钟输出一次状态
                if uptime.total_seconds() % 300 == 0:
                    logger.info(f"💓 实盘交易系统运行时间: {uptime}")
                    logger.info(f"📈 系统状态: {self.system_status}")
                    
                    # 检查交易所健康状态
                    if self.exchange_configs:
                        try:
                            from src.exchanges.multi_exchange_manager import multi_exchange_manager
                            health = multi_exchange_manager.health_check()
                            logger.info(f"🏦 实盘交易所状态: {health['overall_status']}")
                        except:
                            pass
                
                time.sleep(60)  # 每分钟检查一次
                
            except KeyboardInterrupt:
                logger.info("🛑 收到停止信号，正在安全关闭实盘交易系统...")
                break
            except Exception as e:
                logger.error(f"系统监控错误: {e}")
                time.sleep(60)
                
    def print_startup_summary(self):
        """打印启动总结"""
        print(f"\n{Fore.GREEN}🎉 实盘交易系统启动完成!{Style.RESET_ALL}")
        print(f"{Fore.RED}╔══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                          🔥 实盘交易系统状态总览                              ║")
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
            print(f"║ 🏦 实盘交易所: {', '.join(exchange_names)[:50]:50}                    ║")
        
        print(f"║ 🌐 监控界面: http://localhost:8080                                            ║")
        print(f"║ 📊 实时监控: 交易数据、AI状态、系统性能                                        ║")
        print(f"║ 🤖 AI模型: 6大AI融合决策系统                                                  ║")
        print(f"║ 🎯 目标收益: 周收益20%+                                                       ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}💡 使用提示:{Style.RESET_ALL}")
        print("• 访问 http://localhost:8080 查看实时监控面板")
        print("• 按 Ctrl+C 安全停止系统")
        print("• 查看日志了解系统运行状态")
        
        print(f"\n{Fore.RED}🚀 实盘交易模式:{Style.RESET_ALL}")
        print("• ⚠️ 系统正在使用真实资金进行交易")
        print("• 🎯 AI信号将同时发送到所有配置的交易所")
        print("• 💰 请确保账户有足够余额进行交易")
        print("• 🔒 建议定期检查交易结果和风险状况")
        
        print(f"\n{Fore.RED}⚠️ 风险提醒:{Style.RESET_ALL}")
        print("• 量化交易存在亏损风险，请谨慎操作")
        print("• 建议先用小资金测试系统稳定性")
        print("• 定期监控系统状态和交易结果")
        print("• 如有异常请立即停止系统")
        
    def run(self):
        """运行启动器"""
        try:
            # 打印启动横幅
            self.print_banner()
            
            # 用户确认
            print(f"\n{Fore.RED}⚠️ 重要确认 ⚠️{Style.RESET_ALL}")
            print("本系统将使用真实资金进行交易，存在亏损风险。")
            confirm = input("请输入 'YES' 确认继续启动实盘交易系统: ")
            
            if confirm != 'YES':
                print("❌ 用户取消启动")
                return False
            
            # 检查环境
            if not self.check_environment():
                return False
                
            # 检查交易所API配置
            if not self.check_exchange_apis():
                return False
            
            # 检查其他API配置
            self.check_api_configuration()
            
            # 配置交易所连接
            if not self.configure_exchanges():
                return False
                
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
            logger.info("🛑 用户中断，正在安全关闭实盘交易系统...")
        except Exception as e:
            logger.error(f"❌ 实盘交易系统启动失败: {e}")
            return False
        finally:
            logger.info("👋 实盘交易系统已关闭")
            
        return True

def main():
    """主函数"""
    launcher = ProductionSystemLauncher()
    return launcher.run()

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
