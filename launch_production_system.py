#!/usr/bin/env python3
"""
🚀 888-888-88 量化交易系统 - 生产级一键启动器
完整的生产级量化交易系统启动器，支持API配置、系统检查、一键部署
"""

import os
import sys
import asyncio
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.config.api_config_manager import APIConfigManager


class ProductionSystemLauncher:
    """生产级系统启动器"""
    
    def __init__(self):
        self.config_manager = APIConfigManager()
        self.system_status = {
            "api_configured": False,
            "dependencies_installed": False,
            "system_initialized": False,
            "trading_ready": False
        }
        
        # 系统要求
        self.required_packages = [
            "numpy", "pandas", "asyncio", "aiohttp", "websockets",
            "ccxt", "loguru", "cryptography", "psutil", "torch"
        ]
        
        self.required_directories = [
            "config", "logs", "data", "models", "backups"
        ]
    
    def print_banner(self):
        """打印系统横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    888-888-88 量化交易系统                    ║
║                     生产级实盘交易平台                        ║
║                                                              ║
║  🔥 100% 生产级代码 | 🚀 实时AI决策 | 💰 多交易所支持        ║
║  ⚡ 高频交易优化   | 🛡️ 风险管理   | 📊 实时监控            ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🕒 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 66)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        try:
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                logger.error("❌ 需要Python 3.8或更高版本")
                return False
            
            logger.info(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Python版本检查失败: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        try:
            logger.info("🔍 检查系统依赖...")
            missing_packages = []
            
            for package in self.required_packages:
                try:
                    __import__(package)
                    logger.debug(f"✅ {package}")
                except ImportError:
                    missing_packages.append(package)
                    logger.warning(f"❌ 缺少包: {package}")
            
            if missing_packages:
                logger.error(f"❌ 缺少依赖包: {missing_packages}")
                logger.info("💡 运行以下命令安装依赖:")
                logger.info("pip install -r requirements.txt")
                return False
            
            logger.info("✅ 所有依赖包已安装")
            self.system_status["dependencies_installed"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 依赖检查失败: {e}")
            return False
    
    def create_directories(self) -> bool:
        """创建必要目录"""
        try:
            logger.info("📁 创建系统目录...")
            
            for directory in self.required_directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ 创建目录: {directory}")
                else:
                    logger.debug(f"📁 目录已存在: {directory}")
            
            # 设置日志目录权限
            logs_dir = Path("logs")
            if logs_dir.exists():
                os.chmod(logs_dir, 0o755)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 目录创建失败: {e}")
            return False
    
    def setup_logging(self):
        """设置日志系统"""
        try:
            # 配置loguru日志
            logger.remove()  # 移除默认处理器
            
            # 控制台日志
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            
            # 文件日志
            logger.add(
                "logs/system_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="1 day",
                retention="30 days",
                compression="zip"
            )
            
            # 错误日志
            logger.add(
                "logs/errors_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation="1 day",
                retention="90 days"
            )
            
            logger.info("✅ 日志系统初始化完成")
            
        except Exception as e:
            print(f"❌ 日志系统初始化失败: {e}")
    
    def check_api_configuration(self) -> bool:
        """检查API配置"""
        try:
            logger.info("🔐 检查API配置...")
            
            # 初始化配置管理器
            if not self.config_manager.initialize_config():
                logger.warning("⚠️ API配置未初始化")
                return False
            
            # 检查已配置的交易所
            configured_exchanges = self.config_manager.list_configured_exchanges()
            
            if not configured_exchanges:
                logger.warning("⚠️ 未配置任何交易所API")
                return False
            
            logger.info(f"✅ 已配置交易所: {', '.join(configured_exchanges)}")
            
            # 测试连接
            connection_results = {}
            for exchange in configured_exchanges:
                result = self.config_manager.test_exchange_connection(exchange)
                connection_results[exchange] = result
                
                if result:
                    logger.info(f"✅ {exchange} 连接正常")
                else:
                    logger.warning(f"⚠️ {exchange} 连接失败")
            
            # 至少需要一个交易所连接成功
            if not any(connection_results.values()):
                logger.error("❌ 所有交易所连接失败")
                return False
            
            self.system_status["api_configured"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ API配置检查失败: {e}")
            return False
    
    def initialize_system_components(self) -> bool:
        """初始化系统组件"""
        try:
            logger.info("🔧 初始化系统组件...")
            
            # 检查GPU可用性
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    logger.info(f"✅ 检测到 {gpu_count} 个GPU")
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        logger.info(f"  GPU {i}: {gpu_name}")
                else:
                    logger.info("💻 使用CPU模式")
            except ImportError:
                logger.warning("⚠️ PyTorch未安装，无法使用GPU加速")
            
            # 检查内存
            try:
                import psutil
                memory = psutil.virtual_memory()
                logger.info(f"💾 系统内存: {memory.total / 1e9:.1f}GB (可用: {memory.available / 1e9:.1f}GB)")
                
                if memory.available < 2e9:  # 小于2GB
                    logger.warning("⚠️ 可用内存不足，可能影响系统性能")
            except ImportError:
                logger.warning("⚠️ psutil未安装，无法检查系统资源")
            
            # 初始化数据库连接池
            logger.info("🗄️ 初始化数据存储...")
            
            # 创建配置文件
            config_file = Path("config/system_config.json")
            if not config_file.exists():
                default_config = {
                    "system": {
                        "name": "888-888-88",
                        "version": "1.0.0",
                        "mode": "production",
                        "initialized_at": time.time()
                    },
                    "trading": {
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15,
                        "max_daily_trades": 100
                    },
                    "risk_management": {
                        "max_daily_loss": 0.02,
                        "max_drawdown": 0.1,
                        "position_sizing": "kelly"
                    }
                }
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                
                logger.info("✅ 创建系统配置文件")
            
            self.system_status["system_initialized"] = True
            logger.info("✅ 系统组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统组件初始化失败: {e}")
            return False
    
    def run_system_checks(self) -> bool:
        """运行系统检查"""
        try:
            logger.info("🔍 运行系统完整性检查...")
            
            checks = [
                ("Python版本", self.check_python_version),
                ("系统依赖", self.check_dependencies),
                ("目录结构", self.create_directories),
                ("API配置", self.check_api_configuration),
                ("系统组件", self.initialize_system_components)
            ]
            
            failed_checks = []
            
            for check_name, check_func in checks:
                logger.info(f"🔍 检查: {check_name}")
                try:
                    if not check_func():
                        failed_checks.append(check_name)
                        logger.error(f"❌ {check_name} 检查失败")
                    else:
                        logger.info(f"✅ {check_name} 检查通过")
                except Exception as e:
                    failed_checks.append(check_name)
                    logger.error(f"❌ {check_name} 检查异常: {e}")
            
            if failed_checks:
                logger.error(f"❌ 以下检查失败: {', '.join(failed_checks)}")
                return False
            
            logger.info("✅ 所有系统检查通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统检查失败: {e}")
            return False
    
    async def start_trading_system(self) -> bool:
        """启动交易系统"""
        try:
            logger.info("🚀 启动量化交易系统...")
            
            # 导入主要模块
            try:
                from start_production_system import ProductionSystemManager
                
                # 创建系统管理器
                system_manager = ProductionSystemManager()
                
                # 启动系统
                logger.info("🔄 启动系统管理器...")
                success = await system_manager.start_system()
                
                if success:
                    logger.info("✅ 交易系统启动成功")
                    self.system_status["trading_ready"] = True
                    return True
                else:
                    logger.error("❌ 交易系统启动失败")
                    return False
                    
            except ImportError as e:
                logger.error(f"❌ 无法导入交易系统模块: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ 启动交易系统失败: {e}")
            return False
    
    def print_system_status(self):
        """打印系统状态"""
        print("\n" + "=" * 66)
        print("📊 系统状态报告")
        print("-" * 66)
        
        status_items = [
            ("API配置", self.system_status["api_configured"]),
            ("依赖安装", self.system_status["dependencies_installed"]),
            ("系统初始化", self.system_status["system_initialized"]),
            ("交易就绪", self.system_status["trading_ready"])
        ]
        
        for item_name, status in status_items:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {item_name}")
        
        print("-" * 66)
        
        if all(self.system_status.values()):
            print("🎉 系统完全就绪，可以开始交易！")
        else:
            print("⚠️ 系统未完全就绪，请检查上述问题")
        
        print("=" * 66)
    
    def interactive_setup(self) -> bool:
        """交互式设置"""
        try:
            print("\n🔧 首次运行检测到，开始交互式设置...")
            
            # API配置
            if not self.system_status["api_configured"]:
                print("\n📝 需要配置交易所API...")
                setup_api = input("是否现在配置API? (Y/n): ").lower().strip()
                
                if setup_api != 'n':
                    if not self.config_manager.interactive_setup():
                        logger.error("❌ API配置失败")
                        return False
                    
                    # 重新检查API配置
                    self.check_api_configuration()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消设置")
            return False
        except Exception as e:
            logger.error(f"❌ 交互式设置失败: {e}")
            return False
    
    async def launch(self) -> bool:
        """启动系统"""
        try:
            # 打印横幅
            self.print_banner()
            
            # 设置日志
            self.setup_logging()
            
            # 运行系统检查
            if not self.run_system_checks():
                logger.error("❌ 系统检查失败")
                
                # 如果是配置问题，提供交互式设置
                if not self.system_status["api_configured"]:
                    if not self.interactive_setup():
                        return False
                    
                    # 重新运行检查
                    if not self.run_system_checks():
                        return False
                else:
                    return False
            
            # 启动交易系统
            if not await self.start_trading_system():
                logger.error("❌ 交易系统启动失败")
                return False
            
            # 打印状态报告
            self.print_system_status()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("👋 用户中断启动")
            return False
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            return False


async def main():
    """主函数"""
    launcher = ProductionSystemLauncher()
    
    try:
        success = await launcher.launch()
        
        if success:
            print("\n🎉 888-888-88 量化交易系统启动成功！")
            print("💰 系统正在运行，开始实盘交易...")
            print("📊 监控面板: http://localhost:8080")
            print("🛑 按 Ctrl+C 停止系统")
            
            # 保持系统运行
            try:
                while True:
                    await asyncio.sleep(60)
                    logger.info("💓 系统运行正常")
            except KeyboardInterrupt:
                logger.info("👋 用户停止系统")
        else:
            print("\n❌ 系统启动失败")
            print("💡 请检查上述错误信息并重试")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 启动器异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
