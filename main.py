#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 主启动程序
史诗级AI驱动的量化交易平台
专为生产级实盘交易设计，生产级标准
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from loguru import logger
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入系统组件
from src.system.startup_manager import StartupManager
from src.system.ai_scheduler import AIScheduler
from src.trading.trading_engine import TradingEngine
from src.trading.order_manager import OrderManager
from src.risk.risk_manager import RiskManager

# 导入Web应用
from web.app import create_app

class FoxAITradingSystem:
    """🦊 猎狐AI量化交易系统主类"""
    
    def __init__(self):
        self.config = self._load_config()
        self.components = {}
        self.ai_models = {}
        self.web_app = None
        self.running = False
        
        # 设置日志
        self._setup_logging()
        
        logger.info("🦊 猎狐AI量化交易系统初始化...")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载系统配置"""
        config = {
            # 交易所配置
            'exchanges': {
                'binance': {
                    'enabled': os.getenv('BINANCE_API_KEY') is not None,
                    'api_key': os.getenv('BINANCE_API_KEY'),
                    'secret_key': os.getenv('BINANCE_SECRET_KEY'),
                    'sandbox': os.getenv('BINANCE_SANDBOX', 'true').lower() == 'true'
                }
            },
            
            # 系统配置
            'system': {
                'max_order_size': float(os.getenv('MAX_ORDER_SIZE', '10000')),
                'max_daily_orders': int(os.getenv('MAX_DAILY_ORDERS', '1000')),
                'max_single_position': float(os.getenv('MAX_SINGLE_POSITION', '0.3')),
                'max_total_position': float(os.getenv('MAX_TOTAL_POSITION', '0.8')),
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.03')),
                'cpu_cores': int(os.getenv('CPU_CORES', '20')),
                'gpu_memory_gb': float(os.getenv('GPU_MEMORY_GB', '12'))
            },
            
            # Web配置
            'web': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': os.getenv('DEBUG', 'false').lower() == 'true'
            }
        }
        
        return config
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 配置loguru
        logger.remove()  # 移除默认处理器
        
        # 控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # 文件输出
        logger.add(
            log_dir / "system.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="30 days"
        )
    
    async def initialize_components(self):
        """初始化系统组件"""
        try:
            logger.info("🔧 初始化系统组件...")
            
            # 1. 启动管理器
            self.components['startup_manager'] = StartupManager(self.config)
            
            # 2. AI调度中心
            self.components['ai_scheduler'] = AIScheduler(self.config)
            
            # 3. 交易执行组件
            self.components['trading_engine'] = TradingEngine(self.config)
            self.components['order_manager'] = OrderManager(
                self.components['trading_engine'], 
                self.config
            )
            
            # 4. 风险管理
            self.components['risk_manager'] = RiskManager(self.config)
            
            logger.success("✅ 系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 系统组件初始化失败: {e}")
            raise
    
    async def start_web_interface(self):
        """启动Web界面"""
        try:
            logger.info("🌐 启动Web界面...")
            
            # 创建Web应用
            self.web_app = create_app()
            
            # 启动Web服务器
            import uvicorn
            
            config = uvicorn.Config(
                self.web_app,
                host=self.config['web']['host'],
                port=self.config['web']['port'],
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            # 在后台启动服务器
            asyncio.create_task(server.serve())
            
            logger.success(f"✅ Web界面已启动: http://{self.config['web']['host']}:{self.config['web']['port']}")
            
        except Exception as e:
            logger.error(f"❌ Web界面启动失败: {e}")
            raise
    
    async def start_system(self):
        """启动完整系统"""
        try:
            logger.info("🚀 开始启动猎狐AI量化交易系统...")
            
            # 1. 执行60秒启动序列
            startup_manager = self.components['startup_manager']
            
            # 添加进度回调
            def progress_callback(progress_info):
                logger.info(f"📊 启动进度: {progress_info['total_progress']:.1f}%")
            
            startup_manager.add_progress_callback(progress_callback)
            
            # 执行启动序列
            startup_success = await startup_manager.start_system()
            
            if not startup_success:
                raise Exception("系统启动序列失败")
            
            # 2. 启动核心组件
            await self.components['ai_scheduler'].start()
            await self.components['trading_engine'].start()
            await self.components['order_manager'].start()
            
            # 3. 启动Web界面
            await self.start_web_interface()
            
            self.running = True
            logger.success("🎉 猎狐AI量化交易系统启动完成！")
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            raise
    
    async def stop_system(self):
        """停止系统"""
        try:
            logger.info("🛑 正在停止猎狐AI量化交易系统...")
            
            self.running = False
            
            # 停止组件
            if 'order_manager' in self.components:
                await self.components['order_manager'].stop()
            
            if 'trading_engine' in self.components:
                await self.components['trading_engine'].stop()
            
            if 'ai_scheduler' in self.components:
                await self.components['ai_scheduler'].stop()
            
            logger.success("✅ 系统已安全停止")
            
        except Exception as e:
            logger.error(f"❌ 系统停止异常: {e}")
    
    async def run(self):
        """运行系统主循环"""
        try:
            # 初始化组件
            await self.initialize_components()
            
            # 启动系统
            await self.start_system()
            
            # 主循环
            while self.running:
                try:
                    # 等待1秒
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("📝 收到停止信号，正在安全关闭系统...")
                    break
                except Exception as e:
                    logger.error(f"❌ 主循环异常: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ 系统运行异常: {e}")
        finally:
            await self.stop_system()

def setup_signal_handlers(system: FoxAITradingSystem):
    """设置信号处理器"""
    def signal_handler(signum, frame):
        logger.info(f"📝 收到信号 {signum}，正在安全关闭系统...")
        system.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """主函数"""
    try:
        # 显示启动横幅
        print("""
🦊 =============================================== 🦊
   猎狐AI量化交易系统 - Fox AI Trading System
   史诗级AI驱动的量化交易平台
   
   🧠 8大AI智能体 | ⚡ <50ms超低延迟
   🛡️ 五层风控矩阵 | 🌟 豪华黑金界面
🦊 =============================================== 🦊
        """)
        
        # 创建系统实例
        system = FoxAITradingSystem()
        
        # 设置信号处理器
        setup_signal_handlers(system)
        
        # 运行系统
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("📝 用户中断，系统退出")
    except Exception as e:
        logger.error(f"❌ 系统异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
