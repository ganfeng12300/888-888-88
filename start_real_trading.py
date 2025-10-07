#!/usr/bin/env python3
"""
🚀 888-888-88 真实实盘交易启动器
Real Trading System Launcher
"""

import os
import sys
import asyncio
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

class RealTradingLauncher:
    """真实实盘交易启动器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.processes = []
        self.system_status = {}
        
        # 配置日志
        self._setup_logging()
        
        logger.info("🚀 888-888-88 真实实盘交易启动器初始化")
    
    def _setup_logging(self):
        """配置日志"""
        try:
            # 创建日志目录
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # 配置loguru
            logger.remove()
            
            # 控制台输出
            logger.add(
                sys.stdout,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                level="INFO"
            )
            
            # 文件输出
            logger.add(
                log_dir / "real_trading_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",
                rotation="1 day"
            )
            
        except Exception as e:
            print(f"❌ 配置日志失败: {e}")
    
    async def launch_real_trading(self):
        """启动真实实盘交易"""
        try:
            logger.info("🎯 开始启动888-888-88真实实盘交易系统")
            
            # 1. 系统预检查
            await self.pre_launch_checks()
            
            # 2. 加载配置
            await self.load_configurations()
            
            # 3. 初始化核心组件
            await self.initialize_core_components()
            
            # 4. 启动Web服务器
            await self.start_web_server()
            
            # 5. 启动AI系统
            await self.start_ai_systems()
            
            # 6. 启动交易引擎
            await self.start_trading_engine()
            
            # 7. 系统健康检查
            await self.perform_system_health_check()
            
            # 8. 生成启动报告
            await self.generate_launch_report()
            
            # 9. 实时监控循环
            await self.monitoring_loop()
            
        except KeyboardInterrupt:
            logger.info("👋 用户中断，正在安全关闭系统...")
            await self.safe_shutdown()
        except Exception as e:
            logger.error(f"❌ 实盘交易启动失败: {e}")
            await self.emergency_shutdown()
            sys.exit(1)
    
    async def pre_launch_checks(self):
        """启动前检查"""
        try:
            logger.info("🔍 执行启动前系统检查...")
            
            # 检查Python版本
            python_version = sys.version_info
            if python_version.major < 3 or python_version.minor < 8:
                raise RuntimeError(f"需要Python 3.8+，当前版本: {python_version.major}.{python_version.minor}")
            
            logger.info(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # 检查必要目录
            required_dirs = ['src', 'config', 'logs', 'models', 'data']
            for dir_name in required_dirs:
                Path(dir_name).mkdir(exist_ok=True)
                logger.info(f"✅ 目录检查: {dir_name}")
            
            # 检查核心文件
            core_files = [
                'src/config/api_config.py',
                'src/web/enhanced_app.py',
                'src/ai/ai_model_manager.py',
                'src/core/error_handling_system.py'
            ]
            
            missing_files = []
            for file_path in core_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
                else:
                    logger.info(f"✅ 核心文件: {file_path}")
            
            if missing_files:
                raise RuntimeError(f"缺少核心文件: {missing_files}")
            
            # 检查依赖包
            required_packages = ['fastapi', 'uvicorn', 'ccxt', 'numpy', 'pandas', 'loguru']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                    logger.info(f"✅ 依赖包: {package}")
                except ImportError:
                    missing_packages.append(package)
                    logger.error(f"❌ 缺少依赖: {package}")
            
            if missing_packages:
                raise RuntimeError(f"缺少必要依赖包: {missing_packages}")
            
            logger.info("✅ 启动前检查完成")
            
        except Exception as e:
            logger.error(f"❌ 启动前检查失败: {e}")
            raise
    
    async def load_configurations(self):
        """加载配置"""
        try:
            logger.info("⚙️ 加载系统配置...")
            
            # 加载交易所配置
            exchanges_file = Path("config/exchanges.json")
            if exchanges_file.exists():
                with open(exchanges_file, 'r', encoding='utf-8') as f:
                    self.exchanges_config = json.load(f)
                logger.info(f"✅ 交易所配置: {len(self.exchanges_config)} 个交易所")
            else:
                logger.warning("⚠️ 交易所配置文件不存在，使用默认配置")
                self.exchanges_config = {}
            
            # 加载交易配置
            trading_file = Path("config/trading.json")
            if trading_file.exists():
                with open(trading_file, 'r', encoding='utf-8') as f:
                    self.trading_config = json.load(f)
                logger.info(f"✅ 交易配置: {len(self.trading_config.get('allowed_symbols', []))} 个交易对")
            else:
                logger.warning("⚠️ 交易配置文件不存在，使用默认配置")
                self.trading_config = {}
            
            # 加载AI配置
            ai_file = Path("config/ai.json")
            if ai_file.exists():
                with open(ai_file, 'r', encoding='utf-8') as f:
                    self.ai_config = json.load(f)
                logger.info("✅ AI配置加载完成")
            else:
                logger.warning("⚠️ AI配置文件不存在，使用默认配置")
                self.ai_config = {}
            
            # 加载监控配置
            monitoring_file = Path("config/monitoring.json")
            if monitoring_file.exists():
                with open(monitoring_file, 'r', encoding='utf-8') as f:
                    self.monitoring_config = json.load(f)
                logger.info("✅ 监控配置加载完成")
            else:
                logger.warning("⚠️ 监控配置文件不存在，使用默认配置")
                self.monitoring_config = {}
            
            logger.info("✅ 配置加载完成")
            
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise
    
    async def initialize_core_components(self):
        """初始化核心组件"""
        try:
            logger.info("🔧 初始化核心系统组件...")
            
            # 初始化API配置管理器
            try:
                from src.config.api_config import api_config_manager
                await api_config_manager.initialize()
                logger.info("✅ API配置管理器初始化完成")
                self.system_status['api_config'] = 'initialized'
            except Exception as e:
                logger.error(f"❌ API配置管理器初始化失败: {e}")
                self.system_status['api_config'] = 'failed'
            
            # 初始化错误处理系统
            try:
                from src.core.error_handling_system import error_handler
                logger.info("✅ 错误处理系统初始化完成")
                self.system_status['error_handler'] = 'initialized'
            except Exception as e:
                logger.error(f"❌ 错误处理系统初始化失败: {e}")
                self.system_status['error_handler'] = 'failed'
            
            # 初始化系统监控
            try:
                from src.monitoring.system_monitor import system_monitor
                await system_monitor.start_monitoring()
                logger.info("✅ 系统监控初始化完成")
                self.system_status['system_monitor'] = 'running'
            except Exception as e:
                logger.error(f"❌ 系统监控初始化失败: {e}")
                self.system_status['system_monitor'] = 'failed'
            
            logger.info("✅ 核心组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 核心组件初始化失败: {e}")
            raise
    
    async def start_web_server(self):
        """启动Web服务器"""
        try:
            logger.info("🌐 启动Web管理服务器...")
            
            # 启动Web服务器进程
            web_process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "uvicorn",
                "src.web.enhanced_app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload", "False",
                "--log-level", "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes.append(web_process)
            
            # 等待Web服务器启动
            await asyncio.sleep(5)
            
            # 检查Web服务器状态
            if web_process.returncode is None:
                logger.info("✅ Web服务器启动成功")
                logger.info("🌐 Web界面地址:")
                logger.info("   - 主界面: http://localhost:8000")
                logger.info("   - API文档: http://localhost:8000/api/docs")
                logger.info("   - 健康检查: http://localhost:8000/health")
                logger.info("   - 系统状态: http://localhost:8000/api/system/status")
                self.system_status['web_server'] = 'running'
            else:
                raise RuntimeError("Web服务器启动失败")
            
        except Exception as e:
            logger.error(f"❌ Web服务器启动失败: {e}")
            self.system_status['web_server'] = 'failed'
            raise
    
    async def start_ai_systems(self):
        """启动AI系统"""
        try:
            logger.info("🤖 启动AI系统组件...")
            
            # 初始化AI模型管理器
            try:
                from src.ai.ai_model_manager import ai_model_manager
                await ai_model_manager.initialize()
                logger.info("✅ AI模型管理器启动完成")
                self.system_status['ai_model_manager'] = 'running'
            except Exception as e:
                logger.error(f"❌ AI模型管理器启动失败: {e}")
                self.system_status['ai_model_manager'] = 'failed'
            
            # 初始化AI性能监控器
            try:
                from src.ai.ai_performance_monitor import ai_performance_monitor
                logger.info("✅ AI性能监控器启动完成")
                self.system_status['ai_performance_monitor'] = 'running'
            except Exception as e:
                logger.error(f"❌ AI性能监控器启动失败: {e}")
                self.system_status['ai_performance_monitor'] = 'failed'
            
            # 初始化AI融合引擎
            try:
                from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine
                await enhanced_ai_fusion_engine.initialize()
                logger.info("✅ AI融合引擎启动完成")
                self.system_status['ai_fusion_engine'] = 'running'
            except Exception as e:
                logger.error(f"❌ AI融合引擎启动失败: {e}")
                self.system_status['ai_fusion_engine'] = 'failed'
            
            logger.info("✅ AI系统启动完成")
            
        except Exception as e:
            logger.error(f"❌ AI系统启动失败: {e}")
            raise
    
    async def start_trading_engine(self):
        """启动交易引擎"""
        try:
            logger.info("💰 启动交易引擎...")
            
            # 检查交易所配置
            active_exchanges = []
            for exchange_name, config in self.exchanges_config.items():
                if config.get('api_key') and config.get('secret'):
                    active_exchanges.append(exchange_name)
                    logger.info(f"✅ 交易所配置: {exchange_name}")
                else:
                    logger.warning(f"⚠️ 交易所缺少API配置: {exchange_name}")
            
            if not active_exchanges:
                logger.warning("⚠️ 没有配置有效的交易所API，交易引擎将以模拟模式运行")
                self.system_status['trading_engine'] = 'simulation'
            else:
                logger.info(f"✅ 交易引擎启动完成，活跃交易所: {active_exchanges}")
                self.system_status['trading_engine'] = 'running'
            
            # 显示交易配置
            if self.trading_config:
                logger.info("📊 交易配置:")
                logger.info(f"   - 最大仓位: {self.trading_config.get('max_position_size', 0.1)}")
                logger.info(f"   - 每日最大交易: {self.trading_config.get('max_daily_trades', 50)}")
                logger.info(f"   - 风险比例: {self.trading_config.get('risk_per_trade', 0.02)}")
                logger.info(f"   - 止损比例: {self.trading_config.get('stop_loss_pct', 0.02)}")
                logger.info(f"   - 止盈比例: {self.trading_config.get('take_profit_pct', 0.04)}")
                logger.info(f"   - 交易对数量: {len(self.trading_config.get('allowed_symbols', []))}")
            
        except Exception as e:
            logger.error(f"❌ 交易引擎启动失败: {e}")
            self.system_status['trading_engine'] = 'failed'
            raise
    
    async def perform_system_health_check(self):
        """执行系统健康检查"""
        try:
            logger.info("🏥 执行系统健康检查...")
            
            # 统计组件状态
            total_components = len(self.system_status)
            running_components = sum(1 for status in self.system_status.values() 
                                   if status in ['running', 'initialized'])
            failed_components = sum(1 for status in self.system_status.values() 
                                  if status == 'failed')
            
            health_score = (running_components / total_components * 100) if total_components > 0 else 0
            
            logger.info("📊 系统健康检查结果:")
            logger.info(f"   - 总组件数: {total_components}")
            logger.info(f"   - 运行正常: {running_components}")
            logger.info(f"   - 运行失败: {failed_components}")
            logger.info(f"   - 健康度: {health_score:.1f}%")
            
            # 详细状态
            for component, status in self.system_status.items():
                status_icon = "✅" if status in ['running', 'initialized'] else "❌" if status == 'failed' else "⚠️"
                logger.info(f"   {status_icon} {component}: {status}")
            
            if health_score >= 80:
                logger.info("✅ 系统健康状态良好，可以开始交易")
            elif health_score >= 60:
                logger.warning("⚠️ 系统健康状态一般，建议检查失败组件")
            else:
                logger.error("❌ 系统健康状态不佳，不建议进行交易")
            
            self.health_score = health_score
            
        except Exception as e:
            logger.error(f"❌ 系统健康检查失败: {e}")
    
    async def generate_launch_report(self):
        """生成启动报告"""
        try:
            logger.info("📋 生成系统启动报告...")
            
            launch_duration = (datetime.now() - self.start_time).total_seconds()
            
            # 创建启动报告
            report = {
                "launch_info": {
                    "start_time": self.start_time.isoformat(),
                    "completion_time": datetime.now().isoformat(),
                    "duration_seconds": launch_duration,
                    "system_name": "888-888-88 量化交易系统",
                    "version": "2.0.0",
                    "mode": "实盘交易"
                },
                "system_status": self.system_status,
                "health_score": getattr(self, 'health_score', 0),
                "configurations": {
                    "exchanges": len(self.exchanges_config),
                    "trading_pairs": len(self.trading_config.get('allowed_symbols', [])),
                    "ai_models": self.ai_config.get('max_models_loaded', 0)
                },
                "access_urls": {
                    "web_interface": "http://localhost:8000",
                    "api_docs": "http://localhost:8000/api/docs",
                    "health_check": "http://localhost:8000/health",
                    "system_status": "http://localhost:8000/api/system/status",
                    "ai_models": "http://localhost:8000/api/ai/models"
                },
                "next_steps": [
                    "访问Web界面监控系统状态",
                    "检查AI模型运行状态",
                    "监控交易信号和执行",
                    "查看实时性能指标"
                ]
            }
            
            # 保存报告
            report_file = Path(f"real_trading_launch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 启动报告已保存: {report_file}")
            
            # 显示启动摘要
            logger.info("=" * 80)
            logger.info("🎉 888-888-88 真实实盘交易系统启动完成！")
            logger.info("=" * 80)
            logger.info(f"⏱️  启动耗时: {launch_duration:.1f} 秒")
            logger.info(f"🏥 系统健康度: {getattr(self, 'health_score', 0):.1f}%")
            logger.info(f"🔧 运行组件: {sum(1 for s in self.system_status.values() if s in ['running', 'initialized'])}/{len(self.system_status)}")
            logger.info("")
            logger.info("🌐 Web界面访问地址:")
            logger.info("   📊 主界面: http://localhost:8000")
            logger.info("   📚 API文档: http://localhost:8000/api/docs")
            logger.info("   🏥 健康检查: http://localhost:8000/health")
            logger.info("   📈 系统状态: http://localhost:8000/api/system/status")
            logger.info("   🤖 AI模型: http://localhost:8000/api/ai/models")
            logger.info("")
            logger.info("💰 交易配置:")
            logger.info(f"   📊 交易所: {len(self.exchanges_config)} 个")
            logger.info(f"   💱 交易对: {len(self.trading_config.get('allowed_symbols', []))} 个")
            logger.info(f"   🎯 最大仓位: {self.trading_config.get('max_position_size', 0.1)}")
            logger.info(f"   ⚠️ 风险比例: {self.trading_config.get('risk_per_trade', 0.02)}")
            logger.info("")
            logger.info("🚀 系统已就绪，开始实盘交易监控！")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ 生成启动报告失败: {e}")
    
    async def monitoring_loop(self):
        """实时监控循环"""
        try:
            logger.info("🔄 开始实时系统监控...")
            logger.info("💡 按 Ctrl+C 安全关闭系统")
            
            monitor_count = 0
            
            while True:
                monitor_count += 1
                
                # 检查进程状态
                for i, process in enumerate(self.processes):
                    if process.returncode is not None:
                        logger.warning(f"⚠️ 进程 {i} 异常退出，返回码: {process.returncode}")
                
                # 每5分钟显示一次状态
                if monitor_count % 60 == 0:  # 每60个5秒循环 = 5分钟
                    runtime = (datetime.now() - self.start_time).total_seconds()
                    logger.info(f"💓 系统运行正常 - 运行时间: {runtime:.0f}秒 ({runtime/3600:.1f}小时)")
                    
                    # 显示系统状态
                    running_components = sum(1 for s in self.system_status.values() 
                                           if s in ['running', 'initialized'])
                    logger.info(f"🔧 活跃组件: {running_components}/{len(self.system_status)}")
                
                await asyncio.sleep(5)  # 每5秒检查一次
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"❌ 监控循环错误: {e}")
            raise
    
    async def safe_shutdown(self):
        """安全关闭系统"""
        try:
            logger.info("🛑 开始安全关闭系统...")
            
            # 关闭所有进程
            for i, process in enumerate(self.processes):
                try:
                    logger.info(f"🔄 关闭进程 {i}...")
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                    logger.info(f"✅ 进程 {i} 已安全关闭")
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ 进程 {i} 关闭超时，强制终止")
                    process.kill()
                except Exception as e:
                    logger.error(f"❌ 关闭进程 {i} 失败: {e}")
            
            # 关闭AI组件
            try:
                from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine
                from src.ai.ai_model_manager import ai_model_manager
                
                await enhanced_ai_fusion_engine.shutdown()
                await ai_model_manager.shutdown()
                logger.info("✅ AI组件已安全关闭")
            except Exception as e:
                logger.error(f"❌ 关闭AI组件失败: {e}")
            
            runtime = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"🎯 系统已安全关闭，总运行时间: {runtime:.1f}秒 ({runtime/3600:.1f}小时)")
            
        except Exception as e:
            logger.error(f"❌ 安全关闭失败: {e}")
    
    async def emergency_shutdown(self):
        """紧急关闭系统"""
        try:
            logger.error("🚨 执行紧急关闭...")
            
            # 强制终止所有进程
            for i, process in enumerate(self.processes):
                try:
                    process.kill()
                    logger.warning(f"⚠️ 强制终止进程 {i}")
                except Exception as e:
                    logger.error(f"❌ 强制终止进程 {i} 失败: {e}")
            
            logger.error("🚨 紧急关闭完成")
            
        except Exception as e:
            logger.error(f"❌ 紧急关闭失败: {e}")

async def main():
    """主函数"""
    try:
        launcher = RealTradingLauncher()
        await launcher.launch_real_trading()
    except Exception as e:
        logger.error(f"❌ 实盘交易启动器运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # 设置事件循环策略（Windows兼容性）
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 运行主函数
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序安全退出")
    except Exception as e:
        print(f"❌ 程序异常退出: {e}")
        sys.exit(1)
