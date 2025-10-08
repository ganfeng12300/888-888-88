#!/usr/bin/env python3
"""
🚀 888-888-88 一键启动脚本
One-Click Production System Launcher
"""

import os
import sys
import asyncio
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

class OneClickLauncher:
    """一键启动器"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.processes = []
        self.components_status = {}
        
        # 配置日志
        self._setup_logging()
        
        logger.info("🚀 888-888-88 一键启动器初始化")
    
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
                log_dir / "startup_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",
                rotation="1 day"
            )
            
        except Exception as e:
            print(f"❌ 配置日志失败: {e}")
    
    async def run(self):
        """运行一键启动"""
        try:
            logger.info("🎯 开始一键启动888-888-88量化交易系统")
            
            # 1. 环境检查
            await self.check_environment()
            
            # 2. 依赖检查
            await self.check_dependencies()
            
            # 3. 配置初始化
            await self.initialize_configs()
            
            # 4. 启动核心系统
            await self.start_core_system()
            
            # 5. 启动Web界面
            await self.start_web_interface()
            
            # 6. 系统健康检查
            await self.perform_health_check()
            
            # 7. 生成启动报告
            await self.generate_startup_report()
            
            # 8. 保持运行
            await self.keep_running()
            
        except KeyboardInterrupt:
            logger.info("👋 用户中断，正在关闭系统...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"❌ 一键启动失败: {e}")
            await self.shutdown()
            sys.exit(1)
    
    async def check_environment(self):
        """检查环境"""
        try:
            logger.info("🔍 检查运行环境...")
            
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
            
            # 检查环境变量
            env_vars = [
                'DEFAULT_MASTER_PASSWORD',
                'JWT_SECRET_KEY'
            ]
            
            missing_vars = []
            for var in env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(f"⚠️ 缺少环境变量: {missing_vars}")
                logger.info("💡 系统将使用默认配置")
            
            logger.info("✅ 环境检查完成")
            
        except Exception as e:
            logger.error(f"❌ 环境检查失败: {e}")
            raise
    
    async def check_dependencies(self):
        """检查依赖"""
        try:
            logger.info("📦 检查依赖包...")
            
            # 核心依赖
            core_deps = [
                'fastapi',
                'uvicorn',
                'loguru',
                'ccxt',
                'numpy',
                'pandas'
            ]
            
            missing_deps = []
            for dep in core_deps:
                try:
                    __import__(dep)
                    logger.info(f"✅ {dep}")
                except ImportError:
                    missing_deps.append(dep)
                    logger.warning(f"⚠️ 缺少依赖: {dep}")
            
            # 可选依赖
            optional_deps = [
                'tensorflow',
                'torch',
                'xgboost',
                'lightgbm',
                'scikit-learn'
            ]
            
            available_optional = []
            for dep in optional_deps:
                try:
                    __import__(dep)
                    available_optional.append(dep)
                    logger.info(f"✅ {dep} (可选)")
                except ImportError:
                    logger.info(f"ℹ️ {dep} (可选，未安装)")
            
            if missing_deps:
                logger.error(f"❌ 缺少核心依赖: {missing_deps}")
                logger.info("💡 请运行: pip install -r requirements.txt")
                raise RuntimeError("缺少必要依赖")
            
            logger.info(f"✅ 依赖检查完成，可用AI库: {available_optional}")
            
        except Exception as e:
            logger.error(f"❌ 依赖检查失败: {e}")
            raise
    
    async def initialize_configs(self):
        """初始化配置"""
        try:
            logger.info("⚙️ 初始化系统配置...")
            
            # 导入配置管理器
            from src.config.api_config import api_config_manager
            
            # 初始化配置
            await api_config_manager.initialize()
            
            # 获取配置摘要
            config_summary = api_config_manager.get_config_summary()
            
            logger.info("✅ 配置初始化完成")
            logger.info(f"📊 配置摘要: {len(config_summary.get('exchanges', {}))} 个交易所")
            
            self.components_status['config_manager'] = 'initialized'
            
        except Exception as e:
            logger.error(f"❌ 配置初始化失败: {e}")
            raise
    
    async def start_core_system(self):
        """启动核心系统"""
        try:
            logger.info("🔧 启动核心系统组件...")
            
            # 导入核心组件
            from src.core.error_handling_system import error_handler
            from src.monitoring.system_monitor import system_monitor
            from src.ai.ai_model_manager import ai_model_manager
            from src.ai.ai_performance_monitor import ai_performance_monitor
            from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine
            
            # 启动系统监控
            await system_monitor.start_monitoring()
            logger.info("✅ 系统监控已启动")
            self.components_status['system_monitor'] = 'running'
            
            # 初始化AI模型管理器
            await ai_model_manager.initialize()
            logger.info("✅ AI模型管理器已初始化")
            self.components_status['ai_model_manager'] = 'running'
            
            # 初始化AI融合引擎
            await enhanced_ai_fusion_engine.initialize()
            logger.info("✅ AI融合引擎已初始化")
            self.components_status['ai_fusion_engine'] = 'running'
            
            logger.info("✅ 核心系统启动完成")
            
        except Exception as e:
            logger.error(f"❌ 核心系统启动失败: {e}")
            raise
    
    async def start_web_interface(self):
        """启动Web界面"""
        try:
            logger.info("🌐 启动Web管理界面...")
            
            # 启动Web服务器
            web_process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "uvicorn",
                "src.web.enhanced_app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload", "False",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes.append(web_process)
            
            # 等待Web服务器启动
            await asyncio.sleep(3)
            
            # 检查Web服务器状态
            if web_process.returncode is None:
                logger.info("✅ Web界面启动成功")
                logger.info("🌐 访问地址: http://localhost:8000")
                logger.info("📚 API文档: http://localhost:8000/api/docs")
                self.components_status['web_interface'] = 'running'
            else:
                raise RuntimeError("Web服务器启动失败")
            
        except Exception as e:
            logger.error(f"❌ Web界面启动失败: {e}")
            raise
    
    async def perform_health_check(self):
        """执行健康检查"""
        try:
            logger.info("🏥 执行系统健康检查...")
            
            # 检查各组件状态
            health_results = {}
            
            for component, status in self.components_status.items():
                if status == 'running' or status == 'initialized':
                    health_results[component] = 'healthy'
                else:
                    health_results[component] = 'unhealthy'
            
            # 检查Web接口
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://localhost:8000/health', timeout=5) as response:
                        if response.status == 200:
                            health_results['web_api'] = 'healthy'
                        else:
                            health_results['web_api'] = 'unhealthy'
            except:
                health_results['web_api'] = 'unhealthy'
            
            # 计算健康度
            healthy_count = sum(1 for status in health_results.values() if status == 'healthy')
            total_count = len(health_results)
            health_score = (healthy_count / total_count * 100) if total_count > 0 else 0
            
            logger.info(f"📊 系统健康度: {health_score:.1f}% ({healthy_count}/{total_count})")
            
            if health_score >= 80:
                logger.info("✅ 系统健康状态良好")
            elif health_score >= 60:
                logger.warning("⚠️ 系统健康状态一般")
            else:
                logger.error("❌ 系统健康状态不佳")
            
            self.health_results = health_results
            self.health_score = health_score
            
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
    
    async def generate_startup_report(self):
        """生成启动报告"""
        try:
            logger.info("📋 生成启动报告...")
            
            startup_duration = (datetime.now() - self.start_time).total_seconds()
            
            report = {
                "startup_info": {
                    "start_time": self.start_time.isoformat(),
                    "completion_time": datetime.now().isoformat(),
                    "duration_seconds": startup_duration,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "platform": sys.platform
                },
                "components_status": self.components_status,
                "health_check": {
                    "results": getattr(self, 'health_results', {}),
                    "score": getattr(self, 'health_score', 0)
                },
                "access_info": {
                    "web_interface": "http://localhost:8000",
                    "api_docs": "http://localhost:8000/api/docs",
                    "health_check": "http://localhost:8000/health"
                },
                "next_steps": [
                    "访问Web界面查看系统状态",
                    "配置交易所API密钥",
                    "检查AI模型状态",
                    "开始实盘交易"
                ]
            }
            
            # 保存报告
            report_file = Path("startup_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 启动报告已保存: {report_file}")
            
            # 显示启动摘要
            logger.info("=" * 60)
            logger.info("🎉 888-888-88 量化交易系统启动完成！")
            logger.info("=" * 60)
            logger.info(f"⏱️  启动耗时: {startup_duration:.1f} 秒")
            logger.info(f"🏥 系统健康度: {getattr(self, 'health_score', 0):.1f}%")
            logger.info(f"🔧 活跃组件: {len([s for s in self.components_status.values() if s == 'running'])}")
            logger.info("")
            logger.info("🌐 访问地址:")
            logger.info("   主界面: http://localhost:8000")
            logger.info("   API文档: http://localhost:8000/api/docs")
            logger.info("   健康检查: http://localhost:8000/health")
            logger.info("")
            logger.info("🚀 系统已就绪，可以开始使用！")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ 生成启动报告失败: {e}")
    
    async def keep_running(self):
        """保持运行"""
        try:
            logger.info("🔄 系统运行中，按 Ctrl+C 退出...")
            
            while True:
                # 检查进程状态
                for process in self.processes:
                    if process.returncode is not None:
                        logger.warning(f"⚠️ 进程异常退出: {process.pid}")
                
                # 定期状态更新
                await asyncio.sleep(60)
                logger.info(f"💓 系统运行正常 - 运行时间: {(datetime.now() - self.start_time).total_seconds():.0f}秒")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"❌ 系统运行错误: {e}")
            raise
    
    async def shutdown(self):
        """关闭系统"""
        try:
            logger.info("🛑 正在关闭系统...")
            
            # 终止所有进程
            for process in self.processes:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                    logger.info(f"✅ 进程已关闭: {process.pid}")
                except asyncio.TimeoutError:
                    process.kill()
                    logger.warning(f"⚠️ 强制终止进程: {process.pid}")
                except Exception as e:
                    logger.error(f"❌ 关闭进程失败: {e}")
            
            # 关闭AI组件
            try:
                from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine
                from src.ai.ai_model_manager import ai_model_manager
                
                await enhanced_ai_fusion_engine.shutdown()
                await ai_model_manager.shutdown()
                logger.info("✅ AI组件已关闭")
            except Exception as e:
                logger.error(f"❌ 关闭AI组件失败: {e}")
            
            runtime = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"🎯 系统已关闭，总运行时间: {runtime:.1f}秒")
            
        except Exception as e:
            logger.error(f"❌ 系统关闭失败: {e}")

async def main():
    """主函数"""
    try:
        launcher = OneClickLauncher()
        await launcher.run()
    except Exception as e:
        logger.error(f"❌ 启动器运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # 设置事件循环策略（Windows兼容性）
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 运行主函数
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"❌ 程序异常退出: {e}")
        sys.exit(1)
