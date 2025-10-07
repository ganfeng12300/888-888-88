#!/usr/bin/env python3
"""
🚀 888-888-88 生产级系统启动脚本
Production-Grade System Startup Script
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

class ProductionSystemManager:
    """生产级系统管理器"""
    
    def __init__(self):
        self.components = {}
        self.is_running = False
        self.startup_time = None
        
        # 配置日志
        self._setup_logging()
        logger.info("🚀 生产级系统管理器初始化")
    
    def _setup_logging(self):
        """配置日志系统"""
        try:
            # 创建日志目录
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # 配置loguru
            logger.remove()  # 移除默认处理器
            
            # 控制台输出
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            
            # 文件输出
            logger.add(
                log_dir / "system_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
                level="DEBUG",
                rotation="1 day",
                retention="30 days"
            )
            
        except Exception as e:
            print(f"❌ 配置日志系统失败: {e}")
    
    async def initialize_components(self):
        """初始化所有组件"""
        try:
            logger.info("🔧 开始初始化系统组件...")
            
            # 导入核心组件
            from src.core.error_handling_system import error_handler
            from src.monitoring.system_monitor import system_monitor
            from src.ai.ai_model_manager import ai_model_manager
            from src.ai.ai_performance_monitor import ai_performance_monitor
            from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine
            
            # 1. 初始化错误处理系统
            self.components['error_handler'] = error_handler
            logger.info("✅ 错误处理系统已就绪")
            
            # 2. 初始化系统监控
            self.components['system_monitor'] = system_monitor
            await system_monitor.start_monitoring()
            logger.info("✅ 系统监控已启动")
            
            # 3. 初始化AI模型管理器
            self.components['ai_model_manager'] = ai_model_manager
            await ai_model_manager.initialize()
            logger.info("✅ AI模型管理器已初始化")
            
            # 4. 初始化AI性能监控器
            self.components['ai_performance_monitor'] = ai_performance_monitor
            logger.info("✅ AI性能监控器已就绪")
            
            # 5. 初始化AI融合引擎
            self.components['ai_fusion_engine'] = enhanced_ai_fusion_engine
            await enhanced_ai_fusion_engine.initialize()
            logger.info("✅ AI融合引擎已初始化")
            
            logger.info("🎉 所有系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    async def start_system(self):
        """启动系统"""
        try:
            self.startup_time = datetime.now()
            logger.info("🚀 启动888-888-88量化交易系统...")
            
            # 检查环境
            await self._check_environment()
            
            # 初始化组件
            await self.initialize_components()
            
            # 启动主循环
            self.is_running = True
            
            # 注册信号处理器
            self._register_signal_handlers()
            
            logger.info("🎉 系统启动完成！")
            logger.info(f"📊 启动时间: {self.startup_time}")
            logger.info(f"🔧 组件数量: {len(self.components)}")
            
            # 启动主循环
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            await self.shutdown_system()
            raise
    
    async def _check_environment(self):
        """检查环境配置"""
        try:
            logger.info("🔍 检查环境配置...")
            
            # 检查目录结构
            required_dirs = ['logs', 'models', 'data', 'config']
            for dir_name in required_dirs:
                Path(dir_name).mkdir(exist_ok=True)
            
            logger.info("✅ 环境检查完成")
            
        except Exception as e:
            logger.error(f"❌ 环境检查失败: {e}")
            raise
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        try:
            def signal_handler(signum, frame):
                logger.info(f"📡 收到信号 {signum}，开始优雅关闭...")
                asyncio.create_task(self.shutdown_system())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            logger.info("📡 信号处理器已注册")
            
        except Exception as e:
            logger.error(f"❌ 注册信号处理器失败: {e}")
    
    async def _main_loop(self):
        """主循环"""
        try:
            logger.info("🔄 进入主循环...")
            
            while self.is_running:
                try:
                    # 定期健康检查
                    await asyncio.sleep(300)  # 每5分钟
                    
                    if self.is_running:
                        await self._periodic_health_check()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ 主循环错误: {e}")
                    await asyncio.sleep(60)  # 错误后等待1分钟
            
        except Exception as e:
            logger.error(f"❌ 主循环异常: {e}")
        finally:
            logger.info("🔄 主循环已退出")
    
    async def _periodic_health_check(self):
        """定期健康检查"""
        try:
            # 获取系统统计
            stats = await self._collect_system_stats()
            
            # 记录统计信息
            logger.debug(f"📊 系统统计: {stats}")
            
        except Exception as e:
            logger.error(f"❌ 定期健康检查失败: {e}")
    
    async def _collect_system_stats(self) -> Dict[str, Any]:
        """收集系统统计信息"""
        try:
            stats = {
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                "components_count": len(self.components),
                "is_running": self.is_running
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 收集系统统计失败: {e}")
            return {}
    
    async def shutdown_system(self):
        """关闭系统"""
        try:
            if not self.is_running:
                return
            
            logger.info("🛑 开始关闭系统...")
            self.is_running = False
            
            # 关闭各组件
            shutdown_order = [
                'ai_fusion_engine',
                'ai_model_manager', 
                'system_monitor'
            ]
            
            for component_name in shutdown_order:
                if component_name in self.components:
                    try:
                        component = self.components[component_name]
                        if hasattr(component, 'shutdown'):
                            await component.shutdown()
                        logger.info(f"✅ {component_name} 已关闭")
                    except Exception as e:
                        logger.error(f"❌ 关闭 {component_name} 失败: {e}")
            
            logger.info("🎯 系统已完全关闭")
            
        except Exception as e:
            logger.error(f"❌ 系统关闭失败: {e}")

async def main():
    """主函数"""
    try:
        # 创建系统管理器
        system_manager = ProductionSystemManager()
        
        # 启动系统
        await system_manager.start_system()
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭系统...")
    except Exception as e:
        logger.error(f"❌ 系统运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # 设置事件循环策略（Windows兼容性）
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 运行主函数
        asyncio.run(main())
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)
