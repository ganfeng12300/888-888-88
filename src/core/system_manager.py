"""
🎛️ 系统管理器
统一管理所有系统组件的生命周期
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

from src.core.config import settings
from src.data.data_manager import DataManager
from src.ai.ai_engine import AIEngine
from src.trading.trading_engine import TradingEngine
from src.risk.risk_manager import RiskManager
from src.web.web_server import WebServer


class SystemManager:
    """系统管理器 - 协调所有系统组件"""
    
    def __init__(self):
        self.settings = settings
        self.components: Dict[str, Any] = {}
        self.running = False
        
        # 初始化各个组件
        self.data_manager = DataManager()
        self.ai_engine = AIEngine()
        self.trading_engine = TradingEngine()
        self.risk_manager = RiskManager()
        self.web_server = WebServer()
        
        # 组件注册
        self.components = {
            "data_manager": self.data_manager,
            "ai_engine": self.ai_engine,
            "trading_engine": self.trading_engine,
            "risk_manager": self.risk_manager,
            "web_server": self.web_server,
        }
        
        logger.info("系统管理器初始化完成")
    
    async def initialize_databases(self) -> bool:
        """初始化数据库连接"""
        try:
            logger.info("正在初始化数据库连接...")
            await self.data_manager.initialize_databases()
            logger.success("数据库连接初始化完成")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False
    
    async def initialize_ai_models(self) -> bool:
        """初始化AI模型"""
        try:
            logger.info("正在加载AI模型...")
            await self.ai_engine.initialize_models()
            logger.success("AI模型加载完成")
            return True
        except Exception as e:
            logger.error(f"AI模型初始化失败: {e}")
            return False
    
    async def initialize_exchanges(self) -> bool:
        """初始化交易所连接"""
        try:
            logger.info("正在连接交易所...")
            await self.trading_engine.initialize_exchanges()
            logger.success("交易所连接完成")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False
    
    async def initialize_risk_management(self) -> bool:
        """初始化风险管理系统"""
        try:
            logger.info("正在启动风险管理系统...")
            await self.risk_manager.initialize()
            logger.success("风险管理系统启动完成")
            return True
        except Exception as e:
            logger.error(f"风险管理系统初始化失败: {e}")
            return False
    
    async def initialize_web_interface(self) -> bool:
        """初始化Web界面"""
        try:
            logger.info("正在启动Web界面...")
            await self.web_server.initialize()
            logger.success("Web界面启动完成")
            return True
        except Exception as e:
            logger.error(f"Web界面初始化失败: {e}")
            return False
    
    async def start_data_collection(self):
        """启动数据采集"""
        logger.info("启动数据采集服务...")
        await self.data_manager.start_collection()
    
    async def start_ai_decision_engine(self):
        """启动AI决策引擎"""
        logger.info("启动AI决策引擎...")
        await self.ai_engine.start_decision_loop()
    
    async def start_trading_engine(self):
        """启动交易引擎"""
        logger.info("启动交易引擎...")
        await self.trading_engine.start_trading()
    
    async def start_risk_monitoring(self):
        """启动风险监控"""
        logger.info("启动风险监控...")
        await self.risk_manager.start_monitoring()
    
    async def start_web_server(self):
        """启动Web服务器"""
        logger.info("启动Web服务器...")
        await self.web_server.start_server()
    
    async def shutdown(self):
        """安全关闭所有系统组件"""
        logger.info("开始关闭系统...")
        self.running = False
        
        # 按顺序关闭组件
        shutdown_order = [
            "trading_engine",  # 先停止交易
            "ai_engine",       # 停止AI决策
            "risk_manager",    # 停止风险监控
            "data_manager",    # 停止数据采集
            "web_server",      # 最后停止Web服务
        ]
        
        for component_name in shutdown_order:
            try:
                component = self.components.get(component_name)
                if component and hasattr(component, 'shutdown'):
                    logger.info(f"正在关闭 {component_name}...")
                    await component.shutdown()
                    logger.success(f"{component_name} 已安全关闭")
            except Exception as e:
                logger.error(f"关闭 {component_name} 时出错: {e}")
        
        logger.success("系统已安全关闭")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "running": self.running,
            "components": {},
            "performance": {},
            "errors": []
        }
        
        # 获取各组件状态
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    status["components"][name] = component.get_status()
                else:
                    status["components"][name] = {"status": "unknown"}
            except Exception as e:
                status["components"][name] = {"status": "error", "error": str(e)}
                status["errors"].append(f"{name}: {e}")
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            "healthy": True,
            "timestamp": asyncio.get_event_loop().time(),
            "components": {},
            "issues": []
        }
        
        # 检查各组件健康状态
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_status["components"][name] = component_health
                    
                    if not component_health.get("healthy", False):
                        health_status["healthy"] = False
                        health_status["issues"].append(f"{name}: {component_health.get('error', 'Unknown issue')}")
                else:
                    health_status["components"][name] = {"healthy": True, "status": "no_health_check"}
            except Exception as e:
                health_status["healthy"] = False
                health_status["components"][name] = {"healthy": False, "error": str(e)}
                health_status["issues"].append(f"{name}: {e}")
        
        return health_status
    
    async def restart_component(self, component_name: str) -> bool:
        """重启指定组件"""
        try:
            component = self.components.get(component_name)
            if not component:
                logger.error(f"组件 {component_name} 不存在")
                return False
            
            logger.info(f"正在重启组件 {component_name}...")
            
            # 停止组件
            if hasattr(component, 'shutdown'):
                await component.shutdown()
            
            # 重新初始化组件
            if hasattr(component, 'initialize'):
                await component.initialize()
            
            # 启动组件
            if hasattr(component, 'start'):
                await component.start()
            
            logger.success(f"组件 {component_name} 重启成功")
            return True
            
        except Exception as e:
            logger.error(f"重启组件 {component_name} 失败: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {
            "system": {
                "uptime": 0,  # TODO: 实现运行时间计算
                "memory_usage": 0,  # TODO: 实现内存使用率
                "cpu_usage": 0,  # TODO: 实现CPU使用率
                "gpu_usage": 0,  # TODO: 实现GPU使用率
            },
            "trading": {
                "total_trades": 0,  # TODO: 从交易引擎获取
                "success_rate": 0,  # TODO: 计算成功率
                "profit_loss": 0,  # TODO: 计算盈亏
                "drawdown": 0,  # TODO: 计算回撤
            },
            "ai": {
                "prediction_accuracy": 0,  # TODO: 从AI引擎获取
                "model_performance": {},  # TODO: 各模型性能
                "training_status": {},  # TODO: 训练状态
            }
        }
        
        # TODO: 实现具体的性能指标收集逻辑
        
        return metrics
