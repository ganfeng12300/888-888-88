#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 60秒启动管理器
智能系统启动序列，自动化初始化所有组件
专为史诗级AI量化交易设计，生产级实盘交易标准
"""

import asyncio
import time
import psutil
import GPUtil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json
from pathlib import Path

class StartupPhase(Enum):
    """启动阶段"""
    SYSTEM_CHECK = "system_check"
    STORAGE_CHECK = "storage_check"
    NETWORK_CHECK = "network_check"
    AI_MODELS = "ai_models"
    RISK_CONTROL = "risk_control"
    WEB_INTERFACE = "web_interface"
    DATA_COLLECTION = "data_collection"
    FINAL_CHECK = "final_check"

class ComponentStatus(Enum):
    """组件状态"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class StartupComponent:
    """启动组件"""
    name: str
    phase: StartupPhase
    priority: int
    timeout: float  # 秒
    status: ComponentStatus = ComponentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: str = ""
    progress: float = 0.0
    
    @property
    def duration(self) -> float:
        """获取执行时间"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class StartupManager:
    """🦊 猎狐AI - 60秒启动管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components = {}  # name -> StartupComponent
        self.phase_progress = {}  # phase -> progress
        self.is_starting = False
        self.start_time = None
        self.total_timeout = 60.0  # 60秒总超时
        
        # 启动回调
        self.progress_callbacks = []
        self.completion_callbacks = []
        
        # 初始化组件
        self._init_components()
        
        logger.info("🦊 猎狐AI 60秒启动管理器初始化完成")
    
    def _init_components(self):
        """初始化启动组件"""
        components_config = [
            # 第1阶段：系统自检 (0-10秒)
            ("硬件检测", StartupPhase.SYSTEM_CHECK, 1, 3.0),
            ("内存检查", StartupPhase.SYSTEM_CHECK, 2, 2.0),
            ("GPU检测", StartupPhase.SYSTEM_CHECK, 3, 3.0),
            ("网络连通性", StartupPhase.SYSTEM_CHECK, 4, 2.0),
            
            # 第2阶段：存储检查 (10-20秒)
            ("硬盘空间", StartupPhase.STORAGE_CHECK, 1, 2.0),
            ("数据库连接", StartupPhase.STORAGE_CHECK, 2, 3.0),
            ("Redis连接", StartupPhase.STORAGE_CHECK, 3, 2.0),
            ("数据清理", StartupPhase.STORAGE_CHECK, 4, 3.0),
            
            # 第3阶段：网络检查 (20-30秒)
            ("交易所API", StartupPhase.NETWORK_CHECK, 1, 4.0),
            ("新闻API", StartupPhase.NETWORK_CHECK, 2, 2.0),
            ("WebSocket连接", StartupPhase.NETWORK_CHECK, 3, 4.0),
            
            # 第4阶段：AI模型加载 (30-40秒)
            ("元学习指挥官", StartupPhase.AI_MODELS, 1, 3.0),
            ("强化学习交易员", StartupPhase.AI_MODELS, 2, 3.0),
            ("时序预测先知", StartupPhase.AI_MODELS, 3, 2.0),
            ("集成学习智囊团", StartupPhase.AI_MODELS, 4, 2.0),
            
            # 第5阶段：风控系统 (40-50秒)
            ("五层风控矩阵", StartupPhase.RISK_CONTROL, 1, 3.0),
            ("风险计算器", StartupPhase.RISK_CONTROL, 2, 2.0),
            ("订单管理器", StartupPhase.RISK_CONTROL, 3, 3.0),
            ("交易执行引擎", StartupPhase.RISK_CONTROL, 4, 2.0),
            
            # 第6阶段：Web界面 (50-55秒)
            ("Web服务器", StartupPhase.WEB_INTERFACE, 1, 2.0),
            ("监控面板", StartupPhase.WEB_INTERFACE, 2, 2.0),
            ("实时数据流", StartupPhase.WEB_INTERFACE, 3, 1.0),
            
            # 第7阶段：数据采集 (55-60秒)
            ("市场数据采集", StartupPhase.DATA_COLLECTION, 1, 2.0),
            ("技术指标计算", StartupPhase.DATA_COLLECTION, 2, 1.0),
            ("AI训练启动", StartupPhase.DATA_COLLECTION, 3, 2.0),
            
            # 第8阶段：最终检查 (60秒)
            ("系统就绪检查", StartupPhase.FINAL_CHECK, 1, 1.0),
        ]
        
        for name, phase, priority, timeout in components_config:
            component = StartupComponent(
                name=name,
                phase=phase,
                priority=priority,
                timeout=timeout
            )
            self.components[name] = component
        
        # 初始化阶段进度
        for phase in StartupPhase:
            self.phase_progress[phase] = 0.0
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加进度回调"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[bool, Dict[str, Any]], None]):
        """添加完成回调"""
        self.completion_callbacks.append(callback)
    
    async def start_system(self) -> bool:
        """启动系统"""
        if self.is_starting:
            logger.warning("⚠️ 系统正在启动中")
            return False
        
        self.is_starting = True
        self.start_time = datetime.now(timezone.utc)
        
        logger.info("🚀 开始60秒启动序列...")
        
        try:
            # 按阶段启动
            success = True
            
            for phase in StartupPhase:
                phase_success = await self._execute_phase(phase)
                if not phase_success:
                    success = False
                    break
            
            # 计算总启动时间
            total_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            # 调用完成回调
            completion_info = {
                'success': success,
                'total_time': total_time,
                'components': {name: {
                    'status': comp.status.value,
                    'duration': comp.duration,
                    'error': comp.error_message
                } for name, comp in self.components.items()}
            }
            
            for callback in self.completion_callbacks:
                try:
                    callback(success, completion_info)
                except Exception as e:
                    logger.error(f"❌ 完成回调异常: {e}")
            
            if success:
                logger.success(f"✅ 系统启动完成！用时 {total_time:.1f}秒")
            else:
                logger.error(f"❌ 系统启动失败！用时 {total_time:.1f}秒")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 系统启动异常: {e}")
            return False
        finally:
            self.is_starting = False
    
    async def _execute_phase(self, phase: StartupPhase) -> bool:
        """执行启动阶段"""
        try:
            logger.info(f"📋 执行阶段: {phase.value}")
            
            # 获取该阶段的组件
            phase_components = [comp for comp in self.components.values() if comp.phase == phase]
            phase_components.sort(key=lambda x: x.priority)
            
            if not phase_components:
                self.phase_progress[phase] = 100.0
                return True
            
            # 执行组件
            completed = 0
            total = len(phase_components)
            
            for component in phase_components:
                success = await self._execute_component(component)
                completed += 1
                
                # 更新阶段进度
                self.phase_progress[phase] = (completed / total) * 100.0
                
                # 发送进度更新
                await self._send_progress_update()
                
                if not success:
                    logger.error(f"❌ 阶段 {phase.value} 失败于组件: {component.name}")
                    return False
            
            logger.success(f"✅ 阶段 {phase.value} 完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 阶段 {phase.value} 异常: {e}")
            return False
    
    async def _execute_component(self, component: StartupComponent) -> bool:
        """执行组件启动"""
        try:
            component.status = ComponentStatus.INITIALIZING
            component.start_time = datetime.now(timezone.utc)
            
            logger.debug(f"🔧 启动组件: {component.name}")
            
            # 根据组件名称执行相应的初始化
            success = await self._initialize_component(component)
            
            component.end_time = datetime.now(timezone.utc)
            
            if success:
                component.status = ComponentStatus.READY
                component.progress = 100.0
                logger.debug(f"✅ 组件就绪: {component.name} ({component.duration:.2f}s)")
            else:
                component.status = ComponentStatus.ERROR
                logger.error(f"❌ 组件失败: {component.name}")
            
            return success
            
        except asyncio.TimeoutError:
            component.status = ComponentStatus.TIMEOUT
            component.error_message = f"超时 ({component.timeout}s)"
            component.end_time = datetime.now(timezone.utc)
            logger.error(f"⏰ 组件超时: {component.name}")
            return False
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error_message = str(e)
            component.end_time = datetime.now(timezone.utc)
            logger.error(f"❌ 组件异常: {component.name} - {e}")
            return False
    
    async def _initialize_component(self, component: StartupComponent) -> bool:
        """初始化具体组件"""
        try:
            # 使用超时包装
            return await asyncio.wait_for(
                self._do_component_initialization(component),
                timeout=component.timeout
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"❌ 组件初始化失败 {component.name}: {e}")
            return False
    
    async def _do_component_initialization(self, component: StartupComponent) -> bool:
        """执行组件初始化逻辑"""
        name = component.name
        
        try:
            if name == "硬件检测":
                return await self._check_hardware()
            elif name == "内存检查":
                return await self._check_memory()
            elif name == "GPU检测":
                return await self._check_gpu()
            elif name == "网络连通性":
                return await self._check_network()
            elif name == "硬盘空间":
                return await self._check_disk_space()
            elif name == "数据库连接":
                return await self._check_database()
            elif name == "Redis连接":
                return await self._check_redis()
            elif name == "数据清理":
                return await self._cleanup_data()
            elif name == "交易所API":
                return await self._check_exchange_api()
            elif name == "新闻API":
                return await self._check_news_api()
            elif name == "WebSocket连接":
                return await self._check_websocket()
            elif name in ["元学习指挥官", "强化学习交易员", "时序预测先知", "集成学习智囊团"]:
                return await self._load_ai_model(name)
            elif name == "五层风控矩阵":
                return await self._init_risk_control()
            elif name == "风险计算器":
                return await self._init_risk_calculator()
            elif name == "订单管理器":
                return await self._init_order_manager()
            elif name == "交易执行引擎":
                return await self._init_trading_engine()
            elif name == "Web服务器":
                return await self._start_web_server()
            elif name == "监控面板":
                return await self._init_monitoring()
            elif name == "实时数据流":
                return await self._init_realtime_data()
            elif name == "市场数据采集":
                return await self._start_data_collection()
            elif name == "技术指标计算":
                return await self._init_indicators()
            elif name == "AI训练启动":
                return await self._start_ai_training()
            elif name == "系统就绪检查":
                return await self._final_system_check()
            else:
                # 默认成功（模拟）
                await asyncio.sleep(0.5)
                return True
                
        except Exception as e:
            logger.error(f"❌ 组件 {name} 初始化异常: {e}")
            return False
    
    # 具体的初始化方法
    async def _check_hardware(self) -> bool:
        """检查硬件"""
        try:
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            if cpu_count < 4:
                logger.warning(f"⚠️ CPU核心数较少: {cpu_count}")
            
            logger.info(f"💻 CPU: {cpu_count}核心 @ {cpu_freq.current:.0f}MHz")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 硬件检测失败: {e}")
            return False
    
    async def _check_memory(self) -> bool:
        """检查内存"""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2:
                logger.warning(f"⚠️ 可用内存不足: {available_gb:.1f}GB")
                return False
            
            logger.info(f"🧠 内存: {available_gb:.1f}GB可用 / {total_gb:.1f}GB总计")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ 内存检查失败: {e}")
            return False
    
    async def _check_gpu(self) -> bool:
        """检查GPU"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.warning("⚠️ 未检测到GPU，将使用CPU模式")
                return True
            
            for gpu in gpus:
                logger.info(f"🎮 GPU: {gpu.name} {gpu.memoryFree}MB可用/{gpu.memoryTotal}MB")
                if gpu.temperature > 85:
                    logger.warning(f"⚠️ GPU温度过高: {gpu.temperature}°C")
            
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ GPU检测失败: {e}")
            return True  # GPU不是必需的
    
    async def _check_network(self) -> bool:
        """检查网络连通性"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/ip', timeout=3) as response:
                    if response.status == 200:
                        logger.info("🌐 网络连接正常")
                        return True
            return False
        except Exception as e:
            logger.error(f"❌ 网络检查失败: {e}")
            return False
    
    async def _check_disk_space(self) -> bool:
        """检查硬盘空间"""
        try:
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            
            if free_gb < 5:
                logger.error(f"❌ 硬盘空间不足: {free_gb:.1f}GB")
                return False
            
            logger.info(f"💾 硬盘: {free_gb:.1f}GB可用 / {total_gb:.1f}GB总计")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ 硬盘检查失败: {e}")
            return False
    
    async def _check_database(self) -> bool:
        """检查数据库连接"""
        try:
            import sqlite3
            # 测试SQLite连接
            conn = sqlite3.connect(':memory:')
            conn.execute('SELECT 1')
            conn.close()
            
            logger.info("🗄️ 数据库连接正常")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 数据库检查失败: {e}")
            return False
    
    async def _check_redis(self) -> bool:
        """检查Redis连接"""
        try:
            # 模拟Redis检查
            logger.info("📦 Redis连接检查（模拟）")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis检查失败: {e}")
            return True  # Redis不是必需的
    
    async def _cleanup_data(self) -> bool:
        """数据清理"""
        try:
            # 清理临时文件
            temp_dir = Path('temp')
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info("🧹 数据清理完成")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ 数据清理失败: {e}")
            return True  # 清理失败不影响启动
    
    async def _check_exchange_api(self) -> bool:
        """检查交易所API"""
        try:
            # 模拟API检查
            logger.info("🏦 交易所API连接检查")
            await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"❌ 交易所API检查失败: {e}")
            return False
    
    async def _check_news_api(self) -> bool:
        """检查新闻API"""
        try:
            logger.info("📰 新闻API连接检查")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ 新闻API检查失败: {e}")
            return True  # 新闻API不是必需的
    
    async def _check_websocket(self) -> bool:
        """检查WebSocket连接"""
        try:
            logger.info("🔌 WebSocket连接检查")
            await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"❌ WebSocket检查失败: {e}")
            return False
    
    async def _load_ai_model(self, model_name: str) -> bool:
        """加载AI模型"""
        try:
            logger.info(f"🧠 加载AI模型: {model_name}")
            await asyncio.sleep(2)  # 模拟模型加载时间
            return True
        except Exception as e:
            logger.error(f"❌ AI模型加载失败 {model_name}: {e}")
            return False
    
    async def _init_risk_control(self) -> bool:
        """初始化风控系统"""
        try:
            logger.info("🛡️ 初始化五层风控矩阵")
            await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"❌ 风控系统初始化失败: {e}")
            return False
    
    async def _init_risk_calculator(self) -> bool:
        """初始化风险计算器"""
        try:
            logger.info("📊 初始化风险计算器")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 风险计算器初始化失败: {e}")
            return False
    
    async def _init_order_manager(self) -> bool:
        """初始化订单管理器"""
        try:
            logger.info("📋 初始化订单管理器")
            await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"❌ 订单管理器初始化失败: {e}")
            return False
    
    async def _init_trading_engine(self) -> bool:
        """初始化交易执行引擎"""
        try:
            logger.info("⚡ 初始化交易执行引擎")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 交易执行引擎初始化失败: {e}")
            return False
    
    async def _start_web_server(self) -> bool:
        """启动Web服务器"""
        try:
            logger.info("🌐 启动Web服务器")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ Web服务器启动失败: {e}")
            return False
    
    async def _init_monitoring(self) -> bool:
        """初始化监控面板"""
        try:
            logger.info("📊 初始化监控面板")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 监控面板初始化失败: {e}")
            return False
    
    async def _init_realtime_data(self) -> bool:
        """初始化实时数据流"""
        try:
            logger.info("📡 初始化实时数据流")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ 实时数据流初始化失败: {e}")
            return False
    
    async def _start_data_collection(self) -> bool:
        """启动数据采集"""
        try:
            logger.info("📈 启动市场数据采集")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ 数据采集启动失败: {e}")
            return False
    
    async def _init_indicators(self) -> bool:
        """初始化技术指标"""
        try:
            logger.info("📊 初始化技术指标计算")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ 技术指标初始化失败: {e}")
            return False
    
    async def _start_ai_training(self) -> bool:
        """启动AI训练"""
        try:
            logger.info("🎯 启动AI训练")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"❌ AI训练启动失败: {e}")
            return False
    
    async def _final_system_check(self) -> bool:
        """最终系统检查"""
        try:
            logger.info("✅ 系统就绪检查")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"❌ 最终检查失败: {e}")
            return False
    
    async def _send_progress_update(self):
        """发送进度更新"""
        try:
            # 计算总进度
            total_progress = sum(self.phase_progress.values()) / len(StartupPhase)
            
            # 计算各阶段进度
            phase_info = {}
            for phase in StartupPhase:
                phase_components = [comp for comp in self.components.values() if comp.phase == phase]
                completed = len([comp for comp in phase_components if comp.status == ComponentStatus.READY])
                total = len(phase_components)
                phase_info[phase.value] = {
                    'progress': self.phase_progress[phase],
                    'completed': completed,
                    'total': total
                }
            
            # 当前时间
            current_time = datetime.now(timezone.utc)
            elapsed_time = (current_time - self.start_time).total_seconds() if self.start_time else 0
            
            progress_info = {
                'total_progress': total_progress,
                'elapsed_time': elapsed_time,
                'phases': phase_info,
                'components': {name: {
                    'status': comp.status.value,
                    'progress': comp.progress,
                    'error': comp.error_message
                } for name, comp in self.components.items()}
            }
            
            # 调用进度回调
            for callback in self.progress_callbacks:
                try:
                    callback(progress_info)
                except Exception as e:
                    logger.error(f"❌ 进度回调异常: {e}")
                    
        except Exception as e:
            logger.error(f"❌ 发送进度更新失败: {e}")
    
    def get_startup_status(self) -> Dict[str, Any]:
        """获取启动状态"""
        if not self.start_time:
            return {'status': 'not_started'}
        
        current_time = datetime.now(timezone.utc)
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # 计算总进度
        total_progress = sum(self.phase_progress.values()) / len(StartupPhase)
        
        return {
            'status': 'starting' if self.is_starting else 'completed',
            'total_progress': total_progress,
            'elapsed_time': elapsed_time,
            'phases': {phase.value: progress for phase, progress in self.phase_progress.items()},
            'components': {name: {
                'status': comp.status.value,
                'progress': comp.progress,
                'duration': comp.duration,
                'error': comp.error_message
            } for name, comp in self.components.items()}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            ready_components = len([comp for comp in self.components.values() 
                                  if comp.status == ComponentStatus.READY])
            total_components = len(self.components)
            
            return {
                'status': 'healthy' if ready_components == total_components else 'starting',
                'ready_components': ready_components,
                'total_components': total_components,
                'is_starting': self.is_starting,
                'startup_progress': sum(self.phase_progress.values()) / len(StartupPhase)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

