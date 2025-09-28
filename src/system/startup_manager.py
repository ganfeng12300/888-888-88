"""
🚀 系统启动管理器
生产级系统启动流程管理，实现多进程启动、资源管理、依赖注入等完整功能
支持优雅启动、依赖检查、资源分配和故障恢复
"""

import asyncio
import multiprocessing as mp
import threading
import time
import os
import sys
import signal
import psutil
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings


class ComponentStatus(Enum):
    """组件状态"""
    INITIALIZING = "initializing"   # 初始化中
    STARTING = "starting"           # 启动中
    RUNNING = "running"             # 运行中
    STOPPING = "stopping"           # 停止中
    STOPPED = "stopped"             # 已停止
    ERROR = "error"                 # 错误状态
    FAILED = "failed"               # 启动失败


class ComponentType(Enum):
    """组件类型"""
    CORE_SERVICE = "core_service"           # 核心服务
    DATA_SERVICE = "data_service"           # 数据服务
    STRATEGY_SERVICE = "strategy_service"   # 策略服务
    RISK_SERVICE = "risk_service"           # 风险服务
    EXECUTION_SERVICE = "execution_service" # 执行服务
    MONITORING_SERVICE = "monitoring_service" # 监控服务
    EXTERNAL_SERVICE = "external_service"   # 外部服务


@dataclass
class ComponentConfig:
    """组件配置"""
    name: str                               # 组件名称
    component_type: ComponentType           # 组件类型
    module_path: str                        # 模块路径
    class_name: str                         # 类名
    dependencies: List[str] = field(default_factory=list)  # 依赖组件
    startup_timeout: int = 30               # 启动超时(秒)
    health_check_interval: int = 10         # 健康检查间隔(秒)
    max_restart_attempts: int = 3           # 最大重启尝试次数
    process_type: str = "thread"            # 进程类型: thread/process
    cpu_cores: Optional[List[int]] = None   # CPU核心绑定
    memory_limit: Optional[int] = None      # 内存限制(MB)
    priority: int = 0                       # 启动优先级
    config: Dict[str, Any] = field(default_factory=dict)  # 组件配置
    environment: Dict[str, str] = field(default_factory=dict)  # 环境变量


@dataclass
class ComponentInstance:
    """组件实例"""
    config: ComponentConfig                 # 组件配置
    status: ComponentStatus                 # 当前状态
    instance: Optional[Any] = None          # 组件实例
    process: Optional[mp.Process] = None    # 进程对象
    thread: Optional[threading.Thread] = None  # 线程对象
    pid: Optional[int] = None               # 进程ID
    start_time: Optional[float] = None      # 启动时间
    last_health_check: Optional[float] = None  # 最后健康检查时间
    restart_count: int = 0                  # 重启次数
    error_message: Optional[str] = None     # 错误信息
    metrics: Dict[str, Any] = field(default_factory=dict)  # 性能指标


class DependencyResolver:
    """依赖解析器"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, List[str]] = {}
        self.resolved_order: List[str] = []
    
    def add_component(self, name: str, dependencies: List[str]):
        """添加组件依赖"""
        self.dependency_graph[name] = dependencies
    
    def resolve_dependencies(self) -> List[str]:
        """解析依赖顺序"""
        visited = set()
        temp_visited = set()
        self.resolved_order = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"循环依赖检测到: {node}")
            
            if node not in visited:
                temp_visited.add(node)
                
                # 访问依赖
                for dependency in self.dependency_graph.get(node, []):
                    visit(dependency)
                
                temp_visited.remove(node)
                visited.add(node)
                self.resolved_order.append(node)
        
        # 访问所有节点
        for node in self.dependency_graph:
            if node not in visited:
                visit(node)
        
        return self.resolved_order


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.cpu_cores = list(range(psutil.cpu_count()))
        self.allocated_cores: Dict[str, List[int]] = {}
        self.memory_usage: Dict[str, int] = {}
        self.total_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
        self.allocated_memory = 0
        
        logger.info(f"资源管理器初始化 - CPU核心: {len(self.cpu_cores)}, 内存: {self.total_memory}MB")
    
    def allocate_cpu_cores(self, component_name: str, requested_cores: Optional[List[int]] = None) -> List[int]:
        """分配CPU核心"""
        if requested_cores:
            # 检查请求的核心是否可用
            available_cores = [core for core in requested_cores if core not in sum(self.allocated_cores.values(), [])]
            if len(available_cores) == len(requested_cores):
                self.allocated_cores[component_name] = requested_cores
                logger.info(f"为 {component_name} 分配指定CPU核心: {requested_cores}")
                return requested_cores
            else:
                logger.warning(f"请求的CPU核心不完全可用，分配可用核心")
        
        # 自动分配可用核心
        allocated_cores = sum(self.allocated_cores.values(), [])
        available_cores = [core for core in self.cpu_cores if core not in allocated_cores]
        
        if available_cores:
            allocated_core = [available_cores[0]]
            self.allocated_cores[component_name] = allocated_core
            logger.info(f"为 {component_name} 分配CPU核心: {allocated_core}")
            return allocated_core
        else:
            logger.warning(f"无可用CPU核心分配给 {component_name}")
            return []
    
    def allocate_memory(self, component_name: str, requested_memory: int) -> bool:
        """分配内存"""
        if self.allocated_memory + requested_memory <= self.total_memory * 0.8:  # 保留20%内存
            self.memory_usage[component_name] = requested_memory
            self.allocated_memory += requested_memory
            logger.info(f"为 {component_name} 分配内存: {requested_memory}MB")
            return True
        else:
            logger.warning(f"内存不足，无法为 {component_name} 分配 {requested_memory}MB")
            return False
    
    def release_resources(self, component_name: str):
        """释放资源"""
        if component_name in self.allocated_cores:
            cores = self.allocated_cores.pop(component_name)
            logger.info(f"释放 {component_name} 的CPU核心: {cores}")
        
        if component_name in self.memory_usage:
            memory = self.memory_usage.pop(component_name)
            self.allocated_memory -= memory
            logger.info(f"释放 {component_name} 的内存: {memory}MB")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            'cpu_cores': {
                'total': len(self.cpu_cores),
                'allocated': self.allocated_cores,
                'available': [core for core in self.cpu_cores if core not in sum(self.allocated_cores.values(), [])]
            },
            'memory': {
                'total_mb': self.total_memory,
                'allocated_mb': self.allocated_memory,
                'usage_by_component': self.memory_usage,
                'available_mb': self.total_memory - self.allocated_memory
            }
        }


class ComponentLoader:
    """组件加载器"""
    
    @staticmethod
    def load_component_class(module_path: str, class_name: str) -> Type:
        """动态加载组件类"""
        try:
            # 导入模块
            module = __import__(module_path, fromlist=[class_name])
            
            # 获取类
            component_class = getattr(module, class_name)
            
            logger.info(f"成功加载组件类: {module_path}.{class_name}")
            return component_class
            
        except ImportError as e:
            logger.error(f"导入模块失败: {module_path} - {e}")
            raise
        except AttributeError as e:
            logger.error(f"类不存在: {class_name} in {module_path} - {e}")
            raise
    
    @staticmethod
    def create_component_instance(component_class: Type, config: Dict[str, Any]) -> Any:
        """创建组件实例"""
        try:
            # 创建实例
            if config:
                instance = component_class(**config)
            else:
                instance = component_class()
            
            logger.info(f"成功创建组件实例: {component_class.__name__}")
            return instance
            
        except Exception as e:
            logger.error(f"创建组件实例失败: {component_class.__name__} - {e}")
            raise


class SystemStartupManager:
    """系统启动管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/system_config.yaml"
        self.components: Dict[str, ComponentInstance] = {}
        self.dependency_resolver = DependencyResolver()
        self.resource_manager = ResourceManager()
        self.component_loader = ComponentLoader()
        
        # 启动状态
        self.is_starting = False
        self.is_running = False
        self.is_stopping = False
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="StartupManager")
        
        # 信号处理
        self._setup_signal_handlers()
        
        logger.info("系统启动管理器初始化完成")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，开始优雅关闭")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'components': [
                {
                    'name': 'data_manager',
                    'component_type': 'data_service',
                    'module_path': 'src.data.data_manager',
                    'class_name': 'DataManager',
                    'dependencies': [],
                    'priority': 1
                },
                {
                    'name': 'risk_manager',
                    'component_type': 'risk_service',
                    'module_path': 'src.risk.risk_manager',
                    'class_name': 'RiskManager',
                    'dependencies': ['data_manager'],
                    'priority': 2
                },
                {
                    'name': 'strategy_engine',
                    'component_type': 'strategy_service',
                    'module_path': 'src.high_frequency.high_frequency_strategies',
                    'class_name': 'HighFrequencyStrategyEngine',
                    'dependencies': ['data_manager', 'risk_manager'],
                    'priority': 3
                },
                {
                    'name': 'execution_engine',
                    'component_type': 'execution_service',
                    'module_path': 'src.execution.execution_engine',
                    'class_name': 'ExecutionEngine',
                    'dependencies': ['data_manager', 'risk_manager', 'strategy_engine'],
                    'priority': 4
                }
            ]
        }
    
    def parse_component_configs(self, config: Dict[str, Any]) -> List[ComponentConfig]:
        """解析组件配置"""
        component_configs = []
        
        for comp_config in config.get('components', []):
            try:
                component_config = ComponentConfig(
                    name=comp_config['name'],
                    component_type=ComponentType(comp_config.get('component_type', 'core_service')),
                    module_path=comp_config['module_path'],
                    class_name=comp_config['class_name'],
                    dependencies=comp_config.get('dependencies', []),
                    startup_timeout=comp_config.get('startup_timeout', 30),
                    health_check_interval=comp_config.get('health_check_interval', 10),
                    max_restart_attempts=comp_config.get('max_restart_attempts', 3),
                    process_type=comp_config.get('process_type', 'thread'),
                    cpu_cores=comp_config.get('cpu_cores'),
                    memory_limit=comp_config.get('memory_limit'),
                    priority=comp_config.get('priority', 0),
                    config=comp_config.get('config', {}),
                    environment=comp_config.get('environment', {})
                )
                
                component_configs.append(component_config)
                
            except Exception as e:
                logger.error(f"解析组件配置失败: {comp_config.get('name', 'unknown')} - {e}")
        
        return component_configs
    
    async def start_system(self) -> bool:
        """启动系统"""
        try:
            if self.is_starting or self.is_running:
                logger.warning("系统已在启动或运行中")
                return False
            
            self.is_starting = True
            logger.info("开始启动系统...")
            
            # 加载配置
            config = self.load_config()
            component_configs = self.parse_component_configs(config)
            
            if not component_configs:
                logger.error("没有找到有效的组件配置")
                return False
            
            # 构建依赖图
            for comp_config in component_configs:
                self.dependency_resolver.add_component(comp_config.name, comp_config.dependencies)
            
            # 解析启动顺序
            try:
                startup_order = self.dependency_resolver.resolve_dependencies()
                logger.info(f"组件启动顺序: {startup_order}")
            except ValueError as e:
                logger.error(f"依赖解析失败: {e}")
                return False
            
            # 按优先级和依赖顺序启动组件
            component_configs_dict = {config.name: config for config in component_configs}
            
            for component_name in startup_order:
                if component_name in component_configs_dict:
                    config = component_configs_dict[component_name]
                    success = await self._start_component(config)
                    
                    if not success:
                        logger.error(f"组件启动失败: {component_name}")
                        await self._rollback_startup()
                        return False
                    
                    # 等待组件稳定
                    await asyncio.sleep(1)
            
            self.is_starting = False
            self.is_running = True
            
            # 启动健康检查
            asyncio.create_task(self._health_check_loop())
            
            logger.info("系统启动完成")
            return True
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            self.is_starting = False
            await self._rollback_startup()
            return False
    
    async def _start_component(self, config: ComponentConfig) -> bool:
        """启动单个组件"""
        try:
            logger.info(f"启动组件: {config.name}")
            
            # 创建组件实例
            component_instance = ComponentInstance(
                config=config,
                status=ComponentStatus.INITIALIZING
            )
            
            self.components[config.name] = component_instance
            
            # 分配资源
            if config.cpu_cores:
                allocated_cores = self.resource_manager.allocate_cpu_cores(config.name, config.cpu_cores)
            else:
                allocated_cores = self.resource_manager.allocate_cpu_cores(config.name)
            
            if config.memory_limit:
                memory_allocated = self.resource_manager.allocate_memory(config.name, config.memory_limit)
                if not memory_allocated:
                    logger.error(f"内存分配失败: {config.name}")
                    return False
            
            # 加载组件类
            component_class = self.component_loader.load_component_class(
                config.module_path, config.class_name
            )
            
            # 创建组件实例
            component_instance.instance = self.component_loader.create_component_instance(
                component_class, config.config
            )
            
            component_instance.status = ComponentStatus.STARTING
            component_instance.start_time = time.time()
            
            # 根据进程类型启动
            if config.process_type == 'process':
                await self._start_component_process(component_instance, allocated_cores)
            else:
                await self._start_component_thread(component_instance)
            
            # 等待启动完成
            start_time = time.time()
            while (time.time() - start_time) < config.startup_timeout:
                if component_instance.status == ComponentStatus.RUNNING:
                    logger.info(f"组件启动成功: {config.name}")
                    return True
                elif component_instance.status == ComponentStatus.FAILED:
                    logger.error(f"组件启动失败: {config.name}")
                    return False
                
                await asyncio.sleep(0.5)
            
            # 启动超时
            logger.error(f"组件启动超时: {config.name}")
            component_instance.status = ComponentStatus.FAILED
            return False
            
        except Exception as e:
            logger.error(f"启动组件失败: {config.name} - {e}")
            if config.name in self.components:
                self.components[config.name].status = ComponentStatus.FAILED
                self.components[config.name].error_message = str(e)
            return False
    
    async def _start_component_thread(self, component_instance: ComponentInstance):
        """以线程方式启动组件"""
        def run_component():
            try:
                # 设置CPU亲和性
                if component_instance.config.cpu_cores:
                    try:
                        process = psutil.Process()
                        process.cpu_affinity(component_instance.config.cpu_cores)
                    except Exception as e:
                        logger.warning(f"设置CPU亲和性失败: {e}")
                
                # 启动组件
                if hasattr(component_instance.instance, 'start'):
                    asyncio.run(component_instance.instance.start())
                
                component_instance.status = ComponentStatus.RUNNING
                
            except Exception as e:
                logger.error(f"组件线程运行失败: {component_instance.config.name} - {e}")
                component_instance.status = ComponentStatus.FAILED
                component_instance.error_message = str(e)
        
        # 创建并启动线程
        thread = threading.Thread(
            target=run_component,
            name=f"Component-{component_instance.config.name}",
            daemon=True
        )
        
        component_instance.thread = thread
        thread.start()
    
    async def _start_component_process(self, component_instance: ComponentInstance, cpu_cores: List[int]):
        """以进程方式启动组件"""
        def run_component_process():
            try:
                # 设置CPU亲和性
                if cpu_cores:
                    try:
                        process = psutil.Process()
                        process.cpu_affinity(cpu_cores)
                    except Exception as e:
                        logger.warning(f"设置CPU亲和性失败: {e}")
                
                # 设置环境变量
                for key, value in component_instance.config.environment.items():
                    os.environ[key] = value
                
                # 启动组件
                if hasattr(component_instance.instance, 'start'):
                    asyncio.run(component_instance.instance.start())
                
                component_instance.status = ComponentStatus.RUNNING
                
            except Exception as e:
                logger.error(f"组件进程运行失败: {component_instance.config.name} - {e}")
                component_instance.status = ComponentStatus.FAILED
                component_instance.error_message = str(e)
        
        # 创建并启动进程
        process = mp.Process(
            target=run_component_process,
            name=f"Component-{component_instance.config.name}"
        )
        
        component_instance.process = process
        process.start()
        component_instance.pid = process.pid
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                for component_name, component_instance in self.components.items():
                    if component_instance.status == ComponentStatus.RUNNING:
                        await self._check_component_health(component_instance)
                
                await asyncio.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(5)
    
    async def _check_component_health(self, component_instance: ComponentInstance):
        """检查组件健康状态"""
        try:
            current_time = time.time()
            
            # 检查进程/线程是否存活
            if component_instance.process:
                if not component_instance.process.is_alive():
                    logger.warning(f"组件进程已停止: {component_instance.config.name}")
                    await self._restart_component(component_instance)
                    return
            
            if component_instance.thread:
                if not component_instance.thread.is_alive():
                    logger.warning(f"组件线程已停止: {component_instance.config.name}")
                    await self._restart_component(component_instance)
                    return
            
            # 调用组件健康检查方法
            if hasattr(component_instance.instance, 'health_check'):
                try:
                    health_status = await component_instance.instance.health_check()
                    if not health_status:
                        logger.warning(f"组件健康检查失败: {component_instance.config.name}")
                        await self._restart_component(component_instance)
                        return
                except Exception as e:
                    logger.error(f"组件健康检查异常: {component_instance.config.name} - {e}")
                    await self._restart_component(component_instance)
                    return
            
            component_instance.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"健康检查失败: {component_instance.config.name} - {e}")
    
    async def _restart_component(self, component_instance: ComponentInstance):
        """重启组件"""
        try:
            if component_instance.restart_count >= component_instance.config.max_restart_attempts:
                logger.error(f"组件重启次数超限: {component_instance.config.name}")
                component_instance.status = ComponentStatus.FAILED
                return
            
            logger.info(f"重启组件: {component_instance.config.name}")
            component_instance.restart_count += 1
            
            # 停止组件
            await self._stop_component(component_instance)
            
            # 等待一段时间
            await asyncio.sleep(2)
            
            # 重新启动
            success = await self._start_component(component_instance.config)
            if success:
                logger.info(f"组件重启成功: {component_instance.config.name}")
            else:
                logger.error(f"组件重启失败: {component_instance.config.name}")
                
        except Exception as e:
            logger.error(f"重启组件失败: {component_instance.config.name} - {e}")
    
    async def _stop_component(self, component_instance: ComponentInstance):
        """停止组件"""
        try:
            component_instance.status = ComponentStatus.STOPPING
            
            # 调用组件停止方法
            if hasattr(component_instance.instance, 'shutdown'):
                try:
                    await component_instance.instance.shutdown()
                except Exception as e:
                    logger.warning(f"组件优雅关闭失败: {component_instance.config.name} - {e}")
            
            # 停止进程/线程
            if component_instance.process and component_instance.process.is_alive():
                component_instance.process.terminate()
                component_instance.process.join(timeout=5)
                
                if component_instance.process.is_alive():
                    component_instance.process.kill()
            
            if component_instance.thread and component_instance.thread.is_alive():
                # 线程无法强制停止，只能等待
                pass
            
            # 释放资源
            self.resource_manager.release_resources(component_instance.config.name)
            
            component_instance.status = ComponentStatus.STOPPED
            logger.info(f"组件已停止: {component_instance.config.name}")
            
        except Exception as e:
            logger.error(f"停止组件失败: {component_instance.config.name} - {e}")
    
    async def _rollback_startup(self):
        """回滚启动过程"""
        logger.info("开始回滚启动过程...")
        
        # 停止所有已启动的组件
        for component_instance in self.components.values():
            if component_instance.status in [ComponentStatus.RUNNING, ComponentStatus.STARTING]:
                await self._stop_component(component_instance)
        
        self.components.clear()
        self.is_starting = False
        self.is_running = False
    
    async def shutdown(self):
        """关闭系统"""
        try:
            if self.is_stopping:
                logger.warning("系统已在关闭中")
                return
            
            self.is_stopping = True
            self.is_running = False
            
            logger.info("开始关闭系统...")
            
            # 按依赖顺序反向停止组件
            startup_order = self.dependency_resolver.resolved_order
            shutdown_order = list(reversed(startup_order))
            
            for component_name in shutdown_order:
                if component_name in self.components:
                    await self._stop_component(self.components[component_name])
                    await asyncio.sleep(1)  # 等待组件停止
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.info("系统关闭完成")
            
        except Exception as e:
            logger.error(f"系统关闭失败: {e}")
        finally:
            self.is_stopping = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        component_status = {}
        for name, instance in self.components.items():
            component_status[name] = {
                'status': instance.status.value,
                'start_time': instance.start_time,
                'restart_count': instance.restart_count,
                'last_health_check': instance.last_health_check,
                'error_message': instance.error_message,
                'pid': instance.pid
            }
        
        return {
            'system_status': {
                'is_starting': self.is_starting,
                'is_running': self.is_running,
                'is_stopping': self.is_stopping
            },
            'components': component_status,
            'resource_usage': self.resource_manager.get_resource_usage()
        }


# 全局启动管理器实例
startup_manager = SystemStartupManager()
