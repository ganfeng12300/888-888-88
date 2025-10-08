"""
🚀 生产级硬件资源管理器
专为20核CPU + GTX3060 12GB + 128GB内存 + 1TB NVMe SSD优化
实现最大化硬件利用率和性能优化
"""

import os
import psutil
import threading
import asyncio
import time
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from loguru import logger
import torch
import multiprocessing as mp

class ResourceType(Enum):
    """资源类型"""
    CPU_CORE = "cpu_core"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    DISK_SPACE = "disk_space"
    NETWORK_BANDWIDTH = "network_bandwidth"

class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 1    # 关键任务：实时交易执行
    HIGH = 2        # 高优先级：AI模型推理
    MEDIUM = 3      # 中优先级：AI模型训练
    LOW = 4         # 低优先级：数据清理、日志

@dataclass
class ResourceAllocation:
    """资源分配配置"""
    cpu_cores: List[int]
    gpu_memory_mb: int
    system_memory_mb: int
    disk_space_gb: int
    priority: TaskPriority
    task_name: str
    max_threads: int = 1

@dataclass
class HardwareSpecs:
    """硬件规格"""
    cpu_cores: int = 20
    gpu_memory_gb: int = 12
    system_memory_gb: int = 128
    disk_space_tb: int = 1
    gpu_model: str = "GTX 3060"

class ProductionResourceManager:
    """生产级资源管理器"""
    
    def __init__(self):
        self.hardware_specs = HardwareSpecs()
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.active_tasks: Dict[str, Any] = {}
        self.resource_monitor = ResourceMonitor()
        
        # 线程池和进程池
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        self.process_pool = ProcessPoolExecutor(max_workers=10)
        
        # 资源锁
        self.resource_lock = threading.RLock()
        
        # 初始化资源分配
        self._initialize_resource_allocation()
        
        logger.info("🚀 生产级资源管理器初始化完成")
        logger.info(f"💻 硬件配置: {self.hardware_specs.cpu_cores}核CPU, {self.hardware_specs.gpu_model} {self.hardware_specs.gpu_memory_gb}GB, {self.hardware_specs.system_memory_gb}GB内存")

    def _initialize_resource_allocation(self):
        """初始化资源分配策略"""
        
        # CPU核心分配策略 (20核)
        cpu_allocations = {
            "real_time_data_collection": [0, 1, 2, 3],      # 核心0-3: 实时数据采集
            "ai_model_inference": [4, 5, 6, 7],             # 核心4-7: AI模型推理
            "ai_model_training": [8, 9, 10, 11],            # 核心8-11: AI模型训练
            "risk_order_management": [12, 13, 14, 15],      # 核心12-15: 风险控制和订单管理
            "web_monitoring": [16, 17, 18, 19]              # 核心16-19: Web服务和监控
        }
        
        # GPU显存分配策略 (12GB)
        gpu_allocations = {
            "reinforcement_learning": 4096,      # 4GB: 强化学习模型
            "time_series_deep": 3072,           # 3GB: 时序深度学习
            "ensemble_learning": 2048,          # 2GB: 集成学习
            "meta_learning": 2048,              # 2GB: 元学习
            "gpu_cache": 896                    # 896MB: GPU缓存和临时计算
        }
        
        # 系统内存分配策略 (128GB)
        memory_allocations = {
            "real_time_data_cache": 32 * 1024,     # 32GB: 实时数据缓存
            "ai_model_parameters": 24 * 1024,      # 24GB: AI模型参数
            "historical_data_cache": 16 * 1024,    # 16GB: 历史数据缓存
            "system_web_services": 16 * 1024,      # 16GB: 系统和Web服务
            "buffer_reserve": 40 * 1024             # 40GB: 预留缓冲区
        }
        
        # 创建资源分配配置
        allocations = [
            # 实时数据采集 - 最高优先级
            ResourceAllocation(
                cpu_cores=cpu_allocations["real_time_data_collection"],
                gpu_memory_mb=0,
                system_memory_mb=memory_allocations["real_time_data_cache"],
                disk_space_gb=50,
                priority=TaskPriority.CRITICAL,
                task_name="real_time_data_collection",
                max_threads=4
            ),
            
            # AI模型推理 - 高优先级
            ResourceAllocation(
                cpu_cores=cpu_allocations["ai_model_inference"],
                gpu_memory_mb=gpu_allocations["reinforcement_learning"],
                system_memory_mb=memory_allocations["ai_model_parameters"] // 2,
                disk_space_gb=20,
                priority=TaskPriority.HIGH,
                task_name="ai_model_inference",
                max_threads=4
            ),
            
            # AI模型训练 - 中优先级
            ResourceAllocation(
                cpu_cores=cpu_allocations["ai_model_training"],
                gpu_memory_mb=gpu_allocations["time_series_deep"] + gpu_allocations["ensemble_learning"],
                system_memory_mb=memory_allocations["ai_model_parameters"] // 2,
                disk_space_gb=100,
                priority=TaskPriority.MEDIUM,
                task_name="ai_model_training",
                max_threads=4
            ),
            
            # 风险控制和订单管理 - 最高优先级
            ResourceAllocation(
                cpu_cores=cpu_allocations["risk_order_management"],
                gpu_memory_mb=0,
                system_memory_mb=8 * 1024,  # 8GB
                disk_space_gb=30,
                priority=TaskPriority.CRITICAL,
                task_name="risk_order_management",
                max_threads=4
            ),
            
            # Web服务和监控 - 低优先级
            ResourceAllocation(
                cpu_cores=cpu_allocations["web_monitoring"],
                gpu_memory_mb=gpu_allocations["gpu_cache"],
                system_memory_mb=memory_allocations["system_web_services"],
                disk_space_gb=50,
                priority=TaskPriority.LOW,
                task_name="web_monitoring",
                max_threads=4
            )
        ]
        
        # 注册资源分配
        for allocation in allocations:
            self.resource_allocations[allocation.task_name] = allocation
            
        logger.info("✅ 资源分配策略初始化完成")
        self._log_resource_allocation()

    def _log_resource_allocation(self):
        """记录资源分配情况"""
        logger.info("📊 资源分配详情:")
        
        for task_name, allocation in self.resource_allocations.items():
            logger.info(f"  🔧 {task_name}:")
            logger.info(f"    CPU核心: {allocation.cpu_cores}")
            logger.info(f"    GPU显存: {allocation.gpu_memory_mb}MB")
            logger.info(f"    系统内存: {allocation.system_memory_mb//1024}GB")
            logger.info(f"    磁盘空间: {allocation.disk_space_gb}GB")
            logger.info(f"    优先级: {allocation.priority.name}")
            logger.info(f"    最大线程: {allocation.max_threads}")

    def allocate_cpu_cores(self, task_name: str) -> List[int]:
        """分配CPU核心"""
        with self.resource_lock:
            if task_name in self.resource_allocations:
                cores = self.resource_allocations[task_name].cpu_cores
                
                # 设置CPU亲和性
                try:
                    os.sched_setaffinity(0, cores)
                    logger.info(f"✅ 为任务 {task_name} 分配CPU核心: {cores}")
                    return cores
                except Exception as e:
                    logger.warning(f"⚠️ 设置CPU亲和性失败: {e}")
                    return cores
            else:
                logger.error(f"❌ 未找到任务 {task_name} 的资源分配配置")
                return []

    def allocate_gpu_memory(self, task_name: str) -> Optional[int]:
        """分配GPU显存"""
        with self.resource_lock:
            if task_name in self.resource_allocations:
                gpu_memory = self.resource_allocations[task_name].gpu_memory_mb
                
                if gpu_memory > 0:
                    try:
                        # 设置PyTorch GPU内存限制
                        if torch.cuda.is_available():
                            torch.cuda.set_per_process_memory_fraction(gpu_memory / (12 * 1024))
                            logger.info(f"✅ 为任务 {task_name} 分配GPU显存: {gpu_memory}MB")
                            return gpu_memory
                    except Exception as e:
                        logger.warning(f"⚠️ 设置GPU内存限制失败: {e}")
                
                return gpu_memory
            else:
                logger.error(f"❌ 未找到任务 {task_name} 的资源分配配置")
                return None

    def allocate_system_memory(self, task_name: str) -> Optional[int]:
        """分配系统内存"""
        with self.resource_lock:
            if task_name in self.resource_allocations:
                memory_mb = self.resource_allocations[task_name].system_memory_mb
                logger.info(f"✅ 为任务 {task_name} 分配系统内存: {memory_mb//1024}GB")
                return memory_mb
            else:
                logger.error(f"❌ 未找到任务 {task_name} 的资源分配配置")
                return None

    def get_thread_pool(self, task_name: str) -> ThreadPoolExecutor:
        """获取线程池"""
        if task_name in self.resource_allocations:
            max_workers = self.resource_allocations[task_name].max_threads
            return ThreadPoolExecutor(max_workers=max_workers)
        else:
            return self.thread_pool

    def get_process_pool(self) -> ProcessPoolExecutor:
        """获取进程池"""
        return self.process_pool

    async def start_task(self, task_name: str, task_func, *args, **kwargs):
        """启动任务并分配资源"""
        with self.resource_lock:
            if task_name in self.active_tasks:
                logger.warning(f"⚠️ 任务 {task_name} 已在运行")
                return
            
            # 分配资源
            cpu_cores = self.allocate_cpu_cores(task_name)
            gpu_memory = self.allocate_gpu_memory(task_name)
            system_memory = self.allocate_system_memory(task_name)
            
            # 启动任务
            try:
                task = asyncio.create_task(task_func(*args, **kwargs))
                self.active_tasks[task_name] = {
                    'task': task,
                    'start_time': time.time(),
                    'cpu_cores': cpu_cores,
                    'gpu_memory': gpu_memory,
                    'system_memory': system_memory
                }
                
                logger.info(f"🚀 任务 {task_name} 启动成功")
                return task
                
            except Exception as e:
                logger.error(f"❌ 启动任务 {task_name} 失败: {e}")
                return None

    async def stop_task(self, task_name: str):
        """停止任务并释放资源"""
        with self.resource_lock:
            if task_name in self.active_tasks:
                task_info = self.active_tasks[task_name]
                task = task_info['task']
                
                # 取消任务
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # 释放资源
                del self.active_tasks[task_name]
                
                run_time = time.time() - task_info['start_time']
                logger.info(f"🛑 任务 {task_name} 已停止，运行时间: {run_time:.2f}秒")
            else:
                logger.warning(f"⚠️ 任务 {task_name} 未在运行")

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return self.resource_monitor.get_current_usage()

    def optimize_performance(self):
        """性能优化"""
        logger.info("🔧 开始性能优化...")
        
        # CPU优化
        self._optimize_cpu()
        
        # GPU优化
        self._optimize_gpu()
        
        # 内存优化
        self._optimize_memory()
        
        # 磁盘优化
        self._optimize_disk()
        
        logger.info("✅ 性能优化完成")

    def _optimize_cpu(self):
        """CPU优化"""
        try:
            # 设置CPU调度策略
            os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            logger.info("✅ CPU调度策略设置为性能模式")
        except Exception as e:
            logger.warning(f"⚠️ CPU优化失败: {e}")

    def _optimize_gpu(self):
        """GPU优化"""
        try:
            if torch.cuda.is_available():
                # 启用GPU性能模式
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ GPU性能优化完成")
        except Exception as e:
            logger.warning(f"⚠️ GPU优化失败: {e}")

    def _optimize_memory(self):
        """内存优化"""
        try:
            # 设置内存交换策略
            os.system("echo 10 | sudo tee /proc/sys/vm/swappiness")
            logger.info("✅ 内存交换策略优化完成")
        except Exception as e:
            logger.warning(f"⚠️ 内存优化失败: {e}")

    def _optimize_disk(self):
        """磁盘优化"""
        try:
            # 设置磁盘调度器
            os.system("echo mq-deadline | sudo tee /sys/block/nvme*/queue/scheduler")
            logger.info("✅ NVMe SSD调度器优化完成")
        except Exception as e:
            logger.warning(f"⚠️ 磁盘优化失败: {e}")


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: int = 5):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("📊 资源监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("📊 资源监控已停止")

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                usage = self.get_current_usage()
                self._log_usage(usage)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"❌ 资源监控异常: {e}")
                time.sleep(interval)

    def get_current_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_avg = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # GPU使用情况
            gpu_usage = self._get_gpu_usage()
            
            return {
                'timestamp': time.time(),
                'cpu': {
                    'per_core': cpu_percent,
                    'average': cpu_avg,
                    'core_count': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu': gpu_usage
            }
            
        except Exception as e:
            logger.error(f"❌ 获取资源使用情况失败: {e}")
            return {}

    def _get_gpu_usage(self) -> Dict[str, Any]:
        """获取GPU使用情况"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 假设只有一个GPU
                return {
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_percent': gpu.load * 100,
                    'temperature': gpu.temperature
                }
            else:
                return {'error': 'No GPU found'}
        except Exception as e:
            return {'error': str(e)}

    def _log_usage(self, usage: Dict[str, Any]):
        """记录使用情况"""
        if not usage:
            return
            
        cpu_avg = usage.get('cpu', {}).get('average', 0)
        memory_percent = usage.get('memory', {}).get('percent', 0)
        disk_percent = usage.get('disk', {}).get('percent', 0)
        gpu_info = usage.get('gpu', {})
        
        logger.info(f"📊 资源使用: CPU {cpu_avg:.1f}%, 内存 {memory_percent:.1f}%, 磁盘 {disk_percent:.1f}%")
        
        if 'memory_percent' in gpu_info:
            logger.info(f"🎮 GPU使用: {gpu_info['gpu_percent']:.1f}%, 显存 {gpu_info['memory_percent']:.1f}%")


# 全局资源管理器实例
_resource_manager = None

def get_resource_manager() -> ProductionResourceManager:
    """获取全局资源管理器实例"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ProductionResourceManager()
    return _resource_manager

def initialize_production_resources():
    """初始化生产环境资源"""
    manager = get_resource_manager()
    manager.optimize_performance()
    manager.resource_monitor.start_monitoring()
    return manager

if __name__ == "__main__":
    # 测试资源管理器
    async def test_resource_manager():
        manager = initialize_production_resources()
        
        # 测试资源分配
        cpu_cores = manager.allocate_cpu_cores("ai_model_inference")
        gpu_memory = manager.allocate_gpu_memory("ai_model_inference")
        system_memory = manager.allocate_system_memory("ai_model_inference")
        
        print(f"分配的CPU核心: {cpu_cores}")
        print(f"分配的GPU显存: {gpu_memory}MB")
        print(f"分配的系统内存: {system_memory//1024}GB")
        
        # 监控资源使用
        await asyncio.sleep(10)
        
        usage = manager.get_resource_usage()
        print(f"当前资源使用情况: {usage}")
        
        manager.resource_monitor.stop_monitoring()

    asyncio.run(test_resource_manager())
