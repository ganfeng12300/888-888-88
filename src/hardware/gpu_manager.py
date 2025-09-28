"""
🎮 GPU显存优化管理器
生产级RTX3060 12GB显存精确分配和性能优化系统
实现GPU显存池管理、动态分配和AI模型优化
"""

import asyncio
import time
import threading
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os

try:
    import torch
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from loguru import logger


class GPUTaskType(Enum):
    """GPU任务类型"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES_DEEP = "time_series_deep"
    INFERENCE_CACHE = "inference_cache"
    MODEL_TRAINING = "model_training"
    DATA_PREPROCESSING = "data_preprocessing"


@dataclass
class GPUMemoryAllocation:
    """GPU显存分配配置"""
    task_type: GPUTaskType
    memory_gb: float
    priority: int  # 优先级 (1-10, 10最高)
    description: str
    allocated_memory: int = 0  # 已分配的显存 (bytes)
    active_models: List[str] = field(default_factory=list)


@dataclass
class GPUMetrics:
    """GPU性能指标"""
    timestamp: float
    gpu_id: int
    usage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_usage_percent: float
    temperature: float
    power_usage: float
    fan_speed: int
    clock_graphics: int
    clock_memory: int


class GPUMemoryManager:
    """GPU显存管理器"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.gpu_available = GPU_AVAILABLE
        self.monitoring = False
        
        if not self.gpu_available:
            logger.error("GPU不可用，请安装PyTorch和pynvml")
            return
        
        # 初始化GPU
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
            
            # 获取GPU内存信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_memory = memory_info.total
            self.total_memory_gb = self.total_memory / (1024**3)
            
            logger.info(f"GPU管理器初始化完成: {self.gpu_name}, 总显存: {self.total_memory_gb:.1f}GB")
            
        except Exception as e:
            logger.error(f"GPU初始化失败: {e}")
            self.gpu_available = False
            return
        
        # 显存分配策略 (基于RTX3060 12GB)
        self.memory_allocations = {
            GPUTaskType.REINFORCEMENT_LEARNING: GPUMemoryAllocation(
                task_type=GPUTaskType.REINFORCEMENT_LEARNING,
                memory_gb=6.0,  # 6GB用于强化学习
                priority=9,
                description="强化学习AI模型(PPO/SAC)"
            ),
            GPUTaskType.TIME_SERIES_DEEP: GPUMemoryAllocation(
                task_type=GPUTaskType.TIME_SERIES_DEEP,
                memory_gb=4.0,  # 4GB用于时序深度学习
                priority=8,
                description="时序深度学习AI(Transformer/LSTM)"
            ),
            GPUTaskType.INFERENCE_CACHE: GPUMemoryAllocation(
                task_type=GPUTaskType.INFERENCE_CACHE,
                memory_gb=1.5,  # 1.5GB用于推理缓存
                priority=7,
                description="模型推理和预测缓存"
            ),
            GPUTaskType.MODEL_TRAINING: GPUMemoryAllocation(
                task_type=GPUTaskType.MODEL_TRAINING,
                memory_gb=0.5,  # 0.5GB用于其他模型训练
                priority=5,
                description="其他AI模型训练"
            ),
        }
        
        # 性能监控数据
        self.metrics_history: List[GPUMetrics] = []
        self.max_history_size = 3600  # 1小时历史数据
        
        # 内存池管理
        self.memory_pool: Dict[str, torch.Tensor] = {}
        self.memory_lock = threading.Lock()
        
        # PyTorch GPU设置
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            # 启用显存增长模式
            torch.cuda.set_per_process_memory_fraction(0.95)
            # 启用内存池
            torch.cuda.empty_cache()
            logger.info("PyTorch GPU设置完成")
    
    def allocate_memory(self, task_type: GPUTaskType, model_name: str, 
                       memory_size_mb: Optional[int] = None) -> Optional[torch.Tensor]:
        """分配GPU显存"""
        if not self.gpu_available:
            return None
        
        try:
            allocation = self.memory_allocations.get(task_type)
            if not allocation:
                logger.error(f"未知的任务类型: {task_type}")
                return None
            
            # 计算需要分配的显存大小
            if memory_size_mb is None:
                memory_size_mb = int(allocation.memory_gb * 1024)
            
            memory_size_bytes = memory_size_mb * 1024 * 1024
            
            # 检查是否有足够的显存
            if allocation.allocated_memory + memory_size_bytes > allocation.memory_gb * (1024**3):
                logger.warning(f"任务类型 {task_type.value} 显存不足")
                return None
            
            with self.memory_lock:
                # 分配显存
                memory_tensor = torch.empty(memory_size_bytes // 4, dtype=torch.float32, device=f'cuda:{self.gpu_id}')
                
                # 记录分配
                allocation.allocated_memory += memory_size_bytes
                allocation.active_models.append(model_name)
                self.memory_pool[model_name] = memory_tensor
            
            logger.info(f"为 {model_name} 分配了 {memory_size_mb}MB 显存 ({task_type.value})")
            return memory_tensor
            
        except Exception as e:
            logger.error(f"分配显存失败: {e}")
            return None
    
    def deallocate_memory(self, model_name: str) -> bool:
        """释放GPU显存"""
        if not self.gpu_available:
            return False
        
        try:
            with self.memory_lock:
                if model_name not in self.memory_pool:
                    return False
                
                # 获取显存大小
                memory_tensor = self.memory_pool[model_name]
                memory_size_bytes = memory_tensor.numel() * 4  # float32 = 4 bytes
                
                # 找到对应的分配记录
                for allocation in self.memory_allocations.values():
                    if model_name in allocation.active_models:
                        allocation.allocated_memory -= memory_size_bytes
                        allocation.active_models.remove(model_name)
                        break
                
                # 释放显存
                del self.memory_pool[model_name]
                del memory_tensor
                torch.cuda.empty_cache()
            
            logger.info(f"已释放 {model_name} 的显存")
            return True
            
        except Exception as e:
            logger.error(f"释放显存失败: {e}")
            return False
    
    async def start_performance_monitoring(self, interval: float = 1.0):
        """启动GPU性能监控"""
        if not self.gpu_available:
            logger.warning("GPU不可用，无法启动性能监控")
            return
        
        self.monitoring = True
        logger.info("开始GPU性能监控...")
        
        while self.monitoring:
            try:
                metrics = await self._collect_gpu_metrics()
                self._store_metrics(metrics)
                
                # 检查性能问题
                await self._check_performance_issues(metrics)
                
                # 动态优化
                await self._optimize_gpu_performance(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"GPU性能监控出错: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_gpu_metrics(self) -> GPUMetrics:
        """收集GPU性能指标"""
        timestamp = time.time()
        
        try:
            # GPU使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            usage_percent = utilization.gpu
            
            # GPU内存
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used_gb = memory_info.used / (1024**3)
            memory_total_gb = memory_info.total / (1024**3)
            memory_usage_percent = (memory_info.used / memory_info.total) * 100
            
            # GPU温度
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # GPU功耗
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            except:
                power_usage = 0
            
            # 风扇转速
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.handle)
            except:
                fan_speed = 0
            
            # GPU时钟频率
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_memory = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            except:
                clock_graphics = 0
                clock_memory = 0
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_id=self.gpu_id,
                usage_percent=usage_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                memory_usage_percent=memory_usage_percent,
                temperature=temperature,
                power_usage=power_usage,
                fan_speed=fan_speed,
                clock_graphics=clock_graphics,
                clock_memory=clock_memory
            )
            
        except Exception as e:
            logger.error(f"收集GPU指标失败: {e}")
            return GPUMetrics(
                timestamp=timestamp,
                gpu_id=self.gpu_id,
                usage_percent=0,
                memory_used_gb=0,
                memory_total_gb=self.total_memory_gb,
                memory_usage_percent=0,
                temperature=0,
                power_usage=0,
                fan_speed=0,
                clock_graphics=0,
                clock_memory=0
            )
    
    def _store_metrics(self, metrics: GPUMetrics):
        """存储性能指标"""
        self.metrics_history.append(metrics)
        
        # 限制历史数据大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    async def _check_performance_issues(self, metrics: GPUMetrics):
        """检查性能问题"""
        issues = []
        
        # 检查GPU使用率
        if metrics.usage_percent > 98:
            issues.append(f"GPU使用率过高: {metrics.usage_percent}%")
        elif metrics.usage_percent < 50:
            issues.append(f"GPU使用率偏低: {metrics.usage_percent}%")
        
        # 检查显存使用率
        if metrics.memory_usage_percent > 95:
            issues.append(f"GPU显存使用率过高: {metrics.memory_usage_percent:.1f}%")
        
        # 检查温度
        if metrics.temperature > 80:
            issues.append(f"GPU温度过高: {metrics.temperature}°C")
        elif metrics.temperature > 75:
            issues.append(f"GPU温度警告: {metrics.temperature}°C")
        
        # 检查功耗
        if metrics.power_usage > 170:  # RTX3060额定功耗170W
            issues.append(f"GPU功耗过高: {metrics.power_usage:.1f}W")
        
        if issues:
            logger.warning(f"GPU性能问题: {'; '.join(issues)}")
    
    async def _optimize_gpu_performance(self, metrics: GPUMetrics):
        """动态优化GPU性能"""
        try:
            # 温度控制
            if metrics.temperature > 75:
                await self._reduce_gpu_power_limit()
            elif metrics.temperature < 65 and metrics.usage_percent > 90:
                await self._increase_gpu_power_limit()
            
            # 显存优化
            if metrics.memory_usage_percent > 90:
                await self._optimize_memory_usage()
            
            # 时钟频率优化
            if metrics.usage_percent > 95 and metrics.temperature < 70:
                await self._increase_gpu_clocks()
            elif metrics.temperature > 80:
                await self._reduce_gpu_clocks()
                
        except Exception as e:
            logger.error(f"GPU性能优化失败: {e}")
    
    async def _reduce_gpu_power_limit(self):
        """降低GPU功耗限制"""
        try:
            subprocess.run(['nvidia-smi', '-pl', '150'], capture_output=True, check=False)
            logger.info("已降低GPU功耗限制到150W")
        except Exception as e:
            logger.debug(f"降低GPU功耗失败: {e}")
    
    async def _increase_gpu_power_limit(self):
        """提高GPU功耗限制"""
        try:
            subprocess.run(['nvidia-smi', '-pl', '170'], capture_output=True, check=False)
            logger.info("已提高GPU功耗限制到170W")
        except Exception as e:
            logger.debug(f"提高GPU功耗失败: {e}")
    
    async def _optimize_memory_usage(self):
        """优化显存使用"""
        try:
            # 清理PyTorch缓存
            torch.cuda.empty_cache()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            logger.info("已优化GPU显存使用")
            
        except Exception as e:
            logger.error(f"优化显存使用失败: {e}")
    
    async def _increase_gpu_clocks(self):
        """提高GPU时钟频率"""
        try:
            # 使用nvidia-smi提高时钟频率
            subprocess.run(['nvidia-smi', '-ac', '6001,1890'], capture_output=True, check=False)
            logger.info("已提高GPU时钟频率")
        except Exception as e:
            logger.debug(f"提高GPU时钟频率失败: {e}")
    
    async def _reduce_gpu_clocks(self):
        """降低GPU时钟频率"""
        try:
            subprocess.run(['nvidia-smi', '-ac', '5001,1590'], capture_output=True, check=False)
            logger.info("已降低GPU时钟频率")
        except Exception as e:
            logger.debug(f"降低GPU时钟频率失败: {e}")
    
    def get_memory_allocation_status(self) -> Dict[str, Any]:
        """获取显存分配状态"""
        status = {}
        
        for task_type, allocation in self.memory_allocations.items():
            allocated_gb = allocation.allocated_memory / (1024**3)
            utilization = (allocated_gb / allocation.memory_gb) * 100 if allocation.memory_gb > 0 else 0
            
            status[task_type.value] = {
                "allocated_gb": round(allocated_gb, 2),
                "total_gb": allocation.memory_gb,
                "utilization_percent": round(utilization, 1),
                "active_models": len(allocation.active_models),
                "model_names": allocation.active_models.copy(),
                "description": allocation.description
            }
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-60:]  # 最近1分钟
        
        avg_usage = sum(m.usage_percent for m in recent_metrics) / len(recent_metrics)
        max_usage = max(m.usage_percent for m in recent_metrics)
        avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
        max_memory = max(m.memory_usage_percent for m in recent_metrics)
        avg_temp = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
        max_temp = max(m.temperature for m in recent_metrics)
        avg_power = sum(m.power_usage for m in recent_metrics) / len(recent_metrics)
        max_power = max(m.power_usage for m in recent_metrics)
        
        return {
            "gpu_name": self.gpu_name,
            "total_memory_gb": self.total_memory_gb,
            "average_usage_1min": round(avg_usage, 1),
            "max_usage_1min": round(max_usage, 1),
            "average_memory_usage": round(avg_memory, 1),
            "max_memory_usage": round(max_memory, 1),
            "average_temperature": round(avg_temp, 1),
            "max_temperature": round(max_temp, 1),
            "average_power": round(avg_power, 1),
            "max_power": round(max_power, 1),
            "memory_allocations": self.get_memory_allocation_status()
        }
    
    def optimize_for_ai_training(self):
        """为AI训练优化GPU设置"""
        try:
            logger.info("开始为AI训练优化GPU设置...")
            
            # 设置PyTorch优化
            if torch.cuda.is_available():
                # 启用cudnn基准模式
                torch.backends.cudnn.benchmark = True
                
                # 启用混合精度训练
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # 设置显存分配策略
                torch.cuda.set_per_process_memory_fraction(0.95)
                
                logger.info("PyTorch GPU优化完成")
            
            # 设置GPU性能模式
            subprocess.run(['nvidia-smi', '-pm', '1'], capture_output=True, check=False)  # 持久模式
            subprocess.run(['nvidia-smi', '-pl', '170'], capture_output=True, check=False)  # 最大功耗
            
            logger.info("GPU AI训练优化完成")
            
        except Exception as e:
            logger.error(f"GPU AI训练优化失败: {e}")
    
    def create_memory_pool(self, pool_name: str, size_gb: float) -> bool:
        """创建显存池"""
        try:
            size_bytes = int(size_gb * 1024**3)
            memory_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=f'cuda:{self.gpu_id}')
            
            with self.memory_lock:
                self.memory_pool[f"pool_{pool_name}"] = memory_tensor
            
            logger.info(f"创建显存池 {pool_name}: {size_gb}GB")
            return True
            
        except Exception as e:
            logger.error(f"创建显存池失败: {e}")
            return False
    
    def get_available_memory(self) -> float:
        """获取可用显存 (GB)"""
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return memory_info.free / (1024**3)
        except:
            return 0.0
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        logger.info("GPU性能监控已停止")
    
    def cleanup_memory(self):
        """清理所有显存"""
        try:
            with self.memory_lock:
                for model_name in list(self.memory_pool.keys()):
                    del self.memory_pool[model_name]
                
                # 重置分配记录
                for allocation in self.memory_allocations.values():
                    allocation.allocated_memory = 0
                    allocation.active_models.clear()
            
            torch.cuda.empty_cache()
            logger.info("已清理所有GPU显存")
            
        except Exception as e:
            logger.error(f"清理GPU显存失败: {e}")


# 全局GPU管理器实例
gpu_manager = GPUMemoryManager()


def allocate_gpu_memory(task_type: GPUTaskType, model_name: str, 
                       memory_size_mb: Optional[int] = None) -> Optional[torch.Tensor]:
    """分配GPU显存"""
    return gpu_manager.allocate_memory(task_type, model_name, memory_size_mb)


def deallocate_gpu_memory(model_name: str) -> bool:
    """释放GPU显存"""
    return gpu_manager.deallocate_memory(model_name)


async def main():
    """测试主函数"""
    logger.info("启动GPU管理器测试...")
    
    if not gpu_manager.gpu_available:
        logger.error("GPU不可用，无法进行测试")
        return
    
    # 优化GPU设置
    gpu_manager.optimize_for_ai_training()
    
    # 启动性能监控
    monitor_task = asyncio.create_task(gpu_manager.start_performance_monitoring())
    
    try:
        # 测试显存分配
        memory1 = allocate_gpu_memory(GPUTaskType.REINFORCEMENT_LEARNING, "test_model_1", 1024)  # 1GB
        memory2 = allocate_gpu_memory(GPUTaskType.TIME_SERIES_DEEP, "test_model_2", 512)  # 512MB
        
        # 运行30秒测试
        await asyncio.sleep(30)
        
        # 获取性能摘要
        summary = gpu_manager.get_performance_summary()
        logger.info(f"GPU性能摘要: {json.dumps(summary, indent=2)}")
        
        # 清理显存
        deallocate_gpu_memory("test_model_1")
        deallocate_gpu_memory("test_model_2")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        gpu_manager.stop_monitoring()
        monitor_task.cancel()
        gpu_manager.cleanup_memory()


if __name__ == "__main__":
    asyncio.run(main())
