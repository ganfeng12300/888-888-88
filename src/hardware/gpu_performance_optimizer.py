#!/usr/bin/env python3
"""
🚀 GPU性能优化器 - 硬件加速系统
GPU Performance Optimizer - Hardware Acceleration System

专为20核CPU + GTX3060 12G配置优化
- GPU加速深度学习训练
- 内存管理优化
- 并行计算调度
- 性能监控和调优
"""

import os
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，GPU加速功能将被禁用")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy未安装，部分GPU计算功能将被禁用")


@dataclass
class GPUStatus:
    """GPU状态信息"""
    device_id: int
    name: str
    total_memory: int
    used_memory: int
    free_memory: int
    utilization: float
    temperature: float
    power_usage: float
    is_available: bool


@dataclass
class CPUStatus:
    """CPU状态信息"""
    core_count: int
    thread_count: int
    usage_percent: float
    frequency: float
    temperature: float
    load_average: List[float]


@dataclass
class PerformanceMetrics:
    """性能指标"""
    gpu_status: GPUStatus
    cpu_status: CPUStatus
    memory_usage: Dict[str, float]
    disk_usage: Dict[str, float]
    network_usage: Dict[str, float]
    timestamp: datetime


class GPUPerformanceOptimizer:
    """GPU性能优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化GPU性能优化器"""
        self.config = config or {}
        self.is_running = False
        self.performance_history = []
        self.optimization_tasks = []
        self.lock = threading.Lock()
        
        # 系统配置
        self.target_gpu_utilization = self.config.get('target_gpu_utilization', 85.0)
        self.max_memory_usage = self.config.get('max_memory_usage', 90.0)
        self.monitoring_interval = self.config.get('monitoring_interval', 5)
        self.optimization_interval = self.config.get('optimization_interval', 30)
        
        # 初始化硬件检测
        self._initialize_hardware()
        
        # 启动监控线程
        self._start_monitoring()
        
        logger.info("🚀 GPU性能优化器初始化完成")
    
    def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化"""
        try:
            optimization_results = {}
            
            # GPU优化
            if self.gpu_available:
                gpu_optimization = self._optimize_gpu_performance()
                optimization_results['gpu'] = gpu_optimization
            
            # CPU优化
            cpu_optimization = self._optimize_cpu_performance()
            optimization_results['cpu'] = cpu_optimization
            
            # 内存优化
            memory_optimization = self._optimize_memory_usage()
            optimization_results['memory'] = memory_optimization
            
            return {
                'status': 'success',
                'optimizations': optimization_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"性能优化失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _optimize_gpu_performance(self) -> Dict[str, Any]:
        """优化GPU性能"""
        try:
            if not self.gpu_available:
                return {'status': 'skipped', 'reason': 'GPU不可用'}
            
            # 清理GPU缓存
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            
            # 获取当前GPU状态
            gpu_status = self.get_gpu_status()
            
            # 动态调整GPU内存分配
            if gpu_status and gpu_status.utilization > 90:
                # GPU使用率过高，清理缓存
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            return {
                'status': 'success',
                'actions': ['cache_cleared', 'memory_optimized'],
                'gpu_utilization': gpu_status.utilization if gpu_status else 0
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _optimize_cpu_performance(self) -> Dict[str, Any]:
        """优化CPU性能"""
        try:
            # 获取CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 动态调整线程数
            if cpu_usage > 80:
                # CPU使用率过高，减少线程数
                new_threads = max(1, self.cpu_threads // 2)
            else:
                # CPU使用率正常，使用最优线程数
                new_threads = min(self.cpu_threads, 32)
            
            if TORCH_AVAILABLE:
                torch.set_num_threads(new_threads)
            
            return {
                'status': 'success',
                'actions': ['thread_count_adjusted'],
                'cpu_usage': cpu_usage,
                'thread_count': new_threads
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            # 获取内存信息
            memory = psutil.virtual_memory()
            
            # 如果内存使用率过高，触发垃圾回收
            if memory.percent > 85:
                import gc
                gc.collect()
                
                # 如果有GPU，也清理GPU内存
                if self.gpu_available and TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            
            return {
                'status': 'success',
                'actions': ['garbage_collection', 'gpu_cache_cleared'] if memory.percent > 85 else ['monitoring'],
                'memory_usage_percent': memory.percent,
                'available_memory_gb': memory.available / (1024**3)
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        try:
            # 检查GPU状态
            if self.gpu_available:
                gpu_status = self.get_gpu_status()
                if gpu_status:
                    if gpu_status.utilization < 30:
                        suggestions.append("GPU利用率较低，考虑增加批处理大小")
                    elif gpu_status.utilization > 90:
                        suggestions.append("GPU利用率过高，考虑减少批处理大小或清理缓存")
                    
                    if gpu_status.free_memory < gpu_status.total_memory * 0.1:
                        suggestions.append("GPU内存不足，建议清理缓存或减少模型大小")
            else:
                suggestions.append("未检测到GPU，建议安装CUDA和PyTorch以启用GPU加速")
            
            # 检查CPU状态
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 80:
                suggestions.append("CPU使用率过高，考虑优化算法或增加硬件资源")
            elif cpu_usage < 20:
                suggestions.append("CPU利用率较低，可以增加并行处理任务")
            
            # 检查内存状态
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                suggestions.append("内存使用率过高，建议增加内存或优化内存使用")
            
            # 检查PyTorch和CuPy安装
            if not TORCH_AVAILABLE:
                suggestions.append("建议安装PyTorch以获得更好的深度学习性能")
            if not CUPY_AVAILABLE:
                suggestions.append("建议安装CuPy以获得更好的GPU数值计算性能")
            
        except Exception as e:
            logger.error(f"生成优化建议失败: {str(e)}")
            suggestions.append("系统状态检查异常，建议重启优化器")
        
        return suggestions if suggestions else ["系统运行状态良好，无需特殊优化"]
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """优化GPU内存"""
        try:
            if not self.gpu_available:
                return {
                    'status': 'skipped',
                    'reason': 'GPU不可用',
                    'recommendations': ['安装CUDA和PyTorch以启用GPU功能']
                }
            
            initial_memory = torch.cuda.memory_allocated() if TORCH_AVAILABLE else 0
            
            # 清理GPU缓存
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            final_memory = torch.cuda.memory_allocated() if TORCH_AVAILABLE else 0
            freed_memory = initial_memory - final_memory
            
            return {
                'status': 'success',
                'initial_memory_mb': initial_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2),
                'freed_memory_mb': freed_memory / (1024**2),
                'actions': ['cache_cleared', 'memory_synchronized']
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'recommendations': ['检查GPU驱动和CUDA安装']
            }
    
    def _initialize_hardware(self):
        """初始化硬件检测"""
        # 检测GPU
        self.gpu_available = False
        self.gpu_devices = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_available = True
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                self.gpu_devices.append({
                    'id': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                })
            
            logger.info(f"检测到 {device_count} 个GPU设备: {[d['name'] for d in self.gpu_devices]}")
        else:
            logger.warning("未检测到可用的GPU设备")
        
        # 检测CPU
        self.cpu_count = psutil.cpu_count()
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        logger.info(f"检测到CPU: {self.cpu_count}核心 {self.cpu_threads}线程")
        
        # 设置优化参数
        self._configure_optimization()
    
    def _configure_optimization(self):
        """配置优化参数"""
        if self.gpu_available:
            # GPU优化配置
            torch.backends.cudnn.benchmark = True  # 优化卷积性能
            torch.backends.cudnn.deterministic = False  # 允许非确定性算法
            
            # 设置GPU内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            logger.info("GPU优化配置已启用")
        
        # CPU优化配置
        if self.cpu_threads >= 16:  # 20核CPU通常有40线程
            # 设置线程数
            torch.set_num_threads(min(self.cpu_threads, 32))
            os.environ['OMP_NUM_THREADS'] = str(min(self.cpu_threads, 32))
            os.environ['MKL_NUM_THREADS'] = str(min(self.cpu_threads, 32))
            
            logger.info(f"CPU多线程优化已启用: {min(self.cpu_threads, 32)}线程")
    
    def _start_monitoring(self):
        """启动性能监控"""
        self.is_running = True
        
        # 性能监控线程
        threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True,
            name="PerformanceMonitorThread"
        ).start()
        
        # 优化调度线程
        threading.Thread(
            target=self._optimization_loop,
            daemon=True,
            name="OptimizationThread"
        ).start()
    
    def get_gpu_status(self) -> Optional[GPUStatus]:
        """获取GPU状态"""
        if not self.gpu_available:
            return None
        
        try:
            device_id = 0  # 使用第一个GPU
            
            # 获取内存信息
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            cached_memory = torch.cuda.memory_reserved(device_id)
            free_memory = total_memory - cached_memory
            
            # 获取利用率（简化版本）
            utilization = (allocated_memory / total_memory) * 100
            
            return GPUStatus(
                device_id=device_id,
                name=self.gpu_devices[device_id]['name'],
                total_memory=total_memory,
                used_memory=allocated_memory,
                free_memory=free_memory,
                utilization=utilization,
                temperature=0.0,  # 需要nvidia-ml-py库获取
                power_usage=0.0,  # 需要nvidia-ml-py库获取
                is_available=True
            )
            
        except Exception as e:
            logger.error(f"获取GPU状态失败: {e}")
            return None
    
    def get_cpu_status(self) -> CPUStatus:
        """获取CPU状态"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            current_freq = cpu_freq.current if cpu_freq else 0.0
            
            # 负载平均值
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            return CPUStatus(
                core_count=self.cpu_count,
                thread_count=self.cpu_threads,
                usage_percent=cpu_percent,
                frequency=current_freq,
                temperature=0.0,  # 需要额外库获取
                load_average=list(load_avg)
            )
            
        except Exception as e:
            logger.error(f"获取CPU状态失败: {e}")
            return CPUStatus(
                core_count=self.cpu_count,
                thread_count=self.cpu_threads,
                usage_percent=0.0,
                frequency=0.0,
                temperature=0.0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def get_memory_status(self) -> Dict[str, float]:
        """获取内存状态"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent
            }
            
        except Exception as e:
            logger.error(f"获取内存状态失败: {e}")
            return {}
    
    def get_disk_status(self) -> Dict[str, float]:
        """获取磁盘状态"""
        try:
            disk = psutil.disk_usage('/')
            
            return {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            }
            
        except Exception as e:
            logger.error(f"获取磁盘状态失败: {e}")
            return {}
    
    def get_network_status(self) -> Dict[str, float]:
        """获取网络状态"""
        try:
            net_io = psutil.net_io_counters()
            
            return {
                'bytes_sent_mb': net_io.bytes_sent / (1024**2),
                'bytes_recv_mb': net_io.bytes_recv / (1024**2),
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errors_in': net_io.errin,
                'errors_out': net_io.errout
            }
            
        except Exception as e:
            logger.error(f"获取网络状态失败: {e}")
            return {}
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取完整性能指标"""
        return PerformanceMetrics(
            gpu_status=self.get_gpu_status(),
            cpu_status=self.get_cpu_status(),
            memory_usage=self.get_memory_status(),
            disk_usage=self.get_disk_status(),
            network_usage=self.get_network_status(),
            timestamp=datetime.now()
        )
    
    def optimize_gpu_memory(self):
        """优化GPU内存"""
        if not self.gpu_available:
            return
        
        try:
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 获取内存状态
            gpu_status = self.get_gpu_status()
            if gpu_status:
                memory_usage = (gpu_status.used_memory / gpu_status.total_memory) * 100
                
                if memory_usage > self.max_memory_usage:
                    logger.warning(f"GPU内存使用率过高: {memory_usage:.1f}%，执行清理")
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # 重新检查
                    gpu_status_after = self.get_gpu_status()
                    if gpu_status_after:
                        new_usage = (gpu_status_after.used_memory / gpu_status_after.total_memory) * 100
                        logger.info(f"GPU内存清理完成: {memory_usage:.1f}% -> {new_usage:.1f}%")
            
        except Exception as e:
            logger.error(f"GPU内存优化失败: {e}")
    
    def optimize_cpu_performance(self):
        """优化CPU性能"""
        try:
            cpu_status = self.get_cpu_status()
            
            # 如果CPU使用率过高，调整线程数
            if cpu_status.usage_percent > 90.0:
                current_threads = torch.get_num_threads()
                new_threads = max(1, current_threads - 2)
                torch.set_num_threads(new_threads)
                
                logger.warning(f"CPU使用率过高: {cpu_status.usage_percent:.1f}%，"
                             f"调整线程数: {current_threads} -> {new_threads}")
            
            elif cpu_status.usage_percent < 50.0:
                current_threads = torch.get_num_threads()
                max_threads = min(self.cpu_threads, 32)
                new_threads = min(max_threads, current_threads + 2)
                
                if new_threads > current_threads:
                    torch.set_num_threads(new_threads)
                    logger.info(f"CPU使用率较低: {cpu_status.usage_percent:.1f}%，"
                               f"增加线程数: {current_threads} -> {new_threads}")
            
        except Exception as e:
            logger.error(f"CPU性能优化失败: {e}")
    
    def _performance_monitor_loop(self):
        """性能监控循环"""
        while self.is_running:
            try:
                # 获取性能指标
                metrics = self.get_performance_metrics()
                
                # 记录历史
                with self.lock:
                    self.performance_history.append(metrics)
                    
                    # 限制历史记录数量
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                
                # 检查性能警告
                self._check_performance_warnings(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"性能监控失败: {e}")
                time.sleep(self.monitoring_interval)
    
    def _optimization_loop(self):
        """优化调度循环"""
        while self.is_running:
            try:
                time.sleep(self.optimization_interval)
                
                # 执行优化
                self.optimize_gpu_memory()
                self.optimize_cpu_performance()
                
                logger.debug("性能优化调度完成")
                
            except Exception as e:
                logger.error(f"性能优化失败: {e}")
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """检查性能警告"""
        warnings = []
        
        # GPU警告
        if metrics.gpu_status:
            gpu = metrics.gpu_status
            memory_usage = (gpu.used_memory / gpu.total_memory) * 100
            
            if memory_usage > 95.0:
                warnings.append(f"GPU内存使用率过高: {memory_usage:.1f}%")
            
            if gpu.utilization > 98.0:
                warnings.append(f"GPU利用率过高: {gpu.utilization:.1f}%")
        
        # CPU警告
        cpu = metrics.cpu_status
        if cpu.usage_percent > 95.0:
            warnings.append(f"CPU使用率过高: {cpu.usage_percent:.1f}%")
        
        # 内存警告
        memory = metrics.memory_usage
        if memory.get('usage_percent', 0) > 90.0:
            warnings.append(f"系统内存使用率过高: {memory['usage_percent']:.1f}%")
        
        # 磁盘警告
        disk = metrics.disk_usage
        if disk.get('usage_percent', 0) > 90.0:
            warnings.append(f"磁盘使用率过高: {disk['usage_percent']:.1f}%")
        
        # 输出警告
        for warning in warnings:
            logger.warning(f"⚠️ 性能警告: {warning}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        with self.lock:
            if not self.performance_history:
                return {"error": "暂无性能数据"}
            
            latest = self.performance_history[-1]
            
            report = {
                "timestamp": latest.timestamp.isoformat(),
                "gpu_info": {
                    "available": self.gpu_available,
                    "devices": len(self.gpu_devices),
                    "current_status": {
                        "name": latest.gpu_status.name if latest.gpu_status else "N/A",
                        "memory_usage_percent": (
                            (latest.gpu_status.used_memory / latest.gpu_status.total_memory) * 100
                            if latest.gpu_status else 0
                        ),
                        "utilization": latest.gpu_status.utilization if latest.gpu_status else 0
                    } if latest.gpu_status else None
                },
                "cpu_info": {
                    "cores": latest.cpu_status.core_count,
                    "threads": latest.cpu_status.thread_count,
                    "usage_percent": latest.cpu_status.usage_percent,
                    "frequency_mhz": latest.cpu_status.frequency,
                    "load_average": latest.cpu_status.load_average
                },
                "memory_info": latest.memory_usage,
                "disk_info": latest.disk_usage,
                "optimization_status": {
                    "gpu_memory_optimized": self.gpu_available,
                    "cpu_threads_optimized": True,
                    "monitoring_active": self.is_running
                }
            }
            
            return report
    
    def shutdown(self):
        """关闭优化器"""
        logger.info("正在关闭GPU性能优化器...")
        self.is_running = False
        
        # 等待线程结束
        time.sleep(2)
        
        logger.info("GPU性能优化器已关闭")


# 全局实例
_optimizer = None

def get_gpu_optimizer(config: Dict[str, Any] = None) -> GPUPerformanceOptimizer:
    """获取GPU优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = GPUPerformanceOptimizer(config)
    return _optimizer


if __name__ == "__main__":
    # 测试代码
    def test_gpu_optimizer():
        """测试GPU优化器"""
        optimizer = get_gpu_optimizer()
        
        # 获取性能报告
        report = optimizer.get_optimization_report()
        print("性能优化报告:")
        import json
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 运行一段时间
        time.sleep(10)
        
        # 再次获取报告
        report = optimizer.get_optimization_report()
        print("\n10秒后的性能报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        optimizer.shutdown()
    
    test_gpu_optimizer()
