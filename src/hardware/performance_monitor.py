"""
💻 硬件性能实时监控系统
生产级硬件资源监控、优化和动态调节系统
支持20核CPU + RTX3060 + 128GB内存 + 1TB NVMe的极限性能释放
"""

import asyncio
import psutil
import platform
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import os
import signal

try:
    import GPUtil
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from loguru import logger


@dataclass
class HardwareMetrics:
    """硬件指标数据结构"""
    timestamp: float
    cpu_usage_percent: float
    cpu_frequency_mhz: float
    cpu_temperature: float
    cpu_cores_usage: List[float]
    
    memory_usage_percent: float
    memory_used_gb: float
    memory_available_gb: float
    memory_cached_gb: float
    
    gpu_usage_percent: float
    gpu_memory_usage_percent: float
    gpu_memory_used_gb: float
    gpu_temperature: float
    gpu_power_usage: float
    
    disk_usage_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    disk_free_gb: float
    
    network_sent_mb_s: float
    network_recv_mb_s: float


@dataclass
class PerformanceTargets:
    """性能目标配置"""
    cpu_usage_target: Tuple[float, float] = (85.0, 90.0)  # 目标使用率范围
    cpu_temp_max: float = 80.0  # CPU最高温度
    cpu_temp_warning: float = 75.0  # CPU警告温度
    
    gpu_usage_target: Tuple[float, float] = (90.0, 95.0)  # GPU目标使用率
    gpu_temp_max: float = 75.0  # GPU最高温度
    gpu_temp_warning: float = 70.0  # GPU警告温度
    
    memory_usage_target: Tuple[float, float] = (80.0, 85.0)  # 内存目标使用率
    disk_usage_max: float = 80.0  # 磁盘最大使用率
    disk_usage_warning: float = 75.0  # 磁盘警告使用率


class HardwarePerformanceMonitor:
    """硬件性能实时监控器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/hardware_config.json"
        self.targets = PerformanceTargets()
        self.monitoring = False
        self.metrics_history: List[HardwareMetrics] = []
        self.max_history_size = 3600  # 保留1小时的历史数据
        
        # 硬件信息
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.memory_total = psutil.virtual_memory().total
        self.gpu_available = GPU_AVAILABLE
        
        # 性能调节参数
        self.cpu_frequency_step = 100  # MHz
        self.gpu_power_step = 5  # %
        
        # 初始化GPU监控
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU监控初始化成功，检测到 {self.gpu_count} 个GPU")
            except Exception as e:
                logger.error(f"GPU监控初始化失败: {e}")
                self.gpu_available = False
        
        # 基准性能数据
        self.baseline_metrics: Optional[HardwareMetrics] = None
        self.performance_adjustments = 0
        
        logger.info("硬件性能监控器初始化完成")
    
    async def start_monitoring(self, interval: float = 1.0):
        """启动硬件监控"""
        self.monitoring = True
        logger.info(f"开始硬件性能监控，监控间隔: {interval}秒")
        
        # 获取基准性能
        await self._establish_baseline()
        
        while self.monitoring:
            try:
                # 收集硬件指标
                metrics = await self._collect_hardware_metrics()
                
                # 存储历史数据
                self._store_metrics(metrics)
                
                # 性能分析和调节
                await self._analyze_and_adjust_performance(metrics)
                
                # 检查警告条件
                await self._check_warning_conditions(metrics)
                
                # 等待下次监控
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"硬件监控循环出错: {e}")
                await asyncio.sleep(interval)
    
    async def _establish_baseline(self):
        """建立基准性能数据"""
        logger.info("建立硬件性能基准...")
        
        # 收集5次数据取平均值
        baseline_samples = []
        for _ in range(5):
            metrics = await self._collect_hardware_metrics()
            baseline_samples.append(metrics)
            await asyncio.sleep(1)
        
        # 计算基准值
        self.baseline_metrics = self._calculate_average_metrics(baseline_samples)
        logger.info("硬件性能基准建立完成")
    
    def _calculate_average_metrics(self, metrics_list: List[HardwareMetrics]) -> HardwareMetrics:
        """计算平均指标"""
        if not metrics_list:
            raise ValueError("指标列表不能为空")
        
        # 计算各项指标的平均值
        avg_metrics = HardwareMetrics(
            timestamp=time.time(),
            cpu_usage_percent=sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list),
            cpu_frequency_mhz=sum(m.cpu_frequency_mhz for m in metrics_list) / len(metrics_list),
            cpu_temperature=sum(m.cpu_temperature for m in metrics_list) / len(metrics_list),
            cpu_cores_usage=[
                sum(m.cpu_cores_usage[i] for m in metrics_list) / len(metrics_list)
                for i in range(len(metrics_list[0].cpu_cores_usage))
            ],
            memory_usage_percent=sum(m.memory_usage_percent for m in metrics_list) / len(metrics_list),
            memory_used_gb=sum(m.memory_used_gb for m in metrics_list) / len(metrics_list),
            memory_available_gb=sum(m.memory_available_gb for m in metrics_list) / len(metrics_list),
            memory_cached_gb=sum(m.memory_cached_gb for m in metrics_list) / len(metrics_list),
            gpu_usage_percent=sum(m.gpu_usage_percent for m in metrics_list) / len(metrics_list),
            gpu_memory_usage_percent=sum(m.gpu_memory_usage_percent for m in metrics_list) / len(metrics_list),
            gpu_memory_used_gb=sum(m.gpu_memory_used_gb for m in metrics_list) / len(metrics_list),
            gpu_temperature=sum(m.gpu_temperature for m in metrics_list) / len(metrics_list),
            gpu_power_usage=sum(m.gpu_power_usage for m in metrics_list) / len(metrics_list),
            disk_usage_percent=sum(m.disk_usage_percent for m in metrics_list) / len(metrics_list),
            disk_read_mb_s=sum(m.disk_read_mb_s for m in metrics_list) / len(metrics_list),
            disk_write_mb_s=sum(m.disk_write_mb_s for m in metrics_list) / len(metrics_list),
            disk_free_gb=sum(m.disk_free_gb for m in metrics_list) / len(metrics_list),
            network_sent_mb_s=sum(m.network_sent_mb_s for m in metrics_list) / len(metrics_list),
            network_recv_mb_s=sum(m.network_recv_mb_s for m in metrics_list) / len(metrics_list),
        )
        
        return avg_metrics
    
    async def _collect_hardware_metrics(self) -> HardwareMetrics:
        """收集硬件指标"""
        timestamp = time.time()
        
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_frequency = cpu_freq.current if cpu_freq else 0
        cpu_cores_usage = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_temperature = await self._get_cpu_temperature()
        
        # 内存指标
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_cached_gb = getattr(memory, 'cached', 0) / (1024**3)
        
        # GPU指标
        gpu_usage = 0
        gpu_memory_usage = 0
        gpu_memory_used = 0
        gpu_temperature = 0
        gpu_power = 0
        
        if self.gpu_available and self.gpu_count > 0:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 使用第一个GPU
                
                # GPU使用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = utilization.gpu
                
                # GPU内存
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage = (memory_info.used / memory_info.total) * 100
                gpu_memory_used = memory_info.used / (1024**3)
                
                # GPU温度
                try:
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    gpu_temperature = 0
                
                # GPU功耗
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    gpu_power = 0
                    
            except Exception as e:
                logger.warning(f"获取GPU指标失败: {e}")
        
        # 磁盘指标
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = disk_usage.percent
        disk_free_gb = disk_usage.free / (1024**3)
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        if hasattr(self, '_last_disk_io'):
            time_delta = timestamp - self._last_disk_timestamp
            disk_read_mb_s = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2) / time_delta
            disk_write_mb_s = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2) / time_delta
        else:
            disk_read_mb_s = 0
            disk_write_mb_s = 0
        
        self._last_disk_io = disk_io
        self._last_disk_timestamp = timestamp
        
        # 网络指标
        network_io = psutil.net_io_counters()
        if hasattr(self, '_last_network_io'):
            time_delta = timestamp - self._last_network_timestamp
            network_sent_mb_s = (network_io.bytes_sent - self._last_network_io.bytes_sent) / (1024**2) / time_delta
            network_recv_mb_s = (network_io.bytes_recv - self._last_network_io.bytes_recv) / (1024**2) / time_delta
        else:
            network_sent_mb_s = 0
            network_recv_mb_s = 0
        
        self._last_network_io = network_io
        self._last_network_timestamp = timestamp
        
        return HardwareMetrics(
            timestamp=timestamp,
            cpu_usage_percent=cpu_percent,
            cpu_frequency_mhz=cpu_frequency,
            cpu_temperature=cpu_temperature,
            cpu_cores_usage=cpu_cores_usage,
            memory_usage_percent=memory_usage_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            memory_cached_gb=memory_cached_gb,
            gpu_usage_percent=gpu_usage,
            gpu_memory_usage_percent=gpu_memory_usage,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_temperature=gpu_temperature,
            gpu_power_usage=gpu_power,
            disk_usage_percent=disk_usage_percent,
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            disk_free_gb=disk_free_gb,
            network_sent_mb_s=network_sent_mb_s,
            network_recv_mb_s=network_recv_mb_s,
        )
    
    async def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            # 尝试从sensors获取温度
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if 'coretemp' in name.lower() or 'cpu' in name.lower():
                            for entry in entries:
                                if entry.current:
                                    return entry.current
            
            # 尝试从系统文件读取
            temp_files = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        if 20 < temp < 100:  # 合理的温度范围
                            return temp
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"获取CPU温度失败: {e}")
            return 0.0
    
    def _store_metrics(self, metrics: HardwareMetrics):
        """存储指标历史数据"""
        self.metrics_history.append(metrics)
        
        # 限制历史数据大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    async def _analyze_and_adjust_performance(self, metrics: HardwareMetrics):
        """分析性能并进行动态调节"""
        adjustments_made = []
        
        # CPU性能调节
        cpu_adjustment = await self._adjust_cpu_performance(metrics)
        if cpu_adjustment:
            adjustments_made.append(cpu_adjustment)
        
        # GPU性能调节
        gpu_adjustment = await self._adjust_gpu_performance(metrics)
        if gpu_adjustment:
            adjustments_made.append(gpu_adjustment)
        
        # 内存优化
        memory_adjustment = await self._optimize_memory_usage(metrics)
        if memory_adjustment:
            adjustments_made.append(memory_adjustment)
        
        if adjustments_made:
            self.performance_adjustments += 1
            logger.info(f"性能调节 #{self.performance_adjustments}: {', '.join(adjustments_made)}")
    
    async def _adjust_cpu_performance(self, metrics: HardwareMetrics) -> Optional[str]:
        """调节CPU性能"""
        target_min, target_max = self.targets.cpu_usage_target
        
        # 温度保护优先
        if metrics.cpu_temperature > self.targets.cpu_temp_max:
            await self._reduce_cpu_frequency()
            return f"CPU温度过高({metrics.cpu_temperature:.1f}°C)，降低频率"
        
        # 使用率调节
        if metrics.cpu_usage_percent < target_min and metrics.cpu_temperature < self.targets.cpu_temp_warning:
            await self._increase_cpu_frequency()
            return f"CPU使用率偏低({metrics.cpu_usage_percent:.1f}%)，提高频率"
        elif metrics.cpu_usage_percent > target_max:
            await self._reduce_cpu_frequency()
            return f"CPU使用率过高({metrics.cpu_usage_percent:.1f}%)，降低频率"
        
        return None
    
    async def _adjust_gpu_performance(self, metrics: HardwareMetrics) -> Optional[str]:
        """调节GPU性能"""
        if not self.gpu_available:
            return None
        
        target_min, target_max = self.targets.gpu_usage_target
        
        # 温度保护优先
        if metrics.gpu_temperature > self.targets.gpu_temp_max:
            await self._reduce_gpu_power_limit()
            return f"GPU温度过高({metrics.gpu_temperature:.1f}°C)，降低功耗限制"
        
        # 使用率调节
        if metrics.gpu_usage_percent < target_min and metrics.gpu_temperature < self.targets.gpu_temp_warning:
            await self._increase_gpu_power_limit()
            return f"GPU使用率偏低({metrics.gpu_usage_percent:.1f}%)，提高功耗限制"
        elif metrics.gpu_usage_percent > target_max:
            await self._reduce_gpu_power_limit()
            return f"GPU使用率过高({metrics.gpu_usage_percent:.1f}%)，降低功耗限制"
        
        return None
    
    async def _optimize_memory_usage(self, metrics: HardwareMetrics) -> Optional[str]:
        """优化内存使用"""
        target_min, target_max = self.targets.memory_usage_target
        
        if metrics.memory_usage_percent > target_max:
            # 触发内存清理
            await self._trigger_memory_cleanup()
            return f"内存使用率过高({metrics.memory_usage_percent:.1f}%)，触发清理"
        
        return None
    
    async def _reduce_cpu_frequency(self):
        """降低CPU频率"""
        try:
            # 在Linux系统上调节CPU频率
            if platform.system() == "Linux":
                # 使用cpufreq-set命令（需要安装cpufrequtils）
                subprocess.run(['sudo', 'cpufreq-set', '-d', '1000000'], 
                             capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"调节CPU频率失败: {e}")
    
    async def _increase_cpu_frequency(self):
        """提高CPU频率"""
        try:
            if platform.system() == "Linux":
                subprocess.run(['sudo', 'cpufreq-set', '-u', '4000000'], 
                             capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"调节CPU频率失败: {e}")
    
    async def _reduce_gpu_power_limit(self):
        """降低GPU功耗限制"""
        try:
            if self.gpu_available:
                # 使用nvidia-smi调节功耗限制
                subprocess.run(['nvidia-smi', '-pl', '150'], 
                             capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"调节GPU功耗失败: {e}")
    
    async def _increase_gpu_power_limit(self):
        """提高GPU功耗限制"""
        try:
            if self.gpu_available:
                subprocess.run(['nvidia-smi', '-pl', '170'], 
                             capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"调节GPU功耗失败: {e}")
    
    async def _trigger_memory_cleanup(self):
        """触发内存清理"""
        try:
            # 清理系统缓存
            if platform.system() == "Linux":
                subprocess.run(['sudo', 'sync'], capture_output=True, check=False)
                subprocess.run(['sudo', 'echo', '1', '>', '/proc/sys/vm/drop_caches'], 
                             shell=True, capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"内存清理失败: {e}")
    
    async def _check_warning_conditions(self, metrics: HardwareMetrics):
        """检查警告条件"""
        warnings = []
        
        # CPU警告
        if metrics.cpu_temperature > self.targets.cpu_temp_warning:
            warnings.append(f"CPU温度警告: {metrics.cpu_temperature:.1f}°C")
        
        # GPU警告
        if metrics.gpu_temperature > self.targets.gpu_temp_warning:
            warnings.append(f"GPU温度警告: {metrics.gpu_temperature:.1f}°C")
        
        # 内存警告
        if metrics.memory_usage_percent > self.targets.memory_usage_target[1]:
            warnings.append(f"内存使用率警告: {metrics.memory_usage_percent:.1f}%")
        
        # 磁盘警告
        if metrics.disk_usage_percent > self.targets.disk_usage_warning:
            warnings.append(f"磁盘使用率警告: {metrics.disk_usage_percent:.1f}%")
        
        if warnings:
            logger.warning(f"硬件警告: {'; '.join(warnings)}")
    
    def get_current_metrics(self) -> Optional[HardwareMetrics]:
        """获取当前指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[HardwareMetrics]:
        """获取指定时间范围内的历史指标"""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(5)  # 最近5分钟
        if not recent_metrics:
            return {}
        
        return {
            "monitoring_duration_hours": (time.time() - self.metrics_history[0].timestamp) / 3600,
            "total_adjustments": self.performance_adjustments,
            "current_metrics": asdict(self.metrics_history[-1]),
            "average_metrics_5min": asdict(self._calculate_average_metrics(recent_metrics)),
            "hardware_status": {
                "cpu_healthy": recent_metrics[-1].cpu_temperature < self.targets.cpu_temp_warning,
                "gpu_healthy": recent_metrics[-1].gpu_temperature < self.targets.gpu_temp_warning,
                "memory_healthy": recent_metrics[-1].memory_usage_percent < self.targets.memory_usage_target[1],
                "disk_healthy": recent_metrics[-1].disk_usage_percent < self.targets.disk_usage_warning,
            }
        }
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        logger.info("硬件性能监控已停止")
    
    def save_metrics_to_file(self, filepath: str):
        """保存指标到文件"""
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"指标数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存指标数据失败: {e}")


# 全局硬件监控实例
hardware_monitor = HardwarePerformanceMonitor()


async def main():
    """测试主函数"""
    logger.info("启动硬件性能监控测试...")
    
    # 启动监控
    monitor_task = asyncio.create_task(hardware_monitor.start_monitoring(interval=2.0))
    
    try:
        # 运行30秒测试
        await asyncio.sleep(30)
        
        # 获取性能摘要
        summary = hardware_monitor.get_performance_summary()
        logger.info(f"性能摘要: {json.dumps(summary, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止监控...")
    finally:
        hardware_monitor.stop_monitoring()
        monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
