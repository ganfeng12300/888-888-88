"""
🔧 硬件信息收集器 (温控集成版)
生产级硬件信息收集和分析系统
支持20核CPU + RTX3060 + 128GB内存 + 1TB NVMe的完整硬件信息收集

集成温控优化功能：
- 温度感知硬件信息收集
- 热负载分析
- 温控驱动的硬件状态评估
- 与thermal_monitor.py深度集成

Version: 2.0.0 (Thermal Control Integration)
"""

import asyncio
import psutil
import platform
import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import GPUtil
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from loguru import logger

# 导入温控监控系统
try:
    from .thermal_monitor import ThermalMonitor, ThermalReading, get_thermal_monitor
    THERMAL_MONITOR_AVAILABLE = True
except ImportError:
    THERMAL_MONITOR_AVAILABLE = False
    logger.warning("温控监控系统不可用")


@dataclass
class CPUInfo:
    """CPU信息"""
    model: str
    cores_physical: int
    cores_logical: int
    frequency_base: float
    frequency_max: float
    frequency_current: float
    architecture: str
    cache_l1: Optional[str] = None
    cache_l2: Optional[str] = None
    cache_l3: Optional[str] = None
    temperature: Optional[float] = None
    thermal_status: Optional[str] = None


@dataclass
class GPUInfo:
    """GPU信息"""
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: float
    temperature: float
    power_draw: float
    power_limit: float
    driver_version: str
    cuda_version: Optional[str] = None
    thermal_status: Optional[str] = None


@dataclass
class MemoryInfo:
    """内存信息"""
    total: int
    available: int
    used: int
    free: int
    usage_percent: float
    swap_total: int
    swap_used: int
    swap_free: int
    swap_percent: float
    estimated_temperature: Optional[float] = None
    thermal_status: Optional[str] = None


@dataclass
class StorageInfo:
    """存储信息"""
    device: str
    mountpoint: str
    filesystem: str
    total: int
    used: int
    free: int
    usage_percent: float
    read_speed: Optional[float] = None
    write_speed: Optional[float] = None


@dataclass
class NetworkInfo:
    """网络信息"""
    interface: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    speed: Optional[int] = None
    duplex: Optional[str] = None
    mtu: Optional[int] = None


@dataclass
class SystemInfo:
    """系统信息"""
    hostname: str
    platform: str
    platform_release: str
    platform_version: str
    architecture: str
    processor: str
    boot_time: float
    uptime: float


@dataclass
class HardwareInfoSnapshot:
    """硬件信息快照"""
    timestamp: float
    system: SystemInfo
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: List[GPUInfo]
    storage: List[StorageInfo]
    network: List[NetworkInfo]
    thermal_summary: Optional[Dict[str, Any]] = None


class HardwareInfoCollector:
    """硬件信息收集器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/hardware_config.json"
        self.gpu_available = GPU_AVAILABLE
        self.collection_history: List[HardwareInfoSnapshot] = []
        self.max_history_size = 100  # 保留最近100次收集记录
        
        # 温控监控集成
        self.thermal_monitor = None
        if THERMAL_MONITOR_AVAILABLE:
            try:
                thermal_config = {
                    "monitor_interval": 1.0,
                    "thresholds": {
                        "cpu_warning": 70.0,
                        "cpu_critical": 75.0,
                        "cpu_emergency": 80.0,
                        "gpu_warning": 75.0,
                        "gpu_critical": 80.0,
                        "gpu_emergency": 85.0
                    }
                }
                self.thermal_monitor = get_thermal_monitor(thermal_config)
                logger.info("温控监控系统集成成功")
            except Exception as e:
                logger.error(f"温控监控系统集成失败: {e}")
                self.thermal_monitor = None
        
        # 初始化GPU监控
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU监控初始化成功，检测到 {self.gpu_count} 个GPU")
            except Exception as e:
                logger.error(f"GPU监控初始化失败: {e}")
                self.gpu_available = False
        
        logger.info("硬件信息收集器初始化完成")
    
    async def collect_full_hardware_info(self) -> HardwareInfoSnapshot:
        """收集完整硬件信息"""
        try:
            timestamp = time.time()
            
            # 并行收集各类硬件信息
            tasks = [
                self._collect_system_info(),
                self._collect_cpu_info(),
                self._collect_memory_info(),
                self._collect_gpu_info(),
                self._collect_storage_info(),
                self._collect_network_info()
            ]
            
            results = await asyncio.gather(*tasks)
            system_info, cpu_info, memory_info, gpu_info, storage_info, network_info = results
            
            # 收集温控摘要
            thermal_summary = None
            if self.thermal_monitor:
                try:
                    thermal_summary = self.thermal_monitor.get_current_status()
                except Exception as e:
                    logger.error(f"获取温控摘要失败: {e}")
            
            # 创建硬件信息快照
            snapshot = HardwareInfoSnapshot(
                timestamp=timestamp,
                system=system_info,
                cpu=cpu_info,
                memory=memory_info,
                gpus=gpu_info,
                storage=storage_info,
                network=network_info,
                thermal_summary=thermal_summary
            )
            
            # 存储历史记录
            self._store_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"收集硬件信息失败: {e}")
            raise
    
    async def _collect_system_info(self) -> SystemInfo:
        """收集系统信息"""
        try:
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            return SystemInfo(
                hostname=platform.node(),
                platform=platform.system(),
                platform_release=platform.release(),
                platform_version=platform.version(),
                architecture=platform.machine(),
                processor=platform.processor(),
                boot_time=boot_time,
                uptime=uptime
            )
        except Exception as e:
            logger.error(f"收集系统信息失败: {e}")
            raise
    
    async def _collect_cpu_info(self) -> CPUInfo:
        """收集CPU信息"""
        try:
            # 基础CPU信息
            cpu_freq = psutil.cpu_freq()
            cpu_count_physical = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            
            # CPU温度
            cpu_temperature = None
            thermal_status = None
            
            if self.thermal_monitor:
                try:
                    thermal_reading = self.thermal_monitor.get_current_reading()
                    if thermal_reading:
                        cpu_temperature = thermal_reading.cpu_temp
                        thermal_status = self._get_thermal_status_description(
                            cpu_temperature, "cpu"
                        )
                except Exception as e:
                    logger.warning(f"获取CPU温度失败: {e}")
            
            # 尝试获取CPU详细信息
            cpu_model = platform.processor()
            if not cpu_model or cpu_model == "":
                try:
                    if platform.system() == "Linux":
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if "model name" in line:
                                    cpu_model = line.split(":")[1].strip()
                                    break
                except Exception:
                    cpu_model = "Unknown"
            
            return CPUInfo(
                model=cpu_model,
                cores_physical=cpu_count_physical or 0,
                cores_logical=cpu_count_logical,
                frequency_base=cpu_freq.current if cpu_freq else 0.0,
                frequency_max=cpu_freq.max if cpu_freq else 0.0,
                frequency_current=cpu_freq.current if cpu_freq else 0.0,
                architecture=platform.machine(),
                temperature=cpu_temperature,
                thermal_status=thermal_status
            )
        except Exception as e:
            logger.error(f"收集CPU信息失败: {e}")
            raise
    
    async def _collect_memory_info(self) -> MemoryInfo:
        """收集内存信息"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # 内存温度估算
            estimated_temperature = None
            thermal_status = None
            
            if self.thermal_monitor:
                try:
                    thermal_reading = self.thermal_monitor.get_current_reading()
                    if thermal_reading:
                        estimated_temperature = thermal_reading.memory_temp_estimated
                        thermal_status = self._get_thermal_status_description(
                            estimated_temperature, "memory"
                        )
                except Exception as e:
                    logger.warning(f"获取内存温度估算失败: {e}")
            
            return MemoryInfo(
                total=memory.total,
                available=memory.available,
                used=memory.used,
                free=memory.free,
                usage_percent=memory.percent,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_free=swap.free,
                swap_percent=swap.percent,
                estimated_temperature=estimated_temperature,
                thermal_status=thermal_status
            )
        except Exception as e:
            logger.error(f"收集内存信息失败: {e}")
            raise
    
    async def _collect_gpu_info(self) -> List[GPUInfo]:
        """收集GPU信息"""
        gpu_list = []
        
        if not self.gpu_available:
            return gpu_list
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 基础GPU信息
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                
                # 驱动版本
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                except:
                    driver_version = "Unknown"
                
                # CUDA版本
                cuda_version = None
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    if cuda_version:
                        major = cuda_version // 1000
                        minor = (cuda_version % 1000) // 10
                        cuda_version = f"{major}.{minor}"
                except:
                    pass
                
                # 温控状态
                thermal_status = self._get_thermal_status_description(temperature, "gpu")
                
                gpu_info = GPUInfo(
                    name=name,
                    memory_total=memory_info.total,
                    memory_used=memory_info.used,
                    memory_free=memory_info.free,
                    utilization=utilization.gpu,
                    temperature=temperature,
                    power_draw=power_draw,
                    power_limit=power_limit,
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    thermal_status=thermal_status
                )
                
                gpu_list.append(gpu_info)
                
        except Exception as e:
            logger.error(f"收集GPU信息失败: {e}")
        
        return gpu_list
    
    async def _collect_storage_info(self) -> List[StorageInfo]:
        """收集存储信息"""
        storage_list = []
        
        try:
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    storage_info = StorageInfo(
                        device=partition.device,
                        mountpoint=partition.mountpoint,
                        filesystem=partition.fstype,
                        total=usage.total,
                        used=usage.used,
                        free=usage.free,
                        usage_percent=(usage.used / usage.total) * 100 if usage.total > 0 else 0
                    )
                    
                    storage_list.append(storage_info)
                    
                except PermissionError:
                    # 跳过无权限访问的分区
                    continue
                except Exception as e:
                    logger.warning(f"收集存储信息失败 {partition.device}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"收集存储信息失败: {e}")
        
        return storage_list
    
    async def _collect_network_info(self) -> List[NetworkInfo]:
        """收集网络信息"""
        network_list = []
        
        try:
            network_stats = psutil.net_io_counters(pernic=True)
            
            for interface, stats in network_stats.items():
                # 跳过回环接口
                if interface.startswith('lo'):
                    continue
                
                network_info = NetworkInfo(
                    interface=interface,
                    bytes_sent=stats.bytes_sent,
                    bytes_recv=stats.bytes_recv,
                    packets_sent=stats.packets_sent,
                    packets_recv=stats.packets_recv
                )
                
                network_list.append(network_info)
                
        except Exception as e:
            logger.error(f"收集网络信息失败: {e}")
        
        return network_list
    
    def _get_thermal_status_description(self, temperature: Optional[float], component: str) -> Optional[str]:
        """获取温控状态描述"""
        if temperature is None:
            return None
        
        try:
            if component == "cpu":
                if temperature >= 80.0:
                    return "emergency"
                elif temperature >= 75.0:
                    return "critical"
                elif temperature >= 70.0:
                    return "warning"
                else:
                    return "normal"
            elif component == "gpu":
                if temperature >= 85.0:
                    return "emergency"
                elif temperature >= 80.0:
                    return "critical"
                elif temperature >= 75.0:
                    return "warning"
                else:
                    return "normal"
            elif component == "memory":
                if temperature >= 70.0:
                    return "warning"
                elif temperature >= 60.0:
                    return "elevated"
                else:
                    return "normal"
            else:
                return "unknown"
                
        except Exception:
            return "error"
    
    def _store_snapshot(self, snapshot: HardwareInfoSnapshot):
        """存储硬件信息快照"""
        try:
            self.collection_history.append(snapshot)
            
            # 限制历史记录大小
            if len(self.collection_history) > self.max_history_size:
                self.collection_history = self.collection_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"存储硬件信息快照失败: {e}")
    
    def get_latest_snapshot(self) -> Optional[HardwareInfoSnapshot]:
        """获取最新的硬件信息快照"""
        if self.collection_history:
            return self.collection_history[-1]
        return None
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """获取硬件摘要"""
        try:
            latest = self.get_latest_snapshot()
            if not latest:
                return {"status": "no_data"}
            
            summary = {
                "timestamp": latest.timestamp,
                "system": {
                    "hostname": latest.system.hostname,
                    "platform": latest.system.platform,
                    "uptime_hours": latest.system.uptime / 3600
                },
                "cpu": {
                    "model": latest.cpu.model,
                    "cores": f"{latest.cpu.cores_physical}P/{latest.cpu.cores_logical}L",
                    "frequency_ghz": latest.cpu.frequency_current / 1000,
                    "temperature": latest.cpu.temperature,
                    "thermal_status": latest.cpu.thermal_status
                },
                "memory": {
                    "total_gb": latest.memory.total / (1024**3),
                    "usage_percent": latest.memory.usage_percent,
                    "estimated_temperature": latest.memory.estimated_temperature,
                    "thermal_status": latest.memory.thermal_status
                },
                "gpus": [
                    {
                        "name": gpu.name,
                        "memory_gb": gpu.memory_total / (1024**3),
                        "utilization": gpu.utilization,
                        "temperature": gpu.temperature,
                        "thermal_status": gpu.thermal_status,
                        "power_usage": f"{gpu.power_draw:.1f}W/{gpu.power_limit:.1f}W"
                    }
                    for gpu in latest.gpus
                ],
                "storage_total_gb": sum(storage.total for storage in latest.storage) / (1024**3),
                "network_interfaces": len(latest.network),
                "thermal_summary": latest.thermal_summary
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取硬件摘要失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_snapshot_to_file(self, filepath: str, snapshot: Optional[HardwareInfoSnapshot] = None):
        """保存硬件信息快照到文件"""
        try:
            if snapshot is None:
                snapshot = self.get_latest_snapshot()
            
            if snapshot is None:
                logger.warning("没有可保存的硬件信息快照")
                return
            
            # 转换为字典格式
            snapshot_dict = asdict(snapshot)
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"硬件信息快照已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存硬件信息快照失败: {e}")
    
    def get_thermal_integration_status(self) -> Dict[str, Any]:
        """获取温控集成状态"""
        try:
            status = {
                "thermal_monitor_available": THERMAL_MONITOR_AVAILABLE,
                "thermal_monitor_active": self.thermal_monitor is not None,
                "collection_history_count": len(self.collection_history),
                "gpu_monitoring": self.gpu_available
            }
            
            if self.thermal_monitor:
                try:
                    thermal_status = self.thermal_monitor.get_current_status()
                    status["thermal_status"] = thermal_status
                except Exception as e:
                    status["thermal_status_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"获取温控集成状态失败: {e}")
            return {"status": "error", "message": str(e)}


# 全局硬件信息收集器实例
_hardware_collector_instance = None


def get_hardware_collector(config: Optional[Dict[str, Any]] = None) -> HardwareInfoCollector:
    """获取硬件信息收集器实例（单例模式）"""
    global _hardware_collector_instance
    
    if _hardware_collector_instance is None:
        config_path = None
        if config and "config_path" in config:
            config_path = config["config_path"]
        
        _hardware_collector_instance = HardwareInfoCollector(config_path)
    
    return _hardware_collector_instance


async def collect_hardware_info_async() -> HardwareInfoSnapshot:
    """异步收集硬件信息的便捷函数"""
    collector = get_hardware_collector()
    return await collector.collect_full_hardware_info()


def collect_hardware_info_sync() -> Dict[str, Any]:
    """同步收集硬件信息摘要的便捷函数"""
    collector = get_hardware_collector()
    return collector.get_hardware_summary()


if __name__ == "__main__":
    # 测试代码
    async def test_hardware_collection():
        collector = get_hardware_collector()
        
        print("🔧 开始收集硬件信息...")
        snapshot = await collector.collect_full_hardware_info()
        
        print("📊 硬件信息摘要:")
        summary = collector.get_hardware_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        print("🌡️ 温控集成状态:")
        thermal_status = collector.get_thermal_integration_status()
        print(json.dumps(thermal_status, indent=2, ensure_ascii=False))
    
    # 运行测试
    asyncio.run(test_hardware_collection())

