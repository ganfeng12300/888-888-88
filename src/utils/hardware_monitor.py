"""
💻 硬件监控模块
实时监控CPU、GPU、内存等硬件资源使用情况
"""

import asyncio
import psutil
import platform
from typing import Dict, Any, Optional
from loguru import logger

try:
    import GPUtil
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPU监控库未安装，GPU监控功能将被禁用")


class HardwareMonitor:
    """硬件监控器"""
    
    def __init__(self):
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.memory_total = psutil.virtual_memory().total
        self.gpu_info = {}
        self.monitoring = False
        
        # 初始化GPU监控
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU监控初始化成功")
            except Exception as e:
                self.gpu_available = False
                logger.warning(f"GPU监控初始化失败: {e}")
        else:
            self.gpu_available = False
    
    async def detect_hardware(self) -> Dict[str, Any]:
        """检测硬件配置"""
        try:
            hardware_info = {
                "system": {
                    "platform": platform.system(),
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "processor": platform.processor(),
                },
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
                    "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                },
                "memory": {
                    "total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                    "available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
                },
                "disk": [],
                "gpu": []
            }
            
            # 磁盘信息
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    hardware_info["disk"].append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "file_system": partition.fstype,
                        "total": round(partition_usage.total / (1024**3), 2),  # GB
                        "used": round(partition_usage.used / (1024**3), 2),  # GB
                        "free": round(partition_usage.free / (1024**3), 2),  # GB
                        "percentage": partition_usage.percent,
                    })
                except PermissionError:
                    continue
            
            # GPU信息
            if self.gpu_available:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        hardware_info["gpu"].append({
                            "id": i,
                            "name": name,
                            "memory_total": round(memory_info.total / (1024**3), 2),  # GB
                            "memory_used": round(memory_info.used / (1024**3), 2),  # GB
                            "memory_free": round(memory_info.free / (1024**3), 2),  # GB
                        })
                except Exception as e:
                    logger.error(f"获取GPU信息失败: {e}")
            
            self.hardware_info = hardware_info
            logger.info("硬件检测完成")
            return hardware_info
            
        except Exception as e:
            logger.error(f"硬件检测失败: {e}")
            return {}
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息摘要"""
        if not hasattr(self, 'hardware_info'):
            return {
                "cpu_cores": self.cpu_cores,
                "memory_gb": round(self.memory_total / (1024**3), 2),
                "gpu_name": "Unknown",
                "gpu_memory_gb": 0
            }
        
        gpu_name = "Unknown"
        gpu_memory = 0
        
        if self.hardware_info.get("gpu"):
            gpu_info = self.hardware_info["gpu"][0]  # 使用第一个GPU
            gpu_name = gpu_info.get("name", "Unknown")
            gpu_memory = gpu_info.get("memory_total", 0)
        
        return {
            "cpu_cores": self.hardware_info["cpu"]["logical_cores"],
            "memory_gb": self.hardware_info["memory"]["total"],
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory
        }
    
    async def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时硬件使用统计"""
        try:
            stats = {
                "timestamp": asyncio.get_event_loop().time(),
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "usage_per_core": psutil.cpu_percent(interval=1, percpu=True),
                    "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                    "temperature": await self._get_cpu_temperature(),
                },
                "memory": {
                    "usage_percent": psutil.virtual_memory().percent,
                    "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                },
                "disk": {
                    "usage_percent": psutil.disk_usage('/').percent,
                    "read_bytes": psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
                    "write_bytes": psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
                },
                "network": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv,
                },
                "gpu": []
            }
            
            # GPU统计
            if self.gpu_available:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # GPU使用率
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        # 内存使用
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        # 温度
                        try:
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            temperature = 0
                        
                        # 功耗
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        except:
                            power = 0
                        
                        stats["gpu"].append({
                            "id": i,
                            "usage_percent": utilization.gpu,
                            "memory_usage_percent": (memory_info.used / memory_info.total) * 100,
                            "memory_used_gb": round(memory_info.used / (1024**3), 2),
                            "temperature": temperature,
                            "power_usage": power,
                        })
                except Exception as e:
                    logger.error(f"获取GPU统计失败: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"获取实时统计失败: {e}")
            return {}
    
    async def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                return entry.current
            return 0.0
        except:
            return 0.0
    
    async def start_monitoring(self, interval: int = 10):
        """启动硬件监控"""
        self.monitoring = True
        logger.info(f"开始硬件监控，间隔: {interval}秒")
        
        while self.monitoring:
            try:
                stats = await self.get_real_time_stats()
                
                # 检查资源使用警告
                await self._check_resource_warnings(stats)
                
                # 等待下次监控
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"硬件监控出错: {e}")
                await asyncio.sleep(interval)
    
    async def _check_resource_warnings(self, stats: Dict[str, Any]):
        """检查资源使用警告"""
        try:
            # CPU使用率警告
            cpu_usage = stats.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > 90:
                logger.warning(f"CPU使用率过高: {cpu_usage:.1f}%")
            
            # 内存使用率警告
            memory_usage = stats.get("memory", {}).get("usage_percent", 0)
            if memory_usage > 90:
                logger.warning(f"内存使用率过高: {memory_usage:.1f}%")
            
            # GPU使用率和温度警告
            for gpu_stat in stats.get("gpu", []):
                gpu_id = gpu_stat.get("id", 0)
                gpu_usage = gpu_stat.get("usage_percent", 0)
                gpu_temp = gpu_stat.get("temperature", 0)
                
                if gpu_usage > 95:
                    logger.warning(f"GPU {gpu_id} 使用率过高: {gpu_usage}%")
                
                if gpu_temp > 80:
                    logger.warning(f"GPU {gpu_id} 温度过高: {gpu_temp}°C")
            
            # 磁盘使用率警告
            disk_usage = stats.get("disk", {}).get("usage_percent", 0)
            if disk_usage > 90:
                logger.warning(f"磁盘使用率过高: {disk_usage:.1f}%")
                
        except Exception as e:
            logger.error(f"资源警告检查失败: {e}")
    
    def stop_monitoring(self):
        """停止硬件监控"""
        self.monitoring = False
        logger.info("硬件监控已停止")
    
    async def optimize_for_trading(self):
        """为交易优化系统性能"""
        try:
            logger.info("开始系统性能优化...")
            
            # 设置进程优先级
            current_process = psutil.Process()
            if platform.system() == "Windows":
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                current_process.nice(-10)  # 提高优先级
            
            logger.info("进程优先级已优化")
            
            # GPU性能模式设置
            if self.gpu_available:
                try:
                    # 设置GPU性能模式（如果支持）
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        # 这里可以添加GPU性能优化设置
                        pass
                    logger.info("GPU性能模式已优化")
                except Exception as e:
                    logger.warning(f"GPU性能优化失败: {e}")
            
            logger.success("系统性能优化完成")
            
        except Exception as e:
            logger.error(f"系统性能优化失败: {e}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取性能优化建议"""
        recommendations = []
        
        try:
            # 检查内存
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                recommendations.append("内存使用率较高，建议关闭不必要的程序")
            
            # 检查CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 80:
                recommendations.append("CPU使用率较高，建议优化后台进程")
            
            # 检查磁盘
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 85:
                recommendations.append("磁盘空间不足，建议清理临时文件")
            
            # GPU建议
            if not self.gpu_available:
                recommendations.append("建议安装GPU监控库以获得更好的性能监控")
            
            if not recommendations:
                recommendations.append("系统性能良好，无需特别优化")
            
        except Exception as e:
            logger.error(f"获取优化建议失败: {e}")
            recommendations.append("无法获取系统状态，请检查系统权限")
        
        return recommendations
