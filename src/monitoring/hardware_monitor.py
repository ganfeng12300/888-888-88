"""
🖥️ 硬件性能监控器 - 生产级实盘交易硬件资源实时监控系统
监控CPU、GPU、内存、存储、网络等硬件资源使用情况
提供性能预警、资源优化建议、硬件健康状态评估
"""
import asyncio
import time
import threading
import psutil
import platform
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from loguru import logger

try:
    import GPUtil
    import pynvml
    GPU_MONITORING_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    print("GPU monitoring libraries not available, GPU monitoring will be limited")

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    CRITICAL = "critical"  # 关键
    EMERGENCY = "emergency"  # 紧急

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"  # CPU
    GPU = "gpu"  # GPU
    MEMORY = "memory"  # 内存
    DISK = "disk"  # 磁盘
    NETWORK = "network"  # 网络

@dataclass
class HardwareAlert:
    """硬件告警"""
    alert_id: str  # 告警ID
    resource_type: ResourceType  # 资源类型
    alert_level: AlertLevel  # 告警级别
    message: str  # 告警消息
    current_value: float  # 当前值
    threshold: float  # 阈值
    timestamp: float = field(default_factory=time.time)  # 时间戳
    is_resolved: bool = False  # 是否已解决

@dataclass
class CPUMetrics:
    """CPU指标"""
    usage_percent: float  # 使用率百分比
    frequency: float  # 频率 (MHz)
    temperature: float  # 温度 (°C)
    core_count: int  # 核心数
    thread_count: int  # 线程数
    load_average: Tuple[float, float, float]  # 负载平均值 (1min, 5min, 15min)
    per_core_usage: List[float]  # 每核心使用率
    timestamp: float = field(default_factory=time.time)

@dataclass
class GPUMetrics:
    """GPU指标"""
    gpu_id: int  # GPU ID
    name: str  # GPU名称
    usage_percent: float  # 使用率百分比
    memory_used: float  # 已使用显存 (MB)
    memory_total: float  # 总显存 (MB)
    memory_percent: float  # 显存使用率百分比
    temperature: float  # 温度 (°C)
    power_usage: float  # 功耗 (W)
    fan_speed: float  # 风扇转速百分比
    timestamp: float = field(default_factory=time.time)

@dataclass
class MemoryMetrics:
    """内存指标"""
    total: float  # 总内存 (GB)
    available: float  # 可用内存 (GB)
    used: float  # 已使用内存 (GB)
    usage_percent: float  # 使用率百分比
    swap_total: float  # 总交换空间 (GB)
    swap_used: float  # 已使用交换空间 (GB)
    swap_percent: float  # 交换空间使用率
    timestamp: float = field(default_factory=time.time)

@dataclass
class DiskMetrics:
    """磁盘指标"""
    device: str  # 设备名称
    mountpoint: str  # 挂载点
    total: float  # 总空间 (GB)
    used: float  # 已使用空间 (GB)
    free: float  # 可用空间 (GB)
    usage_percent: float  # 使用率百分比
    read_speed: float  # 读取速度 (MB/s)
    write_speed: float  # 写入速度 (MB/s)
    iops: float  # IOPS
    timestamp: float = field(default_factory=time.time)

@dataclass
class NetworkMetrics:
    """网络指标"""
    interface: str  # 网络接口
    bytes_sent: float  # 发送字节数
    bytes_recv: float  # 接收字节数
    packets_sent: int  # 发送包数
    packets_recv: int  # 接收包数
    send_speed: float  # 发送速度 (MB/s)
    recv_speed: float  # 接收速度 (MB/s)
    connections: int  # 连接数
    timestamp: float = field(default_factory=time.time)

class CPUMonitor:
    """CPU监控器"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.previous_net_io = None
        self.previous_disk_io = None
        
        logger.info(f"CPU监控器初始化完成: {self.cpu_count}核心/{self.cpu_count_logical}线程")
    
    def get_cpu_metrics(self) -> CPUMetrics:
        """获取CPU指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            per_cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0
            
            # 负载平均值
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows系统没有getloadavg
                load_avg = (0, 0, 0)
            
            # CPU温度（尝试获取）
            temperature = self._get_cpu_temperature()
            
            return CPUMetrics(
                usage_percent=cpu_percent,
                frequency=frequency,
                temperature=temperature,
                core_count=self.cpu_count,
                thread_count=self.cpu_count_logical,
                load_average=load_avg,
                per_core_usage=per_cpu_percent
            )
        
        except Exception as e:
            logger.error(f"获取CPU指标失败: {e}")
            return CPUMetrics(
                usage_percent=0,
                frequency=0,
                temperature=0,
                core_count=self.cpu_count,
                thread_count=self.cpu_count_logical,
                load_average=(0, 0, 0),
                per_core_usage=[]
            )
    
    def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # 尝试获取CPU温度
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                return entries[0].current
                    
                    # 如果没有找到CPU相关的，返回第一个温度传感器
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            
            return 0.0  # 无法获取温度
        
        except Exception as e:
            logger.debug(f"获取CPU温度失败: {e}")
            return 0.0

class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self):
        self.gpu_available = GPU_MONITORING_AVAILABLE
        self.gpu_count = 0
        
        if self.gpu_available:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU监控器初始化完成: {self.gpu_count}个GPU")
            except Exception as e:
                logger.error(f"GPU监控器初始化失败: {e}")
                self.gpu_available = False
        else:
            logger.warning("GPU监控不可用")
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """获取GPU指标"""
        if not self.gpu_available:
            return []
        
        gpu_metrics = []
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU名称
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # GPU使用率
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                except:
                    gpu_usage = 0
                
                # 显存信息
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = mem_info.total / 1024 / 1024  # MB
                    memory_used = mem_info.used / 1024 / 1024  # MB
                    memory_percent = (memory_used / memory_total) * 100
                except:
                    memory_total = memory_used = memory_percent = 0
                
                # GPU温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                
                # 功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
                except:
                    power_usage = 0
                
                # 风扇转速
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = 0
                
                gpu_metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    usage_percent=gpu_usage,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    memory_percent=memory_percent,
                    temperature=temperature,
                    power_usage=power_usage,
                    fan_speed=fan_speed
                ))
        
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
        
        return gpu_metrics

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        logger.info("内存监控器初始化完成")
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """获取内存指标"""
        try:
            # 虚拟内存
            virtual_mem = psutil.virtual_memory()
            
            # 交换空间
            swap_mem = psutil.swap_memory()
            
            return MemoryMetrics(
                total=virtual_mem.total / 1024 / 1024 / 1024,  # GB
                available=virtual_mem.available / 1024 / 1024 / 1024,  # GB
                used=virtual_mem.used / 1024 / 1024 / 1024,  # GB
                usage_percent=virtual_mem.percent,
                swap_total=swap_mem.total / 1024 / 1024 / 1024,  # GB
                swap_used=swap_mem.used / 1024 / 1024 / 1024,  # GB
                swap_percent=swap_mem.percent
            )
        
        except Exception as e:
            logger.error(f"获取内存指标失败: {e}")
            return MemoryMetrics(
                total=0, available=0, used=0, usage_percent=0,
                swap_total=0, swap_used=0, swap_percent=0
            )

class DiskMonitor:
    """磁盘监控器"""
    
    def __init__(self):
        self.previous_disk_io = {}
        logger.info("磁盘监控器初始化完成")
    
    def get_disk_metrics(self) -> List[DiskMetrics]:
        """获取磁盘指标"""
        disk_metrics = []
        
        try:
            # 磁盘使用情况
            disk_partitions = psutil.disk_partitions()
            
            for partition in disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    # 磁盘IO统计
                    disk_io = psutil.disk_io_counters(perdisk=True)
                    device_name = partition.device.replace('\\', '').replace(':', '')
                    
                    read_speed = write_speed = iops = 0
                    
                    if device_name in disk_io:
                        current_io = disk_io[device_name]
                        
                        if device_name in self.previous_disk_io:
                            prev_io = self.previous_disk_io[device_name]
                            time_delta = time.time() - prev_io['timestamp']
                            
                            if time_delta > 0:
                                read_speed = (current_io.read_bytes - prev_io['read_bytes']) / time_delta / 1024 / 1024  # MB/s
                                write_speed = (current_io.write_bytes - prev_io['write_bytes']) / time_delta / 1024 / 1024  # MB/s
                                iops = (current_io.read_count + current_io.write_count - prev_io['read_count'] - prev_io['write_count']) / time_delta
                        
                        self.previous_disk_io[device_name] = {
                            'read_bytes': current_io.read_bytes,
                            'write_bytes': current_io.write_bytes,
                            'read_count': current_io.read_count,
                            'write_count': current_io.write_count,
                            'timestamp': time.time()
                        }
                    
                    disk_metrics.append(DiskMetrics(
                        device=partition.device,
                        mountpoint=partition.mountpoint,
                        total=usage.total / 1024 / 1024 / 1024,  # GB
                        used=usage.used / 1024 / 1024 / 1024,  # GB
                        free=usage.free / 1024 / 1024 / 1024,  # GB
                        usage_percent=(usage.used / usage.total) * 100,
                        read_speed=max(0, read_speed),
                        write_speed=max(0, write_speed),
                        iops=max(0, iops)
                    ))
                
                except Exception as e:
                    logger.debug(f"获取磁盘 {partition.device} 指标失败: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"获取磁盘指标失败: {e}")
        
        return disk_metrics

class NetworkMonitor:
    """网络监控器"""
    
    def __init__(self):
        self.previous_net_io = {}
        logger.info("网络监控器初始化完成")
    
    def get_network_metrics(self) -> List[NetworkMetrics]:
        """获取网络指标"""
        network_metrics = []
        
        try:
            # 网络IO统计
            net_io = psutil.net_io_counters(pernic=True)
            
            # 网络连接数
            connections = len(psutil.net_connections())
            
            for interface, stats in net_io.items():
                send_speed = recv_speed = 0
                
                if interface in self.previous_net_io:
                    prev_stats = self.previous_net_io[interface]
                    time_delta = time.time() - prev_stats['timestamp']
                    
                    if time_delta > 0:
                        send_speed = (stats.bytes_sent - prev_stats['bytes_sent']) / time_delta / 1024 / 1024  # MB/s
                        recv_speed = (stats.bytes_recv - prev_stats['bytes_recv']) / time_delta / 1024 / 1024  # MB/s
                
                self.previous_net_io[interface] = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'timestamp': time.time()
                }
                
                network_metrics.append(NetworkMetrics(
                    interface=interface,
                    bytes_sent=stats.bytes_sent / 1024 / 1024,  # MB
                    bytes_recv=stats.bytes_recv / 1024 / 1024,  # MB
                    packets_sent=stats.packets_sent,
                    packets_recv=stats.packets_recv,
                    send_speed=max(0, send_speed),
                    recv_speed=max(0, recv_speed),
                    connections=connections
                ))
        
        except Exception as e:
            logger.error(f"获取网络指标失败: {e}")
        
        return network_metrics

class HardwareMonitor:
    """硬件监控器主类"""
    
    def __init__(self):
        self.cpu_monitor = CPUMonitor()
        self.gpu_monitor = GPUMonitor()
        self.memory_monitor = MemoryMonitor()
        self.disk_monitor = DiskMonitor()
        self.network_monitor = NetworkMonitor()
        
        # 告警配置
        self.alert_thresholds = {
            'cpu_usage': 90.0,  # CPU使用率告警阈值
            'cpu_temperature': 80.0,  # CPU温度告警阈值
            'gpu_usage': 95.0,  # GPU使用率告警阈值
            'gpu_temperature': 85.0,  # GPU温度告警阈值
            'memory_usage': 90.0,  # 内存使用率告警阈值
            'disk_usage': 85.0,  # 磁盘使用率告警阈值
            'network_speed': 100.0  # 网络速度告警阈值 (MB/s)
        }
        
        # 告警历史
        self.alerts: List[HardwareAlert] = []
        self.alert_callbacks: List[Callable[[HardwareAlert], None]] = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_interval = 5  # 监控间隔（秒）
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("硬件监控器初始化完成")
    
    def start_monitoring(self):
        """启动监控"""
        try:
            self.is_monitoring = True
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            logger.info("硬件监控启动")
        
        except Exception as e:
            logger.error(f"启动硬件监控失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("硬件监控停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 获取所有硬件指标
                metrics = self.get_all_metrics()
                
                # 检查告警
                self._check_alerts(metrics)
                
                # 等待下次监控
                time.sleep(self.monitor_interval)
            
            except Exception as e:
                logger.error(f"监控循环失败: {e}")
                time.sleep(self.monitor_interval)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有硬件指标"""
        try:
            with self.lock:
                return {
                    'cpu': self.cpu_monitor.get_cpu_metrics(),
                    'gpu': self.gpu_monitor.get_gpu_metrics(),
                    'memory': self.memory_monitor.get_memory_metrics(),
                    'disk': self.disk_monitor.get_disk_metrics(),
                    'network': self.network_monitor.get_network_metrics(),
                    'timestamp': time.time()
                }
        
        except Exception as e:
            logger.error(f"获取硬件指标失败: {e}")
            return {}
    
    def update_all_metrics(self) -> Dict[str, Any]:
        """更新并获取所有硬件指标"""
        try:
            with self.lock:
                # 更新各个监控器的指标
                cpu_metrics = self.cpu_monitor.get_cpu_metrics()
                gpu_metrics = self.gpu_monitor.get_gpu_metrics()
                memory_metrics = self.memory_monitor.get_memory_metrics()
                disk_metrics = self.disk_monitor.get_disk_metrics()
                network_metrics = self.network_monitor.get_network_metrics()
                
                # 组合所有指标
                all_metrics = {
                    'cpu': cpu_metrics,
                    'gpu': gpu_metrics,
                    'memory': memory_metrics,
                    'disk': disk_metrics,
                    'network': network_metrics,
                    'timestamp': time.time()
                }
                
                # 检查告警
                self._check_alerts(all_metrics)
                
                return all_metrics
        
        except Exception as e:
            logger.error(f"更新硬件指标失败: {e}")
            return {}
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警"""
        try:
            alerts = []
            
            # CPU告警检查
            cpu_metrics = metrics.get('cpu')
            if cpu_metrics:
                if cpu_metrics.usage_percent > self.alert_thresholds['cpu_usage']:
                    alerts.append(HardwareAlert(
                        alert_id=f"cpu_usage_{int(time.time())}",
                        resource_type=ResourceType.CPU,
                        alert_level=AlertLevel.WARNING,
                        message=f"CPU使用率过高: {cpu_metrics.usage_percent:.1f}%",
                        current_value=cpu_metrics.usage_percent,
                        threshold=self.alert_thresholds['cpu_usage']
                    ))
                
                if cpu_metrics.temperature > self.alert_thresholds['cpu_temperature']:
                    alerts.append(HardwareAlert(
                        alert_id=f"cpu_temp_{int(time.time())}",
                        resource_type=ResourceType.CPU,
                        alert_level=AlertLevel.CRITICAL,
                        message=f"CPU温度过高: {cpu_metrics.temperature:.1f}°C",
                        current_value=cpu_metrics.temperature,
                        threshold=self.alert_thresholds['cpu_temperature']
                    ))
            
            # GPU告警检查
            gpu_metrics_list = metrics.get('gpu', [])
            for gpu_metrics in gpu_metrics_list:
                if gpu_metrics.usage_percent > self.alert_thresholds['gpu_usage']:
                    alerts.append(HardwareAlert(
                        alert_id=f"gpu_{gpu_metrics.gpu_id}_usage_{int(time.time())}",
                        resource_type=ResourceType.GPU,
                        alert_level=AlertLevel.WARNING,
                        message=f"GPU{gpu_metrics.gpu_id}使用率过高: {gpu_metrics.usage_percent:.1f}%",
                        current_value=gpu_metrics.usage_percent,
                        threshold=self.alert_thresholds['gpu_usage']
                    ))
                
                if gpu_metrics.temperature > self.alert_thresholds['gpu_temperature']:
                    alerts.append(HardwareAlert(
                        alert_id=f"gpu_{gpu_metrics.gpu_id}_temp_{int(time.time())}",
                        resource_type=ResourceType.GPU,
                        alert_level=AlertLevel.CRITICAL,
                        message=f"GPU{gpu_metrics.gpu_id}温度过高: {gpu_metrics.temperature:.1f}°C",
                        current_value=gpu_metrics.temperature,
                        threshold=self.alert_thresholds['gpu_temperature']
                    ))
            
            # 内存告警检查
            memory_metrics = metrics.get('memory')
            if memory_metrics and memory_metrics.usage_percent > self.alert_thresholds['memory_usage']:
                alerts.append(HardwareAlert(
                    alert_id=f"memory_usage_{int(time.time())}",
                    resource_type=ResourceType.MEMORY,
                    alert_level=AlertLevel.WARNING,
                    message=f"内存使用率过高: {memory_metrics.usage_percent:.1f}%",
                    current_value=memory_metrics.usage_percent,
                    threshold=self.alert_thresholds['memory_usage']
                ))
            
            # 磁盘告警检查
            disk_metrics_list = metrics.get('disk', [])
            for disk_metrics in disk_metrics_list:
                if disk_metrics.usage_percent > self.alert_thresholds['disk_usage']:
                    alerts.append(HardwareAlert(
                        alert_id=f"disk_{disk_metrics.device.replace(':', '')}_{int(time.time())}",
                        resource_type=ResourceType.DISK,
                        alert_level=AlertLevel.WARNING,
                        message=f"磁盘{disk_metrics.device}使用率过高: {disk_metrics.usage_percent:.1f}%",
                        current_value=disk_metrics.usage_percent,
                        threshold=self.alert_thresholds['disk_usage']
                    ))
            
            # 处理告警
            for alert in alerts:
                self._process_alert(alert)
        
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
    
    def _process_alert(self, alert: HardwareAlert):
        """处理告警"""
        try:
            # 添加到告警历史
            self.alerts.append(alert)
            
            # 保持告警历史在合理范围内
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
            
            # 调用告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调执行失败: {e}")
            
            # 记录日志
            if alert.alert_level == AlertLevel.EMERGENCY:
                logger.critical(f"紧急硬件告警: {alert.message}")
            elif alert.alert_level == AlertLevel.CRITICAL:
                logger.error(f"关键硬件告警: {alert.message}")
            elif alert.alert_level == AlertLevel.WARNING:
                logger.warning(f"硬件告警: {alert.message}")
            else:
                logger.info(f"硬件信息: {alert.message}")
        
        except Exception as e:
            logger.error(f"处理告警失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[HardwareAlert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """设置告警阈值"""
        if metric_name in self.alert_thresholds:
            self.alert_thresholds[metric_name] = threshold
            logger.info(f"设置告警阈值: {metric_name} = {threshold}")
        else:
            logger.warning(f"未知的告警指标: {metric_name}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'gpu_count': self.gpu_monitor.gpu_count,
                'boot_time': psutil.boot_time(),
                'uptime': time.time() - psutil.boot_time()
            }
        
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {}
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """获取硬件摘要"""
        try:
            with self.lock:
                metrics = self.get_all_metrics()
                
                # 最近告警统计
                recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]
                alert_counts = {}
                for alert in recent_alerts:
                    level = alert.alert_level.value
                    alert_counts[level] = alert_counts.get(level, 0) + 1
                
                return {
                    'system_info': self.get_system_info(),
                    'current_metrics': metrics,
                    'alert_thresholds': self.alert_thresholds,
                    'total_alerts': len(self.alerts),
                    'recent_alerts': len(recent_alerts),
                    'alert_counts': alert_counts,
                    'monitoring_status': self.is_monitoring,
                    'monitor_interval': self.monitor_interval
                }
        
        except Exception as e:
            logger.error(f"获取硬件摘要失败: {e}")
            return {}
    
    def get_recent_alerts(self, limit: int = 50) -> List[HardwareAlert]:
        """获取最近的告警"""
        with self.lock:
            return sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

# 全局硬件监控器实例
hardware_monitor = HardwareMonitor()
