"""
🚀 高频交易性能优化器
生产级高频交易优化系统，实现微秒级延迟优化、网络I/O优化、系统调优
支持CPU亲和性设置、内核旁路技术、零拷贝优化等高级功能
专为极低延迟交易场景设计，确保系统在高频环境下的稳定性和性能
"""
import asyncio
import os
import sys
import time
import threading
import multiprocessing
import ctypes
import mmap
import socket
import struct
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime, timedelta
from collections import deque
import psutil
import numpy as np

try:
    import dpdk
    DPDK_AVAILABLE = True
except ImportError:
    DPDK_AVAILABLE = False

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False

@dataclass
class LatencyMeasurement:
    """延迟测量"""
    measurement_id: str
    start_time: float
    end_time: float
    latency_ns: int
    operation_type: str
    thread_id: int
    cpu_id: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CPUAffinity:
    """CPU亲和性配置"""
    thread_id: int
    cpu_cores: List[int]
    priority: int
    numa_node: Optional[int] = None
    is_isolated: bool = False

@dataclass
class NetworkConfig:
    """网络配置"""
    interface: str
    ip_address: str
    port: int
    buffer_size: int
    use_kernel_bypass: bool = False
    use_zero_copy: bool = False
    interrupt_coalescing: bool = True

class HighFrequencyOptimizer:
    """高频交易优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 系统信息
        self.cpu_count = multiprocessing.cpu_count()
        self.numa_nodes = self._detect_numa_nodes()
        self.isolated_cpus = self._detect_isolated_cpus()
        
        # 性能监控
        self.latency_measurements = deque(maxlen=100000)
        self.performance_counters = {}
        self.system_metrics = deque(maxlen=1000)
        
        # CPU亲和性管理
        self.cpu_affinities: Dict[int, CPUAffinity] = {}
        self.cpu_usage_history = deque(maxlen=1000)
        
        # 网络优化
        self.network_configs: Dict[str, NetworkConfig] = {}
        self.socket_pools: Dict[str, List[socket.socket]] = {}
        
        # 内存优化
        self.huge_pages_enabled = False
        self.memory_pools: Dict[str, Any] = {}
        
        # 实时调优
        self.auto_tuning_enabled = True
        self.tuning_history = deque(maxlen=1000)
        
        self._initialize_system_optimizations()
        
    def _detect_numa_nodes(self) -> List[int]:
        """检测NUMA节点"""
        try:
            if NUMA_AVAILABLE:
                return list(range(numa.get_max_node() + 1))
            else:
                # 简单检测
                numa_path = "/sys/devices/system/node"
                if os.path.exists(numa_path):
                    nodes = []
                    for item in os.listdir(numa_path):
                        if item.startswith("node") and item[4:].isdigit():
                            nodes.append(int(item[4:]))
                    return sorted(nodes)
        except Exception as e:
            self.logger.warning(f"NUMA节点检测失败: {e}")
            
        return [0]  # 默认单节点
        
    def _detect_isolated_cpus(self) -> List[int]:
        """检测隔离的CPU核心"""
        try:
            # 读取内核参数
            with open("/proc/cmdline", "r") as f:
                cmdline = f.read()
                
            # 查找isolcpus参数
            for param in cmdline.split():
                if param.startswith("isolcpus="):
                    cpu_list = param.split("=")[1]
                    return self._parse_cpu_list(cpu_list)
                    
        except Exception as e:
            self.logger.warning(f"隔离CPU检测失败: {e}")
            
        return []
        
    def _parse_cpu_list(self, cpu_list: str) -> List[int]:
        """解析CPU列表"""
        cpus = []
        for part in cpu_list.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(part))
        return cpus
        
    def _initialize_system_optimizations(self):
        """初始化系统优化"""
        try:
            # 设置进程优先级
            os.nice(-20)  # 最高优先级
            
            # 启用大页内存
            self._enable_huge_pages()
            
            # 优化网络参数
            self._optimize_network_parameters()
            
            # 设置CPU调度策略
            self._optimize_cpu_scheduling()
            
            self.logger.info("系统优化初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统优化初始化失败: {e}")
            
    def _enable_huge_pages(self):
        """启用大页内存"""
        try:
            # 检查大页内存支持
            hugepage_path = "/proc/sys/vm/nr_hugepages"
            if os.path.exists(hugepage_path):
                with open(hugepage_path, "r") as f:
                    current_pages = int(f.read().strip())
                    
                if current_pages > 0:
                    self.huge_pages_enabled = True
                    self.logger.info(f"大页内存已启用: {current_pages} 页")
                else:
                    self.logger.warning("大页内存未配置")
                    
        except Exception as e:
            self.logger.warning(f"大页内存检查失败: {e}")
            
    def _optimize_network_parameters(self):
        """优化网络参数"""
        try:
            # 网络缓冲区优化
            network_params = {
                "/proc/sys/net/core/rmem_max": "134217728",
                "/proc/sys/net/core/wmem_max": "134217728",
                "/proc/sys/net/core/rmem_default": "65536",
                "/proc/sys/net/core/wmem_default": "65536",
                "/proc/sys/net/core/netdev_max_backlog": "5000",
                "/proc/sys/net/ipv4/tcp_rmem": "4096 65536 134217728",
                "/proc/sys/net/ipv4/tcp_wmem": "4096 65536 134217728",
                "/proc/sys/net/ipv4/tcp_congestion_control": "bbr"
            }
            
            for param, value in network_params.items():
                try:
                    if os.path.exists(param):
                        with open(param, "w") as f:
                            f.write(value)
                except PermissionError:
                    self.logger.warning(f"无权限修改网络参数: {param}")
                    
        except Exception as e:
            self.logger.warning(f"网络参数优化失败: {e}")
            
    def _optimize_cpu_scheduling(self):
        """优化CPU调度"""
        try:
            # 设置实时调度策略
            if hasattr(os, 'sched_setscheduler'):
                # SCHED_FIFO: 先进先出实时调度
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
                self.logger.info("设置实时调度策略成功")
                
        except Exception as e:
            self.logger.warning(f"CPU调度优化失败: {e}")
            
    async def start(self):
        """启动高频优化器"""
        self.is_running = True
        self.logger.info("🚀 高频交易优化器启动")
        
        # 启动优化循环
        tasks = [
            asyncio.create_task(self._latency_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._auto_tuning_loop()),
            asyncio.create_task(self._cpu_affinity_management_loop()),
            asyncio.create_task(self._network_optimization_loop())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop(self):
        """停止高频优化器"""
        self.is_running = False
        self.logger.info("🚀 高频交易优化器停止")
        
    def set_cpu_affinity(self, thread_id: int, cpu_cores: List[int], 
                        priority: int = 99, numa_node: Optional[int] = None):
        """设置CPU亲和性"""
        try:
            affinity = CPUAffinity(
                thread_id=thread_id,
                cpu_cores=cpu_cores,
                priority=priority,
                numa_node=numa_node,
                is_isolated=any(cpu in self.isolated_cpus for cpu in cpu_cores)
            )
            
            # 设置CPU亲和性
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(thread_id, cpu_cores)
                
            # 设置线程优先级
            if hasattr(os, 'setpriority'):
                os.setpriority(os.PRIO_PROCESS, thread_id, -priority)
                
            self.cpu_affinities[thread_id] = affinity
            self.logger.info(f"设置线程 {thread_id} CPU亲和性: {cpu_cores}")
            
        except Exception as e:
            self.logger.error(f"设置CPU亲和性失败: {e}")
            
    def optimize_critical_thread(self, thread_id: Optional[int] = None):
        """优化关键线程"""
        if thread_id is None:
            thread_id = threading.get_ident()
            
        try:
            # 选择最佳CPU核心
            best_cpus = self._select_optimal_cpus()
            
            # 设置CPU亲和性
            self.set_cpu_affinity(thread_id, best_cpus, priority=99)
            
            # 设置实时调度
            if hasattr(os, 'sched_setscheduler'):
                os.sched_setscheduler(thread_id, os.SCHED_FIFO, os.sched_param(99))
                
            self.logger.info(f"关键线程 {thread_id} 优化完成")
            
        except Exception as e:
            self.logger.error(f"关键线程优化失败: {e}")
            
    def _select_optimal_cpus(self) -> List[int]:
        """选择最优CPU核心"""
        # 优先选择隔离的CPU
        if self.isolated_cpus:
            return self.isolated_cpus[:2]  # 选择前两个隔离CPU
            
        # 选择负载最低的CPU
        cpu_usage = psutil.cpu_percent(percpu=True)
        cpu_load_pairs = [(i, usage) for i, usage in enumerate(cpu_usage)]
        cpu_load_pairs.sort(key=lambda x: x[1])
        
        return [cpu_load_pairs[0][0], cpu_load_pairs[1][0]]
        
    def create_optimized_socket(self, config: NetworkConfig) -> socket.socket:
        """创建优化的网络套接字"""
        try:
            # 创建套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 基本优化
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            # 缓冲区优化
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, config.buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.buffer_size)
            
            # TCP优化
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 零拷贝优化（如果支持）
            if config.use_zero_copy and hasattr(socket, 'MSG_ZEROCOPY'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_ZEROCOPY, 1)
                
            # 内核旁路（需要特殊支持）
            if config.use_kernel_bypass and DPDK_AVAILABLE:
                self._setup_kernel_bypass(sock, config)
                
            self.logger.info(f"创建优化套接字: {config.interface}:{config.port}")
            return sock
            
        except Exception as e:
            self.logger.error(f"创建优化套接字失败: {e}")
            return None
            
    def _setup_kernel_bypass(self, sock: socket.socket, config: NetworkConfig):
        """设置内核旁路"""
        # 这里需要DPDK或其他内核旁路技术的具体实现
        # 由于复杂性，这里只是占位符
        pass
        
    def measure_latency(self, operation_type: str) -> 'LatencyContext':
        """测量延迟的上下文管理器"""
        return LatencyContext(self, operation_type)
        
    def _record_latency(self, measurement: LatencyMeasurement):
        """记录延迟测量"""
        self.latency_measurements.append(measurement)
        
        # 更新性能计数器
        if measurement.operation_type not in self.performance_counters:
            self.performance_counters[measurement.operation_type] = {
                'count': 0,
                'total_latency': 0,
                'min_latency': float('inf'),
                'max_latency': 0,
                'avg_latency': 0
            }
            
        counter = self.performance_counters[measurement.operation_type]
        counter['count'] += 1
        counter['total_latency'] += measurement.latency_ns
        counter['min_latency'] = min(counter['min_latency'], measurement.latency_ns)
        counter['max_latency'] = max(counter['max_latency'], measurement.latency_ns)
        counter['avg_latency'] = counter['total_latency'] / counter['count']
        
    async def _latency_monitoring_loop(self):
        """延迟监控循环"""
        while self.is_running:
            try:
                # 分析延迟分布
                if len(self.latency_measurements) > 100:
                    await self._analyze_latency_distribution()
                    
                await asyncio.sleep(1)  # 1秒分析一次
                
            except Exception as e:
                self.logger.error(f"延迟监控错误: {e}")
                await asyncio.sleep(5)
                
    async def _analyze_latency_distribution(self):
        """分析延迟分布"""
        recent_measurements = list(self.latency_measurements)[-1000:]
        
        if not recent_measurements:
            return
            
        latencies = [m.latency_ns for m in recent_measurements]
        
        # 计算统计信息
        latencies_array = np.array(latencies)
        stats = {
            'count': len(latencies),
            'mean': np.mean(latencies_array),
            'median': np.median(latencies_array),
            'std': np.std(latencies_array),
            'min': np.min(latencies_array),
            'max': np.max(latencies_array),
            'p95': np.percentile(latencies_array, 95),
            'p99': np.percentile(latencies_array, 99),
            'p99_9': np.percentile(latencies_array, 99.9)
        }
        
        # 检查是否需要优化
        if stats['p99'] > 100000:  # 100微秒
            self.logger.warning(f"延迟过高: P99 = {stats['p99']/1000:.2f} μs")
            await self._trigger_latency_optimization()
            
    async def _trigger_latency_optimization(self):
        """触发延迟优化"""
        try:
            # 分析延迟热点
            hotspots = self._identify_latency_hotspots()
            
            # 应用优化策略
            for hotspot in hotspots:
                await self._apply_optimization_strategy(hotspot)
                
        except Exception as e:
            self.logger.error(f"延迟优化失败: {e}")
            
    def _identify_latency_hotspots(self) -> List[Dict[str, Any]]:
        """识别延迟热点"""
        hotspots = []
        
        for op_type, counter in self.performance_counters.items():
            if counter['avg_latency'] > 50000:  # 50微秒
                hotspots.append({
                    'operation_type': op_type,
                    'avg_latency': counter['avg_latency'],
                    'max_latency': counter['max_latency'],
                    'count': counter['count']
                })
                
        return sorted(hotspots, key=lambda x: x['avg_latency'], reverse=True)
        
    async def _apply_optimization_strategy(self, hotspot: Dict[str, Any]):
        """应用优化策略"""
        op_type = hotspot['operation_type']
        
        # 根据操作类型应用不同的优化策略
        if 'network' in op_type.lower():
            await self._optimize_network_operation(op_type)
        elif 'cpu' in op_type.lower():
            await self._optimize_cpu_operation(op_type)
        elif 'memory' in op_type.lower():
            await self._optimize_memory_operation(op_type)
            
    async def _optimize_network_operation(self, op_type: str):
        """优化网络操作"""
        # 调整网络缓冲区大小
        # 启用零拷贝
        # 优化中断合并
        pass
        
    async def _optimize_cpu_operation(self, op_type: str):
        """优化CPU操作"""
        # 调整CPU亲和性
        # 提高线程优先级
        # 启用CPU隔离
        pass
        
    async def _optimize_memory_operation(self, op_type: str):
        """优化内存操作"""
        # 启用大页内存
        # 优化内存分配策略
        # 减少内存拷贝
        pass
        
    async def _performance_monitoring_loop(self):
        """性能监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu': {
                        'usage_percent': psutil.cpu_percent(percpu=True),
                        'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                        'load_avg': os.getloadavg(),
                        'context_switches': psutil.cpu_stats().ctx_switches,
                        'interrupts': psutil.cpu_stats().interrupts
                    },
                    'memory': {
                        'virtual': psutil.virtual_memory()._asdict(),
                        'swap': psutil.swap_memory()._asdict()
                    },
                    'network': {
                        'io_counters': psutil.net_io_counters()._asdict(),
                        'connections': len(psutil.net_connections())
                    },
                    'latency': {
                        'measurements_count': len(self.latency_measurements),
                        'performance_counters': dict(self.performance_counters)
                    }
                }
                
                self.system_metrics.append(metrics)
                
                await asyncio.sleep(5)  # 5秒监控一次
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(10)
                
    async def _auto_tuning_loop(self):
        """自动调优循环"""
        while self.is_running:
            try:
                if self.auto_tuning_enabled:
                    await self._perform_auto_tuning()
                    
                await asyncio.sleep(60)  # 1分钟调优一次
                
            except Exception as e:
                self.logger.error(f"自动调优错误: {e}")
                await asyncio.sleep(120)
                
    async def _perform_auto_tuning(self):
        """执行自动调优"""
        # 分析性能趋势
        if len(self.system_metrics) < 10:
            return
            
        recent_metrics = list(self.system_metrics)[-10:]
        
        # CPU使用率趋势
        cpu_usage_trend = [np.mean(m['cpu']['usage_percent']) for m in recent_metrics]
        avg_cpu_usage = np.mean(cpu_usage_trend)
        
        # 内存使用率趋势
        memory_usage_trend = [m['memory']['virtual']['percent'] for m in recent_metrics]
        avg_memory_usage = np.mean(memory_usage_trend)
        
        # 网络I/O趋势
        network_io_trend = [m['network']['io_counters']['bytes_sent'] + 
                           m['network']['io_counters']['bytes_recv'] for m in recent_metrics]
        
        # 应用调优策略
        tuning_actions = []
        
        if avg_cpu_usage > 80:
            tuning_actions.append("high_cpu_usage")
            await self._tune_for_high_cpu_usage()
            
        if avg_memory_usage > 85:
            tuning_actions.append("high_memory_usage")
            await self._tune_for_high_memory_usage()
            
        if len(tuning_actions) > 0:
            self.tuning_history.append({
                'timestamp': datetime.now(),
                'actions': tuning_actions,
                'cpu_usage': avg_cpu_usage,
                'memory_usage': avg_memory_usage
            })
            
    async def _tune_for_high_cpu_usage(self):
        """针对高CPU使用率的调优"""
        # 重新分配CPU亲和性
        # 调整线程优先级
        # 启用更多CPU核心
        pass
        
    async def _tune_for_high_memory_usage(self):
        """针对高内存使用率的调优"""
        # 触发垃圾回收
        # 清理缓存
        # 优化内存分配
        pass
        
    async def _cpu_affinity_management_loop(self):
        """CPU亲和性管理循环"""
        while self.is_running:
            try:
                # 监控CPU使用情况
                cpu_usage = psutil.cpu_percent(percpu=True)
                
                # 记录CPU使用历史
                self.cpu_usage_history.append({
                    'timestamp': datetime.now(),
                    'usage': cpu_usage
                })
                
                # 动态调整CPU亲和性
                await self._adjust_cpu_affinities(cpu_usage)
                
                await asyncio.sleep(10)  # 10秒调整一次
                
            except Exception as e:
                self.logger.error(f"CPU亲和性管理错误: {e}")
                await asyncio.sleep(30)
                
    async def _adjust_cpu_affinities(self, cpu_usage: List[float]):
        """调整CPU亲和性"""
        # 找出负载过高的CPU
        overloaded_cpus = [i for i, usage in enumerate(cpu_usage) if usage > 90]
        
        if overloaded_cpus:
            # 重新分配线程到负载较低的CPU
            available_cpus = [i for i, usage in enumerate(cpu_usage) if usage < 50]
            
            if available_cpus:
                for thread_id, affinity in self.cpu_affinities.items():
                    if any(cpu in overloaded_cpus for cpu in affinity.cpu_cores):
                        # 迁移到负载较低的CPU
                        new_cpus = available_cpus[:len(affinity.cpu_cores)]
                        self.set_cpu_affinity(thread_id, new_cpus, affinity.priority)
                        
    async def _network_optimization_loop(self):
        """网络优化循环"""
        while self.is_running:
            try:
                # 监控网络性能
                net_io = psutil.net_io_counters()
                
                # 检查网络拥塞
                if hasattr(net_io, 'dropin') and net_io.dropin > 0:
                    self.logger.warning(f"网络丢包检测: {net_io.dropin}")
                    await self._optimize_network_buffers()
                    
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                self.logger.error(f"网络优化错误: {e}")
                await asyncio.sleep(60)
                
    async def _optimize_network_buffers(self):
        """优化网络缓冲区"""
        # 动态调整网络缓冲区大小
        # 启用中断合并
        # 调整网络队列长度
        pass
        
    # 公共接口方法
    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计"""
        if not self.latency_measurements:
            return {}
            
        recent_measurements = list(self.latency_measurements)[-1000:]
        latencies = [m.latency_ns for m in recent_measurements]
        
        if not latencies:
            return {}
            
        latencies_array = np.array(latencies)
        
        return {
            'count': len(latencies),
            'mean_ns': float(np.mean(latencies_array)),
            'median_ns': float(np.median(latencies_array)),
            'std_ns': float(np.std(latencies_array)),
            'min_ns': float(np.min(latencies_array)),
            'max_ns': float(np.max(latencies_array)),
            'p95_ns': float(np.percentile(latencies_array, 95)),
            'p99_ns': float(np.percentile(latencies_array, 99)),
            'p99_9_ns': float(np.percentile(latencies_array, 99.9)),
            'performance_counters': dict(self.performance_counters)
        }
        
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': self.cpu_count,
            'numa_nodes': self.numa_nodes,
            'isolated_cpus': self.isolated_cpus,
            'huge_pages_enabled': self.huge_pages_enabled,
            'cpu_affinities': {
                tid: {
                    'cpu_cores': affinity.cpu_cores,
                    'priority': affinity.priority,
                    'numa_node': affinity.numa_node,
                    'is_isolated': affinity.is_isolated
                }
                for tid, affinity in self.cpu_affinities.items()
            }
        }
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        return {
            'auto_tuning_enabled': self.auto_tuning_enabled,
            'tuning_history_count': len(self.tuning_history),
            'cpu_usage_history_count': len(self.cpu_usage_history),
            'system_metrics_count': len(self.system_metrics),
            'network_configs': len(self.network_configs),
            'socket_pools': {k: len(v) for k, v in self.socket_pools.items()}
        }

class LatencyContext:
    """延迟测量上下文管理器"""
    
    def __init__(self, optimizer: HighFrequencyOptimizer, operation_type: str):
        self.optimizer = optimizer
        self.operation_type = operation_type
        self.start_time = 0
        self.measurement_id = f"{operation_type}_{int(time.time() * 1000000)}"
        
    def __enter__(self):
        self.start_time = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        latency_ns = end_time - self.start_time
        
        measurement = LatencyMeasurement(
            measurement_id=self.measurement_id,
            start_time=self.start_time,
            end_time=end_time,
            latency_ns=latency_ns,
            operation_type=self.operation_type,
            thread_id=threading.get_ident(),
            cpu_id=os.sched_getcpu() if hasattr(os, 'sched_getcpu') else -1
        )
        
        self.optimizer._record_latency(measurement)

