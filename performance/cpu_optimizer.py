"""
⚡ CPU性能优化系统
针对20核CPU的高频交易优化，实现CPU亲和性绑定、进程调度优化、NUMA优化
支持实时性能监控、动态负载均衡和CPU资源管理
"""

import os
import psutil
import threading
import multiprocessing
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from loguru import logger


class CPUPriority(Enum):
    """CPU优先级"""
    REALTIME = "realtime"       # 实时优先级
    HIGH = "high"               # 高优先级
    NORMAL = "normal"           # 普通优先级
    LOW = "low"                 # 低优先级


class ProcessType(Enum):
    """进程类型"""
    TRADING_ENGINE = "trading_engine"       # 交易引擎
    MARKET_DATA = "market_data"             # 市场数据
    STRATEGY = "strategy"                   # 策略计算
    RISK_MANAGEMENT = "risk_management"     # 风险管理
    ORDER_MANAGEMENT = "order_management"   # 订单管理
    MONITORING = "monitoring"               # 监控系统
    LOGGING = "logging"                     # 日志系统
    GENERAL = "general"                     # 通用任务


@dataclass
class CPUCore:
    """CPU核心信息"""
    core_id: int                            # 核心ID
    physical_id: int                        # 物理CPU ID
    is_hyperthread: bool                    # 是否超线程
    frequency: float                        # 频率(MHz)
    usage: float = 0.0                      # 使用率
    temperature: float = 0.0                # 温度
    assigned_processes: Set[int] = field(default_factory=set)  # 分配的进程


@dataclass
class ProcessConfig:
    """进程配置"""
    process_type: ProcessType               # 进程类型
    priority: CPUPriority                   # 优先级
    cpu_affinity: List[int]                 # CPU亲和性
    memory_limit: Optional[int] = None      # 内存限制(MB)
    nice_value: int = 0                     # Nice值
    ionice_class: int = 1                   # IO优先级类
    ionice_value: int = 4                   # IO优先级值


class CPUTopologyAnalyzer:
    """CPU拓扑分析器"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.cores: Dict[int, CPUCore] = {}
        self.numa_nodes: Dict[int, List[int]] = {}
        self._analyze_topology()
        
    def _analyze_topology(self):
        """分析CPU拓扑结构"""
        try:
            # 获取CPU信息
            cpu_info = self._get_cpu_info()
            
            # 分析每个逻辑CPU
            for cpu_id in range(self.cpu_count):
                core_info = cpu_info.get(cpu_id, {})
                
                # 创建CPU核心对象
                core = CPUCore(
                    core_id=cpu_id,
                    physical_id=core_info.get('physical_id', cpu_id // 2),
                    is_hyperthread=self._is_hyperthread(cpu_id),
                    frequency=self._get_cpu_frequency(cpu_id)
                )
                
                self.cores[cpu_id] = core
            
            # 分析NUMA节点
            self._analyze_numa_topology()
            
            logger.info(f"CPU拓扑分析完成: {self.cpu_count}逻辑核心, {self.physical_cpu_count}物理核心")
            
        except Exception as e:
            logger.error(f"CPU拓扑分析失败: {e}")
    
    def _get_cpu_info(self) -> Dict[int, Dict[str, Any]]:
        """获取CPU信息"""
        cpu_info = {}
        
        try:
            # 尝试从/proc/cpuinfo读取
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    current_cpu = {}
                    for line in f:
                        if line.strip() == '':
                            if 'processor' in current_cpu:
                                cpu_id = int(current_cpu['processor'])
                                cpu_info[cpu_id] = current_cpu.copy()
                            current_cpu = {}
                        else:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                current_cpu[key] = value
        except Exception as e:
            logger.warning(f"读取CPU信息失败: {e}")
        
        return cpu_info
    
    def _is_hyperthread(self, cpu_id: int) -> bool:
        """判断是否为超线程"""
        # 简单判断：如果逻辑核心数是物理核心数的2倍，则有超线程
        return self.cpu_count == self.physical_cpu_count * 2 and cpu_id >= self.physical_cpu_count
    
    def _get_cpu_frequency(self, cpu_id: int) -> float:
        """获取CPU频率"""
        try:
            freq_info = psutil.cpu_freq(percpu=True)
            if freq_info and cpu_id < len(freq_info):
                return freq_info[cpu_id].current
        except Exception:
            pass
        
        return 0.0
    
    def _analyze_numa_topology(self):
        """分析NUMA拓扑"""
        try:
            # 尝试读取NUMA信息
            numa_path = '/sys/devices/system/node'
            if os.path.exists(numa_path):
                for node_dir in os.listdir(numa_path):
                    if node_dir.startswith('node'):
                        node_id = int(node_dir[4:])
                        cpulist_path = os.path.join(numa_path, node_dir, 'cpulist')
                        
                        if os.path.exists(cpulist_path):
                            with open(cpulist_path, 'r') as f:
                                cpulist = f.read().strip()
                                cpu_ids = self._parse_cpu_list(cpulist)
                                self.numa_nodes[node_id] = cpu_ids
            
            # 如果没有NUMA信息，创建单个节点
            if not self.numa_nodes:
                self.numa_nodes[0] = list(range(self.cpu_count))
                
        except Exception as e:
            logger.warning(f"NUMA拓扑分析失败: {e}")
            self.numa_nodes[0] = list(range(self.cpu_count))
    
    def _parse_cpu_list(self, cpulist: str) -> List[int]:
        """解析CPU列表字符串"""
        cpu_ids = []
        
        for part in cpulist.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                cpu_ids.extend(range(start, end + 1))
            else:
                cpu_ids.append(int(part))
        
        return cpu_ids
    
    def get_physical_cores(self) -> List[int]:
        """获取物理核心列表"""
        return [core_id for core_id, core in self.cores.items() if not core.is_hyperthread]
    
    def get_hyperthread_cores(self) -> List[int]:
        """获取超线程核心列表"""
        return [core_id for core_id, core in self.cores.items() if core.is_hyperthread]
    
    def get_numa_nodes(self) -> Dict[int, List[int]]:
        """获取NUMA节点信息"""
        return self.numa_nodes.copy()
    
    def get_core_siblings(self, core_id: int) -> List[int]:
        """获取核心的兄弟核心（同一物理核心的超线程）"""
        if core_id not in self.cores:
            return []
        
        physical_id = self.cores[core_id].physical_id
        siblings = []
        
        for cid, core in self.cores.items():
            if core.physical_id == physical_id and cid != core_id:
                siblings.append(cid)
        
        return siblings


class CPUAffinityManager:
    """CPU亲和性管理器"""
    
    def __init__(self, topology: CPUTopologyAnalyzer):
        self.topology = topology
        self.process_assignments: Dict[int, ProcessConfig] = {}
        self.core_allocations: Dict[int, Set[int]] = {
            core_id: set() for core_id in range(topology.cpu_count)
        }
        
        # 预定义的进程类型CPU分配策略
        self.allocation_strategy = self._create_allocation_strategy()
        
    def _create_allocation_strategy(self) -> Dict[ProcessType, Dict[str, Any]]:
        """创建CPU分配策略"""
        physical_cores = self.topology.get_physical_cores()
        hyperthread_cores = self.topology.get_hyperthread_cores()
        
        # 为不同类型的进程分配不同的CPU核心
        strategy = {
            ProcessType.TRADING_ENGINE: {
                'cores': physical_cores[:4],  # 前4个物理核心
                'priority': CPUPriority.REALTIME,
                'exclusive': True
            },
            ProcessType.MARKET_DATA: {
                'cores': physical_cores[4:8],  # 接下来4个物理核心
                'priority': CPUPriority.HIGH,
                'exclusive': True
            },
            ProcessType.STRATEGY: {
                'cores': physical_cores[8:12],  # 策略计算核心
                'priority': CPUPriority.HIGH,
                'exclusive': False
            },
            ProcessType.RISK_MANAGEMENT: {
                'cores': physical_cores[12:14],  # 风险管理核心
                'priority': CPUPriority.HIGH,
                'exclusive': False
            },
            ProcessType.ORDER_MANAGEMENT: {
                'cores': physical_cores[14:16],  # 订单管理核心
                'priority': CPUPriority.NORMAL,
                'exclusive': False
            },
            ProcessType.MONITORING: {
                'cores': hyperthread_cores[:4],  # 超线程核心用于监控
                'priority': CPUPriority.LOW,
                'exclusive': False
            },
            ProcessType.LOGGING: {
                'cores': hyperthread_cores[4:6],  # 日志系统
                'priority': CPUPriority.LOW,
                'exclusive': False
            },
            ProcessType.GENERAL: {
                'cores': hyperthread_cores[6:],  # 其余超线程核心
                'priority': CPUPriority.NORMAL,
                'exclusive': False
            }
        }
        
        return strategy
    
    def assign_process(self, pid: int, process_type: ProcessType) -> bool:
        """为进程分配CPU亲和性"""
        try:
            strategy = self.allocation_strategy.get(process_type)
            if not strategy:
                logger.error(f"未找到进程类型 {process_type} 的分配策略")
                return False
            
            # 获取可用的CPU核心
            available_cores = self._get_available_cores(strategy['cores'], strategy['exclusive'])
            if not available_cores:
                logger.warning(f"没有可用的CPU核心分配给进程 {pid}")
                return False
            
            # 创建进程配置
            config = ProcessConfig(
                process_type=process_type,
                priority=strategy['priority'],
                cpu_affinity=available_cores,
                nice_value=self._get_nice_value(strategy['priority']),
                ionice_class=1,
                ionice_value=4
            )
            
            # 应用CPU亲和性
            if self._apply_cpu_affinity(pid, config):
                self.process_assignments[pid] = config
                
                # 更新核心分配
                for core_id in available_cores:
                    self.core_allocations[core_id].add(pid)
                
                logger.info(f"进程 {pid} ({process_type.value}) 分配到CPU核心: {available_cores}")
                return True
            
        except Exception as e:
            logger.error(f"分配进程CPU亲和性失败: {e}")
        
        return False
    
    def _get_available_cores(self, preferred_cores: List[int], exclusive: bool) -> List[int]:
        """获取可用的CPU核心"""
        available = []
        
        for core_id in preferred_cores:
            if core_id >= self.topology.cpu_count:
                continue
            
            # 如果需要独占且核心已被占用，跳过
            if exclusive and self.core_allocations[core_id]:
                continue
            
            available.append(core_id)
        
        return available
    
    def _get_nice_value(self, priority: CPUPriority) -> int:
        """获取Nice值"""
        nice_values = {
            CPUPriority.REALTIME: -20,
            CPUPriority.HIGH: -10,
            CPUPriority.NORMAL: 0,
            CPUPriority.LOW: 10
        }
        return nice_values.get(priority, 0)
    
    def _apply_cpu_affinity(self, pid: int, config: ProcessConfig) -> bool:
        """应用CPU亲和性设置"""
        try:
            process = psutil.Process(pid)
            
            # 设置CPU亲和性
            process.cpu_affinity(config.cpu_affinity)
            
            # 设置进程优先级
            if config.priority == CPUPriority.REALTIME:
                # 实时优先级需要特殊处理
                try:
                    os.system(f"chrt -f -p 99 {pid}")
                except Exception as e:
                    logger.warning(f"设置实时优先级失败: {e}")
            else:
                process.nice(config.nice_value)
            
            # 设置IO优先级
            try:
                os.system(f"ionice -c {config.ionice_class} -n {config.ionice_value} -p {pid}")
            except Exception as e:
                logger.warning(f"设置IO优先级失败: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"应用CPU亲和性失败: {e}")
            return False
    
    def remove_process(self, pid: int):
        """移除进程分配"""
        try:
            if pid in self.process_assignments:
                config = self.process_assignments[pid]
                
                # 从核心分配中移除
                for core_id in config.cpu_affinity:
                    if core_id in self.core_allocations:
                        self.core_allocations[core_id].discard(pid)
                
                # 移除进程配置
                del self.process_assignments[pid]
                
                logger.info(f"移除进程 {pid} 的CPU分配")
                
        except Exception as e:
            logger.error(f"移除进程分配失败: {e}")
    
    def get_core_usage(self) -> Dict[int, Dict[str, Any]]:
        """获取核心使用情况"""
        usage = {}
        
        for core_id in range(self.topology.cpu_count):
            assigned_processes = self.core_allocations[core_id]
            core_info = self.topology.cores[core_id]
            
            usage[core_id] = {
                'physical_id': core_info.physical_id,
                'is_hyperthread': core_info.is_hyperthread,
                'frequency': core_info.frequency,
                'assigned_processes': len(assigned_processes),
                'process_pids': list(assigned_processes),
                'usage_percent': psutil.cpu_percent(percpu=True)[core_id] if core_id < len(psutil.cpu_percent(percpu=True)) else 0.0
            }
        
        return usage


class CPUPerformanceMonitor:
    """CPU性能监控器"""
    
    def __init__(self, topology: CPUTopologyAnalyzer, affinity_manager: CPUAffinityManager):
        self.topology = topology
        self.affinity_manager = affinity_manager
        self.monitoring = False
        self.monitor_task = None
        self.performance_data: List[Dict[str, Any]] = []
        self.max_history = 1000
        
    async def start_monitoring(self, interval: float = 1.0):
        """启动性能监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("CPU性能监控已启动")
    
    async def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("CPU性能监控已停止")
    
    async def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集性能数据
                data = self._collect_performance_data()
                
                # 添加到历史记录
                self.performance_data.append(data)
                if len(self.performance_data) > self.max_history:
                    self.performance_data.pop(0)
                
                # 检查性能异常
                self._check_performance_anomalies(data)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CPU性能监控失败: {e}")
                await asyncio.sleep(5)
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """收集性能数据"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(percpu=True)
            
            # 负载平均值
            load_avg = psutil.getloadavg()
            
            # CPU频率
            cpu_freq = psutil.cpu_freq(percpu=True)
            
            # 上下文切换
            cpu_stats = psutil.cpu_stats()
            
            # 核心使用情况
            core_usage = self.affinity_manager.get_core_usage()
            
            data = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                },
                'cpu_frequency': [freq.current if freq else 0.0 for freq in cpu_freq] if cpu_freq else [],
                'context_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'soft_interrupts': cpu_stats.soft_interrupts,
                'core_usage': core_usage,
                'total_processes': len(psutil.pids())
            }
            
            return data
            
        except Exception as e:
            logger.error(f"收集性能数据失败: {e}")
            return {'timestamp': time.time(), 'error': str(e)}
    
    def _check_performance_anomalies(self, data: Dict[str, Any]):
        """检查性能异常"""
        try:
            # 检查CPU使用率异常
            if 'cpu_percent' in data:
                high_usage_cores = [
                    i for i, usage in enumerate(data['cpu_percent'])
                    if usage > 95.0
                ]
                
                if high_usage_cores:
                    logger.warning(f"CPU核心使用率过高: {high_usage_cores}")
            
            # 检查负载异常
            if 'load_average' in data:
                load_1min = data['load_average']['1min']
                if load_1min > self.topology.cpu_count * 0.8:
                    logger.warning(f"系统负载过高: {load_1min}")
            
            # 检查频率异常
            if 'cpu_frequency' in data:
                low_freq_cores = [
                    i for i, freq in enumerate(data['cpu_frequency'])
                    if freq > 0 and freq < 2000  # 低于2GHz
                ]
                
                if low_freq_cores:
                    logger.warning(f"CPU频率过低: {low_freq_cores}")
            
        except Exception as e:
            logger.error(f"检查性能异常失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_data:
            return {}
        
        try:
            recent_data = self.performance_data[-60:]  # 最近60个数据点
            
            # 计算平均值
            avg_cpu_usage = 0.0
            avg_load = 0.0
            
            if NUMPY_AVAILABLE:
                cpu_usages = []
                loads = []
                
                for data in recent_data:
                    if 'cpu_percent' in data:
                        cpu_usages.extend(data['cpu_percent'])
                    if 'load_average' in data:
                        loads.append(data['load_average']['1min'])
                
                if cpu_usages:
                    avg_cpu_usage = np.mean(cpu_usages)
                if loads:
                    avg_load = np.mean(loads)
            
            return {
                'monitoring_active': self.monitoring,
                'data_points': len(self.performance_data),
                'average_cpu_usage': avg_cpu_usage,
                'average_load': avg_load,
                'core_count': self.topology.cpu_count,
                'physical_cores': len(self.topology.get_physical_cores()),
                'assigned_processes': len(self.affinity_manager.process_assignments)
            }
            
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {'error': str(e)}


class CPUOptimizer:
    """CPU优化器主类"""
    
    def __init__(self):
        self.topology = CPUTopologyAnalyzer()
        self.affinity_manager = CPUAffinityManager(self.topology)
        self.performance_monitor = CPUPerformanceMonitor(self.topology, self.affinity_manager)
        
        # 系统优化设置
        self._apply_system_optimizations()
        
        logger.info("CPU优化器初始化完成")
    
    def _apply_system_optimizations(self):
        """应用系统级优化"""
        try:
            # 设置CPU调度器
            self._set_cpu_governor()
            
            # 禁用不必要的服务
            self._disable_unnecessary_services()
            
            # 优化内核参数
            self._optimize_kernel_parameters()
            
            logger.info("系统级CPU优化已应用")
            
        except Exception as e:
            logger.error(f"应用系统优化失败: {e}")
    
    def _set_cpu_governor(self):
        """设置CPU调度器为性能模式"""
        try:
            # 设置为performance模式
            for cpu_id in range(self.topology.cpu_count):
                governor_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
                if os.path.exists(governor_path):
                    with open(governor_path, 'w') as f:
                        f.write('performance')
            
            logger.info("CPU调度器设置为性能模式")
            
        except Exception as e:
            logger.warning(f"设置CPU调度器失败: {e}")
    
    def _disable_unnecessary_services(self):
        """禁用不必要的服务"""
        try:
            # 禁用CPU节能功能
            services_to_disable = [
                'irqbalance',  # IRQ平衡服务
                'cpuspeed',    # CPU速度调节
            ]
            
            for service in services_to_disable:
                try:
                    os.system(f"systemctl stop {service} 2>/dev/null")
                    os.system(f"systemctl disable {service} 2>/dev/null")
                except Exception:
                    pass
            
            logger.info("不必要的服务已禁用")
            
        except Exception as e:
            logger.warning(f"禁用服务失败: {e}")
    
    def _optimize_kernel_parameters(self):
        """优化内核参数"""
        try:
            # 优化调度参数
            kernel_params = {
                '/proc/sys/kernel/sched_migration_cost_ns': '5000000',  # 调度迁移成本
                '/proc/sys/kernel/sched_min_granularity_ns': '10000000',  # 最小调度粒度
                '/proc/sys/kernel/sched_wakeup_granularity_ns': '15000000',  # 唤醒粒度
            }
            
            for param_path, value in kernel_params.items():
                try:
                    if os.path.exists(param_path):
                        with open(param_path, 'w') as f:
                            f.write(value)
                except Exception:
                    pass
            
            logger.info("内核参数优化完成")
            
        except Exception as e:
            logger.warning(f"优化内核参数失败: {e}")
    
    def optimize_process(self, pid: int, process_type: ProcessType) -> bool:
        """优化进程"""
        return self.affinity_manager.assign_process(pid, process_type)
    
    def optimize_current_process(self, process_type: ProcessType) -> bool:
        """优化当前进程"""
        return self.optimize_process(os.getpid(), process_type)
    
    async def start_monitoring(self, interval: float = 1.0):
        """启动性能监控"""
        await self.performance_monitor.start_monitoring(interval)
    
    async def stop_monitoring(self):
        """停止性能监控"""
        await self.performance_monitor.stop_monitoring()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'topology': {
                'total_cores': self.topology.cpu_count,
                'physical_cores': len(self.topology.get_physical_cores()),
                'hyperthread_cores': len(self.topology.get_hyperthread_cores()),
                'numa_nodes': len(self.topology.numa_nodes)
            },
            'assignments': {
                'total_processes': len(self.affinity_manager.process_assignments),
                'core_usage': self.affinity_manager.get_core_usage()
            },
            'performance': self.performance_monitor.get_performance_summary()
        }


# 全局CPU优化器实例
cpu_optimizer = CPUOptimizer()
