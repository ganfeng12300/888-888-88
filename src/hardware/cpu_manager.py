"""
🔥 CPU核心分配管理器
生产级20核心CPU资源精确分配和性能优化系统
实现CPU亲和性绑定、负载均衡和动态调度
"""

import os
import psutil
import threading
import multiprocessing
import time
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess
from loguru import logger


class CPUTaskType(Enum):
    """CPU任务类型"""
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    AI_TRAINING_LIGHT = "ai_training_light"
    AI_TRAINING_MEDIUM = "ai_training_medium"
    SYSTEM_MONITORING = "system_monitoring"
    TRADING_EXECUTION = "trading_execution"
    BACKGROUND = "background"


@dataclass
class CPUCoreAllocation:
    """CPU核心分配配置"""
    task_type: CPUTaskType
    core_ids: List[int]
    priority: int  # 进程优先级 (-20 到 19)
    max_threads: int
    description: str
    active_processes: Set[int] = field(default_factory=set)


@dataclass
class CPUPerformanceMetrics:
    """CPU性能指标"""
    timestamp: float
    total_usage: float
    per_core_usage: List[float]
    frequency: float
    temperature: float
    context_switches: int
    interrupts: int
    load_average: Tuple[float, float, float]


class CPUCoreManager:
    """CPU核心管理器"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)
        self.monitoring = False
        
        # 核心分配策略 (基于20核心配置)
        self.core_allocations = {
            CPUTaskType.DATA_COLLECTION: CPUCoreAllocation(
                task_type=CPUTaskType.DATA_COLLECTION,
                core_ids=[0, 1, 2, 3],  # 核心1-4
                priority=-5,  # 高优先级
                max_threads=4,
                description="实时数据采集和预处理"
            ),
            CPUTaskType.FEATURE_ENGINEERING: CPUCoreAllocation(
                task_type=CPUTaskType.FEATURE_ENGINEERING,
                core_ids=[4, 5, 6, 7],  # 核心5-8
                priority=-3,
                max_threads=4,
                description="特征工程和数据清洗"
            ),
            CPUTaskType.AI_TRAINING_LIGHT: CPUCoreAllocation(
                task_type=CPUTaskType.AI_TRAINING_LIGHT,
                core_ids=[8, 9, 10, 11],  # 核心9-12
                priority=0,
                max_threads=4,
                description="轻量级AI模型训练"
            ),
            CPUTaskType.AI_TRAINING_MEDIUM: CPUCoreAllocation(
                task_type=CPUTaskType.AI_TRAINING_MEDIUM,
                core_ids=[12, 13, 14, 15],  # 核心13-16
                priority=0,
                max_threads=4,
                description="中等AI模型训练"
            ),
            CPUTaskType.SYSTEM_MONITORING: CPUCoreAllocation(
                task_type=CPUTaskType.SYSTEM_MONITORING,
                core_ids=[16, 17],  # 核心17-18
                priority=-10,  # 最高优先级
                max_threads=2,
                description="系统监控和风控计算"
            ),
            CPUTaskType.TRADING_EXECUTION: CPUCoreAllocation(
                task_type=CPUTaskType.TRADING_EXECUTION,
                core_ids=[18, 19],  # 核心19-20
                priority=-10,  # 最高优先级
                max_threads=2,
                description="交易执行和订单管理"
            ),
        }
        
        # 性能监控数据
        self.performance_history: List[CPUPerformanceMetrics] = []
        self.max_history_size = 3600  # 1小时历史数据
        
        # 进程管理
        self.managed_processes: Dict[int, CPUTaskType] = {}
        self.process_lock = threading.Lock()
        
        logger.info(f"CPU核心管理器初始化完成，检测到 {self.cpu_count} 个逻辑核心，{self.physical_cores} 个物理核心")
    
    def allocate_process_to_cores(self, pid: int, task_type: CPUTaskType) -> bool:
        """将进程分配到指定的CPU核心"""
        try:
            if task_type not in self.core_allocations:
                logger.error(f"未知的任务类型: {task_type}")
                return False
            
            allocation = self.core_allocations[task_type]
            
            # 检查是否超过最大线程数
            if len(allocation.active_processes) >= allocation.max_threads:
                logger.warning(f"任务类型 {task_type.value} 已达到最大线程数限制 ({allocation.max_threads})")
                return False
            
            # 设置CPU亲和性
            process = psutil.Process(pid)
            process.cpu_affinity(allocation.core_ids)
            
            # 设置进程优先级
            if os.name == 'posix':  # Linux/Unix
                os.setpriority(os.PRIO_PROCESS, pid, allocation.priority)
            else:  # Windows
                process.nice(allocation.priority)
            
            # 记录进程分配
            with self.process_lock:
                allocation.active_processes.add(pid)
                self.managed_processes[pid] = task_type
            
            logger.info(f"进程 {pid} 已分配到 {task_type.value} 核心: {allocation.core_ids}")
            return True
            
        except Exception as e:
            logger.error(f"分配进程 {pid} 到核心失败: {e}")
            return False
    
    def deallocate_process(self, pid: int) -> bool:
        """释放进程的核心分配"""
        try:
            with self.process_lock:
                if pid not in self.managed_processes:
                    return False
                
                task_type = self.managed_processes[pid]
                allocation = self.core_allocations[task_type]
                
                allocation.active_processes.discard(pid)
                del self.managed_processes[pid]
            
            logger.info(f"进程 {pid} 的核心分配已释放")
            return True
            
        except Exception as e:
            logger.error(f"释放进程 {pid} 的核心分配失败: {e}")
            return False
    
    def create_optimized_process(self, target_function, task_type: CPUTaskType, 
                                args: Tuple = (), kwargs: Dict = None) -> Optional[multiprocessing.Process]:
        """创建优化的进程"""
        try:
            if kwargs is None:
                kwargs = {}
            
            # 创建进程
            process = multiprocessing.Process(target=target_function, args=args, kwargs=kwargs)
            process.start()
            
            # 分配CPU核心
            if self.allocate_process_to_cores(process.pid, task_type):
                logger.info(f"创建并优化进程 {process.pid} 用于 {task_type.value}")
                return process
            else:
                process.terminate()
                return None
                
        except Exception as e:
            logger.error(f"创建优化进程失败: {e}")
            return None
    
    async def start_performance_monitoring(self, interval: float = 1.0):
        """启动CPU性能监控"""
        self.monitoring = True
        logger.info("开始CPU性能监控...")
        
        while self.monitoring:
            try:
                metrics = await self._collect_cpu_metrics()
                self._store_performance_metrics(metrics)
                
                # 检查性能异常
                await self._check_performance_issues(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"CPU性能监控出错: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_cpu_metrics(self) -> CPUPerformanceMetrics:
        """收集CPU性能指标"""
        timestamp = time.time()
        
        # CPU使用率
        total_usage = psutil.cpu_percent(interval=0.1)
        per_core_usage = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # CPU频率
        cpu_freq = psutil.cpu_freq()
        frequency = cpu_freq.current if cpu_freq else 0
        
        # CPU温度
        temperature = await self._get_cpu_temperature()
        
        # 系统统计
        cpu_stats = psutil.cpu_stats()
        context_switches = cpu_stats.ctx_switches
        interrupts = cpu_stats.interrupts
        
        # 负载平均值
        if hasattr(os, 'getloadavg'):
            load_average = os.getloadavg()
        else:
            load_average = (0.0, 0.0, 0.0)
        
        return CPUPerformanceMetrics(
            timestamp=timestamp,
            total_usage=total_usage,
            per_core_usage=per_core_usage,
            frequency=frequency,
            temperature=temperature,
            context_switches=context_switches,
            interrupts=interrupts,
            load_average=load_average
        )
    
    async def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if 'coretemp' in name.lower() or 'cpu' in name.lower():
                            for entry in entries:
                                if entry.current:
                                    return entry.current
            return 0.0
        except:
            return 0.0
    
    def _store_performance_metrics(self, metrics: CPUPerformanceMetrics):
        """存储性能指标"""
        self.performance_history.append(metrics)
        
        # 限制历史数据大小
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    async def _check_performance_issues(self, metrics: CPUPerformanceMetrics):
        """检查性能问题"""
        issues = []
        
        # 检查总体CPU使用率
        if metrics.total_usage > 95:
            issues.append(f"CPU总使用率过高: {metrics.total_usage:.1f}%")
        
        # 检查单核心使用率
        for i, usage in enumerate(metrics.per_core_usage):
            if usage > 98:
                issues.append(f"CPU核心 {i} 使用率过高: {usage:.1f}%")
        
        # 检查负载平均值
        if metrics.load_average[0] > self.cpu_count * 1.5:
            issues.append(f"系统负载过高: {metrics.load_average[0]:.2f}")
        
        # 检查温度
        if metrics.temperature > 80:
            issues.append(f"CPU温度过高: {metrics.temperature:.1f}°C")
        
        if issues:
            logger.warning(f"CPU性能问题: {'; '.join(issues)}")
            await self._handle_performance_issues(issues)
    
    async def _handle_performance_issues(self, issues: List[str]):
        """处理性能问题"""
        for issue in issues:
            if "使用率过高" in issue:
                await self._rebalance_cpu_load()
            elif "温度过高" in issue:
                await self._reduce_cpu_frequency()
            elif "负载过高" in issue:
                await self._optimize_process_scheduling()
    
    async def _rebalance_cpu_load(self):
        """重新平衡CPU负载"""
        try:
            logger.info("开始重新平衡CPU负载...")
            
            # 获取当前各核心使用率
            per_core_usage = psutil.cpu_percent(interval=1.0, percpu=True)
            
            # 找出负载最高和最低的核心
            high_load_cores = [i for i, usage in enumerate(per_core_usage) if usage > 90]
            low_load_cores = [i for i, usage in enumerate(per_core_usage) if usage < 50]
            
            if high_load_cores and low_load_cores:
                # 尝试迁移一些进程
                await self._migrate_processes(high_load_cores, low_load_cores)
            
        except Exception as e:
            logger.error(f"重新平衡CPU负载失败: {e}")
    
    async def _migrate_processes(self, high_load_cores: List[int], low_load_cores: List[int]):
        """迁移进程到负载较低的核心"""
        try:
            # 这里实现进程迁移逻辑
            # 由于复杂性，这里只是记录日志
            logger.info(f"建议将进程从高负载核心 {high_load_cores} 迁移到低负载核心 {low_load_cores}")
        except Exception as e:
            logger.error(f"进程迁移失败: {e}")
    
    async def _reduce_cpu_frequency(self):
        """降低CPU频率"""
        try:
            if os.name == 'posix':  # Linux
                # 使用cpufreq工具降低频率
                subprocess.run(['sudo', 'cpufreq-set', '-d', '1000000'], 
                             capture_output=True, check=False)
                logger.info("已降低CPU频率以控制温度")
        except Exception as e:
            logger.debug(f"降低CPU频率失败: {e}")
    
    async def _optimize_process_scheduling(self):
        """优化进程调度"""
        try:
            # 调整进程优先级
            with self.process_lock:
                for pid, task_type in self.managed_processes.items():
                    try:
                        allocation = self.core_allocations[task_type]
                        process = psutil.Process(pid)
                        
                        # 根据当前负载调整优先级
                        if task_type in [CPUTaskType.SYSTEM_MONITORING, CPUTaskType.TRADING_EXECUTION]:
                            # 关键任务保持高优先级
                            continue
                        else:
                            # 其他任务降低优先级
                            if os.name == 'posix':
                                os.setpriority(os.PRIO_PROCESS, pid, allocation.priority + 2)
                    except:
                        continue
            
            logger.info("已优化进程调度")
            
        except Exception as e:
            logger.error(f"优化进程调度失败: {e}")
    
    def get_core_allocation_status(self) -> Dict[str, Any]:
        """获取核心分配状态"""
        status = {}
        
        for task_type, allocation in self.core_allocations.items():
            status[task_type.value] = {
                "core_ids": allocation.core_ids,
                "active_processes": len(allocation.active_processes),
                "max_threads": allocation.max_threads,
                "utilization": len(allocation.active_processes) / allocation.max_threads * 100,
                "description": allocation.description
            }
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-60:]  # 最近1分钟
        
        avg_usage = sum(m.total_usage for m in recent_metrics) / len(recent_metrics)
        max_usage = max(m.total_usage for m in recent_metrics)
        avg_temp = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
        max_temp = max(m.temperature for m in recent_metrics)
        
        return {
            "cpu_count": self.cpu_count,
            "physical_cores": self.physical_cores,
            "average_usage_1min": avg_usage,
            "max_usage_1min": max_usage,
            "average_temperature": avg_temp,
            "max_temperature": max_temp,
            "managed_processes": len(self.managed_processes),
            "core_allocations": self.get_core_allocation_status()
        }
    
    def optimize_for_trading(self):
        """为交易优化CPU设置"""
        try:
            logger.info("开始为交易优化CPU设置...")
            
            # 设置CPU调度策略为实时调度
            if os.name == 'posix':
                # 设置实时调度策略
                subprocess.run(['sudo', 'sysctl', '-w', 'kernel.sched_rt_runtime_us=950000'], 
                             capture_output=True, check=False)
                
                # 禁用CPU节能模式
                subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'], 
                             capture_output=True, check=False)
                
                # 设置CPU亲和性隔离
                subprocess.run(['sudo', 'tuna', '-c', '18,19', '--isolate'], 
                             capture_output=True, check=False)
            
            logger.info("CPU交易优化完成")
            
        except Exception as e:
            logger.error(f"CPU交易优化失败: {e}")
    
    def cleanup_dead_processes(self):
        """清理已死亡的进程"""
        dead_pids = []
        
        with self.process_lock:
            for pid in list(self.managed_processes.keys()):
                try:
                    if not psutil.pid_exists(pid):
                        dead_pids.append(pid)
                except:
                    dead_pids.append(pid)
        
        for pid in dead_pids:
            self.deallocate_process(pid)
        
        if dead_pids:
            logger.info(f"清理了 {len(dead_pids)} 个已死亡的进程")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        logger.info("CPU性能监控已停止")


# 全局CPU管理器实例
cpu_manager = CPUCoreManager()


def allocate_current_process(task_type: CPUTaskType) -> bool:
    """为当前进程分配CPU核心"""
    return cpu_manager.allocate_process_to_cores(os.getpid(), task_type)


def create_data_collection_process(target_function, *args, **kwargs) -> Optional[multiprocessing.Process]:
    """创建数据采集进程"""
    return cpu_manager.create_optimized_process(target_function, CPUTaskType.DATA_COLLECTION, args, kwargs)


def create_ai_training_process(target_function, light_weight: bool = True, *args, **kwargs) -> Optional[multiprocessing.Process]:
    """创建AI训练进程"""
    task_type = CPUTaskType.AI_TRAINING_LIGHT if light_weight else CPUTaskType.AI_TRAINING_MEDIUM
    return cpu_manager.create_optimized_process(target_function, task_type, args, kwargs)


def create_trading_process(target_function, *args, **kwargs) -> Optional[multiprocessing.Process]:
    """创建交易执行进程"""
    return cpu_manager.create_optimized_process(target_function, CPUTaskType.TRADING_EXECUTION, args, kwargs)


async def main():
    """测试主函数"""
    logger.info("启动CPU核心管理器测试...")
    
    # 为当前进程分配核心
    allocate_current_process(CPUTaskType.SYSTEM_MONITORING)
    
    # 启动性能监控
    monitor_task = asyncio.create_task(cpu_manager.start_performance_monitoring())
    
    try:
        # 运行30秒测试
        await asyncio.sleep(30)
        
        # 获取性能摘要
        summary = cpu_manager.get_performance_summary()
        logger.info(f"CPU性能摘要: {summary}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        cpu_manager.stop_monitoring()
        monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
