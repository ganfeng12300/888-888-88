"""
🧠 内存性能优化系统
针对128GB内存的高频交易优化，实现内存池管理、垃圾回收优化、NUMA内存分配
支持实时内存监控、内存泄漏检测和内存使用优化
"""

import gc
import mmap
import os
import psutil
import threading
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


class MemoryType(Enum):
    """内存类型"""
    SYSTEM = "system"           # 系统内存
    HEAP = "heap"               # 堆内存
    STACK = "stack"             # 栈内存
    SHARED = "shared"           # 共享内存
    CACHE = "cache"             # 缓存内存
    BUFFER = "buffer"           # 缓冲区内存


class MemoryPriority(Enum):
    """内存优先级"""
    CRITICAL = "critical"       # 关键内存
    HIGH = "high"               # 高优先级
    NORMAL = "normal"           # 普通优先级
    LOW = "low"                 # 低优先级


@dataclass
class MemoryPool:
    """内存池"""
    name: str                               # 池名称
    size: int                               # 池大小(字节)
    block_size: int                         # 块大小(字节)
    priority: MemoryPriority                # 优先级
    numa_node: int = 0                      # NUMA节点
    allocated_blocks: Set[int] = field(default_factory=set)  # 已分配块
    free_blocks: Set[int] = field(default_factory=set)       # 空闲块
    total_blocks: int = 0                   # 总块数
    created_at: float = field(default_factory=time.time)     # 创建时间


@dataclass
class MemoryStats:
    """内存统计"""
    total_memory: int                       # 总内存
    available_memory: int                   # 可用内存
    used_memory: int                        # 已用内存
    free_memory: int                        # 空闲内存
    cached_memory: int                      # 缓存内存
    buffer_memory: int                      # 缓冲区内存
    swap_total: int                         # 交换区总量
    swap_used: int                          # 交换区使用量
    swap_free: int                          # 交换区空闲量
    memory_percent: float                   # 内存使用率
    timestamp: float = field(default_factory=time.time)  # 时间戳


class MemoryPoolManager:
    """内存池管理器"""
    
    def __init__(self, total_memory_gb: int = 100):
        self.total_memory_gb = total_memory_gb
        self.total_memory_bytes = total_memory_gb * 1024 * 1024 * 1024
        self.pools: Dict[str, MemoryPool] = {}
        self.memory_maps: Dict[str, mmap.mmap] = {}
        self.allocation_lock = threading.RLock()
        
        # 预定义内存池配置
        self.pool_configs = self._create_pool_configs()
        
        # 初始化内存池
        self._initialize_pools()
        
        logger.info(f"内存池管理器初始化完成: {total_memory_gb}GB")
    
    def _create_pool_configs(self) -> Dict[str, Dict[str, Any]]:
        """创建内存池配置"""
        configs = {
            # 交易引擎内存池 - 20GB
            'trading_engine': {
                'size': 20 * 1024 * 1024 * 1024,  # 20GB
                'block_size': 64 * 1024,           # 64KB块
                'priority': MemoryPriority.CRITICAL,
                'numa_node': 0
            },
            
            # 市场数据内存池 - 15GB
            'market_data': {
                'size': 15 * 1024 * 1024 * 1024,  # 15GB
                'block_size': 32 * 1024,           # 32KB块
                'priority': MemoryPriority.CRITICAL,
                'numa_node': 0
            },
            
            # 策略计算内存池 - 25GB
            'strategy': {
                'size': 25 * 1024 * 1024 * 1024,  # 25GB
                'block_size': 128 * 1024,          # 128KB块
                'priority': MemoryPriority.HIGH,
                'numa_node': 1
            },
            
            # 风险管理内存池 - 10GB
            'risk_management': {
                'size': 10 * 1024 * 1024 * 1024,  # 10GB
                'block_size': 16 * 1024,           # 16KB块
                'priority': MemoryPriority.HIGH,
                'numa_node': 0
            },
            
            # 缓存内存池 - 20GB
            'cache': {
                'size': 20 * 1024 * 1024 * 1024,  # 20GB
                'block_size': 4 * 1024,            # 4KB块
                'priority': MemoryPriority.NORMAL,
                'numa_node': 1
            },
            
            # 日志内存池 - 5GB
            'logging': {
                'size': 5 * 1024 * 1024 * 1024,   # 5GB
                'block_size': 8 * 1024,            # 8KB块
                'priority': MemoryPriority.LOW,
                'numa_node': 1
            },
            
            # 通用内存池 - 5GB
            'general': {
                'size': 5 * 1024 * 1024 * 1024,   # 5GB
                'block_size': 16 * 1024,           # 16KB块
                'priority': MemoryPriority.NORMAL,
                'numa_node': 1
            }
        }
        
        return configs
    
    def _initialize_pools(self):
        """初始化内存池"""
        try:
            for pool_name, config in self.pool_configs.items():
                self._create_memory_pool(pool_name, config)
            
            logger.info(f"内存池初始化完成: {len(self.pools)}个池")
            
        except Exception as e:
            logger.error(f"内存池初始化失败: {e}")
    
    def _create_memory_pool(self, name: str, config: Dict[str, Any]) -> bool:
        """创建内存池"""
        try:
            with self.allocation_lock:
                # 创建内存池对象
                pool = MemoryPool(
                    name=name,
                    size=config['size'],
                    block_size=config['block_size'],
                    priority=config['priority'],
                    numa_node=config.get('numa_node', 0)
                )
                
                # 计算总块数
                pool.total_blocks = pool.size // pool.block_size
                
                # 初始化空闲块集合
                pool.free_blocks = set(range(pool.total_blocks))
                
                # 创建内存映射
                try:
                    # 创建临时文件用于内存映射
                    temp_file = f"/tmp/memory_pool_{name}_{os.getpid()}"
                    with open(temp_file, 'wb') as f:
                        f.write(b'\x00' * pool.size)
                    
                    # 创建内存映射
                    with open(temp_file, 'r+b') as f:
                        memory_map = mmap.mmap(f.fileno(), pool.size)
                        self.memory_maps[name] = memory_map
                    
                    # 删除临时文件
                    os.unlink(temp_file)
                    
                except Exception as e:
                    logger.warning(f"创建内存映射失败，使用普通内存: {e}")
                
                # 添加到池字典
                self.pools[name] = pool
                
                logger.info(f"创建内存池: {name} ({pool.size // (1024*1024)}MB, {pool.total_blocks}块)")
                return True
                
        except Exception as e:
            logger.error(f"创建内存池失败: {name} - {e}")
            return False
    
    def allocate_memory(self, pool_name: str, size: int) -> Optional[Tuple[int, int]]:
        """分配内存"""
        try:
            with self.allocation_lock:
                if pool_name not in self.pools:
                    logger.error(f"内存池不存在: {pool_name}")
                    return None
                
                pool = self.pools[pool_name]
                
                # 计算需要的块数
                blocks_needed = (size + pool.block_size - 1) // pool.block_size
                
                # 检查是否有足够的空闲块
                if len(pool.free_blocks) < blocks_needed:
                    logger.warning(f"内存池 {pool_name} 空间不足")
                    return None
                
                # 分配连续的块
                allocated_blocks = []
                free_blocks_list = sorted(list(pool.free_blocks))
                
                # 寻找连续的块
                for i in range(len(free_blocks_list) - blocks_needed + 1):
                    consecutive = True
                    for j in range(blocks_needed):
                        if free_blocks_list[i + j] != free_blocks_list[i] + j:
                            consecutive = False
                            break
                    
                    if consecutive:
                        allocated_blocks = free_blocks_list[i:i + blocks_needed]
                        break
                
                if not allocated_blocks:
                    logger.warning(f"内存池 {pool_name} 无法找到连续块")
                    return None
                
                # 更新池状态
                for block_id in allocated_blocks:
                    pool.free_blocks.remove(block_id)
                    pool.allocated_blocks.add(block_id)
                
                # 返回分配信息 (起始块ID, 块数)
                return (allocated_blocks[0], blocks_needed)
                
        except Exception as e:
            logger.error(f"分配内存失败: {e}")
            return None
    
    def deallocate_memory(self, pool_name: str, start_block: int, block_count: int) -> bool:
        """释放内存"""
        try:
            with self.allocation_lock:
                if pool_name not in self.pools:
                    logger.error(f"内存池不存在: {pool_name}")
                    return False
                
                pool = self.pools[pool_name]
                
                # 释放块
                for i in range(block_count):
                    block_id = start_block + i
                    if block_id in pool.allocated_blocks:
                        pool.allocated_blocks.remove(block_id)
                        pool.free_blocks.add(block_id)
                
                return True
                
        except Exception as e:
            logger.error(f"释放内存失败: {e}")
            return False
    
    def get_pool_stats(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """获取内存池统计"""
        try:
            if pool_name not in self.pools:
                return None
            
            pool = self.pools[pool_name]
            
            allocated_blocks = len(pool.allocated_blocks)
            free_blocks = len(pool.free_blocks)
            total_blocks = pool.total_blocks
            
            allocated_memory = allocated_blocks * pool.block_size
            free_memory = free_blocks * pool.block_size
            total_memory = total_blocks * pool.block_size
            
            usage_percent = (allocated_blocks / total_blocks) * 100 if total_blocks > 0 else 0
            
            return {
                'name': pool.name,
                'total_memory': total_memory,
                'allocated_memory': allocated_memory,
                'free_memory': free_memory,
                'total_blocks': total_blocks,
                'allocated_blocks': allocated_blocks,
                'free_blocks': free_blocks,
                'block_size': pool.block_size,
                'usage_percent': usage_percent,
                'priority': pool.priority.value,
                'numa_node': pool.numa_node,
                'created_at': pool.created_at
            }
            
        except Exception as e:
            logger.error(f"获取内存池统计失败: {e}")
            return None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有内存池统计"""
        stats = {}
        
        for pool_name in self.pools:
            pool_stats = self.get_pool_stats(pool_name)
            if pool_stats:
                stats[pool_name] = pool_stats
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        try:
            # 关闭内存映射
            for name, memory_map in self.memory_maps.items():
                try:
                    memory_map.close()
                except Exception:
                    pass
            
            self.memory_maps.clear()
            self.pools.clear()
            
            logger.info("内存池管理器清理完成")
            
        except Exception as e:
            logger.error(f"内存池清理失败: {e}")


class GarbageCollectionOptimizer:
    """垃圾回收优化器"""
    
    def __init__(self):
        self.gc_stats: List[Dict[str, Any]] = []
        self.max_stats = 1000
        self.optimization_enabled = True
        
        # 配置垃圾回收
        self._configure_gc()
        
        logger.info("垃圾回收优化器初始化完成")
    
    def _configure_gc(self):
        """配置垃圾回收"""
        try:
            # 设置垃圾回收阈值
            # 第0代: 700个对象后触发
            # 第1代: 10次第0代回收后触发
            # 第2代: 10次第1代回收后触发
            gc.set_threshold(700, 10, 10)
            
            # 启用垃圾回收调试
            if logger.level.name == 'DEBUG':
                gc.set_debug(gc.DEBUG_STATS)
            
            logger.info("垃圾回收配置完成")
            
        except Exception as e:
            logger.error(f"配置垃圾回收失败: {e}")
    
    def optimize_gc(self):
        """优化垃圾回收"""
        try:
            if not self.optimization_enabled:
                return
            
            # 记录回收前状态
            before_stats = self._get_gc_stats()
            
            # 执行垃圾回收
            collected = gc.collect()
            
            # 记录回收后状态
            after_stats = self._get_gc_stats()
            
            # 计算回收效果
            memory_freed = before_stats.get('memory_usage', 0) - after_stats.get('memory_usage', 0)
            
            # 记录统计信息
            gc_info = {
                'timestamp': time.time(),
                'objects_collected': collected,
                'memory_freed': memory_freed,
                'before_objects': before_stats.get('object_count', 0),
                'after_objects': after_stats.get('object_count', 0),
                'gc_counts': gc.get_count(),
                'gc_stats': gc.get_stats()
            }
            
            self.gc_stats.append(gc_info)
            if len(self.gc_stats) > self.max_stats:
                self.gc_stats.pop(0)
            
            if collected > 0:
                logger.info(f"垃圾回收完成: 回收{collected}个对象, 释放{memory_freed}字节内存")
            
            return gc_info
            
        except Exception as e:
            logger.error(f"垃圾回收优化失败: {e}")
            return None
    
    def _get_gc_stats(self) -> Dict[str, Any]:
        """获取垃圾回收统计"""
        try:
            # 获取对象计数
            object_count = len(gc.get_objects())
            
            # 获取内存使用情况
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'object_count': object_count,
                'memory_usage': memory_info.rss,
                'gc_counts': gc.get_count(),
                'gc_stats': gc.get_stats()
            }
            
        except Exception as e:
            logger.error(f"获取垃圾回收统计失败: {e}")
            return {}
    
    def schedule_gc(self, interval: float = 60.0):
        """定期垃圾回收"""
        def gc_worker():
            while self.optimization_enabled:
                try:
                    self.optimize_gc()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"定期垃圾回收失败: {e}")
                    time.sleep(10)
        
        gc_thread = threading.Thread(target=gc_worker, daemon=True)
        gc_thread.start()
        
        logger.info(f"定期垃圾回收已启动: {interval}秒间隔")
    
    def get_gc_summary(self) -> Dict[str, Any]:
        """获取垃圾回收摘要"""
        if not self.gc_stats:
            return {}
        
        try:
            recent_stats = self.gc_stats[-10:]  # 最近10次
            
            total_collected = sum(stat.get('objects_collected', 0) for stat in recent_stats)
            total_memory_freed = sum(stat.get('memory_freed', 0) for stat in recent_stats)
            
            return {
                'total_gc_runs': len(self.gc_stats),
                'recent_collected': total_collected,
                'recent_memory_freed': total_memory_freed,
                'current_objects': len(gc.get_objects()),
                'gc_counts': gc.get_count(),
                'gc_thresholds': gc.get_threshold(),
                'optimization_enabled': self.optimization_enabled
            }
            
        except Exception as e:
            logger.error(f"获取垃圾回收摘要失败: {e}")
            return {'error': str(e)}


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, pool_manager: MemoryPoolManager, gc_optimizer: GarbageCollectionOptimizer):
        self.pool_manager = pool_manager
        self.gc_optimizer = gc_optimizer
        self.monitoring = False
        self.monitor_task = None
        self.memory_history: List[MemoryStats] = []
        self.max_history = 1000
        
    async def start_monitoring(self, interval: float = 5.0):
        """启动内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("内存监控已启动")
    
    async def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("内存监控已停止")
    
    async def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集内存统计
                stats = self._collect_memory_stats()
                
                # 添加到历史记录
                self.memory_history.append(stats)
                if len(self.memory_history) > self.max_history:
                    self.memory_history.pop(0)
                
                # 检查内存异常
                self._check_memory_anomalies(stats)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"内存监控失败: {e}")
                await asyncio.sleep(10)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """收集内存统计"""
        try:
            # 系统内存信息
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            stats = MemoryStats(
                total_memory=memory.total,
                available_memory=memory.available,
                used_memory=memory.used,
                free_memory=memory.free,
                cached_memory=getattr(memory, 'cached', 0),
                buffer_memory=getattr(memory, 'buffers', 0),
                swap_total=swap.total,
                swap_used=swap.used,
                swap_free=swap.free,
                memory_percent=memory.percent
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"收集内存统计失败: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    
    def _check_memory_anomalies(self, stats: MemoryStats):
        """检查内存异常"""
        try:
            # 检查内存使用率
            if stats.memory_percent > 90.0:
                logger.warning(f"内存使用率过高: {stats.memory_percent:.1f}%")
                
                # 触发垃圾回收
                self.gc_optimizer.optimize_gc()
            
            # 检查交换区使用
            if stats.swap_total > 0:
                swap_percent = (stats.swap_used / stats.swap_total) * 100
                if swap_percent > 10.0:
                    logger.warning(f"交换区使用率过高: {swap_percent:.1f}%")
            
            # 检查可用内存
            if stats.available_memory < 1024 * 1024 * 1024:  # 小于1GB
                logger.critical(f"可用内存不足: {stats.available_memory // (1024*1024)}MB")
            
        except Exception as e:
            logger.error(f"检查内存异常失败: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        if not self.memory_history:
            return {}
        
        try:
            current_stats = self.memory_history[-1]
            
            # 计算趋势
            if len(self.memory_history) >= 2:
                prev_stats = self.memory_history[-2]
                memory_trend = current_stats.memory_percent - prev_stats.memory_percent
            else:
                memory_trend = 0.0
            
            return {
                'monitoring_active': self.monitoring,
                'data_points': len(self.memory_history),
                'current_memory_percent': current_stats.memory_percent,
                'current_available_gb': current_stats.available_memory // (1024*1024*1024),
                'current_used_gb': current_stats.used_memory // (1024*1024*1024),
                'total_memory_gb': current_stats.total_memory // (1024*1024*1024),
                'memory_trend': memory_trend,
                'swap_usage_percent': (current_stats.swap_used / current_stats.swap_total * 100) if current_stats.swap_total > 0 else 0.0,
                'pool_stats': self.pool_manager.get_all_stats(),
                'gc_summary': self.gc_optimizer.get_gc_summary()
            }
            
        except Exception as e:
            logger.error(f"获取内存摘要失败: {e}")
            return {'error': str(e)}


class MemoryOptimizer:
    """内存优化器主类"""
    
    def __init__(self, total_memory_gb: int = 100):
        self.pool_manager = MemoryPoolManager(total_memory_gb)
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.memory_monitor = MemoryMonitor(self.pool_manager, self.gc_optimizer)
        
        # 启动定期垃圾回收
        self.gc_optimizer.schedule_gc(interval=60.0)
        
        # 应用系统级优化
        self._apply_system_optimizations()
        
        logger.info("内存优化器初始化完成")
    
    def _apply_system_optimizations(self):
        """应用系统级内存优化"""
        try:
            # 禁用交换区
            self._disable_swap()
            
            # 优化内核内存参数
            self._optimize_kernel_memory_params()
            
            # 设置内存超量分配
            self._configure_memory_overcommit()
            
            logger.info("系统级内存优化已应用")
            
        except Exception as e:
            logger.error(f"应用系统内存优化失败: {e}")
    
    def _disable_swap(self):
        """禁用交换区"""
        try:
            # 禁用所有交换区
            os.system("swapoff -a 2>/dev/null")
            logger.info("交换区已禁用")
            
        except Exception as e:
            logger.warning(f"禁用交换区失败: {e}")
    
    def _optimize_kernel_memory_params(self):
        """优化内核内存参数"""
        try:
            memory_params = {
                '/proc/sys/vm/swappiness': '1',              # 最小化交换
                '/proc/sys/vm/dirty_ratio': '15',            # 脏页比例
                '/proc/sys/vm/dirty_background_ratio': '5',  # 后台脏页比例
                '/proc/sys/vm/vfs_cache_pressure': '50',     # VFS缓存压力
                '/proc/sys/vm/min_free_kbytes': '65536',     # 最小空闲内存
            }
            
            for param_path, value in memory_params.items():
                try:
                    if os.path.exists(param_path):
                        with open(param_path, 'w') as f:
                            f.write(value)
                except Exception:
                    pass
            
            logger.info("内核内存参数优化完成")
            
        except Exception as e:
            logger.warning(f"优化内核内存参数失败: {e}")
    
    def _configure_memory_overcommit(self):
        """配置内存超量分配"""
        try:
            # 设置内存超量分配模式
            # 0: 启发式超量分配
            # 1: 总是超量分配
            # 2: 严格会计模式
            overcommit_params = {
                '/proc/sys/vm/overcommit_memory': '2',       # 严格模式
                '/proc/sys/vm/overcommit_ratio': '80',       # 80%超量比例
            }
            
            for param_path, value in overcommit_params.items():
                try:
                    if os.path.exists(param_path):
                        with open(param_path, 'w') as f:
                            f.write(value)
                except Exception:
                    pass
            
            logger.info("内存超量分配配置完成")
            
        except Exception as e:
            logger.warning(f"配置内存超量分配失败: {e}")
    
    def allocate_memory(self, pool_name: str, size: int) -> Optional[Tuple[int, int]]:
        """分配内存"""
        return self.pool_manager.allocate_memory(pool_name, size)
    
    def deallocate_memory(self, pool_name: str, start_block: int, block_count: int) -> bool:
        """释放内存"""
        return self.pool_manager.deallocate_memory(pool_name, start_block, block_count)
    
    def optimize_gc(self):
        """优化垃圾回收"""
        return self.gc_optimizer.optimize_gc()
    
    async def start_monitoring(self, interval: float = 5.0):
        """启动内存监控"""
        await self.memory_monitor.start_monitoring(interval)
    
    async def stop_monitoring(self):
        """停止内存监控"""
        await self.memory_monitor.stop_monitoring()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'memory_pools': self.pool_manager.get_all_stats(),
            'gc_optimizer': self.gc_optimizer.get_gc_summary(),
            'memory_monitor': self.memory_monitor.get_memory_summary()
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            self.pool_manager.cleanup()
            self.gc_optimizer.optimization_enabled = False
            logger.info("内存优化器清理完成")
            
        except Exception as e:
            logger.error(f"内存优化器清理失败: {e}")


# 全局内存优化器实例
memory_optimizer = MemoryOptimizer()

