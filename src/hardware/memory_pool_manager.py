"""
⚡ 内存池管理系统
生产级内存管理系统，支持智能内存池分配、多级缓存优化、内存泄漏检测
实现完整的内存生命周期管理、预取机制、自动回收等功能
专为高频量化交易场景优化，确保内存使用效率和系统稳定性
"""
import asyncio
import threading
import time
import mmap
import os
import gc
import weakref
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import psutil
import numpy as np

@dataclass
class MemoryBlock:
    """内存块"""
    block_id: str
    size: int
    address: int
    is_free: bool = True
    allocated_at: Optional[datetime] = None
    freed_at: Optional[datetime] = None
    access_count: int = 0
    last_access: Optional[datetime] = None
    pool_id: str = ""
    data_type: str = "generic"

@dataclass
class MemoryPool:
    """内存池"""
    pool_id: str
    block_size: int
    max_blocks: int
    allocated_blocks: int = 0
    free_blocks: List[MemoryBlock] = field(default_factory=list)
    used_blocks: Dict[str, MemoryBlock] = field(default_factory=dict)
    total_allocations: int = 0
    total_deallocations: int = 0
    peak_usage: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    size: int
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    ttl: Optional[float] = None
    priority: int = 1

class MemoryPoolManager:
    """内存池管理器"""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.logger = logging.getLogger(__name__)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.is_running = False
        
        # 内存池管理
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.pool_locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.RLock()
        
        # 缓存系统
        self.l1_cache: Dict[str, CacheEntry] = {}  # 最热数据
        self.l2_cache: Dict[str, CacheEntry] = {}  # 热数据
        self.l3_cache: Dict[str, CacheEntry] = {}  # 温数据
        self.cache_locks = {
            'l1': threading.Lock(),
            'l2': threading.Lock(),
            'l3': threading.Lock()
        }
        
        # 内存映射文件
        self.mmap_files: Dict[str, mmap.mmap] = {}
        self.mmap_locks: Dict[str, threading.Lock] = {}
        
        # 性能监控
        self.allocation_stats = deque(maxlen=10000)
        self.memory_usage_history = deque(maxlen=1000)
        self.gc_stats = deque(maxlen=1000)
        
        # 预取系统
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_cache: Dict[str, Any] = {}
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # 内存泄漏检测
        self.tracked_objects = weakref.WeakSet()
        self.allocation_traces: Dict[int, Dict] = {}
        
        # 配置参数
        self.cache_sizes = {
            'l1': 64 * 1024 * 1024,   # 64MB L1缓存
            'l2': 256 * 1024 * 1024,  # 256MB L2缓存
            'l3': 1024 * 1024 * 1024  # 1GB L3缓存
        }
        
        self._initialize_pools()
        
    def _initialize_pools(self):
        """初始化内存池"""
        # 小对象池 (1KB - 64KB)
        for size_kb in [1, 2, 4, 8, 16, 32, 64]:
            pool_id = f"small_{size_kb}kb"
            self._create_pool(pool_id, size_kb * 1024, 1000)
            
        # 中等对象池 (128KB - 8MB)
        for size_kb in [128, 256, 512, 1024, 2048, 4096, 8192]:
            pool_id = f"medium_{size_kb}kb"
            self._create_pool(pool_id, size_kb * 1024, 100)
            
        # 大对象池 (16MB - 256MB)
        for size_mb in [16, 32, 64, 128, 256]:
            pool_id = f"large_{size_mb}mb"
            self._create_pool(pool_id, size_mb * 1024 * 1024, 10)
            
        self.logger.info(f"初始化了 {len(self.memory_pools)} 个内存池")
        
    def _create_pool(self, pool_id: str, block_size: int, max_blocks: int):
        """创建内存池"""
        pool = MemoryPool(
            pool_id=pool_id,
            block_size=block_size,
            max_blocks=max_blocks
        )
        
        self.memory_pools[pool_id] = pool
        self.pool_locks[pool_id] = threading.Lock()
        
        # 预分配一些内存块
        self._preallocate_blocks(pool_id, min(max_blocks // 4, 10))
        
    def _preallocate_blocks(self, pool_id: str, count: int):
        """预分配内存块"""
        pool = self.memory_pools[pool_id]
        
        for i in range(count):
            try:
                # 分配内存
                data = bytearray(pool.block_size)
                address = id(data)
                
                block = MemoryBlock(
                    block_id=f"{pool_id}_{i}",
                    size=pool.block_size,
                    address=address,
                    pool_id=pool_id
                )
                
                pool.free_blocks.append(block)
                
            except MemoryError:
                self.logger.warning(f"预分配内存块失败: {pool_id}")
                break
                
    async def start(self):
        """启动内存管理器"""
        self.is_running = True
        self.logger.info("⚡ 内存池管理器启动")
        
        # 启动管理循环
        tasks = [
            asyncio.create_task(self._memory_monitoring_loop()),
            asyncio.create_task(self._cache_management_loop()),
            asyncio.create_task(self._prefetch_loop()),
            asyncio.create_task(self._garbage_collection_loop()),
            asyncio.create_task(self._leak_detection_loop())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop(self):
        """停止内存管理器"""
        self.is_running = False
        
        # 清理所有缓存
        await self._cleanup_all_caches()
        
        # 释放所有内存池
        self._cleanup_all_pools()
        
        self.logger.info("⚡ 内存池管理器停止")
        
    def allocate(self, size: int, data_type: str = "generic") -> Optional[MemoryBlock]:
        """分配内存"""
        # 选择合适的内存池
        pool_id = self._select_pool(size)
        
        if pool_id is None:
            # 没有合适的池，直接分配
            return self._direct_allocate(size, data_type)
            
        with self.pool_locks[pool_id]:
            pool = self.memory_pools[pool_id]
            
            # 尝试从空闲块中获取
            if pool.free_blocks:
                block = pool.free_blocks.pop()
                block.is_free = False
                block.allocated_at = datetime.now()
                block.data_type = data_type
                
                pool.used_blocks[block.block_id] = block
                pool.allocated_blocks += 1
                pool.total_allocations += 1
                
                # 更新峰值使用量
                current_usage = pool.allocated_blocks * pool.block_size
                pool.peak_usage = max(pool.peak_usage, current_usage)
                
                # 记录分配统计
                self._record_allocation_stats(block)
                
                return block
                
            # 没有空闲块，尝试创建新块
            if pool.allocated_blocks < pool.max_blocks:
                return self._create_new_block(pool_id, size, data_type)
                
        return None
        
    def _select_pool(self, size: int) -> Optional[str]:
        """选择合适的内存池"""
        best_pool = None
        best_waste = float('inf')
        
        for pool_id, pool in self.memory_pools.items():
            if pool.block_size >= size:
                waste = pool.block_size - size
                if waste < best_waste:
                    best_waste = waste
                    best_pool = pool_id
                    
        return best_pool
        
    def _direct_allocate(self, size: int, data_type: str) -> MemoryBlock:
        """直接分配内存"""
        try:
            data = bytearray(size)
            address = id(data)
            
            block = MemoryBlock(
                block_id=f"direct_{int(time.time() * 1000000)}",
                size=size,
                address=address,
                is_free=False,
                allocated_at=datetime.now(),
                data_type=data_type,
                pool_id="direct"
            )
            
            self._record_allocation_stats(block)
            return block
            
        except MemoryError:
            self.logger.error(f"直接内存分配失败: {size} bytes")
            return None
            
    def _create_new_block(self, pool_id: str, size: int, data_type: str) -> Optional[MemoryBlock]:
        """创建新的内存块"""
        pool = self.memory_pools[pool_id]
        
        try:
            data = bytearray(pool.block_size)
            address = id(data)
            
            block = MemoryBlock(
                block_id=f"{pool_id}_{pool.total_allocations}",
                size=pool.block_size,
                address=address,
                is_free=False,
                allocated_at=datetime.now(),
                data_type=data_type,
                pool_id=pool_id
            )
            
            pool.used_blocks[block.block_id] = block
            pool.allocated_blocks += 1
            pool.total_allocations += 1
            
            self._record_allocation_stats(block)
            return block
            
        except MemoryError:
            self.logger.error(f"创建内存块失败: {pool_id}")
            return None
            
    def deallocate(self, block: MemoryBlock):
        """释放内存"""
        if block.pool_id == "direct":
            # 直接分配的内存，标记为已释放
            block.is_free = True
            block.freed_at = datetime.now()
            return
            
        pool_id = block.pool_id
        if pool_id not in self.memory_pools:
            return
            
        with self.pool_locks[pool_id]:
            pool = self.memory_pools[pool_id]
            
            if block.block_id in pool.used_blocks:
                # 从使用中移除
                del pool.used_blocks[block.block_id]
                
                # 重置块状态
                block.is_free = True
                block.freed_at = datetime.now()
                block.access_count = 0
                block.last_access = None
                
                # 添加到空闲列表
                pool.free_blocks.append(block)
                pool.allocated_blocks -= 1
                pool.total_deallocations += 1
                
    def _record_allocation_stats(self, block: MemoryBlock):
        """记录分配统计"""
        stats = {
            'timestamp': datetime.now(),
            'block_id': block.block_id,
            'size': block.size,
            'pool_id': block.pool_id,
            'data_type': block.data_type,
            'action': 'allocate'
        }
        
        self.allocation_stats.append(stats)
        
    # 缓存管理方法
    async def cache_put(self, key: str, data: Any, level: str = "l2", 
                       ttl: Optional[float] = None, priority: int = 1):
        """放入缓存"""
        if level not in self.cache_locks:
            level = "l2"
            
        # 计算数据大小
        size = self._calculate_data_size(data)
        
        entry = CacheEntry(
            key=key,
            data=data,
            size=size,
            ttl=ttl,
            priority=priority
        )
        
        with self.cache_locks[level]:
            cache = getattr(self, f"{level}_cache")
            
            # 检查缓存容量
            await self._ensure_cache_capacity(level, size)
            
            # 添加到缓存
            cache[key] = entry
            
            # 记录访问模式
            self._record_access_pattern(key)
            
    async def cache_get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        # 按优先级检查各级缓存
        for level in ['l1', 'l2', 'l3']:
            with self.cache_locks[level]:
                cache = getattr(self, f"{level}_cache")
                
                if key in cache:
                    entry = cache[key]
                    
                    # 检查TTL
                    if entry.ttl is not None:
                        age = (datetime.now() - entry.created_at).total_seconds()
                        if age > entry.ttl:
                            del cache[key]
                            continue
                            
                    # 更新访问信息
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    
                    # 提升到更高级缓存
                    if level != 'l1' and entry.access_count > 5:
                        await self._promote_cache_entry(key, entry, level)
                        
                    return entry.data
                    
        return None
        
    async def _ensure_cache_capacity(self, level: str, required_size: int):
        """确保缓存容量"""
        cache = getattr(self, f"{level}_cache")
        max_size = self.cache_sizes[level]
        
        current_size = sum(entry.size for entry in cache.values())
        
        if current_size + required_size > max_size:
            # 需要清理缓存
            await self._evict_cache_entries(level, required_size)
            
    async def _evict_cache_entries(self, level: str, required_size: int):
        """驱逐缓存条目"""
        cache = getattr(self, f"{level}_cache")
        
        # 按LRU策略排序
        entries = list(cache.items())
        entries.sort(key=lambda x: (x[1].last_access, x[1].access_count))
        
        freed_size = 0
        for key, entry in entries:
            if freed_size >= required_size:
                break
                
            del cache[key]
            freed_size += entry.size
            
    async def _promote_cache_entry(self, key: str, entry: CacheEntry, current_level: str):
        """提升缓存条目到更高级"""
        if current_level == 'l3':
            target_level = 'l2'
        elif current_level == 'l2':
            target_level = 'l1'
        else:
            return
            
        # 从当前级别移除
        current_cache = getattr(self, f"{current_level}_cache")
        if key in current_cache:
            del current_cache[key]
            
        # 添加到目标级别
        await self.cache_put(key, entry.data, target_level, entry.ttl, entry.priority)
        
    def _calculate_data_size(self, data: Any) -> int:
        """计算数据大小"""
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, np.ndarray):
            return data.nbytes
        else:
            # 估算其他类型的大小
            import sys
            return sys.getsizeof(data)
            
    def _record_access_pattern(self, key: str):
        """记录访问模式"""
        # 简化的访问模式记录
        current_time = datetime.now()
        pattern_key = f"{current_time.hour}:{current_time.minute // 10}"
        
        self.access_patterns[pattern_key].append(key)
        
        # 保持模式历史在合理范围内
        if len(self.access_patterns[pattern_key]) > 100:
            self.access_patterns[pattern_key] = self.access_patterns[pattern_key][-50:]
            
    # 预取系统
    async def prefetch_data(self, key: str, data_loader):
        """预取数据"""
        await self.prefetch_queue.put((key, data_loader))
        
    async def _prefetch_loop(self):
        """预取循环"""
        while self.is_running:
            try:
                key, data_loader = await asyncio.wait_for(
                    self.prefetch_queue.get(), timeout=1.0
                )
                
                # 检查是否已经在缓存中
                if await self.cache_get(key) is not None:
                    continue
                    
                # 加载数据
                try:
                    data = await data_loader() if asyncio.iscoroutinefunction(data_loader) else data_loader()
                    
                    # 放入预取缓存
                    self.prefetch_cache[key] = data
                    
                    # 也放入L3缓存
                    await self.cache_put(key, data, "l3")
                    
                except Exception as e:
                    self.logger.error(f"预取数据失败 {key}: {e}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"预取循环错误: {e}")
                await asyncio.sleep(1)
                
    # 监控和管理循环
    async def _memory_monitoring_loop(self):
        """内存监控循环"""
        while self.is_running:
            try:
                # 收集内存使用统计
                memory_info = psutil.virtual_memory()
                
                stats = {
                    'timestamp': datetime.now(),
                    'system_memory': {
                        'total': memory_info.total,
                        'available': memory_info.available,
                        'percent': memory_info.percent,
                        'used': memory_info.used,
                        'free': memory_info.free
                    },
                    'pools': {},
                    'caches': {}
                }
                
                # 收集池统计
                for pool_id, pool in self.memory_pools.items():
                    stats['pools'][pool_id] = {
                        'allocated_blocks': pool.allocated_blocks,
                        'free_blocks': len(pool.free_blocks),
                        'total_allocations': pool.total_allocations,
                        'total_deallocations': pool.total_deallocations,
                        'peak_usage': pool.peak_usage
                    }
                    
                # 收集缓存统计
                for level in ['l1', 'l2', 'l3']:
                    cache = getattr(self, f"{level}_cache")
                    total_size = sum(entry.size for entry in cache.values())
                    
                    stats['caches'][level] = {
                        'entries': len(cache),
                        'total_size': total_size,
                        'max_size': self.cache_sizes[level],
                        'usage_percent': (total_size / self.cache_sizes[level]) * 100
                    }
                    
                self.memory_usage_history.append(stats)
                
                await asyncio.sleep(10)  # 10秒监控一次
                
            except Exception as e:
                self.logger.error(f"内存监控错误: {e}")
                await asyncio.sleep(30)
                
    async def _cache_management_loop(self):
        """缓存管理循环"""
        while self.is_running:
            try:
                # 清理过期缓存
                await self._cleanup_expired_cache()
                
                # 优化缓存分布
                await self._optimize_cache_distribution()
                
                await asyncio.sleep(60)  # 1分钟管理一次
                
            except Exception as e:
                self.logger.error(f"缓存管理错误: {e}")
                await asyncio.sleep(120)
                
    async def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        
        for level in ['l1', 'l2', 'l3']:
            with self.cache_locks[level]:
                cache = getattr(self, f"{level}_cache")
                expired_keys = []
                
                for key, entry in cache.items():
                    if entry.ttl is not None:
                        age = (current_time - entry.created_at).total_seconds()
                        if age > entry.ttl:
                            expired_keys.append(key)
                            
                for key in expired_keys:
                    del cache[key]
                    
                if expired_keys:
                    self.logger.debug(f"清理了 {len(expired_keys)} 个过期的{level}缓存条目")
                    
    async def _optimize_cache_distribution(self):
        """优化缓存分布"""
        # 分析访问模式，调整缓存策略
        # 这里可以实现更复杂的缓存优化算法
        pass
        
    async def _garbage_collection_loop(self):
        """垃圾回收循环"""
        while self.is_running:
            try:
                # 执行垃圾回收
                before = psutil.virtual_memory().used
                
                collected = gc.collect()
                
                after = psutil.virtual_memory().used
                freed = before - after
                
                gc_stats = {
                    'timestamp': datetime.now(),
                    'collected_objects': collected,
                    'memory_freed': freed,
                    'generation_0': gc.get_count()[0],
                    'generation_1': gc.get_count()[1],
                    'generation_2': gc.get_count()[2]
                }
                
                self.gc_stats.append(gc_stats)
                
                if collected > 0:
                    self.logger.debug(f"垃圾回收: 回收了{collected}个对象，释放了{freed}字节内存")
                    
                await asyncio.sleep(300)  # 5分钟回收一次
                
            except Exception as e:
                self.logger.error(f"垃圾回收错误: {e}")
                await asyncio.sleep(600)
                
    async def _leak_detection_loop(self):
        """内存泄漏检测循环"""
        while self.is_running:
            try:
                # 检测长时间未释放的内存块
                current_time = datetime.now()
                leak_threshold = timedelta(hours=1)
                
                potential_leaks = []
                
                for pool_id, pool in self.memory_pools.items():
                    with self.pool_locks[pool_id]:
                        for block_id, block in pool.used_blocks.items():
                            if block.allocated_at and (current_time - block.allocated_at) > leak_threshold:
                                potential_leaks.append(block)
                                
                if potential_leaks:
                    self.logger.warning(f"检测到 {len(potential_leaks)} 个潜在内存泄漏")
                    
                await asyncio.sleep(1800)  # 30分钟检测一次
                
            except Exception as e:
                self.logger.error(f"内存泄漏检测错误: {e}")
                await asyncio.sleep(3600)
                
    async def _cleanup_all_caches(self):
        """清理所有缓存"""
        for level in ['l1', 'l2', 'l3']:
            with self.cache_locks[level]:
                cache = getattr(self, f"{level}_cache")
                cache.clear()
                
        self.prefetch_cache.clear()
        
    def _cleanup_all_pools(self):
        """清理所有内存池"""
        with self.global_lock:
            for pool_id, pool in self.memory_pools.items():
                with self.pool_locks[pool_id]:
                    pool.free_blocks.clear()
                    pool.used_blocks.clear()
                    
    # 公共接口方法
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        if not self.memory_usage_history:
            return {}
            
        latest_stats = self.memory_usage_history[-1]
        
        return {
            'system_memory': latest_stats['system_memory'],
            'pools': latest_stats['pools'],
            'caches': latest_stats['caches'],
            'total_pools': len(self.memory_pools),
            'allocation_stats_count': len(self.allocation_stats),
            'gc_stats_count': len(self.gc_stats)
        }
        
    def get_pool_info(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """获取内存池信息"""
        if pool_id not in self.memory_pools:
            return None
            
        pool = self.memory_pools[pool_id]
        
        return {
            'pool_id': pool.pool_id,
            'block_size': pool.block_size,
            'max_blocks': pool.max_blocks,
            'allocated_blocks': pool.allocated_blocks,
            'free_blocks': len(pool.free_blocks),
            'total_allocations': pool.total_allocations,
            'total_deallocations': pool.total_deallocations,
            'peak_usage': pool.peak_usage,
            'created_at': pool.created_at.isoformat()
        }
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        stats = {}
        
        for level in ['l1', 'l2', 'l3']:
            cache = getattr(self, f"{level}_cache")
            total_size = sum(entry.size for entry in cache.values())
            
            stats[level] = {
                'entries': len(cache),
                'total_size': total_size,
                'max_size': self.cache_sizes[level],
                'usage_percent': (total_size / self.cache_sizes[level]) * 100 if self.cache_sizes[level] > 0 else 0,
                'hit_ratio': 0  # 需要实现命中率统计
            }
            
        return stats

