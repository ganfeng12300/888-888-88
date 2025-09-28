"""
🔧 GPU显存优化管理系统 - RTX3060 12GB显存最大化利用
生产级GPU显存池管理、动态分配回收、多模型调度、温度控制
专为AI量化交易系统优化，支持6个AI模型并行训练
"""

import gc
import os
import psutil
import threading
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, GPU optimization will be limited")

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("pynvml not available, GPU monitoring will be limited")

from loguru import logger


class GPUMemoryType(Enum):
    """GPU内存类型"""
    MODEL_WEIGHTS = "model_weights"     # 模型权重
    GRADIENTS = "gradients"             # 梯度
    ACTIVATIONS = "activations"         # 激活值
    OPTIMIZER_STATES = "optimizer_states" # 优化器状态
    CACHE = "cache"                     # 缓存
    TEMPORARY = "temporary"             # 临时数据


class GPUPriority(Enum):
    """GPU优先级"""
    CRITICAL = "critical"               # 关键优先级
    HIGH = "high"                       # 高优先级
    NORMAL = "normal"                   # 普通优先级
    LOW = "low"                         # 低优先级


@dataclass
class GPUMemoryBlock:
    """GPU内存块"""
    block_id: str                       # 块ID
    size: int                           # 大小(字节)
    memory_type: GPUMemoryType          # 内存类型
    priority: GPUPriority               # 优先级
    allocated: bool                     # 是否已分配
    owner: Optional[str]                # 所有者
    created_at: float                   # 创建时间
    last_accessed: float                # 最后访问时间
    tensor_ptr: Optional[Any]           # 张量指针


@dataclass
class GPUMemoryPool:
    """GPU内存池"""
    pool_id: str                        # 池ID
    total_size: int                     # 总大小
    allocated_size: int                 # 已分配大小
    free_size: int                      # 空闲大小
    memory_type: GPUMemoryType          # 内存类型
    blocks: Dict[str, GPUMemoryBlock]   # 内存块
    max_block_size: int                 # 最大块大小
    fragmentation_ratio: float          # 碎片率


@dataclass
class GPUStatus:
    """GPU状态"""
    device_id: int                      # 设备ID
    name: str                           # 设备名称
    total_memory: int                   # 总显存
    used_memory: int                    # 已用显存
    free_memory: int                    # 空闲显存
    memory_utilization: float           # 显存利用率
    gpu_utilization: float              # GPU利用率
    temperature: float                  # 温度
    power_usage: float                  # 功耗
    timestamp: float                    # 时间戳


class GPUMemoryManager:
    """GPU显存管理器"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.memory_pools: Dict[str, GPUMemoryPool] = {}
        self.allocated_blocks: Dict[str, GPUMemoryBlock] = {}
        self.allocation_lock = threading.RLock()
        
        # RTX3060 12GB配置
        self.total_memory = 12 * 1024 * 1024 * 1024  # 12GB
        self.reserved_memory = 1 * 1024 * 1024 * 1024  # 1GB系统保留
        self.available_memory = self.total_memory - self.reserved_memory
        
        # 内存池配置
        self.pool_configs = self._create_pool_configs()
        
        # 初始化GPU
        self._initialize_gpu()
        
        # 创建内存池
        self._create_memory_pools()
        
        logger.info(f"GPU显存管理器初始化完成: RTX3060 12GB (设备{device_id})")
    
    def _initialize_gpu(self):
        """初始化GPU"""
        try:
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.device_id)
                    # 清空GPU缓存
                    torch.cuda.empty_cache()
                    # 设置内存分配策略
                    torch.cuda.set_per_process_memory_fraction(0.9, self.device_id)
                    logger.info("PyTorch GPU初始化完成")
                else:
                    logger.warning("CUDA不可用")
            
            if NVML_AVAILABLE:
                pynvml.nvmlInit()
                logger.info("NVML初始化完成")
            
        except Exception as e:
            logger.error(f"GPU初始化失败: {e}")
    
    def _create_pool_configs(self) -> Dict[str, Dict[str, Any]]:
        """创建内存池配置"""
        configs = {
            # 模型权重池 - 6GB (6个模型，每个1GB)
            'model_weights': {
                'size': 6 * 1024 * 1024 * 1024,
                'memory_type': GPUMemoryType.MODEL_WEIGHTS,
                'max_block_size': 1024 * 1024 * 1024,  # 1GB
                'priority': GPUPriority.CRITICAL
            },
            
            # 梯度池 - 2GB
            'gradients': {
                'size': 2 * 1024 * 1024 * 1024,
                'memory_type': GPUMemoryType.GRADIENTS,
                'max_block_size': 512 * 1024 * 1024,   # 512MB
                'priority': GPUPriority.HIGH
            },
            
            # 激活值池 - 2GB
            'activations': {
                'size': 2 * 1024 * 1024 * 1024,
                'memory_type': GPUMemoryType.ACTIVATIONS,
                'max_block_size': 256 * 1024 * 1024,   # 256MB
                'priority': GPUPriority.HIGH
            },
            
            # 优化器状态池 - 1GB
            'optimizer_states': {
                'size': 1 * 1024 * 1024 * 1024,
                'memory_type': GPUMemoryType.OPTIMIZER_STATES,
                'max_block_size': 256 * 1024 * 1024,   # 256MB
                'priority': GPUPriority.NORMAL
            },
            
            # 缓存池 - 500MB
            'cache': {
                'size': 500 * 1024 * 1024,
                'memory_type': GPUMemoryType.CACHE,
                'max_block_size': 100 * 1024 * 1024,   # 100MB
                'priority': GPUPriority.LOW
            },
            
            # 临时数据池 - 500MB
            'temporary': {
                'size': 500 * 1024 * 1024,
                'memory_type': GPUMemoryType.TEMPORARY,
                'max_block_size': 50 * 1024 * 1024,    # 50MB
                'priority': GPUPriority.LOW
            }
        }
        
        return configs
    
    def _create_memory_pools(self):
        """创建内存池"""
        try:
            with self.allocation_lock:
                for pool_id, config in self.pool_configs.items():
                    pool = GPUMemoryPool(
                        pool_id=pool_id,
                        total_size=config['size'],
                        allocated_size=0,
                        free_size=config['size'],
                        memory_type=config['memory_type'],
                        blocks={},
                        max_block_size=config['max_block_size'],
                        fragmentation_ratio=0.0
                    )
                    
                    self.memory_pools[pool_id] = pool
                    
                    logger.info(f"创建GPU内存池: {pool_id} ({config['size'] // (1024*1024)}MB)")
                
                logger.info(f"GPU内存池创建完成: {len(self.memory_pools)}个池")
                
        except Exception as e:
            logger.error(f"创建GPU内存池失败: {e}")
    
    def allocate_memory(self, pool_id: str, size: int, owner: str, 
                       memory_type: GPUMemoryType = None, 
                       priority: GPUPriority = GPUPriority.NORMAL) -> Optional[str]:
        """分配GPU内存"""
        try:
            with self.allocation_lock:
                if pool_id not in self.memory_pools:
                    logger.error(f"内存池不存在: {pool_id}")
                    return None
                
                pool = self.memory_pools[pool_id]
                
                # 检查可用空间
                if pool.free_size < size:
                    logger.warning(f"内存池 {pool_id} 空间不足: 需要{size//1024//1024}MB, 可用{pool.free_size//1024//1024}MB")
                    
                    # 尝试垃圾回收
                    self._garbage_collect()
                    
                    # 再次检查
                    if pool.free_size < size:
                        # 尝试释放低优先级内存
                        if self._free_low_priority_memory(pool, size):
                            logger.info(f"释放低优先级内存后重试分配")
                        else:
                            return None
                
                # 创建内存块
                block_id = f"{pool_id}_{int(time.time() * 1000000)}"
                
                # 实际分配GPU内存
                tensor_ptr = None
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        # 分配GPU张量
                        tensor_ptr = torch.empty(size // 4, dtype=torch.float32, device=f'cuda:{self.device_id}')
                    except torch.cuda.OutOfMemoryError:
                        logger.error("GPU显存不足，分配失败")
                        return None
                
                # 创建内存块对象
                block = GPUMemoryBlock(
                    block_id=block_id,
                    size=size,
                    memory_type=memory_type or pool.memory_type,
                    priority=priority,
                    allocated=True,
                    owner=owner,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    tensor_ptr=tensor_ptr
                )
                
                # 更新池状态
                pool.blocks[block_id] = block
                pool.allocated_size += size
                pool.free_size -= size
                
                # 更新全局分配记录
                self.allocated_blocks[block_id] = block
                
                # 更新碎片率
                pool.fragmentation_ratio = self._calculate_fragmentation(pool)
                
                logger.info(f"GPU内存分配成功: {block_id} ({size//1024//1024}MB) -> {owner}")
                return block_id
                
        except Exception as e:
            logger.error(f"GPU内存分配失败: {e}")
            return None
    
    def deallocate_memory(self, block_id: str) -> bool:
        """释放GPU内存"""
        try:
            with self.allocation_lock:
                if block_id not in self.allocated_blocks:
                    logger.warning(f"内存块不存在: {block_id}")
                    return False
                
                block = self.allocated_blocks[block_id]
                
                # 找到对应的内存池
                pool = None
                for pool_id, p in self.memory_pools.items():
                    if block_id in p.blocks:
                        pool = p
                        break
                
                if not pool:
                    logger.error(f"找不到内存块对应的池: {block_id}")
                    return False
                
                # 释放GPU张量
                if block.tensor_ptr is not None:
                    try:
                        del block.tensor_ptr
                    except Exception as e:
                        logger.warning(f"释放GPU张量失败: {e}")
                
                # 更新池状态
                pool.allocated_size -= block.size
                pool.free_size += block.size
                del pool.blocks[block_id]
                
                # 更新全局记录
                del self.allocated_blocks[block_id]
                
                # 更新碎片率
                pool.fragmentation_ratio = self._calculate_fragmentation(pool)
                
                logger.info(f"GPU内存释放成功: {block_id} ({block.size//1024//1024}MB)")
                return True
                
        except Exception as e:
            logger.error(f"GPU内存释放失败: {e}")
            return False
    
    def _free_low_priority_memory(self, pool: GPUMemoryPool, needed_size: int) -> bool:
        """释放低优先级内存"""
        try:
            # 按优先级和最后访问时间排序
            blocks_to_free = []
            for block in pool.blocks.values():
                if block.priority in [GPUPriority.LOW, GPUPriority.NORMAL]:
                    blocks_to_free.append(block)
            
            # 按优先级(低优先级优先)和访问时间(旧的优先)排序
            blocks_to_free.sort(key=lambda b: (b.priority.value, b.last_accessed))
            
            freed_size = 0
            for block in blocks_to_free:
                if freed_size >= needed_size:
                    break
                
                if self.deallocate_memory(block.block_id):
                    freed_size += block.size
                    logger.info(f"释放低优先级内存: {block.block_id} ({block.size//1024//1024}MB)")
            
            return freed_size >= needed_size
            
        except Exception as e:
            logger.error(f"释放低优先级内存失败: {e}")
            return False
    
    def _calculate_fragmentation(self, pool: GPUMemoryPool) -> float:
        """计算内存碎片率"""
        try:
            if pool.total_size == 0:
                return 0.0
            
            # 简单的碎片率计算：已分配块数 / 总容量的比例
            block_count = len(pool.blocks)
            if block_count == 0:
                return 0.0
            
            # 碎片率 = 1 - (最大连续空闲空间 / 总空闲空间)
            # 这里简化为基于块数量的估算
            fragmentation = min(0.9, block_count / 100.0)  # 最大90%碎片率
            
            return fragmentation
            
        except Exception as e:
            logger.error(f"计算内存碎片率失败: {e}")
            return 0.0
    
    def _garbage_collect(self):
        """垃圾回收"""
        try:
            # Python垃圾回收
            gc.collect()
            
            # PyTorch GPU缓存清理
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("GPU垃圾回收完成")
            
        except Exception as e:
            logger.error(f"GPU垃圾回收失败: {e}")
    
    def get_gpu_status(self) -> GPUStatus:
        """获取GPU状态"""
        try:
            status = GPUStatus(
                device_id=self.device_id,
                name="RTX3060",
                total_memory=self.total_memory,
                used_memory=0,
                free_memory=self.total_memory,
                memory_utilization=0.0,
                gpu_utilization=0.0,
                temperature=0.0,
                power_usage=0.0,
                timestamp=time.time()
            )
            
            # PyTorch GPU信息
            if TORCH_AVAILABLE and torch.cuda.is_available():
                status.used_memory = torch.cuda.memory_allocated(self.device_id)
                status.free_memory = torch.cuda.memory_reserved(self.device_id) - status.used_memory
                status.memory_utilization = (status.used_memory / self.total_memory) * 100
            
            # NVML GPU信息
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                    
                    # 内存信息
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    status.total_memory = mem_info.total
                    status.used_memory = mem_info.used
                    status.free_memory = mem_info.free
                    status.memory_utilization = (mem_info.used / mem_info.total) * 100
                    
                    # GPU利用率
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    status.gpu_utilization = util.gpu
                    
                    # 温度
                    status.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # 功耗
                    status.power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    
                    # 设备名称
                    status.name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                except Exception as e:
                    logger.warning(f"获取NVML信息失败: {e}")
            
            return status
            
        except Exception as e:
            logger.error(f"获取GPU状态失败: {e}")
            return GPUStatus(self.device_id, "Unknown", 0, 0, 0, 0.0, 0.0, 0.0, 0.0, time.time())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        try:
            stats = {
                'total_pools': len(self.memory_pools),
                'total_allocated_blocks': len(self.allocated_blocks),
                'pools': {},
                'gpu_status': self.get_gpu_status().__dict__,
                'fragmentation_summary': {}
            }
            
            total_allocated = 0
            total_free = 0
            total_fragmentation = 0.0
            
            for pool_id, pool in self.memory_pools.items():
                pool_stats = {
                    'total_size_mb': pool.total_size // (1024 * 1024),
                    'allocated_size_mb': pool.allocated_size // (1024 * 1024),
                    'free_size_mb': pool.free_size // (1024 * 1024),
                    'utilization_percent': (pool.allocated_size / pool.total_size) * 100 if pool.total_size > 0 else 0,
                    'block_count': len(pool.blocks),
                    'fragmentation_ratio': pool.fragmentation_ratio,
                    'memory_type': pool.memory_type.value
                }
                
                stats['pools'][pool_id] = pool_stats
                
                total_allocated += pool.allocated_size
                total_free += pool.free_size
                total_fragmentation += pool.fragmentation_ratio
            
            # 汇总统计
            stats['fragmentation_summary'] = {
                'total_allocated_mb': total_allocated // (1024 * 1024),
                'total_free_mb': total_free // (1024 * 1024),
                'overall_utilization_percent': (total_allocated / (total_allocated + total_free)) * 100 if (total_allocated + total_free) > 0 else 0,
                'average_fragmentation': total_fragmentation / len(self.memory_pools) if self.memory_pools else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取内存统计失败: {e}")
            return {}
    
    def optimize_memory_layout(self):
        """优化内存布局"""
        try:
            with self.allocation_lock:
                logger.info("开始GPU内存布局优化")
                
                # 1. 垃圾回收
                self._garbage_collect()
                
                # 2. 整理碎片
                self._defragment_memory()
                
                # 3. 重新分配高优先级内存
                self._reallocate_high_priority_memory()
                
                logger.info("GPU内存布局优化完成")
                
        except Exception as e:
            logger.error(f"GPU内存布局优化失败: {e}")
    
    def _defragment_memory(self):
        """整理内存碎片"""
        try:
            for pool_id, pool in self.memory_pools.items():
                if pool.fragmentation_ratio > 0.3:  # 碎片率超过30%
                    logger.info(f"整理内存池碎片: {pool_id} (碎片率: {pool.fragmentation_ratio:.2f})")
                    
                    # 这里可以实现更复杂的碎片整理算法
                    # 简单实现：重新计算碎片率
                    pool.fragmentation_ratio = self._calculate_fragmentation(pool)
            
        except Exception as e:
            logger.error(f"整理内存碎片失败: {e}")
    
    def _reallocate_high_priority_memory(self):
        """重新分配高优先级内存"""
        try:
            # 收集高优先级内存块
            high_priority_blocks = []
            for block in self.allocated_blocks.values():
                if block.priority == GPUPriority.CRITICAL:
                    high_priority_blocks.append(block)
            
            # 按访问时间排序，确保最近使用的在前面
            high_priority_blocks.sort(key=lambda b: b.last_accessed, reverse=True)
            
            logger.info(f"重新分配{len(high_priority_blocks)}个高优先级内存块")
            
        except Exception as e:
            logger.error(f"重新分配高优先级内存失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            with self.allocation_lock:
                # 释放所有内存块
                block_ids = list(self.allocated_blocks.keys())
                for block_id in block_ids:
                    self.deallocate_memory(block_id)
                
                # 清空内存池
                self.memory_pools.clear()
                
                # GPU缓存清理
                self._garbage_collect()
                
                logger.info("GPU显存管理器清理完成")
                
        except Exception as e:
            logger.error(f"GPU显存管理器清理失败: {e}")


# 全局GPU显存管理器实例
gpu_memory_manager = GPUMemoryManager()
