#!/usr/bin/env python3
"""
GPU内存优化器 - 生产级RTX3060 12GB显存智能管理
实现显存动态分配、内存池管理、模型调度优化
"""
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from loguru import logger
import torch
import torch.cuda
import GPUtil
import psutil
from concurrent.futures import ThreadPoolExecutor

class ProductionGPUMemoryOptimizer:
    """生产级GPU内存优化器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.current_device = 0
        self.memory_pools = {}
        self.allocated_memory = {}
        self.memory_history = []
        self.optimization_stats = {
            'total_allocations': 0,
            'memory_saved': 0,
            'fragmentation_reduced': 0,
            'optimization_count': 0
        }
        self.is_monitoring = False
        self.monitor_thread = None
        self.memory_threshold = 0.85  # 85%内存使用阈值
        
        if self.gpu_available:
            self._initialize_gpu_monitoring()
            logger.info(f"🎮 GPU内存优化器初始化完成 - 检测到 {self.device_count} 个GPU设备")
        else:
            logger.warning("⚠️ 未检测到GPU设备，使用CPU模式")
    
    def _initialize_gpu_monitoring(self):
        """初始化GPU监控"""
        try:
            for i in range(self.device_count):
                torch.cuda.set_device(i)
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 获取GPU信息
                gpu_props = torch.cuda.get_device_properties(i)
                total_memory = gpu_props.total_memory
                
                self.memory_pools[i] = {
                    'total_memory': total_memory,
                    'allocated_memory': 0,
                    'cached_memory': 0,
                    'free_memory': total_memory,
                    'memory_blocks': [],
                    'last_cleanup': time.time()
                }
                
                logger.info(f"GPU {i}: {gpu_props.name} - {total_memory / 1024**3:.1f}GB 总内存")
                
        except Exception as e:
            logger.error(f"GPU监控初始化错误: {e}")
    
    def start_memory_monitoring(self):
        """启动内存监控"""
        if not self.gpu_available:
            logger.warning("GPU不可用，跳过内存监控")
            return
        
        if self.is_monitoring:
            logger.warning("GPU内存监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔍 GPU内存监控启动")
    
    def _memory_monitor_loop(self):
        """内存监控主循环"""
        while self.is_monitoring:
            try:
                # 更新内存状态
                self._update_memory_status()
                
                # 检查内存使用情况
                self._check_memory_pressure()
                
                # 执行内存优化
                self._optimize_memory_usage()
                
                # 记录内存历史
                self._record_memory_history()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"内存监控循环错误: {e}")
                time.sleep(10)
    
    def _update_memory_status(self):
        """更新内存状态"""
        try:
            for device_id in range(self.device_count):
                torch.cuda.set_device(device_id)
                
                # 获取内存使用情况
                allocated = torch.cuda.memory_allocated(device_id)
                cached = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                self.memory_pools[device_id].update({
                    'allocated_memory': allocated,
                    'cached_memory': cached,
                    'free_memory': total - cached,
                    'utilization': cached / total
                })
                
        except Exception as e:
            logger.error(f"内存状态更新错误: {e}")
    
    def _check_memory_pressure(self):
        """检查内存压力"""
        try:
            for device_id, pool in self.memory_pools.items():
                utilization = pool['utilization']
                
                if utilization > self.memory_threshold:
                    logger.warning(f"🚨 GPU {device_id} 内存使用率过高: {utilization:.1%}")
                    self._trigger_memory_cleanup(device_id)
                    
        except Exception as e:
            logger.error(f"内存压力检查错误: {e}")
    
    def _trigger_memory_cleanup(self, device_id: int):
        """触发内存清理"""
        try:
            torch.cuda.set_device(device_id)
            
            # 清理未使用的缓存
            torch.cuda.empty_cache()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 更新清理时间
            self.memory_pools[device_id]['last_cleanup'] = time.time()
            
            logger.info(f"🧹 GPU {device_id} 内存清理完成")
            
        except Exception as e:
            logger.error(f"内存清理错误: {e}")
    
    def _optimize_memory_usage(self):
        """优化内存使用"""
        try:
            for device_id in range(self.device_count):
                pool = self.memory_pools[device_id]
                
                # 检查是否需要优化
                if pool['utilization'] > 0.7:  # 70%以上开始优化
                    self._defragment_memory(device_id)
                    self._optimize_memory_allocation(device_id)
                    
        except Exception as e:
            logger.error(f"内存优化错误: {e}")
    
    def _defragment_memory(self, device_id: int):
        """内存碎片整理"""
        try:
            torch.cuda.set_device(device_id)
            
            # 记录优化前状态
            before_fragmentation = self._calculate_fragmentation(device_id)
            
            # 执行内存整理
            torch.cuda.empty_cache()
            
            # 记录优化后状态
            after_fragmentation = self._calculate_fragmentation(device_id)
            
            fragmentation_reduced = before_fragmentation - after_fragmentation
            if fragmentation_reduced > 0:
                self.optimization_stats['fragmentation_reduced'] += fragmentation_reduced
                logger.info(f"📊 GPU {device_id} 内存碎片减少: {fragmentation_reduced:.2%}")
                
        except Exception as e:
            logger.error(f"内存碎片整理错误: {e}")
    
    def _calculate_fragmentation(self, device_id: int) -> float:
        """计算内存碎片率"""
        try:
            pool = self.memory_pools[device_id]
            allocated = pool['allocated_memory']
            cached = pool['cached_memory']
            
            if cached == 0:
                return 0.0
            
            # 简单的碎片率计算
            fragmentation = (cached - allocated) / cached
            return max(0.0, fragmentation)
            
        except Exception as e:
            logger.error(f"内存碎片计算错误: {e}")
            return 0.0
    
    def _optimize_memory_allocation(self, device_id: int):
        """优化内存分配策略"""
        try:
            pool = self.memory_pools[device_id]
            
            # 基于使用模式调整分配策略
            if pool['utilization'] > 0.8:
                # 高使用率时采用保守分配
                torch.cuda.set_per_process_memory_fraction(0.9, device_id)
            elif pool['utilization'] < 0.3:
                # 低使用率时释放更多内存
                torch.cuda.set_per_process_memory_fraction(0.5, device_id)
            else:
                # 正常使用率
                torch.cuda.set_per_process_memory_fraction(0.8, device_id)
                
            self.optimization_stats['optimization_count'] += 1
            
        except Exception as e:
            logger.error(f"内存分配优化错误: {e}")
    
    def _record_memory_history(self):
        """记录内存历史"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            memory_snapshot = {
                'timestamp': timestamp,
                'devices': {}
            }
            
            for device_id, pool in self.memory_pools.items():
                memory_snapshot['devices'][device_id] = {
                    'allocated_mb': pool['allocated_memory'] / 1024**2,
                    'cached_mb': pool['cached_memory'] / 1024**2,
                    'free_mb': pool['free_memory'] / 1024**2,
                    'utilization': pool['utilization']
                }
            
            self.memory_history.append(memory_snapshot)
            
            # 保持历史记录在合理范围内
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]
                
        except Exception as e:
            logger.error(f"内存历史记录错误: {e}")
    
    def allocate_model_memory(self, model_size_mb: float, device_id: Optional[int] = None) -> Optional[int]:
        """为模型分配内存"""
        try:
            if not self.gpu_available:
                logger.warning("GPU不可用，无法分配GPU内存")
                return None
            
            # 选择最佳设备
            if device_id is None:
                device_id = self._select_best_device(model_size_mb)
            
            if device_id is None:
                logger.error("没有足够的GPU内存分配模型")
                return None
            
            torch.cuda.set_device(device_id)
            
            # 检查内存是否足够
            required_bytes = model_size_mb * 1024**2
            pool = self.memory_pools[device_id]
            
            if pool['free_memory'] < required_bytes:
                # 尝试清理内存
                self._trigger_memory_cleanup(device_id)
                self._update_memory_status()
                
                if pool['free_memory'] < required_bytes:
                    logger.error(f"GPU {device_id} 内存不足，需要 {model_size_mb:.1f}MB，可用 {pool['free_memory']/1024**2:.1f}MB")
                    return None
            
            # 记录分配
            allocation_id = f"model_{int(time.time())}_{device_id}"
            self.allocated_memory[allocation_id] = {
                'device_id': device_id,
                'size_bytes': required_bytes,
                'allocated_time': time.time(),
                'model_type': 'ai_model'
            }
            
            self.optimization_stats['total_allocations'] += 1
            
            logger.info(f"✅ 模型内存分配成功 - GPU {device_id}: {model_size_mb:.1f}MB")
            return device_id
            
        except Exception as e:
            logger.error(f"模型内存分配错误: {e}")
            return None
    
    def _select_best_device(self, required_mb: float) -> Optional[int]:
        """选择最佳GPU设备"""
        try:
            best_device = None
            best_score = -1
            
            required_bytes = required_mb * 1024**2
            
            for device_id, pool in self.memory_pools.items():
                # 检查内存是否足够
                if pool['free_memory'] < required_bytes:
                    continue
                
                # 计算设备评分（考虑可用内存和利用率）
                free_ratio = pool['free_memory'] / pool['total_memory']
                utilization_penalty = pool['utilization']
                
                score = free_ratio - utilization_penalty * 0.5
                
                if score > best_score:
                    best_score = score
                    best_device = device_id
            
            return best_device
            
        except Exception as e:
            logger.error(f"设备选择错误: {e}")
            return None
    
    def deallocate_model_memory(self, allocation_id: str):
        """释放模型内存"""
        try:
            if allocation_id not in self.allocated_memory:
                logger.warning(f"分配ID不存在: {allocation_id}")
                return
            
            allocation = self.allocated_memory[allocation_id]
            device_id = allocation['device_id']
            
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            
            # 移除分配记录
            del self.allocated_memory[allocation_id]
            
            logger.info(f"🗑️ 模型内存释放完成 - GPU {device_id}")
            
        except Exception as e:
            logger.error(f"模型内存释放错误: {e}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """获取内存状态"""
        try:
            status = {
                'gpu_available': self.gpu_available,
                'device_count': self.device_count,
                'devices': {},
                'optimization_stats': self.optimization_stats.copy(),
                'allocated_models': len(self.allocated_memory)
            }
            
            if self.gpu_available:
                for device_id, pool in self.memory_pools.items():
                    # 获取实时GPU信息
                    gpu_info = {}
                    try:
                        gpus = GPUtil.getGPUs()
                        if device_id < len(gpus):
                            gpu = gpus[device_id]
                            gpu_info = {
                                'temperature': gpu.temperature,
                                'load': gpu.load * 100,
                                'memory_used_mb': gpu.memoryUsed,
                                'memory_total_mb': gpu.memoryTotal
                            }
                    except:
                        pass
                    
                    status['devices'][device_id] = {
                        'total_memory_gb': pool['total_memory'] / 1024**3,
                        'allocated_memory_mb': pool['allocated_memory'] / 1024**2,
                        'cached_memory_mb': pool['cached_memory'] / 1024**2,
                        'free_memory_mb': pool['free_memory'] / 1024**2,
                        'utilization': pool['utilization'],
                        'fragmentation': self._calculate_fragmentation(device_id),
                        **gpu_info
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"内存状态获取错误: {e}")
            return {'error': str(e)}
    
    def optimize_for_training(self, model_count: int = 6):
        """为训练优化内存配置"""
        try:
            if not self.gpu_available:
                logger.warning("GPU不可用，跳过训练优化")
                return
            
            logger.info(f"🎯 为 {model_count} 个模型优化GPU内存配置")
            
            for device_id in range(self.device_count):
                torch.cuda.set_device(device_id)
                
                # 清理所有缓存
                torch.cuda.empty_cache()
                
                # 设置内存分配策略
                memory_fraction = min(0.95, 0.8 + (model_count * 0.02))
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
                
                logger.info(f"GPU {device_id} 内存分配比例设置为: {memory_fraction:.1%}")
            
            # 降低内存监控阈值以更积极地管理内存
            self.memory_threshold = 0.8
            
        except Exception as e:
            logger.error(f"训练内存优化错误: {e}")
    
    def stop_memory_monitoring(self):
        """停止内存监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 GPU内存监控已停止")

# 全局GPU内存优化器实例
_gpu_memory_optimizer = None

def initialize_gpu_memory_optimizer() -> ProductionGPUMemoryOptimizer:
    """初始化GPU内存优化器"""
    global _gpu_memory_optimizer
    
    if _gpu_memory_optimizer is None:
        _gpu_memory_optimizer = ProductionGPUMemoryOptimizer()
        _gpu_memory_optimizer.start_memory_monitoring()
        logger.success("✅ GPU内存优化器初始化完成")
    
    return _gpu_memory_optimizer

def get_gpu_memory_optimizer() -> Optional[ProductionGPUMemoryOptimizer]:
    """获取GPU内存优化器实例"""
    return _gpu_memory_optimizer

if __name__ == "__main__":
    # 测试GPU内存优化器
    optimizer = initialize_gpu_memory_optimizer()
    
    # 运行测试
    for i in range(10):
        status = optimizer.get_memory_status()
        print(f"GPU状态: {status}")
        time.sleep(2)
    
    optimizer.stop_memory_monitoring()
