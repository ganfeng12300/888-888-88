"""
🤖 GPU多模型调度器 - 6个AI模型并行训练调度系统
智能调度6个AI模型在RTX3060 12GB上并行训练，支持动态资源分配、优先级管理、温度控制
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from loguru import logger
from .gpu_memory_optimizer import GPUMemoryManager, GPUPriority, GPUMemoryType


class ModelType(Enum):
    """模型类型"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"   # 强化学习
    TRANSFORMER = "transformer"                         # Transformer
    LSTM = "lstm"                                       # LSTM
    ENSEMBLE = "ensemble"                               # 集成学习
    META_LEARNING = "meta_learning"                     # 元学习
    TRANSFER_LEARNING = "transfer_learning"             # 迁移学习


class ModelStatus(Enum):
    """模型状态"""
    IDLE = "idle"                   # 空闲
    LOADING = "loading"             # 加载中
    TRAINING = "training"           # 训练中
    INFERENCE = "inference"         # 推理中
    PAUSED = "paused"              # 暂停
    ERROR = "error"                # 错误


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str                   # 模型ID
    model_type: ModelType           # 模型类型
    memory_requirement: int         # 内存需求(MB)
    priority: GPUPriority           # 优先级
    max_batch_size: int             # 最大批次大小
    training_enabled: bool          # 是否启用训练
    inference_enabled: bool         # 是否启用推理
    checkpoint_interval: int        # 检查点间隔(秒)
    max_training_time: int          # 最大训练时间(秒)


@dataclass
class ModelInstance:
    """模型实例"""
    config: ModelConfig             # 模型配置
    status: ModelStatus             # 当前状态
    model: Optional[Any]            # 模型对象
    optimizer: Optional[Any]        # 优化器
    memory_blocks: List[str]        # 分配的内存块
    gpu_utilization: float          # GPU利用率
    memory_usage: int               # 内存使用量(MB)
    training_steps: int             # 训练步数
    last_checkpoint: float          # 最后检查点时间
    created_at: float               # 创建时间
    last_active: float              # 最后活跃时间
    error_count: int                # 错误次数


class GPUModelScheduler:
    """GPU多模型调度器"""
    
    def __init__(self, memory_manager: GPUMemoryManager):
        self.memory_manager = memory_manager
        self.models: Dict[str, ModelInstance] = {}
        self.scheduler_lock = threading.RLock()
        self.running = False
        self.scheduler_task = None
        
        # 调度配置
        self.max_concurrent_models = 6         # 最大并发模型数
        self.scheduling_interval = 1.0         # 调度间隔(秒)
        self.temperature_threshold = 80.0      # 温度阈值(°C)
        self.memory_threshold = 0.9            # 内存阈值(90%)
        
        # 预定义模型配置
        self.model_configs = self._create_model_configs()
        
        logger.info("GPU多模型调度器初始化完成")
    
    def _create_model_configs(self) -> Dict[str, ModelConfig]:
        """创建模型配置"""
        configs = {
            # 强化学习模型 - 高优先级
            'rl_model': ModelConfig(
                model_id='rl_model',
                model_type=ModelType.REINFORCEMENT_LEARNING,
                memory_requirement=1024,  # 1GB
                priority=GPUPriority.CRITICAL,
                max_batch_size=64,
                training_enabled=True,
                inference_enabled=True,
                checkpoint_interval=300,  # 5分钟
                max_training_time=3600    # 1小时
            ),
            
            # Transformer模型 - 高优先级
            'transformer_model': ModelConfig(
                model_id='transformer_model',
                model_type=ModelType.TRANSFORMER,
                memory_requirement=1024,  # 1GB
                priority=GPUPriority.CRITICAL,
                max_batch_size=32,
                training_enabled=True,
                inference_enabled=True,
                checkpoint_interval=300,
                max_training_time=3600
            ),
            
            # LSTM模型 - 中等优先级
            'lstm_model': ModelConfig(
                model_id='lstm_model',
                model_type=ModelType.LSTM,
                memory_requirement=512,   # 512MB
                priority=GPUPriority.HIGH,
                max_batch_size=128,
                training_enabled=True,
                inference_enabled=True,
                checkpoint_interval=600,  # 10分钟
                max_training_time=7200    # 2小时
            ),
            
            # 集成学习模型 - 中等优先级
            'ensemble_model': ModelConfig(
                model_id='ensemble_model',
                model_type=ModelType.ENSEMBLE,
                memory_requirement=768,   # 768MB
                priority=GPUPriority.HIGH,
                max_batch_size=96,
                training_enabled=True,
                inference_enabled=True,
                checkpoint_interval=600,
                max_training_time=7200
            ),
            
            # 元学习模型 - 低优先级
            'meta_model': ModelConfig(
                model_id='meta_model',
                model_type=ModelType.META_LEARNING,
                memory_requirement=512,   # 512MB
                priority=GPUPriority.NORMAL,
                max_batch_size=64,
                training_enabled=True,
                inference_enabled=False,
                checkpoint_interval=900,  # 15分钟
                max_training_time=10800   # 3小时
            ),
            
            # 迁移学习模型 - 低优先级
            'transfer_model': ModelConfig(
                model_id='transfer_model',
                model_type=ModelType.TRANSFER_LEARNING,
                memory_requirement=512,   # 512MB
                priority=GPUPriority.NORMAL,
                max_batch_size=64,
                training_enabled=True,
                inference_enabled=False,
                checkpoint_interval=900,
                max_training_time=10800
            )
        }
        
        return configs
    
    async def start_scheduler(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # 初始化所有模型
        await self._initialize_models()
        
        logger.info("GPU多模型调度器已启动")
    
    async def stop_scheduler(self):
        """停止调度器"""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 清理所有模型
        await self._cleanup_models()
        
        logger.info("GPU多模型调度器已停止")
    
    async def _scheduler_loop(self):
        """调度循环"""
        while self.running:
            try:
                # 检查GPU状态
                gpu_status = self.memory_manager.get_gpu_status()
                
                # 温度控制
                if gpu_status.temperature > self.temperature_threshold:
                    await self._handle_overheating()
                
                # 内存管理
                if gpu_status.memory_utilization > self.memory_threshold * 100:
                    await self._handle_memory_pressure()
                
                # 模型调度
                await self._schedule_models()
                
                # 检查点保存
                await self._save_checkpoints()
                
                await asyncio.sleep(self.scheduling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _initialize_models(self):
        """初始化所有模型"""
        try:
            with self.scheduler_lock:
                for model_id, config in self.model_configs.items():
                    await self._load_model(model_id, config)
            
            logger.info(f"初始化完成: {len(self.models)}个模型")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
    
    async def _load_model(self, model_id: str, config: ModelConfig) -> bool:
        """加载模型"""
        try:
            if model_id in self.models:
                logger.warning(f"模型已存在: {model_id}")
                return False
            
            # 分配内存
            memory_blocks = []
            
            # 模型权重内存
            weight_block = self.memory_manager.allocate_memory(
                'model_weights',
                config.memory_requirement * 1024 * 1024,  # 转换为字节
                model_id,
                GPUMemoryType.MODEL_WEIGHTS,
                config.priority
            )
            
            if weight_block:
                memory_blocks.append(weight_block)
            else:
                logger.error(f"模型 {model_id} 内存分配失败")
                return False
            
            # 创建模型实例
            model_instance = ModelInstance(
                config=config,
                status=ModelStatus.LOADING,
                model=None,
                optimizer=None,
                memory_blocks=memory_blocks,
                gpu_utilization=0.0,
                memory_usage=config.memory_requirement,
                training_steps=0,
                last_checkpoint=time.time(),
                created_at=time.time(),
                last_active=time.time(),
                error_count=0
            )
            
            # 创建实际模型
            if TORCH_AVAILABLE:
                model_instance.model = self._create_torch_model(config)
                if config.training_enabled:
                    model_instance.optimizer = self._create_optimizer(model_instance.model)
            
            model_instance.status = ModelStatus.IDLE
            self.models[model_id] = model_instance
            
            logger.info(f"模型加载成功: {model_id} ({config.memory_requirement}MB)")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {model_id} - {e}")
            return False
    
    def _create_torch_model(self, config: ModelConfig) -> Optional[nn.Module]:
        """创建PyTorch模型"""
        try:
            if not TORCH_AVAILABLE:
                return None
            
            # 根据模型类型创建不同的模型架构
            if config.model_type == ModelType.REINFORCEMENT_LEARNING:
                # 简单的DQN网络
                model = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
                )
            
            elif config.model_type == ModelType.TRANSFORMER:
                # 简单的Transformer
                model = nn.Sequential(
                    nn.Linear(100, 512),
                    nn.TransformerEncoderLayer(512, 8),
                    nn.Linear(512, 10)
                )
            
            elif config.model_type == ModelType.LSTM:
                # LSTM网络
                model = nn.Sequential(
                    nn.LSTM(100, 256, batch_first=True),
                    nn.Linear(256, 10)
                )
            
            else:
                # 默认全连接网络
                model = nn.Sequential(
                    nn.Linear(100, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
                )
            
            # 移动到GPU
            device = f'cuda:{self.memory_manager.device_id}'
            model = model.to(device)
            
            return model
            
        except Exception as e:
            logger.error(f"创建PyTorch模型失败: {e}")
            return None
    
    def _create_optimizer(self, model: nn.Module) -> Optional[Any]:
        """创建优化器"""
        try:
            if not TORCH_AVAILABLE or model is None:
                return None
            
            return torch.optim.Adam(model.parameters(), lr=0.001)
            
        except Exception as e:
            logger.error(f"创建优化器失败: {e}")
            return None
    
    async def _schedule_models(self):
        """调度模型"""
        try:
            with self.scheduler_lock:
                # 获取活跃模型
                active_models = [m for m in self.models.values() 
                               if m.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]]
                
                # 如果活跃模型数量超过限制，暂停低优先级模型
                if len(active_models) > self.max_concurrent_models:
                    await self._pause_low_priority_models(active_models)
                
                # 启动空闲的高优先级模型
                await self._activate_high_priority_models()
                
                # 更新模型状态
                self._update_model_status()
            
        except Exception as e:
            logger.error(f"模型调度失败: {e}")
    
    async def _pause_low_priority_models(self, active_models: List[ModelInstance]):
        """暂停低优先级模型"""
        try:
            # 按优先级排序，低优先级在前
            sorted_models = sorted(active_models, 
                                 key=lambda m: (m.config.priority.value, m.last_active))
            
            paused_count = 0
            target_pause = len(active_models) - self.max_concurrent_models
            
            for model in sorted_models:
                if paused_count >= target_pause:
                    break
                
                if model.config.priority in [GPUPriority.LOW, GPUPriority.NORMAL]:
                    model.status = ModelStatus.PAUSED
                    paused_count += 1
                    logger.info(f"暂停低优先级模型: {model.config.model_id}")
            
        except Exception as e:
            logger.error(f"暂停低优先级模型失败: {e}")
    
    async def _activate_high_priority_models(self):
        """激活高优先级模型"""
        try:
            # 获取空闲和暂停的高优先级模型
            inactive_models = [m for m in self.models.values() 
                             if m.status in [ModelStatus.IDLE, ModelStatus.PAUSED] 
                             and m.config.priority in [GPUPriority.CRITICAL, GPUPriority.HIGH]]
            
            # 按优先级排序，高优先级在前
            sorted_models = sorted(inactive_models, 
                                 key=lambda m: (m.config.priority.value, -m.last_active), 
                                 reverse=True)
            
            active_count = len([m for m in self.models.values() 
                              if m.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]])
            
            for model in sorted_models:
                if active_count >= self.max_concurrent_models:
                    break
                
                # 检查是否有足够资源
                if self._has_sufficient_resources(model):
                    if model.config.training_enabled:
                        model.status = ModelStatus.TRAINING
                    else:
                        model.status = ModelStatus.INFERENCE
                    
                    model.last_active = time.time()
                    active_count += 1
                    logger.info(f"激活高优先级模型: {model.config.model_id}")
            
        except Exception as e:
            logger.error(f"激活高优先级模型失败: {e}")
    
    def _has_sufficient_resources(self, model: ModelInstance) -> bool:
        """检查是否有足够资源"""
        try:
            gpu_status = self.memory_manager.get_gpu_status()
            
            # 检查内存
            required_memory = model.config.memory_requirement * 1024 * 1024  # 转换为字节
            if gpu_status.free_memory < required_memory:
                return False
            
            # 检查温度
            if gpu_status.temperature > self.temperature_threshold * 0.9:  # 90%阈值
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查资源失败: {e}")
            return False
    
    def _update_model_status(self):
        """更新模型状态"""
        try:
            current_time = time.time()
            
            for model in self.models.values():
                # 检查训练超时
                if (model.status == ModelStatus.TRAINING and 
                    current_time - model.last_active > model.config.max_training_time):
                    model.status = ModelStatus.IDLE
                    logger.info(f"模型训练超时，切换为空闲: {model.config.model_id}")
                
                # 更新GPU利用率（模拟）
                if model.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]:
                    model.gpu_utilization = min(100.0, model.gpu_utilization + 5.0)
                else:
                    model.gpu_utilization = max(0.0, model.gpu_utilization - 10.0)
            
        except Exception as e:
            logger.error(f"更新模型状态失败: {e}")
    
    async def _handle_overheating(self):
        """处理过热"""
        try:
            logger.warning("GPU温度过高，启动降温措施")
            
            # 暂停所有低优先级模型
            with self.scheduler_lock:
                for model in self.models.values():
                    if (model.config.priority in [GPUPriority.LOW, GPUPriority.NORMAL] and
                        model.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]):
                        model.status = ModelStatus.PAUSED
                        logger.info(f"因过热暂停模型: {model.config.model_id}")
            
            # 等待降温
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"处理过热失败: {e}")
    
    async def _handle_memory_pressure(self):
        """处理内存压力"""
        try:
            logger.warning("GPU内存压力过大，启动内存清理")
            
            # 优化内存布局
            self.memory_manager.optimize_memory_layout()
            
            # 暂停部分模型
            with self.scheduler_lock:
                paused_count = 0
                for model in sorted(self.models.values(), 
                                  key=lambda m: (m.config.priority.value, m.last_active)):
                    if paused_count >= 2:  # 最多暂停2个模型
                        break
                    
                    if (model.config.priority in [GPUPriority.LOW, GPUPriority.NORMAL] and
                        model.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]):
                        model.status = ModelStatus.PAUSED
                        paused_count += 1
                        logger.info(f"因内存压力暂停模型: {model.config.model_id}")
            
        except Exception as e:
            logger.error(f"处理内存压力失败: {e}")
    
    async def _save_checkpoints(self):
        """保存检查点"""
        try:
            current_time = time.time()
            
            for model in self.models.values():
                if (model.status == ModelStatus.TRAINING and
                    current_time - model.last_checkpoint > model.config.checkpoint_interval):
                    
                    await self._save_model_checkpoint(model)
                    model.last_checkpoint = current_time
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    async def _save_model_checkpoint(self, model: ModelInstance):
        """保存模型检查点"""
        try:
            if not TORCH_AVAILABLE or model.model is None:
                return
            
            checkpoint_path = f"checkpoints/{model.config.model_id}_checkpoint.pth"
            
            # 创建检查点目录
            import os
            os.makedirs("checkpoints", exist_ok=True)
            
            # 保存模型状态
            checkpoint = {
                'model_state_dict': model.model.state_dict(),
                'training_steps': model.training_steps,
                'timestamp': time.time()
            }
            
            if model.optimizer:
                checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存检查点: {model.config.model_id}")
            
        except Exception as e:
            logger.error(f"保存模型检查点失败: {e}")
    
    async def _cleanup_models(self):
        """清理所有模型"""
        try:
            with self.scheduler_lock:
                for model_id, model in self.models.items():
                    # 释放内存块
                    for block_id in model.memory_blocks:
                        self.memory_manager.deallocate_memory(block_id)
                    
                    # 清理模型对象
                    if model.model:
                        del model.model
                    if model.optimizer:
                        del model.optimizer
                    
                    logger.info(f"清理模型: {model_id}")
                
                self.models.clear()
            
            logger.info("所有模型清理完成")
            
        except Exception as e:
            logger.error(f"清理模型失败: {e}")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计"""
        try:
            stats = {
                'total_models': len(self.models),
                'running': self.running,
                'model_status': {},
                'resource_usage': {},
                'performance_summary': {}
            }
            
            # 模型状态统计
            status_counts = {}
            total_memory_usage = 0
            total_gpu_utilization = 0.0
            
            for model in self.models.values():
                status = model.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                total_memory_usage += model.memory_usage
                total_gpu_utilization += model.gpu_utilization
                
                stats['model_status'][model.config.model_id] = {
                    'status': status,
                    'priority': model.config.priority.value,
                    'memory_usage_mb': model.memory_usage,
                    'gpu_utilization': model.gpu_utilization,
                    'training_steps': model.training_steps,
                    'error_count': model.error_count,
                    'uptime': time.time() - model.created_at
                }
            
            stats['resource_usage'] = {
                'total_memory_usage_mb': total_memory_usage,
                'avg_gpu_utilization': total_gpu_utilization / len(self.models) if self.models else 0,
                'status_distribution': status_counts
            }
            
            # GPU状态
            gpu_status = self.memory_manager.get_gpu_status()
            stats['performance_summary'] = {
                'gpu_temperature': gpu_status.temperature,
                'gpu_utilization': gpu_status.gpu_utilization,
                'memory_utilization': gpu_status.memory_utilization,
                'active_models': len([m for m in self.models.values() 
                                    if m.status in [ModelStatus.TRAINING, ModelStatus.INFERENCE]])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取调度器统计失败: {e}")
            return {}


# 全局GPU模型调度器实例
gpu_model_scheduler = None

def initialize_gpu_scheduler(memory_manager: GPUMemoryManager):
    """初始化GPU模型调度器"""
    global gpu_model_scheduler
    gpu_model_scheduler = GPUModelScheduler(memory_manager)
    return gpu_model_scheduler
