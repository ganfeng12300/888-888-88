#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - AI模型调度中心
8大AI智能体统一协调调度，智能决策融合
专为史诗级AI量化交易设计，生产级实盘交易标准
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import psutil
import GPUtil

class AIModelStatus(Enum):
    """AI模型状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"
    OFFLINE = "offline"

class AIModelPriority(Enum):
    """AI模型优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AIModelInfo:
    """AI模型信息"""
    model_id: str
    model_name: str
    model_type: str
    status: AIModelStatus
    priority: AIModelPriority
    cpu_cores: int
    gpu_memory: float  # GB
    last_prediction: Optional[Dict[str, Any]] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    performance_score: float = 0.0
    accuracy: float = 0.0
    latency_ms: float = 0.0
    error_count: int = 0
    prediction_count: int = 0
    is_active: bool = True

@dataclass
class SchedulingTask:
    """调度任务"""
    task_id: str
    model_id: str
    task_type: str  # 'predict', 'train', 'update'
    priority: int
    data: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: str = ""

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.cpu_cores = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.gpu_info = self._get_gpu_info()
        
        # 资源分配
        self.allocated_cpu = {}  # model_id -> cores
        self.allocated_gpu = {}  # model_id -> memory_gb
        self.cpu_usage = {}     # core_id -> usage
        self.gpu_usage = {}     # gpu_id -> usage
        
        logger.info(f"🖥️ 资源管理器初始化: CPU {self.cpu_cores}核, 内存 {self.total_memory:.1f}GB")
        
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                })
            return gpu_info
        except Exception as e:
            logger.warning(f"⚠️ 获取GPU信息失败: {e}")
            return []
    
    def allocate_resources(self, model_id: str, cpu_cores: int, gpu_memory: float) -> bool:
        """分配资源"""
        try:
            # 检查CPU资源
            allocated_cores = sum(self.allocated_cpu.values())
            if allocated_cores + cpu_cores > self.cpu_cores:
                logger.warning(f"⚠️ CPU资源不足: 需要{cpu_cores}核，可用{self.cpu_cores - allocated_cores}核")
                return False
            
            # 检查GPU资源
            if gpu_memory > 0 and self.gpu_info:
                total_allocated_gpu = sum(self.allocated_gpu.values())
                available_gpu = self.gpu_info[0]['memory_total'] / 1024  # 转换为GB
                
                if total_allocated_gpu + gpu_memory > available_gpu:
                    logger.warning(f"⚠️ GPU内存不足: 需要{gpu_memory}GB，可用{available_gpu - total_allocated_gpu}GB")
                    return False
            
            # 分配资源
            self.allocated_cpu[model_id] = cpu_cores
            if gpu_memory > 0:
                self.allocated_gpu[model_id] = gpu_memory
            
            logger.info(f"✅ 资源分配成功: {model_id} CPU {cpu_cores}核, GPU {gpu_memory}GB")
            return True
            
        except Exception as e:
            logger.error(f"❌ 资源分配失败: {e}")
            return False
    
    def release_resources(self, model_id: str):
        """释放资源"""
        try:
            if model_id in self.allocated_cpu:
                del self.allocated_cpu[model_id]
            if model_id in self.allocated_gpu:
                del self.allocated_gpu[model_id]
            
            logger.info(f"🔄 资源已释放: {model_id}")
            
        except Exception as e:
            logger.error(f"❌ 资源释放失败: {e}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 更新GPU信息
            gpu_info = self._get_gpu_info()
            
            return {
                'cpu': {
                    'cores': self.cpu_cores,
                    'allocated': sum(self.allocated_cpu.values()),
                    'usage_percent': cpu_percent,
                    'available': self.cpu_cores - sum(self.allocated_cpu.values())
                },
                'memory': {
                    'total_gb': self.total_memory,
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'usage_percent': memory.percent
                },
                'gpu': gpu_info
            }
            
        except Exception as e:
            logger.error(f"❌ 获取资源使用情况失败: {e}")
            return {}

class AIScheduler:
    """🦊 猎狐AI - AI模型调度中心"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.resource_manager = ResourceManager()
        
        # AI模型注册表
        self.ai_models = {}  # model_id -> AIModelInfo
        self.model_instances = {}  # model_id -> model_instance
        
        # 任务队列
        self.task_queue = asyncio.Queue()
        self.completed_tasks = {}  # task_id -> SchedulingTask
        self.running_tasks = {}   # task_id -> SchedulingTask
        
        # 调度器状态
        self.is_running = False
        self.scheduler_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_latency': 0.0,
            'active_models': 0,
            'total_predictions': 0
        }
        
        logger.info("🦊 猎狐AI模型调度中心初始化完成")
    
    async def register_ai_model(self, model_id: str, model_name: str, model_type: str,
                               model_instance: Any, cpu_cores: int = 2, 
                               gpu_memory: float = 1.0, priority: AIModelPriority = AIModelPriority.NORMAL) -> bool:
        """注册AI模型"""
        try:
            # 分配资源
            if not self.resource_manager.allocate_resources(model_id, cpu_cores, gpu_memory):
                return False
            
            # 创建模型信息
            model_info = AIModelInfo(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                status=AIModelStatus.INITIALIZING,
                priority=priority,
                cpu_cores=cpu_cores,
                gpu_memory=gpu_memory
            )
            
            # 注册模型
            self.ai_models[model_id] = model_info
            self.model_instances[model_id] = model_instance
            
            # 初始化模型
            await self._initialize_model(model_id)
            
            self.stats['active_models'] = len([m for m in self.ai_models.values() if m.is_active])
            
            logger.success(f"✅ AI模型注册成功: {model_name} ({model_id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ AI模型注册失败 {model_id}: {e}")
            return False
    
    async def _initialize_model(self, model_id: str):
        """初始化AI模型"""
        try:
            model_info = self.ai_models[model_id]
            model_instance = self.model_instances[model_id]
            
            model_info.status = AIModelStatus.INITIALIZING
            
            # 如果模型有初始化方法，调用它
            if hasattr(model_instance, 'initialize'):
                await model_instance.initialize()
            
            model_info.status = AIModelStatus.READY
            model_info.last_update = datetime.now(timezone.utc)
            
            logger.info(f"✅ AI模型初始化完成: {model_id}")
            
        except Exception as e:
            logger.error(f"❌ AI模型初始化失败 {model_id}: {e}")
            self.ai_models[model_id].status = AIModelStatus.ERROR
            self.ai_models[model_id].error_count += 1
    
    async def schedule_prediction(self, model_id: str, data: Dict[str, Any], 
                                priority: int = 2) -> Optional[str]:
        """调度预测任务"""
        try:
            if model_id not in self.ai_models:
                logger.error(f"❌ AI模型不存在: {model_id}")
                return None
            
            # 创建任务
            task = SchedulingTask(
                task_id=f"pred_{int(time.time() * 1000)}_{model_id}",
                model_id=model_id,
                task_type='predict',
                priority=priority,
                data=data,
                created_at=datetime.now(timezone.utc)
            )
            
            # 添加到队列
            await self.task_queue.put(task)
            self.stats['total_tasks'] += 1
            
            logger.debug(f"📋 预测任务已调度: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"❌ 调度预测任务失败: {e}")
            return None
    
    async def schedule_training(self, model_id: str, data: Dict[str, Any],
                              priority: int = 1) -> Optional[str]:
        """调度训练任务"""
        try:
            if model_id not in self.ai_models:
                logger.error(f"❌ AI模型不存在: {model_id}")
                return None
            
            # 创建任务
            task = SchedulingTask(
                task_id=f"train_{int(time.time() * 1000)}_{model_id}",
                model_id=model_id,
                task_type='train',
                priority=priority,
                data=data,
                created_at=datetime.now(timezone.utc)
            )
            
            # 添加到队列
            await self.task_queue.put(task)
            self.stats['total_tasks'] += 1
            
            logger.debug(f"📋 训练任务已调度: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"❌ 调度训练任务失败: {e}")
            return None
    
    async def get_ensemble_prediction(self, data: Dict[str, Any], 
                                    model_ids: List[str] = None) -> Dict[str, Any]:
        """获取集成预测结果"""
        try:
            if model_ids is None:
                # 使用所有可用的模型
                model_ids = [mid for mid, info in self.ai_models.items() 
                           if info.status == AIModelStatus.READY and info.is_active]
            
            if not model_ids:
                logger.warning("⚠️ 没有可用的AI模型进行集成预测")
                return {'prediction': 0.0, 'confidence': 0.0, 'models_used': []}
            
            # 并行调度预测任务
            task_ids = []
            for model_id in model_ids:
                task_id = await self.schedule_prediction(model_id, data, priority=3)
                if task_id:
                    task_ids.append(task_id)
            
            if not task_ids:
                return {'prediction': 0.0, 'confidence': 0.0, 'models_used': []}
            
            # 等待所有任务完成
            predictions = []
            confidences = []
            weights = []
            used_models = []
            
            timeout = 30  # 30秒超时
            start_time = time.time()
            
            while len(predictions) < len(task_ids) and (time.time() - start_time) < timeout:
                for task_id in task_ids:
                    if task_id in self.completed_tasks:
                        task = self.completed_tasks[task_id]
                        if task.result and task_id not in [t['task_id'] for t in predictions]:
                            model_info = self.ai_models[task.model_id]
                            
                            predictions.append({
                                'task_id': task_id,
                                'model_id': task.model_id,
                                'prediction': task.result.get('prediction', 0.0),
                                'confidence': task.result.get('confidence', 0.0)
                            })
                            
                            confidences.append(task.result.get('confidence', 0.0))
                            weights.append(model_info.performance_score)
                            used_models.append(task.model_id)
                
                await asyncio.sleep(0.1)
            
            if not predictions:
                return {'prediction': 0.0, 'confidence': 0.0, 'models_used': []}
            
            # 加权平均集成
            pred_values = [p['prediction'] for p in predictions]
            conf_values = confidences
            
            # 如果没有权重，使用等权重
            if not any(weights):
                weights = [1.0] * len(predictions)
            
            # 标准化权重
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(predictions)] * len(predictions)
            
            # 计算加权预测
            ensemble_prediction = sum(p * w for p, w in zip(pred_values, weights))
            ensemble_confidence = sum(c * w for c, w in zip(conf_values, weights))
            
            result = {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'models_used': used_models,
                'individual_predictions': predictions,
                'weights': dict(zip(used_models, weights))
            }
            
            logger.info(f"🧠 集成预测完成: {len(used_models)}个模型, 预测值: {ensemble_prediction:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 集成预测失败: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'models_used': []}
    
    async def start(self):
        """启动调度器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动调度器任务
        self.scheduler_tasks = [
            asyncio.create_task(self._task_scheduler()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._health_checker())
        ]
        
        logger.success("🚀 AI模型调度中心已启动")
    
    async def stop(self):
        """停止调度器"""
        self.is_running = False
        
        # 取消所有任务
        for task in self.scheduler_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.scheduler_tasks, return_exceptions=True)
        
        # 释放所有资源
        for model_id in list(self.ai_models.keys()):
            self.resource_manager.release_resources(model_id)
        
        self.executor.shutdown(wait=True)
        
        logger.info("🛑 AI模型调度中心已停止")
    
    async def _task_scheduler(self):
        """任务调度器主循环"""
        while self.is_running:
            try:
                # 获取任务（带超时）
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # 检查模型状态
                if task.model_id not in self.ai_models:
                    logger.error(f"❌ 任务模型不存在: {task.model_id}")
                    continue
                
                model_info = self.ai_models[task.model_id]
                if model_info.status != AIModelStatus.READY:
                    logger.warning(f"⚠️ 模型状态不可用: {task.model_id} - {model_info.status}")
                    continue
                
                # 执行任务
                task.scheduled_at = datetime.now(timezone.utc)
                self.running_tasks[task.task_id] = task
                
                # 异步执行任务
                asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                logger.error(f"❌ 任务调度异常: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: SchedulingTask):
        """执行任务"""
        start_time = time.time()
        
        try:
            model_info = self.ai_models[task.model_id]
            model_instance = self.model_instances[task.model_id]
            
            # 更新模型状态
            if task.task_type == 'predict':
                model_info.status = AIModelStatus.PREDICTING
            elif task.task_type == 'train':
                model_info.status = AIModelStatus.TRAINING
            
            # 执行任务
            result = None
            
            if task.task_type == 'predict':
                if hasattr(model_instance, 'predict'):
                    result = await model_instance.predict(task.data)
                elif hasattr(model_instance, 'get_prediction'):
                    result = await model_instance.get_prediction(task.data)
                else:
                    raise ValueError(f"模型 {task.model_id} 没有预测方法")
                    
            elif task.task_type == 'train':
                if hasattr(model_instance, 'train'):
                    result = await model_instance.train(task.data)
                elif hasattr(model_instance, 'update_model'):
                    result = await model_instance.update_model(task.data)
                else:
                    raise ValueError(f"模型 {task.model_id} 没有训练方法")
            
            # 任务完成
            execution_time = (time.time() - start_time) * 1000  # 毫秒
            
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            
            # 更新模型信息
            model_info.status = AIModelStatus.READY
            model_info.last_prediction = result
            model_info.last_update = datetime.now(timezone.utc)
            model_info.latency_ms = execution_time
            model_info.prediction_count += 1
            
            # 移动任务
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            # 更新统计
            self.stats['completed_tasks'] += 1
            self.stats['total_predictions'] += 1
            
            # 更新平均延迟
            if self.stats['completed_tasks'] > 0:
                self.stats['average_latency'] = (
                    (self.stats['average_latency'] * (self.stats['completed_tasks'] - 1) + execution_time) /
                    self.stats['completed_tasks']
                )
            
            logger.debug(f"✅ 任务执行完成: {task.task_id} 用时 {execution_time:.2f}ms")
            
        except Exception as e:
            # 任务失败
            execution_time = (time.time() - start_time) * 1000
            
            task.error_message = str(e)
            task.completed_at = datetime.now(timezone.utc)
            
            # 更新模型信息
            model_info = self.ai_models[task.model_id]
            model_info.status = AIModelStatus.ERROR
            model_info.error_count += 1
            model_info.latency_ms = execution_time
            
            # 移动任务
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            # 更新统计
            self.stats['failed_tasks'] += 1
            
            logger.error(f"❌ 任务执行失败: {task.task_id} - {e}")
    
    async def _performance_monitor(self):
        """性能监控"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 更新模型性能分数
                for model_id, model_info in self.ai_models.items():
                    if model_info.prediction_count > 0:
                        # 简单的性能评分：基于准确率和延迟
                        accuracy_score = model_info.accuracy
                        latency_score = max(0, 1 - model_info.latency_ms / 1000)  # 延迟越低分数越高
                        error_penalty = max(0, 1 - model_info.error_count / max(model_info.prediction_count, 1))
                        
                        model_info.performance_score = (accuracy_score * 0.5 + 
                                                       latency_score * 0.3 + 
                                                       error_penalty * 0.2)
                
                # 清理旧的完成任务
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                old_tasks = [tid for tid, task in self.completed_tasks.items() 
                           if task.completed_at and task.completed_at < cutoff_time]
                
                for task_id in old_tasks:
                    del self.completed_tasks[task_id]
                
                logger.debug(f"📊 性能监控完成，清理了 {len(old_tasks)} 个旧任务")
                
            except Exception as e:
                logger.error(f"❌ 性能监控异常: {e}")
    
    async def _health_checker(self):
        """健康检查"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                # 检查模型健康状态
                for model_id, model_info in self.ai_models.items():
                    # 如果模型长时间没有更新，标记为离线
                    if model_info.last_update:
                        time_since_update = datetime.now(timezone.utc) - model_info.last_update
                        if time_since_update > timedelta(minutes=5):
                            if model_info.status != AIModelStatus.OFFLINE:
                                model_info.status = AIModelStatus.OFFLINE
                                logger.warning(f"⚠️ 模型离线: {model_id}")
                    
                    # 如果错误率过高，暂时禁用模型
                    if model_info.prediction_count > 10:
                        error_rate = model_info.error_count / model_info.prediction_count
                        if error_rate > 0.5:  # 错误率超过50%
                            model_info.is_active = False
                            logger.warning(f"⚠️ 模型错误率过高，已禁用: {model_id}")
                
                # 更新活跃模型数量
                self.stats['active_models'] = len([m for m in self.ai_models.values() if m.is_active])
                
            except Exception as e:
                logger.error(f"❌ 健康检查异常: {e}")
    
    def get_model_status(self, model_id: str = None) -> Dict[str, Any]:
        """获取模型状态"""
        if model_id:
            if model_id in self.ai_models:
                model_info = self.ai_models[model_id]
                return {
                    'model_id': model_info.model_id,
                    'model_name': model_info.model_name,
                    'status': model_info.status.value,
                    'performance_score': model_info.performance_score,
                    'accuracy': model_info.accuracy,
                    'latency_ms': model_info.latency_ms,
                    'prediction_count': model_info.prediction_count,
                    'error_count': model_info.error_count,
                    'is_active': model_info.is_active,
                    'last_update': model_info.last_update.isoformat()
                }
            else:
                return {}
        else:
            # 返回所有模型状态
            return {
                mid: {
                    'model_id': info.model_id,
                    'model_name': info.model_name,
                    'status': info.status.value,
                    'performance_score': info.performance_score,
                    'accuracy': info.accuracy,
                    'latency_ms': info.latency_ms,
                    'prediction_count': info.prediction_count,
                    'error_count': info.error_count,
                    'is_active': info.is_active,
                    'last_update': info.last_update.isoformat()
                }
                for mid, info in self.ai_models.items()
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        resource_usage = self.resource_manager.get_resource_usage()
        
        return {
            'scheduler_stats': self.stats,
            'resource_usage': resource_usage,
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'registered_models': len(self.ai_models),
            'active_models': self.stats['active_models']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            return {
                'status': 'healthy' if self.is_running else 'stopped',
                'scheduler_running': self.is_running,
                'active_models': self.stats['active_models'],
                'queue_size': self.task_queue.qsize(),
                'average_latency': f"{self.stats['average_latency']:.2f}ms",
                'success_rate': (
                    (self.stats['completed_tasks'] / max(self.stats['total_tasks'], 1)) * 100
                    if self.stats['total_tasks'] > 0 else 0
                )
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

