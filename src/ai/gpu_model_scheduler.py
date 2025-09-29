#!/usr/bin/env python3
"""
GPU模型调度器 - 生产级多AI模型并行训练调度管理
实现6个AI模型智能调度、负载均衡、资源优化
"""
import asyncio
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import queue
import numpy as np
from loguru import logger
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import GPUtil

class ModelTask:
    """模型任务类"""
    
    def __init__(self, task_id: str, model_type: str, priority: int = 1, 
                 config: Dict = None, callback: Callable = None):
        self.task_id = task_id
        self.model_type = model_type
        self.priority = priority
        self.config = config or {}
        self.callback = callback
        self.status = 'pending'
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.device_id = None
        self.memory_allocated = 0
        
    def __lt__(self, other):
        # 优先级队列排序（优先级高的先执行）
        return self.priority > other.priority

class ProductionGPUModelScheduler:
    """生产级GPU模型调度器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_cores = psutil.cpu_count()
        
        # 任务队列
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # 资源管理
        self.device_status = {}
        self.cpu_slots = min(6, self.cpu_cores // 2)  # CPU训练槽位
        self.gpu_slots = min(2, self.device_count) if self.gpu_available else 0  # GPU训练槽位
        self.used_cpu_slots = 0
        self.used_gpu_slots = 0
        
        # 调度统计
        self.scheduler_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'gpu_utilization': 0.0,
            'cpu_utilization': 0.0
        }
        
        # 线程池
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.cpu_slots)
        self.gpu_executor = ThreadPoolExecutor(max_workers=self.gpu_slots)
        
        # 调度控制
        self.is_running = False
        self.scheduler_thread = None
        self.monitor_thread = None
        
        # AI模型类型配置
        self.model_configs = {
            'reinforcement_learning': {
                'memory_requirement_mb': 2048,
                'preferred_device': 'gpu',
                'training_time_estimate': 300,
                'priority_base': 5
            },
            'deep_learning': {
                'memory_requirement_mb': 1536,
                'preferred_device': 'gpu',
                'training_time_estimate': 240,
                'priority_base': 4
            },
            'ensemble_learning': {
                'memory_requirement_mb': 512,
                'preferred_device': 'cpu',
                'training_time_estimate': 180,
                'priority_base': 3
            },
            'expert_system': {
                'memory_requirement_mb': 256,
                'preferred_device': 'cpu',
                'training_time_estimate': 120,
                'priority_base': 2
            },
            'meta_learning': {
                'memory_requirement_mb': 1024,
                'preferred_device': 'gpu',
                'training_time_estimate': 360,
                'priority_base': 6
            },
            'transfer_learning': {
                'memory_requirement_mb': 768,
                'preferred_device': 'gpu',
                'training_time_estimate': 200,
                'priority_base': 4
            }
        }
        
        self._initialize_device_status()
        logger.info(f"🎯 GPU模型调度器初始化完成 - CPU槽位: {self.cpu_slots}, GPU槽位: {self.gpu_slots}")
    
    def _initialize_device_status(self):
        """初始化设备状态"""
        try:
            # 初始化CPU状态
            self.device_status['cpu'] = {
                'available_slots': self.cpu_slots,
                'used_slots': 0,
                'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                'available_memory_gb': psutil.virtual_memory().available / 1024**3,
                'utilization': 0.0,
                'temperature': 0.0,
                'running_tasks': []
            }
            
            # 初始化GPU状态
            if self.gpu_available:
                for device_id in range(self.device_count):
                    gpu_props = torch.cuda.get_device_properties(device_id)
                    self.device_status[f'gpu_{device_id}'] = {
                        'device_id': device_id,
                        'name': gpu_props.name,
                        'total_memory_gb': gpu_props.total_memory / 1024**3,
                        'available_memory_gb': gpu_props.total_memory / 1024**3,
                        'utilization': 0.0,
                        'temperature': 0.0,
                        'running_tasks': [],
                        'max_concurrent_tasks': 1  # RTX3060建议单任务
                    }
            
        except Exception as e:
            logger.error(f"设备状态初始化错误: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        if self.is_running:
            logger.warning("模型调度器已在运行中")
            return
        
        self.is_running = True
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 GPU模型调度器启动")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                # 检查是否有待执行任务
                if not self.task_queue.empty():
                    # 尝试调度任务
                    self._schedule_next_task()
                
                # 清理完成的任务
                self._cleanup_completed_tasks()
                
                # 更新调度统计
                self._update_scheduler_stats()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                time.sleep(5)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 更新设备状态
                self._update_device_status()
                
                # 检查任务超时
                self._check_task_timeouts()
                
                # 记录监控信息
                self._log_scheduler_status()
                
                time.sleep(10)  # 每10秒监控一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(15)
    
    def _schedule_next_task(self):
        """调度下一个任务"""
        try:
            # 获取优先级最高的任务
            priority, task = self.task_queue.get_nowait()
            
            # 选择最佳设备
            device_type, device_id = self._select_best_device(task)
            
            if device_type is None:
                # 没有可用设备，重新放回队列
                self.task_queue.put((priority, task))
                return
            
            # 分配资源并执行任务
            self._execute_task(task, device_type, device_id)
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"任务调度错误: {e}")
    
    def _select_best_device(self, task: ModelTask) -> Tuple[Optional[str], Optional[int]]:
        """选择最佳设备"""
        try:
            model_config = self.model_configs.get(task.model_type, {})
            preferred_device = model_config.get('preferred_device', 'cpu')
            memory_requirement = model_config.get('memory_requirement_mb', 512)
            
            # 优先选择推荐设备类型
            if preferred_device == 'gpu' and self.gpu_available:
                # 查找可用GPU
                for device_id in range(self.device_count):
                    device_key = f'gpu_{device_id}'
                    device_status = self.device_status[device_key]
                    
                    # 检查GPU是否可用
                    if (len(device_status['running_tasks']) < device_status['max_concurrent_tasks'] and
                        device_status['available_memory_gb'] * 1024 > memory_requirement):
                        return 'gpu', device_id
            
            # 检查CPU是否可用
            cpu_status = self.device_status['cpu']
            if (cpu_status['used_slots'] < cpu_status['available_slots'] and
                cpu_status['available_memory_gb'] * 1024 > memory_requirement):
                return 'cpu', None
            
            return None, None
            
        except Exception as e:
            logger.error(f"设备选择错误: {e}")
            return None, None
    
    def _execute_task(self, task: ModelTask, device_type: str, device_id: Optional[int]):
        """执行任务"""
        try:
            task.status = 'running'
            task.start_time = time.time()
            task.device_id = device_id
            
            # 更新设备状态
            if device_type == 'gpu':
                device_key = f'gpu_{device_id}'
                self.device_status[device_key]['running_tasks'].append(task.task_id)
                self.used_gpu_slots += 1
            else:
                self.device_status['cpu']['used_slots'] += 1
                self.device_status['cpu']['running_tasks'].append(task.task_id)
                self.used_cpu_slots += 1
            
            # 添加到运行任务列表
            self.running_tasks[task.task_id] = task
            
            # 选择执行器
            executor = self.gpu_executor if device_type == 'gpu' else self.cpu_executor
            
            # 提交任务执行
            future = executor.submit(self._run_model_training, task, device_type, device_id)
            
            # 设置完成回调
            future.add_done_callback(lambda f: self._task_completed(task, f))
            
            logger.info(f"🎯 任务开始执行 - {task.task_id} ({task.model_type}) on {device_type}_{device_id or 'cpu'}")
            
        except Exception as e:
            logger.error(f"任务执行错误: {e}")
            self._task_failed(task, str(e))
    
    def _run_model_training(self, task: ModelTask, device_type: str, device_id: Optional[int]) -> Any:
        """运行模型训练"""
        try:
            model_config = self.model_configs.get(task.model_type, {})
            training_time = model_config.get('training_time_estimate', 120)
            
            # 设置设备
            if device_type == 'gpu' and device_id is not None:
                torch.cuda.set_device(device_id)
                device = torch.device(f'cuda:{device_id}')
            else:
                device = torch.device('cpu')
            
            logger.info(f"🧠 开始训练 {task.model_type} 模型 - 设备: {device}")
            
            # 模拟训练过程（实际实现中会调用具体的AI模型训练代码）
            result = self._simulate_model_training(task, device, training_time)
            
            logger.success(f"✅ 模型训练完成 - {task.task_id}")
            return result
            
        except Exception as e:
            logger.error(f"模型训练错误: {e}")
            raise e
    
    def _simulate_model_training(self, task: ModelTask, device: torch.device, training_time: int) -> Dict[str, Any]:
        """模拟模型训练（实际实现中替换为真实训练代码）"""
        try:
            # 模拟训练过程
            start_time = time.time()
            
            # 分阶段模拟训练
            phases = ['数据加载', '模型初始化', '训练循环', '验证评估', '模型保存']
            phase_time = training_time / len(phases)
            
            results = {
                'model_type': task.model_type,
                'training_phases': [],
                'final_metrics': {},
                'device_used': str(device)
            }
            
            for i, phase in enumerate(phases):
                phase_start = time.time()
                
                # 模拟该阶段的工作
                time.sleep(min(phase_time, 30))  # 最多30秒一个阶段
                
                phase_end = time.time()
                phase_result = {
                    'phase': phase,
                    'duration': phase_end - phase_start,
                    'progress': (i + 1) / len(phases)
                }
                results['training_phases'].append(phase_result)
                
                logger.info(f"📊 {task.task_id} - {phase} 完成 ({(i+1)/len(phases)*100:.1f}%)")
            
            # 模拟最终指标
            results['final_metrics'] = {
                'accuracy': np.random.uniform(0.7, 0.95),
                'loss': np.random.uniform(0.05, 0.3),
                'training_time': time.time() - start_time,
                'convergence_epoch': np.random.randint(10, 100)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"模型训练模拟错误: {e}")
            raise e
    
    def _task_completed(self, task: ModelTask, future):
        """任务完成回调"""
        try:
            task.end_time = time.time()
            
            if future.exception():
                # 任务失败
                task.status = 'failed'
                task.error = str(future.exception())
                self.failed_tasks[task.task_id] = task
                logger.error(f"❌ 任务失败 - {task.task_id}: {task.error}")
            else:
                # 任务成功
                task.status = 'completed'
                task.result = future.result()
                self.completed_tasks[task.task_id] = task
                logger.success(f"✅ 任务完成 - {task.task_id} (耗时: {task.end_time - task.start_time:.1f}s)")
                
                # 执行回调
                if task.callback:
                    try:
                        task.callback(task)
                    except Exception as e:
                        logger.error(f"任务回调错误: {e}")
            
            # 释放资源
            self._release_task_resources(task)
            
            # 从运行任务中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
        except Exception as e:
            logger.error(f"任务完成处理错误: {e}")
    
    def _task_failed(self, task: ModelTask, error_msg: str):
        """任务失败处理"""
        task.status = 'failed'
        task.error = error_msg
        task.end_time = time.time()
        self.failed_tasks[task.task_id] = task
        
        # 释放资源
        self._release_task_resources(task)
        
        # 从运行任务中移除
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        
        logger.error(f"❌ 任务失败 - {task.task_id}: {error_msg}")
    
    def _release_task_resources(self, task: ModelTask):
        """释放任务资源"""
        try:
            if task.device_id is not None:
                # GPU任务
                device_key = f'gpu_{task.device_id}'
                if task.task_id in self.device_status[device_key]['running_tasks']:
                    self.device_status[device_key]['running_tasks'].remove(task.task_id)
                self.used_gpu_slots = max(0, self.used_gpu_slots - 1)
            else:
                # CPU任务
                if task.task_id in self.device_status['cpu']['running_tasks']:
                    self.device_status['cpu']['running_tasks'].remove(task.task_id)
                self.device_status['cpu']['used_slots'] = max(0, self.device_status['cpu']['used_slots'] - 1)
                self.used_cpu_slots = max(0, self.used_cpu_slots - 1)
            
        except Exception as e:
            logger.error(f"资源释放错误: {e}")
    
    def _update_device_status(self):
        """更新设备状态"""
        try:
            # 更新CPU状态
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.device_status['cpu'].update({
                'utilization': cpu_percent,
                'available_memory_gb': memory.available / 1024**3
            })
            
            # 更新GPU状态
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    for device_id in range(min(len(gpus), self.device_count)):
                        gpu = gpus[device_id]
                        device_key = f'gpu_{device_id}'
                        
                        self.device_status[device_key].update({
                            'utilization': gpu.load * 100,
                            'temperature': gpu.temperature,
                            'available_memory_gb': (gpu.memoryTotal - gpu.memoryUsed) / 1024
                        })
                except:
                    pass
            
        except Exception as e:
            logger.error(f"设备状态更新错误: {e}")
    
    def _check_task_timeouts(self):
        """检查任务超时"""
        try:
            current_time = time.time()
            timeout_threshold = 1800  # 30分钟超时
            
            for task_id, task in list(self.running_tasks.items()):
                if task.start_time and (current_time - task.start_time) > timeout_threshold:
                    logger.warning(f"⏰ 任务超时 - {task_id}")
                    self._task_failed(task, "任务执行超时")
                    
        except Exception as e:
            logger.error(f"超时检查错误: {e}")
    
    def _cleanup_completed_tasks(self):
        """清理完成的任务"""
        try:
            # 保持完成任务历史在合理范围内
            if len(self.completed_tasks) > 100:
                # 移除最旧的任务
                oldest_tasks = sorted(self.completed_tasks.items(), 
                                    key=lambda x: x[1].end_time or 0)
                for task_id, _ in oldest_tasks[:50]:
                    del self.completed_tasks[task_id]
            
            # 清理失败任务
            if len(self.failed_tasks) > 50:
                oldest_failed = sorted(self.failed_tasks.items(), 
                                     key=lambda x: x[1].end_time or 0)
                for task_id, _ in oldest_failed[:25]:
                    del self.failed_tasks[task_id]
                    
        except Exception as e:
            logger.error(f"任务清理错误: {e}")
    
    def _update_scheduler_stats(self):
        """更新调度统计"""
        try:
            total_completed = len(self.completed_tasks)
            total_failed = len(self.failed_tasks)
            
            self.scheduler_stats.update({
                'total_tasks': total_completed + total_failed + len(self.running_tasks),
                'completed_tasks': total_completed,
                'failed_tasks': total_failed,
                'running_tasks': len(self.running_tasks),
                'pending_tasks': self.task_queue.qsize()
            })
            
            # 计算平均执行时间
            if self.completed_tasks:
                total_time = sum((task.end_time - task.start_time) 
                               for task in self.completed_tasks.values() 
                               if task.start_time and task.end_time)
                self.scheduler_stats['average_execution_time'] = total_time / len(self.completed_tasks)
            
            # 计算资源利用率
            if self.gpu_available:
                gpu_utilization = sum(self.device_status[f'gpu_{i}']['utilization'] 
                                    for i in range(self.device_count)) / self.device_count
                self.scheduler_stats['gpu_utilization'] = gpu_utilization
            
            self.scheduler_stats['cpu_utilization'] = self.device_status['cpu']['utilization']
            
        except Exception as e:
            logger.error(f"统计更新错误: {e}")
    
    def _log_scheduler_status(self):
        """记录调度器状态"""
        try:
            if int(time.time()) % 60 == 0:  # 每分钟记录一次
                stats = self.scheduler_stats
                logger.info(f"🎯 调度器状态 - 运行: {stats['running_tasks']}, "
                          f"完成: {stats['completed_tasks']}, "
                          f"失败: {stats['failed_tasks']}, "
                          f"等待: {stats['pending_tasks']}")
                
        except Exception as e:
            logger.error(f"状态记录错误: {e}")
    
    def submit_task(self, model_type: str, priority: int = 1, 
                   config: Dict = None, callback: Callable = None) -> str:
        """提交训练任务"""
        try:
            task_id = f"{model_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"
            task = ModelTask(task_id, model_type, priority, config, callback)
            
            # 添加到任务队列
            self.task_queue.put((priority, task))
            
            logger.info(f"📝 任务提交成功 - {task_id} ({model_type}), 优先级: {priority}")
            return task_id
            
        except Exception as e:
            logger.error(f"任务提交错误: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        try:
            # 检查运行中任务
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'model_type': task.model_type,
                    'device_id': task.device_id,
                    'start_time': task.start_time,
                    'running_time': time.time() - task.start_time if task.start_time else 0
                }
            
            # 检查完成任务
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'model_type': task.model_type,
                    'result': task.result,
                    'execution_time': task.end_time - task.start_time if task.start_time and task.end_time else 0
                }
            
            # 检查失败任务
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'model_type': task.model_type,
                    'error': task.error,
                    'execution_time': task.end_time - task.start_time if task.start_time and task.end_time else 0
                }
            
            return None
            
        except Exception as e:
            logger.error(f"任务状态查询错误: {e}")
            return None
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            'scheduler_stats': self.scheduler_stats.copy(),
            'device_status': self.device_status.copy(),
            'resource_usage': {
                'cpu_slots_used': self.used_cpu_slots,
                'cpu_slots_total': self.cpu_slots,
                'gpu_slots_used': self.used_gpu_slots,
                'gpu_slots_total': self.gpu_slots
            },
            'queue_status': {
                'pending_tasks': self.task_queue.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            }
        }
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        
        # 等待线程结束
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        # 关闭线程池
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)
        
        logger.info("🛑 GPU模型调度器已停止")

# 全局模型调度器实例
_gpu_model_scheduler = None

def initialize_gpu_model_scheduler() -> ProductionGPUModelScheduler:
    """初始化GPU模型调度器"""
    global _gpu_model_scheduler
    
    if _gpu_model_scheduler is None:
        _gpu_model_scheduler = ProductionGPUModelScheduler()
        _gpu_model_scheduler.start_scheduler()
        logger.success("✅ GPU模型调度器初始化完成")
    
    return _gpu_model_scheduler

def get_gpu_model_scheduler() -> Optional[ProductionGPUModelScheduler]:
    """获取GPU模型调度器实例"""
    return _gpu_model_scheduler

if __name__ == "__main__":
    # 测试GPU模型调度器
    scheduler = initialize_gpu_model_scheduler()
    
    # 提交测试任务
    task_ids = []
    for model_type in ['reinforcement_learning', 'deep_learning', 'ensemble_learning']:
        task_id = scheduler.submit_task(model_type, priority=5)
        task_ids.append(task_id)
    
    # 监控任务状态
    for i in range(30):
        status = scheduler.get_scheduler_status()
        print(f"调度器状态: {status['queue_status']}")
        time.sleep(10)
    
    scheduler.stop_scheduler()
