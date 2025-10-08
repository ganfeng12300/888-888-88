#!/usr/bin/env python3
"""
🚀 GPU加速器 - 生产级GPU计算优化
GPU Accelerator - Production-Grade GPU Computing Optimization

生产级特性：
- CUDA/OpenCL GPU计算加速
- 内存池管理和优化
- 批处理和并行计算
- GPU资源监控
- 自动回退CPU计算
"""

import numpy as np
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import deque
import logging
import json

# GPU计算库导入
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

try:
    import numba
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    cuda = None
    jit = None

from ..monitoring.unified_logging_system import UnifiedLogger

@dataclass
class GPUInfo:
    """GPU信息数据结构"""
    device_id: int
    name: str
    memory_total: int
    memory_free: int
    memory_used: int
    compute_capability: str
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dim: Tuple[int, int, int]
    max_grid_dim: Tuple[int, int, int]

@dataclass
class ComputeTask:
    """计算任务数据结构"""
    task_id: str
    function_name: str
    data: Any
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

class GPUMemoryPool:
    """GPU内存池管理器"""
    
    def __init__(self, device_id: int = 0):
        self.logger = UnifiedLogger("GPUMemoryPool")
        self.device_id = device_id
        self.allocated_blocks = {}
        self.free_blocks = {}
        self.total_allocated = 0
        self.peak_usage = 0
        self._lock = threading.Lock()
        
        if CUPY_AVAILABLE:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
    
    def allocate(self, size: int, dtype=np.float32) -> Optional[Any]:
        """分配GPU内存"""
        try:
            with self._lock:
                if CUPY_AVAILABLE:
                    # 使用CuPy内存池
                    array = cp.empty(size, dtype=dtype)
                    block_id = id(array)
                    self.allocated_blocks[block_id] = {
                        'array': array,
                        'size': size,
                        'dtype': dtype,
                        'allocated_at': datetime.now()
                    }
                    self.total_allocated += array.nbytes
                    self.peak_usage = max(self.peak_usage, self.total_allocated)
                    return array
                else:
                    # 回退到CPU内存
                    array = np.empty(size, dtype=dtype)
                    return array
                    
        except Exception as e:
            self.logger.error(f"GPU内存分配失败: {e}")
            # 回退到CPU内存
            return np.empty(size, dtype=dtype)
    
    def deallocate(self, array: Any):
        """释放GPU内存"""
        try:
            with self._lock:
                if CUPY_AVAILABLE and hasattr(array, '__array_interface__'):
                    block_id = id(array)
                    if block_id in self.allocated_blocks:
                        block_info = self.allocated_blocks[block_id]
                        self.total_allocated -= block_info['array'].nbytes
                        del self.allocated_blocks[block_id]
                        del array
                        
        except Exception as e:
            self.logger.error(f"GPU内存释放失败: {e}")
    
    def get_memory_info(self) -> Dict:
        """获取内存使用信息"""
        try:
            if CUPY_AVAILABLE:
                mempool_info = self.mempool.get_limit()
                used_bytes = self.mempool.used_bytes()
                total_bytes = self.mempool.total_bytes()
                
                return {
                    'device_id': self.device_id,
                    'total_allocated': self.total_allocated,
                    'peak_usage': self.peak_usage,
                    'active_blocks': len(self.allocated_blocks),
                    'mempool_used': used_bytes,
                    'mempool_total': total_bytes,
                    'mempool_limit': mempool_info
                }
            else:
                return {
                    'device_id': self.device_id,
                    'gpu_available': False,
                    'using_cpu_fallback': True
                }
                
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
            return {}
    
    def cleanup(self):
        """清理内存池"""
        try:
            with self._lock:
                if CUPY_AVAILABLE:
                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()
                
                self.allocated_blocks.clear()
                self.total_allocated = 0
                
            self.logger.info("GPU内存池已清理")
            
        except Exception as e:
            self.logger.error(f"清理内存池失败: {e}")

class CUDAAccelerator:
    """CUDA加速器"""
    
    def __init__(self, device_id: int = 0):
        self.logger = UnifiedLogger("CUDAAccelerator")
        self.device_id = device_id
        self.available = CUPY_AVAILABLE
        self.memory_pool = GPUMemoryPool(device_id) if self.available else None
        
        if self.available:
            try:
                cp.cuda.Device(device_id).use()
                self.logger.info(f"CUDA设备 {device_id} 初始化成功")
            except Exception as e:
                self.logger.error(f"CUDA设备初始化失败: {e}")
                self.available = False
    
    def get_device_info(self) -> Optional[GPUInfo]:
        """获取GPU设备信息"""
        if not self.available:
            return None
        
        try:
            device = cp.cuda.Device(self.device_id)
            attrs = device.attributes
            
            return GPUInfo(
                device_id=self.device_id,
                name=device.name.decode('utf-8'),
                memory_total=device.mem_info[1],
                memory_free=device.mem_info[0],
                memory_used=device.mem_info[1] - device.mem_info[0],
                compute_capability=f"{attrs['ComputeCapabilityMajor']}.{attrs['ComputeCapabilityMinor']}",
                multiprocessor_count=attrs['MultiProcessorCount'],
                max_threads_per_block=attrs['MaxThreadsPerBlock'],
                max_block_dim=(
                    attrs['MaxBlockDimX'],
                    attrs['MaxBlockDimY'],
                    attrs['MaxBlockDimZ']
                ),
                max_grid_dim=(
                    attrs['MaxGridDimX'],
                    attrs['MaxGridDimY'],
                    attrs['MaxGridDimZ']
                )
            )
            
        except Exception as e:
            self.logger.error(f"获取GPU设备信息失败: {e}")
            return None
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU矩阵乘法"""
        try:
            if not self.available:
                return np.dot(a, b)
            
            # 转换到GPU
            gpu_a = cp.asarray(a)
            gpu_b = cp.asarray(b)
            
            # GPU矩阵乘法
            gpu_result = cp.dot(gpu_a, gpu_b)
            
            # 转换回CPU
            result = cp.asnumpy(gpu_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU矩阵乘法失败: {e}")
            # 回退到CPU计算
            return np.dot(a, b)
    
    def element_wise_operations(self, a: np.ndarray, b: np.ndarray, operation: str) -> np.ndarray:
        """GPU元素级运算"""
        try:
            if not self.available:
                return self._cpu_element_wise(a, b, operation)
            
            # 转换到GPU
            gpu_a = cp.asarray(a)
            gpu_b = cp.asarray(b)
            
            # GPU元素级运算
            if operation == 'add':
                gpu_result = gpu_a + gpu_b
            elif operation == 'subtract':
                gpu_result = gpu_a - gpu_b
            elif operation == 'multiply':
                gpu_result = gpu_a * gpu_b
            elif operation == 'divide':
                gpu_result = gpu_a / gpu_b
            elif operation == 'power':
                gpu_result = cp.power(gpu_a, gpu_b)
            else:
                raise ValueError(f"不支持的运算: {operation}")
            
            # 转换回CPU
            result = cp.asnumpy(gpu_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU元素级运算失败: {e}")
            # 回退到CPU计算
            return self._cpu_element_wise(a, b, operation)
    
    def _cpu_element_wise(self, a: np.ndarray, b: np.ndarray, operation: str) -> np.ndarray:
        """CPU元素级运算回退"""
        if operation == 'add':
            return a + b
        elif operation == 'subtract':
            return a - b
        elif operation == 'multiply':
            return a * b
        elif operation == 'divide':
            return a / b
        elif operation == 'power':
            return np.power(a, b)
        else:
            raise ValueError(f"不支持的运算: {operation}")
    
    def reduce_operations(self, data: np.ndarray, operation: str, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """GPU归约运算"""
        try:
            if not self.available:
                return self._cpu_reduce(data, operation, axis)
            
            # 转换到GPU
            gpu_data = cp.asarray(data)
            
            # GPU归约运算
            if operation == 'sum':
                gpu_result = cp.sum(gpu_data, axis=axis)
            elif operation == 'mean':
                gpu_result = cp.mean(gpu_data, axis=axis)
            elif operation == 'max':
                gpu_result = cp.max(gpu_data, axis=axis)
            elif operation == 'min':
                gpu_result = cp.min(gpu_data, axis=axis)
            elif operation == 'std':
                gpu_result = cp.std(gpu_data, axis=axis)
            else:
                raise ValueError(f"不支持的归约运算: {operation}")
            
            # 转换回CPU
            if hasattr(gpu_result, 'ndim') and gpu_result.ndim > 0:
                result = cp.asnumpy(gpu_result)
            else:
                result = float(gpu_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU归约运算失败: {e}")
            # 回退到CPU计算
            return self._cpu_reduce(data, operation, axis)
    
    def _cpu_reduce(self, data: np.ndarray, operation: str, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """CPU归约运算回退"""
        if operation == 'sum':
            return np.sum(data, axis=axis)
        elif operation == 'mean':
            return np.mean(data, axis=axis)
        elif operation == 'max':
            return np.max(data, axis=axis)
        elif operation == 'min':
            return np.min(data, axis=axis)
        elif operation == 'std':
            return np.std(data, axis=axis)
        else:
            raise ValueError(f"不支持的归约运算: {operation}")
    
    def fft_transform(self, data: np.ndarray, inverse: bool = False) -> np.ndarray:
        """GPU快速傅里叶变换"""
        try:
            if not self.available:
                return np.fft.ifft(data) if inverse else np.fft.fft(data)
            
            # 转换到GPU
            gpu_data = cp.asarray(data)
            
            # GPU FFT
            if inverse:
                gpu_result = cp.fft.ifft(gpu_data)
            else:
                gpu_result = cp.fft.fft(gpu_data)
            
            # 转换回CPU
            result = cp.asnumpy(gpu_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU FFT失败: {e}")
            # 回退到CPU计算
            return np.fft.ifft(data) if inverse else np.fft.fft(data)

class NumbaAccelerator:
    """Numba JIT加速器"""
    
    def __init__(self):
        self.logger = UnifiedLogger("NumbaAccelerator")
        self.available = NUMBA_AVAILABLE and cuda is not None
        self.compiled_functions = {}
        
        if self.available:
            try:
                # 检查CUDA设备
                cuda.detect()
                self.logger.info("Numba CUDA加速器初始化成功")
            except Exception as e:
                self.logger.error(f"Numba CUDA初始化失败: {e}")
                self.available = False
    
    def compile_function(self, func: callable, signature: str = None) -> callable:
        """编译函数为GPU内核"""
        try:
            if not self.available:
                return jit(nopython=True)(func) if NUMBA_AVAILABLE else func
            
            func_name = func.__name__
            if func_name not in self.compiled_functions:
                if signature:
                    compiled_func = cuda.jit(signature)(func)
                else:
                    compiled_func = cuda.jit(func)
                
                self.compiled_functions[func_name] = compiled_func
                self.logger.info(f"函数 {func_name} 编译为GPU内核成功")
            
            return self.compiled_functions[func_name]
            
        except Exception as e:
            self.logger.error(f"编译GPU内核失败: {e}")
            # 回退到CPU JIT
            return jit(nopython=True)(func) if NUMBA_AVAILABLE else func
    
    def parallel_compute(self, data: np.ndarray, func: callable, *args, **kwargs) -> np.ndarray:
        """并行计算"""
        try:
            if not self.available:
                return self._cpu_parallel_compute(data, func, *args, **kwargs)
            
            # 分配GPU内存
            gpu_data = cuda.to_device(data)
            gpu_result = cuda.device_array_like(data)
            
            # 配置线程块和网格
            threads_per_block = 256
            blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block
            
            # 编译并执行GPU内核
            compiled_func = self.compile_function(func)
            compiled_func[blocks_per_grid, threads_per_block](gpu_data, gpu_result, *args, **kwargs)
            
            # 同步并复制结果
            cuda.synchronize()
            result = gpu_result.copy_to_host()
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU并行计算失败: {e}")
            # 回退到CPU计算
            return self._cpu_parallel_compute(data, func, *args, **kwargs)
    
    def _cpu_parallel_compute(self, data: np.ndarray, func: callable, *args, **kwargs) -> np.ndarray:
        """CPU并行计算回退"""
        try:
            if NUMBA_AVAILABLE:
                # 使用Numba CPU并行
                compiled_func = jit(nopython=True, parallel=True)(func)
                return compiled_func(data, *args, **kwargs)
            else:
                # 普通计算
                return func(data, *args, **kwargs)
                
        except Exception as e:
            self.logger.error(f"CPU并行计算失败: {e}")
            return func(data, *args, **kwargs)

class GPUTaskScheduler:
    """GPU任务调度器"""
    
    def __init__(self, max_concurrent_tasks: int = 4):
        self.logger = UnifiedLogger("GPUTaskScheduler")
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = deque()
        self.running_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self._running = False
        self._scheduler_thread = None
        self._lock = threading.Lock()
        
        # 初始化加速器
        self.cuda_accelerator = CUDAAccelerator()
        self.numba_accelerator = NumbaAccelerator()
    
    def start(self):
        """启动任务调度器"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("GPU任务调度器已启动")
    
    def stop(self):
        """停止任务调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        self.logger.info("GPU任务调度器已停止")
    
    def submit_task(self, function_name: str, data: Any, parameters: Dict[str, Any] = None, priority: int = 1) -> str:
        """提交计算任务"""
        task = ComputeTask(
            task_id=f"{function_name}_{int(time.time() * 1000000)}",
            function_name=function_name,
            data=data,
            parameters=parameters or {},
            priority=priority,
            created_at=datetime.now()
        )
        
        with self._lock:
            # 按优先级插入任务队列
            inserted = False
            for i, existing_task in enumerate(self.task_queue):
                if task.priority > existing_task.priority:
                    self.task_queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self.task_queue.append(task)
        
        self.logger.info(f"任务已提交: {task.task_id}")
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[ComputeTask]:
        """获取任务结果"""
        # 检查运行中的任务
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        
        # 检查已完成的任务
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 清理已完成的任务
                self._cleanup_completed_tasks()
                
                # 检查是否有可用的执行槽位
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    with self._lock:
                        if self.task_queue:
                            task = self.task_queue.popleft()
                            self.running_tasks[task.task_id] = task
                            
                            # 在新线程中执行任务
                            task_thread = threading.Thread(
                                target=self._execute_task,
                                args=(task,),
                                daemon=True
                            )
                            task_thread.start()
                
                time.sleep(0.1)  # 短暂休眠
                
            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: ComputeTask):
        """执行计算任务"""
        try:
            task.started_at = datetime.now()
            
            # 根据函数名选择执行方法
            if task.function_name == 'matrix_multiply':
                result = self._execute_matrix_multiply(task)
            elif task.function_name == 'element_wise':
                result = self._execute_element_wise(task)
            elif task.function_name == 'reduce':
                result = self._execute_reduce(task)
            elif task.function_name == 'fft':
                result = self._execute_fft(task)
            elif task.function_name == 'parallel_compute':
                result = self._execute_parallel_compute(task)
            else:
                raise ValueError(f"不支持的函数: {task.function_name}")
            
            task.result = result
            task.completed_at = datetime.now()
            
            self.logger.info(f"任务完成: {task.task_id}")
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.now()
            self.logger.error(f"任务执行失败 {task.task_id}: {e}")
        
        finally:
            # 移动到已完成任务列表
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.completed_tasks.append(task)
    
    def _execute_matrix_multiply(self, task: ComputeTask) -> np.ndarray:
        """执行矩阵乘法任务"""
        data = task.data
        if isinstance(data, dict) and 'a' in data and 'b' in data:
            return self.cuda_accelerator.matrix_multiply(data['a'], data['b'])
        else:
            raise ValueError("矩阵乘法需要包含'a'和'b'的数据字典")
    
    def _execute_element_wise(self, task: ComputeTask) -> np.ndarray:
        """执行元素级运算任务"""
        data = task.data
        operation = task.parameters.get('operation', 'add')
        
        if isinstance(data, dict) and 'a' in data and 'b' in data:
            return self.cuda_accelerator.element_wise_operations(data['a'], data['b'], operation)
        else:
            raise ValueError("元素级运算需要包含'a'和'b'的数据字典")
    
    def _execute_reduce(self, task: ComputeTask) -> Union[np.ndarray, float]:
        """执行归约运算任务"""
        data = task.data
        operation = task.parameters.get('operation', 'sum')
        axis = task.parameters.get('axis', None)
        
        return self.cuda_accelerator.reduce_operations(data, operation, axis)
    
    def _execute_fft(self, task: ComputeTask) -> np.ndarray:
        """执行FFT任务"""
        data = task.data
        inverse = task.parameters.get('inverse', False)
        
        return self.cuda_accelerator.fft_transform(data, inverse)
    
    def _execute_parallel_compute(self, task: ComputeTask) -> np.ndarray:
        """执行并行计算任务"""
        data = task.data
        func = task.parameters.get('function')
        args = task.parameters.get('args', ())
        kwargs = task.parameters.get('kwargs', {})
        
        if func is None:
            raise ValueError("并行计算需要指定函数")
        
        return self.numba_accelerator.parallel_compute(data, func, *args, **kwargs)
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        # 保留最近1小时的任务
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self._lock:
            # 过滤已完成任务
            filtered_tasks = deque()
            for task in self.completed_tasks:
                if task.completed_at and task.completed_at >= cutoff_time:
                    filtered_tasks.append(task)
            
            self.completed_tasks = filtered_tasks
    
    def get_scheduler_status(self) -> Dict:
        """获取调度器状态"""
        return {
            'running': self._running,
            'queued_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'cuda_available': self.cuda_accelerator.available,
            'numba_available': self.numba_accelerator.available
        }

class GPUAccelerator:
    """GPU加速器主类"""
    
    def __init__(self, device_id: int = 0, max_concurrent_tasks: int = 4):
        self.logger = UnifiedLogger("GPUAccelerator")
        self.device_id = device_id
        
        # 初始化组件
        self.cuda_accelerator = CUDAAccelerator(device_id)
        self.numba_accelerator = NumbaAccelerator()
        self.task_scheduler = GPUTaskScheduler(max_concurrent_tasks)
        
        # 启动调度器
        self.task_scheduler.start()
        
        self.logger.info("GPU加速器初始化完成")
    
    def get_system_info(self) -> Dict:
        """获取GPU系统信息"""
        info = {
            'cuda_available': CUPY_AVAILABLE,
            'opencl_available': OPENCL_AVAILABLE,
            'numba_available': NUMBA_AVAILABLE,
            'device_id': self.device_id
        }
        
        # 获取GPU设备信息
        gpu_info = self.cuda_accelerator.get_device_info()
        if gpu_info:
            info['gpu_info'] = gpu_info.__dict__
        
        # 获取内存信息
        if self.cuda_accelerator.memory_pool:
            info['memory_info'] = self.cuda_accelerator.memory_pool.get_memory_info()
        
        # 获取调度器状态
        info['scheduler_status'] = self.task_scheduler.get_scheduler_status()
        
        return info
    
    def submit_computation(self, function_name: str, data: Any, parameters: Dict[str, Any] = None, priority: int = 1) -> str:
        """提交GPU计算任务"""
        return self.task_scheduler.submit_task(function_name, data, parameters, priority)
    
    def get_result(self, task_id: str) -> Optional[ComputeTask]:
        """获取计算结果"""
        return self.task_scheduler.get_task_result(task_id)
    
    def wait_for_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """等待计算结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.get_result(task_id)
            if task and task.completed_at:
                if task.error:
                    raise RuntimeError(f"任务执行失败: {task.error}")
                return task.result
            
            time.sleep(0.1)
        
        raise TimeoutError(f"任务 {task_id} 超时")
    
    def cleanup(self):
        """清理资源"""
        try:
            self.task_scheduler.stop()
            
            if self.cuda_accelerator.memory_pool:
                self.cuda_accelerator.memory_pool.cleanup()
            
            self.logger.info("GPU加速器资源已清理")
            
        except Exception as e:
            self.logger.error(f"清理GPU资源失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建GPU加速器
    gpu_accelerator = GPUAccelerator()
    
    try:
        # 获取系统信息
        system_info = gpu_accelerator.get_system_info()
        print("GPU系统信息:", json.dumps(system_info, indent=2, default=str, ensure_ascii=False))
        
        # 测试矩阵乘法
        a = np.random.rand(1000, 1000).astype(np.float32)
        b = np.random.rand(1000, 1000).astype(np.float32)
        
        task_id = gpu_accelerator.submit_computation(
            'matrix_multiply',
            {'a': a, 'b': b},
            priority=1
        )
        
        print(f"提交矩阵乘法任务: {task_id}")
        
        # 等待结果
        result = gpu_accelerator.wait_for_result(task_id, timeout=30)
        print(f"矩阵乘法结果形状: {result.shape}")
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        gpu_accelerator.cleanup()
