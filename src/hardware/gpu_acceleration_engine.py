"""
🔧 GPU加速计算引擎
生产级GPU加速计算系统，支持CUDA/OpenCL多平台加速
实现完整的GPU内存管理、多GPU并行计算、性能监控等功能
专为高频量化交易场景优化，提供微秒级计算响应
"""
import asyncio
import numpy as np
import cupy as cp
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
from collections import deque
import psutil
import GPUtil

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

@dataclass
class GPUDevice:
    """GPU设备信息"""
    device_id: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    max_threads_per_block: int
    is_available: bool = True
    utilization: float = 0.0
    temperature: float = 0.0

@dataclass
class ComputeTask:
    """计算任务"""
    task_id: str
    task_type: str  # 'matrix_multiply', 'convolution', 'fft', 'reduction'
    input_data: np.ndarray
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    device_id: Optional[int] = None

@dataclass
class ComputeResult:
    """计算结果"""
    task_id: str
    result_data: np.ndarray
    execution_time: float
    device_id: int
    memory_used: int
    completed_at: datetime = field(default_factory=datetime.now)

class CUDAKernelManager:
    """CUDA内核管理器"""
    
    def __init__(self):
        self.kernels = {}
        self._compile_kernels()
        
    def _compile_kernels(self):
        """编译CUDA内核"""
        if not CUDA_AVAILABLE:
            return
            
        # 矩阵乘法内核
        matrix_multiply_kernel = """
        __global__ void matrix_multiply(float *A, float *B, float *C, int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
        """
        
        # 卷积内核
        convolution_kernel = """
        __global__ void convolution_2d(float *input, float *kernel, float *output,
                                     int input_height, int input_width,
                                     int kernel_height, int kernel_width,
                                     int output_height, int output_width) {
            int out_row = blockIdx.y * blockDim.y + threadIdx.y;
            int out_col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (out_row < output_height && out_col < output_width) {
                float sum = 0.0f;
                for (int k_row = 0; k_row < kernel_height; k_row++) {
                    for (int k_col = 0; k_col < kernel_width; k_col++) {
                        int in_row = out_row + k_row;
                        int in_col = out_col + k_col;
                        if (in_row < input_height && in_col < input_width) {
                            sum += input[in_row * input_width + in_col] * 
                                   kernel[k_row * kernel_width + k_col];
                        }
                    }
                }
                output[out_row * output_width + out_col] = sum;
            }
        }
        """
        
        # 归约内核
        reduction_kernel = """
        __global__ void reduce_sum(float *input, float *output, int n) {
            extern __shared__ float sdata[];
            
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            sdata[tid] = (i < n) ? input[i] : 0;
            __syncthreads();
            
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) output[blockIdx.x] = sdata[0];
        }
        """
        
        try:
            self.kernels['matrix_multiply'] = SourceModule(matrix_multiply_kernel)
            self.kernels['convolution'] = SourceModule(convolution_kernel)
            self.kernels['reduction'] = SourceModule(reduction_kernel)
        except Exception as e:
            logging.error(f"CUDA内核编译失败: {e}")

class GPUAccelerationEngine:
    """GPU加速计算引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # GPU设备管理
        self.gpu_devices: List[GPUDevice] = []
        self.current_device = 0
        self.device_locks = {}
        
        # 计算任务管理
        self.task_queue = asyncio.Queue()
        self.result_cache = {}
        self.active_tasks = {}
        
        # 性能监控
        self.performance_metrics = deque(maxlen=10000)
        self.device_utilization = {}
        
        # 内核管理
        self.kernel_manager = CUDAKernelManager()
        
        # 内存管理
        self.memory_pools = {}
        self.memory_usage = {}
        
        self._initialize_devices()
        
    def _initialize_devices(self):
        """初始化GPU设备"""
        try:
            if CUDA_AVAILABLE:
                self._initialize_cuda_devices()
            elif OPENCL_AVAILABLE:
                self._initialize_opencl_devices()
            else:
                self.logger.warning("未检测到GPU加速支持，将使用CPU计算")
                
        except Exception as e:
            self.logger.error(f"GPU设备初始化失败: {e}")
            
    def _initialize_cuda_devices(self):
        """初始化CUDA设备"""
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                device = GPUDevice(
                    device_id=i,
                    name=gpu.name,
                    memory_total=int(gpu.memoryTotal),
                    memory_free=int(gpu.memoryFree),
                    compute_capability=(gpu.driver, 0),  # 简化版本
                    multiprocessor_count=0,  # 需要通过CUDA API获取
                    max_threads_per_block=1024,  # 默认值
                    utilization=gpu.load * 100,
                    temperature=gpu.temperature
                )
                self.gpu_devices.append(device)
                self.device_locks[i] = threading.Lock()
                self.memory_pools[i] = {}
                self.memory_usage[i] = 0
                
            self.logger.info(f"初始化了 {len(self.gpu_devices)} 个CUDA设备")
            
        except Exception as e:
            self.logger.error(f"CUDA设备初始化失败: {e}")
            
    def _initialize_opencl_devices(self):
        """初始化OpenCL设备"""
        try:
            platforms = cl.get_platforms()
            device_id = 0
            
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                for device in devices:
                    gpu_device = GPUDevice(
                        device_id=device_id,
                        name=device.name.strip(),
                        memory_total=device.global_mem_size // (1024 * 1024),
                        memory_free=device.global_mem_size // (1024 * 1024),
                        compute_capability=(1, 0),
                        multiprocessor_count=device.max_compute_units,
                        max_threads_per_block=device.max_work_group_size
                    )
                    self.gpu_devices.append(gpu_device)
                    self.device_locks[device_id] = threading.Lock()
                    self.memory_pools[device_id] = {}
                    self.memory_usage[device_id] = 0
                    device_id += 1
                    
            self.logger.info(f"初始化了 {len(self.gpu_devices)} 个OpenCL设备")
            
        except Exception as e:
            self.logger.error(f"OpenCL设备初始化失败: {e}")
            
    async def start(self):
        """启动GPU加速引擎"""
        self.is_running = True
        self.logger.info("🔧 GPU加速引擎启动")
        
        # 启动任务处理循环
        tasks = [
            asyncio.create_task(self._task_processing_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._memory_management_loop()),
            asyncio.create_task(self._device_health_monitoring_loop())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop(self):
        """停止GPU加速引擎"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("🔧 GPU加速引擎停止")
        
    async def _task_processing_loop(self):
        """任务处理循环"""
        while self.is_running:
            try:
                # 获取任务
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # 选择最优设备
                device_id = await self._select_optimal_device(task)
                
                if device_id is not None:
                    # 执行任务
                    result = await self._execute_task(task, device_id)
                    
                    # 缓存结果
                    self.result_cache[task.task_id] = result
                    
                    # 记录性能指标
                    await self._record_performance_metrics(task, result)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"任务处理错误: {e}")
                await asyncio.sleep(1)
                
    async def _select_optimal_device(self, task: ComputeTask) -> Optional[int]:
        """选择最优GPU设备"""
        if not self.gpu_devices:
            return None
            
        # 如果指定了设备，直接使用
        if task.device_id is not None and task.device_id < len(self.gpu_devices):
            return task.device_id
            
        # 根据设备负载和内存使用情况选择
        best_device = None
        best_score = float('inf')
        
        for device in self.gpu_devices:
            if not device.is_available:
                continue
                
            # 计算设备评分（负载 + 内存使用率）
            memory_usage_ratio = self.memory_usage[device.device_id] / device.memory_total
            score = device.utilization + memory_usage_ratio * 100
            
            if score < best_score:
                best_score = score
                best_device = device.device_id
                
        return best_device
        
    async def _execute_task(self, task: ComputeTask, device_id: int) -> ComputeResult:
        """执行计算任务"""
        start_time = time.time()
        
        try:
            with self.device_locks[device_id]:
                # 设置当前设备
                if CUDA_AVAILABLE:
                    cp.cuda.Device(device_id).use()
                    
                # 根据任务类型执行计算
                if task.task_type == 'matrix_multiply':
                    result_data = await self._execute_matrix_multiply(task, device_id)
                elif task.task_type == 'convolution':
                    result_data = await self._execute_convolution(task, device_id)
                elif task.task_type == 'fft':
                    result_data = await self._execute_fft(task, device_id)
                elif task.task_type == 'reduction':
                    result_data = await self._execute_reduction(task, device_id)
                else:
                    raise ValueError(f"不支持的任务类型: {task.task_type}")
                    
                execution_time = time.time() - start_time
                
                return ComputeResult(
                    task_id=task.task_id,
                    result_data=result_data,
                    execution_time=execution_time,
                    device_id=device_id,
                    memory_used=result_data.nbytes
                )
                
        except Exception as e:
            self.logger.error(f"任务执行失败 {task.task_id}: {e}")
            raise
            
    async def _execute_matrix_multiply(self, task: ComputeTask, device_id: int) -> np.ndarray:
        """执行矩阵乘法"""
        input_data = task.input_data
        
        if len(input_data) != 2:
            raise ValueError("矩阵乘法需要两个输入矩阵")
            
        A, B = input_data[0], input_data[1]
        
        if CUDA_AVAILABLE:
            # 使用CuPy进行GPU计算
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)
            C_gpu = cp.dot(A_gpu, B_gpu)
            result = cp.asnumpy(C_gpu)
        else:
            # 回退到CPU计算
            result = np.dot(A, B)
            
        return result
        
    async def _execute_convolution(self, task: ComputeTask, device_id: int) -> np.ndarray:
        """执行卷积计算"""
        input_data = task.input_data
        kernel = task.parameters.get('kernel')
        
        if kernel is None:
            raise ValueError("卷积计算需要提供kernel参数")
            
        if CUDA_AVAILABLE:
            # 使用CuPy进行GPU卷积
            input_gpu = cp.asarray(input_data)
            kernel_gpu = cp.asarray(kernel)
            
            # 简化的2D卷积实现
            result_gpu = cp.zeros((
                input_data.shape[0] - kernel.shape[0] + 1,
                input_data.shape[1] - kernel.shape[1] + 1
            ))
            
            for i in range(result_gpu.shape[0]):
                for j in range(result_gpu.shape[1]):
                    result_gpu[i, j] = cp.sum(
                        input_gpu[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel_gpu
                    )
                    
            result = cp.asnumpy(result_gpu)
        else:
            # 回退到CPU计算
            from scipy import ndimage
            result = ndimage.convolve(input_data, kernel)
            
        return result
        
    async def _execute_fft(self, task: ComputeTask, device_id: int) -> np.ndarray:
        """执行FFT计算"""
        input_data = task.input_data
        
        if CUDA_AVAILABLE:
            # 使用CuPy进行GPU FFT
            input_gpu = cp.asarray(input_data)
            result_gpu = cp.fft.fft(input_gpu)
            result = cp.asnumpy(result_gpu)
        else:
            # 回退到CPU计算
            result = np.fft.fft(input_data)
            
        return result
        
    async def _execute_reduction(self, task: ComputeTask, device_id: int) -> np.ndarray:
        """执行归约计算"""
        input_data = task.input_data
        operation = task.parameters.get('operation', 'sum')
        
        if CUDA_AVAILABLE:
            input_gpu = cp.asarray(input_data)
            
            if operation == 'sum':
                result_gpu = cp.sum(input_gpu)
            elif operation == 'mean':
                result_gpu = cp.mean(input_gpu)
            elif operation == 'max':
                result_gpu = cp.max(input_gpu)
            elif operation == 'min':
                result_gpu = cp.min(input_gpu)
            else:
                raise ValueError(f"不支持的归约操作: {operation}")
                
            result = cp.asnumpy(result_gpu)
        else:
            # 回退到CPU计算
            if operation == 'sum':
                result = np.sum(input_data)
            elif operation == 'mean':
                result = np.mean(input_data)
            elif operation == 'max':
                result = np.max(input_data)
            elif operation == 'min':
                result = np.min(input_data)
            else:
                raise ValueError(f"不支持的归约操作: {operation}")
                
        return np.array([result])
        
    async def _performance_monitoring_loop(self):
        """性能监控循环"""
        while self.is_running:
            try:
                # 更新设备状态
                await self._update_device_status()
                
                # 记录性能指标
                await self._collect_performance_metrics()
                
                await asyncio.sleep(5)  # 5秒更新一次
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(10)
                
    async def _update_device_status(self):
        """更新设备状态"""
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if i < len(self.gpu_devices):
                    device = self.gpu_devices[i]
                    device.memory_free = int(gpu.memoryFree)
                    device.utilization = gpu.load * 100
                    device.temperature = gpu.temperature
                    device.is_available = gpu.temperature < 85  # 温度保护
                    
        except Exception as e:
            self.logger.error(f"设备状态更新失败: {e}")
            
    async def _collect_performance_metrics(self):
        """收集性能指标"""
        metrics = {
            'timestamp': datetime.now(),
            'devices': [],
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
        for device in self.gpu_devices:
            device_metrics = {
                'device_id': device.device_id,
                'name': device.name,
                'utilization': device.utilization,
                'memory_used': device.memory_total - device.memory_free,
                'memory_total': device.memory_total,
                'temperature': device.temperature,
                'is_available': device.is_available
            }
            metrics['devices'].append(device_metrics)
            
        self.performance_metrics.append(metrics)
        
    async def _memory_management_loop(self):
        """内存管理循环"""
        while self.is_running:
            try:
                # 清理过期的结果缓存
                await self._cleanup_result_cache()
                
                # 优化内存使用
                await self._optimize_memory_usage()
                
                await asyncio.sleep(30)  # 30秒清理一次
                
            except Exception as e:
                self.logger.error(f"内存管理错误: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_result_cache(self):
        """清理结果缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for task_id, result in self.result_cache.items():
            # 清理超过1小时的缓存
            if (current_time - result.completed_at).total_seconds() > 3600:
                expired_keys.append(task_id)
                
        for key in expired_keys:
            del self.result_cache[key]
            
        if expired_keys:
            self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存")
            
    async def _optimize_memory_usage(self):
        """优化内存使用"""
        for device_id in self.memory_pools:
            # 检查内存使用情况
            if self.memory_usage[device_id] > self.gpu_devices[device_id].memory_total * 0.8:
                # 内存使用超过80%，进行垃圾回收
                if CUDA_AVAILABLE:
                    cp.cuda.Device(device_id).use()
                    cp.get_default_memory_pool().free_all_blocks()
                    
                self.logger.info(f"设备 {device_id} 执行内存优化")
                
    async def _device_health_monitoring_loop(self):
        """设备健康监控循环"""
        while self.is_running:
            try:
                for device in self.gpu_devices:
                    # 检查设备温度
                    if device.temperature > 80:
                        self.logger.warning(f"设备 {device.device_id} 温度过高: {device.temperature}°C")
                        
                    # 检查设备利用率
                    if device.utilization > 95:
                        self.logger.warning(f"设备 {device.device_id} 利用率过高: {device.utilization}%")
                        
                    # 检查内存使用
                    memory_usage_ratio = (device.memory_total - device.memory_free) / device.memory_total
                    if memory_usage_ratio > 0.9:
                        self.logger.warning(f"设备 {device.device_id} 内存使用过高: {memory_usage_ratio:.1%}")
                        
                await asyncio.sleep(10)  # 10秒检查一次
                
            except Exception as e:
                self.logger.error(f"设备健康监控错误: {e}")
                await asyncio.sleep(30)
                
    async def _record_performance_metrics(self, task: ComputeTask, result: ComputeResult):
        """记录性能指标"""
        metrics = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'device_id': result.device_id,
            'execution_time': result.execution_time,
            'memory_used': result.memory_used,
            'throughput': result.result_data.size / result.execution_time,
            'timestamp': result.completed_at
        }
        
        # 这里可以发送到监控系统
        self.logger.debug(f"任务性能: {metrics}")
        
    # 公共接口方法
    async def submit_task(self, task: ComputeTask) -> str:
        """提交计算任务"""
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        return task.task_id
        
    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[ComputeResult]:
        """获取计算结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.result_cache:
                result = self.result_cache[task_id]
                # 清理已获取的结果
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                return result
                
            await asyncio.sleep(0.1)
            
        return None
        
    async def matrix_multiply_async(self, A: np.ndarray, B: np.ndarray, 
                                  device_id: Optional[int] = None) -> np.ndarray:
        """异步矩阵乘法"""
        task = ComputeTask(
            task_id=f"matmul_{int(time.time() * 1000000)}",
            task_type='matrix_multiply',
            input_data=[A, B],
            parameters={},
            device_id=device_id
        )
        
        task_id = await self.submit_task(task)
        result = await self.get_result(task_id)
        
        if result is None:
            raise TimeoutError("矩阵乘法计算超时")
            
        return result.result_data
        
    async def convolution_async(self, input_data: np.ndarray, kernel: np.ndarray,
                              device_id: Optional[int] = None) -> np.ndarray:
        """异步卷积计算"""
        task = ComputeTask(
            task_id=f"conv_{int(time.time() * 1000000)}",
            task_type='convolution',
            input_data=input_data,
            parameters={'kernel': kernel},
            device_id=device_id
        )
        
        task_id = await self.submit_task(task)
        result = await self.get_result(task_id)
        
        if result is None:
            raise TimeoutError("卷积计算超时")
            
        return result.result_data
        
    def get_device_info(self) -> List[Dict[str, Any]]:
        """获取设备信息"""
        return [
            {
                'device_id': device.device_id,
                'name': device.name,
                'memory_total': device.memory_total,
                'memory_free': device.memory_free,
                'utilization': device.utilization,
                'temperature': device.temperature,
                'is_available': device.is_available
            }
            for device in self.gpu_devices
        ]
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.performance_metrics:
            return {}
            
        recent_metrics = list(self.performance_metrics)[-100:]  # 最近100条记录
        
        return {
            'total_devices': len(self.gpu_devices),
            'active_tasks': len(self.active_tasks),
            'cached_results': len(self.result_cache),
            'recent_metrics_count': len(recent_metrics),
            'average_device_utilization': np.mean([
                np.mean([d['utilization'] for d in m['devices']]) 
                for m in recent_metrics if m['devices']
            ]) if recent_metrics else 0
        }
