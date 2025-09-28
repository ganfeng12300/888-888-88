"""
🔄 低延迟执行引擎
生产级毫秒级订单执行系统，支持异步并发处理
实现网络延迟优化、执行队列管理和实时性能监控
"""

import asyncio
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import concurrent.futures
import socket
import ssl
import json
import websockets
import aiohttp

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.trading_execution.smart_order_router import ExecutionRoute, RouteSegment, OrderStatus


class ExecutionPriority(Enum):
    """执行优先级"""
    CRITICAL = 1    # 紧急订单
    HIGH = 2        # 高优先级
    NORMAL = 3      # 普通优先级
    LOW = 4         # 低优先级


class ExecutionMethod(Enum):
    """执行方法"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    FIX_PROTOCOL = "fix_protocol"
    DIRECT_MARKET_ACCESS = "dma"


@dataclass
class ExecutionTask:
    """执行任务"""
    task_id: str
    route_segment: RouteSegment
    priority: ExecutionPriority
    method: ExecutionMethod
    retry_count: int = 0
    max_retries: int = 3
    timeout_ms: float = 1000.0
    callback: Optional[Callable] = None
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class ExecutionResult:
    """执行结果"""
    task_id: str
    success: bool
    order_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    execution_time_ms: float = 0.0
    network_latency_ms: float = 0.0
    error_message: Optional[str] = None
    exchange_response: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyMetrics:
    """延迟指标"""
    timestamp: float
    exchange: str
    method: ExecutionMethod
    network_latency_ms: float
    processing_latency_ms: float
    total_latency_ms: float
    success: bool


class ConnectionPool:
    """连接池管理"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, List[Any]] = {}
        self.connection_stats: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    async def get_connection(self, exchange: str, connection_type: str = "rest") -> Any:
        """获取连接"""
        with self.lock:
            key = f"{exchange}_{connection_type}"
            
            if key not in self.connections:
                self.connections[key] = []
                self.connection_stats[key] = {
                    'total_created': 0,
                    'active_count': 0,
                    'last_used': time.time()
                }
            
            # 尝试复用现有连接
            if self.connections[key]:
                conn = self.connections[key].pop()
                self.connection_stats[key]['active_count'] += 1
                self.connection_stats[key]['last_used'] = time.time()
                return conn
            
            # 创建新连接
            if self.connection_stats[key]['total_created'] < self.max_connections:
                conn = await self._create_connection(exchange, connection_type)
                if conn:
                    self.connection_stats[key]['total_created'] += 1
                    self.connection_stats[key]['active_count'] += 1
                    self.connection_stats[key]['last_used'] = time.time()
                    return conn
            
            return None
    
    async def _create_connection(self, exchange: str, connection_type: str) -> Any:
        """创建连接"""
        try:
            if connection_type == "rest":
                # 创建HTTP会话
                timeout = aiohttp.ClientTimeout(total=5.0, connect=1.0)
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=20,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector
                )
                return session
            
            elif connection_type == "websocket":
                # WebSocket连接配置
                ws_urls = {
                    'binance': 'wss://stream.binance.com:9443/ws',
                    'okx': 'wss://ws.okx.com:8443/ws/v5/public',
                    'bybit': 'wss://stream.bybit.com/v5/public/spot'
                }
                
                if exchange in ws_urls:
                    websocket = await websockets.connect(
                        ws_urls[exchange],
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=10
                    )
                    return websocket
            
            return None
            
        except Exception as e:
            logger.error(f"创建连接失败 {exchange}-{connection_type}: {e}")
            return None
    
    def return_connection(self, exchange: str, connection_type: str, connection: Any):
        """归还连接"""
        with self.lock:
            key = f"{exchange}_{connection_type}"
            
            if key in self.connections:
                self.connections[key].append(connection)
                self.connection_stats[key]['active_count'] -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        with self.lock:
            return {
                'connections': {k: len(v) for k, v in self.connections.items()},
                'stats': self.connection_stats.copy()
            }


class LowLatencyExecutionEngine:
    """低延迟执行引擎"""
    
    def __init__(self, exchange_connectors: Dict[str, Any]):
        self.exchange_connectors = exchange_connectors
        self.connection_pool = ConnectionPool(max_connections=20)
        
        # 执行队列 (按优先级)
        self.execution_queues: Dict[ExecutionPriority, queue.PriorityQueue] = {
            ExecutionPriority.CRITICAL: queue.PriorityQueue(),
            ExecutionPriority.HIGH: queue.PriorityQueue(),
            ExecutionPriority.NORMAL: queue.PriorityQueue(),
            ExecutionPriority.LOW: queue.PriorityQueue()
        }
        
        # 执行结果
        self.execution_results: Dict[str, ExecutionResult] = {}
        self.latency_metrics: deque = deque(maxlen=1000)
        
        # 性能配置
        self.max_concurrent_executions = 50
        self.execution_timeout_ms = 5000.0
        self.retry_delay_ms = 100.0
        self.circuit_breaker_threshold = 0.8  # 80%成功率阈值
        
        # 执行器线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=20,
            thread_name_prefix="execution_worker"
        )
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.ORDER_EXECUTION, [9, 10, 11, 12])
        
        # 启动执行循环
        self.running = False
        self.execution_tasks = []
        self._start_execution_loops()
        
        logger.info("低延迟执行引擎初始化完成")
    
    def _start_execution_loops(self):
        """启动执行循环"""
        self.running = True
        
        # 为每个优先级启动执行循环
        for priority in ExecutionPriority:
            task = asyncio.create_task(self._execution_loop(priority))
            self.execution_tasks.append(task)
        
        # 启动延迟监控
        monitor_task = asyncio.create_task(self._latency_monitoring_loop())
        self.execution_tasks.append(monitor_task)
        
        logger.info("执行循环已启动")
    
    async def _execution_loop(self, priority: ExecutionPriority):
        """执行循环"""
        logger.info(f"启动{priority.name}优先级执行循环")
        
        while self.running:
            try:
                # 获取执行任务
                try:
                    # 非阻塞获取任务
                    _, task = self.execution_queues[priority].get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)  # 1ms
                    continue
                
                # 执行任务
                await self._execute_task(task)
                
            except Exception as e:
                logger.error(f"执行循环出错 {priority.name}: {e}")
                await asyncio.sleep(0.01)
    
    async def _execute_task(self, task: ExecutionTask):
        """执行单个任务"""
        try:
            task.start_time = time.time()
            task.status = OrderStatus.ROUTING
            
            # 选择执行方法
            if task.method == ExecutionMethod.REST_API:
                result = await self._execute_via_rest(task)
            elif task.method == ExecutionMethod.WEBSOCKET:
                result = await self._execute_via_websocket(task)
            else:
                result = ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    error_message=f"不支持的执行方法: {task.method}"
                )
            
            task.end_time = time.time()
            task.status = OrderStatus.FILLED if result.success else OrderStatus.FAILED
            
            # 记录结果
            self.execution_results[task.task_id] = result
            
            # 记录延迟指标
            if task.start_time and task.end_time:
                execution_time = (task.end_time - task.start_time) * 1000
                
                latency_metric = LatencyMetrics(
                    timestamp=time.time(),
                    exchange=task.route_segment.exchange,
                    method=task.method,
                    network_latency_ms=result.network_latency_ms,
                    processing_latency_ms=execution_time - result.network_latency_ms,
                    total_latency_ms=execution_time,
                    success=result.success
                )
                
                self.latency_metrics.append(latency_metric)
            
            # 调用回调
            if task.callback:
                try:
                    await task.callback(result)
                except Exception as e:
                    logger.error(f"执行回调失败: {e}")
            
            logger.info(f"任务执行完成: {task.task_id}, "
                       f"成功={result.success}, 延迟={result.execution_time_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"执行任务失败 {task.task_id}: {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(self.retry_delay_ms / 1000.0)
                await self.submit_execution_task(task)
    
    async def _execute_via_rest(self, task: ExecutionTask) -> ExecutionResult:
        """通过REST API执行"""
        start_time = time.time()
        
        try:
            # 获取连接
            session = await self.connection_pool.get_connection(
                task.route_segment.exchange, "rest"
            )
            
            if not session:
                return ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    error_message="无法获取REST连接"
                )
            
            # 构建订单参数
            order_params = self._build_order_params(task)
            
            # 获取交易所API端点
            api_url = self._get_api_endpoint(task.route_segment.exchange, "place_order")
            
            # 发送订单
            network_start = time.time()
            
            async with session.post(
                api_url,
                json=order_params,
                timeout=aiohttp.ClientTimeout(total=task.timeout_ms / 1000.0)
            ) as response:
                response_data = await response.json()
                network_end = time.time()
                
                network_latency = (network_end - network_start) * 1000
                total_time = (network_end - start_time) * 1000
                
                # 归还连接
                self.connection_pool.return_connection(
                    task.route_segment.exchange, "rest", session
                )
                
                # 解析响应
                if response.status == 200 and response_data.get('success'):
                    return ExecutionResult(
                        task_id=task.task_id,
                        success=True,
                        order_id=response_data.get('orderId'),
                        filled_quantity=float(response_data.get('executedQty', 0)),
                        avg_price=float(response_data.get('price', task.route_segment.price)),
                        execution_time_ms=total_time,
                        network_latency_ms=network_latency,
                        exchange_response=response_data
                    )
                else:
                    return ExecutionResult(
                        task_id=task.task_id,
                        success=False,
                        execution_time_ms=total_time,
                        network_latency_ms=network_latency,
                        error_message=response_data.get('msg', f'HTTP {response.status}'),
                        exchange_response=response_data
                    )
        
        except asyncio.TimeoutError:
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message="执行超时",
                execution_time_ms=task.timeout_ms
            )
        
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                execution_time_ms=total_time,
                error_message=str(e)
            )
    
    async def _execute_via_websocket(self, task: ExecutionTask) -> ExecutionResult:
        """通过WebSocket执行"""
        start_time = time.time()
        
        try:
            # 获取WebSocket连接
            websocket = await self.connection_pool.get_connection(
                task.route_segment.exchange, "websocket"
            )
            
            if not websocket:
                return ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    error_message="无法获取WebSocket连接"
                )
            
            # 构建订单消息
            order_message = self._build_websocket_message(task)
            
            # 发送订单
            network_start = time.time()
            await websocket.send(json.dumps(order_message))
            
            # 等待响应
            response_text = await asyncio.wait_for(
                websocket.recv(),
                timeout=task.timeout_ms / 1000.0
            )
            
            network_end = time.time()
            response_data = json.loads(response_text)
            
            network_latency = (network_end - network_start) * 1000
            total_time = (network_end - start_time) * 1000
            
            # 归还连接
            self.connection_pool.return_connection(
                task.route_segment.exchange, "websocket", websocket
            )
            
            # 解析响应
            if response_data.get('success') or response_data.get('result'):
                return ExecutionResult(
                    task_id=task.task_id,
                    success=True,
                    order_id=response_data.get('id') or response_data.get('orderId'),
                    filled_quantity=float(response_data.get('filled', 0)),
                    avg_price=float(response_data.get('price', task.route_segment.price)),
                    execution_time_ms=total_time,
                    network_latency_ms=network_latency,
                    exchange_response=response_data
                )
            else:
                return ExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    execution_time_ms=total_time,
                    network_latency_ms=network_latency,
                    error_message=response_data.get('error', 'WebSocket执行失败'),
                    exchange_response=response_data
                )
        
        except asyncio.TimeoutError:
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message="WebSocket执行超时",
                execution_time_ms=task.timeout_ms
            )
        
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                execution_time_ms=total_time,
                error_message=str(e)
            )
    
    def _build_order_params(self, task: ExecutionTask) -> Dict[str, Any]:
        """构建订单参数"""
        segment = task.route_segment
        
        # 基础参数
        params = {
            'symbol': segment.exchange.replace('/', ''),  # 移除斜杠
            'side': 'BUY' if segment.quantity > 0 else 'SELL',
            'type': 'MARKET',  # 市价单
            'quantity': abs(segment.quantity),
            'timestamp': int(time.time() * 1000)
        }
        
        # 交易所特定参数
        if segment.exchange.lower() == 'binance':
            params.update({
                'timeInForce': 'IOC',  # 立即成交或取消
                'newOrderRespType': 'RESULT'
            })
        elif segment.exchange.lower() == 'okx':
            params.update({
                'instId': segment.exchange,
                'tdMode': 'cash',
                'ordType': 'market'
            })
        
        return params
    
    def _build_websocket_message(self, task: ExecutionTask) -> Dict[str, Any]:
        """构建WebSocket消息"""
        segment = task.route_segment
        
        message = {
            'id': task.task_id,
            'method': 'order.place',
            'params': {
                'symbol': segment.exchange,
                'side': 'buy' if segment.quantity > 0 else 'sell',
                'type': 'market',
                'amount': abs(segment.quantity),
                'timestamp': int(time.time() * 1000)
            }
        }
        
        return message
    
    def _get_api_endpoint(self, exchange: str, endpoint_type: str) -> str:
        """获取API端点"""
        endpoints = {
            'binance': {
                'place_order': 'https://api.binance.com/api/v3/order',
                'cancel_order': 'https://api.binance.com/api/v3/order',
                'order_status': 'https://api.binance.com/api/v3/order'
            },
            'okx': {
                'place_order': 'https://www.okx.com/api/v5/trade/order',
                'cancel_order': 'https://www.okx.com/api/v5/trade/cancel-order',
                'order_status': 'https://www.okx.com/api/v5/trade/order'
            },
            'bybit': {
                'place_order': 'https://api.bybit.com/v5/order/create',
                'cancel_order': 'https://api.bybit.com/v5/order/cancel',
                'order_status': 'https://api.bybit.com/v5/order/realtime'
            }
        }
        
        return endpoints.get(exchange.lower(), {}).get(endpoint_type, '')
    
    async def submit_execution_task(self, task: ExecutionTask):
        """提交执行任务"""
        try:
            # 添加到对应优先级队列
            priority_value = task.priority.value + task.created_time  # 时间戳作为次要排序
            self.execution_queues[task.priority].put((priority_value, task))
            
            logger.info(f"任务已提交: {task.task_id}, 优先级={task.priority.name}")
            
        except Exception as e:
            logger.error(f"提交执行任务失败: {e}")
    
    async def execute_route(self, route: ExecutionRoute, 
                           priority: ExecutionPriority = ExecutionPriority.NORMAL) -> List[str]:
        """执行路由"""
        try:
            task_ids = []
            
            for i, segment in enumerate(route.segments):
                # 创建执行任务
                task = ExecutionTask(
                    task_id=f"{route.order_id}_seg_{i}",
                    route_segment=segment,
                    priority=priority,
                    method=ExecutionMethod.REST_API,  # 默认使用REST
                    timeout_ms=min(5000.0, segment.estimated_fill_time * 1000 * 2)
                )
                
                # 提交任务
                await self.submit_execution_task(task)
                task_ids.append(task.task_id)
            
            logger.info(f"路由执行已提交: {route.order_id}, {len(task_ids)}个任务")
            
            return task_ids
            
        except Exception as e:
            logger.error(f"执行路由失败: {e}")
            return []
    
    async def _latency_monitoring_loop(self):
        """延迟监控循环"""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # 每10秒监控一次
                
                if len(self.latency_metrics) < 10:
                    continue
                
                # 分析延迟指标
                recent_metrics = list(self.latency_metrics)[-100:]  # 最近100个
                
                # 按交易所分组
                exchange_metrics = {}
                for metric in recent_metrics:
                    if metric.exchange not in exchange_metrics:
                        exchange_metrics[metric.exchange] = []
                    exchange_metrics[metric.exchange].append(metric)
                
                # 计算统计信息
                for exchange, metrics in exchange_metrics.items():
                    if not metrics:
                        continue
                    
                    latencies = [m.total_latency_ms for m in metrics]
                    success_rate = sum(1 for m in metrics if m.success) / len(metrics)
                    
                    avg_latency = np.mean(latencies)
                    p95_latency = np.percentile(latencies, 95)
                    p99_latency = np.percentile(latencies, 99)
                    
                    # 检查性能问题
                    if avg_latency > 1000:  # 平均延迟超过1秒
                        logger.warning(f"{exchange} 平均延迟过高: {avg_latency:.1f}ms")
                    
                    if success_rate < self.circuit_breaker_threshold:
                        logger.warning(f"{exchange} 成功率过低: {success_rate:.2%}")
                    
                    if int(time.time()) % 60 == 0:  # 每分钟记录一次
                        logger.info(f"{exchange} 延迟统计: "
                                   f"平均={avg_latency:.1f}ms, "
                                   f"P95={p95_latency:.1f}ms, "
                                   f"成功率={success_rate:.2%}")
                
            except Exception as e:
                logger.error(f"延迟监控出错: {e}")
    
    def get_execution_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取执行结果"""
        return self.execution_results.get(task_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.latency_metrics:
            return {}
        
        recent_metrics = list(self.latency_metrics)[-100:]
        
        # 总体统计
        total_latencies = [m.total_latency_ms for m in recent_metrics]
        network_latencies = [m.network_latency_ms for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)
        
        # 按交易所统计
        exchange_stats = {}
        for metric in recent_metrics:
            if metric.exchange not in exchange_stats:
                exchange_stats[metric.exchange] = {
                    'latencies': [],
                    'successes': 0,
                    'total': 0
                }
            
            exchange_stats[metric.exchange]['latencies'].append(metric.total_latency_ms)
            exchange_stats[metric.exchange]['total'] += 1
            if metric.success:
                exchange_stats[metric.exchange]['successes'] += 1
        
        # 计算交易所统计
        for exchange, stats in exchange_stats.items():
            if stats['latencies']:
                stats['avg_latency'] = np.mean(stats['latencies'])
                stats['p95_latency'] = np.percentile(stats['latencies'], 95)
                stats['success_rate'] = stats['successes'] / stats['total']
        
        return {
            'total_executions': len(recent_metrics),
            'success_rate': success_count / len(recent_metrics) if recent_metrics else 0,
            'avg_total_latency': np.mean(total_latencies) if total_latencies else 0,
            'avg_network_latency': np.mean(network_latencies) if network_latencies else 0,
            'p95_latency': np.percentile(total_latencies, 95) if total_latencies else 0,
            'p99_latency': np.percentile(total_latencies, 99) if total_latencies else 0,
            'exchange_stats': exchange_stats,
            'connection_pool_stats': self.connection_pool.get_stats(),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.execution_queues.items()
            }
        }
    
    async def stop(self):
        """停止执行引擎"""
        self.running = False
        
        # 等待所有任务完成
        for task in self.execution_tasks:
            task.cancel()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("低延迟执行引擎已停止")


# 全局执行引擎实例
execution_engine = None


def create_execution_engine(exchange_connectors: Dict[str, Any]) -> LowLatencyExecutionEngine:
    """创建执行引擎"""
    global execution_engine
    execution_engine = LowLatencyExecutionEngine(exchange_connectors)
    return execution_engine


async def main():
    """测试主函数"""
    logger.info("启动低延迟执行引擎测试...")
    
    # 模拟交易所连接器
    mock_connectors = {
        'binance': None,
        'okx': None,
        'bybit': None
    }
    
    # 创建执行引擎
    engine = create_execution_engine(mock_connectors)
    
    try:
        # 运行测试
        await asyncio.sleep(30)
        
        # 获取性能指标
        metrics = engine.get_performance_metrics()
        logger.info(f"性能指标: {json.dumps(metrics, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())

