"""
📊 Prometheus指标收集系统
生产级监控指标收集，实现系统指标、应用指标、业务指标的完整收集
支持自定义指标、指标推送、指标聚合和性能优化
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import warnings
warnings.filterwarnings('ignore')

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        push_to_gateway, delete_from_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from loguru import logger
from src.core.config import settings


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"         # 计数器
    GAUGE = "gauge"             # 仪表盘
    HISTOGRAM = "histogram"     # 直方图
    SUMMARY = "summary"         # 摘要
    INFO = "info"               # 信息


@dataclass
class MetricConfig:
    """指标配置"""
    name: str                               # 指标名称
    metric_type: MetricType                 # 指标类型
    description: str                        # 指标描述
    labels: List[str] = field(default_factory=list)  # 标签列表
    buckets: Optional[List[float]] = None   # 直方图桶
    objectives: Optional[Dict[float, float]] = None  # 摘要目标
    namespace: str = "trading"              # 命名空间
    subsystem: str = ""                     # 子系统
    unit: str = ""                          # 单位


class SystemMetricsCollector:
    """系统指标收集器"""
    
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self.metrics = {}
        self._setup_system_metrics()
        
    def _setup_system_metrics(self):
        """设置系统指标"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端不可用，跳过系统指标设置")
            return
        
        # CPU指标
        self.metrics['cpu_usage_percent'] = Gauge(
            'system_cpu_usage_percent',
            'CPU使用率百分比',
            ['cpu'],
            registry=self.registry
        )
        
        self.metrics['cpu_count'] = Gauge(
            'system_cpu_count',
            'CPU核心数',
            registry=self.registry
        )
        
        self.metrics['load_average'] = Gauge(
            'system_load_average',
            '系统负载平均值',
            ['period'],
            registry=self.registry
        )
        
        # 内存指标
        self.metrics['memory_usage_bytes'] = Gauge(
            'system_memory_usage_bytes',
            '内存使用量(字节)',
            ['type'],
            registry=self.registry
        )
        
        self.metrics['memory_usage_percent'] = Gauge(
            'system_memory_usage_percent',
            '内存使用率百分比',
            registry=self.registry
        )
        
        # 磁盘指标
        self.metrics['disk_usage_bytes'] = Gauge(
            'system_disk_usage_bytes',
            '磁盘使用量(字节)',
            ['device', 'type'],
            registry=self.registry
        )
        
        self.metrics['disk_usage_percent'] = Gauge(
            'system_disk_usage_percent',
            '磁盘使用率百分比',
            ['device'],
            registry=self.registry
        )
        
        self.metrics['disk_io_bytes'] = Counter(
            'system_disk_io_bytes_total',
            '磁盘IO字节数',
            ['device', 'direction'],
            registry=self.registry
        )
        
        # 网络指标
        self.metrics['network_bytes'] = Counter(
            'system_network_bytes_total',
            '网络字节数',
            ['interface', 'direction'],
            registry=self.registry
        )
        
        self.metrics['network_packets'] = Counter(
            'system_network_packets_total',
            '网络包数',
            ['interface', 'direction'],
            registry=self.registry
        )
        
        # 进程指标
        self.metrics['process_count'] = Gauge(
            'system_process_count',
            '进程数量',
            registry=self.registry
        )
        
        self.metrics['process_memory_bytes'] = Gauge(
            'process_memory_bytes',
            '进程内存使用量',
            ['pid', 'name'],
            registry=self.registry
        )
        
        self.metrics['process_cpu_percent'] = Gauge(
            'process_cpu_percent',
            '进程CPU使用率',
            ['pid', 'name'],
            registry=self.registry
        )
    
    def collect_system_metrics(self):
        """收集系统指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, percent in enumerate(cpu_percent):
                self.metrics['cpu_usage_percent'].labels(cpu=f'cpu{i}').set(percent)
            
            self.metrics['cpu_count'].set(psutil.cpu_count())
            
            # 负载平均值
            load_avg = psutil.getloadavg()
            self.metrics['load_average'].labels(period='1m').set(load_avg[0])
            self.metrics['load_average'].labels(period='5m').set(load_avg[1])
            self.metrics['load_average'].labels(period='15m').set(load_avg[2])
            
            # 内存指标
            memory = psutil.virtual_memory()
            self.metrics['memory_usage_bytes'].labels(type='total').set(memory.total)
            self.metrics['memory_usage_bytes'].labels(type='available').set(memory.available)
            self.metrics['memory_usage_bytes'].labels(type='used').set(memory.used)
            self.metrics['memory_usage_bytes'].labels(type='free').set(memory.free)
            self.metrics['memory_usage_percent'].set(memory.percent)
            
            # 磁盘指标
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace('/', '_')
                    
                    self.metrics['disk_usage_bytes'].labels(device=device, type='total').set(disk_usage.total)
                    self.metrics['disk_usage_bytes'].labels(device=device, type='used').set(disk_usage.used)
                    self.metrics['disk_usage_bytes'].labels(device=device, type='free').set(disk_usage.free)
                    
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    self.metrics['disk_usage_percent'].labels(device=device).set(usage_percent)
                except PermissionError:
                    continue
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters(perdisk=True)
            for device, io_counters in disk_io.items():
                device_clean = device.replace('/', '_')
                self.metrics['disk_io_bytes'].labels(device=device_clean, direction='read').inc(io_counters.read_bytes)
                self.metrics['disk_io_bytes'].labels(device=device_clean, direction='write').inc(io_counters.write_bytes)
            
            # 网络指标
            network_io = psutil.net_io_counters(pernic=True)
            for interface, io_counters in network_io.items():
                self.metrics['network_bytes'].labels(interface=interface, direction='sent').inc(io_counters.bytes_sent)
                self.metrics['network_bytes'].labels(interface=interface, direction='recv').inc(io_counters.bytes_recv)
                self.metrics['network_packets'].labels(interface=interface, direction='sent').inc(io_counters.packets_sent)
                self.metrics['network_packets'].labels(interface=interface, direction='recv').inc(io_counters.packets_recv)
            
            # 进程指标
            self.metrics['process_count'].set(len(psutil.pids()))
            
            # 当前进程指标
            current_process = psutil.Process()
            self.metrics['process_memory_bytes'].labels(
                pid=current_process.pid, 
                name=current_process.name()
            ).set(current_process.memory_info().rss)
            
            self.metrics['process_cpu_percent'].labels(
                pid=current_process.pid, 
                name=current_process.name()
            ).set(current_process.cpu_percent())
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")


class ApplicationMetricsCollector:
    """应用指标收集器"""
    
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self.metrics = {}
        self._setup_application_metrics()
    
    def _setup_application_metrics(self):
        """设置应用指标"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端不可用，跳过应用指标设置")
            return
        
        # HTTP请求指标
        self.metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'HTTP请求总数',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['http_request_duration_seconds'] = Histogram(
            'http_request_duration_seconds',
            'HTTP请求持续时间',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # 数据库指标
        self.metrics['database_connections'] = Gauge(
            'database_connections',
            '数据库连接数',
            ['database', 'state'],
            registry=self.registry
        )
        
        self.metrics['database_query_duration_seconds'] = Histogram(
            'database_query_duration_seconds',
            '数据库查询持续时间',
            ['database', 'operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        # 缓存指标
        self.metrics['cache_operations_total'] = Counter(
            'cache_operations_total',
            '缓存操作总数',
            ['cache', 'operation', 'result'],
            registry=self.registry
        )
        
        self.metrics['cache_size_bytes'] = Gauge(
            'cache_size_bytes',
            '缓存大小(字节)',
            ['cache'],
            registry=self.registry
        )
        
        # 消息队列指标
        self.metrics['message_queue_size'] = Gauge(
            'message_queue_size',
            '消息队列大小',
            ['queue'],
            registry=self.registry
        )
        
        self.metrics['messages_processed_total'] = Counter(
            'messages_processed_total',
            '处理的消息总数',
            ['queue', 'status'],
            registry=self.registry
        )
        
        # 任务指标
        self.metrics['tasks_total'] = Counter(
            'tasks_total',
            '任务总数',
            ['task_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['task_duration_seconds'] = Histogram(
            'task_duration_seconds',
            '任务执行时间',
            ['task_type'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # 错误指标
        self.metrics['errors_total'] = Counter(
            'errors_total',
            '错误总数',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # 应用信息
        self.metrics['app_info'] = Info(
            'app_info',
            '应用信息',
            registry=self.registry
        )
        
        # 设置应用信息
        self.metrics['app_info'].info({
            'version': '1.0.0',
            'environment': 'production',
            'build_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': '3.11'
        })


class TradingMetricsCollector:
    """交易指标收集器"""
    
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self.metrics = {}
        self._setup_trading_metrics()
    
    def _setup_trading_metrics(self):
        """设置交易指标"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端不可用，跳过交易指标设置")
            return
        
        # 订单指标
        self.metrics['orders_total'] = Counter(
            'trading_orders_total',
            '订单总数',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.metrics['order_value_total'] = Counter(
            'trading_order_value_total',
            '订单价值总计',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        self.metrics['order_latency_seconds'] = Histogram(
            'trading_order_latency_seconds',
            '订单延迟时间',
            ['symbol', 'order_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # 持仓指标
        self.metrics['positions'] = Gauge(
            'trading_positions',
            '当前持仓',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        self.metrics['position_value'] = Gauge(
            'trading_position_value',
            '持仓价值',
            ['symbol'],
            registry=self.registry
        )
        
        # 盈亏指标
        self.metrics['pnl_realized'] = Counter(
            'trading_pnl_realized_total',
            '已实现盈亏',
            ['symbol'],
            registry=self.registry
        )
        
        self.metrics['pnl_unrealized'] = Gauge(
            'trading_pnl_unrealized',
            '未实现盈亏',
            ['symbol'],
            registry=self.registry
        )
        
        # 策略指标
        self.metrics['strategy_signals_total'] = Counter(
            'trading_strategy_signals_total',
            '策略信号总数',
            ['strategy', 'signal_type'],
            registry=self.registry
        )
        
        self.metrics['strategy_performance'] = Gauge(
            'trading_strategy_performance',
            '策略表现',
            ['strategy', 'metric'],
            registry=self.registry
        )
        
        # 风险指标
        self.metrics['risk_exposure'] = Gauge(
            'trading_risk_exposure',
            '风险敞口',
            ['risk_type'],
            registry=self.registry
        )
        
        self.metrics['risk_violations_total'] = Counter(
            'trading_risk_violations_total',
            '风险违规总数',
            ['risk_type', 'severity'],
            registry=self.registry
        )
        
        # 市场数据指标
        self.metrics['market_data_updates_total'] = Counter(
            'trading_market_data_updates_total',
            '市场数据更新总数',
            ['symbol', 'data_type'],
            registry=self.registry
        )
        
        self.metrics['market_data_latency_seconds'] = Histogram(
            'trading_market_data_latency_seconds',
            '市场数据延迟',
            ['symbol', 'data_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )


class PrometheusMetricsManager:
    """Prometheus指标管理器"""
    
    def __init__(self, pushgateway_url: Optional[str] = None, job_name: str = "trading-system"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        
        # 创建注册表
        self.registry = CollectorRegistry()
        
        # 初始化收集器
        self.system_collector = SystemMetricsCollector(self.registry)
        self.app_collector = ApplicationMetricsCollector(self.registry)
        self.trading_collector = TradingMetricsCollector(self.registry)
        
        # 自定义指标
        self.custom_metrics: Dict[str, Any] = {}
        
        # 收集任务
        self.collection_task = None
        self.collection_interval = 10  # 10秒收集一次
        self.running = False
        
        logger.info("Prometheus指标管理器初始化完成")
    
    def create_metric(self, config: MetricConfig) -> Optional[Any]:
        """创建自定义指标"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端不可用，无法创建指标")
            return None
        
        try:
            metric_name = f"{config.namespace}_{config.subsystem}_{config.name}".strip('_')
            
            if config.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_name,
                    config.description,
                    config.labels,
                    buckets=config.buckets,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry
                )
            elif config.metric_type == MetricType.INFO:
                metric = Info(
                    metric_name,
                    config.description,
                    registry=self.registry
                )
            else:
                logger.error(f"不支持的指标类型: {config.metric_type}")
                return None
            
            self.custom_metrics[config.name] = metric
            logger.info(f"创建自定义指标: {metric_name}")
            return metric
            
        except Exception as e:
            logger.error(f"创建指标失败: {config.name} - {e}")
            return None
    
    def get_metric(self, name: str) -> Optional[Any]:
        """获取指标"""
        # 先查找自定义指标
        if name in self.custom_metrics:
            return self.custom_metrics[name]
        
        # 查找系统指标
        if name in self.system_collector.metrics:
            return self.system_collector.metrics[name]
        
        # 查找应用指标
        if name in self.app_collector.metrics:
            return self.app_collector.metrics[name]
        
        # 查找交易指标
        if name in self.trading_collector.metrics:
            return self.trading_collector.metrics[name]
        
        return None
    
    async def start_collection(self):
        """启动指标收集"""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("指标收集已启动")
    
    async def stop_collection(self):
        """停止指标收集"""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("指标收集已停止")
    
    async def _collection_loop(self):
        """指标收集循环"""
        while self.running:
            try:
                # 收集系统指标
                self.system_collector.collect_system_metrics()
                
                # 推送到Pushgateway
                if self.pushgateway_url:
                    await self._push_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
                await asyncio.sleep(5)
    
    async def _push_metrics(self):
        """推送指标到Pushgateway"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            def push_job():
                push_to_gateway(
                    self.pushgateway_url,
                    job=self.job_name,
                    registry=self.registry
                )
            
            # 在线程池中执行推送
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, push_job)
            
        except Exception as e:
            logger.error(f"推送指标失败: {e}")
    
    def get_metrics_text(self) -> str:
        """获取指标文本格式"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"生成指标文本失败: {e}")
            return ""
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录HTTP请求指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.app_collector.metrics['http_requests_total'].labels(
                method=method, endpoint=endpoint, status=str(status)
            ).inc()
            
            self.app_collector.metrics['http_request_duration_seconds'].labels(
                method=method, endpoint=endpoint
            ).observe(duration)
        except Exception as e:
            logger.error(f"记录HTTP请求指标失败: {e}")
    
    def record_database_query(self, database: str, operation: str, duration: float):
        """记录数据库查询指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.app_collector.metrics['database_query_duration_seconds'].labels(
                database=database, operation=operation
            ).observe(duration)
        except Exception as e:
            logger.error(f"记录数据库查询指标失败: {e}")
    
    def record_trading_order(self, symbol: str, side: str, status: str, value: float, latency: float):
        """记录交易订单指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.trading_collector.metrics['orders_total'].labels(
                symbol=symbol, side=side, status=status
            ).inc()
            
            self.trading_collector.metrics['order_value_total'].labels(
                symbol=symbol, side=side
            ).inc(value)
            
            self.trading_collector.metrics['order_latency_seconds'].labels(
                symbol=symbol, order_type='market'
            ).observe(latency)
        except Exception as e:
            logger.error(f"记录交易订单指标失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'pushgateway_url': self.pushgateway_url,
            'job_name': self.job_name,
            'collection_running': self.running,
            'collection_interval': self.collection_interval,
            'custom_metrics_count': len(self.custom_metrics),
            'system_metrics_count': len(self.system_collector.metrics),
            'app_metrics_count': len(self.app_collector.metrics),
            'trading_metrics_count': len(self.trading_collector.metrics)
        }


# 全局指标管理器实例
metrics_manager = PrometheusMetricsManager()
