"""
🏥 健康监控系统
生产级系统健康监控，实现自动恢复、故障转移、监控告警等完整功能
支持组件健康检查、系统指标监控、异常检测和自动修复
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"         # 健康
    WARNING = "warning"         # 警告
    CRITICAL = "critical"       # 严重
    UNKNOWN = "unknown"         # 未知
    DEGRADED = "degraded"       # 降级


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"               # 信息
    WARNING = "warning"         # 警告
    ERROR = "error"             # 错误
    CRITICAL = "critical"       # 严重


@dataclass
class HealthMetric:
    """健康指标"""
    name: str                               # 指标名称
    value: float                            # 指标值
    unit: str = ""                          # 单位
    threshold_warning: Optional[float] = None    # 警告阈值
    threshold_critical: Optional[float] = None   # 严重阈值
    timestamp: float = field(default_factory=time.time)  # 时间戳
    tags: Dict[str, str] = field(default_factory=dict)   # 标签


@dataclass
class HealthCheck:
    """健康检查"""
    component_id: str                       # 组件ID
    check_name: str                         # 检查名称
    status: HealthStatus                    # 健康状态
    message: str = ""                       # 状态消息
    metrics: List[HealthMetric] = field(default_factory=list)  # 指标列表
    timestamp: float = field(default_factory=time.time)  # 时间戳
    duration_ms: float = 0.0                # 检查耗时(毫秒)
    error: Optional[str] = None             # 错误信息


@dataclass
class Alert:
    """告警"""
    id: str                                 # 告警ID
    component_id: str                       # 组件ID
    level: AlertLevel                       # 告警级别
    title: str                              # 告警标题
    message: str                            # 告警消息
    metric_name: Optional[str] = None       # 相关指标
    metric_value: Optional[float] = None    # 指标值
    threshold: Optional[float] = None       # 阈值
    timestamp: float = field(default_factory=time.time)  # 时间戳
    resolved: bool = False                  # 是否已解决
    resolved_at: Optional[float] = None     # 解决时间
    tags: Dict[str, str] = field(default_factory=dict)  # 标签


class SystemMetricsCollector:
    """系统指标收集器"""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def collect_system_metrics(self) -> List[HealthMetric]:
        """收集系统指标"""
        metrics = []
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                tags={"type": "system", "resource": "cpu"}
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                name="memory_usage_percent",
                value=memory.percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                tags={"type": "system", "resource": "memory"}
            ))
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                name="disk_usage_percent",
                value=disk_percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
                tags={"type": "system", "resource": "disk"}
            ))
            
            # 网络IO
            net_io = psutil.net_io_counters()
            metrics.append(HealthMetric(
                name="network_bytes_sent",
                value=net_io.bytes_sent,
                unit="bytes",
                tags={"type": "system", "resource": "network", "direction": "sent"}
            ))
            
            metrics.append(HealthMetric(
                name="network_bytes_recv",
                value=net_io.bytes_recv,
                unit="bytes",
                tags={"type": "system", "resource": "network", "direction": "recv"}
            ))
            
            # 进程数
            process_count = len(psutil.pids())
            metrics.append(HealthMetric(
                name="process_count",
                value=process_count,
                unit="count",
                threshold_warning=500,
                threshold_critical=1000,
                tags={"type": "system", "resource": "process"}
            ))
            
            # 负载平均值
            load_avg = psutil.getloadavg()
            metrics.append(HealthMetric(
                name="load_average_1m",
                value=load_avg[0],
                unit="",
                threshold_warning=psutil.cpu_count() * 0.8,
                threshold_critical=psutil.cpu_count() * 1.2,
                tags={"type": "system", "resource": "load", "period": "1m"}
            ))
            
            # 存储指标历史
            for metric in metrics:
                self.metrics_history[metric.name].append({
                    'value': metric.value,
                    'timestamp': metric.timestamp
                })
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
        
        return metrics
    
    def get_metric_history(self, metric_name: str, duration_seconds: int = 300) -> List[Dict[str, Any]]:
        """获取指标历史"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = time.time() - duration_seconds
        history = list(self.metrics_history[metric_name])
        
        return [
            point for point in history 
            if point['timestamp'] >= cutoff_time
        ]


class ComponentHealthChecker:
    """组件健康检查器"""
    
    def __init__(self):
        self.health_checks: Dict[str, List[Callable]] = defaultdict(list)
        self.last_check_results: Dict[str, HealthCheck] = {}
        
    def register_health_check(self, component_id: str, check_func: Callable):
        """注册健康检查函数"""
        self.health_checks[component_id].append(check_func)
        logger.info(f"注册健康检查: {component_id}")
    
    async def check_component_health(self, component_id: str) -> HealthCheck:
        """检查组件健康状态"""
        start_time = time.perf_counter()
        
        try:
            overall_status = HealthStatus.HEALTHY
            messages = []
            all_metrics = []
            
            # 执行所有健康检查
            for check_func in self.health_checks.get(component_id, []):
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        result = await check_func()
                    else:
                        result = check_func()
                    
                    if isinstance(result, dict):
                        status = HealthStatus(result.get('status', 'healthy'))
                        message = result.get('message', '')
                        metrics = result.get('metrics', [])
                    else:
                        status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                        message = "Health check passed" if result else "Health check failed"
                        metrics = []
                    
                    # 更新整体状态
                    if status == HealthStatus.CRITICAL:
                        overall_status = HealthStatus.CRITICAL
                    elif status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.WARNING
                    
                    if message:
                        messages.append(message)
                    
                    all_metrics.extend(metrics)
                    
                except Exception as e:
                    logger.error(f"健康检查执行失败: {component_id} - {e}")
                    overall_status = HealthStatus.CRITICAL
                    messages.append(f"Health check error: {str(e)}")
            
            # 创建健康检查结果
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            health_check = HealthCheck(
                component_id=component_id,
                check_name="component_health",
                status=overall_status,
                message="; ".join(messages) if messages else "All checks passed",
                metrics=all_metrics,
                duration_ms=duration_ms
            )
            
            self.last_check_results[component_id] = health_check
            return health_check
            
        except Exception as e:
            logger.error(f"组件健康检查失败: {component_id} - {e}")
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            health_check = HealthCheck(
                component_id=component_id,
                check_name="component_health",
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {str(e)}",
                duration_ms=duration_ms,
                error=str(e)
            )
            
            self.last_check_results[component_id] = health_check
            return health_check
    
    def get_last_check_result(self, component_id: str) -> Optional[HealthCheck]:
        """获取最后一次检查结果"""
        return self.last_check_results.get(component_id)


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, metric_name: str, threshold: float, level: AlertLevel, 
                      comparison: str = "greater", component_id: str = "*"):
        """添加告警规则"""
        rule = {
            'metric_name': metric_name,
            'threshold': threshold,
            'level': level,
            'comparison': comparison,  # greater, less, equal
            'component_id': component_id
        }
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {metric_name} {comparison} {threshold}")
    
    def check_metric_alerts(self, component_id: str, metrics: List[HealthMetric]):
        """检查指标告警"""
        for metric in metrics:
            # 检查内置阈值
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                self._create_alert(
                    component_id=component_id,
                    level=AlertLevel.CRITICAL,
                    title=f"{metric.name} Critical",
                    message=f"{metric.name} is {metric.value}{metric.unit}, exceeds critical threshold {metric.threshold_critical}{metric.unit}",
                    metric_name=metric.name,
                    metric_value=metric.value,
                    threshold=metric.threshold_critical
                )
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                self._create_alert(
                    component_id=component_id,
                    level=AlertLevel.WARNING,
                    title=f"{metric.name} Warning",
                    message=f"{metric.name} is {metric.value}{metric.unit}, exceeds warning threshold {metric.threshold_warning}{metric.unit}",
                    metric_name=metric.name,
                    metric_value=metric.value,
                    threshold=metric.threshold_warning
                )
            
            # 检查自定义规则
            for rule in self.alert_rules:
                if (rule['component_id'] == "*" or rule['component_id'] == component_id) and \
                   rule['metric_name'] == metric.name:
                    
                    triggered = False
                    if rule['comparison'] == 'greater' and metric.value > rule['threshold']:
                        triggered = True
                    elif rule['comparison'] == 'less' and metric.value < rule['threshold']:
                        triggered = True
                    elif rule['comparison'] == 'equal' and metric.value == rule['threshold']:
                        triggered = True
                    
                    if triggered:
                        self._create_alert(
                            component_id=component_id,
                            level=rule['level'],
                            title=f"{metric.name} Alert",
                            message=f"{metric.name} is {metric.value}{metric.unit}, {rule['comparison']} threshold {rule['threshold']}",
                            metric_name=metric.name,
                            metric_value=metric.value,
                            threshold=rule['threshold']
                        )
    
    def _create_alert(self, component_id: str, level: AlertLevel, title: str, 
                     message: str, metric_name: str = None, metric_value: float = None, 
                     threshold: float = None):
        """创建告警"""
        alert_id = f"{component_id}_{metric_name}_{level.value}"
        
        # 检查是否已存在相同告警
        if alert_id in self.active_alerts:
            # 更新现有告警
            existing_alert = self.active_alerts[alert_id]
            existing_alert.message = message
            existing_alert.metric_value = metric_value
            existing_alert.timestamp = time.time()
        else:
            # 创建新告警
            alert = Alert(
                id=alert_id,
                component_id=component_id,
                level=level,
                title=title,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # 通知告警处理器
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"告警处理器执行失败: {e}")
            
            logger.warning(f"新告警: {title} - {message}")
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self.active_alerts[alert_id]
            
            logger.info(f"告警已解决: {alert.title}")
    
    def get_active_alerts(self, component_id: str = None, level: AlertLevel = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = list(self.active_alerts.values())
        
        if component_id:
            alerts = [alert for alert in alerts if alert.component_id == component_id]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return alerts


class HealthMonitor:
    """健康监控系统"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        
        # 组件
        self.metrics_collector = SystemMetricsCollector()
        self.component_checker = ComponentHealthChecker()
        self.alert_manager = AlertManager()
        
        # 监控任务
        self.monitor_task = None
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'healthy_components': 0,
            'warning_components': 0,
            'critical_components': 0,
            'active_alerts': 0,
            'last_check_time': 0
        }
        
        # 自动修复处理器
        self.recovery_handlers: Dict[str, Callable] = {}
        
        logger.info("健康监控系统初始化完成")
    
    async def start(self):
        """启动健康监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        # 添加默认告警规则
        self._setup_default_alert_rules()
        
        logger.info("健康监控系统已启动")
    
    async def stop(self):
        """停止健康监控"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("健康监控系统已停止")
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        # CPU使用率告警
        self.alert_manager.add_alert_rule(
            metric_name="cpu_usage_percent",
            threshold=90.0,
            level=AlertLevel.CRITICAL,
            comparison="greater"
        )
        
        # 内存使用率告警
        self.alert_manager.add_alert_rule(
            metric_name="memory_usage_percent",
            threshold=90.0,
            level=AlertLevel.CRITICAL,
            comparison="greater"
        )
        
        # 磁盘使用率告警
        self.alert_manager.add_alert_rule(
            metric_name="disk_usage_percent",
            threshold=90.0,
            level=AlertLevel.CRITICAL,
            comparison="greater"
        )
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康监控循环失败: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        try:
            start_time = time.time()
            
            # 收集系统指标
            system_metrics = self.metrics_collector.collect_system_metrics()
            
            # 检查系统指标告警
            self.alert_manager.check_metric_alerts("system", system_metrics)
            
            # 检查所有注册的组件
            component_results = {}
            for component_id in self.component_checker.health_checks.keys():
                health_check = await self.component_checker.check_component_health(component_id)
                component_results[component_id] = health_check
                
                # 检查组件指标告警
                if health_check.metrics:
                    self.alert_manager.check_metric_alerts(component_id, health_check.metrics)
                
                # 检查组件状态告警
                if health_check.status == HealthStatus.CRITICAL:
                    self.alert_manager._create_alert(
                        component_id=component_id,
                        level=AlertLevel.CRITICAL,
                        title=f"Component {component_id} Critical",
                        message=health_check.message
                    )
                elif health_check.status == HealthStatus.WARNING:
                    self.alert_manager._create_alert(
                        component_id=component_id,
                        level=AlertLevel.WARNING,
                        title=f"Component {component_id} Warning",
                        message=health_check.message
                    )
            
            # 更新统计信息
            self._update_stats(component_results)
            
            # 执行自动修复
            await self._perform_auto_recovery(component_results)
            
            self.stats['last_check_time'] = time.time()
            self.stats['total_checks'] += 1
            
            logger.debug(f"健康检查完成，耗时: {time.time() - start_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"执行健康检查失败: {e}")
    
    def _update_stats(self, component_results: Dict[str, HealthCheck]):
        """更新统计信息"""
        healthy_count = 0
        warning_count = 0
        critical_count = 0
        
        for health_check in component_results.values():
            if health_check.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif health_check.status == HealthStatus.WARNING:
                warning_count += 1
            elif health_check.status == HealthStatus.CRITICAL:
                critical_count += 1
        
        self.stats.update({
            'healthy_components': healthy_count,
            'warning_components': warning_count,
            'critical_components': critical_count,
            'active_alerts': len(self.alert_manager.active_alerts)
        })
    
    async def _perform_auto_recovery(self, component_results: Dict[str, HealthCheck]):
        """执行自动修复"""
        for component_id, health_check in component_results.items():
            if health_check.status == HealthStatus.CRITICAL:
                if component_id in self.recovery_handlers:
                    try:
                        logger.info(f"执行自动修复: {component_id}")
                        recovery_handler = self.recovery_handlers[component_id]
                        
                        if asyncio.iscoroutinefunction(recovery_handler):
                            await recovery_handler(health_check)
                        else:
                            recovery_handler(health_check)
                            
                    except Exception as e:
                        logger.error(f"自动修复失败: {component_id} - {e}")
    
    def register_component_health_check(self, component_id: str, check_func: Callable):
        """注册组件健康检查"""
        self.component_checker.register_health_check(component_id, check_func)
    
    def register_recovery_handler(self, component_id: str, handler: Callable):
        """注册自动修复处理器"""
        self.recovery_handlers[component_id] = handler
        logger.info(f"注册自动修复处理器: {component_id}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_manager.add_alert_handler(handler)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """获取系统健康摘要"""
        return {
            'overall_status': self._get_overall_status(),
            'stats': self.stats.copy(),
            'active_alerts': len(self.alert_manager.active_alerts),
            'critical_alerts': len([
                alert for alert in self.alert_manager.active_alerts.values()
                if alert.level == AlertLevel.CRITICAL
            ]),
            'last_check_time': self.stats['last_check_time'],
            'monitoring_enabled': self.running
        }
    
    def _get_overall_status(self) -> str:
        """获取整体状态"""
        if self.stats['critical_components'] > 0:
            return "critical"
        elif self.stats['warning_components'] > 0:
            return "warning"
        elif self.stats['healthy_components'] > 0:
            return "healthy"
        else:
            return "unknown"
    
    def get_component_health(self, component_id: str) -> Optional[HealthCheck]:
        """获取组件健康状态"""
        return self.component_checker.get_last_check_result(component_id)
    
    def get_active_alerts(self, component_id: str = None) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts(component_id)
    
    def get_metric_history(self, metric_name: str, duration_seconds: int = 300) -> List[Dict[str, Any]]:
        """获取指标历史"""
        return self.metrics_collector.get_metric_history(metric_name, duration_seconds)


# 全局健康监控实例
health_monitor = HealthMonitor()
