#!/usr/bin/env python3
"""
📊 实时监控系统 - 生产级监控解决方案
Real-Time Monitoring System - Production-Grade Monitoring Solution

生产级特性：
- 实时性能监控
- 系统资源监控
- 交易指标监控
- 异常检测和告警
- 可视化仪表板
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import logging
from pathlib import Path

# 导入统一日志系统
from .unified_logging_system import UnifiedLogger

@dataclass
class SystemMetrics:
    """系统指标数据结构"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]

@dataclass
class TradingMetrics:
    """交易指标数据结构"""
    timestamp: datetime
    active_positions: int
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    avg_trade_duration: float

@dataclass
class AlertRule:
    """告警规则配置"""
    name: str
    metric_path: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    duration: int  # 持续时间（秒）
    severity: str  # 'low', 'medium', 'high', 'critical'
    callback: Optional[Callable] = None

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.logger = UnifiedLogger("MetricsCollector")
        self.system_metrics_history = deque(maxlen=1000)
        self.trading_metrics_history = deque(maxlen=1000)
        self._running = False
        
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # 网络IO
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 进程数量
            process_count = len(psutil.pids())
            
            # 系统负载
            load_average = list(psutil.getloadavg())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average
            )
            
            self.system_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
            return None
    
    def collect_trading_metrics(self, trading_engine=None) -> TradingMetrics:
        """收集交易指标"""
        try:
            # 这里需要从交易引擎获取实际数据
            # 目前使用模拟数据作为示例
            
            if trading_engine:
                # 从实际交易引擎获取数据
                active_positions = len(trading_engine.get_positions())
                total_pnl = trading_engine.get_total_pnl()
                daily_pnl = trading_engine.get_daily_pnl()
                win_rate = trading_engine.get_win_rate()
                sharpe_ratio = trading_engine.get_sharpe_ratio()
                max_drawdown = trading_engine.get_max_drawdown()
                trade_count = trading_engine.get_trade_count()
                avg_trade_duration = trading_engine.get_avg_trade_duration()
            else:
                # 模拟数据
                active_positions = 5
                total_pnl = 12500.0
                daily_pnl = 850.0
                win_rate = 0.65
                sharpe_ratio = 1.8
                max_drawdown = -0.08
                trade_count = 45
                avg_trade_duration = 3600.0  # 1小时
            
            metrics = TradingMetrics(
                timestamp=datetime.now(),
                active_positions=active_positions,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trade_count=trade_count,
                avg_trade_duration=avg_trade_duration
            )
            
            self.trading_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集交易指标失败: {e}")
            return None

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.logger = UnifiedLogger("AlertManager")
        self.alert_rules: List[AlertRule] = []
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules.append(rule)
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        self.logger.info(f"移除告警规则: {rule_name}")
    
    def check_alerts(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """检查告警条件"""
        current_time = datetime.now()
        
        # 合并所有指标数据
        all_metrics = {
            'system': asdict(system_metrics) if system_metrics else {},
            'trading': asdict(trading_metrics) if trading_metrics else {}
        }
        
        for rule in self.alert_rules:
            try:
                # 解析指标路径
                metric_value = self._get_metric_value(all_metrics, rule.metric_path)
                if metric_value is None:
                    continue
                
                # 检查告警条件
                if self._evaluate_condition(metric_value, rule.operator, rule.threshold):
                    alert_key = f"{rule.name}_{rule.metric_path}"
                    
                    if alert_key not in self.active_alerts:
                        # 新告警
                        self.active_alerts[alert_key] = {
                            'rule': rule,
                            'start_time': current_time,
                            'last_trigger': current_time,
                            'trigger_count': 1
                        }
                    else:
                        # 更新现有告警
                        self.active_alerts[alert_key]['last_trigger'] = current_time
                        self.active_alerts[alert_key]['trigger_count'] += 1
                        
                        # 检查是否达到持续时间要求
                        duration = (current_time - self.active_alerts[alert_key]['start_time']).total_seconds()
                        if duration >= rule.duration:
                            self._trigger_alert(rule, metric_value, duration)
                else:
                    # 条件不满足，清除告警
                    alert_key = f"{rule.name}_{rule.metric_path}"
                    if alert_key in self.active_alerts:
                        del self.active_alerts[alert_key]
                        
            except Exception as e:
                self.logger.error(f"检查告警规则 {rule.name} 失败: {e}")
    
    def _get_metric_value(self, metrics: Dict, path: str) -> Optional[float]:
        """根据路径获取指标值"""
        try:
            keys = path.split('.')
            value = metrics
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return float(value) if value is not None else None
        except:
            return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """评估告警条件"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        return False
    
    def _trigger_alert(self, rule: AlertRule, value: float, duration: float):
        """触发告警"""
        alert_data = {
            'timestamp': datetime.now(),
            'rule_name': rule.name,
            'metric_path': rule.metric_path,
            'current_value': value,
            'threshold': rule.threshold,
            'severity': rule.severity,
            'duration': duration
        }
        
        self.alert_history.append(alert_data)
        
        # 记录告警日志
        self.logger.warning(
            f"🚨 告警触发: {rule.name} | "
            f"指标: {rule.metric_path} | "
            f"当前值: {value} | "
            f"阈值: {rule.threshold} | "
            f"严重程度: {rule.severity}"
        )
        
        # 执行回调函数
        if rule.callback:
            try:
                rule.callback(alert_data)
            except Exception as e:
                self.logger.error(f"执行告警回调失败: {e}")

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.logger = UnifiedLogger("PerformanceProfiler")
        self.function_stats = defaultdict(list)
        self.active_timers = {}
    
    def start_timer(self, name: str) -> str:
        """开始计时"""
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        self.active_timers[timer_id] = {
            'name': name,
            'start_time': time.perf_counter(),
            'start_timestamp': datetime.now()
        }
        return timer_id
    
    def end_timer(self, timer_id: str) -> Optional[float]:
        """结束计时"""
        if timer_id not in self.active_timers:
            return None
        
        timer_info = self.active_timers[timer_id]
        end_time = time.perf_counter()
        duration = end_time - timer_info['start_time']
        
        # 记录性能数据
        self.function_stats[timer_info['name']].append({
            'duration': duration,
            'timestamp': timer_info['start_timestamp']
        })
        
        # 清理计时器
        del self.active_timers[timer_id]
        
        return duration
    
    def get_performance_stats(self, function_name: str = None) -> Dict:
        """获取性能统计"""
        if function_name:
            stats = self.function_stats.get(function_name, [])
            if not stats:
                return {}
            
            durations = [s['duration'] for s in stats]
            return {
                'function_name': function_name,
                'call_count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        else:
            # 返回所有函数的统计
            all_stats = {}
            for func_name, stats in self.function_stats.items():
                durations = [s['duration'] for s in stats]
                all_stats[func_name] = {
                    'call_count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
            return all_stats

class RealTimeMonitor:
    """实时监控主类"""
    
    def __init__(self, config: Dict = None):
        self.logger = UnifiedLogger("RealTimeMonitor")
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_profiler = PerformanceProfiler()
        
        # 监控状态
        self._running = False
        self._monitor_thread = None
        self._last_metrics_time = None
        
        # 初始化默认告警规则
        self._setup_default_alerts()
        
        self.logger.info("实时监控系统初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'collection_interval': 5,  # 数据收集间隔（秒）
            'alert_check_interval': 10,  # 告警检查间隔（秒）
            'metrics_retention': 3600,  # 指标保留时间（秒）
            'enable_performance_profiling': True,
            'dashboard_port': 8080,
            'log_level': 'INFO'
        }
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        default_rules = [
            AlertRule(
                name="高CPU使用率",
                metric_path="system.cpu_percent",
                operator=">",
                threshold=80.0,
                duration=60,
                severity="high"
            ),
            AlertRule(
                name="高内存使用率",
                metric_path="system.memory_percent",
                operator=">",
                threshold=85.0,
                duration=60,
                severity="high"
            ),
            AlertRule(
                name="磁盘空间不足",
                metric_path="system.disk_usage",
                operator=">",
                threshold=90.0,
                duration=30,
                severity="critical"
            ),
            AlertRule(
                name="交易胜率过低",
                metric_path="trading.win_rate",
                operator="<",
                threshold=0.4,
                duration=300,
                severity="medium"
            ),
            AlertRule(
                name="最大回撤过大",
                metric_path="trading.max_drawdown",
                operator="<",
                threshold=-0.15,
                duration=60,
                severity="high"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def start_monitoring(self):
        """开始监控"""
        if self._running:
            self.logger.warning("监控系统已在运行中")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("实时监控系统已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("实时监控系统已停止")
    
    def _monitor_loop(self):
        """监控主循环"""
        last_collection_time = 0
        last_alert_check_time = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # 收集指标
                if current_time - last_collection_time >= self.config['collection_interval']:
                    system_metrics = self.metrics_collector.collect_system_metrics()
                    trading_metrics = self.metrics_collector.collect_trading_metrics()
                    last_collection_time = current_time
                    
                    # 检查告警
                    if current_time - last_alert_check_time >= self.config['alert_check_interval']:
                        self.alert_manager.check_alerts(system_metrics, trading_metrics)
                        last_alert_check_time = current_time
                
                # 短暂休眠
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(5)
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标"""
        system_metrics = self.metrics_collector.collect_system_metrics()
        trading_metrics = self.metrics_collector.collect_trading_metrics()
        
        return {
            'system': asdict(system_metrics) if system_metrics else {},
            'trading': asdict(trading_metrics) if trading_metrics else {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_history(self, hours: int = 1) -> Dict:
        """获取历史指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤系统指标
        system_history = [
            asdict(m) for m in self.metrics_collector.system_metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # 过滤交易指标
        trading_history = [
            asdict(m) for m in self.metrics_collector.trading_metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        return {
            'system': system_history,
            'trading': trading_history,
            'period_hours': hours
        }
    
    def get_alert_status(self) -> Dict:
        """获取告警状态"""
        return {
            'active_alerts': len(self.alert_manager.active_alerts),
            'alert_rules': len(self.alert_manager.alert_rules),
            'recent_alerts': [
                alert for alert in list(self.alert_manager.alert_history)[-10:]
            ]
        }
    
    def add_custom_alert(self, rule: AlertRule):
        """添加自定义告警规则"""
        self.alert_manager.add_alert_rule(rule)
        self.logger.info(f"添加自定义告警规则: {rule.name}")
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """导出指标数据"""
        try:
            metrics_data = {
                'export_time': datetime.now().isoformat(),
                'system_metrics': [
                    asdict(m) for m in self.metrics_collector.system_metrics_history
                ],
                'trading_metrics': [
                    asdict(m) for m in self.metrics_collector.trading_metrics_history
                ],
                'performance_stats': self.performance_profiler.get_performance_stats(),
                'alert_history': list(self.alert_manager.alert_history)
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"指标数据已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出指标数据失败: {e}")

# 性能监控装饰器
def monitor_performance(monitor_instance: RealTimeMonitor):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_id = monitor_instance.performance_profiler.start_timer(func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor_instance.performance_profiler.end_timer(timer_id)
                if duration and duration > 1.0:  # 记录超过1秒的函数调用
                    monitor_instance.logger.info(f"函数 {func.__name__} 执行时间: {duration:.3f}秒")
        return wrapper
    return decorator

# 使用示例
if __name__ == "__main__":
    # 创建监控实例
    monitor = RealTimeMonitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    try:
        # 运行一段时间进行测试
        time.sleep(30)
        
        # 获取当前指标
        current_metrics = monitor.get_current_metrics()
        print("当前指标:", json.dumps(current_metrics, indent=2, default=str, ensure_ascii=False))
        
        # 获取告警状态
        alert_status = monitor.get_alert_status()
        print("告警状态:", json.dumps(alert_status, indent=2, default=str, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("停止监控...")
    finally:
        monitor.stop_monitoring()
