#!/usr/bin/env python3
"""
📈 性能指标收集器 - 生产级性能监控
Performance Metrics Collector - Production-Grade Performance Monitoring

生产级特性：
- 多维度性能指标收集
- 实时性能基线建立
- 性能趋势分析
- 异常检测和预警
- 性能报告生成
"""

import time
import psutil
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import statistics
import numpy as np
from pathlib import Path

from .unified_logging_system import UnifiedLogger

@dataclass
class PerformanceMetric:
    """性能指标数据结构"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None
    source: str = "system"

@dataclass
class PerformanceBaseline:
    """性能基线数据结构"""
    metric_name: str
    avg_value: float
    min_value: float
    max_value: float
    std_deviation: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    created_at: datetime
    updated_at: datetime

class SystemMetricsCollector:
    """系统指标收集器"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "SystemMetricsCollector")
        self.metrics_history = deque(maxlen=10000)
        self._collection_interval = 1.0
        self._running = False
        self._collection_thread = None
    
    def start_collection(self):
        """开始收集系统指标"""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        self.logger.info("系统指标收集已启动")
    
    def stop_collection(self):
        """停止收集系统指标"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        self.logger.info("系统指标收集已停止")
    
    def _collection_loop(self):
        """指标收集循环"""
        while self._running:
            try:
                self._collect_cpu_metrics()
                self._collect_memory_metrics()
                self._collect_disk_metrics()
                self._collect_network_metrics()
                self._collect_process_metrics()
                
                time.sleep(self._collection_interval)
                
            except Exception as e:
                self.logger.error(f"收集系统指标异常: {e}")
                time.sleep(5)
    
    def _collect_cpu_metrics(self):
        """收集CPU指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            self._add_metric("cpu.usage_percent", cpu_percent, "%")
            
            # CPU核心使用率
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            for i, usage in enumerate(cpu_per_core):
                self._add_metric("cpu.core_usage_percent", usage, "%", {"core": str(i)})
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self._add_metric("cpu.frequency_mhz", cpu_freq.current, "MHz")
            
            # 系统负载
            load_avg = psutil.getloadavg()
            self._add_metric("cpu.load_1min", load_avg[0], "")
            self._add_metric("cpu.load_5min", load_avg[1], "")
            self._add_metric("cpu.load_15min", load_avg[2], "")
            
        except Exception as e:
            self.logger.error(f"收集CPU指标失败: {e}")
    
    def _collect_memory_metrics(self):
        """收集内存指标"""
        try:
            # 虚拟内存
            memory = psutil.virtual_memory()
            self._add_metric("memory.total_bytes", memory.total, "bytes")
            self._add_metric("memory.available_bytes", memory.available, "bytes")
            self._add_metric("memory.used_bytes", memory.used, "bytes")
            self._add_metric("memory.usage_percent", memory.percent, "%")
            
            # 交换内存
            swap = psutil.swap_memory()
            self._add_metric("swap.total_bytes", swap.total, "bytes")
            self._add_metric("swap.used_bytes", swap.used, "bytes")
            self._add_metric("swap.usage_percent", swap.percent, "%")
            
        except Exception as e:
            self.logger.error(f"收集内存指标失败: {e}")
    
    def _collect_disk_metrics(self):
        """收集磁盘指标"""
        try:
            # 磁盘使用情况
            disk_usage = psutil.disk_usage('/')
            self._add_metric("disk.total_bytes", disk_usage.total, "bytes")
            self._add_metric("disk.used_bytes", disk_usage.used, "bytes")
            self._add_metric("disk.free_bytes", disk_usage.free, "bytes")
            self._add_metric("disk.usage_percent", (disk_usage.used / disk_usage.total) * 100, "%")
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._add_metric("disk.read_bytes", disk_io.read_bytes, "bytes")
                self._add_metric("disk.write_bytes", disk_io.write_bytes, "bytes")
                self._add_metric("disk.read_count", disk_io.read_count, "count")
                self._add_metric("disk.write_count", disk_io.write_count, "count")
                self._add_metric("disk.read_time_ms", disk_io.read_time, "ms")
                self._add_metric("disk.write_time_ms", disk_io.write_time, "ms")
            
        except Exception as e:
            self.logger.error(f"收集磁盘指标失败: {e}")
    
    def _collect_network_metrics(self):
        """收集网络指标"""
        try:
            # 网络IO
            network_io = psutil.net_io_counters()
            if network_io:
                self._add_metric("network.bytes_sent", network_io.bytes_sent, "bytes")
                self._add_metric("network.bytes_recv", network_io.bytes_recv, "bytes")
                self._add_metric("network.packets_sent", network_io.packets_sent, "count")
                self._add_metric("network.packets_recv", network_io.packets_recv, "count")
                self._add_metric("network.errin", network_io.errin, "count")
                self._add_metric("network.errout", network_io.errout, "count")
                self._add_metric("network.dropin", network_io.dropin, "count")
                self._add_metric("network.dropout", network_io.dropout, "count")
            
            # 网络连接数
            connections = psutil.net_connections()
            connection_states = defaultdict(int)
            for conn in connections:
                if conn.status:
                    connection_states[conn.status] += 1
            
            for state, count in connection_states.items():
                self._add_metric("network.connections", count, "count", {"state": state})
            
        except Exception as e:
            self.logger.error(f"收集网络指标失败: {e}")
    
    def _collect_process_metrics(self):
        """收集进程指标"""
        try:
            # 进程数量
            process_count = len(psutil.pids())
            self._add_metric("process.total_count", process_count, "count")
            
            # 当前进程指标
            current_process = psutil.Process()
            
            # 内存使用
            memory_info = current_process.memory_info()
            self._add_metric("process.memory_rss_bytes", memory_info.rss, "bytes")
            self._add_metric("process.memory_vms_bytes", memory_info.vms, "bytes")
            
            # CPU使用
            cpu_percent = current_process.cpu_percent()
            self._add_metric("process.cpu_percent", cpu_percent, "%")
            
            # 文件描述符
            try:
                num_fds = current_process.num_fds()
                self._add_metric("process.file_descriptors", num_fds, "count")
            except:
                pass  # Windows不支持
            
            # 线程数
            num_threads = current_process.num_threads()
            self._add_metric("process.thread_count", num_threads, "count")
            
        except Exception as e:
            self.logger.error(f"收集进程指标失败: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """添加指标"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            source="system"
        )
        self.metrics_history.append(metric)

class ApplicationMetricsCollector:
    """应用指标收集器"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "ApplicationMetricsCollector")
        self.metrics_history = deque(maxlen=10000)
        self.function_timers = {}
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
    
    def start_timer(self, name: str) -> str:
        """开始计时器"""
        timer_id = f"{name}_{int(time.time() * 1000000)}"
        self.function_timers[timer_id] = {
            'name': name,
            'start_time': time.perf_counter(),
            'start_timestamp': datetime.now()
        }
        return timer_id
    
    def end_timer(self, timer_id: str) -> Optional[float]:
        """结束计时器"""
        if timer_id not in self.function_timers:
            return None
        
        timer_info = self.function_timers[timer_id]
        end_time = time.perf_counter()
        duration = end_time - timer_info['start_time']
        
        # 记录指标
        self._add_metric(f"function.{timer_info['name']}.duration", duration * 1000, "ms")
        
        # 添加到直方图
        self.histograms[f"function.{timer_info['name']}.duration"].append(duration * 1000)
        
        # 清理计时器
        del self.function_timers[timer_id]
        
        return duration
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """增加计数器"""
        self.counters[name] += value
        self._add_metric(f"counter.{name}", self.counters[name], "count", tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表"""
        self.gauges[name] = value
        self._add_metric(f"gauge.{name}", value, "value", tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录直方图"""
        self.histograms[name].append(value)
        self._add_metric(f"histogram.{name}", value, "value", tags)
    
    def _add_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """添加指标"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            source="application"
        )
        self.metrics_history.append(metric)

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "PerformanceAnalyzer")
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.anomaly_threshold = 2.0  # 标准差倍数
    
    def create_baseline(self, metrics: List[PerformanceMetric], metric_name: str) -> PerformanceBaseline:
        """创建性能基线"""
        try:
            # 过滤指定指标
            filtered_metrics = [m for m in metrics if m.metric_name == metric_name]
            
            if len(filtered_metrics) < 10:
                self.logger.warning(f"指标 {metric_name} 样本数量不足，无法创建基线")
                return None
            
            values = [m.value for m in filtered_metrics]
            
            baseline = PerformanceBaseline(
                metric_name=metric_name,
                avg_value=statistics.mean(values),
                min_value=min(values),
                max_value=max(values),
                std_deviation=statistics.stdev(values) if len(values) > 1 else 0,
                percentile_95=np.percentile(values, 95),
                percentile_99=np.percentile(values, 99),
                sample_count=len(values),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.baselines[metric_name] = baseline
            self.logger.info(f"创建性能基线: {metric_name}")
            return baseline
            
        except Exception as e:
            self.logger.error(f"创建性能基线失败: {e}")
            return None
    
    def update_baseline(self, metrics: List[PerformanceMetric], metric_name: str):
        """更新性能基线"""
        if metric_name not in self.baselines:
            return self.create_baseline(metrics, metric_name)
        
        try:
            # 获取最近的指标
            recent_metrics = [
                m for m in metrics 
                if m.metric_name == metric_name and 
                (datetime.now() - m.timestamp).total_seconds() < 3600  # 最近1小时
            ]
            
            if len(recent_metrics) < 10:
                return
            
            values = [m.value for m in recent_metrics]
            baseline = self.baselines[metric_name]
            
            # 更新基线
            baseline.avg_value = statistics.mean(values)
            baseline.min_value = min(values)
            baseline.max_value = max(values)
            baseline.std_deviation = statistics.stdev(values) if len(values) > 1 else 0
            baseline.percentile_95 = np.percentile(values, 95)
            baseline.percentile_99 = np.percentile(values, 99)
            baseline.sample_count = len(values)
            baseline.updated_at = datetime.now()
            
            self.logger.debug(f"更新性能基线: {metric_name}")
            
        except Exception as e:
            self.logger.error(f"更新性能基线失败: {e}")
    
    def detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[Dict]:
        """检测性能异常"""
        anomalies = []
        
        try:
            # 按指标名称分组
            metrics_by_name = defaultdict(list)
            for metric in metrics:
                metrics_by_name[metric.metric_name].append(metric)
            
            for metric_name, metric_list in metrics_by_name.items():
                if metric_name not in self.baselines:
                    continue
                
                baseline = self.baselines[metric_name]
                
                # 检查最近的指标值
                for metric in metric_list[-10:]:  # 检查最近10个值
                    deviation = abs(metric.value - baseline.avg_value)
                    threshold = baseline.std_deviation * self.anomaly_threshold
                    
                    if deviation > threshold and baseline.std_deviation > 0:
                        anomaly = {
                            'metric_name': metric_name,
                            'timestamp': metric.timestamp,
                            'value': metric.value,
                            'baseline_avg': baseline.avg_value,
                            'deviation': deviation,
                            'threshold': threshold,
                            'severity': self._calculate_anomaly_severity(deviation, threshold)
                        }
                        anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"检测性能异常失败: {e}")
        
        return anomalies
    
    def _calculate_anomaly_severity(self, deviation: float, threshold: float) -> str:
        """计算异常严重程度"""
        ratio = deviation / threshold
        
        if ratio >= 5:
            return "critical"
        elif ratio >= 3:
            return "high"
        elif ratio >= 2:
            return "medium"
        else:
            return "low"
    
    def generate_performance_report(self, metrics: List[PerformanceMetric], hours: int = 24) -> Dict:
        """生成性能报告"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 过滤时间范围内的指标
            filtered_metrics = [
                m for m in metrics 
                if m.timestamp >= cutoff_time
            ]
            
            # 按指标名称分组统计
            metrics_stats = {}
            metrics_by_name = defaultdict(list)
            
            for metric in filtered_metrics:
                metrics_by_name[metric.metric_name].append(metric.value)
            
            for metric_name, values in metrics_by_name.items():
                if not values:
                    continue
                
                metrics_stats[metric_name] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
            
            # 检测异常
            anomalies = self.detect_anomalies(filtered_metrics)
            
            # 生成报告
            report = {
                'report_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'total_metrics': len(filtered_metrics),
                'unique_metric_names': len(metrics_stats),
                'metrics_statistics': metrics_stats,
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies,
                'baselines': {name: asdict(baseline) for name, baseline in self.baselines.items()},
                'top_metrics_by_variance': self._get_top_metrics_by_variance(metrics_stats)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")
            return {}
    
    def _get_top_metrics_by_variance(self, metrics_stats: Dict, top_n: int = 10) -> List[Dict]:
        """获取方差最大的指标"""
        try:
            variance_metrics = []
            
            for metric_name, stats in metrics_stats.items():
                if stats['std'] > 0:
                    # 计算变异系数 (CV = std / mean)
                    cv = stats['std'] / abs(stats['avg']) if stats['avg'] != 0 else float('inf')
                    variance_metrics.append({
                        'metric_name': metric_name,
                        'coefficient_of_variation': cv,
                        'std_deviation': stats['std'],
                        'mean': stats['avg']
                    })
            
            # 按变异系数排序
            variance_metrics.sort(key=lambda x: x['coefficient_of_variation'], reverse=True)
            return variance_metrics[:top_n]
            
        except Exception as e:
            self.logger.error(f"计算方差指标失败: {e}")
            return []

class PerformanceMetricsManager:
    """性能指标管理器"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "PerformanceMetricsManager")
        
        # 初始化组件
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        self.analyzer = PerformanceAnalyzer()
        
        # 配置
        self.baseline_update_interval = 3600  # 1小时更新一次基线
        self.anomaly_check_interval = 300     # 5分钟检查一次异常
        
        # 状态
        self._running = False
        self._analysis_thread = None
        
        self.logger.info("性能指标管理器初始化完成")
    
    def start(self):
        """启动性能监控"""
        if self._running:
            return
        
        self._running = True
        
        # 启动系统指标收集
        self.system_collector.start_collection()
        
        # 启动分析线程
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        
        self.logger.info("性能监控已启动")
    
    def stop(self):
        """停止性能监控"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止系统指标收集
        self.system_collector.stop_collection()
        
        # 停止分析线程
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        
        self.logger.info("性能监控已停止")
    
    def _analysis_loop(self):
        """分析循环"""
        last_baseline_update = 0
        last_anomaly_check = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # 获取所有指标
                all_metrics = list(self.system_collector.metrics_history) + list(self.app_collector.metrics_history)
                
                # 更新基线
                if current_time - last_baseline_update >= self.baseline_update_interval:
                    self._update_all_baselines(all_metrics)
                    last_baseline_update = current_time
                
                # 检查异常
                if current_time - last_anomaly_check >= self.anomaly_check_interval:
                    anomalies = self.analyzer.detect_anomalies(all_metrics)
                    if anomalies:
                        self._handle_anomalies(anomalies)
                    last_anomaly_check = current_time
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self.logger.error(f"性能分析循环异常: {e}")
                time.sleep(60)
    
    def _update_all_baselines(self, metrics: List[PerformanceMetric]):
        """更新所有基线"""
        try:
            # 获取所有唯一的指标名称
            metric_names = set(m.metric_name for m in metrics)
            
            for metric_name in metric_names:
                self.analyzer.update_baseline(metrics, metric_name)
            
            self.logger.info(f"更新了 {len(metric_names)} 个指标的基线")
            
        except Exception as e:
            self.logger.error(f"更新基线失败: {e}")
    
    def _handle_anomalies(self, anomalies: List[Dict]):
        """处理异常"""
        for anomaly in anomalies:
            self.logger.warning(
                f"检测到性能异常: {anomaly['metric_name']} | "
                f"当前值: {anomaly['value']:.2f} | "
                f"基线: {anomaly['baseline_avg']:.2f} | "
                f"偏差: {anomaly['deviation']:.2f} | "
                f"严重程度: {anomaly['severity']}"
            )
    
    def get_current_metrics(self) -> Dict:
        """获取当前指标"""
        system_metrics = list(self.system_collector.metrics_history)[-100:]  # 最近100个
        app_metrics = list(self.app_collector.metrics_history)[-100:]
        
        return {
            'system_metrics': [asdict(m) for m in system_metrics],
            'application_metrics': [asdict(m) for m in app_metrics],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self, hours: int = 24) -> Dict:
        """生成性能报告"""
        all_metrics = list(self.system_collector.metrics_history) + list(self.app_collector.metrics_history)
        return self.analyzer.generate_performance_report(all_metrics, hours)
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """导出指标数据"""
        try:
            report = self.generate_report(hours)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"性能指标已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出性能指标失败: {e}")

# 性能监控装饰器
def monitor_performance(metrics_manager: PerformanceMetricsManager):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_id = metrics_manager.app_collector.start_timer(func.__name__)
            try:
                result = func(*args, **kwargs)
                metrics_manager.app_collector.increment_counter(f"{func.__name__}.success_count")
                return result
            except Exception as e:
                metrics_manager.app_collector.increment_counter(f"{func.__name__}.error_count")
                raise
            finally:
                duration = metrics_manager.app_collector.end_timer(timer_id)
                if duration and duration > 1.0:  # 记录超过1秒的函数调用
                    metrics_manager.logger.info(f"函数 {func.__name__} 执行时间: {duration:.3f}秒")
        return wrapper
    return decorator

# 使用示例
if __name__ == "__main__":
    # 创建性能指标管理器
    metrics_manager = PerformanceMetricsManager()
    
    # 启动监控
    metrics_manager.start()
    
    try:
        # 运行一段时间进行测试
        time.sleep(60)
        
        # 生成报告
        report = metrics_manager.generate_report(hours=1)
        print("性能报告:", json.dumps(report, indent=2, default=str, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("停止监控...")
    finally:
        metrics_manager.stop()
