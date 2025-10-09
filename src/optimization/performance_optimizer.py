#!/usr/bin/env python3
"""
⚡ 性能优化器 - 生产级系统性能优化
Performance Optimizer - Production-Grade System Performance Optimization

生产级特性：
- 自动性能分析
- 智能资源调优
- 缓存策略优化
- 数据库查询优化
- 系统瓶颈识别
"""

import psutil
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import gc

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float

@dataclass
class OptimizationRecommendation:
    """优化建议"""
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    impact: str
    implementation: str
    estimated_improvement: float

class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "PerformanceOptimizer")
        
        # 性能数据收集
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread = None
        
        # 优化配置
        self.optimization_config = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'response_time_threshold': 2.0,
            'error_rate_threshold': 0.05,
            'optimization_interval': 300  # 5分钟
        }
        
        # 缓存管理
        self.cache_stats = defaultdict(dict)
        self.query_stats = defaultdict(list)
        
        self.logger.info("性能优化器初始化完成")
    
    def start_monitoring(self):
        """启动性能监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self._monitoring:
            try:
                # 收集性能指标
                metrics = self._collect_performance_metrics()
                self.metrics_history.append(metrics)
                
                # 分析性能问题
                issues = self._analyze_performance_issues(metrics)
                
                # 生成优化建议
                if issues:
                    recommendations = self._generate_optimization_recommendations(issues)
                    
                    # 自动应用低风险优化
                    self._apply_automatic_optimizations(recommendations)
                
                time.sleep(60)  # 每分钟收集一次
                
            except Exception as e:
                self.logger.error(f"性能监控循环异常: {e}")
                time.sleep(60)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
            
            # 网络I/O
            network_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': network_io.bytes_sent if network_io else 0,
                'bytes_recv': network_io.bytes_recv if network_io else 0,
                'packets_sent': network_io.packets_sent if network_io else 0,
                'packets_recv': network_io.packets_recv if network_io else 0
            }
            
            # 模拟应用性能指标
            response_time = self._calculate_average_response_time()
            throughput = self._calculate_throughput()
            error_rate = self._calculate_error_rate()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_metrics,
                network_io=network_metrics,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate
            )
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            return None
    
    def _calculate_average_response_time(self) -> float:
        """计算平均响应时间"""
        # 这里应该从实际的应用监控中获取数据
        # 现在返回模拟数据
        if len(self.metrics_history) > 0:
            recent_metrics = list(self.metrics_history)[-10:]
            response_times = [m.response_time for m in recent_metrics if m.response_time > 0]
            if response_times:
                return statistics.mean(response_times)
        
        return 0.5  # 默认响应时间
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量"""
        # 这里应该从实际的应用监控中获取数据
        return 1000.0  # 默认吞吐量 (requests/second)
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        # 这里应该从实际的应用监控中获取数据
        return 0.01  # 默认错误率 1%
    
    def _analyze_performance_issues(self, metrics: PerformanceMetrics) -> List[str]:
        """分析性能问题"""
        issues = []
        
        if not metrics:
            return issues
        
        # CPU使用率过高
        if metrics.cpu_usage > self.optimization_config['cpu_threshold']:
            issues.append('high_cpu_usage')
            self.logger.warning(f"CPU使用率过高: {metrics.cpu_usage:.1f}%")
        
        # 内存使用率过高
        if metrics.memory_usage > self.optimization_config['memory_threshold']:
            issues.append('high_memory_usage')
            self.logger.warning(f"内存使用率过高: {metrics.memory_usage:.1f}%")
        
        # 响应时间过长
        if metrics.response_time > self.optimization_config['response_time_threshold']:
            issues.append('slow_response_time')
            self.logger.warning(f"响应时间过长: {metrics.response_time:.2f}s")
        
        # 错误率过高
        if metrics.error_rate > self.optimization_config['error_rate_threshold']:
            issues.append('high_error_rate')
            self.logger.warning(f"错误率过高: {metrics.error_rate:.2%}")
        
        # 分析趋势
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            # CPU使用率上升趋势
            cpu_trend = [m.cpu_usage for m in recent_metrics]
            if self._is_increasing_trend(cpu_trend):
                issues.append('cpu_usage_trend')
            
            # 内存使用率上升趋势
            memory_trend = [m.memory_usage for m in recent_metrics]
            if self._is_increasing_trend(memory_trend):
                issues.append('memory_usage_trend')
        
        return issues
    
    def _is_increasing_trend(self, values: List[float], threshold: float = 0.7) -> bool:
        """检查是否为上升趋势"""
        if len(values) < 5:
            return False
        
        # 计算线性回归斜率
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        return slope > threshold
    
    def _generate_optimization_recommendations(self, issues: List[str]) -> List[OptimizationRecommendation]:
        """生成优化建议"""
        recommendations = []
        
        for issue in issues:
            if issue == 'high_cpu_usage':
                recommendations.extend([
                    OptimizationRecommendation(
                        category='CPU优化',
                        priority='high',
                        description='启用CPU密集型任务的异步处理',
                        impact='减少CPU阻塞，提高并发处理能力',
                        implementation='使用asyncio或多进程处理',
                        estimated_improvement=20.0
                    ),
                    OptimizationRecommendation(
                        category='CPU优化',
                        priority='medium',
                        description='优化算法复杂度',
                        impact='减少计算开销',
                        implementation='代码重构和算法优化',
                        estimated_improvement=15.0
                    )
                ])
            
            elif issue == 'high_memory_usage':
                recommendations.extend([
                    OptimizationRecommendation(
                        category='内存优化',
                        priority='high',
                        description='启用内存缓存清理',
                        impact='释放未使用的内存',
                        implementation='定期执行垃圾回收',
                        estimated_improvement=25.0
                    ),
                    OptimizationRecommendation(
                        category='内存优化',
                        priority='medium',
                        description='优化数据结构',
                        impact='减少内存占用',
                        implementation='使用更高效的数据结构',
                        estimated_improvement=18.0
                    )
                ])
            
            elif issue == 'slow_response_time':
                recommendations.extend([
                    OptimizationRecommendation(
                        category='响应时间优化',
                        priority='high',
                        description='启用查询缓存',
                        impact='减少数据库查询时间',
                        implementation='Redis缓存热点数据',
                        estimated_improvement=40.0
                    ),
                    OptimizationRecommendation(
                        category='响应时间优化',
                        priority='medium',
                        description='数据库索引优化',
                        impact='提高查询效率',
                        implementation='添加合适的数据库索引',
                        estimated_improvement=30.0
                    )
                ])
            
            elif issue == 'high_error_rate':
                recommendations.extend([
                    OptimizationRecommendation(
                        category='错误率优化',
                        priority='high',
                        description='增强错误处理',
                        impact='提高系统稳定性',
                        implementation='添加重试机制和熔断器',
                        estimated_improvement=50.0
                    )
                ])
        
        return recommendations
    
    def _apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """自动应用优化"""
        for recommendation in recommendations:
            if recommendation.priority == 'high' and recommendation.category in ['内存优化']:
                try:
                    if '内存缓存清理' in recommendation.description:
                        self._perform_memory_cleanup()
                        self.logger.info(f"自动应用优化: {recommendation.description}")
                        
                        # 记录优化历史
                        self.optimization_history.append({
                            'timestamp': datetime.now(),
                            'recommendation': asdict(recommendation),
                            'applied': True,
                            'result': 'success'
                        })
                        
                except Exception as e:
                    self.logger.error(f"自动优化失败: {e}")
    
    def _perform_memory_cleanup(self):
        """执行内存清理"""
        try:
            # 强制垃圾回收
            collected = gc.collect()
            self.logger.info(f"垃圾回收完成，清理对象数: {collected}")
            
            # 清理缓存
            self._cleanup_caches()
            
        except Exception as e:
            self.logger.error(f"内存清理失败: {e}")
    
    def _cleanup_caches(self):
        """清理缓存"""
        try:
            # 清理过期的缓存统计
            current_time = datetime.now()
            for cache_name in list(self.cache_stats.keys()):
                cache_data = self.cache_stats[cache_name]
                if 'last_access' in cache_data:
                    last_access = cache_data['last_access']
                    if isinstance(last_access, datetime) and (current_time - last_access).total_seconds() > 3600:
                        del self.cache_stats[cache_name]
            
            # 清理旧的查询统计
            for query_type in self.query_stats:
                if len(self.query_stats[query_type]) > 100:
                    self.query_stats[query_type] = self.query_stats[query_type][-50:]
            
            self.logger.info("缓存清理完成")
            
        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")
    
    def optimize_database_queries(self, query_stats: Dict[str, List[float]]):
        """优化数据库查询"""
        try:
            recommendations = []
            
            for query_type, execution_times in query_stats.items():
                if len(execution_times) < 10:
                    continue
                
                avg_time = statistics.mean(execution_times)
                max_time = max(execution_times)
                
                # 查询时间过长
                if avg_time > 1.0:  # 1秒
                    recommendations.append({
                        'query_type': query_type,
                        'issue': 'slow_query',
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'suggestion': '考虑添加索引或优化查询逻辑'
                    })
                
                # 查询时间波动大
                if max_time > avg_time * 3:
                    recommendations.append({
                        'query_type': query_type,
                        'issue': 'inconsistent_performance',
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'suggestion': '检查查询计划和数据分布'
                    })
            
            if recommendations:
                self.logger.info(f"数据库查询优化建议: {len(recommendations)}条")
                for rec in recommendations:
                    self.logger.info(f"  {rec['query_type']}: {rec['suggestion']}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"数据库查询优化失败: {e}")
            return []
    
    def optimize_cache_strategy(self, cache_hit_rates: Dict[str, float]):
        """优化缓存策略"""
        try:
            recommendations = []
            
            for cache_name, hit_rate in cache_hit_rates.items():
                if hit_rate < 0.7:  # 命中率低于70%
                    recommendations.append({
                        'cache_name': cache_name,
                        'hit_rate': hit_rate,
                        'suggestion': '考虑调整缓存策略或增加缓存容量'
                    })
                elif hit_rate > 0.95:  # 命中率过高
                    recommendations.append({
                        'cache_name': cache_name,
                        'hit_rate': hit_rate,
                        'suggestion': '缓存效果良好，可以考虑扩展到更多数据'
                    })
            
            if recommendations:
                self.logger.info(f"缓存策略优化建议: {len(recommendations)}条")
                for rec in recommendations:
                    self.logger.info(f"  {rec['cache_name']}: {rec['suggestion']}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"缓存策略优化失败: {e}")
            return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            if not self.metrics_history:
                return {'error': '暂无性能数据'}
            
            recent_metrics = list(self.metrics_history)[-60:]  # 最近60个数据点
            
            # 计算统计信息
            cpu_values = [m.cpu_usage for m in recent_metrics]
            memory_values = [m.memory_usage for m in recent_metrics]
            response_times = [m.response_time for m in recent_metrics]
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(recent_metrics),
                'time_range_minutes': len(recent_metrics),
                'cpu_stats': {
                    'average': statistics.mean(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values),
                    'current': cpu_values[-1] if cpu_values else 0
                },
                'memory_stats': {
                    'average': statistics.mean(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values),
                    'current': memory_values[-1] if memory_values else 0
                },
                'response_time_stats': {
                    'average': statistics.mean(response_times),
                    'max': max(response_times),
                    'min': min(response_times),
                    'current': response_times[-1] if response_times else 0
                },
                'optimization_history': [
                    {
                        'timestamp': opt['timestamp'].isoformat(),
                        'description': opt['recommendation']['description'],
                        'applied': opt['applied'],
                        'result': opt['result']
                    }
                    for opt in list(self.optimization_history)[-10:]
                ],
                'system_health': self._calculate_system_health_score(recent_metrics)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}
    
    def _calculate_system_health_score(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """计算系统健康评分"""
        try:
            if not metrics:
                return {'score': 0, 'status': 'unknown'}
            
            scores = []
            
            # CPU健康评分
            cpu_values = [m.cpu_usage for m in metrics]
            avg_cpu = statistics.mean(cpu_values)
            cpu_score = max(0, 100 - avg_cpu)
            scores.append(cpu_score)
            
            # 内存健康评分
            memory_values = [m.memory_usage for m in metrics]
            avg_memory = statistics.mean(memory_values)
            memory_score = max(0, 100 - avg_memory)
            scores.append(memory_score)
            
            # 响应时间健康评分
            response_times = [m.response_time for m in metrics]
            avg_response_time = statistics.mean(response_times)
            response_score = max(0, 100 - (avg_response_time * 50))  # 2秒响应时间 = 0分
            scores.append(response_score)
            
            # 错误率健康评分
            error_rates = [m.error_rate for m in metrics]
            avg_error_rate = statistics.mean(error_rates)
            error_score = max(0, 100 - (avg_error_rate * 1000))  # 10%错误率 = 0分
            scores.append(error_score)
            
            # 综合评分
            overall_score = statistics.mean(scores)
            
            # 确定状态
            if overall_score >= 80:
                status = 'excellent'
            elif overall_score >= 60:
                status = 'good'
            elif overall_score >= 40:
                status = 'fair'
            elif overall_score >= 20:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'score': round(overall_score, 1),
                'status': status,
                'components': {
                    'cpu': round(cpu_score, 1),
                    'memory': round(memory_score, 1),
                    'response_time': round(response_score, 1),
                    'error_rate': round(error_score, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"计算系统健康评分失败: {e}")
            return {'score': 0, 'status': 'error'}
    
    def manual_optimization(self, optimization_type: str, parameters: Dict[str, Any] = None) -> bool:
        """手动执行优化"""
        try:
            parameters = parameters or {}
            
            if optimization_type == 'memory_cleanup':
                self._perform_memory_cleanup()
                return True
            
            elif optimization_type == 'cache_cleanup':
                self._cleanup_caches()
                return True
            
            elif optimization_type == 'gc_collect':
                collected = gc.collect()
                self.logger.info(f"手动垃圾回收完成，清理对象数: {collected}")
                return True
            
            else:
                self.logger.warning(f"未知的优化类型: {optimization_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"手动优化失败: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    try:
        # 启动监控
        optimizer.start_monitoring()
        
        # 等待收集数据
        time.sleep(120)  # 等待2分钟
        
        # 获取性能报告
        report = optimizer.get_performance_report()
        print("性能报告:", json.dumps(report, indent=2, default=str, ensure_ascii=False))
        
        # 手动执行优化
        optimizer.manual_optimization('memory_cleanup')
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        optimizer.stop_monitoring()
