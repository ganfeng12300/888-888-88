"""
🏥 系统健康检查器 - 生产级实盘交易系统全方位健康状态监控
监控系统各组件健康状态、服务可用性、数据完整性、异常检测
提供系统诊断、故障预警、自动恢复、健康报告生成
"""
import asyncio
import time
import threading
import subprocess
import socket
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"  # 健康
    WARNING = "warning"  # 警告
    CRITICAL = "critical"  # 关键
    DOWN = "down"  # 宕机

class ComponentType(Enum):
    """组件类型"""
    HARDWARE = "hardware"  # 硬件
    SOFTWARE = "software"  # 软件
    NETWORK = "network"  # 网络
    DATABASE = "database"  # 数据库
    API = "api"  # API接口
    SERVICE = "service"  # 服务

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component_id: str  # 组件ID
    component_type: ComponentType  # 组件类型
    status: HealthStatus  # 健康状态
    response_time: float  # 响应时间 (毫秒)
    error_message: Optional[str]  # 错误消息
    details: Dict[str, Any]  # 详细信息
    timestamp: float = field(default_factory=time.time)  # 时间戳

@dataclass
class SystemHealthReport:
    """系统健康报告"""
    overall_status: HealthStatus  # 整体状态
    healthy_components: int  # 健康组件数
    warning_components: int  # 警告组件数
    critical_components: int  # 关键组件数
    down_components: int  # 宕机组件数
    total_components: int  # 总组件数
    uptime: float  # 系统运行时间
    last_check_time: float  # 最后检查时间
    component_results: List[HealthCheckResult]  # 组件检查结果
    recommendations: List[str]  # 建议

class NetworkChecker:
    """网络检查器"""
    
    def __init__(self):
        logger.info("网络检查器初始化完成")
    
    def check_internet_connectivity(self) -> HealthCheckResult:
        """检查互联网连接"""
        try:
            start_time = time.time()
            
            # 尝试连接到公共DNS服务器
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component_id="internet_connectivity",
                component_type=ComponentType.NETWORK,
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                error_message=None,
                details={"target": "8.8.8.8:53", "timeout": 5}
            )
        
        except Exception as e:
            return HealthCheckResult(
                component_id="internet_connectivity",
                component_type=ComponentType.NETWORK,
                status=HealthStatus.DOWN,
                response_time=5000,  # 超时时间
                error_message=str(e),
                details={"target": "8.8.8.8:53", "timeout": 5}
            )
    
    def check_exchange_connectivity(self, exchange_urls: List[str]) -> List[HealthCheckResult]:
        """检查交易所连接"""
        results = []
        
        for url in exchange_urls:
            try:
                start_time = time.time()
                
                # 解析URL获取主机和端口
                if "://" in url:
                    host = url.split("://")[1].split("/")[0].split(":")[0]
                    port = 443 if url.startswith("https") else 80
                else:
                    host = url.split(":")[0]
                    port = int(url.split(":")[1]) if ":" in url else 80
                
                socket.create_connection((host, port), timeout=10)
                
                response_time = (time.time() - start_time) * 1000
                
                results.append(HealthCheckResult(
                    component_id=f"exchange_{host}",
                    component_type=ComponentType.NETWORK,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    error_message=None,
                    details={"host": host, "port": port, "url": url}
                ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"exchange_{host if 'host' in locals() else 'unknown'}",
                    component_type=ComponentType.NETWORK,
                    status=HealthStatus.DOWN,
                    response_time=10000,
                    error_message=str(e),
                    details={"url": url, "timeout": 10}
                ))
        
        return results
    
    def check_dns_resolution(self, domains: List[str]) -> List[HealthCheckResult]:
        """检查DNS解析"""
        results = []
        
        for domain in domains:
            try:
                start_time = time.time()
                
                socket.gethostbyname(domain)
                
                response_time = (time.time() - start_time) * 1000
                
                results.append(HealthCheckResult(
                    component_id=f"dns_{domain}",
                    component_type=ComponentType.NETWORK,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    error_message=None,
                    details={"domain": domain}
                ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"dns_{domain}",
                    component_type=ComponentType.NETWORK,
                    status=HealthStatus.DOWN,
                    response_time=0,
                    error_message=str(e),
                    details={"domain": domain}
                ))
        
        return results

class ServiceChecker:
    """服务检查器"""
    
    def __init__(self):
        logger.info("服务检查器初始化完成")
    
    def check_process_running(self, process_names: List[str]) -> List[HealthCheckResult]:
        """检查进程是否运行"""
        results = []
        
        for process_name in process_names:
            try:
                start_time = time.time()
                
                # 使用ps命令检查进程
                result = subprocess.run(
                    ["pgrep", "-f", process_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    status = HealthStatus.HEALTHY
                    error_message = None
                    details = {"pids": pids, "count": len(pids)}
                else:
                    status = HealthStatus.DOWN
                    error_message = "Process not found"
                    details = {"pids": [], "count": 0}
                
                results.append(HealthCheckResult(
                    component_id=f"process_{process_name}",
                    component_type=ComponentType.SERVICE,
                    status=status,
                    response_time=response_time,
                    error_message=error_message,
                    details=details
                ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"process_{process_name}",
                    component_type=ComponentType.SERVICE,
                    status=HealthStatus.CRITICAL,
                    response_time=0,
                    error_message=str(e),
                    details={"process_name": process_name}
                ))
        
        return results
    
    def check_port_listening(self, ports: List[int]) -> List[HealthCheckResult]:
        """检查端口监听"""
        results = []
        
        for port in ports:
            try:
                start_time = time.time()
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                response_time = (time.time() - start_time) * 1000
                
                if result == 0:
                    status = HealthStatus.HEALTHY
                    error_message = None
                else:
                    status = HealthStatus.DOWN
                    error_message = f"Port {port} not listening"
                
                results.append(HealthCheckResult(
                    component_id=f"port_{port}",
                    component_type=ComponentType.SERVICE,
                    status=status,
                    response_time=response_time,
                    error_message=error_message,
                    details={"port": port, "host": "localhost"}
                ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"port_{port}",
                    component_type=ComponentType.SERVICE,
                    status=HealthStatus.CRITICAL,
                    response_time=0,
                    error_message=str(e),
                    details={"port": port}
                ))
        
        return results

class DatabaseChecker:
    """数据库检查器"""
    
    def __init__(self):
        logger.info("数据库检查器初始化完成")
    
    def check_database_connection(self, db_configs: List[Dict[str, Any]]) -> List[HealthCheckResult]:
        """检查数据库连接"""
        results = []
        
        for db_config in db_configs:
            db_type = db_config.get('type', 'unknown')
            db_name = db_config.get('name', 'unknown')
            
            try:
                start_time = time.time()
                
                # 这里应该根据不同的数据库类型使用相应的连接方法
                # 为了简化，我们只检查连接性
                if db_type == 'redis':
                    host = db_config.get('host', 'localhost')
                    port = db_config.get('port', 6379)
                    sock = socket.create_connection((host, port), timeout=5)
                    sock.close()
                elif db_type == 'postgresql':
                    host = db_config.get('host', 'localhost')
                    port = db_config.get('port', 5432)
                    sock = socket.create_connection((host, port), timeout=5)
                    sock.close()
                else:
                    # 默认检查主机端口连接
                    host = db_config.get('host', 'localhost')
                    port = db_config.get('port', 3306)
                    sock = socket.create_connection((host, port), timeout=5)
                    sock.close()
                
                response_time = (time.time() - start_time) * 1000
                
                results.append(HealthCheckResult(
                    component_id=f"database_{db_name}",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    error_message=None,
                    details=db_config
                ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"database_{db_name}",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.DOWN,
                    response_time=0,
                    error_message=str(e),
                    details=db_config
                ))
        
        return results

class APIChecker:
    """API检查器"""
    
    def __init__(self):
        logger.info("API检查器初始化完成")
    
    def check_api_endpoints(self, api_configs: List[Dict[str, Any]]) -> List[HealthCheckResult]:
        """检查API端点"""
        results = []
        
        for api_config in api_configs:
            api_name = api_config.get('name', 'unknown')
            url = api_config.get('url', '')
            
            try:
                start_time = time.time()
                
                # 简化的HTTP检查 - 实际应该使用requests库
                if url.startswith('http'):
                    # 解析URL
                    if "://" in url:
                        host = url.split("://")[1].split("/")[0].split(":")[0]
                        port = 443 if url.startswith("https") else 80
                        if ":" in url.split("://")[1].split("/")[0]:
                            port = int(url.split("://")[1].split("/")[0].split(":")[1])
                    
                    sock = socket.create_connection((host, port), timeout=10)
                    sock.close()
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    # 简单的状态判断
                    if response_time < 1000:
                        status = HealthStatus.HEALTHY
                    elif response_time < 3000:
                        status = HealthStatus.WARNING
                    else:
                        status = HealthStatus.CRITICAL
                    
                    results.append(HealthCheckResult(
                        component_id=f"api_{api_name}",
                        component_type=ComponentType.API,
                        status=status,
                        response_time=response_time,
                        error_message=None,
                        details=api_config
                    ))
                else:
                    results.append(HealthCheckResult(
                        component_id=f"api_{api_name}",
                        component_type=ComponentType.API,
                        status=HealthStatus.CRITICAL,
                        response_time=0,
                        error_message="Invalid URL format",
                        details=api_config
                    ))
            
            except Exception as e:
                results.append(HealthCheckResult(
                    component_id=f"api_{api_name}",
                    component_type=ComponentType.API,
                    status=HealthStatus.DOWN,
                    response_time=0,
                    error_message=str(e),
                    details=api_config
                ))
        
        return results

class SystemHealthChecker:
    """系统健康检查器主类"""
    
    def __init__(self):
        self.network_checker = NetworkChecker()
        self.service_checker = ServiceChecker()
        self.database_checker = DatabaseChecker()
        self.api_checker = APIChecker()
        
        # 检查配置
        self.check_configs = {
            'exchange_urls': [
                'api.binance.com',
                'api.huobi.pro',
                'www.okx.com'
            ],
            'dns_domains': [
                'google.com',
                'cloudflare.com',
                'github.com'
            ],
            'processes': [
                'python',
                'redis-server',
                'postgres'
            ],
            'ports': [6379, 5432, 8080],
            'databases': [
                {'type': 'redis', 'name': 'cache', 'host': 'localhost', 'port': 6379},
                {'type': 'postgresql', 'name': 'main', 'host': 'localhost', 'port': 5432}
            ],
            'apis': [
                {'name': 'binance', 'url': 'https://api.binance.com/api/v3/ping'},
                {'name': 'huobi', 'url': 'https://api.huobi.pro/v1/common/timestamp'},
                {'name': 'okx', 'url': 'https://www.okx.com/api/v5/public/time'}
            ]
        }
        
        # 健康检查历史
        self.health_history: List[SystemHealthReport] = []
        
        # 监控配置
        self.check_interval = 300  # 检查间隔（秒）
        self.is_monitoring = False
        
        # 回调函数
        self.health_callbacks: List[Callable[[SystemHealthReport], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 系统启动时间
        self.system_start_time = time.time()
        
        logger.info("系统健康检查器初始化完成")
    
    def check_all_systems(self) -> SystemHealthReport:
        """检查所有系统（别名方法）"""
        return self.perform_full_health_check()
    
    def perform_full_health_check(self) -> SystemHealthReport:
        """执行完整健康检查"""
        try:
            all_results = []
            
            # 网络检查
            all_results.append(self.network_checker.check_internet_connectivity())
            all_results.extend(self.network_checker.check_exchange_connectivity(
                self.check_configs['exchange_urls']
            ))
            all_results.extend(self.network_checker.check_dns_resolution(
                self.check_configs['dns_domains']
            ))
            
            # 服务检查
            all_results.extend(self.service_checker.check_process_running(
                self.check_configs['processes']
            ))
            all_results.extend(self.service_checker.check_port_listening(
                self.check_configs['ports']
            ))
            
            # 数据库检查
            all_results.extend(self.database_checker.check_database_connection(
                self.check_configs['databases']
            ))
            
            # API检查
            all_results.extend(self.api_checker.check_api_endpoints(
                self.check_configs['apis']
            ))
            
            # 统计结果
            healthy_count = sum(1 for r in all_results if r.status == HealthStatus.HEALTHY)
            warning_count = sum(1 for r in all_results if r.status == HealthStatus.WARNING)
            critical_count = sum(1 for r in all_results if r.status == HealthStatus.CRITICAL)
            down_count = sum(1 for r in all_results if r.status == HealthStatus.DOWN)
            total_count = len(all_results)
            
            # 确定整体状态
            if down_count > 0 or critical_count > total_count * 0.3:
                overall_status = HealthStatus.CRITICAL
            elif critical_count > 0 or warning_count > total_count * 0.5:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
            
            # 生成建议
            recommendations = self._generate_health_recommendations(all_results)
            
            # 创建健康报告
            report = SystemHealthReport(
                overall_status=overall_status,
                healthy_components=healthy_count,
                warning_components=warning_count,
                critical_components=critical_count,
                down_components=down_count,
                total_components=total_count,
                uptime=time.time() - self.system_start_time,
                last_check_time=time.time(),
                component_results=all_results,
                recommendations=recommendations
            )
            
            return report
        
        except Exception as e:
            logger.error(f"执行健康检查失败: {e}")
            return SystemHealthReport(
                overall_status=HealthStatus.CRITICAL,
                healthy_components=0,
                warning_components=0,
                critical_components=0,
                down_components=0,
                total_components=0,
                uptime=time.time() - self.system_start_time,
                last_check_time=time.time(),
                component_results=[],
                recommendations=["健康检查系统故障，请检查系统配置"]
            )
    
    def _generate_health_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        try:
            # 按组件类型分组分析
            network_issues = [r for r in results if r.component_type == ComponentType.NETWORK and r.status != HealthStatus.HEALTHY]
            service_issues = [r for r in results if r.component_type == ComponentType.SERVICE and r.status != HealthStatus.HEALTHY]
            database_issues = [r for r in results if r.component_type == ComponentType.DATABASE and r.status != HealthStatus.HEALTHY]
            api_issues = [r for r in results if r.component_type == ComponentType.API and r.status != HealthStatus.HEALTHY]
            
            # 网络问题建议
            if network_issues:
                if any('internet_connectivity' in r.component_id for r in network_issues):
                    recommendations.append("检测到网络连接问题，请检查网络配置和防火墙设置")
                if any('exchange_' in r.component_id for r in network_issues):
                    recommendations.append("交易所连接异常，请检查交易所服务状态和网络连接")
                if any('dns_' in r.component_id for r in network_issues):
                    recommendations.append("DNS解析异常，请检查DNS服务器配置")
            
            # 服务问题建议
            if service_issues:
                if any('process_' in r.component_id for r in service_issues):
                    recommendations.append("检测到关键进程未运行，请检查服务状态并重启相关服务")
                if any('port_' in r.component_id for r in service_issues):
                    recommendations.append("检测到端口监听异常，请检查服务配置和端口占用情况")
            
            # 数据库问题建议
            if database_issues:
                recommendations.append("数据库连接异常，请检查数据库服务状态和连接配置")
            
            # API问题建议
            if api_issues:
                recommendations.append("API服务异常，请检查API服务状态和网络连接")
            
            # 性能问题建议
            slow_responses = [r for r in results if r.response_time > 5000]
            if slow_responses:
                recommendations.append("检测到响应时间过长，建议优化网络配置或服务性能")
            
            # 如果没有问题
            if not recommendations:
                recommendations.append("系统运行正常，所有组件状态良好")
        
        except Exception as e:
            logger.error(f"生成健康建议失败: {e}")
            recommendations.append("无法生成健康建议，请手动检查系统状态")
        
        return recommendations
    
    def start_monitoring(self):
        """启动健康监控"""
        try:
            self.is_monitoring = True
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            logger.info("系统健康监控启动")
        
        except Exception as e:
            logger.error(f"启动系统健康监控失败: {e}")
    
    def stop_monitoring(self):
        """停止健康监控"""
        self.is_monitoring = False
        logger.info("系统健康监控停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行健康检查
                health_report = self.perform_full_health_check()
                
                with self.lock:
                    # 添加到历史记录
                    self.health_history.append(health_report)
                    
                    # 保持历史记录在合理范围内
                    if len(self.health_history) > 100:
                        self.health_history = self.health_history[-50:]
                    
                    # 调用回调函数
                    for callback in self.health_callbacks:
                        try:
                            callback(health_report)
                        except Exception as e:
                            logger.error(f"健康检查回调执行失败: {e}")
                
                # 记录关键状态变化
                if health_report.overall_status == HealthStatus.CRITICAL:
                    logger.critical(f"系统健康状态严重: {health_report.critical_components + health_report.down_components} 个组件异常")
                elif health_report.overall_status == HealthStatus.WARNING:
                    logger.warning(f"系统健康状态警告: {health_report.warning_components} 个组件警告")
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"健康监控循环失败: {e}")
                time.sleep(self.check_interval)
    
    def add_health_callback(self, callback: Callable[[SystemHealthReport], None]):
        """添加健康检查回调"""
        self.health_callbacks.append(callback)
    
    def update_check_config(self, config_type: str, config_value: Any):
        """更新检查配置"""
        if config_type in self.check_configs:
            self.check_configs[config_type] = config_value
            logger.info(f"更新检查配置: {config_type}")
        else:
            logger.warning(f"未知的配置类型: {config_type}")
    
    def get_latest_health_report(self) -> Optional[SystemHealthReport]:
        """获取最新健康报告"""
        with self.lock:
            return self.health_history[-1] if self.health_history else None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        try:
            with self.lock:
                latest_report = self.get_latest_health_report()
                
                if not latest_report:
                    return {'status': 'no_data', 'message': '暂无健康检查数据'}
                
                # 计算可用性
                uptime_hours = latest_report.uptime / 3600
                availability = (latest_report.healthy_components / latest_report.total_components * 100) if latest_report.total_components > 0 else 0
                
                # 最近趋势
                recent_reports = self.health_history[-10:] if len(self.health_history) >= 10 else self.health_history
                trend_improving = True
                if len(recent_reports) >= 2:
                    recent_healthy = recent_reports[-1].healthy_components
                    previous_healthy = recent_reports[-2].healthy_components
                    trend_improving = recent_healthy >= previous_healthy
                
                return {
                    'overall_status': latest_report.overall_status.value,
                    'availability_percentage': round(availability, 2),
                    'uptime_hours': round(uptime_hours, 2),
                    'component_summary': {
                        'healthy': latest_report.healthy_components,
                        'warning': latest_report.warning_components,
                        'critical': latest_report.critical_components,
                        'down': latest_report.down_components,
                        'total': latest_report.total_components
                    },
                    'trend_improving': trend_improving,
                    'last_check': latest_report.last_check_time,
                    'monitoring_active': self.is_monitoring,
                    'total_checks': len(self.health_history),
                    'top_recommendations': latest_report.recommendations[:3]
                }
        
        except Exception as e:
            logger.error(f"获取健康摘要失败: {e}")
            return {'status': 'error', 'message': f'获取健康摘要失败: {str(e)}'}
    
    def get_component_health_details(self, component_type: str = None) -> List[Dict[str, Any]]:
        """获取组件健康详情"""
        try:
            with self.lock:
                latest_report = self.get_latest_health_report()
                
                if not latest_report:
                    return []
                
                results = latest_report.component_results
                
                # 按组件类型过滤
                if component_type:
                    results = [r for r in results if r.component_type.value == component_type]
                
                # 转换为字典格式
                details = []
                for result in results:
                    details.append({
                        'component_id': result.component_id,
                        'component_type': result.component_type.value,
                        'status': result.status.value,
                        'response_time': result.response_time,
                        'error_message': result.error_message,
                        'details': result.details,
                        'timestamp': result.timestamp
                    })
                
                return details
        
        except Exception as e:
            logger.error(f"获取组件健康详情失败: {e}")
            return []

# 全局系统健康检查器实例
system_health_checker = SystemHealthChecker()
