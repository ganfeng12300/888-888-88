#!/usr/bin/env python3
"""
📊 888-888-88 系统监控器
生产级系统性能监控和报警系统
"""

import asyncio
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from loguru import logger
import sqlite3
from collections import deque


class AlertLevel(Enum):
    """报警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    temperature: Optional[float] = None


@dataclass
class Alert:
    """报警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, db_path: str = "data/monitoring.db"):
        self.db_path = db_path
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 监控配置
        self.check_interval = 30  # 30秒检查间隔
        self.alert_thresholds = {
            'cpu_usage': 80.0,      # CPU使用率阈值
            'memory_usage': 85.0,   # 内存使用率阈值
            'disk_usage': 90.0,     # 磁盘使用率阈值
            'temperature': 70.0     # 温度阈值
        }
        
        # 数据存储
        self.metrics_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'total_alerts': 0,
            'resolved_alerts': 0,
            'uptime_start': datetime.now(),
            'last_check': None
        }
        
        self.init_database()
        logger.info("📊 系统监控器初始化完成")
    
    def init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_sent REAL,
                network_recv REAL,
                process_count INTEGER,
                uptime REAL,
                temperature REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.monitoring:
            logger.warning("⚠️ 系统监控已在运行")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 系统监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ 系统监控已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                
                # 存储指标
                self.metrics_history.append(metrics)
                self._save_metrics(metrics)
                
                # 检查报警条件
                self._check_alerts(metrics)
                
                # 更新统计
                self.stats['last_check'] = datetime.now()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ 监控循环异常: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # 网络IO
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv
        }
        
        # 进程数量
        process_count = len(psutil.pids())
        
        # 系统运行时间
        uptime = time.time() - psutil.boot_time()
        
        # 温度（如果可用）
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # 获取CPU温度
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        temperature = entries[0].current
                        break
        except:
            pass
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            uptime=uptime,
            temperature=temperature
        )
    
    def _save_metrics(self, metrics: SystemMetrics) -> None:
        """保存指标到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, disk_usage, 
                 network_sent, network_recv, process_count, uptime, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(metrics.timestamp.timestamp()),
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.network_io['bytes_sent'],
                metrics.network_io['bytes_recv'],
                metrics.process_count,
                metrics.uptime,
                metrics.temperature
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存指标失败: {e}")
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """检查报警条件"""
        # CPU使用率报警
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            self._create_alert(
                "high_cpu_usage",
                AlertLevel.WARNING,
                "CPU使用率过高",
                f"CPU使用率达到 {metrics.cpu_usage:.1f}%",
                "system_monitor",
                {'cpu_usage': metrics.cpu_usage}
            )
        
        # 内存使用率报警
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            self._create_alert(
                "high_memory_usage",
                AlertLevel.WARNING,
                "内存使用率过高",
                f"内存使用率达到 {metrics.memory_usage:.1f}%",
                "system_monitor",
                {'memory_usage': metrics.memory_usage}
            )
        
        # 磁盘使用率报警
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            self._create_alert(
                "high_disk_usage",
                AlertLevel.ERROR,
                "磁盘使用率过高",
                f"磁盘使用率达到 {metrics.disk_usage:.1f}%",
                "system_monitor",
                {'disk_usage': metrics.disk_usage}
            )
        
        # 温度报警
        if (metrics.temperature and 
            metrics.temperature > self.alert_thresholds['temperature']):
            self._create_alert(
                "high_temperature",
                AlertLevel.CRITICAL,
                "系统温度过高",
                f"系统温度达到 {metrics.temperature:.1f}°C",
                "system_monitor",
                {'temperature': metrics.temperature}
            )
    
    def _create_alert(self, alert_id: str, level: AlertLevel, title: str,
                     message: str, source: str, metadata: Dict[str, Any]) -> None:
        """创建报警"""
        if alert_id in self.active_alerts:
            return  # 避免重复报警
        
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata
        )
        
        self.active_alerts[alert_id] = alert
        self.stats['total_alerts'] += 1
        
        # 保存到数据库
        self._save_alert(alert)
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"❌ 报警回调失败: {e}")
        
        logger.warning(f"🚨 {level.value.upper()}: {title} - {message}")
    
    def _save_alert(self, alert: Alert) -> None:
        """保存报警到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, level, title, message, timestamp, source, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.level.value,
                alert.title,
                alert.message,
                int(alert.timestamp.timestamp()),
                alert.source,
                1 if alert.resolved else 0,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存报警失败: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决报警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            self._save_alert(alert)
            del self.active_alerts[alert_id]
            self.stats['resolved_alerts'] += 1
            logger.info(f"✅ 报警已解决: {alert_id}")
            return True
        return False
    
    def add_alert_callback(self, callback: Callable) -> None:
        """添加报警回调"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """获取历史指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃报警"""
        return list(self.active_alerts.values())
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        current_metrics = self.get_current_metrics()
        uptime_seconds = (datetime.now() - self.stats['uptime_start']).total_seconds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': self.monitoring,
            'uptime_seconds': uptime_seconds,
            'current_metrics': {
                'cpu_usage': current_metrics.cpu_usage if current_metrics else 0,
                'memory_usage': current_metrics.memory_usage if current_metrics else 0,
                'disk_usage': current_metrics.disk_usage if current_metrics else 0,
                'process_count': current_metrics.process_count if current_metrics else 0,
                'temperature': current_metrics.temperature if current_metrics else None
            } if current_metrics else {},
            'alert_summary': {
                'total_alerts': self.stats['total_alerts'],
                'active_alerts': len(self.active_alerts),
                'resolved_alerts': self.stats['resolved_alerts']
            },
            'thresholds': self.alert_thresholds.copy(),
            'last_check': self.stats['last_check'].isoformat() if self.stats['last_check'] else None
        }


if __name__ == "__main__":
    # 测试系统监控器
    monitor = SystemMonitor()
    
    # 添加报警回调
    def on_alert(alert: Alert):
        print(f"收到报警: {alert.level.value} - {alert.title}")
    
    monitor.add_alert_callback(on_alert)
    
    # 开始监控
    monitor.start_monitoring()
    
    try:
        # 运行一段时间
        time.sleep(60)
        
        # 获取报告
        report = monitor.get_monitoring_report()
        print(f"监控报告: {json.dumps(report, indent=2, default=str)}")
        
    finally:
        # 停止监控
        monitor.stop_monitoring()

