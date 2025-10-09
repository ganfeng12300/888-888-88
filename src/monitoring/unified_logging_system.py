#!/usr/bin/env python3
"""
📊 统一日志系统 - 生产级日志管理
Unified Logging System - Production-Grade Log Management

生产级特性：
- 结构化日志记录
- 多级别日志分类
- 自动日志轮转
- 性能监控集成
- 异常追踪
- 实时日志流
"""

import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor
import queue
import gzip
import shutil

from loguru import logger as loguru_logger


class LogLevel(Enum):
    """日志级别枚举"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志分类枚举"""
    SYSTEM = "SYSTEM"          # 系统级日志
    TRADING = "TRADING"        # 交易相关日志
    AI = "AI"                 # AI相关日志
    RISK = "RISK"             # 风险管理日志
    PERFORMANCE = "PERFORMANCE" # 性能监控日志
    NETWORK = "NETWORK"        # 网络通信日志
    DATABASE = "DATABASE"      # 数据库操作日志
    SECURITY = "SECURITY"      # 安全相关日志
    BUSINESS = "BUSINESS"      # 业务逻辑日志
    AUDIT = "AUDIT"           # 审计日志
    ALERT = "ALERT"           # 告警日志


@dataclass
class LogEntry:
    """日志条目数据结构"""
    timestamp: str
    level: str
    category: str
    module: str
    function: str
    line: int
    message: str
    extra_data: Dict[str, Any]
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None


@dataclass
class LogConfig:
    """日志配置"""
    log_dir: str = "logs"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10
    compression: bool = True
    console_output: bool = True
    file_output: bool = True
    json_format: bool = True
    include_trace: bool = True
    buffer_size: int = 1000
    flush_interval: float = 5.0
    enable_performance_logging: bool = True
    enable_audit_logging: bool = True
    log_retention_days: int = 30


class LogBuffer:
    """日志缓冲区"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def add(self, log_entry: LogEntry) -> bool:
        """添加日志条目到缓冲区"""
        try:
            self.buffer.put_nowait(log_entry)
            return True
        except queue.Full:
            # 缓冲区满时，移除最旧的条目
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(log_entry)
                return True
            except queue.Empty:
                return False
    
    def get_all(self) -> List[LogEntry]:
        """获取所有缓冲的日志条目"""
        entries = []
        while not self.buffer.empty():
            try:
                entries.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return entries
    
    def size(self) -> int:
        """获取缓冲区大小"""
        return self.buffer.qsize()


class LogFormatter:
    """日志格式化器"""
    
    @staticmethod
    def format_json(log_entry: LogEntry) -> str:
        """格式化为JSON"""
        return json.dumps(asdict(log_entry), ensure_ascii=False, default=str)
    
    @staticmethod
    def format_text(log_entry: LogEntry) -> str:
        """格式化为文本"""
        base_info = f"{log_entry.timestamp} | {log_entry.level:<8} | {log_entry.category:<12} | {log_entry.module}:{log_entry.function}:{log_entry.line}"
        
        if log_entry.execution_time:
            base_info += f" | {log_entry.execution_time:.3f}s"
        
        if log_entry.trace_id:
            base_info += f" | trace:{log_entry.trace_id[:8]}"
        
        message_part = f" | {log_entry.message}"
        
        if log_entry.extra_data:
            extra_part = f" | {json.dumps(log_entry.extra_data, ensure_ascii=False)}"
        else:
            extra_part = ""
        
        return base_info + message_part + extra_part
    
    @staticmethod
    def format_console(log_entry: LogEntry) -> str:
        """格式化为控制台输出"""
        # 添加颜色代码
        level_colors = {
            "TRACE": "\033[90m",      # 灰色
            "DEBUG": "\033[36m",      # 青色
            "INFO": "\033[32m",       # 绿色
            "SUCCESS": "\033[92m",    # 亮绿色
            "WARNING": "\033[33m",    # 黄色
            "ERROR": "\033[31m",      # 红色
            "CRITICAL": "\033[91m",   # 亮红色
        }
        
        reset_color = "\033[0m"
        color = level_colors.get(log_entry.level, "")
        
        time_str = log_entry.timestamp.split('T')[1][:12]  # 只显示时间部分
        
        formatted = f"{color}{time_str} | {log_entry.level:<8} | {log_entry.category:<10} | {log_entry.message}{reset_color}"
        
        return formatted


class LogRotator:
    """日志轮转管理器"""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def should_rotate(self, file_path: Path) -> bool:
        """检查是否需要轮转"""
        if not file_path.exists():
            return False
        
        return file_path.stat().st_size >= self.config.max_file_size
    
    def rotate_file(self, file_path: Path) -> Path:
        """轮转日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        rotated_path = file_path.parent / rotated_name
        
        # 移动当前文件
        shutil.move(str(file_path), str(rotated_path))
        
        # 压缩文件
        if self.config.compression:
            compressed_path = rotated_path.with_suffix(rotated_path.suffix + '.gz')
            with open(rotated_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            rotated_path.unlink()  # 删除未压缩文件
            rotated_path = compressed_path
        
        return rotated_path
    
    def cleanup_old_logs(self):
        """清理过期日志"""
        cutoff_date = datetime.now() - timedelta(days=self.config.log_retention_days)
        
        for log_file in self.log_dir.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    print(f"删除过期日志文件: {log_file}")
            except Exception as e:
                print(f"清理日志文件失败 {log_file}: {e}")


class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self):
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_tracking(self, trace_id: str):
        """开始性能追踪"""
        with self.lock:
            self.start_times[trace_id] = time.time()
    
    def end_tracking(self, trace_id: str) -> Optional[float]:
        """结束性能追踪"""
        with self.lock:
            start_time = self.start_times.pop(trace_id, None)
            if start_time:
                return time.time() - start_time
            return None


class UnifiedLoggingSystem:
    """统一日志系统"""
    
    def __init__(self, config: LogConfig = None):
        self.config = config or LogConfig()
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.buffer = LogBuffer(self.config.buffer_size)
        self.formatter = LogFormatter()
        self.rotator = LogRotator(self.config)
        self.performance_tracker = PerformanceTracker()
        
        # 线程池用于异步日志处理
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="LogWorker")
        
        # 文件句柄缓存
        self.file_handlers = {}
        self.handler_locks = {}
        
        # 控制标志
        self.is_running = True
        self.flush_thread = None
        
        # 初始化日志系统
        self._setup_logging()
        self._start_flush_thread()
        
        # 注册清理函数
        import atexit
        atexit.register(self.shutdown)
    
    def _setup_logging(self):
        """设置日志系统"""
        # 配置loguru
        loguru_logger.remove()  # 移除默认处理器
        
        if self.config.console_output:
            loguru_logger.add(
                sys.stdout,
                format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
                level="DEBUG",
                colorize=True
            )
        
        # 创建分类日志文件
        for category in LogCategory:
            log_file = self.log_dir / f"{category.value.lower()}.log"
            self.file_handlers[category.value] = log_file
            self.handler_locks[category.value] = threading.Lock()
    
    def _start_flush_thread(self):
        """启动日志刷新线程"""
        def flush_worker():
            while self.is_running:
                try:
                    self._flush_buffer()
                    time.sleep(self.config.flush_interval)
                except Exception as e:
                    print(f"日志刷新线程异常: {e}")
        
        self.flush_thread = threading.Thread(
            target=flush_worker,
            daemon=True,
            name="LogFlushThread"
        )
        self.flush_thread.start()
    
    def _flush_buffer(self):
        """刷新日志缓冲区"""
        entries = self.buffer.get_all()
        if not entries:
            return
        
        # 按分类分组
        categorized_entries = {}
        for entry in entries:
            category = entry.category
            if category not in categorized_entries:
                categorized_entries[category] = []
            categorized_entries[category].append(entry)
        
        # 写入文件
        for category, category_entries in categorized_entries.items():
            self._write_to_file(category, category_entries)
    
    def _write_to_file(self, category: str, entries: List[LogEntry]):
        """写入日志文件"""
        if not self.config.file_output:
            return
        
        log_file = self.file_handlers.get(category)
        if not log_file:
            return
        
        lock = self.handler_locks.get(category)
        if not lock:
            return
        
        with lock:
            try:
                # 检查是否需要轮转
                if self.rotator.should_rotate(log_file):
                    self.rotator.rotate_file(log_file)
                
                # 写入日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    for entry in entries:
                        if self.config.json_format:
                            line = self.formatter.format_json(entry)
                        else:
                            line = self.formatter.format_text(entry)
                        f.write(line + '\n')
                    f.flush()
                    
            except Exception as e:
                print(f"写入日志文件失败 {log_file}: {e}")
    
    def _get_caller_info(self) -> tuple:
        """获取调用者信息"""
        frame = sys._getframe(3)  # 跳过装饰器和日志方法
        return (
            frame.f_code.co_filename.split('/')[-1],  # 模块名
            frame.f_code.co_name,                     # 函数名
            frame.f_lineno                            # 行号
        )
    
    def _create_log_entry(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        extra_data: Dict[str, Any] = None,
        trace_id: str = None,
        user_id: str = None,
        session_id: str = None,
        request_id: str = None,
        execution_time: float = None
    ) -> LogEntry:
        """创建日志条目"""
        module, function, line = self._get_caller_info()
        
        # 获取系统资源信息
        memory_usage = None
        cpu_usage = None
        
        if self.config.enable_performance_logging:
            try:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss
                cpu_usage = process.cpu_percent()
            except:
                pass
        
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            module=module,
            function=function,
            line=line,
            message=message,
            extra_data=extra_data or {},
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
    
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        extra_data: Dict[str, Any] = None,
        **kwargs
    ):
        """记录日志"""
        try:
            log_entry = self._create_log_entry(
                level=level,
                category=category,
                message=message,
                extra_data=extra_data,
                **kwargs
            )
            
            # 添加到缓冲区
            self.buffer.add(log_entry)
            
            # 控制台输出
            if self.config.console_output:
                console_msg = self.formatter.format_console(log_entry)
                print(console_msg)
            
            # 关键日志立即刷新
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.executor.submit(self._flush_buffer)
                
        except Exception as e:
            print(f"日志记录失败: {e}")
    
    # 便捷方法
    def trace(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录TRACE级别日志"""
        self.log(LogLevel.TRACE, category, message, **kwargs)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录DEBUG级别日志"""
        self.log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录INFO级别日志"""
        self.log(LogLevel.INFO, category, message, **kwargs)
    
    def success(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录SUCCESS级别日志"""
        self.log(LogLevel.SUCCESS, category, message, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录WARNING级别日志"""
        self.log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录ERROR级别日志"""
        # 自动添加异常信息
        if 'extra_data' not in kwargs:
            kwargs['extra_data'] = {}
        
        if self.config.include_trace:
            kwargs['extra_data']['traceback'] = traceback.format_exc()
        
        self.log(LogLevel.ERROR, category, message, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """记录CRITICAL级别日志"""
        # 自动添加异常信息
        if 'extra_data' not in kwargs:
            kwargs['extra_data'] = {}
        
        if self.config.include_trace:
            kwargs['extra_data']['traceback'] = traceback.format_exc()
        
        self.log(LogLevel.CRITICAL, category, message, **kwargs)
    
    # 特殊日志方法
    def trading_log(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs):
        """交易日志"""
        self.log(level, LogCategory.TRADING, message, **kwargs)
    
    def risk_log(self, message: str, level: LogLevel = LogLevel.WARNING, **kwargs):
        """风险日志"""
        self.log(level, LogCategory.RISK, message, **kwargs)
    
    def performance_log(self, message: str, execution_time: float = None, **kwargs):
        """性能日志"""
        self.log(LogLevel.INFO, LogCategory.PERFORMANCE, message, execution_time=execution_time, **kwargs)
    
    def security_log(self, message: str, level: LogLevel = LogLevel.WARNING, **kwargs):
        """安全日志"""
        self.log(level, LogCategory.SECURITY, message, **kwargs)
    
    def audit_log(self, message: str, user_id: str = None, **kwargs):
        """审计日志"""
        if not self.config.enable_audit_logging:
            return
        self.log(LogLevel.INFO, LogCategory.AUDIT, message, user_id=user_id, **kwargs)
    
    def alert_log(self, message: str, level: LogLevel = LogLevel.ERROR, **kwargs):
        """告警日志"""
        self.log(level, LogCategory.ALERT, message, **kwargs)
    
    # 性能追踪装饰器
    def track_performance(self, category: LogCategory = LogCategory.PERFORMANCE):
        """性能追踪装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                trace_id = f"{func.__name__}_{int(time.time() * 1000000)}"
                self.performance_tracker.start_tracking(trace_id)
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = self.performance_tracker.end_tracking(trace_id)
                    
                    self.performance_log(
                        f"函数 {func.__name__} 执行完成",
                        execution_time=execution_time,
                        extra_data={
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        },
                        trace_id=trace_id
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = self.performance_tracker.end_tracking(trace_id)
                    
                    self.error(
                        f"函数 {func.__name__} 执行异常: {str(e)}",
                        category=category,
                        execution_time=execution_time,
                        extra_data={
                            'function': func.__name__,
                            'exception': str(e),
                            'exception_type': type(e).__name__
                        },
                        trace_id=trace_id
                    )
                    raise
            
            return wrapper
        return decorator
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = {
            'buffer_size': self.buffer.size(),
            'log_files': {},
            'system_info': {
                'log_dir': str(self.log_dir),
                'config': asdict(self.config),
                'is_running': self.is_running
            }
        }
        
        # 统计各分类日志文件大小
        for category, log_file in self.file_handlers.items():
            if log_file.exists():
                stats['log_files'][category] = {
                    'size_bytes': log_file.stat().st_size,
                    'size_mb': round(log_file.stat().st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
        
        return stats
    
    def search_logs(
        self,
        category: str = None,
        level: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        keyword: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索日志"""
        results = []
        
        # 确定要搜索的文件
        if category:
            log_files = [self.file_handlers.get(category.upper())]
        else:
            log_files = list(self.file_handlers.values())
        
        for log_file in log_files:
            if not log_file or not log_file.exists():
                continue
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        try:
                            if self.config.json_format:
                                log_data = json.loads(line.strip())
                            else:
                                # 简单文本解析
                                continue
                            
                            # 过滤条件
                            if level and log_data.get('level') != level.upper():
                                continue
                            
                            if keyword and keyword.lower() not in log_data.get('message', '').lower():
                                continue
                            
                            if start_time or end_time:
                                log_time = datetime.fromisoformat(log_data.get('timestamp', ''))
                                if start_time and log_time < start_time:
                                    continue
                                if end_time and log_time > end_time:
                                    continue
                            
                            results.append(log_data)
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
                            
            except Exception as e:
                self.error(f"搜索日志文件失败 {log_file}: {e}")
        
        return results
    
    def export_logs(
        self,
        output_file: str,
        category: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        format: str = 'json'
    ) -> bool:
        """导出日志"""
        try:
            logs = self.search_logs(
                category=category,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
            elif format.lower() == 'csv':
                import csv
                if logs:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        writer.writerows(logs)
            
            self.info(f"日志导出成功: {output_file}, 共{len(logs)}条记录")
            return True
            
        except Exception as e:
            self.error(f"日志导出失败: {e}")
            return False
    
    def cleanup_logs(self):
        """清理过期日志"""
        self.rotator.cleanup_old_logs()
    
    def shutdown(self):
        """关闭日志系统"""
        self.info("正在关闭统一日志系统...")
        
        self.is_running = False
        
        # 刷新剩余日志
        self._flush_buffer()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 等待刷新线程结束
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5)
        
        self.info("统一日志系统已关闭")


# 全局日志实例
_global_logger = None


def get_logger(config: LogConfig = None) -> UnifiedLoggingSystem:
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLoggingSystem(config)
    return _global_logger


def setup_logging(config: LogConfig = None) -> UnifiedLoggingSystem:
    """设置全局日志系统"""
    global _global_logger
    _global_logger = UnifiedLoggingSystem(config)
    return _global_logger


# 便捷函数
def log_info(message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
    """记录信息日志"""
    get_logger().info(message, category, **kwargs)


def log_error(message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
    """记录错误日志"""
    get_logger().error(message, category, **kwargs)


def log_warning(message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
    """记录警告日志"""
    get_logger().warning(message, category, **kwargs)


def log_trading(message: str, **kwargs):
    """记录交易日志"""
    get_logger().trading_log(message, **kwargs)


def log_risk(message: str, **kwargs):
    """记录风险日志"""
    get_logger().risk_log(message, **kwargs)


def log_performance(message: str, execution_time: float = None, **kwargs):
    """记录性能日志"""
    get_logger().performance_log(message, execution_time, **kwargs)


def track_performance(category: LogCategory = LogCategory.PERFORMANCE):
    """性能追踪装饰器"""
    return get_logger().track_performance(category)


if __name__ == "__main__":
    # 测试代码
    config = LogConfig(
        log_dir="test_logs",
        console_output=True,
        json_format=True
    )
    
    logger_system = UnifiedLoggingSystem(config)
    
    # 测试各种日志
    logger_system.info("系统启动", LogCategory.SYSTEM)
    logger_system.trading_log("开始交易", extra_data={"symbol": "BTCUSDT", "price": 50000})
    logger_system.risk_log("风险警告", extra_data={"risk_score": 75})
    logger_system.performance_log("性能测试", execution_time=0.123)
    
    # 测试性能追踪装饰器
    @logger_system.track_performance(LogCategory.TRADING)
    def test_function():
        time.sleep(0.1)
        return "测试完成"
    
    result = test_function()
    
    # 获取统计信息
    stats = logger_system.get_log_stats()
    print(f"日志统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 关闭系统
    logger_system.shutdown()
