#!/usr/bin/env python3
"""
🛡️ 888-888-88 量化交易系统错误处理系统
Production-Grade Error Handling System
"""

import os
import sys
import traceback
import functools
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from loguru import logger

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误分类"""
    TRADING = "trading"
    NETWORK = "network"
    DATABASE = "database"
    AI_MODEL = "ai_model"
    EXCHANGE_API = "exchange_api"
    SYSTEM = "system"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"

@dataclass
class ErrorContext:
    """错误上下文信息"""
    timestamp: datetime
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    function_name: str
    module_name: str
    error_message: str
    stack_trace: str
    user_data: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False

class ErrorRecoveryStrategy:
    """错误恢复策略基类"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """执行错误恢复"""
        raise NotImplementedError

class NetworkErrorRecovery(ErrorRecoveryStrategy):
    """网络错误恢复策略"""
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """网络错误恢复"""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.backoff_factor ** attempt)
                # 测试网络连接
                response = requests.get("https://httpbin.org/status/200", timeout=5)
                if response.status_code == 200:
                    logger.info(f"🌐 网络连接恢复，尝试次数: {attempt + 1}")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ 网络恢复尝试 {attempt + 1} 失败: {e}")
                continue
        return False

class DatabaseErrorRecovery(ErrorRecoveryStrategy):
    """数据库错误恢复策略"""
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """数据库错误恢复"""
        try:
            # 重新初始化数据库连接
            from src.database.database_manager import DatabaseManager
            db_manager = DatabaseManager()
            await db_manager.reconnect()
            logger.info("🗄️ 数据库连接已恢复")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库恢复失败: {e}")
            return False

class ExchangeAPIErrorRecovery(ErrorRecoveryStrategy):
    """交易所API错误恢复策略"""
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """交易所API错误恢复"""
        try:
            # 重新初始化交易所连接
            from src.trading.exchange_manager import ExchangeManager
            exchange_manager = ExchangeManager()
            await exchange_manager.reconnect_all()
            logger.info("🏦 交易所连接已恢复")
            return True
        except Exception as e:
            logger.error(f"❌ 交易所连接恢复失败: {e}")
            return False

class AIModelErrorRecovery(ErrorRecoveryStrategy):
    """AI模型错误恢复策略"""
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """AI模型错误恢复"""
        try:
            # 重新加载AI模型
            from src.ai.ai_model_manager import AIModelManager
            model_manager = AIModelManager()
            await model_manager.reload_models()
            logger.info("🤖 AI模型已重新加载")
            return True
        except Exception as e:
            logger.error(f"❌ AI模型恢复失败: {e}")
            return False

class ErrorNotificationService:
    """错误通知服务"""
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        self.notification_enabled = bool(self.smtp_username or self.slack_webhook)
    
    async def send_email_notification(self, error_context: ErrorContext):
        """发送邮件通知"""
        if not self.smtp_username:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = os.getenv("ERROR_NOTIFICATION_EMAIL", self.smtp_username)
            msg['Subject'] = f"🚨 888-888-88 系统错误 - {error_context.severity.value.upper()}"
            
            body = f"""
            错误详情:
            - 错误ID: {error_context.error_id}
            - 时间: {error_context.timestamp}
            - 严重程度: {error_context.severity.value}
            - 分类: {error_context.category.value}
            - 模块: {error_context.module_name}
            - 函数: {error_context.function_name}
            - 错误信息: {error_context.error_message}
            - 恢复尝试: {'是' if error_context.recovery_attempted else '否'}
            - 恢复成功: {'是' if error_context.recovery_successful else '否'}
            
            堆栈跟踪:
            {error_context.stack_trace}
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 错误通知邮件已发送: {error_context.error_id}")
            
        except Exception as e:
            logger.error(f"❌ 发送邮件通知失败: {e}")
    
    async def send_slack_notification(self, error_context: ErrorContext):
        """发送Slack通知"""
        if not self.slack_webhook:
            return
        
        try:
            severity_emoji = {
                ErrorSeverity.LOW: "🟡",
                ErrorSeverity.MEDIUM: "🟠", 
                ErrorSeverity.HIGH: "🔴",
                ErrorSeverity.CRITICAL: "🚨"
            }
            
            payload = {
                "text": f"{severity_emoji.get(error_context.severity, '⚠️')} 888-888-88 系统错误",
                "attachments": [{
                    "color": "danger" if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else "warning",
                    "fields": [
                        {"title": "错误ID", "value": error_context.error_id, "short": True},
                        {"title": "严重程度", "value": error_context.severity.value, "short": True},
                        {"title": "分类", "value": error_context.category.value, "short": True},
                        {"title": "模块", "value": error_context.module_name, "short": True},
                        {"title": "函数", "value": error_context.function_name, "short": True},
                        {"title": "时间", "value": error_context.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "错误信息", "value": error_context.error_message[:500], "short": False},
                        {"title": "恢复状态", "value": f"尝试: {'是' if error_context.recovery_attempted else '否'}, 成功: {'是' if error_context.recovery_successful else '否'}", "short": False}
                    ]
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"📱 Slack通知已发送: {error_context.error_id}")
            else:
                logger.error(f"❌ Slack通知发送失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ 发送Slack通知失败: {e}")

class ErrorHandlingSystem:
    """错误处理系统"""
    
    def __init__(self):
        self.recovery_strategies = {
            ErrorCategory.NETWORK: NetworkErrorRecovery(),
            ErrorCategory.DATABASE: DatabaseErrorRecovery(),
            ErrorCategory.EXCHANGE_API: ExchangeAPIErrorRecovery(),
            ErrorCategory.AI_MODEL: AIModelErrorRecovery(),
        }
        self.notification_service = ErrorNotificationService()
        self.error_history: List[ErrorContext] = []
        self.error_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "last_24h_errors": 0
        }
        
        # 创建错误日志目录
        self.error_log_dir = Path("logs/errors")
        self.error_log_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_error_id(self) -> str:
        """生成错误ID"""
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_history):04d}"
    
    def categorize_error(self, error: Exception, function_name: str) -> ErrorCategory:
        """错误分类"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['database', 'sql', 'connection pool']):
            return ErrorCategory.DATABASE
        elif any(keyword in error_message for keyword in ['api', 'exchange', 'binance', 'okx', 'bybit']):
            return ErrorCategory.EXCHANGE_API
        elif any(keyword in error_message for keyword in ['model', 'prediction', 'ai', 'ml']):
            return ErrorCategory.AI_MODEL
        elif any(keyword in error_message for keyword in ['auth', 'permission', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        else:
            return ErrorCategory.SYSTEM
    
    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """确定错误严重程度"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # 关键错误
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'emergency']):
            return ErrorSeverity.CRITICAL
        
        # 高严重性错误
        if category in [ErrorCategory.TRADING, ErrorCategory.EXCHANGE_API]:
            return ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ['security', 'unauthorized', 'permission']):
            return ErrorSeverity.HIGH
        
        # 中等严重性错误
        if category in [ErrorCategory.DATABASE, ErrorCategory.AI_MODEL]:
            return ErrorSeverity.MEDIUM
        elif error_type in ['ConnectionError', 'TimeoutError']:
            return ErrorSeverity.MEDIUM
        
        # 低严重性错误
        return ErrorSeverity.LOW
    
    async def handle_error(self, 
                          error: Exception, 
                          function_name: str, 
                          module_name: str,
                          user_data: Optional[Dict[str, Any]] = None,
                          system_state: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """处理错误"""
        
        # 创建错误上下文
        category = self.categorize_error(error, function_name)
        severity = self.determine_severity(error, category)
        
        error_context = ErrorContext(
            timestamp=datetime.now(),
            error_id=self.generate_error_id(),
            severity=severity,
            category=category,
            function_name=function_name,
            module_name=module_name,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            user_data=user_data or {},
            system_state=system_state or {}
        )
        
        # 记录错误
        self.error_history.append(error_context)
        self.error_stats["total_errors"] += 1
        
        if severity == ErrorSeverity.CRITICAL:
            self.error_stats["critical_errors"] += 1
        
        # 记录到日志文件
        await self._log_error_to_file(error_context)
        
        # 尝试错误恢复
        if category in self.recovery_strategies:
            try:
                error_context.recovery_attempted = True
                recovery_successful = await self.recovery_strategies[category].recover(error_context)
                error_context.recovery_successful = recovery_successful
                
                if recovery_successful:
                    self.error_stats["recovered_errors"] += 1
                    logger.info(f"✅ 错误恢复成功: {error_context.error_id}")
                else:
                    logger.warning(f"⚠️ 错误恢复失败: {error_context.error_id}")
                    
            except Exception as recovery_error:
                logger.error(f"❌ 错误恢复过程中发生异常: {recovery_error}")
        
        # 发送通知
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self.notification_service.send_email_notification(error_context)
            await self.notification_service.send_slack_notification(error_context)
        
        # 记录错误日志
        logger.error(f"🚨 系统错误 [{error_context.error_id}] {severity.value}: {error}")
        
        return error_context
    
    async def _log_error_to_file(self, error_context: ErrorContext):
        """记录错误到文件"""
        try:
            log_file = self.error_log_dir / f"error_{error_context.timestamp.strftime('%Y%m%d')}.json"
            
            error_data = {
                "error_id": error_context.error_id,
                "timestamp": error_context.timestamp.isoformat(),
                "severity": error_context.severity.value,
                "category": error_context.category.value,
                "function_name": error_context.function_name,
                "module_name": error_context.module_name,
                "error_message": error_context.error_message,
                "stack_trace": error_context.stack_trace,
                "user_data": error_context.user_data,
                "system_state": error_context.system_state,
                "recovery_attempted": error_context.recovery_attempted,
                "recovery_successful": error_context.recovery_successful
            }
            
            # 追加到日志文件
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(error_data)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"❌ 记录错误日志失败: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        # 统计最近24小时的错误
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= last_24h
        ]
        
        self.error_stats["last_24h_errors"] = len(recent_errors)
        
        # 按分类统计
        category_stats = {}
        for category in ErrorCategory:
            category_errors = [
                error for error in self.error_history 
                if error.category == category
            ]
            category_stats[category.value] = len(category_errors)
        
        # 按严重程度统计
        severity_stats = {}
        for severity in ErrorSeverity:
            severity_errors = [
                error for error in self.error_history 
                if error.severity == severity
            ]
            severity_stats[severity.value] = len(severity_errors)
        
        return {
            **self.error_stats,
            "category_stats": category_stats,
            "severity_stats": severity_stats,
            "recovery_rate": (self.error_stats["recovered_errors"] / max(1, self.error_stats["total_errors"])) * 100
        }

# 全局错误处理系统实例
error_handler = ErrorHandlingSystem()

def handle_errors(category: Optional[ErrorCategory] = None, 
                 severity: Optional[ErrorSeverity] = None,
                 notify: bool = True):
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = await error_handler.handle_error(
                    error=e,
                    function_name=func.__name__,
                    module_name=func.__module__,
                    user_data={"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
                )
                
                # 根据严重程度决定是否重新抛出异常
                if error_context.severity == ErrorSeverity.CRITICAL:
                    raise
                elif error_context.recovery_successful:
                    logger.info(f"✅ 函数 {func.__name__} 错误已恢复，继续执行")
                    return await func(*args, **kwargs)  # 重试一次
                else:
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 同步函数的错误处理
                asyncio.create_task(error_handler.handle_error(
                    error=e,
                    function_name=func.__name__,
                    module_name=func.__module__,
                    user_data={"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
                ))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def critical_section(func: Callable) -> Callable:
    """关键代码段装饰器 - 任何错误都会触发紧急处理"""
    return handle_errors(severity=ErrorSeverity.CRITICAL, notify=True)(func)

def trading_operation(func: Callable) -> Callable:
    """交易操作装饰器 - 交易相关的错误处理"""
    return handle_errors(category=ErrorCategory.TRADING, severity=ErrorSeverity.HIGH, notify=True)(func)

def ai_operation(func: Callable) -> Callable:
    """AI操作装饰器 - AI相关的错误处理"""
    return handle_errors(category=ErrorCategory.AI_MODEL, severity=ErrorSeverity.MEDIUM, notify=False)(func)

# 导出主要组件
__all__ = [
    'ErrorHandlingSystem',
    'ErrorSeverity', 
    'ErrorCategory',
    'ErrorContext',
    'handle_errors',
    'critical_section',
    'trading_operation', 
    'ai_operation',
    'error_handler'
]

