#!/usr/bin/env python3
"""
🚨 告警系统 - 生产级告警管理
Alert System - Production-Grade Alert Management

生产级特性：
- 多渠道告警通知
- 告警升级机制
- 告警抑制和去重
- 告警恢复通知
- 告警统计分析
"""

import asyncio
import smtplib
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import logging

from .unified_logging_system import UnifiedLogger

@dataclass
class AlertChannel:
    """告警通道配置"""
    name: str
    type: str  # 'email', 'webhook', 'sms', 'telegram'
    config: Dict[str, Any]
    enabled: bool = True
    priority_filter: List[str] = None  # 优先级过滤

@dataclass
class Alert:
    """告警数据结构"""
    id: str
    title: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    source: str
    timestamp: datetime
    tags: Dict[str, str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    notification_count: int = 0

class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = UnifiedLogger("EmailNotifier")
    
    async def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """发送告警邮件"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # 构建邮件内容
            body = self._build_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # 发送邮件
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config.get('use_tls', True):
                    server.starttls()
                if self.config.get('username') and self.config.get('password'):
                    server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            
            self.logger.info(f"告警邮件发送成功: {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送告警邮件失败: {e}")
            return False
    
    def _build_email_body(self, alert: Alert) -> str:
        """构建邮件内容"""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107', 
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">🚨 系统告警</h2>
                    <p style="margin: 5px 0 0 0;">严重程度: {alert.severity.upper()}</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                    <h3 style="color: #495057; margin-top: 0;">{alert.title}</h3>
                    <p style="color: #6c757d; line-height: 1.6;">{alert.message}</p>
                    
                    <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                        <tr>
                            <td style="padding: 8px; border: 1px solid #dee2e6; background-color: #e9ecef; font-weight: bold;">告警ID</td>
                            <td style="padding: 8px; border: 1px solid #dee2e6;">{alert.id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #dee2e6; background-color: #e9ecef; font-weight: bold;">来源</td>
                            <td style="padding: 8px; border: 1px solid #dee2e6;">{alert.source}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #dee2e6; background-color: #e9ecef; font-weight: bold;">时间</td>
                            <td style="padding: 8px; border: 1px solid #dee2e6;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                    </table>
                    
                    {self._build_tags_html(alert.tags) if alert.tags else ''}
                </div>
                
                <div style="background-color: #e9ecef; padding: 15px; border-radius: 0 0 5px 5px; text-align: center; color: #6c757d;">
                    <small>此邮件由量化交易系统自动发送</small>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _build_tags_html(self, tags: Dict[str, str]) -> str:
        """构建标签HTML"""
        if not tags:
            return ""
        
        tags_html = "<div style='margin-top: 15px;'><strong>标签:</strong><br>"
        for key, value in tags.items():
            tags_html += f"<span style='background-color: #007bff; color: white; padding: 2px 8px; border-radius: 3px; margin: 2px; display: inline-block;'>{key}: {value}</span>"
        tags_html += "</div>"
        return tags_html

class WebhookNotifier:
    """Webhook通知器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = UnifiedLogger("WebhookNotifier")
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'tags': alert.tags or {},
                'resolved': alert.resolved
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'QuantTradingSystem/1.0'
            }
            
            # 添加自定义头部
            if 'headers' in self.config:
                headers.update(self.config['headers'])
            
            response = requests.post(
                self.config['url'],
                json=payload,
                headers=headers,
                timeout=self.config.get('timeout', 30)
            )
            
            response.raise_for_status()
            self.logger.info(f"Webhook告警发送成功: {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送Webhook告警失败: {e}")
            return False

class TelegramNotifier:
    """Telegram通知器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = UnifiedLogger("TelegramNotifier")
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送Telegram告警"""
        try:
            severity_emojis = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }
            
            emoji = severity_emojis.get(alert.severity, '⚪')
            
            message = f"""
{emoji} *系统告警*

*标题:* {alert.title}
*严重程度:* {alert.severity.upper()}
*来源:* {alert.source}
*时间:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

*详情:*
{alert.message}

*告警ID:* `{alert.id}`
            """.strip()
            
            url = f"https://api.telegram.org/bot{self.config['bot_token']}/sendMessage"
            payload = {
                'chat_id': self.config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            self.logger.info(f"Telegram告警发送成功: {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送Telegram告警失败: {e}")
            return False

class AlertManager:
    """告警管理器"""
    
    def __init__(self, config_file: str = None):
        self.logger = UnifiedLogger("AlertManager")
        self.config = self._load_config(config_file)
        
        # 告警存储
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=10000)
        self.suppressed_alerts = set()
        
        # 通知器
        self.notifiers = self._initialize_notifiers()
        
        # 告警统计
        self.alert_stats = defaultdict(int)
        self.escalation_rules = self._load_escalation_rules()
        
        # 去重配置
        self.dedup_window = self.config.get('dedup_window', 300)  # 5分钟
        self.dedup_cache = {}
        
        self.logger.info("告警管理器初始化完成")
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置"""
        default_config = {
            'channels': [],
            'escalation_enabled': True,
            'escalation_interval': 1800,  # 30分钟
            'max_escalation_level': 3,
            'dedup_window': 300,
            'auto_resolve_timeout': 3600,  # 1小时
            'notification_rate_limit': 10  # 每分钟最多10条
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
        
        return default_config
    
    def _initialize_notifiers(self) -> Dict:
        """初始化通知器"""
        notifiers = {}
        
        for channel in self.config.get('channels', []):
            if not channel.get('enabled', True):
                continue
                
            channel_type = channel['type']
            channel_config = channel['config']
            
            try:
                if channel_type == 'email':
                    notifiers[channel['name']] = EmailNotifier(channel_config)
                elif channel_type == 'webhook':
                    notifiers[channel['name']] = WebhookNotifier(channel_config)
                elif channel_type == 'telegram':
                    notifiers[channel['name']] = TelegramNotifier(channel_config)
                else:
                    self.logger.warning(f"不支持的通知渠道类型: {channel_type}")
                    
            except Exception as e:
                self.logger.error(f"初始化通知器 {channel['name']} 失败: {e}")
        
        return notifiers
    
    def _load_escalation_rules(self) -> List[Dict]:
        """加载升级规则"""
        return [
            {
                'level': 1,
                'delay': 900,  # 15分钟
                'channels': ['email'],
                'severity_filter': ['medium', 'high', 'critical']
            },
            {
                'level': 2,
                'delay': 1800,  # 30分钟
                'channels': ['email', 'telegram'],
                'severity_filter': ['high', 'critical']
            },
            {
                'level': 3,
                'delay': 3600,  # 1小时
                'channels': ['email', 'telegram', 'webhook'],
                'severity_filter': ['critical']
            }
        ]
    
    async def create_alert(self, 
                          title: str,
                          message: str,
                          severity: str,
                          source: str,
                          tags: Dict[str, str] = None) -> str:
        """创建告警"""
        
        # 生成告警ID
        alert_id = f"{source}_{int(time.time() * 1000)}"
        
        # 检查去重
        dedup_key = self._generate_dedup_key(title, source, tags)
        if self._is_duplicate(dedup_key):
            self.logger.debug(f"告警去重: {dedup_key}")
            return None
        
        # 创建告警对象
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        # 存储告警
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 更新统计
        self.alert_stats[f"total_{severity}"] += 1
        self.alert_stats["total_alerts"] += 1
        
        # 发送通知
        await self._send_notifications(alert)
        
        self.logger.info(f"创建告警: {alert_id} - {title}")
        return alert_id
    
    async def resolve_alert(self, alert_id: str, message: str = None) -> bool:
        """解决告警"""
        if alert_id not in self.active_alerts:
            self.logger.warning(f"告警不存在: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        # 发送恢复通知
        recovery_alert = Alert(
            id=f"{alert_id}_recovery",
            title=f"✅ 告警恢复: {alert.title}",
            message=message or f"告警 {alert_id} 已恢复",
            severity="info",
            source=alert.source,
            timestamp=datetime.now(),
            tags=alert.tags
        )
        
        await self._send_notifications(recovery_alert)
        
        # 从活跃告警中移除
        del self.active_alerts[alert_id]
        
        self.logger.info(f"告警已解决: {alert_id}")
        return True
    
    def _generate_dedup_key(self, title: str, source: str, tags: Dict[str, str]) -> str:
        """生成去重键"""
        tag_str = json.dumps(tags or {}, sort_keys=True)
        return f"{source}:{title}:{hash(tag_str)}"
    
    def _is_duplicate(self, dedup_key: str) -> bool:
        """检查是否重复告警"""
        current_time = time.time()
        
        # 清理过期的去重记录
        expired_keys = [
            key for key, timestamp in self.dedup_cache.items()
            if current_time - timestamp > self.dedup_window
        ]
        for key in expired_keys:
            del self.dedup_cache[key]
        
        # 检查是否重复
        if dedup_key in self.dedup_cache:
            return True
        
        # 记录新的去重键
        self.dedup_cache[dedup_key] = current_time
        return False
    
    async def _send_notifications(self, alert: Alert):
        """发送通知"""
        try:
            # 检查是否被抑制
            if alert.id in self.suppressed_alerts:
                return
            
            # 根据严重程度选择通知渠道
            channels_to_notify = self._get_notification_channels(alert.severity)
            
            # 并发发送通知
            tasks = []
            for channel_name in channels_to_notify:
                if channel_name in self.notifiers:
                    notifier = self.notifiers[channel_name]
                    if hasattr(notifier, 'send_alert'):
                        tasks.append(notifier.send_alert(alert))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
                self.logger.info(f"通知发送完成: {success_count}/{len(tasks)} 成功")
            
        except Exception as e:
            self.logger.error(f"发送通知失败: {e}")
    
    def _get_notification_channels(self, severity: str) -> List[str]:
        """根据严重程度获取通知渠道"""
        channel_mapping = {
            'low': ['email'],
            'medium': ['email'],
            'high': ['email', 'telegram'],
            'critical': ['email', 'telegram', 'webhook']
        }
        return channel_mapping.get(severity, ['email'])
    
    def suppress_alert(self, alert_id: str, duration: int = 3600):
        """抑制告警"""
        self.suppressed_alerts.add(alert_id)
        
        # 设置自动取消抑制
        async def auto_unsuppress():
            await asyncio.sleep(duration)
            self.suppressed_alerts.discard(alert_id)
        
        asyncio.create_task(auto_unsuppress())
        self.logger.info(f"告警已抑制: {alert_id}, 持续时间: {duration}秒")
    
    def get_alert_statistics(self) -> Dict:
        """获取告警统计"""
        now = datetime.now()
        
        # 计算最近24小时的告警
        recent_alerts = [
            alert for alert in self.alert_history
            if (now - alert.timestamp).total_seconds() < 86400
        ]
        
        # 按严重程度统计
        severity_stats = defaultdict(int)
        for alert in recent_alerts:
            severity_stats[alert.severity] += 1
        
        # 按来源统计
        source_stats = defaultdict(int)
        for alert in recent_alerts:
            source_stats[alert.source] += 1
        
        return {
            'total_active_alerts': len(self.active_alerts),
            'total_suppressed_alerts': len(self.suppressed_alerts),
            'recent_24h_alerts': len(recent_alerts),
            'severity_distribution': dict(severity_stats),
            'source_distribution': dict(source_stats),
            'alert_rate_per_hour': len(recent_alerts) / 24,
            'top_alert_sources': sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def export_alert_data(self, filepath: str, hours: int = 24):
        """导出告警数据"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 过滤指定时间范围的告警
            filtered_alerts = [
                asdict(alert) for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'total_alerts': len(filtered_alerts),
                'alerts': filtered_alerts,
                'statistics': self.get_alert_statistics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"告警数据已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出告警数据失败: {e}")

# 告警装饰器
def alert_on_exception(alert_manager: AlertManager, severity: str = 'high'):
    """异常告警装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await alert_manager.create_alert(
                    title=f"函数异常: {func.__name__}",
                    message=f"函数 {func.__name__} 执行异常: {str(e)}",
                    severity=severity,
                    source="function_monitor",
                    tags={'function': func.__name__, 'exception_type': type(e).__name__}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 同步函数需要在事件循环中运行异步代码
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(alert_manager.create_alert(
                        title=f"函数异常: {func.__name__}",
                        message=f"函数 {func.__name__} 执行异常: {str(e)}",
                        severity=severity,
                        source="function_monitor",
                        tags={'function': func.__name__, 'exception_type': type(e).__name__}
                    ))
                except:
                    pass  # 如果无法创建告警，至少不影响原函数的异常抛出
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 使用示例
if __name__ == "__main__":
    async def main():
        # 创建告警管理器
        alert_manager = AlertManager()
        
        # 创建测试告警
        alert_id = await alert_manager.create_alert(
            title="系统测试告警",
            message="这是一个测试告警消息",
            severity="medium",
            source="test_system",
            tags={'environment': 'production', 'service': 'trading'}
        )
        
        print(f"创建告警: {alert_id}")
        
        # 等待一段时间
        await asyncio.sleep(5)
        
        # 解决告警
        if alert_id:
            await alert_manager.resolve_alert(alert_id, "测试完成，告警已解决")
        
        # 获取统计信息
        stats = alert_manager.get_alert_statistics()
        print("告警统计:", json.dumps(stats, indent=2, ensure_ascii=False))
    
    asyncio.run(main())
