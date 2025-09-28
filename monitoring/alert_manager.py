"""
🚨 AlertManager告警管理系统
生产级告警管理，实现邮件、Slack、钉钉等多渠道通知
支持告警规则、告警分组、告警抑制和告警路由
"""

import asyncio
import json
import time
import smtplib
import ssl
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

from loguru import logger
from src.core.config import settings


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"       # 触发中
    RESOLVED = "resolved"   # 已解决
    PENDING = "pending"     # 待处理
    SILENCED = "silenced"   # 已静默


@dataclass
class Alert:
    """告警对象"""
    id: str                                 # 告警ID
    name: str                               # 告警名称
    description: str                        # 告警描述
    severity: AlertSeverity                 # 严重程度
    status: AlertStatus                     # 告警状态
    labels: Dict[str, str] = field(default_factory=dict)  # 标签
    annotations: Dict[str, str] = field(default_factory=dict)  # 注释
    starts_at: float = field(default_factory=time.time)  # 开始时间
    ends_at: Optional[float] = None         # 结束时间
    generator_url: str = ""                 # 生成器URL
    fingerprint: str = ""                   # 指纹
    source: str = "trading-system"          # 来源


@dataclass
class NotificationChannel:
    """通知渠道配置"""
    name: str                               # 渠道名称
    type: str                               # 渠道类型: email/slack/dingtalk/webhook
    config: Dict[str, Any] = field(default_factory=dict)  # 渠道配置
    enabled: bool = True                    # 是否启用
    severity_filter: List[AlertSeverity] = field(default_factory=list)  # 严重程度过滤


class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
        
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警邮件"""
        try:
            # 创建邮件内容
            subject = f"[{alert.severity.value.upper()}] {alert.name}"
            
            html_content = self._create_html_content(alert)
            text_content = self._create_text_content(alert)
            
            # 创建邮件消息
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.from_email
            message["To"] = ", ".join(self.to_emails)
            
            # 添加文本和HTML内容
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # 发送邮件
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, message.as_string())
            
            logger.info(f"邮件告警发送成功: {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """创建HTML邮件内容"""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>告警通知</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .alert-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .label {{ display: inline-block; background-color: #e9ecef; padding: 2px 8px; border-radius: 3px; margin: 2px; font-size: 12px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 {alert.severity.value.upper()} 告警</h1>
                    <h2>{alert.name}</h2>
                </div>
                <div class="content">
                    <div class="alert-info">
                        <h3>告警详情</h3>
                        <p><strong>描述:</strong> {alert.description}</p>
                        <p><strong>状态:</strong> {alert.status.value}</p>
                        <p><strong>开始时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.starts_at))}</p>
                        <p><strong>来源:</strong> {alert.source}</p>
                    </div>
                    
                    {self._format_labels_html(alert.labels)}
                    {self._format_annotations_html(alert.annotations)}
                </div>
                <div class="footer">
                    <p>此邮件由AI量化交易系统自动发送</p>
                    <p>时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_content(self, alert: Alert) -> str:
        """创建文本邮件内容"""
        content = f"""
🚨 {alert.severity.value.upper()} 告警

告警名称: {alert.name}
告警描述: {alert.description}
告警状态: {alert.status.value}
开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.starts_at))}
来源系统: {alert.source}

标签信息:
{self._format_labels_text(alert.labels)}

注释信息:
{self._format_annotations_text(alert.annotations)}

---
此邮件由AI量化交易系统自动发送
发送时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return content.strip()
    
    def _format_labels_html(self, labels: Dict[str, str]) -> str:
        """格式化标签HTML"""
        if not labels:
            return ""
        
        html = "<div class='alert-info'><h3>标签</h3>"
        for key, value in labels.items():
            html += f"<span class='label'>{key}: {value}</span>"
        html += "</div>"
        
        return html
    
    def _format_annotations_html(self, annotations: Dict[str, str]) -> str:
        """格式化注释HTML"""
        if not annotations:
            return ""
        
        html = "<div class='alert-info'><h3>注释</h3>"
        for key, value in annotations.items():
            html += f"<p><strong>{key}:</strong> {value}</p>"
        html += "</div>"
        
        return html
    
    def _format_labels_text(self, labels: Dict[str, str]) -> str:
        """格式化标签文本"""
        if not labels:
            return "无"
        
        return "\n".join([f"  {key}: {value}" for key, value in labels.items()])
    
    def _format_annotations_text(self, annotations: Dict[str, str]) -> str:
        """格式化注释文本"""
        if not annotations:
            return "无"
        
        return "\n".join([f"  {key}: {value}" for key, value in annotations.items()])


class SlackNotifier:
    """Slack通知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'AlertBot')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
        
    async def send_alert(self, alert: Alert) -> bool:
        """发送Slack告警"""
        if not HTTP_AVAILABLE:
            logger.warning("HTTP库不可用，无法发送Slack通知")
            return False
        
        try:
            # 创建Slack消息
            payload = self._create_slack_payload(alert)
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack告警发送成功: {alert.name}")
                        return True
                    else:
                        logger.error(f"Slack告警发送失败: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Slack告警发送失败: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """创建Slack消息载荷"""
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        
        color = severity_colors.get(alert.severity, "#cccccc")
        
        # 创建字段
        fields = [
            {
                "title": "严重程度",
                "value": alert.severity.value.upper(),
                "short": True
            },
            {
                "title": "状态",
                "value": alert.status.value,
                "short": True
            },
            {
                "title": "开始时间",
                "value": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.starts_at)),
                "short": True
            },
            {
                "title": "来源",
                "value": alert.source,
                "short": True
            }
        ]
        
        # 添加标签字段
        if alert.labels:
            labels_text = "\n".join([f"• {k}: {v}" for k, v in alert.labels.items()])
            fields.append({
                "title": "标签",
                "value": labels_text,
                "short": False
            })
        
        # 创建附件
        attachment = {
            "color": color,
            "title": f"🚨 {alert.name}",
            "text": alert.description,
            "fields": fields,
            "footer": "AI量化交易系统",
            "ts": int(alert.starts_at)
        }
        
        # 创建完整载荷
        payload = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment]
        }
        
        return payload


class DingTalkNotifier:
    """钉钉通知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url', '')
        self.secret = config.get('secret', '')
        
    async def send_alert(self, alert: Alert) -> bool:
        """发送钉钉告警"""
        if not HTTP_AVAILABLE:
            logger.warning("HTTP库不可用，无法发送钉钉通知")
            return False
        
        try:
            # 创建钉钉消息
            payload = self._create_dingtalk_payload(alert)
            
            # 计算签名
            if self.secret:
                timestamp = str(round(time.time() * 1000))
                sign = self._calculate_sign(timestamp)
                url = f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"
            else:
                url = self.webhook_url
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('errcode') == 0:
                            logger.info(f"钉钉告警发送成功: {alert.name}")
                            return True
                        else:
                            logger.error(f"钉钉告警发送失败: {result.get('errmsg')}")
                            return False
                    else:
                        logger.error(f"钉钉告警发送失败: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"钉钉告警发送失败: {e}")
            return False
    
    def _create_dingtalk_payload(self, alert: Alert) -> Dict[str, Any]:
        """创建钉钉消息载荷"""
        # 创建Markdown内容
        markdown_text = f"""
# 🚨 {alert.severity.value.upper()} 告警

**告警名称:** {alert.name}

**告警描述:** {alert.description}

**告警状态:** {alert.status.value}

**开始时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.starts_at))}

**来源系统:** {alert.source}

"""
        
        # 添加标签信息
        if alert.labels:
            markdown_text += "**标签信息:**\n"
            for key, value in alert.labels.items():
                markdown_text += f"- {key}: {value}\n"
            markdown_text += "\n"
        
        # 添加注释信息
        if alert.annotations:
            markdown_text += "**注释信息:**\n"
            for key, value in alert.annotations.items():
                markdown_text += f"- {key}: {value}\n"
            markdown_text += "\n"
        
        markdown_text += f"---\n*发送时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        
        # 创建载荷
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{alert.severity.value.upper()} 告警: {alert.name}",
                "text": markdown_text
            }
        }
        
        return payload
    
    def _calculate_sign(self, timestamp: str) -> str:
        """计算钉钉签名"""
        import hmac
        import hashlib
        import base64
        import urllib.parse
        
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        
        return sign


class WebhookNotifier:
    """Webhook通知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.url = config.get('url', '')
        self.method = config.get('method', 'POST')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
        
    async def send_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        if not HTTP_AVAILABLE:
            logger.warning("HTTP库不可用，无法发送Webhook通知")
            return False
        
        try:
            # 创建载荷
            payload = {
                'id': alert.id,
                'name': alert.name,
                'description': alert.description,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'labels': alert.labels,
                'annotations': alert.annotations,
                'starts_at': alert.starts_at,
                'ends_at': alert.ends_at,
                'source': alert.source,
                'timestamp': time.time()
            }
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    self.method,
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook告警发送成功: {alert.name}")
                        return True
                    else:
                        logger.error(f"Webhook告警发送失败: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Webhook告警发送失败: {e}")
            return False


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.notifiers: Dict[str, Any] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 10000
        
        # 告警规则
        self.alert_rules: List[Dict[str, Any]] = []
        
        # 告警抑制
        self.inhibit_rules: List[Dict[str, Any]] = []
        
        # 告警分组
        self.group_rules: List[Dict[str, Any]] = []
        
        # 静默规则
        self.silence_rules: List[Dict[str, Any]] = []
        
        logger.info("告警管理器初始化完成")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        try:
            self.channels[channel.name] = channel
            
            # 创建对应的通知器
            if channel.type == 'email':
                self.notifiers[channel.name] = EmailNotifier(channel.config)
            elif channel.type == 'slack':
                self.notifiers[channel.name] = SlackNotifier(channel.config)
            elif channel.type == 'dingtalk':
                self.notifiers[channel.name] = DingTalkNotifier(channel.config)
            elif channel.type == 'webhook':
                self.notifiers[channel.name] = WebhookNotifier(channel.config)
            else:
                logger.error(f"不支持的通知渠道类型: {channel.type}")
                return
            
            logger.info(f"添加通知渠道: {channel.name} ({channel.type})")
            
        except Exception as e:
            logger.error(f"添加通知渠道失败: {e}")
    
    async def send_alert(self, alert: Alert):
        """发送告警"""
        try:
            # 检查是否被静默
            if self._is_silenced(alert):
                logger.info(f"告警被静默: {alert.name}")
                return
            
            # 检查是否被抑制
            if self._is_inhibited(alert):
                logger.info(f"告警被抑制: {alert.name}")
                return
            
            # 更新活跃告警
            self.active_alerts[alert.id] = alert
            
            # 添加到历史记录
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # 发送到所有启用的通知渠道
            for channel_name, channel in self.channels.items():
                if not channel.enabled:
                    continue
                
                # 检查严重程度过滤
                if channel.severity_filter and alert.severity not in channel.severity_filter:
                    continue
                
                # 发送通知
                notifier = self.notifiers.get(channel_name)
                if notifier:
                    try:
                        await notifier.send_alert(alert)
                    except Exception as e:
                        logger.error(f"通知渠道 {channel_name} 发送失败: {e}")
            
            logger.info(f"告警发送完成: {alert.name}")
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.ends_at = time.time()
                
                # 从活跃告警中移除
                del self.active_alerts[alert_id]
                
                logger.info(f"告警已解决: {alert.name}")
                
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
    
    def _is_silenced(self, alert: Alert) -> bool:
        """检查告警是否被静默"""
        for rule in self.silence_rules:
            if self._match_rule(alert, rule):
                return True
        return False
    
    def _is_inhibited(self, alert: Alert) -> bool:
        """检查告警是否被抑制"""
        for rule in self.inhibit_rules:
            # 检查是否有更高优先级的告警
            source_match = rule.get('source_match', {})
            target_match = rule.get('target_match', {})
            
            # 检查当前告警是否匹配目标
            if self._match_labels(alert.labels, target_match):
                # 检查是否有匹配源的活跃告警
                for active_alert in self.active_alerts.values():
                    if self._match_labels(active_alert.labels, source_match):
                        return True
        
        return False
    
    def _match_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """匹配规则"""
        matchers = rule.get('matchers', {})
        return self._match_labels(alert.labels, matchers)
    
    def _match_labels(self, labels: Dict[str, str], matchers: Dict[str, str]) -> bool:
        """匹配标签"""
        for key, value in matchers.items():
            if key not in labels or labels[key] != value:
                return False
        return True
    
    def add_silence_rule(self, matchers: Dict[str, str], duration: int = 3600):
        """添加静默规则"""
        rule = {
            'matchers': matchers,
            'starts_at': time.time(),
            'ends_at': time.time() + duration
        }
        self.silence_rules.append(rule)
        logger.info(f"添加静默规则: {matchers}")
    
    def add_inhibit_rule(self, source_match: Dict[str, str], target_match: Dict[str, str]):
        """添加抑制规则"""
        rule = {
            'source_match': source_match,
            'target_match': target_match
        }
        self.inhibit_rules.append(rule)
        logger.info(f"添加抑制规则: {source_match} -> {target_match}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self.alert_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_history': len(self.alert_history),
            'notification_channels': len(self.channels),
            'silence_rules': len(self.silence_rules),
            'inhibit_rules': len(self.inhibit_rules),
            'group_rules': len(self.group_rules)
        }


# 全局告警管理器实例
alert_manager = AlertManager()
