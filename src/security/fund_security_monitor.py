"""
💰 资金安全监控系统 - 生产级实盘交易资金安全监控和保护系统
提供账户余额监控、资金流向追踪、异常转账检测、风险预警等全方位资金安全功能
确保交易账户资金安全，防范各类资金风险和异常操作
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class FundEventType(Enum):
    """资金事件类型"""
    DEPOSIT = "deposit"  # 充值
    WITHDRAWAL = "withdrawal"  # 提现
    TRADE_PROFIT = "trade_profit"  # 交易盈利
    TRADE_LOSS = "trade_loss"  # 交易亏损
    FEE = "fee"  # 手续费
    TRANSFER_IN = "transfer_in"  # 转入
    TRANSFER_OUT = "transfer_out"  # 转出
    UNKNOWN = "unknown"  # 未知

class SecurityLevel(Enum):
    """安全级别"""
    SAFE = "safe"  # 安全
    WARNING = "warning"  # 警告
    DANGER = "danger"  # 危险
    CRITICAL = "critical"  # 关键

@dataclass
class FundEvent:
    """资金事件"""
    event_id: str  # 事件ID
    event_type: FundEventType  # 事件类型
    exchange: str  # 交易所
    currency: str  # 币种
    amount: float  # 金额
    balance_before: float  # 变动前余额
    balance_after: float  # 变动后余额
    timestamp: float  # 时间戳
    transaction_id: Optional[str] = None  # 交易ID
    address: Optional[str] = None  # 地址（提现/充值）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class SecurityAlert:
    """安全警报"""
    alert_id: str  # 警报ID
    security_level: SecurityLevel  # 安全级别
    title: str  # 标题
    description: str  # 描述
    exchange: str  # 交易所
    currency: str  # 币种
    amount: float  # 涉及金额
    timestamp: float = field(default_factory=time.time)  # 时间戳
    is_resolved: bool = False  # 是否已解决
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class AccountBalance:
    """账户余额"""
    exchange: str  # 交易所
    currency: str  # 币种
    total_balance: float  # 总余额
    available_balance: float  # 可用余额
    frozen_balance: float  # 冻结余额
    timestamp: float = field(default_factory=time.time)  # 时间戳

class BalanceMonitor:
    """余额监控器"""
    
    def __init__(self):
        self.balances: Dict[str, Dict[str, AccountBalance]] = {}  # exchange -> currency -> balance
        self.balance_history: Dict[str, List[AccountBalance]] = {}  # key -> history
        self.balance_thresholds: Dict[str, float] = {}  # 余额阈值
        
        # 默认阈值设置
        self.default_min_balance_threshold = 0.01  # 最小余额阈值
        self.balance_change_threshold = 0.1  # 余额变化阈值 (10%)
        
        logger.info("余额监控器初始化完成")
    
    def update_balance(self, balance: AccountBalance) -> List[SecurityAlert]:
        """更新余额并检查安全"""
        alerts = []
        
        try:
            # 初始化存储结构
            if balance.exchange not in self.balances:
                self.balances[balance.exchange] = {}
            
            balance_key = f"{balance.exchange}_{balance.currency}"
            
            # 获取之前的余额
            previous_balance = self.balances[balance.exchange].get(balance.currency)
            
            # 更新当前余额
            self.balances[balance.exchange][balance.currency] = balance
            
            # 添加到历史记录
            if balance_key not in self.balance_history:
                self.balance_history[balance_key] = []
            
            self.balance_history[balance_key].append(balance)
            
            # 保持历史记录在合理范围内
            if len(self.balance_history[balance_key]) > 1000:
                self.balance_history[balance_key] = self.balance_history[balance_key][-500:]
            
            # 检查余额安全
            alerts.extend(self._check_balance_security(balance, previous_balance))
            
            return alerts
        
        except Exception as e:
            logger.error(f"更新余额失败: {e}")
            return []
    
    def _check_balance_security(self, current: AccountBalance, 
                               previous: Optional[AccountBalance]) -> List[SecurityAlert]:
        """检查余额安全"""
        alerts = []
        
        try:
            # 检查最小余额
            min_threshold = self.balance_thresholds.get(
                f"{current.exchange}_{current.currency}", 
                self.default_min_balance_threshold
            )
            
            if current.total_balance < min_threshold:
                alerts.append(SecurityAlert(
                    alert_id=f"low_balance_{current.exchange}_{current.currency}_{int(time.time())}",
                    security_level=SecurityLevel.WARNING,
                    title="余额过低",
                    description=f"{current.exchange} {current.currency} 余额过低: {current.total_balance}",
                    exchange=current.exchange,
                    currency=current.currency,
                    amount=current.total_balance,
                    metadata={'threshold': min_threshold}
                ))
            
            # 检查余额异常变化
            if previous:
                balance_change = abs(current.total_balance - previous.total_balance)
                change_ratio = balance_change / previous.total_balance if previous.total_balance > 0 else 0
                
                if change_ratio > self.balance_change_threshold:
                    security_level = SecurityLevel.DANGER if change_ratio > 0.5 else SecurityLevel.WARNING
                    
                    alerts.append(SecurityAlert(
                        alert_id=f"balance_change_{current.exchange}_{current.currency}_{int(time.time())}",
                        security_level=security_level,
                        title="余额异常变化",
                        description=f"{current.exchange} {current.currency} 余额异常变化: {change_ratio:.2%}",
                        exchange=current.exchange,
                        currency=current.currency,
                        amount=balance_change,
                        metadata={
                            'previous_balance': previous.total_balance,
                            'current_balance': current.total_balance,
                            'change_ratio': change_ratio
                        }
                    ))
            
            # 检查冻结余额异常
            if current.frozen_balance > current.total_balance * 0.8:
                alerts.append(SecurityAlert(
                    alert_id=f"high_frozen_{current.exchange}_{current.currency}_{int(time.time())}",
                    security_level=SecurityLevel.WARNING,
                    title="冻结余额过高",
                    description=f"{current.exchange} {current.currency} 冻结余额过高: {current.frozen_balance}",
                    exchange=current.exchange,
                    currency=current.currency,
                    amount=current.frozen_balance,
                    metadata={'frozen_ratio': current.frozen_balance / current.total_balance}
                ))
            
            return alerts
        
        except Exception as e:
            logger.error(f"检查余额安全失败: {e}")
            return []
    
    def set_balance_threshold(self, exchange: str, currency: str, threshold: float):
        """设置余额阈值"""
        key = f"{exchange}_{currency}"
        self.balance_thresholds[key] = threshold
        logger.info(f"设置余额阈值: {key} = {threshold}")
    
    def get_balance_summary(self) -> Dict[str, Any]:
        """获取余额摘要"""
        try:
            summary = {
                'total_exchanges': len(self.balances),
                'total_currencies': 0,
                'balances_by_exchange': {},
                'low_balance_alerts': 0
            }
            
            for exchange, currencies in self.balances.items():
                summary['total_currencies'] += len(currencies)
                summary['balances_by_exchange'][exchange] = {}
                
                for currency, balance in currencies.items():
                    summary['balances_by_exchange'][exchange][currency] = {
                        'total_balance': balance.total_balance,
                        'available_balance': balance.available_balance,
                        'frozen_balance': balance.frozen_balance
                    }
                    
                    # 检查低余额
                    min_threshold = self.balance_thresholds.get(
                        f"{exchange}_{currency}", 
                        self.default_min_balance_threshold
                    )
                    if balance.total_balance < min_threshold:
                        summary['low_balance_alerts'] += 1
            
            return summary
        
        except Exception as e:
            logger.error(f"获取余额摘要失败: {e}")
            return {}

class TransactionMonitor:
    """交易监控器"""
    
    def __init__(self):
        self.fund_events: List[FundEvent] = []
        self.suspicious_patterns: Dict[str, Any] = {}
        
        # 监控阈值
        self.large_withdrawal_threshold = 1000  # 大额提现阈值
        self.frequent_transaction_threshold = 10  # 频繁交易阈值
        self.unusual_time_window = 3600  # 异常时间窗口（秒）
        
        logger.info("交易监控器初始化完成")
    
    def record_fund_event(self, event: FundEvent) -> List[SecurityAlert]:
        """记录资金事件并检查安全"""
        alerts = []
        
        try:
            # 添加到事件历史
            self.fund_events.append(event)
            
            # 保持历史记录在合理范围内
            if len(self.fund_events) > 10000:
                self.fund_events = self.fund_events[-5000:]
            
            # 检查可疑模式
            alerts.extend(self._check_suspicious_patterns(event))
            
            return alerts
        
        except Exception as e:
            logger.error(f"记录资金事件失败: {e}")
            return []
    
    def _check_suspicious_patterns(self, event: FundEvent) -> List[SecurityAlert]:
        """检查可疑模式"""
        alerts = []
        current_time = time.time()
        
        try:
            # 检查大额提现
            if event.event_type == FundEventType.WITHDRAWAL and event.amount > self.large_withdrawal_threshold:
                alerts.append(SecurityAlert(
                    alert_id=f"large_withdrawal_{event.exchange}_{int(current_time)}",
                    security_level=SecurityLevel.DANGER,
                    title="大额提现",
                    description=f"检测到大额提现: {event.amount} {event.currency}",
                    exchange=event.exchange,
                    currency=event.currency,
                    amount=event.amount,
                    metadata={'event_id': event.event_id, 'address': event.address}
                ))
            
            # 检查频繁交易
            recent_events = [
                e for e in self.fund_events 
                if (current_time - e.timestamp < self.unusual_time_window and 
                    e.exchange == event.exchange and 
                    e.currency == event.currency)
            ]
            
            if len(recent_events) > self.frequent_transaction_threshold:
                alerts.append(SecurityAlert(
                    alert_id=f"frequent_transactions_{event.exchange}_{int(current_time)}",
                    security_level=SecurityLevel.WARNING,
                    title="频繁交易",
                    description=f"检测到频繁交易: {len(recent_events)}次 在 {self.unusual_time_window/60:.0f}分钟内",
                    exchange=event.exchange,
                    currency=event.currency,
                    amount=sum(e.amount for e in recent_events),
                    metadata={'transaction_count': len(recent_events)}
                ))
            
            # 检查异常时间交易
            if self._is_unusual_time(event.timestamp):
                alerts.append(SecurityAlert(
                    alert_id=f"unusual_time_{event.exchange}_{int(current_time)}",
                    security_level=SecurityLevel.WARNING,
                    title="异常时间交易",
                    description=f"检测到异常时间交易: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}",
                    exchange=event.exchange,
                    currency=event.currency,
                    amount=event.amount,
                    metadata={'event_time': event.timestamp}
                ))
            
            # 检查未知来源交易
            if event.event_type == FundEventType.UNKNOWN and event.amount > 100:
                alerts.append(SecurityAlert(
                    alert_id=f"unknown_transaction_{event.exchange}_{int(current_time)}",
                    security_level=SecurityLevel.DANGER,
                    title="未知来源交易",
                    description=f"检测到未知来源交易: {event.amount} {event.currency}",
                    exchange=event.exchange,
                    currency=event.currency,
                    amount=event.amount,
                    metadata={'event_id': event.event_id}
                ))
            
            return alerts
        
        except Exception as e:
            logger.error(f"检查可疑模式失败: {e}")
            return []
    
    def _is_unusual_time(self, timestamp: float) -> bool:
        """检查是否为异常时间"""
        # 检查是否在深夜时间（凌晨2-6点）
        hour = time.localtime(timestamp).tm_hour
        return 2 <= hour <= 6
    
    def get_transaction_summary(self) -> Dict[str, Any]:
        """获取交易摘要"""
        try:
            if not self.fund_events:
                return {}
            
            # 最近24小时的事件
            recent_events = [
                e for e in self.fund_events 
                if time.time() - e.timestamp < 86400
            ]
            
            # 按类型统计
            event_counts = {}
            total_amounts = {}
            
            for event in recent_events:
                event_type = event.event_type.value
                currency = event.currency
                
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                if currency not in total_amounts:
                    total_amounts[currency] = {}
                total_amounts[currency][event_type] = total_amounts[currency].get(event_type, 0) + event.amount
            
            return {
                'total_events': len(self.fund_events),
                'recent_events_24h': len(recent_events),
                'event_type_counts': event_counts,
                'total_amounts_by_currency': total_amounts
            }
        
        except Exception as e:
            logger.error(f"获取交易摘要失败: {e}")
            return {}

class AddressWhitelist:
    """地址白名单管理"""
    
    def __init__(self):
        self.whitelisted_addresses: Dict[str, Dict[str, List[str]]] = {}  # exchange -> currency -> addresses
        self.address_labels: Dict[str, str] = {}  # address -> label
        
        logger.info("地址白名单管理器初始化完成")
    
    def add_address(self, exchange: str, currency: str, address: str, label: str = ""):
        """添加白名单地址"""
        try:
            if exchange not in self.whitelisted_addresses:
                self.whitelisted_addresses[exchange] = {}
            
            if currency not in self.whitelisted_addresses[exchange]:
                self.whitelisted_addresses[exchange][currency] = []
            
            if address not in self.whitelisted_addresses[exchange][currency]:
                self.whitelisted_addresses[exchange][currency].append(address)
                
                if label:
                    self.address_labels[address] = label
                
                logger.info(f"添加白名单地址: {exchange} {currency} {address} ({label})")
        
        except Exception as e:
            logger.error(f"添加白名单地址失败: {e}")
    
    def remove_address(self, exchange: str, currency: str, address: str):
        """移除白名单地址"""
        try:
            if (exchange in self.whitelisted_addresses and 
                currency in self.whitelisted_addresses[exchange] and
                address in self.whitelisted_addresses[exchange][currency]):
                
                self.whitelisted_addresses[exchange][currency].remove(address)
                self.address_labels.pop(address, None)
                
                logger.info(f"移除白名单地址: {exchange} {currency} {address}")
        
        except Exception as e:
            logger.error(f"移除白名单地址失败: {e}")
    
    def is_whitelisted(self, exchange: str, currency: str, address: str) -> bool:
        """检查地址是否在白名单中"""
        try:
            return (exchange in self.whitelisted_addresses and
                    currency in self.whitelisted_addresses[exchange] and
                    address in self.whitelisted_addresses[exchange][currency])
        
        except Exception as e:
            logger.error(f"检查白名单地址失败: {e}")
            return False
    
    def get_whitelist_summary(self) -> Dict[str, Any]:
        """获取白名单摘要"""
        try:
            summary = {
                'total_exchanges': len(self.whitelisted_addresses),
                'total_addresses': sum(
                    len(addresses) 
                    for exchange_data in self.whitelisted_addresses.values()
                    for addresses in exchange_data.values()
                ),
                'addresses_by_exchange': {}
            }
            
            for exchange, currencies in self.whitelisted_addresses.items():
                summary['addresses_by_exchange'][exchange] = {}
                for currency, addresses in currencies.items():
                    summary['addresses_by_exchange'][exchange][currency] = [
                        {
                            'address': addr,
                            'label': self.address_labels.get(addr, "")
                        }
                        for addr in addresses
                    ]
            
            return summary
        
        except Exception as e:
            logger.error(f"获取白名单摘要失败: {e}")
            return {}

class FundSecurityMonitor:
    """资金安全监控系统主类"""
    
    def __init__(self):
        self.balance_monitor = BalanceMonitor()
        self.transaction_monitor = TransactionMonitor()
        self.address_whitelist = AddressWhitelist()
        
        # 安全警报历史
        self.security_alerts: List[SecurityAlert] = []
        
        # 回调函数
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("资金安全监控系统初始化完成")
    
    def update_account_balance(self, balance: AccountBalance) -> List[SecurityAlert]:
        """更新账户余额"""
        try:
            with self.lock:
                alerts = self.balance_monitor.update_balance(balance)
                self._process_alerts(alerts)
                return alerts
        
        except Exception as e:
            logger.error(f"更新账户余额失败: {e}")
            return []
    
    def record_fund_transaction(self, event: FundEvent) -> List[SecurityAlert]:
        """记录资金交易"""
        try:
            with self.lock:
                alerts = self.transaction_monitor.record_fund_event(event)
                
                # 检查提现地址白名单
                if (event.event_type == FundEventType.WITHDRAWAL and 
                    event.address and 
                    not self.address_whitelist.is_whitelisted(event.exchange, event.currency, event.address)):
                    
                    alerts.append(SecurityAlert(
                        alert_id=f"non_whitelist_withdrawal_{event.exchange}_{int(time.time())}",
                        security_level=SecurityLevel.CRITICAL,
                        title="非白名单地址提现",
                        description=f"检测到向非白名单地址提现: {event.address}",
                        exchange=event.exchange,
                        currency=event.currency,
                        amount=event.amount,
                        metadata={'address': event.address, 'event_id': event.event_id}
                    ))
                
                self._process_alerts(alerts)
                return alerts
        
        except Exception as e:
            logger.error(f"记录资金交易失败: {e}")
            return []
    
    def add_whitelist_address(self, exchange: str, currency: str, address: str, label: str = ""):
        """添加白名单地址"""
        self.address_whitelist.add_address(exchange, currency, address, label)
    
    def remove_whitelist_address(self, exchange: str, currency: str, address: str):
        """移除白名单地址"""
        self.address_whitelist.remove_address(exchange, currency, address)
    
    def set_balance_threshold(self, exchange: str, currency: str, threshold: float):
        """设置余额阈值"""
        self.balance_monitor.set_balance_threshold(exchange, currency, threshold)
    
    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def _process_alerts(self, alerts: List[SecurityAlert]):
        """处理安全警报"""
        for alert in alerts:
            try:
                # 添加到历史记录
                self.security_alerts.append(alert)
                
                # 保持历史记录在合理范围内
                if len(self.security_alerts) > 10000:
                    self.security_alerts = self.security_alerts[-5000:]
                
                # 调用回调函数
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"警报回调函数执行失败: {e}")
                
                # 记录日志
                if alert.security_level == SecurityLevel.CRITICAL:
                    logger.critical(f"关键安全警报: {alert.title} - {alert.description}")
                elif alert.security_level == SecurityLevel.DANGER:
                    logger.error(f"危险安全警报: {alert.title} - {alert.description}")
                elif alert.security_level == SecurityLevel.WARNING:
                    logger.warning(f"警告安全警报: {alert.title} - {alert.description}")
                else:
                    logger.info(f"安全提示: {alert.title} - {alert.description}")
            
            except Exception as e:
                logger.error(f"处理安全警报失败: {e}")
    
    def resolve_alert(self, alert_id: str):
        """解决安全警报"""
        try:
            with self.lock:
                for alert in self.security_alerts:
                    if alert.alert_id == alert_id:
                        alert.is_resolved = True
                        logger.info(f"安全警报已解决: {alert_id}")
                        break
        
        except Exception as e:
            logger.error(f"解决安全警报失败: {e}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """获取安全摘要"""
        try:
            with self.lock:
                # 未解决的警报
                unresolved_alerts = [a for a in self.security_alerts if not a.is_resolved]
                
                # 最近警报统计
                recent_alerts = [a for a in self.security_alerts if time.time() - a.timestamp < 86400]
                
                # 按级别统计
                level_counts = {}
                for alert in recent_alerts:
                    level = alert.security_level.value
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                return {
                    'total_alerts': len(self.security_alerts),
                    'unresolved_alerts': len(unresolved_alerts),
                    'recent_alerts_24h': len(recent_alerts),
                    'alert_level_counts': level_counts,
                    'balance_summary': self.balance_monitor.get_balance_summary(),
                    'transaction_summary': self.transaction_monitor.get_transaction_summary(),
                    'whitelist_summary': self.address_whitelist.get_whitelist_summary()
                }
        
        except Exception as e:
            logger.error(f"获取安全摘要失败: {e}")
            return {}
    
    def get_recent_alerts(self, limit: int = 50) -> List[SecurityAlert]:
        """获取最近的安全警报"""
        with self.lock:
            return sorted(self.security_alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

# 全局资金安全监控系统实例
fund_security_monitor = FundSecurityMonitor()

