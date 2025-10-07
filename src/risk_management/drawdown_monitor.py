#!/usr/bin/env python3
"""
📉 888-888-88 回撤监控系统
生产级回撤监控和保护机制
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from loguru import logger
import json


class DrawdownLevel(Enum):
    """回撤等级"""
    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class ProtectionAction(Enum):
    """保护动作"""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    STOP_NEW_TRADES = "stop_new_trades"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class DrawdownEvent:
    """回撤事件"""
    start_time: datetime
    end_time: Optional[datetime] = None
    peak_value: float = 0.0
    trough_value: float = 0.0
    max_drawdown: float = 0.0
    duration: Optional[timedelta] = None
    recovery_time: Optional[timedelta] = None
    is_active: bool = True


@dataclass
class DrawdownMetrics:
    """回撤指标"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_frequency: float = 0.0
    avg_recovery_time: float = 0.0
    max_recovery_time: float = 0.0
    underwater_time: float = 0.0  # 水下时间占比
    calmar_ratio: float = 0.0
    sterling_ratio: float = 0.0
    pain_index: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProtectionSettings:
    """保护设置"""
    warning_threshold: float = 0.05    # 5%回撤警告
    danger_threshold: float = 0.10     # 10%回撤危险
    critical_threshold: float = 0.15   # 15%回撤临界
    max_daily_drawdown: float = 0.03   # 日最大回撤3%
    max_weekly_drawdown: float = 0.08  # 周最大回撤8%
    max_monthly_drawdown: float = 0.12 # 月最大回撤12%
    recovery_timeout: int = 30         # 恢复超时天数
    position_reduction_pct: float = 0.5 # 仓位削减比例
    enable_auto_protection: bool = True # 启用自动保护


class DrawdownMonitor:
    """生产级回撤监控器"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.settings = ProtectionSettings()
        
        # 历史数据
        self.balance_history: List[Tuple[datetime, float]] = []
        self.drawdown_events: List[DrawdownEvent] = []
        self.daily_returns: List[float] = []
        
        # 当前状态
        self.current_drawdown_event: Optional[DrawdownEvent] = None
        self.protection_level = DrawdownLevel.NORMAL
        self.protection_actions: List[ProtectionAction] = []
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info("📉 回撤监控系统初始化完成")
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔄 回撤监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ 回撤监控已停止")
    
    def update_balance(self, new_balance: float, timestamp: datetime = None) -> None:
        """更新账户余额"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            old_balance = self.current_balance
            self.current_balance = new_balance
            
            # 记录历史
            self.balance_history.append((timestamp, new_balance))
            
            # 计算日收益率
            if len(self.balance_history) > 1:
                prev_balance = self.balance_history[-2][1]
                daily_return = (new_balance - prev_balance) / prev_balance
                self.daily_returns.append(daily_return)
            
            # 更新峰值
            if new_balance > self.peak_balance:
                self.peak_balance = new_balance
                # 如果有活跃的回撤事件，结束它
                if self.current_drawdown_event and self.current_drawdown_event.is_active:
                    self._end_drawdown_event(timestamp)
            
            # 检查回撤
            self._check_drawdown(timestamp)
            
            # 保持历史数据长度
            if len(self.balance_history) > 10000:
                self.balance_history = self.balance_history[-10000:]
            if len(self.daily_returns) > 1000:
                self.daily_returns = self.daily_returns[-1000:]
            
            logger.debug(f"💰 余额更新: {old_balance:.2f} → {new_balance:.2f}")
    
    def _check_drawdown(self, timestamp: datetime) -> None:
        """检查回撤状态"""
        current_drawdown = self.calculate_current_drawdown()
        
        # 检查是否需要开始新的回撤事件
        if current_drawdown > 0.001 and not self.current_drawdown_event:
            self._start_drawdown_event(timestamp)
        
        # 更新保护等级
        old_level = self.protection_level
        self.protection_level = self._determine_protection_level(current_drawdown)
        
        if self.protection_level != old_level:
            logger.warning(f"⚠️ 保护等级变更: {old_level.value} → {self.protection_level.value}")
            self._trigger_protection_actions()
    
    def _start_drawdown_event(self, timestamp: datetime) -> None:
        """开始回撤事件"""
        self.current_drawdown_event = DrawdownEvent(
            start_time=timestamp,
            peak_value=self.peak_balance,
            trough_value=self.current_balance,
            max_drawdown=self.calculate_current_drawdown()
        )
        logger.warning(f"📉 回撤事件开始: {self.current_drawdown_event.max_drawdown:.2%}")
    
    def _end_drawdown_event(self, timestamp: datetime) -> None:
        """结束回撤事件"""
        if self.current_drawdown_event:
            self.current_drawdown_event.end_time = timestamp
            self.current_drawdown_event.duration = timestamp - self.current_drawdown_event.start_time
            self.current_drawdown_event.recovery_time = self.current_drawdown_event.duration
            self.current_drawdown_event.is_active = False
            
            # 添加到历史记录
            self.drawdown_events.append(self.current_drawdown_event)
            
            logger.info(f"📈 回撤事件结束: 最大回撤 {self.current_drawdown_event.max_drawdown:.2%}, "
                       f"持续时间 {self.current_drawdown_event.duration}")
            
            self.current_drawdown_event = None
    
    def _determine_protection_level(self, drawdown: float) -> DrawdownLevel:
        """确定保护等级"""
        if drawdown >= self.settings.critical_threshold:
            return DrawdownLevel.CRITICAL
        elif drawdown >= self.settings.danger_threshold:
            return DrawdownLevel.DANGER
        elif drawdown >= self.settings.warning_threshold:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL
    
    def _trigger_protection_actions(self) -> None:
        """触发保护动作"""
        if not self.settings.enable_auto_protection:
            return
        
        self.protection_actions.clear()
        
        if self.protection_level == DrawdownLevel.WARNING:
            self.protection_actions.append(ProtectionAction.REDUCE_SIZE)
        elif self.protection_level == DrawdownLevel.DANGER:
            self.protection_actions.extend([
                ProtectionAction.REDUCE_SIZE,
                ProtectionAction.STOP_NEW_TRADES
            ])
        elif self.protection_level == DrawdownLevel.CRITICAL:
            self.protection_actions.extend([
                ProtectionAction.CLOSE_POSITIONS,
                ProtectionAction.EMERGENCY_STOP
            ])
        
        logger.warning(f"🛡️ 触发保护动作: {[action.value for action in self.protection_actions]}")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                with self.lock:
                    # 检查时间相关的回撤限制
                    self._check_time_based_limits()
                    
                    # 检查恢复超时
                    self._check_recovery_timeout()
                    
                    # 更新当前回撤事件
                    if self.current_drawdown_event:
                        current_dd = self.calculate_current_drawdown()
                        if current_dd > self.current_drawdown_event.max_drawdown:
                            self.current_drawdown_event.max_drawdown = current_dd
                            self.current_drawdown_event.trough_value = self.current_balance
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"❌ 监控循环异常: {e}")
                time.sleep(60)
    
    def _check_time_based_limits(self) -> None:
        """检查基于时间的限制"""
        now = datetime.now()
        
        # 检查日回撤
        daily_dd = self._calculate_period_drawdown(timedelta(days=1))
        if daily_dd > self.settings.max_daily_drawdown:
            logger.critical(f"🚨 日回撤超限: {daily_dd:.2%}")
            if ProtectionAction.EMERGENCY_STOP not in self.protection_actions:
                self.protection_actions.append(ProtectionAction.EMERGENCY_STOP)
        
        # 检查周回撤
        weekly_dd = self._calculate_period_drawdown(timedelta(weeks=1))
        if weekly_dd > self.settings.max_weekly_drawdown:
            logger.critical(f"🚨 周回撤超限: {weekly_dd:.2%}")
            if ProtectionAction.STOP_NEW_TRADES not in self.protection_actions:
                self.protection_actions.append(ProtectionAction.STOP_NEW_TRADES)
        
        # 检查月回撤
        monthly_dd = self._calculate_period_drawdown(timedelta(days=30))
        if monthly_dd > self.settings.max_monthly_drawdown:
            logger.critical(f"🚨 月回撤超限: {monthly_dd:.2%}")
            if ProtectionAction.REDUCE_SIZE not in self.protection_actions:
                self.protection_actions.append(ProtectionAction.REDUCE_SIZE)
    
    def _check_recovery_timeout(self) -> None:
        """检查恢复超时"""
        if self.current_drawdown_event:
            duration = datetime.now() - self.current_drawdown_event.start_time
            if duration.days > self.settings.recovery_timeout:
                logger.critical(f"🚨 回撤恢复超时: {duration.days}天")
                if ProtectionAction.CLOSE_POSITIONS not in self.protection_actions:
                    self.protection_actions.append(ProtectionAction.CLOSE_POSITIONS)
    
    def _calculate_period_drawdown(self, period: timedelta) -> float:
        """计算指定期间的回撤"""
        if not self.balance_history:
            return 0.0
        
        cutoff_time = datetime.now() - period
        period_history = [(t, b) for t, b in self.balance_history if t >= cutoff_time]
        
        if not period_history:
            return 0.0
        
        period_peak = max(balance for _, balance in period_history)
        current_balance = period_history[-1][1]
        
        if period_peak <= 0:
            return 0.0
        
        return (period_peak - current_balance) / period_peak
    
    def calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if self.peak_balance <= 0:
            return 0.0
        return max(0.0, (self.peak_balance - self.current_balance) / self.peak_balance)
    
    def calculate_max_drawdown(self) -> float:
        """计算历史最大回撤"""
        if not self.balance_history:
            return 0.0
        
        peak = self.balance_history[0][1]
        max_dd = 0.0
        
        for _, balance in self.balance_history:
            if balance > peak:
                peak = balance
            if peak > 0:
                drawdown = (peak - balance) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_drawdown_metrics(self) -> DrawdownMetrics:
        """计算回撤指标"""
        current_dd = self.calculate_current_drawdown()
        max_dd = self.calculate_max_drawdown()
        
        # 计算平均回撤
        completed_events = [e for e in self.drawdown_events if not e.is_active]
        avg_dd = np.mean([e.max_drawdown for e in completed_events]) if completed_events else 0.0
        
        # 计算回撤频率（每年）
        if self.balance_history:
            total_days = (self.balance_history[-1][0] - self.balance_history[0][0]).days
            dd_frequency = len(completed_events) * 365 / max(total_days, 1)
        else:
            dd_frequency = 0.0
        
        # 计算平均恢复时间
        recovery_times = [e.recovery_time.days for e in completed_events 
                         if e.recovery_time is not None]
        avg_recovery = np.mean(recovery_times) if recovery_times else 0.0
        max_recovery = max(recovery_times) if recovery_times else 0.0
        
        # 计算水下时间
        underwater_days = sum((e.end_time - e.start_time).days for e in completed_events 
                             if e.end_time is not None)
        if self.current_drawdown_event:
            underwater_days += (datetime.now() - self.current_drawdown_event.start_time).days
        
        total_days = (datetime.now() - self.balance_history[0][0]).days if self.balance_history else 1
        underwater_time = underwater_days / max(total_days, 1)
        
        # 计算Calmar比率
        if self.daily_returns and max_dd > 0:
            annual_return = np.mean(self.daily_returns) * 252
            calmar_ratio = annual_return / max_dd
        else:
            calmar_ratio = 0.0
        
        # 计算Sterling比率
        if max_dd > 0:
            sterling_ratio = calmar_ratio  # 简化版本
        else:
            sterling_ratio = 0.0
        
        # 计算Pain Index
        pain_index = self._calculate_pain_index()
        
        return DrawdownMetrics(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_frequency=dd_frequency,
            avg_recovery_time=avg_recovery,
            max_recovery_time=max_recovery,
            underwater_time=underwater_time,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            pain_index=pain_index
        )
    
    def _calculate_pain_index(self) -> float:
        """计算痛苦指数"""
        if not self.balance_history:
            return 0.0
        
        total_pain = 0.0
        peak = self.balance_history[0][1]
        
        for _, balance in self.balance_history:
            if balance > peak:
                peak = balance
            if peak > 0:
                drawdown = (peak - balance) / peak
                total_pain += drawdown
        
        return total_pain / len(self.balance_history) if self.balance_history else 0.0
    
    def get_protection_recommendations(self) -> List[str]:
        """获取保护建议"""
        recommendations = []
        
        current_dd = self.calculate_current_drawdown()
        
        if current_dd > self.settings.warning_threshold:
            recommendations.append(f"当前回撤 {current_dd:.2%} 超过警告线，建议减少仓位")
        
        if self.current_drawdown_event:
            duration = datetime.now() - self.current_drawdown_event.start_time
            if duration.days > 7:
                recommendations.append(f"回撤持续 {duration.days} 天，建议检查策略")
        
        if len(self.drawdown_events) > 0:
            recent_events = [e for e in self.drawdown_events 
                           if e.start_time > datetime.now() - timedelta(days=30)]
            if len(recent_events) > 3:
                recommendations.append("近期回撤频繁，建议降低风险敞口")
        
        return recommendations
    
    def get_drawdown_report(self) -> Dict[str, Any]:
        """获取回撤报告"""
        metrics = self.calculate_drawdown_metrics()
        recommendations = self.get_protection_recommendations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "protection_level": self.protection_level.value,
            "protection_actions": [action.value for action in self.protection_actions],
            "metrics": {
                "current_drawdown": metrics.current_drawdown,
                "max_drawdown": metrics.max_drawdown,
                "avg_drawdown": metrics.avg_drawdown,
                "drawdown_frequency": metrics.drawdown_frequency,
                "avg_recovery_time": metrics.avg_recovery_time,
                "underwater_time": metrics.underwater_time,
                "calmar_ratio": metrics.calmar_ratio,
                "pain_index": metrics.pain_index
            },
            "current_event": {
                "active": self.current_drawdown_event is not None,
                "duration_days": (datetime.now() - self.current_drawdown_event.start_time).days 
                                if self.current_drawdown_event else 0,
                "max_drawdown": self.current_drawdown_event.max_drawdown 
                               if self.current_drawdown_event else 0
            },
            "settings": {
                "warning_threshold": self.settings.warning_threshold,
                "danger_threshold": self.settings.danger_threshold,
                "critical_threshold": self.settings.critical_threshold,
                "auto_protection": self.settings.enable_auto_protection
            },
            "recommendations": recommendations,
            "total_events": len(self.drawdown_events)
        }
    
    def reset_peak(self) -> None:
        """重置峰值（用于新的交易周期）"""
        with self.lock:
            self.peak_balance = self.current_balance
            if self.current_drawdown_event:
                self._end_drawdown_event(datetime.now())
            logger.info(f"🔄 峰值重置为: {self.peak_balance:.2f}")


# 全局回撤监控器实例
drawdown_monitor = None

def get_drawdown_monitor(initial_balance: float = 100000.0) -> DrawdownMonitor:
    """获取回撤监控器实例"""
    global drawdown_monitor
    if drawdown_monitor is None:
        drawdown_monitor = DrawdownMonitor(initial_balance)
    return drawdown_monitor


if __name__ == "__main__":
    # 测试回撤监控器
    monitor = DrawdownMonitor(100000.0)
    monitor.start_monitoring()
    
    # 模拟余额变化
    balances = [100000, 98000, 95000, 92000, 96000, 99000, 101000]
    
    for i, balance in enumerate(balances):
        time.sleep(1)
        monitor.update_balance(balance)
        print(f"余额: {balance}, 当前回撤: {monitor.calculate_current_drawdown():.2%}")
    
    # 生成报告
    report = monitor.get_drawdown_report()
    print(json.dumps(report, indent=2, default=str))
    
    monitor.stop_monitoring()

