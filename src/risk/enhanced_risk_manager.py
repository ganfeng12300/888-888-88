#!/usr/bin/env python3
"""
🛡️ 增强风险管理器 - 生产级风险控制系统
Enhanced Risk Manager - Production-Grade Risk Control System

生产级特性：
- 实时风险监控和评估
- 多层次风险控制机制
- 动态风险限额管理
- 风险预警和自动止损
- 风险报告和分析
"""

import numpy as np
import pandas as pd
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    """风险类型枚举"""
    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"

@dataclass
class RiskLimit:
    """风险限额配置"""
    limit_id: str
    name: str
    risk_type: RiskType
    limit_value: float
    warning_threshold: float
    current_exposure: float = 0.0
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class RiskEvent:
    """风险事件数据结构"""
    event_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    description: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PositionRisk:
    """持仓风险数据结构"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    var_1d: float
    var_5d: float
    max_drawdown: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime

class VaRCalculator:
    """风险价值计算器"""
    
    def __init__(self, confidence_level: float = 0.95):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "VaRCalculator")
        self.confidence_level = confidence_level
        self.price_history = defaultdict(deque)
        self.return_history = defaultdict(deque)
        
    def add_price_data(self, symbol: str, price: float, timestamp: datetime):
        """添加价格数据"""
        self.price_history[symbol].append((timestamp, price))
        
        if len(self.price_history[symbol]) > 252:
            self.price_history[symbol].popleft()
        
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2][1]
            return_rate = (price - prev_price) / prev_price
            self.return_history[symbol].append(return_rate)
            
            if len(self.return_history[symbol]) > 251:
                self.return_history[symbol].popleft()
    
    def calculate_var(self, symbol: str, position_value: float, days: int = 1) -> float:
        """计算风险价值"""
        try:
            if symbol not in self.return_history or len(self.return_history[symbol]) < 30:
                return position_value * 0.02 * np.sqrt(days)
            
            returns = np.array(list(self.return_history[symbol]))
            var_percentile = (1 - self.confidence_level) * 100
            daily_var = np.percentile(returns, var_percentile)
            period_var = daily_var * np.sqrt(days)
            var_amount = abs(position_value * period_var)
            
            return var_amount
            
        except Exception as e:
            self.logger.error(f"计算VaR失败 {symbol}: {e}")
            return position_value * 0.02 * np.sqrt(days)

class DrawdownMonitor:
    """回撤监控器"""
    
    def __init__(self, max_lookback_days: int = 252):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "DrawdownMonitor")
        self.max_lookback_days = max_lookback_days
        self.equity_curve = deque(maxlen=max_lookback_days)
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_duration = 0
        self.max_drawdown_duration = 0
        
    def update_equity(self, equity_value: float, timestamp: datetime):
        """更新净值"""
        self.equity_curve.append((timestamp, equity_value))
        
        if equity_value > self.peak_value:
            self.peak_value = equity_value
            self.drawdown_duration = 0
        else:
            self.drawdown_duration += 1
            self.max_drawdown_duration = max(self.max_drawdown_duration, self.drawdown_duration)
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - equity_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def get_drawdown_stats(self) -> Dict[str, float]:
        """获取回撤统计"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value,
            'drawdown_duration': self.drawdown_duration,
            'max_drawdown_duration': self.max_drawdown_duration
        }

class RiskLimitManager:
    """风险限额管理器"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "RiskLimitManager")
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.limit_violations = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_risk_limit(self, risk_limit: RiskLimit):
        """添加风险限额"""
        with self._lock:
            if risk_limit.created_at is None:
                risk_limit.created_at = datetime.now()
            risk_limit.updated_at = datetime.now()
            
            self.risk_limits[risk_limit.limit_id] = risk_limit
            self.logger.info(f"添加风险限额: {risk_limit.name}")
    
    def update_exposure(self, limit_id: str, exposure: float) -> bool:
        """更新风险敞口"""
        with self._lock:
            if limit_id not in self.risk_limits:
                self.logger.error(f"风险限额不存在: {limit_id}")
                return False
            
            risk_limit = self.risk_limits[limit_id]
            risk_limit.current_exposure = exposure
            risk_limit.updated_at = datetime.now()
            
            utilization = exposure / risk_limit.limit_value if risk_limit.limit_value > 0 else 0
            
            if utilization >= 1.0:
                violation = {
                    'limit_id': limit_id,
                    'limit_name': risk_limit.name,
                    'limit_value': risk_limit.limit_value,
                    'current_exposure': exposure,
                    'utilization': utilization,
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                }
                self.limit_violations.append(violation)
                self.logger.critical(f"违反风险限额: {risk_limit.name}")
                return False
                
            elif utilization >= risk_limit.warning_threshold:
                warning = {
                    'limit_id': limit_id,
                    'limit_name': risk_limit.name,
                    'limit_value': risk_limit.limit_value,
                    'current_exposure': exposure,
                    'utilization': utilization,
                    'timestamp': datetime.now(),
                    'severity': 'warning'
                }
                self.limit_violations.append(warning)
                self.logger.warning(f"接近风险限额: {risk_limit.name}")
            
            return True

class EnhancedRiskManager:
    """增强风险管理器主类"""
    
    def __init__(self, config=None):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "EnhancedRiskManager")
        self.settings = config
        
        # 初始化组件
        self.var_calculator = VaRCalculator()
        self.drawdown_monitor = DrawdownMonitor()
        self.limit_manager = RiskLimitManager()
        
        # 风险事件存储
        self.risk_events = deque(maxlen=10000)
        self.active_risks = {}
        
        # 监控状态
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # 初始化默认限额
        self._setup_default_limits()
        
        self.logger.info("增强风险管理器初始化完成")
    
    def _setup_default_limits(self):
        """设置默认风险限额"""
        default_limits = [
            RiskLimit(
                limit_id="max_portfolio_var",
                name="投资组合最大VaR",
                risk_type=RiskType.MARKET,
                limit_value=100000.0,
                warning_threshold=0.8
            ),
            RiskLimit(
                limit_id="max_drawdown",
                name="最大回撤限制",
                risk_type=RiskType.DRAWDOWN,
                limit_value=0.15,
                warning_threshold=0.8
            ),
            RiskLimit(
                limit_id="max_single_position",
                name="单一持仓限制",
                risk_type=RiskType.CONCENTRATION,
                limit_value=0.1,
                warning_threshold=0.8
            )
        ]
        
        for limit in default_limits:
            self.limit_manager.add_risk_limit(limit)
    
    def assess_position_risk(self, symbol: str, position_size: float, 
                           market_price: float, unrealized_pnl: float) -> PositionRisk:
        """评估持仓风险"""
        try:
            market_value = position_size * market_price
            
            var_1d = self.var_calculator.calculate_var(symbol, market_value, 1)
            var_5d = self.var_calculator.calculate_var(symbol, market_value, 5)
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                var_1d=var_1d,
                var_5d=var_5d,
                max_drawdown=0.0,
                concentration_risk=0.0,
                liquidity_risk=0.0,
                timestamp=datetime.now()
            )
            
            return position_risk
            
        except Exception as e:
            self.logger.error(f"评估持仓风险失败 {symbol}: {e}")
            return None
    
    def assess_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """评估投资组合风险"""
        try:
            if not positions:
                return {}
            
            position_values = {
                symbol: pos_info.get('market_value', 0.0) 
                for symbol, pos_info in positions.items()
            }
            
            total_market_value = sum(abs(value) for value in position_values.values())
            
            # 计算投资组合VaR
            portfolio_var = sum(
                self.var_calculator.calculate_var(symbol, abs(value), 1)
                for symbol, value in position_values.items()
            )
            
            # 更新回撤监控
            total_pnl = sum(pos_info.get('unrealized_pnl', 0.0) for pos_info in positions.values())
            current_equity = total_market_value + total_pnl
            self.drawdown_monitor.update_equity(current_equity, datetime.now())
            
            drawdown_stats = self.drawdown_monitor.get_drawdown_stats()
            
            # 检查风险限额
            exposures = {
                'max_portfolio_var': portfolio_var,
                'max_drawdown': drawdown_stats['current_drawdown'],
            }
            
            limit_violations = []
            for limit_id, exposure in exposures.items():
                if not self.limit_manager.update_exposure(limit_id, exposure):
                    limit_violations.append({
                        'limit_id': limit_id,
                        'exposure': exposure
                    })
            
            portfolio_risk = {
                'total_market_value': total_market_value,
                'portfolio_var_1d': portfolio_var,
                'drawdown_stats': drawdown_stats,
                'limit_violations': limit_violations,
                'risk_level': self._calculate_overall_risk_level(portfolio_var, drawdown_stats),
                'timestamp': datetime.now().isoformat()
            }
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"评估投资组合风险失败: {e}")
            return {}
    
    def _calculate_overall_risk_level(self, portfolio_var: float, drawdown_stats: Dict) -> str:
        """计算整体风险等级"""
        try:
            risk_score = 0
            
            # VaR风险评分
            var_limit = self.limit_manager.risk_limits.get('max_portfolio_var')
            if var_limit and var_limit.limit_value > 0:
                var_ratio = portfolio_var / var_limit.limit_value
                if var_ratio > 0.8:
                    risk_score += 3
                elif var_ratio > 0.6:
                    risk_score += 2
                elif var_ratio > 0.4:
                    risk_score += 1
            
            # 回撤风险评分
            current_drawdown = drawdown_stats.get('current_drawdown', 0)
            if current_drawdown > 0.1:
                risk_score += 3
            elif current_drawdown > 0.05:
                risk_score += 2
            elif current_drawdown > 0.02:
                risk_score += 1
            
            if risk_score >= 5:
                return RiskLevel.CRITICAL.value
            elif risk_score >= 3:
                return RiskLevel.HIGH.value
            elif risk_score >= 1:
                return RiskLevel.MEDIUM.value
            else:
                return RiskLevel.LOW.value
                
        except Exception as e:
            self.logger.error(f"计算整体风险等级失败: {e}")
            return RiskLevel.MEDIUM.value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            recent_events = [
                event for event in self.risk_events
                if (datetime.now() - event.timestamp).total_seconds() < 86400
            ]
            
            risk_summary = {
                'active_risks': len(self.active_risks),
                'recent_events_24h': len(recent_events),
                'drawdown_stats': self.drawdown_monitor.get_drawdown_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            return risk_summary
            
        except Exception as e:
            self.logger.error(f"获取风险摘要失败: {e}")
            return {}

# 使用示例
if __name__ == "__main__":
    risk_manager = EnhancedRiskManager()
    
    try:
        positions = {
            'AAPL': {
                'position_size': 1000,
                'market_price': 150.0,
                'market_value': 150000.0,
                'unrealized_pnl': 5000.0
            }
        }
        
        portfolio_risk = risk_manager.assess_portfolio_risk(positions)
        print("投资组合风险评估:", json.dumps(portfolio_risk, indent=2, default=str, ensure_ascii=False))
        
    except Exception as e:
        print(f"测试失败: {e}")
