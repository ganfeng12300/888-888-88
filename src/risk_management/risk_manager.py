#!/usr/bin/env python3
"""
🛡️ 888-888-88 风险管理系统
生产级风险控制和资金管理模块
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
import json
import threading
from decimal import Decimal, ROUND_DOWN


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """风险处理动作"""
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # 95% VaR
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_size: float = 0.0
    position_concentration: float = 0.0
    leverage_ratio: float = 0.0
    correlation_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """风险限制"""
    max_position_size: float = 0.1  # 单个仓位最大占比
    max_total_exposure: float = 0.8  # 总敞口限制
    max_daily_loss: float = 0.05  # 日最大亏损
    max_drawdown: float = 0.15  # 最大回撤
    max_leverage: float = 3.0  # 最大杠杆
    max_correlation: float = 0.7  # 最大相关性
    min_liquidity: float = 0.1  # 最小流动性保留
    stop_loss_pct: float = 0.02  # 止损百分比
    take_profit_pct: float = 0.06  # 止盈百分比


@dataclass
class TradeRisk:
    """交易风险评估"""
    symbol: str
    side: str
    size: float
    price: float
    risk_score: float
    risk_level: RiskLevel
    risk_factors: List[str]
    recommended_action: RiskAction
    max_allowed_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class RiskManager:
    """生产级风险管理器"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_limits = RiskLimits()
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.risk_metrics = RiskMetrics()
        self.emergency_stop = False
        self.risk_lock = threading.Lock()
        
        # 风险监控参数
        self.price_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_cache: Dict[str, float] = {}
        
        logger.info("🛡️ 风险管理系统初始化完成")
    
    def update_balance(self, new_balance: float) -> None:
        """更新账户余额"""
        with self.risk_lock:
            self.current_balance = new_balance
            logger.info(f"💰 账户余额更新: {new_balance:.2f}")
    
    def add_position(self, symbol: str, side: str, size: float, 
                    entry_price: float, timestamp: datetime = None) -> bool:
        """添加仓位"""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.risk_lock:
            position_value = size * entry_price
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'current_price': entry_price,
                    'unrealized_pnl': 0.0,
                    'timestamp': timestamp,
                    'stop_loss': None,
                    'take_profit': None
                }
            else:
                # 更新现有仓位
                existing = self.positions[symbol]
                if existing['side'] == side:
                    # 同向加仓
                    total_size = existing['size'] + size
                    avg_price = (existing['size'] * existing['entry_price'] + 
                               size * entry_price) / total_size
                    existing['size'] = total_size
                    existing['entry_price'] = avg_price
                else:
                    # 反向交易，减仓或反向
                    if existing['size'] > size:
                        existing['size'] -= size
                    elif existing['size'] < size:
                        existing['size'] = size - existing['size']
                        existing['side'] = side
                        existing['entry_price'] = entry_price
                    else:
                        # 完全平仓
                        del self.positions[symbol]
                        return True
            
            logger.info(f"📊 仓位更新: {symbol} {side} {size} @ {entry_price}")
            return True
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """更新仓位当前价格"""
        with self.risk_lock:
            if symbol in self.positions:
                position = self.positions[symbol]
                position['current_price'] = current_price
                
                # 计算未实现盈亏
                if position['side'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']
                
                position['unrealized_pnl'] = pnl
                
                # 更新价格历史
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(current_price)
                
                # 保持最近1000个价格点
                if len(self.price_history[symbol]) > 1000:
                    self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def calculate_position_risk(self, symbol: str, side: str, size: float, 
                              price: float) -> TradeRisk:
        """计算交易风险"""
        risk_factors = []
        risk_score = 0.0
        
        # 1. 仓位大小风险
        position_value = size * price
        position_ratio = position_value / self.current_balance
        
        if position_ratio > self.risk_limits.max_position_size:
            risk_factors.append(f"仓位过大: {position_ratio:.2%}")
            risk_score += 30
        
        # 2. 总敞口风险
        total_exposure = self._calculate_total_exposure()
        new_exposure = total_exposure + position_value
        exposure_ratio = new_exposure / self.current_balance
        
        if exposure_ratio > self.risk_limits.max_total_exposure:
            risk_factors.append(f"总敞口过大: {exposure_ratio:.2%}")
            risk_score += 25
        
        # 3. 相关性风险
        correlation_risk = self._calculate_correlation_risk(symbol)
        if correlation_risk > self.risk_limits.max_correlation:
            risk_factors.append(f"相关性过高: {correlation_risk:.2f}")
            risk_score += 20
        
        # 4. 波动率风险
        volatility = self._calculate_volatility(symbol)
        if volatility > 0.05:  # 5%日波动率
            risk_factors.append(f"高波动率: {volatility:.2%}")
            risk_score += 15
        
        # 5. 流动性风险
        liquidity_ratio = self._calculate_liquidity_ratio()
        if liquidity_ratio < self.risk_limits.min_liquidity:
            risk_factors.append(f"流动性不足: {liquidity_ratio:.2%}")
            risk_score += 20
        
        # 6. 回撤风险
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.risk_limits.max_drawdown * 0.8:
            risk_factors.append(f"接近最大回撤: {current_drawdown:.2%}")
            risk_score += 25
        
        # 确定风险等级
        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
            action = RiskAction.EMERGENCY_STOP
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
            action = RiskAction.BLOCK
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM
            action = RiskAction.REDUCE
        else:
            risk_level = RiskLevel.LOW
            action = RiskAction.ALLOW
        
        # 计算建议仓位大小
        max_allowed_size = self._calculate_max_allowed_size(
            symbol, side, price, risk_score
        )
        
        # 计算止损止盈
        stop_loss, take_profit = self._calculate_stop_levels(
            side, price, volatility
        )
        
        return TradeRisk(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommended_action=action,
            max_allowed_size=max_allowed_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def validate_trade(self, symbol: str, side: str, size: float, 
                      price: float) -> Tuple[bool, TradeRisk]:
        """验证交易是否可执行"""
        if self.emergency_stop:
            risk = TradeRisk(
                symbol=symbol, side=side, size=size, price=price,
                risk_score=100, risk_level=RiskLevel.CRITICAL,
                risk_factors=["紧急停止状态"], 
                recommended_action=RiskAction.EMERGENCY_STOP,
                max_allowed_size=0
            )
            return False, risk
        
        trade_risk = self.calculate_position_risk(symbol, side, size, price)
        
        # 根据风险等级决定是否允许交易
        if trade_risk.recommended_action == RiskAction.ALLOW:
            return True, trade_risk
        elif trade_risk.recommended_action == RiskAction.REDUCE:
            # 允许交易但建议减少仓位
            return True, trade_risk
        else:
            # 阻止交易
            return False, trade_risk
    
    def update_risk_metrics(self) -> RiskMetrics:
        """更新风险指标"""
        with self.risk_lock:
            # 计算总敞口
            total_exposure = self._calculate_total_exposure()
            
            # 计算当日盈亏
            daily_pnl = self._calculate_daily_pnl()
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown()
            
            # 计算VaR
            var_95 = self._calculate_var(confidence=0.95)
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # 计算胜率
            win_rate = self._calculate_win_rate()
            
            # 计算平均交易规模
            avg_trade_size = self._calculate_avg_trade_size()
            
            # 计算仓位集中度
            position_concentration = self._calculate_position_concentration()
            
            # 计算杠杆比率
            leverage_ratio = total_exposure / self.current_balance
            
            # 计算相关性风险
            correlation_risk = self._calculate_avg_correlation_risk()
            
            self.risk_metrics = RiskMetrics(
                total_exposure=total_exposure,
                daily_pnl=daily_pnl,
                max_drawdown=max_drawdown,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                avg_trade_size=avg_trade_size,
                position_concentration=position_concentration,
                leverage_ratio=leverage_ratio,
                correlation_risk=correlation_risk,
                timestamp=datetime.now()
            )
            
            return self.risk_metrics
    
    def check_emergency_conditions(self) -> bool:
        """检查紧急停止条件"""
        # 检查最大回撤
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.risk_limits.max_drawdown:
            logger.critical(f"🚨 触发紧急停止: 回撤超限 {current_drawdown:.2%}")
            self.emergency_stop = True
            return True
        
        # 检查日亏损限制
        daily_pnl = self._calculate_daily_pnl()
        daily_loss_ratio = abs(daily_pnl) / self.current_balance
        if daily_pnl < 0 and daily_loss_ratio > self.risk_limits.max_daily_loss:
            logger.critical(f"🚨 触发紧急停止: 日亏损超限 {daily_loss_ratio:.2%}")
            self.emergency_stop = True
            return True
        
        # 检查杠杆比率
        leverage = self._calculate_total_exposure() / self.current_balance
        if leverage > self.risk_limits.max_leverage:
            logger.critical(f"🚨 触发紧急停止: 杠杆超限 {leverage:.2f}")
            self.emergency_stop = True
            return True
        
        return False
    
    def reset_emergency_stop(self) -> None:
        """重置紧急停止状态"""
        self.emergency_stop = False
        logger.info("✅ 紧急停止状态已重置")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        self.update_risk_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "emergency_stop": self.emergency_stop,
            "current_balance": self.current_balance,
            "risk_metrics": {
                "total_exposure": self.risk_metrics.total_exposure,
                "daily_pnl": self.risk_metrics.daily_pnl,
                "max_drawdown": self.risk_metrics.max_drawdown,
                "var_95": self.risk_metrics.var_95,
                "sharpe_ratio": self.risk_metrics.sharpe_ratio,
                "win_rate": self.risk_metrics.win_rate,
                "leverage_ratio": self.risk_metrics.leverage_ratio,
                "correlation_risk": self.risk_metrics.correlation_risk
            },
            "risk_limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "max_total_exposure": self.risk_limits.max_total_exposure,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_drawdown": self.risk_limits.max_drawdown,
                "max_leverage": self.risk_limits.max_leverage
            },
            "positions": dict(self.positions),
            "position_count": len(self.positions)
        }
    
    # 私有方法实现
    def _calculate_total_exposure(self) -> float:
        """计算总敞口"""
        total = 0.0
        for position in self.positions.values():
            total += abs(position['size'] * position['current_price'])
        return total
    
    def _calculate_daily_pnl(self) -> float:
        """计算当日盈亏"""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        # 未实现盈亏
        for position in self.positions.values():
            daily_pnl += position['unrealized_pnl']
        
        # 已实现盈亏（当日交易）
        for trade in self.trade_history:
            if trade.get('timestamp', datetime.now()).date() == today:
                daily_pnl += trade.get('realized_pnl', 0.0)
        
        return daily_pnl
    
    def _calculate_max_drawdown(self) -> float:
        """计算历史最大回撤"""
        if not self.trade_history:
            return 0.0
        
        balance_history = [self.initial_balance]
        running_balance = self.initial_balance
        
        for trade in self.trade_history:
            running_balance += trade.get('realized_pnl', 0.0)
            balance_history.append(running_balance)
        
        peak = balance_history[0]
        max_dd = 0.0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        current_value = self.current_balance
        for position in self.positions.values():
            current_value += position['unrealized_pnl']
        
        # 找到历史最高点
        peak = self.initial_balance
        for trade in self.trade_history:
            peak = max(peak, trade.get('balance_after', peak))
        
        if peak <= 0:
            return 0.0
        
        return max(0.0, (peak - current_value) / peak)
    
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if not self.trade_history:
            return 0.0
        
        returns = [trade.get('return_pct', 0.0) for trade in self.trade_history[-100:]]
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        return abs(var * self.current_balance)
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if not self.trade_history:
            return 0.0
        
        returns = [trade.get('return_pct', 0.0) for trade in self.trade_history[-100:]]
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # 假设无风险利率为0
        return mean_return / std_return
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trade_history:
            return 0.0
        
        wins = sum(1 for trade in self.trade_history 
                  if trade.get('realized_pnl', 0.0) > 0)
        total = len(self.trade_history)
        
        return wins / total if total > 0 else 0.0
    
    def _calculate_avg_trade_size(self) -> float:
        """计算平均交易规模"""
        if not self.trade_history:
            return 0.0
        
        sizes = [trade.get('size', 0.0) * trade.get('price', 0.0) 
                for trade in self.trade_history]
        return np.mean(sizes) if sizes else 0.0
    
    def _calculate_position_concentration(self) -> float:
        """计算仓位集中度"""
        if not self.positions:
            return 0.0
        
        total_exposure = self._calculate_total_exposure()
        if total_exposure == 0:
            return 0.0
        
        # 计算最大单个仓位占比
        max_position = max(
            abs(pos['size'] * pos['current_price']) 
            for pos in self.positions.values()
        )
        
        return max_position / total_exposure
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """计算相关性风险"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return 0.0
        
        target_returns = np.diff(np.log(self.price_history[symbol][-20:]))
        max_correlation = 0.0
        
        for other_symbol, prices in self.price_history.items():
            if other_symbol != symbol and len(prices) >= 20:
                other_returns = np.diff(np.log(prices[-20:]))
                if len(other_returns) == len(target_returns):
                    correlation = np.corrcoef(target_returns, other_returns)[0, 1]
                    max_correlation = max(max_correlation, abs(correlation))
        
        return max_correlation
    
    def _calculate_avg_correlation_risk(self) -> float:
        """计算平均相关性风险"""
        correlations = []
        symbols = list(self.price_history.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if (len(self.price_history[symbol1]) >= 20 and 
                    len(self.price_history[symbol2]) >= 20):
                    
                    returns1 = np.diff(np.log(self.price_history[symbol1][-20:]))
                    returns2 = np.diff(np.log(self.price_history[symbol2][-20:]))
                    
                    if len(returns1) == len(returns2):
                        corr = np.corrcoef(returns1, returns2)[0, 1]
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算波动率"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return 0.02  # 默认2%
        
        prices = self.price_history[symbol][-20:]
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(24)  # 日化波动率
        
        self.volatility_cache[symbol] = volatility
        return volatility
    
    def _calculate_liquidity_ratio(self) -> float:
        """计算流动性比率"""
        total_exposure = self._calculate_total_exposure()
        if total_exposure == 0:
            return 1.0
        
        available_cash = self.current_balance - total_exposure
        return available_cash / self.current_balance
    
    def _calculate_max_allowed_size(self, symbol: str, side: str, 
                                  price: float, risk_score: float) -> float:
        """计算最大允许仓位"""
        # 基础最大仓位
        base_max = self.current_balance * self.risk_limits.max_position_size
        
        # 根据风险评分调整
        risk_adjustment = max(0.1, 1.0 - risk_score / 100.0)
        adjusted_max = base_max * risk_adjustment
        
        # 转换为数量
        max_size = adjusted_max / price
        
        return max_size
    
    def _calculate_stop_levels(self, side: str, price: float, 
                             volatility: float) -> Tuple[Optional[float], Optional[float]]:
        """计算止损止盈位"""
        # 动态止损，基于波动率
        stop_pct = max(self.risk_limits.stop_loss_pct, volatility * 2)
        profit_pct = max(self.risk_limits.take_profit_pct, volatility * 3)
        
        if side == 'long':
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + profit_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - profit_pct)
        
        return stop_loss, take_profit


# 全局风险管理器实例
risk_manager = None

def get_risk_manager(initial_balance: float = 100000.0) -> RiskManager:
    """获取风险管理器实例"""
    global risk_manager
    if risk_manager is None:
        risk_manager = RiskManager(initial_balance)
    return risk_manager


if __name__ == "__main__":
    # 测试风险管理器
    rm = RiskManager(100000.0)
    
    # 模拟交易测试
    can_trade, risk = rm.validate_trade("BTC/USDT", "long", 1.0, 50000.0)
    print(f"交易验证: {can_trade}")
    print(f"风险评估: {risk.risk_level.value}, 评分: {risk.risk_score}")
    
    if can_trade:
        rm.add_position("BTC/USDT", "long", 1.0, 50000.0)
        rm.update_position_price("BTC/USDT", 51000.0)
    
    # 生成风险报告
    report = rm.get_risk_report()
    print(json.dumps(report, indent=2, default=str))

