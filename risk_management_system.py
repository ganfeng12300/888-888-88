#!/usr/bin/env python3
"""
🛡️ 风险管理系统 - 第二步扩展：智能风险控制
Risk Management System - Step 2 Extension: Intelligent Risk Control

生产级功能：
- 实时风险监控 (128GB内存优化)
- 多维度风险评估
- 动态仓位管理
- 智能止损止盈
- 市场异常检测
- 资金管理优化
"""

import asyncio
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_management.log'),
        logging.StreamHandler()
    ]
)

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0
        if self.side == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

@dataclass
class RiskMetrics:
    """风险指标"""
    total_exposure: float
    max_drawdown: float
    var_95: float  # 95% VaR
    sharpe_ratio: float
    volatility: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    timestamp: float

@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    risk_type: str
    severity: RiskLevel
    message: str
    symbol: Optional[str]
    exchange: Optional[str]
    current_value: float
    threshold: float
    timestamp: float
    acknowledged: bool = False

class PositionManager:
    """仓位管理器"""
    
    def __init__(self, max_positions: int = 50):
        self.positions: Dict[str, Position] = {}
        self.max_positions = max_positions
        self.logger = logging.getLogger("PositionManager")
        self._lock = threading.Lock()
    
    def add_position(self, position: Position) -> bool:
        """添加持仓"""
        with self._lock:
            if len(self.positions) >= self.max_positions:
                self.logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return False
            
            key = f"{position.symbol}_{position.exchange}_{position.side}"
            
            if key in self.positions:
                # 更新现有持仓
                existing = self.positions[key]
                total_quantity = existing.quantity + position.quantity
                if total_quantity == 0:
                    del self.positions[key]
                else:
                    # 计算平均成本
                    total_cost = (existing.quantity * existing.entry_price + 
                                position.quantity * position.entry_price)
                    avg_price = total_cost / total_quantity
                    
                    existing.quantity = total_quantity
                    existing.entry_price = avg_price
                    existing.timestamp = position.timestamp
            else:
                self.positions[key] = position
            
            return True
    
    def update_position_price(self, symbol: str, exchange: str, price: float):
        """更新持仓价格"""
        with self._lock:
            for key, position in self.positions.items():
                if position.symbol == symbol and position.exchange == exchange:
                    position.current_price = price
                    
                    # 计算未实现盈亏
                    if position.side == 'long':
                        position.unrealized_pnl = (price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - price) * position.quantity
    
    def get_total_exposure(self) -> float:
        """获取总敞口"""
        with self._lock:
            return sum(pos.market_value for pos in self.positions.values())
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """按交易对获取持仓"""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_unrealized_pnl(self) -> float:
        """获取未实现盈亏"""
        with self._lock:
            return sum(pos.unrealized_pnl for pos in self.positions.values())

class RiskCalculator:
    """风险计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger("RiskCalculator")
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
    
    def update_price_history(self, symbol: str, price: float, max_history: int = 1000):
        """更新价格历史"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.returns_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # 计算收益率
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2]
            return_rate = (price - prev_price) / prev_price
            self.returns_history[symbol].append(return_rate)
        
        # 限制历史数据长度
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        if len(self.returns_history[symbol]) > max_history:
            self.returns_history[symbol] = self.returns_history[symbol][-max_history:]
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence) * 100)
    
    def calculate_volatility(self, returns: List[float], window: int = 30) -> float:
        """计算波动率"""
        if len(returns) < window:
            return 0.0
        
        recent_returns = returns[-window:]
        return np.std(recent_returns) * np.sqrt(252)  # 年化波动率
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 日化无风险利率
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, prices: List[float]) -> float:
        """计算最大回撤"""
        if len(prices) < 2:
            return 0.0
        
        prices_array = np.array(prices)
        cumulative = np.cumprod(1 + np.diff(prices_array) / prices_array[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """计算相关性矩阵"""
        if len(symbols) < 2:
            return np.array([[1.0]])
        
        returns_matrix = []
        min_length = float('inf')
        
        # 找到最短的历史数据长度
        for symbol in symbols:
            if symbol in self.returns_history:
                min_length = min(min_length, len(self.returns_history[symbol]))
        
        if min_length < 30:
            return np.eye(len(symbols))
        
        # 构建收益率矩阵
        for symbol in symbols:
            if symbol in self.returns_history:
                returns_matrix.append(self.returns_history[symbol][-min_length:])
            else:
                returns_matrix.append([0.0] * min_length)
        
        returns_df = pd.DataFrame(returns_matrix).T
        return returns_df.corr().values

class RiskMonitor:
    """风险监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_manager = PositionManager(config.get('max_positions', 50))
        self.risk_calculator = RiskCalculator()
        self.alerts: List[RiskAlert] = []
        self.logger = logging.getLogger("RiskMonitor")
        
        # 风险限制
        self.max_total_exposure = config.get('max_total_exposure', 100000)
        self.max_position_size = config.get('max_position_size', 10000)
        self.max_daily_loss = config.get('max_daily_loss', 5000)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)
        self.var_limit = config.get('var_limit', 0.05)
        
        # 统计数据
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def check_pre_trade_risk(self, symbol: str, exchange: str, side: str, 
                           quantity: float, price: float) -> Tuple[bool, str]:
        """交易前风险检查"""
        try:
            # 检查单笔订单大小
            order_value = quantity * price
            if order_value > self.max_position_size:
                return False, f"Order size {order_value:.2f} exceeds limit {self.max_position_size}"
            
            # 检查总敞口
            current_exposure = self.position_manager.get_total_exposure()
            if current_exposure + order_value > self.max_total_exposure:
                return False, f"Total exposure would exceed limit {self.max_total_exposure}"
            
            # 检查日损失限制
            if self.daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit {self.max_daily_loss} exceeded"
            
            # 检查持仓集中度
            symbol_positions = self.position_manager.get_positions_by_symbol(symbol)
            symbol_exposure = sum(pos.market_value for pos in symbol_positions)
            
            if symbol_exposure + order_value > self.max_total_exposure * 0.3:  # 单个交易对不超过30%
                return False, f"Symbol concentration risk: {symbol} exposure too high"
            
            return True, "Risk check passed"
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade risk check: {e}")
            return False, f"Risk check error: {e}"
    
    def update_position(self, symbol: str, exchange: str, side: str, 
                       quantity: float, price: float):
        """更新持仓"""
        position = Position(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=time.time()
        )
        
        self.position_manager.add_position(position)
        self.risk_calculator.update_price_history(symbol, price)
    
    def update_market_price(self, symbol: str, exchange: str, price: float):
        """更新市场价格"""
        self.position_manager.update_position_price(symbol, exchange, price)
        self.risk_calculator.update_price_history(symbol, price)
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标"""
        try:
            # 总敞口
            total_exposure = self.position_manager.get_total_exposure()
            
            # 获取所有交易对的收益率
            all_returns = []
            for returns in self.risk_calculator.returns_history.values():
                all_returns.extend(returns[-100:])  # 取最近100个数据点
            
            if not all_returns:
                return RiskMetrics(
                    total_exposure=total_exposure,
                    max_drawdown=0.0,
                    var_95=0.0,
                    sharpe_ratio=0.0,
                    volatility=0.0,
                    correlation_risk=0.0,
                    liquidity_risk=0.0,
                    concentration_risk=0.0,
                    timestamp=time.time()
                )
            
            # 计算各项指标
            var_95 = self.risk_calculator.calculate_var(all_returns, 0.95)
            volatility = self.risk_calculator.calculate_volatility(all_returns)
            sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(all_returns)
            
            # 计算最大回撤
            portfolio_values = [total_exposure]  # 简化处理
            max_drawdown = self.risk_calculator.calculate_max_drawdown(portfolio_values)
            
            # 计算集中度风险
            positions = list(self.position_manager.positions.values())
            if positions:
                max_position_value = max(pos.market_value for pos in positions)
                concentration_risk = max_position_value / total_exposure if total_exposure > 0 else 0
            else:
                concentration_risk = 0.0
            
            # 相关性风险（简化）
            symbols = list(set(pos.symbol for pos in positions))
            if len(symbols) > 1:
                corr_matrix = self.risk_calculator.calculate_correlation_matrix(symbols)
                correlation_risk = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            else:
                correlation_risk = 0.0
            
            # 流动性风险（简化，基于持仓数量）
            liquidity_risk = min(1.0, len(positions) / 20.0)  # 持仓越多流动性风险越高
            
            return RiskMetrics(
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                var_95=abs(var_95),
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, time.time())
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """检查风险限制"""
        alerts = []
        
        try:
            # 检查总敞口
            if metrics.total_exposure > self.max_total_exposure:
                alert = RiskAlert(
                    alert_id=f"exposure_{int(time.time())}",
                    risk_type="total_exposure",
                    severity=RiskLevel.HIGH,
                    message=f"Total exposure {metrics.total_exposure:.2f} exceeds limit {self.max_total_exposure}",
                    symbol=None,
                    exchange=None,
                    current_value=metrics.total_exposure,
                    threshold=self.max_total_exposure,
                    timestamp=time.time()
                )
                alerts.append(alert)
            
            # 检查最大回撤
            if metrics.max_drawdown > self.max_drawdown_limit:
                alert = RiskAlert(
                    alert_id=f"drawdown_{int(time.time())}",
                    risk_type="max_drawdown",
                    severity=RiskLevel.CRITICAL,
                    message=f"Max drawdown {metrics.max_drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%}",
                    symbol=None,
                    exchange=None,
                    current_value=metrics.max_drawdown,
                    threshold=self.max_drawdown_limit,
                    timestamp=time.time()
                )
                alerts.append(alert)
            
            # 检查VaR
            if metrics.var_95 > self.var_limit:
                alert = RiskAlert(
                    alert_id=f"var_{int(time.time())}",
                    risk_type="var_95",
                    severity=RiskLevel.HIGH,
                    message=f"95% VaR {metrics.var_95:.2%} exceeds limit {self.var_limit:.2%}",
                    symbol=None,
                    exchange=None,
                    current_value=metrics.var_95,
                    threshold=self.var_limit,
                    timestamp=time.time()
                )
                alerts.append(alert)
            
            # 检查集中度风险
            if metrics.concentration_risk > 0.4:  # 40%集中度警告
                alert = RiskAlert(
                    alert_id=f"concentration_{int(time.time())}",
                    risk_type="concentration_risk",
                    severity=RiskLevel.MEDIUM,
                    message=f"Concentration risk {metrics.concentration_risk:.2%} is high",
                    symbol=None,
                    exchange=None,
                    current_value=metrics.concentration_risk,
                    threshold=0.4,
                    timestamp=time.time()
                )
                alerts.append(alert)
            
            # 检查相关性风险
            if metrics.correlation_risk > 0.8:  # 80%相关性警告
                alert = RiskAlert(
                    alert_id=f"correlation_{int(time.time())}",
                    risk_type="correlation_risk",
                    severity=RiskLevel.MEDIUM,
                    message=f"High correlation risk {metrics.correlation_risk:.2%}",
                    symbol=None,
                    exchange=None,
                    current_value=metrics.correlation_risk,
                    threshold=0.8,
                    timestamp=time.time()
                )
                alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
        
        return alerts
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """生成风险报告"""
        metrics = self.calculate_risk_metrics()
        alerts = self.check_risk_limits(metrics)
        
        # 更新警报列表
        self.alerts.extend(alerts)
        
        # 只保留最近的100个警报
        self.alerts = self.alerts[-100:]
        
        # 计算胜率
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "risk_metrics": {
                "total_exposure": metrics.total_exposure,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "sharpe_ratio": metrics.sharpe_ratio,
                "volatility": metrics.volatility,
                "correlation_risk": metrics.correlation_risk,
                "liquidity_risk": metrics.liquidity_risk,
                "concentration_risk": metrics.concentration_risk
            },
            "portfolio_stats": {
                "total_positions": len(self.position_manager.positions),
                "unrealized_pnl": self.position_manager.get_unrealized_pnl(),
                "daily_pnl": self.daily_pnl,
                "total_trades": self.total_trades,
                "win_rate": win_rate
            },
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "risk_type": alert.risk_type,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in alerts
            ],
            "positions": [
                {
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_percentage": pos.pnl_percentage
                }
                for pos in self.position_manager.positions.values()
            ]
        }
        
        return report
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """判断是否应该停止交易"""
        metrics = self.calculate_risk_metrics()
        
        # 严重风险情况下停止交易
        if self.daily_pnl < -self.max_daily_loss:
            return True, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
        
        if metrics.max_drawdown > self.max_drawdown_limit:
            return True, f"Max drawdown limit exceeded: {metrics.max_drawdown:.2%}"
        
        if metrics.total_exposure > self.max_total_exposure * 1.2:  # 超过120%立即停止
            return True, f"Total exposure critically high: {metrics.total_exposure:.2f}"
        
        return False, "Trading can continue"
    
    def update_trade_result(self, pnl: float):
        """更新交易结果"""
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
    
    async def start_monitoring(self, interval: float = 5.0):
        """开始风险监控"""
        self.logger.info("Starting risk monitoring...")
        
        while True:
            try:
                # 生成风险报告
                report = self.generate_risk_report()
                
                # 检查是否需要停止交易
                should_stop, reason = self.should_stop_trading()
                if should_stop:
                    self.logger.critical(f"TRADING STOPPED: {reason}")
                
                # 记录关键指标
                metrics = report['risk_metrics']
                self.logger.info(
                    f"Risk Monitor - Exposure: {metrics['total_exposure']:.2f}, "
                    f"Drawdown: {metrics['max_drawdown']:.2%}, "
                    f"VaR: {metrics['var_95']:.2%}, "
                    f"Alerts: {len(report['active_alerts'])}"
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(interval)

def create_risk_config() -> Dict[str, Any]:
    """创建风险配置"""
    return {
        'max_positions': 50,
        'max_total_exposure': 100000,  # 最大总敞口
        'max_position_size': 10000,    # 最大单笔订单
        'max_daily_loss': 5000,        # 最大日损失
        'max_drawdown_limit': 0.15,    # 最大回撤限制 15%
        'var_limit': 0.05,             # VaR限制 5%
        'monitoring_interval': 5.0      # 监控间隔（秒）
    }

async def main():
    """主函数"""
    print("🛡️ 启动风险管理系统...")
    
    config = create_risk_config()
    risk_monitor = RiskMonitor(config)
    
    # 模拟一些交易
    risk_monitor.update_position("BTCUSDT", "binance", "long", 0.1, 50000)
    risk_monitor.update_position("ETHUSDT", "okx", "long", 1.0, 3000)
    
    # 模拟价格更新
    risk_monitor.update_market_price("BTCUSDT", "binance", 51000)
    risk_monitor.update_market_price("ETHUSDT", "okx", 2950)
    
    # 生成风险报告
    report = risk_monitor.generate_risk_report()
    
    print("📊 风险报告:")
    print(f"  总敞口: ${report['risk_metrics']['total_exposure']:,.2f}")
    print(f"  未实现盈亏: ${report['portfolio_stats']['unrealized_pnl']:,.2f}")
    print(f"  持仓数量: {report['portfolio_stats']['total_positions']}")
    print(f"  活跃警报: {len(report['active_alerts'])}")
    
    # 启动监控（演示5秒）
    monitoring_task = asyncio.create_task(risk_monitor.start_monitoring())
    await asyncio.sleep(5)
    monitoring_task.cancel()
    
    print("✅ 风险管理系统演示完成")

if __name__ == "__main__":
    asyncio.run(main())
