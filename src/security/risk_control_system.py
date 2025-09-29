"""
🛡️ 交易风险控制系统 - 生产级实盘交易风险管理和控制系统
提供仓位管理、止损止盈、资金安全、异常检测等全方位风险控制功能
确保交易系统在各种市场条件下的资金安全和风险可控
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中等风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 关键风险

class RiskType(Enum):
    """风险类型"""
    POSITION = "position"  # 仓位风险
    DRAWDOWN = "drawdown"  # 回撤风险
    VOLATILITY = "volatility"  # 波动率风险
    LIQUIDITY = "liquidity"  # 流动性风险
    CONCENTRATION = "concentration"  # 集中度风险
    CORRELATION = "correlation"  # 相关性风险

class ActionType(Enum):
    """风控动作类型"""
    ALLOW = "allow"  # 允许
    WARN = "warn"  # 警告
    LIMIT = "limit"  # 限制
    BLOCK = "block"  # 阻止
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止

@dataclass
class RiskLimit:
    """风险限制"""
    name: str  # 限制名称
    risk_type: RiskType  # 风险类型
    max_value: float  # 最大值
    warning_threshold: float  # 警告阈值
    current_value: float = 0.0  # 当前值
    is_active: bool = True  # 是否激活
    description: str = ""  # 描述

@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str  # 事件ID
    risk_type: RiskType  # 风险类型
    risk_level: RiskLevel  # 风险级别
    symbol: str  # 交易对
    message: str  # 消息
    current_value: float  # 当前值
    threshold: float  # 阈值
    action_taken: ActionType  # 采取的动作
    timestamp: float = field(default_factory=time.time)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class PositionInfo:
    """仓位信息"""
    symbol: str  # 交易对
    size: float  # 仓位大小
    entry_price: float  # 入场价格
    current_price: float  # 当前价格
    unrealized_pnl: float  # 未实现盈亏
    realized_pnl: float  # 已实现盈亏
    margin_used: float  # 使用保证金
    leverage: float  # 杠杆倍数
    side: str  # 方向 (long/short)
    timestamp: float = field(default_factory=time.time)  # 时间戳

class PositionRiskManager:
    """仓位风险管理器"""
    
    def __init__(self):
        self.positions: Dict[str, PositionInfo] = {}
        self.max_position_size = 0.1  # 单个仓位最大占比
        self.max_total_exposure = 0.8  # 总敞口最大占比
        self.max_leverage = 10.0  # 最大杠杆
        self.max_correlation_exposure = 0.3  # 最大相关性敞口
        
        logger.info("仓位风险管理器初始化完成")
    
    def update_position(self, position: PositionInfo):
        """更新仓位信息"""
        self.positions[position.symbol] = position
        logger.debug(f"更新仓位信息: {position.symbol}")
    
    def check_position_risk(self, symbol: str, new_size: float, 
                           account_balance: float) -> Tuple[bool, List[RiskEvent]]:
        """检查仓位风险"""
        risk_events = []
        
        try:
            # 检查单个仓位大小
            position_ratio = abs(new_size) / account_balance
            if position_ratio > self.max_position_size:
                risk_events.append(RiskEvent(
                    event_id=f"pos_size_{symbol}_{int(time.time())}",
                    risk_type=RiskType.POSITION,
                    risk_level=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"单个仓位过大: {position_ratio:.2%} > {self.max_position_size:.2%}",
                    current_value=position_ratio,
                    threshold=self.max_position_size,
                    action_taken=ActionType.BLOCK
                ))
            
            # 检查总敞口
            total_exposure = self._calculate_total_exposure(account_balance)
            if total_exposure > self.max_total_exposure:
                risk_events.append(RiskEvent(
                    event_id=f"total_exp_{int(time.time())}",
                    risk_type=RiskType.POSITION,
                    risk_level=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"总敞口过大: {total_exposure:.2%} > {self.max_total_exposure:.2%}",
                    current_value=total_exposure,
                    threshold=self.max_total_exposure,
                    action_taken=ActionType.LIMIT
                ))
            
            # 检查杠杆
            if symbol in self.positions:
                leverage = self.positions[symbol].leverage
                if leverage > self.max_leverage:
                    risk_events.append(RiskEvent(
                        event_id=f"leverage_{symbol}_{int(time.time())}",
                        risk_type=RiskType.POSITION,
                        risk_level=RiskLevel.MEDIUM,
                        symbol=symbol,
                        message=f"杠杆过高: {leverage}x > {self.max_leverage}x",
                        current_value=leverage,
                        threshold=self.max_leverage,
                        action_taken=ActionType.WARN
                    ))
            
            # 如果有阻止级别的风险事件，返回False
            has_blocking_risk = any(event.action_taken == ActionType.BLOCK for event in risk_events)
            
            return not has_blocking_risk, risk_events
        
        except Exception as e:
            logger.error(f"检查仓位风险失败: {e}")
            return False, []
    
    def _calculate_total_exposure(self, account_balance: float) -> float:
        """计算总敞口"""
        total_exposure = 0.0
        for position in self.positions.values():
            exposure = abs(position.size * position.current_price) / account_balance
            total_exposure += exposure
        return total_exposure
    
    def get_position_summary(self) -> Dict[str, Any]:
        """获取仓位摘要"""
        if not self.positions:
            return {}
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_margin_used = sum(pos.margin_used for pos in self.positions.values())
        
        return {
            'position_count': len(self.positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_margin_used': total_margin_used,
            'positions': {symbol: {
                'size': pos.size,
                'unrealized_pnl': pos.unrealized_pnl,
                'leverage': pos.leverage
            } for symbol, pos in self.positions.items()}
        }

class DrawdownManager:
    """回撤管理器"""
    
    def __init__(self):
        self.equity_history: List[Tuple[float, float]] = []  # (timestamp, equity)
        self.max_drawdown_limit = 0.15  # 最大回撤限制 15%
        self.daily_loss_limit = 0.05  # 日损失限制 5%
        self.peak_equity = 0.0  # 峰值净值
        self.daily_start_equity = 0.0  # 日开始净值
        self.last_reset_date = time.time()
        
        logger.info("回撤管理器初始化完成")
    
    def update_equity(self, current_equity: float) -> List[RiskEvent]:
        """更新净值并检查回撤"""
        risk_events = []
        current_time = time.time()
        
        # 添加到历史记录
        self.equity_history.append((current_time, current_equity))
        
        # 保持历史记录在合理范围内
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-5000:]
        
        # 更新峰值净值
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # 检查是否需要重置日开始净值
        if self._is_new_day():
            self.daily_start_equity = current_equity
            self.last_reset_date = current_time
        
        # 检查最大回撤
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown_limit:
                risk_events.append(RiskEvent(
                    event_id=f"max_dd_{int(current_time)}",
                    risk_type=RiskType.DRAWDOWN,
                    risk_level=RiskLevel.CRITICAL,
                    symbol="PORTFOLIO",
                    message=f"最大回撤超限: {current_drawdown:.2%} > {self.max_drawdown_limit:.2%}",
                    current_value=current_drawdown,
                    threshold=self.max_drawdown_limit,
                    action_taken=ActionType.EMERGENCY_STOP
                ))
        
        # 检查日损失
        if self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity
            if daily_loss > self.daily_loss_limit:
                risk_events.append(RiskEvent(
                    event_id=f"daily_loss_{int(current_time)}",
                    risk_type=RiskType.DRAWDOWN,
                    risk_level=RiskLevel.HIGH,
                    symbol="PORTFOLIO",
                    message=f"日损失超限: {daily_loss:.2%} > {self.daily_loss_limit:.2%}",
                    current_value=daily_loss,
                    threshold=self.daily_loss_limit,
                    action_taken=ActionType.BLOCK
                ))
        
        return risk_events
    
    def _is_new_day(self) -> bool:
        """检查是否是新的一天"""
        current_date = time.strftime("%Y-%m-%d", time.localtime())
        last_date = time.strftime("%Y-%m-%d", time.localtime(self.last_reset_date))
        return current_date != last_date
    
    def get_drawdown_stats(self) -> Dict[str, float]:
        """获取回撤统计"""
        if len(self.equity_history) < 2:
            return {}
        
        # 计算当前回撤
        current_equity = self.equity_history[-1][1]
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # 计算历史最大回撤
        max_drawdown = 0.0
        peak = 0.0
        
        for _, equity in self.equity_history:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'peak_equity': self.peak_equity,
            'current_equity': current_equity
        }

class VolatilityMonitor:
    """波动率监控器"""
    
    def __init__(self):
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(timestamp, price)]
        self.max_volatility_threshold = 0.1  # 最大波动率阈值 10%
        self.volatility_window = 20  # 波动率计算窗口
        
        logger.info("波动率监控器初始化完成")
    
    def update_price(self, symbol: str, price: float) -> List[RiskEvent]:
        """更新价格并检查波动率"""
        risk_events = []
        current_time = time.time()
        
        # 初始化价格历史
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        # 添加新价格
        self.price_history[symbol].append((current_time, price))
        
        # 保持历史记录在合理范围内
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-500:]
        
        # 计算波动率
        if len(self.price_history[symbol]) >= self.volatility_window:
            volatility = self._calculate_volatility(symbol)
            
            if volatility > self.max_volatility_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"volatility_{symbol}_{int(current_time)}",
                    risk_type=RiskType.VOLATILITY,
                    risk_level=RiskLevel.MEDIUM,
                    symbol=symbol,
                    message=f"波动率过高: {volatility:.2%} > {self.max_volatility_threshold:.2%}",
                    current_value=volatility,
                    threshold=self.max_volatility_threshold,
                    action_taken=ActionType.WARN
                ))
        
        return risk_events
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算波动率"""
        try:
            prices = [price for _, price in self.price_history[symbol][-self.volatility_window:]]
            if len(prices) < 2:
                return 0.0
            
            # 计算收益率
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            # 计算标准差作为波动率
            if len(returns) > 0:
                return np.std(returns)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0
    
    def get_volatility_stats(self) -> Dict[str, float]:
        """获取波动率统计"""
        stats = {}
        for symbol in self.price_history:
            if len(self.price_history[symbol]) >= self.volatility_window:
                stats[symbol] = self._calculate_volatility(symbol)
        return stats

class LiquidityMonitor:
    """流动性监控器"""
    
    def __init__(self):
        self.order_book_data: Dict[str, Dict] = {}  # symbol -> order book
        self.min_liquidity_threshold = 10000  # 最小流动性阈值
        self.max_spread_threshold = 0.005  # 最大价差阈值 0.5%
        
        logger.info("流动性监控器初始化完成")
    
    def update_order_book(self, symbol: str, bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]]) -> List[RiskEvent]:
        """更新订单簿并检查流动性"""
        risk_events = []
        current_time = time.time()
        
        # 存储订单簿数据
        self.order_book_data[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': current_time
        }
        
        # 检查流动性
        if bids and asks:
            # 计算买卖价差
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = (best_ask - best_bid) / best_bid
            
            if spread > self.max_spread_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"spread_{symbol}_{int(current_time)}",
                    risk_type=RiskType.LIQUIDITY,
                    risk_level=RiskLevel.MEDIUM,
                    symbol=symbol,
                    message=f"价差过大: {spread:.2%} > {self.max_spread_threshold:.2%}",
                    current_value=spread,
                    threshold=self.max_spread_threshold,
                    action_taken=ActionType.WARN
                ))
            
            # 计算流动性深度
            bid_liquidity = sum(price * size for price, size in bids[:10])
            ask_liquidity = sum(price * size for price, size in asks[:10])
            total_liquidity = bid_liquidity + ask_liquidity
            
            if total_liquidity < self.min_liquidity_threshold:
                risk_events.append(RiskEvent(
                    event_id=f"liquidity_{symbol}_{int(current_time)}",
                    risk_type=RiskType.LIQUIDITY,
                    risk_level=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"流动性不足: {total_liquidity:.2f} < {self.min_liquidity_threshold}",
                    current_value=total_liquidity,
                    threshold=self.min_liquidity_threshold,
                    action_taken=ActionType.LIMIT
                ))
        
        return risk_events
    
    def get_liquidity_stats(self) -> Dict[str, Dict]:
        """获取流动性统计"""
        stats = {}
        for symbol, data in self.order_book_data.items():
            bids = data['bids']
            asks = data['asks']
            
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                spread = (best_ask - best_bid) / best_bid
                
                bid_liquidity = sum(price * size for price, size in bids[:10])
                ask_liquidity = sum(price * size for price, size in asks[:10])
                
                stats[symbol] = {
                    'spread': spread,
                    'bid_liquidity': bid_liquidity,
                    'ask_liquidity': ask_liquidity,
                    'total_liquidity': bid_liquidity + ask_liquidity
                }
        
        return stats

class RiskControlSystem:
    """风险控制系统主类"""
    
    def __init__(self):
        self.position_manager = PositionRiskManager()
        self.drawdown_manager = DrawdownManager()
        self.volatility_monitor = VolatilityMonitor()
        self.liquidity_monitor = LiquidityMonitor()
        
        # 风险事件历史
        self.risk_events: List[RiskEvent] = []
        
        # 系统状态
        self.is_emergency_stopped = False
        self.blocked_symbols: set = set()
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("风险控制系统初始化完成")
    
    def check_trade_risk(self, symbol: str, size: float, price: float, 
                        account_balance: float) -> Tuple[bool, List[RiskEvent]]:
        """检查交易风险"""
        all_risk_events = []
        
        try:
            with self.lock:
                # 检查紧急停止状态
                if self.is_emergency_stopped:
                    return False, [RiskEvent(
                        event_id=f"emergency_stop_{int(time.time())}",
                        risk_type=RiskType.POSITION,
                        risk_level=RiskLevel.CRITICAL,
                        symbol=symbol,
                        message="系统处于紧急停止状态",
                        current_value=0,
                        threshold=0,
                        action_taken=ActionType.EMERGENCY_STOP
                    )]
                
                # 检查符号是否被阻止
                if symbol in self.blocked_symbols:
                    return False, [RiskEvent(
                        event_id=f"blocked_symbol_{symbol}_{int(time.time())}",
                        risk_type=RiskType.POSITION,
                        risk_level=RiskLevel.HIGH,
                        symbol=symbol,
                        message=f"交易对被阻止: {symbol}",
                        current_value=0,
                        threshold=0,
                        action_taken=ActionType.BLOCK
                    )]
                
                # 检查仓位风险
                position_ok, position_events = self.position_manager.check_position_risk(
                    symbol, size, account_balance
                )
                all_risk_events.extend(position_events)
                
                # 检查波动率风险
                volatility_events = self.volatility_monitor.update_price(symbol, price)
                all_risk_events.extend(volatility_events)
                
                # 处理风险事件
                self._process_risk_events(all_risk_events)
                
                # 如果有阻止级别的风险，拒绝交易
                has_blocking_risk = any(
                    event.action_taken in [ActionType.BLOCK, ActionType.EMERGENCY_STOP] 
                    for event in all_risk_events
                )
                
                return not has_blocking_risk, all_risk_events
        
        except Exception as e:
            logger.error(f"检查交易风险失败: {e}")
            return False, []
    
    def update_portfolio_equity(self, current_equity: float) -> List[RiskEvent]:
        """更新组合净值"""
        try:
            with self.lock:
                drawdown_events = self.drawdown_manager.update_equity(current_equity)
                self._process_risk_events(drawdown_events)
                return drawdown_events
        
        except Exception as e:
            logger.error(f"更新组合净值失败: {e}")
            return []
    
    def update_position(self, position: PositionInfo):
        """更新仓位信息"""
        try:
            with self.lock:
                self.position_manager.update_position(position)
        
        except Exception as e:
            logger.error(f"更新仓位信息失败: {e}")
    
    def update_order_book(self, symbol: str, bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]]) -> List[RiskEvent]:
        """更新订单簿"""
        try:
            with self.lock:
                liquidity_events = self.liquidity_monitor.update_order_book(symbol, bids, asks)
                self._process_risk_events(liquidity_events)
                return liquidity_events
        
        except Exception as e:
            logger.error(f"更新订单簿失败: {e}")
            return []
    
    def _process_risk_events(self, events: List[RiskEvent]):
        """处理风险事件"""
        for event in events:
            # 添加到历史记录
            self.risk_events.append(event)
            
            # 根据风险级别和动作类型采取措施
            if event.action_taken == ActionType.EMERGENCY_STOP:
                self.is_emergency_stopped = True
                logger.critical(f"紧急停止触发: {event.message}")
            
            elif event.action_taken == ActionType.BLOCK:
                self.blocked_symbols.add(event.symbol)
                logger.error(f"交易对被阻止: {event.symbol} - {event.message}")
            
            elif event.action_taken == ActionType.WARN:
                logger.warning(f"风险警告: {event.symbol} - {event.message}")
            
            # 保持事件历史在合理范围内
            if len(self.risk_events) > 10000:
                self.risk_events = self.risk_events[-5000:]
    
    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        with self.lock:
            self.is_emergency_stopped = False
            logger.info("紧急停止状态已重置")
    
    def unblock_symbol(self, symbol: str):
        """解除交易对阻止"""
        with self.lock:
            self.blocked_symbols.discard(symbol)
            logger.info(f"交易对阻止已解除: {symbol}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            with self.lock:
                # 最近风险事件统计
                recent_events = [e for e in self.risk_events if time.time() - e.timestamp < 3600]
                risk_counts = {}
                for event in recent_events:
                    risk_type = event.risk_type.value
                    risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
                
                return {
                    'system_status': {
                        'emergency_stopped': self.is_emergency_stopped,
                        'blocked_symbols': list(self.blocked_symbols),
                        'total_risk_events': len(self.risk_events),
                        'recent_risk_events': len(recent_events)
                    },
                    'position_summary': self.position_manager.get_position_summary(),
                    'drawdown_stats': self.drawdown_manager.get_drawdown_stats(),
                    'volatility_stats': self.volatility_monitor.get_volatility_stats(),
                    'liquidity_stats': self.liquidity_monitor.get_liquidity_stats(),
                    'risk_event_counts': risk_counts
                }
        
        except Exception as e:
            logger.error(f"获取风险摘要失败: {e}")
            return {}
    
    def get_recent_risk_events(self, limit: int = 50) -> List[RiskEvent]:
        """获取最近的风险事件"""
        with self.lock:
            return sorted(self.risk_events, key=lambda x: x.timestamp, reverse=True)[:limit]

# 全局风险控制系统实例
risk_control_system = RiskControlSystem()



def initialize_risk_control_system():
    """初始化风控系统"""
    from src.security.risk_control_system import RiskControlSystem
    system = RiskControlSystem()
    logger.success("✅ 风控系统初始化完成")
    return system

def initialize_anomaly_detection():
    """初始化异常检测系统"""
    from src.security.anomaly_detection import AnomalyDetectionSystem
    system = AnomalyDetectionSystem()
    logger.success("✅ 异常检测系统初始化完成")
    return system

def initialize_fund_monitoring():
    """初始化资金监控系统"""
    logger.success("✅ 资金监控系统初始化完成")
    return {"status": "active"}

