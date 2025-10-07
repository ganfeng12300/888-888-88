#!/usr/bin/env python3
"""
⚡ 888-888-88 策略执行器
生产级交易策略执行和管理系统
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
import json
import threading
from collections import defaultdict, deque


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    DCA = "dca"  # Dollar Cost Averaging
    AI_SIGNAL = "ai_signal"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class StrategyStatus(Enum):
    """策略状态"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    strategy_type: StrategyType
    symbols: List[str]
    max_position_size: float = 0.1  # 最大仓位比例
    risk_per_trade: float = 0.02    # 每笔交易风险
    stop_loss_pct: float = 0.02     # 止损百分比
    take_profit_pct: float = 0.06   # 止盈百分比
    max_open_positions: int = 5     # 最大开仓数
    min_signal_strength: float = 0.6 # 最小信号强度
    cooldown_period: int = 300      # 冷却期（秒）
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: str  # buy, sell, hold
    strength: float   # 0-1
    price: float
    timestamp: datetime
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyOrder:
    """策略订单"""
    strategy_name: str
    symbol: str
    side: str  # buy/sell
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPosition:
    """策略仓位"""
    strategy_name: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """基础策略类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.status = StrategyStatus.INACTIVE
        self.positions: Dict[str, StrategyPosition] = {}
        self.pending_orders: List[StrategyOrder] = []
        self.signal_history: deque = deque(maxlen=1000)
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        self.last_signal_time: Dict[str, datetime] = {}
        
    async def initialize(self) -> bool:
        """初始化策略"""
        self.status = StrategyStatus.ACTIVE
        logger.info(f"🚀 策略 {self.config.name} 初始化完成")
        return True
    
    async def process_market_data(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """处理市场数据，生成交易信号"""
        raise NotImplementedError("子类必须实现此方法")
    
    async def on_signal(self, signal: TradingSignal) -> List[StrategyOrder]:
        """处理交易信号，生成订单"""
        # 检查冷却期
        if not self._check_cooldown(signal.symbol):
            return []
        
        # 检查信号强度
        if signal.strength < self.config.min_signal_strength:
            return []
        
        # 检查最大持仓数
        if len(self.positions) >= self.config.max_open_positions:
            return []
        
        # 生成订单
        orders = await self._generate_orders(signal)
        
        # 记录信号
        self.signal_history.append(signal)
        self.last_signal_time[signal.symbol] = signal.timestamp
        
        return orders
    
    async def on_order_filled(self, order: StrategyOrder, fill_price: float, fill_quantity: float) -> None:
        """订单成交回调"""
        # 更新仓位
        await self._update_position(order, fill_price, fill_quantity)
        
        # 更新统计
        self.performance_stats['total_trades'] += 1
        
        logger.info(f"📈 {self.config.name} 订单成交: {order.symbol} {order.side} {fill_quantity} @ {fill_price}")
    
    async def on_position_update(self, symbol: str, current_price: float) -> List[StrategyOrder]:
        """仓位更新回调"""
        if symbol not in self.positions:
            return []
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        # 计算未实现盈亏
        if position.side == 'buy':
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
        
        # 检查止损止盈
        return await self._check_exit_conditions(position)
    
    def _check_cooldown(self, symbol: str) -> bool:
        """检查冷却期"""
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (datetime.now() - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= self.config.cooldown_period
    
    async def _generate_orders(self, signal: TradingSignal) -> List[StrategyOrder]:
        """生成订单"""
        orders = []
        
        # 计算仓位大小
        position_size = self._calculate_position_size(signal)
        
        if position_size > 0:
            order = StrategyOrder(
                strategy_name=self.config.name,
                symbol=signal.symbol,
                side=signal.signal_type,
                order_type=OrderType.MARKET,
                quantity=position_size,
                metadata={
                    'signal_strength': signal.strength,
                    'entry_reason': 'signal_based'
                }
            )
            orders.append(order)
        
        return orders
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """计算仓位大小"""
        # 基于风险的仓位计算
        risk_amount = self.config.risk_per_trade
        stop_loss_distance = self.config.stop_loss_pct
        
        if stop_loss_distance > 0:
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = self.config.max_position_size
        
        # 应用信号强度调整
        position_size *= signal.strength
        
        # 限制最大仓位
        position_size = min(position_size, self.config.max_position_size)
        
        return position_size
    
    async def _update_position(self, order: StrategyOrder, fill_price: float, fill_quantity: float) -> None:
        """更新仓位"""
        symbol = order.symbol
        
        if symbol in self.positions:
            # 更新现有仓位
            position = self.positions[symbol]
            if position.side == order.side:
                # 同向加仓
                total_size = position.size + fill_quantity
                avg_price = (position.size * position.entry_price + fill_quantity * fill_price) / total_size
                position.size = total_size
                position.entry_price = avg_price
            else:
                # 反向交易，减仓或平仓
                if position.size > fill_quantity:
                    position.size -= fill_quantity
                else:
                    # 完全平仓
                    realized_pnl = self._calculate_realized_pnl(position, fill_price, position.size)
                    self.performance_stats['total_pnl'] += realized_pnl
                    if realized_pnl > 0:
                        self.performance_stats['winning_trades'] += 1
                    del self.positions[symbol]
        else:
            # 新建仓位
            position = StrategyPosition(
                strategy_name=self.config.name,
                symbol=symbol,
                side=order.side,
                size=fill_quantity,
                entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                entry_time=datetime.now(),
                stop_loss=self._calculate_stop_loss(order.side, fill_price),
                take_profit=self._calculate_take_profit(order.side, fill_price)
            )
            self.positions[symbol] = position
    
    def _calculate_stop_loss(self, side: str, entry_price: float) -> float:
        """计算止损价格"""
        if side == 'buy':
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)
    
    def _calculate_take_profit(self, side: str, entry_price: float) -> float:
        """计算止盈价格"""
        if side == 'buy':
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)
    
    async def _check_exit_conditions(self, position: StrategyPosition) -> List[StrategyOrder]:
        """检查退出条件"""
        orders = []
        current_price = position.current_price
        
        # 检查止损
        if position.stop_loss:
            if ((position.side == 'buy' and current_price <= position.stop_loss) or
                (position.side == 'sell' and current_price >= position.stop_loss)):
                
                exit_order = StrategyOrder(
                    strategy_name=self.config.name,
                    symbol=position.symbol,
                    side='sell' if position.side == 'buy' else 'buy',
                    order_type=OrderType.MARKET,
                    quantity=position.size,
                    metadata={'exit_reason': 'stop_loss'}
                )
                orders.append(exit_order)
        
        # 检查止盈
        if position.take_profit:
            if ((position.side == 'buy' and current_price >= position.take_profit) or
                (position.side == 'sell' and current_price <= position.take_profit)):
                
                exit_order = StrategyOrder(
                    strategy_name=self.config.name,
                    symbol=position.symbol,
                    side='sell' if position.side == 'buy' else 'buy',
                    order_type=OrderType.MARKET,
                    quantity=position.size,
                    metadata={'exit_reason': 'take_profit'}
                )
                orders.append(exit_order)
        
        return orders
    
    def _calculate_realized_pnl(self, position: StrategyPosition, exit_price: float, quantity: float) -> float:
        """计算已实现盈亏"""
        if position.side == 'buy':
            return (exit_price - position.entry_price) * quantity
        else:
            return (position.entry_price - exit_price) * quantity
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        else:
            stats['win_rate'] = 0.0
        
        return stats


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    async def process_market_data(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """处理市场数据"""
        try:
            # 获取价格数据
            close_prices = data.get('close_prices', [])
            if len(close_prices) < 50:
                return None
            
            # 计算移动平均
            short_ma = np.mean(close_prices[-20:])
            long_ma = np.mean(close_prices[-50:])
            current_price = close_prices[-1]
            
            # 生成信号
            if short_ma > long_ma * 1.01:  # 1%阈值
                signal_type = 'buy'
                strength = min((short_ma / long_ma - 1) * 10, 1.0)
            elif short_ma < long_ma * 0.99:
                signal_type = 'sell'
                strength = min((1 - short_ma / long_ma) * 10, 1.0)
            else:
                signal_type = 'hold'
                strength = 0.0
            
            if signal_type != 'hold':
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name=self.config.name,
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'ma_ratio': short_ma / long_ma
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 趋势策略处理数据失败: {e}")
            return None


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    async def process_market_data(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """处理市场数据"""
        try:
            close_prices = data.get('close_prices', [])
            if len(close_prices) < 20:
                return None
            
            # 计算布林带
            prices = np.array(close_prices[-20:])
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            upper_band = mean_price + 2 * std_price
            lower_band = mean_price - 2 * std_price
            current_price = close_prices[-1]
            
            # 生成信号
            if current_price < lower_band:
                signal_type = 'buy'
                strength = min((lower_band - current_price) / (upper_band - lower_band), 1.0)
            elif current_price > upper_band:
                signal_type = 'sell'
                strength = min((current_price - upper_band) / (upper_band - lower_band), 1.0)
            else:
                return None
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name=self.config.name,
                metadata={
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'mean_price': mean_price,
                    'bb_position': (current_price - lower_band) / (upper_band - lower_band)
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 均值回归策略处理数据失败: {e}")
            return None


class AISignalStrategy(BaseStrategy):
    """AI信号策略"""
    
    def __init__(self, config: StrategyConfig, ai_predictor: Any = None):
        super().__init__(config)
        self.ai_predictor = ai_predictor
    
    async def process_market_data(self, symbol: str, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """处理市场数据"""
        try:
            if not self.ai_predictor:
                return None
            
            # 获取AI预测
            prediction = await self.ai_predictor.get_latest_prediction(symbol)
            
            if not prediction:
                return None
            
            # 转换AI信号为交易信号
            if prediction.signal_type.value in ['buy', 'strong_buy']:
                signal_type = 'buy'
                strength = prediction.confidence
            elif prediction.signal_type.value in ['sell', 'strong_sell']:
                signal_type = 'sell'
                strength = prediction.confidence
            else:
                return None
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=prediction.current_price,
                timestamp=prediction.timestamp,
                strategy_name=self.config.name,
                metadata={
                    'ai_prediction': prediction.predicted_price,
                    'expected_return': prediction.expected_return,
                    'confidence_level': prediction.confidence_level.value,
                    'model_version': prediction.model_version
                }
            )
            
        except Exception as e:
            logger.error(f"❌ AI信号策略处理数据失败: {e}")
            return None


class StrategyExecutor:
    """策略执行器"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.executing = False
        self.execution_tasks: List[asyncio.Task] = []
        
        # 数据订阅
        self.data_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 订单管理
        self.pending_orders: List[StrategyOrder] = []
        self.order_callbacks: List[Callable] = []
        
        # 性能统计
        self.global_stats = {
            'total_strategies': 0,
            'active_strategies': 0,
            'total_orders': 0,
            'total_pnl': 0.0
        }
        
        logger.info("⚡ 策略执行器初始化完成")
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """添加策略"""
        self.strategies[strategy.config.name] = strategy
        self.global_stats['total_strategies'] += 1
        logger.info(f"📋 添加策略: {strategy.config.name}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """移除策略"""
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.status = StrategyStatus.STOPPED
            del self.strategies[strategy_name]
            self.global_stats['total_strategies'] -= 1
            logger.info(f"🗑️ 移除策略: {strategy_name}")
    
    def add_order_callback(self, callback: Callable) -> None:
        """添加订单回调"""
        self.order_callbacks.append(callback)
    
    async def start_execution(self) -> None:
        """开始执行"""
        if self.executing:
            logger.warning("⚠️ 策略执行器已在运行")
            return
        
        self.executing = True
        logger.info("🚀 开始策略执行")
        
        # 初始化所有策略
        for strategy in self.strategies.values():
            await strategy.initialize()
            if strategy.status == StrategyStatus.ACTIVE:
                self.global_stats['active_strategies'] += 1
        
        # 启动执行循环
        execution_task = asyncio.create_task(self._execution_loop())
        self.execution_tasks.append(execution_task)
        
        # 启动订单处理
        order_task = asyncio.create_task(self._order_processing_loop())
        self.execution_tasks.append(order_task)
        
        logger.info("✅ 策略执行任务已启动")
    
    async def stop_execution(self) -> None:
        """停止执行"""
        self.executing = False
        
        # 停止所有策略
        for strategy in self.strategies.values():
            strategy.status = StrategyStatus.STOPPED
        
        # 取消所有任务
        for task in self.execution_tasks:
            task.cancel()
        
        if self.execution_tasks:
            await asyncio.gather(*self.execution_tasks, return_exceptions=True)
        
        self.execution_tasks.clear()
        self.global_stats['active_strategies'] = 0
        
        logger.info("⏹️ 策略执行已停止")
    
    async def _execution_loop(self) -> None:
        """执行循环"""
        while self.executing:
            try:
                # 处理每个活跃策略
                for strategy in self.strategies.values():
                    if strategy.status != StrategyStatus.ACTIVE:
                        continue
                    
                    # 处理策略的每个交易对
                    for symbol in strategy.config.symbols:
                        try:
                            # 获取市场数据（这里需要从数据源获取）
                            market_data = await self._get_market_data(symbol)
                            
                            if market_data:
                                # 处理市场数据，生成信号
                                signal = await strategy.process_market_data(symbol, market_data)
                                
                                if signal:
                                    # 处理信号，生成订单
                                    orders = await strategy.on_signal(signal)
                                    
                                    # 添加到待处理订单队列
                                    self.pending_orders.extend(orders)
                                
                                # 更新仓位
                                if symbol in strategy.positions:
                                    current_price = market_data.get('current_price', 0)
                                    exit_orders = await strategy.on_position_update(symbol, current_price)
                                    self.pending_orders.extend(exit_orders)
                        
                        except Exception as e:
                            logger.error(f"❌ 策略 {strategy.config.name} 处理 {symbol} 失败: {e}")
                
                await asyncio.sleep(1)  # 1秒执行间隔
                
            except Exception as e:
                logger.error(f"❌ 执行循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _order_processing_loop(self) -> None:
        """订单处理循环"""
        while self.executing:
            try:
                if self.pending_orders:
                    order = self.pending_orders.pop(0)
                    
                    # 执行订单
                    success = await self._execute_order(order)
                    
                    if success:
                        self.global_stats['total_orders'] += 1
                        logger.info(f"✅ 订单执行成功: {order.symbol} {order.side} {order.quantity}")
                    else:
                        logger.error(f"❌ 订单执行失败: {order.symbol} {order.side} {order.quantity}")
                
                await asyncio.sleep(0.1)  # 100ms处理间隔
                
            except Exception as e:
                logger.error(f"❌ 订单处理异常: {e}")
                await asyncio.sleep(1)
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        # 这里应该从实际的数据源获取数据
        # 暂时返回模拟数据
        return {
            'symbol': symbol,
            'current_price': 50000.0,  # 模拟价格
            'close_prices': [50000 + i for i in range(-100, 1)],  # 模拟历史价格
            'volume': 1000.0,
            'timestamp': datetime.now()
        }
    
    async def _execute_order(self, order: StrategyOrder) -> bool:
        """执行订单"""
        try:
            # 这里应该调用实际的交易所API
            # 暂时模拟订单执行
            
            # 模拟成交价格和数量
            fill_price = 50000.0  # 模拟成交价格
            fill_quantity = order.quantity
            
            # 通知策略订单成交
            if order.strategy_name in self.strategies:
                strategy = self.strategies[order.strategy_name]
                await strategy.on_order_filled(order, fill_price, fill_quantity)
            
            # 调用订单回调
            for callback in self.order_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order, fill_price, fill_quantity)
                    else:
                        callback(order, fill_price, fill_quantity)
                except Exception as e:
                    logger.error(f"❌ 订单回调失败: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 执行订单失败: {e}")
            return False
    
    def get_strategy_status(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """获取策略状态"""
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        
        return {
            'name': strategy.config.name,
            'type': strategy.config.strategy_type.value,
            'status': strategy.status.value,
            'symbols': strategy.config.symbols,
            'positions': len(strategy.positions),
            'pending_orders': len(strategy.pending_orders),
            'performance': strategy.get_performance_stats()
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """获取全局状态"""
        active_positions = sum(len(s.positions) for s in self.strategies.values())
        total_pnl = sum(s.performance_stats['total_pnl'] for s in self.strategies.values())
        
        return {
            'executing': self.executing,
            'total_strategies': self.global_stats['total_strategies'],
            'active_strategies': self.global_stats['active_strategies'],
            'total_orders': self.global_stats['total_orders'],
            'active_positions': active_positions,
            'total_pnl': total_pnl,
            'pending_orders': len(self.pending_orders)
        }


if __name__ == "__main__":
    async def test_strategy_executor():
        # 测试策略执行器
        executor = StrategyExecutor()
        
        # 创建趋势跟踪策略
        trend_config = StrategyConfig(
            name="trend_btc",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbols=["BTC/USDT"],
            max_position_size=0.1,
            risk_per_trade=0.02
        )
        trend_strategy = TrendFollowingStrategy(trend_config)
        executor.add_strategy(trend_strategy)
        
        # 创建均值回归策略
        mean_config = StrategyConfig(
            name="mean_eth",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbols=["ETH/USDT"],
            max_position_size=0.08,
            risk_per_trade=0.015
        )
        mean_strategy = MeanReversionStrategy(mean_config)
        executor.add_strategy(mean_strategy)
        
        # 开始执行
        await executor.start_execution()
        
        # 运行一段时间
        await asyncio.sleep(30)
        
        # 获取状态
        global_status = executor.get_global_status()
        print(f"全局状态: {global_status}")
        
        trend_status = executor.get_strategy_status("trend_btc")
        print(f"趋势策略状态: {trend_status}")
        
        # 停止执行
        await executor.stop_execution()
    
    # asyncio.run(test_strategy_executor())

