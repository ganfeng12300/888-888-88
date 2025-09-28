"""
🎯 高频交易策略引擎
生产级高频策略系统，实现统计套利、市场做市、动量反转等完整策略
支持微秒级信号生成、实时风险控制和策略性能监控
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings
from src.high_frequency.low_latency_engine import (
    Order, OrderType, OrderSide, OrderStatus, MarketData, Trade, OrderBook
)


class StrategyType(Enum):
    """策略类型"""
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"    # 统计套利
    MARKET_MAKING = "market_making"                     # 市场做市
    MOMENTUM_REVERSAL = "momentum_reversal"             # 动量反转
    PAIRS_TRADING = "pairs_trading"                     # 配对交易
    MEAN_REVERSION = "mean_reversion"                   # 均值回归
    TREND_FOLLOWING = "trend_following"                 # 趋势跟踪
    SCALPING = "scalping"                               # 剥头皮
    CROSS_EXCHANGE_ARBITRAGE = "cross_exchange_arb"     # 跨交易所套利


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """交易信号"""
    strategy_id: str                    # 策略ID
    symbol: str                         # 交易对
    signal_type: SignalType             # 信号类型
    strength: float                     # 信号强度 (0-1)
    price: float                        # 建议价格
    quantity: float                     # 建议数量
    confidence: float                   # 置信度 (0-1)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """策略性能指标"""
    strategy_id: str                    # 策略ID
    total_trades: int = 0               # 总交易数
    winning_trades: int = 0             # 盈利交易数
    losing_trades: int = 0              # 亏损交易数
    total_pnl: float = 0.0              # 总盈亏
    max_drawdown: float = 0.0           # 最大回撤
    sharpe_ratio: float = 0.0           # 夏普比率
    win_rate: float = 0.0               # 胜率
    avg_win: float = 0.0                # 平均盈利
    avg_loss: float = 0.0               # 平均亏损
    profit_factor: float = 0.0          # 盈利因子
    signals_generated: int = 0          # 生成信号数
    signals_executed: int = 0           # 执行信号数
    execution_rate: float = 0.0         # 执行率
    last_update: float = field(default_factory=time.time)


class BaseStrategy:
    """基础策略类"""
    
    def __init__(self, strategy_id: str, strategy_type: StrategyType):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.is_active = False
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # 策略参数
        self.parameters = {}
        
        # 历史数据缓存
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=100)
        
        # 性能指标
        self.metrics = StrategyMetrics(strategy_id=strategy_id)
        
        # 风险控制
        self.max_position = 1000.0
        self.max_loss_per_trade = 100.0
        self.daily_loss_limit = 1000.0
        self.daily_pnl = 0.0
        
        logger.info(f"策略初始化: {strategy_id} ({strategy_type.value})")
    
    async def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """生成交易信号 - 子类需要实现"""
        raise NotImplementedError("子类必须实现generate_signal方法")
    
    async def update_market_data(self, market_data: MarketData):
        """更新市场数据"""
        self.price_history.append(market_data.last_price)
        self.volume_history.append(market_data.volume)
        
        # 更新未实现盈亏
        if self.position != 0:
            self.unrealized_pnl = (market_data.last_price - self.entry_price) * self.position
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """执行交易信号"""
        try:
            # 风险检查
            if not self._risk_check(signal):
                return False
            
            # 更新仓位
            if signal.signal_type == SignalType.BUY:
                self.position += signal.quantity
                self.entry_price = signal.price
            elif signal.signal_type == SignalType.SELL:
                self.position -= signal.quantity
                self.entry_price = signal.price
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                # 平仓
                self.realized_pnl += self.unrealized_pnl
                self.daily_pnl += self.unrealized_pnl
                self.position = 0.0
                self.entry_price = 0.0
                self.unrealized_pnl = 0.0
            
            # 更新指标
            self.metrics.signals_executed += 1
            self.signal_history.append(signal)
            
            return True
            
        except Exception as e:
            logger.error(f"执行信号失败: {e}")
            return False
    
    def _risk_check(self, signal: TradingSignal) -> bool:
        """风险检查"""
        # 检查仓位限制
        new_position = self.position
        if signal.signal_type == SignalType.BUY:
            new_position += signal.quantity
        elif signal.signal_type == SignalType.SELL:
            new_position -= signal.quantity
        
        if abs(new_position) > self.max_position:
            logger.warning(f"仓位超限: {new_position} > {self.max_position}")
            return False
        
        # 检查日损失限制
        if self.daily_pnl < -self.daily_loss_limit:
            logger.warning(f"日损失超限: {self.daily_pnl} < -{self.daily_loss_limit}")
            return False
        
        return True
    
    def update_metrics(self):
        """更新性能指标"""
        if len(self.signal_history) > 0:
            self.metrics.execution_rate = self.metrics.signals_executed / self.metrics.signals_generated
        
        self.metrics.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.metrics.last_update = time.time()


class StatisticalArbitrageStrategy(BaseStrategy):
    """统计套利策略"""
    
    def __init__(self, strategy_id: str):
        super().__init__(strategy_id, StrategyType.STATISTICAL_ARBITRAGE)
        
        # 策略参数
        self.parameters = {
            'lookback_period': 20,          # 回望期
            'z_score_threshold': 2.0,       # Z分数阈值
            'mean_reversion_speed': 0.1,    # 均值回归速度
            'volatility_window': 10,        # 波动率窗口
            'correlation_threshold': 0.8,   # 相关性阈值
        }
        
        # 统计数据
        self.price_mean = 0.0
        self.price_std = 0.0
        self.z_score = 0.0
        self.correlation = 0.0
    
    async def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """生成统计套利信号"""
        try:
            await self.update_market_data(market_data)
            
            if len(self.price_history) < self.parameters['lookback_period']:
                return None
            
            # 计算统计指标
            prices = np.array(list(self.price_history))
            self.price_mean = np.mean(prices[-self.parameters['lookback_period']:])
            self.price_std = np.std(prices[-self.parameters['lookback_period']:])
            
            if self.price_std == 0:
                return None
            
            # 计算Z分数
            current_price = market_data.last_price
            self.z_score = (current_price - self.price_mean) / self.price_std
            
            # 生成信号
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            if self.z_score > self.parameters['z_score_threshold']:
                # 价格过高，卖出信号
                signal_type = SignalType.SELL
                strength = min(abs(self.z_score) / self.parameters['z_score_threshold'], 1.0)
                confidence = strength
            elif self.z_score < -self.parameters['z_score_threshold']:
                # 价格过低，买入信号
                signal_type = SignalType.BUY
                strength = min(abs(self.z_score) / self.parameters['z_score_threshold'], 1.0)
                confidence = strength
            
            if signal_type != SignalType.HOLD:
                quantity = min(100 * strength, self.max_position - abs(self.position))
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=market_data.symbol,
                    signal_type=signal_type,
                    strength=strength,
                    price=current_price,
                    quantity=quantity,
                    confidence=confidence,
                    metadata={
                        'z_score': self.z_score,
                        'price_mean': self.price_mean,
                        'price_std': self.price_std
                    }
                )
                
                self.metrics.signals_generated += 1
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"生成统计套利信号失败: {e}")
            return None


class MarketMakingStrategy(BaseStrategy):
    """市场做市策略"""
    
    def __init__(self, strategy_id: str):
        super().__init__(strategy_id, StrategyType.MARKET_MAKING)
        
        # 策略参数
        self.parameters = {
            'spread_threshold': 0.001,      # 价差阈值
            'inventory_limit': 500.0,       # 库存限制
            'quote_size': 10.0,             # 报价数量
            'skew_factor': 0.5,             # 偏斜因子
            'volatility_adjustment': True,   # 波动率调整
            'adverse_selection_protection': True  # 逆向选择保护
        }
        
        # 做市数据
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        self.inventory = 0.0
        self.volatility = 0.0
    
    async def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """生成做市信号"""
        try:
            await self.update_market_data(market_data)
            
            # 计算中间价和价差
            self.bid_price = market_data.bid_price
            self.ask_price = market_data.ask_price
            self.mid_price = (self.bid_price + self.ask_price) / 2
            self.spread = self.ask_price - self.bid_price
            
            # 检查价差是否足够
            if self.spread < self.parameters['spread_threshold']:
                return None
            
            # 计算波动率
            if len(self.price_history) >= 10:
                prices = np.array(list(self.price_history)[-10:])
                returns = np.diff(np.log(prices))
                self.volatility = np.std(returns) * np.sqrt(252)
            
            # 库存管理
            self.inventory = self.position
            inventory_ratio = self.inventory / self.parameters['inventory_limit']
            
            # 计算最优报价
            optimal_spread = self.spread / 2
            if self.parameters['volatility_adjustment']:
                optimal_spread *= (1 + self.volatility)
            
            # 库存偏斜调整
            skew = self.parameters['skew_factor'] * inventory_ratio
            
            # 生成双边报价信号
            signals = []
            
            # 买入报价 (bid)
            if abs(self.inventory) < self.parameters['inventory_limit']:
                bid_price = self.mid_price - optimal_spread - skew
                if bid_price > self.bid_price:  # 改善最优买价
                    buy_signal = TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=market_data.symbol,
                        signal_type=SignalType.BUY,
                        strength=0.8,
                        price=bid_price,
                        quantity=self.parameters['quote_size'],
                        confidence=0.7,
                        metadata={
                            'signal_source': 'market_making_bid',
                            'spread': self.spread,
                            'inventory': self.inventory,
                            'skew': skew
                        }
                    )
                    signals.append(buy_signal)
            
            # 卖出报价 (ask)
            if abs(self.inventory) < self.parameters['inventory_limit']:
                ask_price = self.mid_price + optimal_spread + skew
                if ask_price < self.ask_price:  # 改善最优卖价
                    sell_signal = TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=market_data.symbol,
                        signal_type=SignalType.SELL,
                        strength=0.8,
                        price=ask_price,
                        quantity=self.parameters['quote_size'],
                        confidence=0.7,
                        metadata={
                            'signal_source': 'market_making_ask',
                            'spread': self.spread,
                            'inventory': self.inventory,
                            'skew': skew
                        }
                    )
                    signals.append(sell_signal)
            
            if signals:
                self.metrics.signals_generated += len(signals)
                return signals[0]  # 返回第一个信号，实际应该同时处理双边
            
            return None
            
        except Exception as e:
            logger.error(f"生成做市信号失败: {e}")
            return None


class MomentumReversalStrategy(BaseStrategy):
    """动量反转策略"""
    
    def __init__(self, strategy_id: str):
        super().__init__(strategy_id, StrategyType.MOMENTUM_REVERSAL)
        
        # 策略参数
        self.parameters = {
            'momentum_window': 5,           # 动量窗口
            'reversal_threshold': 0.02,     # 反转阈值
            'volume_confirmation': True,    # 成交量确认
            'rsi_period': 14,               # RSI周期
            'rsi_overbought': 70,           # RSI超买
            'rsi_oversold': 30,             # RSI超卖
        }
        
        # 技术指标
        self.momentum = 0.0
        self.rsi = 50.0
        self.volume_ratio = 1.0
        self.price_change = 0.0
    
    async def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """生成动量反转信号"""
        try:
            await self.update_market_data(market_data)
            
            if len(self.price_history) < self.parameters['momentum_window']:
                return None
            
            # 计算动量
            prices = np.array(list(self.price_history))
            current_price = market_data.last_price
            
            # 短期动量
            momentum_period = self.parameters['momentum_window']
            if len(prices) >= momentum_period:
                self.momentum = (current_price - prices[-momentum_period]) / prices[-momentum_period]
            
            # 计算RSI
            if len(prices) >= self.parameters['rsi_period']:
                self.rsi = self._calculate_rsi(prices, self.parameters['rsi_period'])
            
            # 计算成交量比率
            if len(self.volume_history) >= 5:
                volumes = np.array(list(self.volume_history))
                current_volume = market_data.volume
                avg_volume = np.mean(volumes[-5:])
                self.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 价格变化
            if len(prices) >= 2:
                self.price_change = (current_price - prices[-2]) / prices[-2]
            
            # 生成信号
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # 动量反转逻辑
            if (abs(self.momentum) > self.parameters['reversal_threshold'] and
                abs(self.price_change) > 0.005):  # 价格有显著变化
                
                # 超买反转 (卖出)
                if (self.momentum > 0 and self.rsi > self.parameters['rsi_overbought']):
                    signal_type = SignalType.SELL
                    strength = min(abs(self.momentum) / self.parameters['reversal_threshold'], 1.0)
                    confidence = (self.rsi - 50) / 50  # RSI越极端，置信度越高
                
                # 超卖反转 (买入)
                elif (self.momentum < 0 and self.rsi < self.parameters['rsi_oversold']):
                    signal_type = SignalType.BUY
                    strength = min(abs(self.momentum) / self.parameters['reversal_threshold'], 1.0)
                    confidence = (50 - self.rsi) / 50  # RSI越极端，置信度越高
            
            # 成交量确认
            if (self.parameters['volume_confirmation'] and 
                signal_type != SignalType.HOLD and 
                self.volume_ratio < 1.2):  # 成交量不足
                return None
            
            if signal_type != SignalType.HOLD:
                quantity = min(50 * strength, self.max_position - abs(self.position))
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=market_data.symbol,
                    signal_type=signal_type,
                    strength=strength,
                    price=current_price,
                    quantity=quantity,
                    confidence=confidence,
                    metadata={
                        'momentum': self.momentum,
                        'rsi': self.rsi,
                        'volume_ratio': self.volume_ratio,
                        'price_change': self.price_change
                    }
                )
                
                self.metrics.signals_generated += 1
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"生成动量反转信号失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class HighFrequencyStrategyEngine:
    """高频策略引擎"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_signals: List[TradingSignal] = []
        self.signal_queue = deque(maxlen=10000)
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="StrategyEngine")
        
        # 性能统计
        self.engine_stats = {
            'total_signals_generated': 0,
            'total_signals_executed': 0,
            'strategies_count': 0,
            'active_strategies': 0,
            'avg_signal_latency_us': 0.0,
            'last_update': time.time()
        }
        
        # 运行状态
        self.running = False
        self.processing_thread = None
        
        logger.info("高频策略引擎初始化完成")
    
    def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        self.strategies[strategy.strategy_id] = strategy
        self.engine_stats['strategies_count'] = len(self.strategies)
        logger.info(f"添加策略: {strategy.strategy_id} ({strategy.strategy_type.value})")
    
    def remove_strategy(self, strategy_id: str):
        """移除策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.engine_stats['strategies_count'] = len(self.strategies)
            logger.info(f"移除策略: {strategy_id}")
    
    def activate_strategy(self, strategy_id: str):
        """激活策略"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].is_active = True
            self._update_active_count()
            logger.info(f"激活策略: {strategy_id}")
    
    def deactivate_strategy(self, strategy_id: str):
        """停用策略"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].is_active = False
            self._update_active_count()
            logger.info(f"停用策略: {strategy_id}")
    
    def _update_active_count(self):
        """更新活跃策略数量"""
        self.engine_stats['active_strategies'] = sum(
            1 for strategy in self.strategies.values() if strategy.is_active
        )
    
    async def process_market_data(self, market_data: MarketData):
        """处理市场数据并生成信号"""
        try:
            start_time = time.perf_counter_ns()
            
            # 并发处理所有活跃策略
            tasks = []
            for strategy in self.strategies.values():
                if strategy.is_active:
                    task = asyncio.create_task(strategy.generate_signal(market_data))
                    tasks.append((strategy.strategy_id, task))
            
            # 等待所有策略完成
            for strategy_id, task in tasks:
                try:
                    signal = await task
                    if signal:
                        self.signal_queue.append(signal)
                        self.active_signals.append(signal)
                        self.engine_stats['total_signals_generated'] += 1
                except Exception as e:
                    logger.error(f"策略 {strategy_id} 处理失败: {e}")
            
            # 更新延迟统计
            end_time = time.perf_counter_ns()
            latency_us = (end_time - start_time) / 1000.0
            self._update_latency_stats(latency_us)
            
        except Exception as e:
            logger.error(f"处理市场数据失败: {e}")
    
    async def execute_signals(self) -> List[Order]:
        """执行交易信号"""
        orders = []
        executed_signals = []
        
        try:
            for signal in self.active_signals:
                # 转换信号为订单
                order = self._signal_to_order(signal)
                if order:
                    orders.append(order)
                    executed_signals.append(signal)
                    
                    # 更新策略
                    if signal.strategy_id in self.strategies:
                        strategy = self.strategies[signal.strategy_id]
                        await strategy.execute_signal(signal)
                    
                    self.engine_stats['total_signals_executed'] += 1
            
            # 清除已执行的信号
            for signal in executed_signals:
                if signal in self.active_signals:
                    self.active_signals.remove(signal)
            
            return orders
            
        except Exception as e:
            logger.error(f"执行信号失败: {e}")
            return []
    
    def _signal_to_order(self, signal: TradingSignal) -> Optional[Order]:
        """将信号转换为订单"""
        try:
            # 确定订单类型
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                order_type = OrderType.LIMIT  # 默认使用限价单
                side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            else:
                return None  # 其他信号类型暂不处理
            
            # 创建订单
            order = Order(
                order_id=f"order_{int(time.time_ns())}",
                symbol=signal.symbol,
                side=side,
                order_type=order_type,
                quantity=signal.quantity,
                price=signal.price,
                metadata={
                    'strategy_id': signal.strategy_id,
                    'signal_strength': signal.strength,
                    'signal_confidence': signal.confidence,
                    'signal_metadata': signal.metadata
                }
            )
            
            return order
            
        except Exception as e:
            logger.error(f"信号转换订单失败: {e}")
            return None
    
    def _update_latency_stats(self, latency_us: float):
        """更新延迟统计"""
        # 简单的移动平均
        alpha = 0.1
        self.engine_stats['avg_signal_latency_us'] = (
            alpha * latency_us + 
            (1 - alpha) * self.engine_stats['avg_signal_latency_us']
        )
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            strategy.update_metrics()
            return strategy.metrics
        return None
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        self.engine_stats['last_update'] = time.time()
        return self.engine_stats.copy()
    
    def get_active_signals(self) -> List[TradingSignal]:
        """获取活跃信号"""
        return self.active_signals.copy()
    
    async def shutdown(self):
        """关闭引擎"""
        try:
            self.running = False
            self.executor.shutdown(wait=True)
            logger.info("高频策略引擎已关闭")
        except Exception as e:
            logger.error(f"关闭高频策略引擎失败: {e}")


# 全局策略引擎实例
hf_strategy_engine = HighFrequencyStrategyEngine()
