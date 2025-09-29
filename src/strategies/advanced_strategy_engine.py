"""
📈 高级策略引擎 - 生产级实盘交易多策略智能引擎
集成多种高级交易策略，支持策略组合、动态优化、风险管理
提供网格交易、趋势跟踪、均值回归、套利等多种策略实现
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class StrategyType(Enum):
    """策略类型"""
    GRID_TRADING = "grid_trading"  # 网格交易
    TREND_FOLLOWING = "trend_following"  # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    ARBITRAGE = "arbitrage"  # 套利
    MOMENTUM = "momentum"  # 动量策略
    SCALPING = "scalping"  # 剥头皮
    SWING_TRADING = "swing_trading"  # 摆动交易

class StrategyStatus(Enum):
    """策略状态"""
    INACTIVE = "inactive"  # 未激活
    ACTIVE = "active"  # 激活
    PAUSED = "paused"  # 暂停
    ERROR = "error"  # 错误

class SignalType(Enum):
    """信号类型"""
    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出
    HOLD = "hold"  # 持有
    CLOSE = "close"  # 平仓

@dataclass
class TradingSignal:
    """交易信号"""
    strategy_id: str  # 策略ID
    symbol: str  # 交易对
    signal_type: SignalType  # 信号类型
    price: float  # 价格
    quantity: float  # 数量
    confidence: float  # 置信度
    stop_loss: Optional[float] = None  # 止损价格
    take_profit: Optional[float] = None  # 止盈价格
    timestamp: float = field(default_factory=time.time)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class StrategyPerformance:
    """策略性能"""
    strategy_id: str  # 策略ID
    total_trades: int  # 总交易次数
    winning_trades: int  # 盈利交易次数
    total_pnl: float  # 总盈亏
    win_rate: float  # 胜率
    avg_profit: float  # 平均盈利
    avg_loss: float  # 平均亏损
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    start_time: float  # 开始时间
    end_time: float  # 结束时间

class BaseStrategy(ABC):
    """基础策略抽象类"""
    
    def __init__(self, strategy_id: str, symbol: str, parameters: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.parameters = parameters
        self.status = StrategyStatus.INACTIVE
        
        # 性能统计
        self.trades_history: List[Dict[str, Any]] = []
        self.signals_history: List[TradingSignal] = []
        self.equity_curve: List[Tuple[float, float]] = []  # (timestamp, equity)
        
        # 风险管理
        self.max_position_size = parameters.get('max_position_size', 0.1)
        self.stop_loss_pct = parameters.get('stop_loss_pct', 0.02)
        self.take_profit_pct = parameters.get('take_profit_pct', 0.04)
        
        logger.info(f"策略初始化: {strategy_id} - {symbol}")
    
    @abstractmethod
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """更新策略参数"""
        pass
    
    def start(self):
        """启动策略"""
        self.status = StrategyStatus.ACTIVE
        logger.info(f"策略启动: {self.strategy_id}")
    
    def pause(self):
        """暂停策略"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"策略暂停: {self.strategy_id}")
    
    def stop(self):
        """停止策略"""
        self.status = StrategyStatus.INACTIVE
        logger.info(f"策略停止: {self.strategy_id}")
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """记录交易"""
        self.trades_history.append(trade_data)
        
        # 保持历史记录在合理范围内
        if len(self.trades_history) > 1000:
            self.trades_history = self.trades_history[-500:]
    
    def record_signal(self, signal: TradingSignal):
        """记录信号"""
        self.signals_history.append(signal)
        
        # 保持历史记录在合理范围内
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-500:]
    
    def calculate_performance(self) -> StrategyPerformance:
        """计算策略性能"""
        try:
            if not self.trades_history:
                return StrategyPerformance(
                    strategy_id=self.strategy_id,
                    total_trades=0,
                    winning_trades=0,
                    total_pnl=0.0,
                    win_rate=0.0,
                    avg_profit=0.0,
                    avg_loss=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    start_time=time.time(),
                    end_time=time.time()
                )
            
            # 计算基础统计
            total_trades = len(self.trades_history)
            winning_trades = sum(1 for trade in self.trades_history if trade.get('pnl', 0) > 0)
            total_pnl = sum(trade.get('pnl', 0) for trade in self.trades_history)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算平均盈利和亏损
            profits = [trade['pnl'] for trade in self.trades_history if trade.get('pnl', 0) > 0]
            losses = [trade['pnl'] for trade in self.trades_history if trade.get('pnl', 0) < 0]
            
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown()
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # 时间范围
            start_time = min(trade.get('timestamp', time.time()) for trade in self.trades_history)
            end_time = max(trade.get('timestamp', time.time()) for trade in self.trades_history)
            
            return StrategyPerformance(
                strategy_id=self.strategy_id,
                total_trades=total_trades,
                winning_trades=winning_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                start_time=start_time,
                end_time=end_time
            )
        
        except Exception as e:
            logger.error(f"计算策略性能失败: {e}")
            return StrategyPerformance(
                strategy_id=self.strategy_id,
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
                win_rate=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                start_time=time.time(),
                end_time=time.time()
            )
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        try:
            if not self.equity_curve:
                return 0.0
            
            equity_values = [equity for _, equity in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0.0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
        
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.trades_history) < 2:
                return 0.0
            
            returns = [trade.get('pnl', 0) for trade in self.trades_history]
            
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # 假设无风险利率为0
            sharpe_ratio = mean_return / std_return
            
            return sharpe_ratio
        
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0

class GridTradingStrategy(BaseStrategy):
    """网格交易策略"""
    
    def __init__(self, strategy_id: str, symbol: str, parameters: Dict[str, Any]):
        super().__init__(strategy_id, symbol, parameters)
        
        # 网格参数
        self.grid_size = parameters.get('grid_size', 0.01)  # 网格间距
        self.grid_levels = parameters.get('grid_levels', 10)  # 网格层数
        self.base_quantity = parameters.get('base_quantity', 100)  # 基础数量
        
        # 网格状态
        self.center_price = 0.0
        self.grid_orders: Dict[float, Dict] = {}  # price -> order_info
        
        logger.info(f"网格交易策略初始化: {strategy_id}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成网格交易信号"""
        try:
            if self.status != StrategyStatus.ACTIVE or len(market_data) == 0:
                return None
            
            current_price = market_data['close'].iloc[-1]
            
            # 初始化中心价格
            if self.center_price == 0:
                self.center_price = current_price
                self._initialize_grid()
            
            # 检查网格触发
            signal = self._check_grid_trigger(current_price)
            
            if signal:
                self.record_signal(signal)
            
            return signal
        
        except Exception as e:
            logger.error(f"网格交易信号生成失败: {e}")
            return None
    
    def _initialize_grid(self):
        """初始化网格"""
        try:
            # 清空现有网格
            self.grid_orders.clear()
            
            # 创建买入网格
            for i in range(1, self.grid_levels + 1):
                buy_price = self.center_price * (1 - self.grid_size * i)
                self.grid_orders[buy_price] = {
                    'type': 'buy',
                    'quantity': self.base_quantity,
                    'triggered': False
                }
            
            # 创建卖出网格
            for i in range(1, self.grid_levels + 1):
                sell_price = self.center_price * (1 + self.grid_size * i)
                self.grid_orders[sell_price] = {
                    'type': 'sell',
                    'quantity': self.base_quantity,
                    'triggered': False
                }
            
            logger.info(f"网格初始化完成: 中心价格 {self.center_price}, 网格数量 {len(self.grid_orders)}")
        
        except Exception as e:
            logger.error(f"网格初始化失败: {e}")
    
    def _check_grid_trigger(self, current_price: float) -> Optional[TradingSignal]:
        """检查网格触发"""
        try:
            for grid_price, order_info in self.grid_orders.items():
                if order_info['triggered']:
                    continue
                
                # 检查买入网格触发
                if (order_info['type'] == 'buy' and 
                    current_price <= grid_price):
                    
                    order_info['triggered'] = True
                    
                    return TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        signal_type=SignalType.BUY,
                        price=grid_price,
                        quantity=order_info['quantity'],
                        confidence=0.8,
                        metadata={'grid_price': grid_price, 'grid_type': 'buy'}
                    )
                
                # 检查卖出网格触发
                elif (order_info['type'] == 'sell' and 
                      current_price >= grid_price):
                    
                    order_info['triggered'] = True
                    
                    return TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        signal_type=SignalType.SELL,
                        price=grid_price,
                        quantity=order_info['quantity'],
                        confidence=0.8,
                        metadata={'grid_price': grid_price, 'grid_type': 'sell'}
                    )
            
            return None
        
        except Exception as e:
            logger.error(f"检查网格触发失败: {e}")
            return None
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """更新策略参数"""
        try:
            self.parameters.update(new_parameters)
            
            # 更新网格参数
            if 'grid_size' in new_parameters:
                self.grid_size = new_parameters['grid_size']
            if 'grid_levels' in new_parameters:
                self.grid_levels = new_parameters['grid_levels']
            if 'base_quantity' in new_parameters:
                self.base_quantity = new_parameters['base_quantity']
            
            # 重新初始化网格
            if self.center_price > 0:
                self._initialize_grid()
            
            logger.info(f"网格策略参数更新: {self.strategy_id}")
        
        except Exception as e:
            logger.error(f"更新网格策略参数失败: {e}")

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, strategy_id: str, symbol: str, parameters: Dict[str, Any]):
        super().__init__(strategy_id, symbol, parameters)
        
        # 趋势参数
        self.fast_ma_period = parameters.get('fast_ma_period', 10)
        self.slow_ma_period = parameters.get('slow_ma_period', 30)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
        
        # 状态跟踪
        self.current_position = 0  # 1: long, -1: short, 0: neutral
        self.last_signal_time = 0
        self.min_signal_interval = parameters.get('min_signal_interval', 300)  # 5分钟
        
        logger.info(f"趋势跟踪策略初始化: {strategy_id}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成趋势跟踪信号"""
        try:
            if (self.status != StrategyStatus.ACTIVE or 
                len(market_data) < max(self.slow_ma_period, self.rsi_period)):
                return None
            
            # 检查信号间隔
            current_time = time.time()
            if current_time - self.last_signal_time < self.min_signal_interval:
                return None
            
            # 计算技术指标
            fast_ma = market_data['close'].rolling(window=self.fast_ma_period).mean()
            slow_ma = market_data['close'].rolling(window=self.slow_ma_period).mean()
            rsi = self._calculate_rsi(market_data['close'], self.rsi_period)
            
            current_price = market_data['close'].iloc[-1]
            current_fast_ma = fast_ma.iloc[-1]
            current_slow_ma = slow_ma.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # 趋势判断
            trend_up = current_fast_ma > current_slow_ma
            trend_down = current_fast_ma < current_slow_ma
            
            # 生成信号
            signal = None
            
            # 买入信号
            if (trend_up and 
                self.current_position <= 0 and 
                current_rsi < self.rsi_overbought):
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    quantity=self.base_quantity,
                    confidence=self._calculate_confidence(current_fast_ma, current_slow_ma, current_rsi),
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    take_profit=current_price * (1 + self.take_profit_pct),
                    metadata={
                        'fast_ma': current_fast_ma,
                        'slow_ma': current_slow_ma,
                        'rsi': current_rsi,
                        'trend': 'up'
                    }
                )
                
                self.current_position = 1
            
            # 卖出信号
            elif (trend_down and 
                  self.current_position >= 0 and 
                  current_rsi > self.rsi_oversold):
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=self.base_quantity,
                    confidence=self._calculate_confidence(current_fast_ma, current_slow_ma, current_rsi),
                    stop_loss=current_price * (1 + self.stop_loss_pct),
                    take_profit=current_price * (1 - self.take_profit_pct),
                    metadata={
                        'fast_ma': current_fast_ma,
                        'slow_ma': current_slow_ma,
                        'rsi': current_rsi,
                        'trend': 'down'
                    }
                )
                
                self.current_position = -1
            
            if signal:
                self.last_signal_time = current_time
                self.record_signal(signal)
            
            return signal
        
        except Exception as e:
            logger.error(f"趋势跟踪信号生成失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_confidence(self, fast_ma: float, slow_ma: float, rsi: float) -> float:
        """计算信号置信度"""
        try:
            # 基于MA差距的置信度
            ma_diff = abs(fast_ma - slow_ma) / slow_ma
            ma_confidence = min(ma_diff * 10, 0.5)  # 最大0.5
            
            # 基于RSI的置信度
            if rsi > 50:
                rsi_confidence = min((rsi - 50) / 50 * 0.3, 0.3)
            else:
                rsi_confidence = min((50 - rsi) / 50 * 0.3, 0.3)
            
            # 基础置信度
            base_confidence = 0.2
            
            total_confidence = base_confidence + ma_confidence + rsi_confidence
            return min(total_confidence, 1.0)
        
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """更新策略参数"""
        try:
            self.parameters.update(new_parameters)
            
            if 'fast_ma_period' in new_parameters:
                self.fast_ma_period = new_parameters['fast_ma_period']
            if 'slow_ma_period' in new_parameters:
                self.slow_ma_period = new_parameters['slow_ma_period']
            if 'rsi_period' in new_parameters:
                self.rsi_period = new_parameters['rsi_period']
            if 'rsi_overbought' in new_parameters:
                self.rsi_overbought = new_parameters['rsi_overbought']
            if 'rsi_oversold' in new_parameters:
                self.rsi_oversold = new_parameters['rsi_oversold']
            
            logger.info(f"趋势跟踪策略参数更新: {self.strategy_id}")
        
        except Exception as e:
            logger.error(f"更新趋势跟踪策略参数失败: {e}")


def initialize_advanced_strategy_engine():
    """初始化高级策略引擎"""
    from src.strategies.advanced_strategy_engine import AdvancedStrategyEngine
    engine = AdvancedStrategyEngine()
    logger.success("✅ 高级策略引擎初始化完成")
    return engine

def initialize_strategy_manager():
    """初始化策略管理器"""
    from src.strategies.strategy_manager import StrategyManager
    manager = StrategyManager()
    logger.success("✅ 策略管理器初始化完成")
    return manager

def initialize_portfolio_optimizer():
    """初始化投资组合优化器"""
    from src.strategies.portfolio_optimizer import PortfolioOptimizer
    optimizer = PortfolioOptimizer()
    logger.success("✅ 投资组合优化器初始化完成")
    return optimizer

