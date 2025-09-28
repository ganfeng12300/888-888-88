"""
🎯 回测引擎系统
生产级历史数据回测引擎，支持策略性能评估和风险指标计算
实现完整的回测流程、性能分析和结果可视化
"""

import asyncio
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import sqlite3
from datetime import datetime, timedelta
import math

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.ai_models.ai_evolution_system import AIEvolutionSystem
from src.risk_management.risk_controller import AILevelRiskController
from src.trading_execution.smart_order_router import SmartOrderRouter


class BacktestStatus(Enum):
    """回测状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_positions: int = 10
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    benchmark_symbol: str = "BTC/USDT"


@dataclass
class MarketData:
    """市场数据"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    
    def __post_init__(self):
        if self.vwap == 0.0:
            self.vwap = (self.high + self.low + self.close) / 3


@dataclass
class BacktestOrder:
    """回测订单"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, price: float):
        """更新价格"""
        self.current_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # 基础指标
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # 收益指标
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 风险指标
    volatility: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    
    # 基准比较
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    
    # 详细数据
    equity_curve: List[Tuple[datetime, float]]
    trades: List[Dict[str, Any]]
    daily_returns: List[float]
    monthly_returns: List[float]


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self):
        # 系统组件
        self.ai_evolution_system: Optional[AIEvolutionSystem] = None
        self.risk_controller: Optional[AILevelRiskController] = None
        self.order_router: Optional[SmartOrderRouter] = None
        
        # 回测状态
        self.current_backtest: Optional[BacktestConfig] = None
        self.backtest_status = BacktestStatus.PENDING
        self.current_time: Optional[datetime] = None
        
        # 市场数据
        self.market_data: Dict[str, List[MarketData]] = {}
        self.current_prices: Dict[str, float] = {}
        
        # 交易状态
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions: Dict[str, Position] = {}
        self.orders: List[BacktestOrder] = []
        self.trades: List[Dict[str, Any]] = []
        
        # 性能记录
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.BACKTESTING, [19, 20])
        
        logger.info("回测引擎初始化完成")
    
    def initialize_components(self, components: Dict[str, Any]):
        """初始化系统组件"""
        self.ai_evolution_system = components.get('ai_evolution_system')
        self.risk_controller = components.get('risk_controller')
        self.order_router = components.get('order_router')
        
        logger.info("回测引擎组件初始化完成")
    
    def load_market_data(self, data_source: str, config: BacktestConfig):
        """加载市场数据"""
        try:
            logger.info(f"加载市场数据: {config.start_date} 到 {config.end_date}")
            
            # 生成模拟历史数据 (生产环境应该从真实数据源加载)
            for symbol in config.symbols:
                self.market_data[symbol] = self._generate_historical_data(
                    symbol, config.start_date, config.end_date
                )
            
            logger.info(f"市场数据加载完成，共{len(config.symbols)}个交易对")
            
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
            raise
    
    def _generate_historical_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime) -> List[MarketData]:
        """生成历史数据 (模拟真实市场数据)"""
        data = []
        current_date = start_date
        
        # 初始价格
        if symbol == "BTC/USDT":
            base_price = 50000.0
        elif symbol == "ETH/USDT":
            base_price = 3000.0
        else:
            base_price = 100.0
        
        current_price = base_price
        
        while current_date <= end_date:
            # 模拟价格波动 (随机游走 + 趋势)
            daily_return = np.random.normal(0.001, 0.02)  # 0.1%均值，2%波动率
            trend_factor = 1 + daily_return
            
            # 计算OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price * trend_factor
            
            # 确保OHLC逻辑正确
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # 模拟成交量
            volume = np.random.uniform(1000000, 5000000)
            
            data.append(MarketData(
                timestamp=current_date,
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
            
            current_price = close_price
            current_date += timedelta(hours=1)  # 1小时K线
        
        return data
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """运行回测"""
        try:
            logger.info("开始回测...")
            start_time = time.time()
            
            # 初始化回测状态
            self.current_backtest = config
            self.backtest_status = BacktestStatus.RUNNING
            self.cash = config.initial_capital
            self.portfolio_value = config.initial_capital
            self.positions.clear()
            self.orders.clear()
            self.trades.clear()
            self.equity_curve.clear()
            self.daily_returns.clear()
            
            # 获取所有时间点
            all_timestamps = set()
            for symbol_data in self.market_data.values():
                for data_point in symbol_data:
                    all_timestamps.add(data_point.timestamp)
            
            sorted_timestamps = sorted(all_timestamps)
            
            # 逐时间点回测
            for i, timestamp in enumerate(sorted_timestamps):
                self.current_time = timestamp
                
                # 更新市场数据
                self._update_current_prices(timestamp)
                
                # 更新持仓
                self._update_positions()
                
                # 处理订单
                await self._process_orders()
                
                # AI决策
                if self.ai_evolution_system:
                    await self._make_trading_decisions()
                
                # 风险控制
                if self.risk_controller:
                    await self._apply_risk_controls()
                
                # 记录权益曲线
                self._record_equity_curve()
                
                # 进度报告
                if i % 1000 == 0:
                    progress = i / len(sorted_timestamps) * 100
                    logger.info(f"回测进度: {progress:.1f}%")
            
            # 平仓所有持仓
            await self._close_all_positions()
            
            # 计算最终结果
            end_time = time.time()
            result = self._calculate_backtest_result(start_time, end_time)
            
            self.backtest_status = BacktestStatus.COMPLETED
            logger.info("回测完成")
            
            return result
            
        except Exception as e:
            self.backtest_status = BacktestStatus.FAILED
            logger.error(f"回测失败: {e}")
            raise
    
    def _update_current_prices(self, timestamp: datetime):
        """更新当前价格"""
        for symbol, data_list in self.market_data.items():
            for data_point in data_list:
                if data_point.timestamp == timestamp:
                    self.current_prices[symbol] = data_point.close
                    break
    
    def _update_positions(self):
        """更新持仓"""
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                position.update_price(self.current_prices[symbol])
    
    async def _process_orders(self):
        """处理订单"""
        for order in self.orders:
            if order.status == OrderStatus.PENDING:
                await self._execute_order(order)
    
    async def _execute_order(self, order: BacktestOrder):
        """执行订单"""
        try:
            if order.symbol not in self.current_prices:
                return
            
            current_price = self.current_prices[order.symbol]
            
            # 计算滑点
            slippage_factor = 1 + (self.current_backtest.slippage_rate if order.side == 'buy' else -self.current_backtest.slippage_rate)
            execution_price = current_price * slippage_factor
            
            # 计算手续费
            commission = order.quantity * execution_price * self.current_backtest.commission_rate
            
            # 检查资金充足性
            if order.side == 'buy':
                required_cash = order.quantity * execution_price + commission
                if required_cash > self.cash:
                    order.status = OrderStatus.REJECTED
                    return
                
                # 扣除现金
                self.cash -= required_cash
                
                # 更新持仓
                if order.symbol in self.positions:
                    # 加仓
                    pos = self.positions[order.symbol]
                    total_quantity = pos.quantity + order.quantity
                    total_cost = pos.quantity * pos.entry_price + order.quantity * execution_price
                    pos.entry_price = total_cost / total_quantity
                    pos.quantity = total_quantity
                else:
                    # 开仓
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        entry_price=execution_price,
                        entry_time=self.current_time,
                        current_price=current_price
                    )
            
            else:  # sell
                if order.symbol not in self.positions:
                    order.status = OrderStatus.REJECTED
                    return
                
                pos = self.positions[order.symbol]
                if pos.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    return
                
                # 计算盈亏
                pnl = (execution_price - pos.entry_price) * order.quantity
                
                # 增加现金
                self.cash += order.quantity * execution_price - commission
                
                # 更新持仓
                pos.quantity -= order.quantity
                pos.realized_pnl += pnl
                
                if pos.quantity == 0:
                    del self.positions[order.symbol]
                
                # 记录交易
                self.trades.append({
                    'timestamp': self.current_time,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price': execution_price,
                    'pnl': pnl,
                    'commission': commission
                })
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.commission = commission
            order.slippage = abs(execution_price - current_price)
            
        except Exception as e:
            logger.error(f"执行订单失败: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _make_trading_decisions(self):
        """AI交易决策"""
        try:
            if not self.ai_evolution_system:
                return
            
            # 构建市场数据
            market_data = {}
            for symbol in self.current_prices:
                market_data[symbol] = {
                    'price': self.current_prices[symbol],
                    'volume': 1000000.0,  # 简化
                    'volatility': 0.02
                }
            
            # 获取AI决策
            for symbol in self.current_backtest.symbols:
                if symbol in market_data:
                    decision = self.ai_evolution_system.get_fusion_decision(market_data[symbol])
                    
                    if decision and decision.get('final_action') in ['buy', 'sell']:
                        await self._place_order_from_decision(symbol, decision)
            
        except Exception as e:
            logger.error(f"AI决策失败: {e}")
    
    async def _place_order_from_decision(self, symbol: str, decision: Dict[str, Any]):
        """根据AI决策下单"""
        try:
            action = decision['final_action']
            confidence = decision.get('confidence', 0.5)
            
            # 计算仓位大小
            position_value = self.portfolio_value * self.current_backtest.position_size_pct * confidence
            current_price = self.current_prices[symbol]
            quantity = position_value / current_price
            
            # 检查最大持仓数限制
            if action == 'buy' and len(self.positions) >= self.current_backtest.max_positions:
                return
            
            # 创建订单
            order = BacktestOrder(
                order_id=f"{symbol}_{self.current_time.timestamp()}_{action}",
                timestamp=self.current_time,
                symbol=symbol,
                side=action,
                quantity=quantity,
                price=current_price
            )
            
            self.orders.append(order)
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
    
    async def _apply_risk_controls(self):
        """应用风险控制"""
        try:
            # 止损止盈检查
            for symbol, position in list(self.positions.items()):
                if symbol not in self.current_prices:
                    continue
                
                current_price = self.current_prices[symbol]
                entry_price = position.entry_price
                
                # 计算收益率
                return_pct = (current_price - entry_price) / entry_price
                
                # 止损
                if return_pct <= -self.current_backtest.stop_loss_pct:
                    await self._close_position(symbol, "stop_loss")
                
                # 止盈
                elif return_pct >= self.current_backtest.take_profit_pct:
                    await self._close_position(symbol, "take_profit")
            
        except Exception as e:
            logger.error(f"风险控制失败: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """平仓"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 创建平仓订单
        order = BacktestOrder(
            order_id=f"{symbol}_{self.current_time.timestamp()}_close_{reason}",
            timestamp=self.current_time,
            symbol=symbol,
            side='sell',
            quantity=position.quantity,
            price=self.current_prices[symbol]
        )
        
        self.orders.append(order)
    
    async def _close_all_positions(self):
        """平仓所有持仓"""
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, "backtest_end")
        
        # 处理剩余订单
        await self._process_orders()
    
    def _record_equity_curve(self):
        """记录权益曲线"""
        # 计算总资产
        total_value = self.cash
        for position in self.positions.values():
            if position.symbol in self.current_prices:
                total_value += position.quantity * self.current_prices[position.symbol]
        
        self.portfolio_value = total_value
        self.equity_curve.append((self.current_time, total_value))
    
    def _calculate_backtest_result(self, start_time: float, end_time: float) -> BacktestResult:
        """计算回测结果"""
        try:
            config = self.current_backtest
            
            # 基础指标
            initial_capital = config.initial_capital
            final_capital = self.portfolio_value
            total_return = final_capital - initial_capital
            total_return_pct = total_return / initial_capital
            
            # 交易统计
            total_trades = len(self.trades)
            winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算日收益率
            daily_returns = self._calculate_daily_returns()
            
            # 风险指标
            volatility = np.std(daily_returns) * np.sqrt(365) if daily_returns else 0
            max_drawdown, max_drawdown_pct = self._calculate_max_drawdown()
            
            # 夏普比率
            risk_free_rate = 0.02  # 假设2%无风险利率
            excess_returns = np.array(daily_returns) - risk_free_rate / 365
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if len(excess_returns) > 0 and np.std(excess_returns) > 0 else 0
            
            # Sortino比率
            downside_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(365) if downside_std > 0 else 0
            
            # Calmar比率
            calmar_ratio = total_return_pct / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
            
            # VaR和CVaR
            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
            cvar_95 = np.mean([r for r in daily_returns if r <= var_95]) if daily_returns else 0
            
            # 基准比较 (简化)
            benchmark_return = 0.1  # 假设基准收益10%
            alpha = total_return_pct - benchmark_return
            beta = 1.0  # 简化
            information_ratio = alpha / volatility if volatility > 0 else 0
            
            return BacktestResult(
                config=config,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                duration_seconds=end_time - start_time,
                
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95,
                
                benchmark_return=benchmark_return,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                
                equity_curve=self.equity_curve,
                trades=self.trades,
                daily_returns=daily_returns,
                monthly_returns=self._calculate_monthly_returns()
            )
            
        except Exception as e:
            logger.error(f"计算回测结果失败: {e}")
            raise
    
    def _calculate_daily_returns(self) -> List[float]:
        """计算日收益率"""
        if len(self.equity_curve) < 2:
            return []
        
        daily_returns = []
        prev_value = self.equity_curve[0][1]
        
        for timestamp, value in self.equity_curve[1:]:
            if prev_value > 0:
                daily_return = (value - prev_value) / prev_value
                daily_returns.append(daily_return)
            prev_value = value
        
        return daily_returns
    
    def _calculate_monthly_returns(self) -> List[float]:
        """计算月收益率"""
        # 简化实现
        daily_returns = self._calculate_daily_returns()
        if not daily_returns:
            return []
        
        # 按30天分组计算月收益
        monthly_returns = []
        for i in range(0, len(daily_returns), 30):
            month_returns = daily_returns[i:i+30]
            if month_returns:
                # 复合收益率
                month_return = np.prod([1 + r for r in month_returns]) - 1
                monthly_returns.append(month_return)
        
        return monthly_returns
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """计算最大回撤"""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        peak = self.equity_curve[0][1]
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        
        for timestamp, value in self.equity_curve:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_pct = drawdown / peak if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        return max_drawdown, max_drawdown_pct
    
    def get_backtest_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """获取回测摘要"""
        return {
            'performance': {
                'total_return': result.total_return,
                'total_return_pct': result.total_return_pct,
                'annualized_return': result.total_return_pct * 365 / (result.end_time - result.start_time).days,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'win_rate': result.win_rate
            },
            'trading': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'avg_trade_return': result.total_return / result.total_trades if result.total_trades > 0 else 0
            },
            'risk': {
                'volatility': result.volatility,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio
            },
            'benchmark': {
                'alpha': result.alpha,
                'beta': result.beta,
                'information_ratio': result.information_ratio
            }
        }


# 全局回测引擎实例
backtest_engine = BacktestEngine()


async def main():
    """测试主函数"""
    logger.info("启动回测引擎测试...")
    
    try:
        # 配置回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            initial_capital=100000.0,
            symbols=["BTC/USDT", "ETH/USDT"],
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # 加载数据
        backtest_engine.load_market_data("mock", config)
        
        # 运行回测
        result = await backtest_engine.run_backtest(config)
        
        # 显示结果
        summary = backtest_engine.get_backtest_summary(result)
        logger.info("回测结果:")
        logger.info(json.dumps(summary, indent=2))
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    except Exception as e:
        logger.error(f"测试出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
