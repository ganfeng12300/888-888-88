"""
🎯 策略管理器 - 生产级实盘交易多策略管理和调度系统
提供策略注册、启动停止、性能监控、资源分配等全方位策略管理功能
支持多策略并行运行、动态调整、风险控制和性能优化
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

from .advanced_strategy_engine import (
    BaseStrategy, TradingSignal, StrategyPerformance, StrategyStatus,
    GridTradingStrategy, TrendFollowingStrategy, StrategyType
)

class StrategyPriority(Enum):
    """策略优先级"""
    LOW = "low"  # 低优先级
    MEDIUM = "medium"  # 中等优先级
    HIGH = "high"  # 高优先级
    CRITICAL = "critical"  # 关键优先级

@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str  # 策略ID
    strategy_type: StrategyType  # 策略类型
    symbol: str  # 交易对
    parameters: Dict[str, Any]  # 策略参数
    priority: StrategyPriority = StrategyPriority.MEDIUM  # 优先级
    max_allocation: float = 0.1  # 最大资金分配比例
    is_enabled: bool = True  # 是否启用
    created_at: float = field(default_factory=time.time)  # 创建时间

@dataclass
class StrategyAllocation:
    """策略资金分配"""
    strategy_id: str  # 策略ID
    allocated_amount: float  # 分配金额
    used_amount: float  # 已使用金额
    available_amount: float  # 可用金额
    allocation_ratio: float  # 分配比例
    last_updated: float = field(default_factory=time.time)  # 最后更新时间

class StrategyFactory:
    """策略工厂"""
    
    @staticmethod
    def create_strategy(config: StrategyConfig) -> Optional[BaseStrategy]:
        """创建策略实例"""
        try:
            if config.strategy_type == StrategyType.GRID_TRADING:
                return GridTradingStrategy(
                    config.strategy_id,
                    config.symbol,
                    config.parameters
                )
            elif config.strategy_type == StrategyType.TREND_FOLLOWING:
                return TrendFollowingStrategy(
                    config.strategy_id,
                    config.symbol,
                    config.parameters
                )
            else:
                logger.error(f"不支持的策略类型: {config.strategy_type}")
                return None
        
        except Exception as e:
            logger.error(f"创建策略失败: {e}")
            return None

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        self.benchmark_returns: List[float] = []
        
        logger.info("性能跟踪器初始化完成")
    
    def update_performance(self, strategy_id: str, performance: StrategyPerformance):
        """更新策略性能"""
        try:
            if strategy_id not in self.performance_history:
                self.performance_history[strategy_id] = []
            
            self.performance_history[strategy_id].append(performance)
            
            # 保持历史记录在合理范围内
            if len(self.performance_history[strategy_id]) > 1000:
                self.performance_history[strategy_id] = self.performance_history[strategy_id][-500:]
            
            logger.debug(f"更新策略性能: {strategy_id}")
        
        except Exception as e:
            logger.error(f"更新策略性能失败: {e}")
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """获取策略排名"""
        try:
            rankings = []
            
            for strategy_id, performances in self.performance_history.items():
                if performances:
                    latest_performance = performances[-1]
                    # 综合评分：夏普比率 * 胜率 * (1 - 最大回撤)
                    score = (latest_performance.sharpe_ratio * 
                            latest_performance.win_rate * 
                            (1 - latest_performance.max_drawdown))
                    rankings.append((strategy_id, score))
            
            # 按评分降序排列
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
        
        except Exception as e:
            logger.error(f"获取策略排名失败: {e}")
            return []
    
    def calculate_portfolio_performance(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """计算组合性能"""
        try:
            if not self.performance_history:
                return {}
            
            total_pnl = 0.0
            total_trades = 0
            winning_trades = 0
            max_drawdown = 0.0
            
            for strategy_id, allocation in allocations.items():
                if strategy_id in self.performance_history and self.performance_history[strategy_id]:
                    performance = self.performance_history[strategy_id][-1]
                    
                    # 按分配比例加权
                    total_pnl += performance.total_pnl * allocation
                    total_trades += performance.total_trades
                    winning_trades += performance.winning_trades
                    max_drawdown = max(max_drawdown, performance.max_drawdown)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown
            }
        
        except Exception as e:
            logger.error(f"计算组合性能失败: {e}")
            return {}

class ResourceManager:
    """资源管理器"""
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.reserved_capital = 0.1  # 保留10%资金
        
        logger.info(f"资源管理器初始化完成: 总资金 {total_capital}")
    
    def allocate_capital(self, strategy_id: str, allocation_ratio: float) -> bool:
        """分配资金"""
        try:
            if allocation_ratio <= 0 or allocation_ratio > 1:
                logger.error(f"无效的分配比例: {allocation_ratio}")
                return False
            
            # 检查可用资金
            used_ratio = sum(alloc.allocation_ratio for alloc in self.allocations.values())
            available_ratio = 1.0 - self.reserved_capital - used_ratio
            
            if allocation_ratio > available_ratio:
                logger.error(f"资金不足: 需要 {allocation_ratio:.2%}, 可用 {available_ratio:.2%}")
                return False
            
            # 分配资金
            allocated_amount = self.total_capital * allocation_ratio
            
            self.allocations[strategy_id] = StrategyAllocation(
                strategy_id=strategy_id,
                allocated_amount=allocated_amount,
                used_amount=0.0,
                available_amount=allocated_amount,
                allocation_ratio=allocation_ratio
            )
            
            logger.info(f"资金分配成功: {strategy_id} - {allocated_amount:.2f} ({allocation_ratio:.2%})")
            return True
        
        except Exception as e:
            logger.error(f"分配资金失败: {e}")
            return False
    
    def update_usage(self, strategy_id: str, used_amount: float):
        """更新资金使用"""
        try:
            if strategy_id in self.allocations:
                allocation = self.allocations[strategy_id]
                allocation.used_amount = used_amount
                allocation.available_amount = allocation.allocated_amount - used_amount
                allocation.last_updated = time.time()
                
                logger.debug(f"更新资金使用: {strategy_id} - 使用 {used_amount:.2f}")
        
        except Exception as e:
            logger.error(f"更新资金使用失败: {e}")
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配摘要"""
        try:
            total_allocated = sum(alloc.allocated_amount for alloc in self.allocations.values())
            total_used = sum(alloc.used_amount for alloc in self.allocations.values())
            
            return {
                'total_capital': self.total_capital,
                'total_allocated': total_allocated,
                'total_used': total_used,
                'available_capital': self.total_capital - total_allocated,
                'utilization_rate': total_used / total_allocated if total_allocated > 0 else 0,
                'allocations': {
                    strategy_id: {
                        'allocated': alloc.allocated_amount,
                        'used': alloc.used_amount,
                        'available': alloc.available_amount,
                        'ratio': alloc.allocation_ratio
                    }
                    for strategy_id, alloc in self.allocations.items()
                }
            }
        
        except Exception as e:
            logger.error(f"获取分配摘要失败: {e}")
            return {}

class StrategyManager:
    """策略管理器主类"""
    
    def __init__(self, total_capital: float):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.performance_tracker = PerformanceTracker()
        self.resource_manager = ResourceManager(total_capital)
        
        # 信号处理
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        
        # 运行状态
        self.is_running = False
        self.update_interval = 60  # 更新间隔（秒）
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("策略管理器初始化完成")
    
    def register_strategy(self, config: StrategyConfig) -> bool:
        """注册策略"""
        try:
            with self.lock:
                # 检查策略ID是否已存在
                if config.strategy_id in self.strategy_configs:
                    logger.error(f"策略ID已存在: {config.strategy_id}")
                    return False
                
                # 创建策略实例
                strategy = StrategyFactory.create_strategy(config)
                if not strategy:
                    return False
                
                # 分配资金
                if not self.resource_manager.allocate_capital(
                    config.strategy_id, config.max_allocation):
                    return False
                
                # 注册策略
                self.strategies[config.strategy_id] = strategy
                self.strategy_configs[config.strategy_id] = config
                
                logger.info(f"策略注册成功: {config.strategy_id}")
                return True
        
        except Exception as e:
            logger.error(f"注册策略失败: {e}")
            return False
    
    def start_strategy(self, strategy_id: str) -> bool:
        """启动策略"""
        try:
            with self.lock:
                if strategy_id not in self.strategies:
                    logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                config = self.strategy_configs[strategy_id]
                if not config.is_enabled:
                    logger.error(f"策略未启用: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.start()
                
                logger.info(f"策略启动成功: {strategy_id}")
                return True
        
        except Exception as e:
            logger.error(f"启动策略失败: {e}")
            return False
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """停止策略"""
        try:
            with self.lock:
                if strategy_id not in self.strategies:
                    logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.stop()
                
                logger.info(f"策略停止成功: {strategy_id}")
                return True
        
        except Exception as e:
            logger.error(f"停止策略失败: {e}")
            return False
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """暂停策略"""
        try:
            with self.lock:
                if strategy_id not in self.strategies:
                    logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.pause()
                
                logger.info(f"策略暂停成功: {strategy_id}")
                return True
        
        except Exception as e:
            logger.error(f"暂停策略失败: {e}")
            return False
    
    def update_strategy_parameters(self, strategy_id: str, new_parameters: Dict[str, Any]) -> bool:
        """更新策略参数"""
        try:
            with self.lock:
                if strategy_id not in self.strategies:
                    logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.update_parameters(new_parameters)
                
                # 更新配置
                self.strategy_configs[strategy_id].parameters.update(new_parameters)
                
                logger.info(f"策略参数更新成功: {strategy_id}")
                return True
        
        except Exception as e:
            logger.error(f"更新策略参数失败: {e}")
            return False
    
    async def process_market_data(self, symbol: str, market_data: pd.DataFrame):
        """处理市场数据"""
        try:
            signals = []
            
            with self.lock:
                # 为相关策略生成信号
                for strategy_id, strategy in self.strategies.items():
                    if (strategy.symbol == symbol and 
                        strategy.status == StrategyStatus.ACTIVE):
                        
                        signal = await strategy.generate_signal(market_data)
                        if signal:
                            signals.append(signal)
            
            # 处理信号
            for signal in signals:
                await self.signal_queue.put(signal)
                
                # 调用回调函数
                for callback in self.signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"信号回调函数执行失败: {e}")
        
        except Exception as e:
            logger.error(f"处理市场数据失败: {e}")
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """添加信号回调函数"""
        self.signal_callbacks.append(callback)
    
    async def start_manager(self):
        """启动管理器"""
        try:
            self.is_running = True
            logger.info("策略管理器启动")
            
            # 启动性能更新任务
            asyncio.create_task(self._performance_update_loop())
            
            # 启动信号处理任务
            asyncio.create_task(self._signal_processing_loop())
        
        except Exception as e:
            logger.error(f"启动策略管理器失败: {e}")
    
    def stop_manager(self):
        """停止管理器"""
        try:
            self.is_running = False
            
            # 停止所有策略
            with self.lock:
                for strategy_id in list(self.strategies.keys()):
                    self.stop_strategy(strategy_id)
            
            logger.info("策略管理器停止")
        
        except Exception as e:
            logger.error(f"停止策略管理器失败: {e}")
    
    async def _performance_update_loop(self):
        """性能更新循环"""
        while self.is_running:
            try:
                with self.lock:
                    for strategy_id, strategy in self.strategies.items():
                        # 计算策略性能
                        performance = strategy.calculate_performance()
                        self.performance_tracker.update_performance(strategy_id, performance)
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"性能更新循环失败: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _signal_processing_loop(self):
        """信号处理循环"""
        while self.is_running:
            try:
                # 从队列获取信号
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                # 处理信号（这里可以添加信号过滤、合并等逻辑）
                logger.info(f"处理交易信号: {signal.strategy_id} - {signal.signal_type.value}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"信号处理循环失败: {e}")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        try:
            with self.lock:
                strategy_statuses = {}
                for strategy_id, strategy in self.strategies.items():
                    strategy_statuses[strategy_id] = {
                        'status': strategy.status.value,
                        'symbol': strategy.symbol,
                        'signals_count': len(strategy.signals_history),
                        'trades_count': len(strategy.trades_history)
                    }
                
                return {
                    'is_running': self.is_running,
                    'total_strategies': len(self.strategies),
                    'active_strategies': sum(1 for s in self.strategies.values() 
                                           if s.status == StrategyStatus.ACTIVE),
                    'strategy_statuses': strategy_statuses,
                    'resource_summary': self.resource_manager.get_allocation_summary(),
                    'strategy_rankings': self.performance_tracker.get_strategy_ranking()
                }
        
        except Exception as e:
            logger.error(f"获取管理器状态失败: {e}")
            return {}
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """获取策略性能"""
        try:
            if strategy_id in self.strategies:
                return self.strategies[strategy_id].calculate_performance()
            return None
        
        except Exception as e:
            logger.error(f"获取策略性能失败: {e}")
            return None

# 全局策略管理器实例（需要在使用前初始化）
strategy_manager = None

def initialize_strategy_manager(total_capital: float):
    """初始化策略管理器"""
    global strategy_manager
    strategy_manager = StrategyManager(total_capital)
    return strategy_manager
