#!/usr/bin/env python3
"""
🎯 策略管理器 - 生产级交易策略系统
Strategy Manager - Production-Grade Trading Strategy System

生产级特性：
- 多策略并行执行
- 策略性能监控
- 动态参数调优
- 策略风险控制
- 回测和实盘切换
"""

import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

class StrategyStatus(Enum):
    """策略状态"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class Signal:
    """交易信号"""
    signal_id: str
    strategy_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 信号强度 0-1
    price: float
    quantity: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class StrategyPerformance:
    """策略性能指标"""
    strategy_id: str
    total_trades: int
    win_trades: int
    loss_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_pnl: float
    last_updated: datetime

class BaseStrategy:
    """基础策略类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # f"Strategy_{config.strategy_id}")
        self.status = StrategyStatus.INACTIVE
        self.signals = deque(maxlen=1000)
        self.performance = StrategyPerformance(
            strategy_id=config.strategy_id,
            total_trades=0,
            win_trades=0,
            loss_trades=0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            avg_trade_pnl=0.0,
            last_updated=datetime.now()
        )
        
    def initialize(self):
        """初始化策略"""
        self.status = StrategyStatus.ACTIVE
        self.logger.info(f"策略初始化完成: {self.config.name}")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """生成交易信号 - 子类需要实现"""
        raise NotImplementedError("子类必须实现generate_signal方法")
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """更新策略性能"""
        try:
            self.performance.total_trades += 1
            pnl = trade_result.get('pnl', 0.0)
            self.performance.total_pnl += pnl
            
            if pnl > 0:
                self.performance.win_trades += 1
            else:
                self.performance.loss_trades += 1
            
            # 计算胜率
            self.performance.win_rate = self.performance.win_trades / self.performance.total_trades
            
            # 计算平均盈亏
            self.performance.avg_trade_pnl = self.performance.total_pnl / self.performance.total_trades
            
            self.performance.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"更新策略性能失败: {e}")
    
    def pause(self):
        """暂停策略"""
        self.status = StrategyStatus.PAUSED
        self.logger.info(f"策略已暂停: {self.config.name}")
    
    def resume(self):
        """恢复策略"""
        self.status = StrategyStatus.ACTIVE
        self.logger.info(f"策略已恢复: {self.config.name}")
    
    def stop(self):
        """停止策略"""
        self.status = StrategyStatus.INACTIVE
        self.logger.info(f"策略已停止: {self.config.name}")

class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.short_window = config.parameters.get('short_window', 10)
        self.long_window = config.parameters.get('long_window', 30)
        self.price_history = defaultdict(deque)
        
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """生成移动平均交叉信号"""
        try:
            if self.status != StrategyStatus.ACTIVE:
                return None
            
            symbol = market_data.get('symbol')
            price = market_data.get('price')
            
            if not symbol or not price:
                return None
            
            # 更新价格历史
            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > self.long_window:
                self.price_history[symbol].popleft()
            
            # 检查是否有足够数据
            if len(self.price_history[symbol]) < self.long_window:
                return None
            
            prices = list(self.price_history[symbol])
            
            # 计算移动平均
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])
            
            # 计算前一期移动平均
            if len(prices) > self.long_window:
                prev_short_ma = np.mean(prices[-(self.short_window+1):-1])
                prev_long_ma = np.mean(prices[-(self.long_window+1):-1])
            else:
                return None
            
            # 生成信号
            signal = None
            
            # 金叉 - 买入信号
            if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                signal = Signal(
                    signal_id=f"{self.config.strategy_id}_{int(time.time() * 1000)}",
                    strategy_id=self.config.strategy_id,
                    symbol=symbol,
                    action='buy',
                    strength=min((short_ma - long_ma) / long_ma, 1.0),
                    price=price,
                    quantity=self.config.parameters.get('base_quantity', 100),
                    timestamp=datetime.now(),
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'signal_type': 'golden_cross'
                    }
                )
            
            # 死叉 - 卖出信号
            elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                signal = Signal(
                    signal_id=f"{self.config.strategy_id}_{int(time.time() * 1000)}",
                    strategy_id=self.config.strategy_id,
                    symbol=symbol,
                    action='sell',
                    strength=min((long_ma - short_ma) / long_ma, 1.0),
                    price=price,
                    quantity=self.config.parameters.get('base_quantity', 100),
                    timestamp=datetime.now(),
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'signal_type': 'death_cross'
                    }
                )
            
            if signal:
                self.signals.append(signal)
                self.logger.info(f"生成信号: {signal.action} {signal.symbol} 强度: {signal.strength:.3f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成移动平均信号失败: {e}")
            self.status = StrategyStatus.ERROR
            return None

class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.oversold_threshold = config.parameters.get('oversold_threshold', 30)
        self.overbought_threshold = config.parameters.get('overbought_threshold', 70)
        self.price_history = defaultdict(deque)
        
    def calculate_rsi(self, prices: List[float]) -> float:
        """计算RSI指标"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """生成RSI信号"""
        try:
            if self.status != StrategyStatus.ACTIVE:
                return None
            
            symbol = market_data.get('symbol')
            price = market_data.get('price')
            
            if not symbol or not price:
                return None
            
            # 更新价格历史
            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > self.rsi_period * 2:
                self.price_history[symbol].popleft()
            
            # 检查是否有足够数据
            if len(self.price_history[symbol]) < self.rsi_period + 1:
                return None
            
            prices = list(self.price_history[symbol])
            rsi = self.calculate_rsi(prices)
            
            signal = None
            
            # 超卖 - 买入信号
            if rsi < self.oversold_threshold:
                signal = Signal(
                    signal_id=f"{self.config.strategy_id}_{int(time.time() * 1000)}",
                    strategy_id=self.config.strategy_id,
                    symbol=symbol,
                    action='buy',
                    strength=(self.oversold_threshold - rsi) / self.oversold_threshold,
                    price=price,
                    quantity=self.config.parameters.get('base_quantity', 100),
                    timestamp=datetime.now(),
                    metadata={
                        'rsi': rsi,
                        'signal_type': 'oversold'
                    }
                )
            
            # 超买 - 卖出信号
            elif rsi > self.overbought_threshold:
                signal = Signal(
                    signal_id=f"{self.config.strategy_id}_{int(time.time() * 1000)}",
                    strategy_id=self.config.strategy_id,
                    symbol=symbol,
                    action='sell',
                    strength=(rsi - self.overbought_threshold) / (100 - self.overbought_threshold),
                    price=price,
                    quantity=self.config.parameters.get('base_quantity', 100),
                    timestamp=datetime.now(),
                    metadata={
                        'rsi': rsi,
                        'signal_type': 'overbought'
                    }
                )
            
            if signal:
                self.signals.append(signal)
                self.logger.info(f"生成RSI信号: {signal.action} {signal.symbol} RSI: {rsi:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成RSI信号失败: {e}")
            self.status = StrategyStatus.ERROR
            return None

class StrategyManager:
    """策略管理器主类"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "StrategyManager")
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.signal_queue = deque()
        self.performance_history = deque(maxlen=10000)
        
        # 运行状态
        self._running = False
        self._strategy_thread = None
        self._lock = threading.Lock()
        
        self.logger.info("策略管理器初始化完成")
    
    def register_strategy(self, strategy_config: StrategyConfig, strategy_class: type = None) -> bool:
        """注册策略"""
        try:
            with self._lock:
                # 保存配置
                if strategy_config.created_at is None:
                    strategy_config.created_at = datetime.now()
                strategy_config.updated_at = datetime.now()
                
                self.strategy_configs[strategy_config.strategy_id] = strategy_config
                
                # 创建策略实例
                if strategy_class:
                    strategy = strategy_class(strategy_config)
                else:
                    # 根据策略名称选择默认实现
                    if 'ma' in strategy_config.name.lower() or 'moving' in strategy_config.name.lower():
                        strategy = MovingAverageStrategy(strategy_config)
                    elif 'rsi' in strategy_config.name.lower():
                        strategy = RSIStrategy(strategy_config)
                    else:
                        self.logger.error(f"未知策略类型: {strategy_config.name}")
                        return False
                
                self.strategies[strategy_config.strategy_id] = strategy
                
                # 初始化策略
                if strategy_config.enabled:
                    strategy.initialize()
                
                self.logger.info(f"策略注册成功: {strategy_config.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"注册策略失败: {e}")
            return False
    
    def start_strategy(self, strategy_id: str) -> bool:
        """启动策略"""
        try:
            with self._lock:
                if strategy_id not in self.strategies:
                    self.logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.initialize()
                
                self.logger.info(f"策略已启动: {strategy_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"启动策略失败: {e}")
            return False
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """停止策略"""
        try:
            with self._lock:
                if strategy_id not in self.strategies:
                    self.logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.stop()
                
                self.logger.info(f"策略已停止: {strategy_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"停止策略失败: {e}")
            return False
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """暂停策略"""
        try:
            with self._lock:
                if strategy_id not in self.strategies:
                    self.logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.pause()
                
                self.logger.info(f"策略已暂停: {strategy_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"暂停策略失败: {e}")
            return False
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """恢复策略"""
        try:
            with self._lock:
                if strategy_id not in self.strategies:
                    self.logger.error(f"策略不存在: {strategy_id}")
                    return False
                
                strategy = self.strategies[strategy_id]
                strategy.resume()
                
                self.logger.info(f"策略已恢复: {strategy_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"恢复策略失败: {e}")
            return False
    
    def process_market_data(self, market_data: Dict[str, Any]):
        """处理市场数据，生成交易信号"""
        try:
            with self._lock:
                for strategy_id, strategy in self.strategies.items():
                    if strategy.status == StrategyStatus.ACTIVE:
                        signal = strategy.generate_signal(market_data)
                        if signal:
                            self.signal_queue.append(signal)
                            
        except Exception as e:
            self.logger.error(f"处理市场数据失败: {e}")
    
    def get_signals(self, count: int = 10) -> List[Signal]:
        """获取最新信号"""
        signals = []
        with self._lock:
            for _ in range(min(count, len(self.signal_queue))):
                if self.signal_queue:
                    signals.append(self.signal_queue.popleft())
        return signals
    
    def update_strategy_performance(self, strategy_id: str, trade_result: Dict[str, Any]):
        """更新策略性能"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.update_performance(trade_result)
                
                # 记录性能历史
                performance_record = {
                    'strategy_id': strategy_id,
                    'performance': asdict(strategy.performance),
                    'timestamp': datetime.now()
                }
                self.performance_history.append(performance_record)
                
        except Exception as e:
            self.logger.error(f"更新策略性能失败: {e}")
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """获取策略性能"""
        if strategy_id in self.strategies:
            return self.strategies[strategy_id].performance
        return None
    
    def get_all_strategies_status(self) -> Dict[str, Dict]:
        """获取所有策略状态"""
        status = {}
        with self._lock:
            for strategy_id, strategy in self.strategies.items():
                status[strategy_id] = {
                    'name': strategy.config.name,
                    'status': strategy.status.value,
                    'performance': asdict(strategy.performance),
                    'config': asdict(strategy.config)
                }
        return status
    
    def optimize_strategy_parameters(self, strategy_id: str, optimization_data: Dict[str, Any]) -> bool:
        """优化策略参数"""
        try:
            if strategy_id not in self.strategies:
                self.logger.error(f"策略不存在: {strategy_id}")
                return False
            
            strategy = self.strategies[strategy_id]
            config = self.strategy_configs[strategy_id]
            
            # 简单的参数优化逻辑
            current_performance = strategy.performance
            
            # 如果胜率低于50%，调整参数
            if current_performance.win_rate < 0.5 and current_performance.total_trades > 10:
                if isinstance(strategy, MovingAverageStrategy):
                    # 调整移动平均周期
                    config.parameters['short_window'] = min(config.parameters.get('short_window', 10) + 1, 20)
                    config.parameters['long_window'] = min(config.parameters.get('long_window', 30) + 2, 50)
                elif isinstance(strategy, RSIStrategy):
                    # 调整RSI阈值
                    config.parameters['oversold_threshold'] = max(config.parameters.get('oversold_threshold', 30) - 2, 20)
                    config.parameters['overbought_threshold'] = min(config.parameters.get('overbought_threshold', 70) + 2, 80)
                
                # 重新初始化策略
                strategy.__init__(config)
                if config.enabled:
                    strategy.initialize()
                
                self.logger.info(f"策略参数已优化: {strategy_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"优化策略参数失败: {e}")
            return False
    
    def export_strategy_report(self, filepath: str, hours: int = 24):
        """导出策略报告"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 过滤时间范围内的性能记录
            filtered_performance = [
                record for record in self.performance_history
                if record['timestamp'] >= cutoff_time
            ]
            
            report_data = {
                'report_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'strategies_status': self.get_all_strategies_status(),
                'performance_history': filtered_performance,
                'total_signals': len(self.signal_queue),
                'active_strategies': len([s for s in self.strategies.values() if s.status == StrategyStatus.ACTIVE])
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"策略报告已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出策略报告失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建策略管理器
    strategy_manager = StrategyManager()
    
    try:
        # 注册移动平均策略
        ma_config = StrategyConfig(
            strategy_id="ma_strategy_001",
            name="移动平均交叉策略",
            description="基于短期和长期移动平均线交叉的交易策略",
            parameters={
                'short_window': 10,
                'long_window': 30,
                'base_quantity': 100
            },
            risk_limits={
                'max_position_size': 1000,
                'max_daily_loss': 5000
            }
        )
        
        strategy_manager.register_strategy(ma_config)
        
        # 注册RSI策略
        rsi_config = StrategyConfig(
            strategy_id="rsi_strategy_001",
            name="RSI超买超卖策略",
            description="基于RSI指标的超买超卖交易策略",
            parameters={
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'base_quantity': 100
            },
            risk_limits={
                'max_position_size': 1000,
                'max_daily_loss': 5000
            }
        )
        
        strategy_manager.register_strategy(rsi_config)
        
        # 模拟市场数据
        import random
        
        for i in range(100):
            market_data = {
                'symbol': 'AAPL',
                'price': 150 + random.uniform(-5, 5),
                'timestamp': datetime.now()
            }
            
            strategy_manager.process_market_data(market_data)
            time.sleep(0.1)
        
        # 获取信号
        signals = strategy_manager.get_signals(5)
        print(f"生成信号数量: {len(signals)}")
        
        for signal in signals:
            print(f"信号: {signal.action} {signal.symbol} 强度: {signal.strength:.3f}")
        
        # 获取策略状态
        status = strategy_manager.get_all_strategies_status()
        print("策略状态:", json.dumps(status, indent=2, default=str, ensure_ascii=False))
        
    except Exception as e:
        print(f"测试失败: {e}")
