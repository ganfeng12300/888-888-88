#!/usr/bin/env python3
"""
📏 888-888-88 仓位管理系统
生产级仓位大小计算和资金分配模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import math
from loguru import logger


class SizingMethod(Enum):
    """仓位计算方法"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"


@dataclass
class SizingParameters:
    """仓位计算参数"""
    method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED
    base_risk_per_trade: float = 0.02  # 每笔交易风险2%
    max_position_size: float = 0.1     # 最大单仓位10%
    min_position_size: float = 0.001   # 最小仓位0.1%
    volatility_lookback: int = 20      # 波动率回看期
    kelly_lookback: int = 100          # 凯利公式回看期
    risk_free_rate: float = 0.02       # 无风险利率
    confidence_level: float = 0.95     # 置信水平
    max_leverage: float = 3.0          # 最大杠杆


@dataclass
class PositionSize:
    """仓位大小结果"""
    symbol: str
    recommended_size: float
    max_size: float
    min_size: float
    risk_amount: float
    expected_return: float
    confidence_score: float
    method_used: SizingMethod
    parameters: Dict[str, Any]


class PositionSizer:
    """生产级仓位管理器"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.parameters = SizingParameters()
        self.price_history: Dict[str, List[float]] = {}
        self.return_history: Dict[str, List[float]] = {}
        self.trade_history: List[Dict] = []
        
        logger.info("📏 仓位管理系统初始化完成")
    
    def update_balance(self, new_balance: float) -> None:
        """更新账户余额"""
        self.balance = new_balance
        logger.debug(f"💰 余额更新: {new_balance:.2f}")
    
    def add_price_data(self, symbol: str, price: float) -> None:
        """添加价格数据"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.return_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # 计算收益率
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2]
            return_rate = (price - prev_price) / prev_price
            self.return_history[symbol].append(return_rate)
        
        # 保持数据长度
        max_length = max(self.parameters.kelly_lookback, 
                        self.parameters.volatility_lookback) + 50
        
        if len(self.price_history[symbol]) > max_length:
            self.price_history[symbol] = self.price_history[symbol][-max_length:]
            self.return_history[symbol] = self.return_history[symbol][-max_length:]
    
    def calculate_position_size(self, symbol: str, entry_price: float,
                              stop_loss: Optional[float] = None,
                              take_profit: Optional[float] = None,
                              signal_strength: float = 1.0) -> PositionSize:
        """计算推荐仓位大小"""
        
        # 获取历史数据
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            logger.warning(f"⚠️ {symbol} 历史数据不足，使用默认仓位")
            return self._default_position_size(symbol, entry_price)
        
        # 根据方法计算仓位
        if self.parameters.method == SizingMethod.FIXED_AMOUNT:
            size_result = self._fixed_amount_sizing(symbol, entry_price)
        elif self.parameters.method == SizingMethod.FIXED_PERCENTAGE:
            size_result = self._fixed_percentage_sizing(symbol, entry_price)
        elif self.parameters.method == SizingMethod.KELLY_CRITERION:
            size_result = self._kelly_criterion_sizing(symbol, entry_price)
        elif self.parameters.method == SizingMethod.VOLATILITY_ADJUSTED:
            size_result = self._volatility_adjusted_sizing(symbol, entry_price)
        elif self.parameters.method == SizingMethod.RISK_PARITY:
            size_result = self._risk_parity_sizing(symbol, entry_price)
        elif self.parameters.method == SizingMethod.OPTIMAL_F:
            size_result = self._optimal_f_sizing(symbol, entry_price)
        else:
            size_result = self._volatility_adjusted_sizing(symbol, entry_price)
        
        # 应用信号强度调整
        size_result.recommended_size *= signal_strength
        
        # 应用止损调整
        if stop_loss:
            size_result = self._adjust_for_stop_loss(
                size_result, entry_price, stop_loss
            )
        
        # 确保在限制范围内
        size_result = self._apply_size_limits(size_result)
        
        logger.info(f"📊 {symbol} 推荐仓位: {size_result.recommended_size:.4f}")
        return size_result
    
    def _fixed_amount_sizing(self, symbol: str, price: float) -> PositionSize:
        """固定金额仓位"""
        fixed_amount = self.balance * self.parameters.base_risk_per_trade
        size = fixed_amount / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.5,
            risk_amount=fixed_amount,
            expected_return=0.0,
            confidence_score=0.8,
            method_used=SizingMethod.FIXED_AMOUNT,
            parameters={"fixed_amount": fixed_amount}
        )
    
    def _fixed_percentage_sizing(self, symbol: str, price: float) -> PositionSize:
        """固定百分比仓位"""
        position_value = self.balance * self.parameters.base_risk_per_trade
        size = position_value / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.5,
            risk_amount=position_value,
            expected_return=0.0,
            confidence_score=0.8,
            method_used=SizingMethod.FIXED_PERCENTAGE,
            parameters={"percentage": self.parameters.base_risk_per_trade}
        )
    
    def _kelly_criterion_sizing(self, symbol: str, price: float) -> PositionSize:
        """凯利公式仓位计算"""
        if len(self.return_history[symbol]) < self.parameters.kelly_lookback:
            return self._volatility_adjusted_sizing(symbol, price)
        
        returns = np.array(self.return_history[symbol][-self.parameters.kelly_lookback:])
        
        # 计算胜率和平均盈亏
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return self._volatility_adjusted_sizing(symbol, price)
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # 凯利公式: f = (bp - q) / b
        # b = 平均盈利/平均亏损, p = 胜率, q = 败率
        if avg_loss == 0:
            return self._volatility_adjusted_sizing(symbol, price)
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 限制在0-25%
        
        position_value = self.balance * kelly_fraction
        size = position_value / price
        
        # 计算置信度
        confidence = min(0.95, len(returns) / self.parameters.kelly_lookback)
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 1.5,
            min_size=size * 0.5,
            risk_amount=position_value,
            expected_return=avg_win * win_rate - avg_loss * (1 - win_rate),
            confidence_score=confidence,
            method_used=SizingMethod.KELLY_CRITERION,
            parameters={
                "kelly_fraction": kelly_fraction,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            }
        )
    
    def _volatility_adjusted_sizing(self, symbol: str, price: float) -> PositionSize:
        """波动率调整仓位"""
        if len(self.return_history[symbol]) < self.parameters.volatility_lookback:
            return self._fixed_percentage_sizing(symbol, price)
        
        returns = np.array(self.return_history[symbol][-self.parameters.volatility_lookback:])
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 目标波动率为2%
        target_volatility = self.parameters.base_risk_per_trade
        
        if volatility == 0:
            volatility_adjustment = 1.0
        else:
            volatility_adjustment = target_volatility / volatility
        
        # 限制调整范围
        volatility_adjustment = max(0.1, min(volatility_adjustment, 5.0))
        
        base_position_value = self.balance * self.parameters.base_risk_per_trade
        adjusted_position_value = base_position_value * volatility_adjustment
        size = adjusted_position_value / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.3,
            risk_amount=adjusted_position_value,
            expected_return=np.mean(returns) if len(returns) > 0 else 0.0,
            confidence_score=min(0.9, len(returns) / self.parameters.volatility_lookback),
            method_used=SizingMethod.VOLATILITY_ADJUSTED,
            parameters={
                "volatility": volatility,
                "target_volatility": target_volatility,
                "adjustment_factor": volatility_adjustment
            }
        )
    
    def _risk_parity_sizing(self, symbol: str, price: float) -> PositionSize:
        """风险平价仓位"""
        if len(self.return_history[symbol]) < self.parameters.volatility_lookback:
            return self._volatility_adjusted_sizing(symbol, price)
        
        returns = np.array(self.return_history[symbol][-self.parameters.volatility_lookback:])
        volatility = np.std(returns)
        
        if volatility == 0:
            return self._fixed_percentage_sizing(symbol, price)
        
        # 风险平价：每个资产贡献相等的风险
        target_risk = self.parameters.base_risk_per_trade * self.balance
        position_value = target_risk / volatility
        size = position_value / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.5,
            risk_amount=position_value * volatility,
            expected_return=np.mean(returns) if len(returns) > 0 else 0.0,
            confidence_score=0.85,
            method_used=SizingMethod.RISK_PARITY,
            parameters={
                "volatility": volatility,
                "target_risk": target_risk
            }
        )
    
    def _optimal_f_sizing(self, symbol: str, price: float) -> PositionSize:
        """最优f仓位计算"""
        if len(self.trade_history) < 30:
            return self._volatility_adjusted_sizing(symbol, price)
        
        # 获取该品种的交易历史
        symbol_trades = [t for t in self.trade_history if t.get('symbol') == symbol]
        if len(symbol_trades) < 20:
            return self._volatility_adjusted_sizing(symbol, price)
        
        # 计算每笔交易的盈亏
        pnls = [trade.get('pnl', 0) for trade in symbol_trades[-50:]]
        
        # 寻找最优f
        best_f = 0.0
        best_twr = 0.0
        
        for f in np.arange(0.01, 0.5, 0.01):
            twr = 1.0
            for pnl in pnls:
                if pnl < 0:
                    # 最大可能亏损
                    max_loss = abs(pnl)
                    if max_loss * f >= 1.0:
                        twr = 0.0
                        break
                    twr *= (1 - max_loss * f)
                else:
                    twr *= (1 + pnl * f)
            
            if twr > best_twr:
                best_twr = twr
                best_f = f
        
        # 应用保守系数
        optimal_f = best_f * 0.25  # 使用25%的最优f
        
        position_value = self.balance * optimal_f
        size = position_value / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.3,
            risk_amount=position_value,
            expected_return=np.mean(pnls) if pnls else 0.0,
            confidence_score=min(0.9, len(symbol_trades) / 50),
            method_used=SizingMethod.OPTIMAL_F,
            parameters={
                "optimal_f": optimal_f,
                "trade_count": len(symbol_trades),
                "twr": best_twr
            }
        )
    
    def _adjust_for_stop_loss(self, size_result: PositionSize, 
                            entry_price: float, stop_loss: float) -> PositionSize:
        """根据止损调整仓位"""
        # 计算止损距离
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # 目标风险金额
        target_risk = self.balance * self.parameters.base_risk_per_trade
        
        # 根据止损距离调整仓位
        if stop_distance > 0:
            max_size_for_risk = target_risk / (stop_distance * entry_price)
            size_result.recommended_size = min(size_result.recommended_size, max_size_for_risk)
            size_result.risk_amount = size_result.recommended_size * entry_price * stop_distance
        
        return size_result
    
    def _apply_size_limits(self, size_result: PositionSize) -> PositionSize:
        """应用仓位限制"""
        # 最大仓位限制
        max_position_value = self.balance * self.parameters.max_position_size
        max_size = max_position_value / (size_result.recommended_size * 
                                       (self.price_history[size_result.symbol][-1] 
                                        if size_result.symbol in self.price_history 
                                        else 50000))
        
        # 最小仓位限制
        min_position_value = self.balance * self.parameters.min_position_size
        min_size = min_position_value / (size_result.recommended_size * 
                                       (self.price_history[size_result.symbol][-1] 
                                        if size_result.symbol in self.price_history 
                                        else 50000))
        
        # 应用限制
        size_result.recommended_size = max(min_size, 
                                         min(max_size, size_result.recommended_size))
        size_result.max_size = max_size
        size_result.min_size = min_size
        
        return size_result
    
    def _default_position_size(self, symbol: str, price: float) -> PositionSize:
        """默认仓位大小"""
        position_value = self.balance * self.parameters.base_risk_per_trade
        size = position_value / price
        
        return PositionSize(
            symbol=symbol,
            recommended_size=size,
            max_size=size * 2,
            min_size=size * 0.5,
            risk_amount=position_value,
            expected_return=0.0,
            confidence_score=0.5,
            method_used=SizingMethod.FIXED_PERCENTAGE,
            parameters={"default": True}
        )
    
    def update_trade_history(self, trade: Dict[str, Any]) -> None:
        """更新交易历史"""
        self.trade_history.append(trade)
        
        # 保持历史长度
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def get_portfolio_allocation(self, symbols: List[str], 
                               prices: Dict[str, float]) -> Dict[str, PositionSize]:
        """获取投资组合分配"""
        allocations = {}
        
        for symbol in symbols:
            if symbol in prices:
                allocation = self.calculate_position_size(symbol, prices[symbol])
                allocations[symbol] = allocation
        
        # 标准化分配（确保总和不超过100%）
        total_allocation = sum(alloc.recommended_size * prices[symbol] 
                             for symbol, alloc in allocations.items() 
                             if symbol in prices)
        
        if total_allocation > self.balance * 0.9:  # 最多90%资金
            scale_factor = (self.balance * 0.9) / total_allocation
            for allocation in allocations.values():
                allocation.recommended_size *= scale_factor
        
        return allocations
    
    def get_sizing_report(self) -> Dict[str, Any]:
        """获取仓位管理报告"""
        return {
            "balance": self.balance,
            "parameters": {
                "method": self.parameters.method.value,
                "base_risk_per_trade": self.parameters.base_risk_per_trade,
                "max_position_size": self.parameters.max_position_size,
                "min_position_size": self.parameters.min_position_size
            },
            "data_coverage": {
                symbol: len(prices) for symbol, prices in self.price_history.items()
            },
            "trade_history_count": len(self.trade_history)
        }


# 全局仓位管理器实例
position_sizer = None

def get_position_sizer(initial_balance: float = 100000.0) -> PositionSizer:
    """获取仓位管理器实例"""
    global position_sizer
    if position_sizer is None:
        position_sizer = PositionSizer(initial_balance)
    return position_sizer


if __name__ == "__main__":
    # 测试仓位管理器
    sizer = PositionSizer(100000.0)
    
    # 添加模拟价格数据
    prices = [50000, 50500, 49800, 51200, 50800, 52000, 51500, 50900]
    for price in prices:
        sizer.add_price_data("BTC/USDT", price)
    
    # 计算仓位
    position = sizer.calculate_position_size("BTC/USDT", 51000.0)
    print(f"推荐仓位: {position.recommended_size:.4f}")
    print(f"风险金额: {position.risk_amount:.2f}")
    print(f"置信度: {position.confidence_score:.2f}")
    
    # 生成报告
    report = sizer.get_sizing_report()
    print(f"仓位管理报告: {report}")

