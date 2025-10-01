"""
📊 仓位管理系统
生产级仓位管理系统，支持多交易所、多品种、动态杠杆
实现完整的仓位跟踪、风险控制、盈亏计算等功能
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from loguru import logger

from src.core.config import settings
from src.system.message_bus import message_bus, Message, MessageType, MessagePriority
from src.trading.order_management_system import Fill, OrderSide


class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """仓位对象"""
    symbol: str
    exchange: str
    side: PositionSide
    size: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin: Decimal
    leverage: int
    liquidation_price: Optional[Decimal]
    created_time: datetime
    updated_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> Decimal:
        """名义价值"""
        return abs(self.size) * self.mark_price
    
    @property
    def margin_ratio(self) -> Decimal:
        """保证金比例"""
        if self.notional_value == 0:
            return Decimal('0')
        return self.margin / self.notional_value
    
    @property
    def pnl_percentage(self) -> Decimal:
        """盈亏百分比"""
        if self.margin == 0:
            return Decimal('0')
        return (self.unrealized_pnl / self.margin) * 100


@dataclass
class PositionRisk:
    """仓位风险"""
    symbol: str
    position_value: Decimal
    margin_used: Decimal
    margin_ratio: Decimal
    leverage: int
    liquidation_distance: Decimal
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    max_loss: Decimal
    var_1d: Decimal  # 1日风险价值
    recommendations: List[str] = field(default_factory=list)


class PositionManager:
    """仓位管理器"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}  # key: f"{exchange}:{symbol}"
        self.position_history: List[Dict[str, Any]] = []
        self.total_margin: Decimal = Decimal('0')
        self.total_unrealized_pnl: Decimal = Decimal('0')
        self.total_realized_pnl: Decimal = Decimal('0')
        
        # 风险参数
        self.max_position_size_ratio = Decimal('0.1')  # 单个仓位最大占比10%
        self.max_leverage = 20
        self.liquidation_buffer = Decimal('0.05')  # 5%缓冲
        
        # 统计信息
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_volume': Decimal('0'),
            'total_commission': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'sharpe_ratio': Decimal('0')
        }
        
        logger.info("仓位管理器初始化完成")
    
    def get_position_key(self, exchange: str, symbol: str) -> str:
        """获取仓位键"""
        return f"{exchange}:{symbol}"
    
    async def update_position(self, fill: Fill):
        """更新仓位"""
        try:
            position_key = self.get_position_key(fill.metadata.get('exchange', 'bitget'), fill.symbol)
            
            # 获取或创建仓位
            position = self.positions.get(position_key)
            if not position:
                position = Position(
                    symbol=fill.symbol,
                    exchange=fill.metadata.get('exchange', 'bitget'),
                    side=PositionSide.FLAT,
                    size=Decimal('0'),
                    entry_price=Decimal('0'),
                    mark_price=fill.price,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    margin=Decimal('0'),
                    leverage=fill.metadata.get('leverage', 1),
                    liquidation_price=None,
                    created_time=datetime.now(),
                    updated_time=datetime.now()
                )
                self.positions[position_key] = position
            
            # 计算新仓位
            old_size = position.size
            fill_size = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            new_size = old_size + fill_size
            
            # 更新仓位信息
            if new_size == 0:
                # 平仓
                if old_size != 0:
                    # 计算已实现盈亏
                    if old_size > 0:  # 原来是多头
                        realized_pnl = (fill.price - position.entry_price) * abs(fill_size)
                    else:  # 原来是空头
                        realized_pnl = (position.entry_price - fill.price) * abs(fill_size)
                    
                    position.realized_pnl += realized_pnl
                    self.total_realized_pnl += realized_pnl
                    
                    # 更新统计
                    self.stats['total_trades'] += 1
                    if realized_pnl > 0:
                        self.stats['winning_trades'] += 1
                    else:
                        self.stats['losing_trades'] += 1
                
                position.side = PositionSide.FLAT
                position.size = Decimal('0')
                position.entry_price = Decimal('0')
                position.unrealized_pnl = Decimal('0')
                position.margin = Decimal('0')
                
            elif (old_size >= 0 and new_size > 0) or (old_size <= 0 and new_size < 0):
                # 加仓
                if old_size == 0:
                    # 开仓
                    position.entry_price = fill.price
                else:
                    # 加仓，计算平均成本
                    total_cost = position.entry_price * abs(old_size) + fill.price * abs(fill_size)
                    position.entry_price = total_cost / abs(new_size)
                
                position.size = new_size
                position.side = PositionSide.LONG if new_size > 0 else PositionSide.SHORT
                
            else:
                # 减仓或反向开仓
                if abs(new_size) < abs(old_size):
                    # 减仓
                    close_size = abs(fill_size)
                    if old_size > 0:  # 原来是多头
                        realized_pnl = (fill.price - position.entry_price) * close_size
                    else:  # 原来是空头
                        realized_pnl = (position.entry_price - fill.price) * close_size
                    
                    position.realized_pnl += realized_pnl
                    self.total_realized_pnl += realized_pnl
                    position.size = new_size
                    
                else:
                    # 反向开仓
                    # 先平掉原仓位
                    if old_size > 0:  # 原来是多头
                        realized_pnl = (fill.price - position.entry_price) * abs(old_size)
                    else:  # 原来是空头
                        realized_pnl = (position.entry_price - fill.price) * abs(old_size)
                    
                    position.realized_pnl += realized_pnl
                    self.total_realized_pnl += realized_pnl
                    
                    # 更新统计
                    self.stats['total_trades'] += 1
                    if realized_pnl > 0:
                        self.stats['winning_trades'] += 1
                    else:
                        self.stats['losing_trades'] += 1
                    
                    # 开新仓
                    position.size = new_size
                    position.side = PositionSide.LONG if new_size > 0 else PositionSide.SHORT
                    position.entry_price = fill.price
            
            # 更新保证金
            if position.size != 0:
                position.margin = abs(position.size) * position.entry_price / position.leverage
            
            # 更新时间和标记价格
            position.updated_time = datetime.now()
            position.mark_price = fill.price
            
            # 计算未实现盈亏
            await self.calculate_unrealized_pnl(position)
            
            # 计算强平价格
            position.liquidation_price = self.calculate_liquidation_price(position)
            
            # 更新总计数据
            self.update_totals()
            
            # 更新统计
            self.stats['total_volume'] += fill.quantity
            self.stats['total_commission'] += fill.commission
            
            # 发送仓位更新消息
            await message_bus.publish(Message(
                type=MessageType.POSITION_UPDATED,
                data={'position': position, 'fill': fill},
                priority=MessagePriority.HIGH
            ))
            
            logger.info(f"仓位更新: {position_key} - {position.side.value} {position.size}")
            
        except Exception as e:
            logger.error(f"更新仓位失败: {e}")
    
    async def calculate_unrealized_pnl(self, position: Position):
        """计算未实现盈亏"""
        try:
            if position.size == 0:
                position.unrealized_pnl = Decimal('0')
                return
            
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (position.mark_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - position.mark_price) * abs(position.size)
            
        except Exception as e:
            logger.error(f"计算未实现盈亏失败: {e}")
    
    def calculate_liquidation_price(self, position: Position) -> Optional[Decimal]:
        """计算强平价格"""
        try:
            if position.size == 0:
                return None
            
            # 简化的强平价格计算
            maintenance_margin_rate = Decimal('0.005')  # 0.5%维持保证金率
            
            if position.side == PositionSide.LONG:
                # 多头强平价格 = 开仓价格 * (1 - 1/杠杆 + 维持保证金率)
                liquidation_price = position.entry_price * (
                    Decimal('1') - Decimal('1') / position.leverage + maintenance_margin_rate
                )
            else:
                # 空头强平价格 = 开仓价格 * (1 + 1/杠杆 - 维持保证金率)
                liquidation_price = position.entry_price * (
                    Decimal('1') + Decimal('1') / position.leverage - maintenance_margin_rate
                )
            
            return liquidation_price
            
        except Exception as e:
            logger.error(f"计算强平价格失败: {e}")
            return None
    
    def update_totals(self):
        """更新总计数据"""
        self.total_margin = sum(pos.margin for pos in self.positions.values())
        self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
    
    async def update_mark_prices(self, price_data: Dict[str, Decimal]):
        """更新标记价格"""
        try:
            for position_key, position in self.positions.items():
                if position.size == 0:
                    continue
                
                symbol_price = price_data.get(position.symbol)
                if symbol_price:
                    position.mark_price = symbol_price
                    await self.calculate_unrealized_pnl(position)
                    position.updated_time = datetime.now()
            
            self.update_totals()
            
        except Exception as e:
            logger.error(f"更新标记价格失败: {e}")
    
    def get_position(self, exchange: str, symbol: str) -> Optional[Position]:
        """获取仓位"""
        position_key = self.get_position_key(exchange, symbol)
        return self.positions.get(position_key)
    
    def get_all_positions(self) -> List[Position]:
        """获取所有仓位"""
        return list(self.positions.values())
    
    def get_active_positions(self) -> List[Position]:
        """获取活跃仓位"""
        return [pos for pos in self.positions.values() if pos.size != 0]
    
    def get_positions_by_exchange(self, exchange: str) -> List[Position]:
        """获取指定交易所的仓位"""
        return [pos for pos in self.positions.values() if pos.exchange == exchange]
    
    def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """计算投资组合风险"""
        try:
            active_positions = self.get_active_positions()
            
            if not active_positions:
                return {
                    'total_exposure': Decimal('0'),
                    'total_margin': Decimal('0'),
                    'portfolio_leverage': Decimal('0'),
                    'risk_level': 'LOW',
                    'max_loss': Decimal('0'),
                    'concentration_risk': Decimal('0')
                }
            
            # 计算总敞口
            total_exposure = sum(pos.notional_value for pos in active_positions)
            
            # 计算投资组合杠杆
            portfolio_leverage = total_exposure / max(self.total_margin, Decimal('0.01'))
            
            # 计算最大损失
            max_loss = sum(pos.margin for pos in active_positions)
            
            # 计算集中度风险
            max_position_value = max((pos.notional_value for pos in active_positions), default=Decimal('0'))
            concentration_risk = max_position_value / max(total_exposure, Decimal('0.01'))
            
            # 风险等级评估
            risk_level = 'LOW'
            if portfolio_leverage > 10 or concentration_risk > 0.3:
                risk_level = 'HIGH'
            elif portfolio_leverage > 5 or concentration_risk > 0.2:
                risk_level = 'MEDIUM'
            
            return {
                'total_exposure': total_exposure,
                'total_margin': self.total_margin,
                'portfolio_leverage': portfolio_leverage,
                'risk_level': risk_level,
                'max_loss': max_loss,
                'concentration_risk': concentration_risk,
                'unrealized_pnl': self.total_unrealized_pnl,
                'realized_pnl': self.total_realized_pnl
            }
            
        except Exception as e:
            logger.error(f"计算投资组合风险失败: {e}")
            return {}
    
    def calculate_position_risk(self, position: Position) -> PositionRisk:
        """计算单个仓位风险"""
        try:
            # 计算到强平价格的距离
            liquidation_distance = Decimal('0')
            if position.liquidation_price and position.mark_price:
                liquidation_distance = abs(position.liquidation_price - position.mark_price) / position.mark_price
            
            # 风险等级评估
            risk_level = 'LOW'
            if liquidation_distance < Decimal('0.05'):  # 5%
                risk_level = 'CRITICAL'
            elif liquidation_distance < Decimal('0.1'):  # 10%
                risk_level = 'HIGH'
            elif liquidation_distance < Decimal('0.2'):  # 20%
                risk_level = 'MEDIUM'
            
            # 计算1日风险价值（简化版）
            volatility = Decimal('0.02')  # 假设2%日波动率
            var_1d = position.notional_value * volatility * Decimal('1.65')  # 95%置信度
            
            # 生成建议
            recommendations = []
            if risk_level == 'CRITICAL':
                recommendations.append("立即减仓或平仓")
                recommendations.append("增加保证金")
            elif risk_level == 'HIGH':
                recommendations.append("考虑减仓")
                recommendations.append("密切监控价格")
            elif position.leverage > 10:
                recommendations.append("考虑降低杠杆")
            
            return PositionRisk(
                symbol=position.symbol,
                position_value=position.notional_value,
                margin_used=position.margin,
                margin_ratio=position.margin_ratio,
                leverage=position.leverage,
                liquidation_distance=liquidation_distance,
                risk_level=risk_level,
                max_loss=position.margin,
                var_1d=var_1d,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"计算仓位风险失败: {e}")
            return PositionRisk(
                symbol=position.symbol,
                position_value=Decimal('0'),
                margin_used=Decimal('0'),
                margin_ratio=Decimal('0'),
                leverage=1,
                liquidation_distance=Decimal('0'),
                risk_level='UNKNOWN',
                max_loss=Decimal('0'),
                var_1d=Decimal('0')
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取绩效统计"""
        try:
            total_trades = self.stats['total_trades']
            winning_trades = self.stats['winning_trades']
            losing_trades = self.stats['losing_trades']
            
            win_rate = (winning_trades / max(total_trades, 1)) * 100
            
            # 计算夏普比率（简化版）
            if self.total_margin > 0:
                return_rate = (self.total_realized_pnl / self.total_margin) * 100
                # 假设无风险利率为3%，波动率为20%
                sharpe_ratio = (return_rate - 3) / 20
            else:
                sharpe_ratio = Decimal('0')
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_realized_pnl': self.total_realized_pnl,
                'total_unrealized_pnl': self.total_unrealized_pnl,
                'total_pnl': self.total_realized_pnl + self.total_unrealized_pnl,
                'total_volume': self.stats['total_volume'],
                'total_commission': self.stats['total_commission'],
                'sharpe_ratio': sharpe_ratio,
                'active_positions': len(self.get_active_positions())
            }
            
        except Exception as e:
            logger.error(f"获取绩效统计失败: {e}")
            return {}
    
    async def close_position(self, exchange: str, symbol: str, percentage: Decimal = Decimal('100')) -> bool:
        """平仓"""
        try:
            position = self.get_position(exchange, symbol)
            if not position or position.size == 0:
                logger.warning(f"没有可平仓位: {exchange}:{symbol}")
                return False
            
            close_size = abs(position.size) * (percentage / 100)
            close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            # 这里应该调用订单管理器创建平仓订单
            # 为了简化，这里只是记录日志
            logger.info(f"创建平仓订单: {symbol} {close_side.value} {close_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return False
    
    async def start(self):
        """启动仓位管理器"""
        try:
            # 启动定期任务
            asyncio.create_task(self._periodic_tasks())
            logger.info("仓位管理器启动成功")
            
        except Exception as e:
            logger.error(f"启动仓位管理器失败: {e}")
            raise
    
    async def _periodic_tasks(self):
        """定期任务"""
        while True:
            try:
                # 检查风险
                await self._check_risk()
                await asyncio.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"定期任务失败: {e}")
                await asyncio.sleep(30)
    
    async def _check_risk(self):
        """检查风险"""
        try:
            for position in self.get_active_positions():
                risk = self.calculate_position_risk(position)
                
                if risk.risk_level == 'CRITICAL':
                    # 发送风险警告
                    await message_bus.publish(Message(
                        type=MessageType.RISK_WARNING,
                        data={
                            'position': position,
                            'risk': risk,
                            'message': f"仓位 {position.symbol} 风险等级为 CRITICAL"
                        },
                        priority=MessagePriority.CRITICAL
                    ))
                    
                    logger.warning(f"仓位风险警告: {position.symbol} - {risk.risk_level}")
            
        except Exception as e:
            logger.error(f"风险检查失败: {e}")


# 全局仓位管理器实例
position_manager = PositionManager()

