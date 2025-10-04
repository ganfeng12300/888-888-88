"""
📋 订单管理系统
生产级订单管理系统，支持多交易所、多品种、高频交易
实现完整的订单生命周期管理、风险控制、执行优化等功能
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from decimal import Decimal, ROUND_HALF_UP
from loguru import logger

from src.core.config import settings
from src.system.message_bus import message_bus, Message, MessageType, MessagePriority
from src.security.encryption import config_manager


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWO_WAY = "two_way"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    exchange: str = "bitget"
    leverage: Optional[int] = None
    reduce_only: bool = False
    post_only: bool = False
    iceberg_qty: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None
    expire_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """订单对象"""
    id: str
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    time_in_force: TimeInForce
    exchange: str
    leverage: Optional[int]
    reduce_only: bool
    post_only: bool
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = field(init=False)
    average_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    commission_asset: str = ""
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    filled_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity
    
    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal, commission: Decimal = Decimal('0')):
        """更新成交信息"""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.commission += commission
        self.updated_time = datetime.now()
        
        # 更新平均价格
        if self.average_price is None:
            self.average_price = fill_price
        else:
            total_value = (self.filled_quantity - fill_quantity) * self.average_price + fill_quantity * fill_price
            self.average_price = total_value / self.filled_quantity
        
        # 更新状态
        if self.remaining_quantity <= Decimal('0'):
            self.status = OrderStatus.FILLED
            self.filled_time = datetime.now()
        elif self.filled_quantity > Decimal('0'):
            self.status = OrderStatus.PARTIAL_FILLED


@dataclass
class Fill:
    """成交记录"""
    id: str
    order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: datetime
    trade_id: str
    is_maker: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """订单管理器"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, Fill] = {}
        self.order_callbacks: Dict[str, List[Callable]] = {}
        self.exchange_managers = {}
        self.risk_manager = None
        self.position_manager = None
        
        # 订单统计
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': Decimal('0'),
            'total_commission': Decimal('0')
        }
        
        logger.info("订单管理器初始化完成")
    
    def set_exchange_manager(self, exchange: str, manager):
        """设置交易所管理器"""
        self.exchange_managers[exchange] = manager
        logger.info(f"设置交易所管理器: {exchange}")
    
    def set_risk_manager(self, risk_manager):
        """设置风险管理器"""
        self.risk_manager = risk_manager
        logger.info("设置风险管理器")
    
    def set_position_manager(self, position_manager):
        """设置仓位管理器"""
        self.position_manager = position_manager
        logger.info("设置仓位管理器")
    
    async def create_order(self, request: OrderRequest) -> Optional[Order]:
        """创建订单"""
        try:
            # 生成订单ID
            order_id = str(uuid.uuid4())
            client_order_id = request.client_order_id or f"order_{int(time.time() * 1000)}"
            
            # 风险检查
            if self.risk_manager:
                risk_result = await self.risk_manager.check_order_risk(request)
                if not risk_result.approved:
                    logger.warning(f"订单风险检查失败: {risk_result.reason}")
                    return None
            
            # 创建订单对象
            order = Order(
                id=order_id,
                client_order_id=client_order_id,
                exchange_order_id=None,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                stop_price=request.stop_price,
                status=OrderStatus.PENDING,
                time_in_force=request.time_in_force,
                exchange=request.exchange,
                leverage=request.leverage,
                reduce_only=request.reduce_only,
                post_only=request.post_only,
                metadata=request.metadata
            )
            
            # 保存订单
            self.orders[order_id] = order
            self.stats['total_orders'] += 1
            
            # 发送订单创建消息
            await message_bus.publish(Message(
                type=MessageType.ORDER_CREATED,
                data={'order': order},
                priority=MessagePriority.HIGH
            ))
            
            logger.info(f"订单创建成功: {order_id} - {request.symbol} {request.side.value} {request.quantity}")
            return order
            
        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            return None
    
    async def submit_order(self, order_id: str) -> bool:
        """提交订单到交易所"""
        try:
            order = self.orders.get(order_id)
            if not order:
                logger.error(f"订单不存在: {order_id}")
                return False
            
            if order.status != OrderStatus.PENDING:
                logger.warning(f"订单状态不正确: {order_id} - {order.status}")
                return False
            
            # 获取交易所管理器
            exchange_manager = self.exchange_managers.get(order.exchange)
            if not exchange_manager:
                logger.error(f"交易所管理器不存在: {order.exchange}")
                return False
            
            # 提交到交易所
            exchange_order_id = await exchange_manager.submit_order(order)
            if not exchange_order_id:
                order.status = OrderStatus.REJECTED
                self.stats['rejected_orders'] += 1
                logger.error(f"订单提交失败: {order_id}")
                return False
            
            # 更新订单状态
            order.exchange_order_id = exchange_order_id
            order.status = OrderStatus.SUBMITTED
            order.updated_time = datetime.now()
            
            # 发送订单提交消息
            await message_bus.publish(Message(
                type=MessageType.ORDER_SUBMITTED,
                data={'order': order},
                priority=MessagePriority.HIGH
            ))
            
            logger.info(f"订单提交成功: {order_id} - 交易所订单ID: {exchange_order_id}")
            return True
            
        except Exception as e:
            logger.error(f"提交订单失败: {order_id} - {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            order = self.orders.get(order_id)
            if not order:
                logger.error(f"订单不存在: {order_id}")
                return False
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"订单无法取消: {order_id} - {order.status}")
                return False
            
            # 如果已提交到交易所，需要向交易所发送取消请求
            if order.status == OrderStatus.SUBMITTED and order.exchange_order_id:
                exchange_manager = self.exchange_managers.get(order.exchange)
                if exchange_manager:
                    success = await exchange_manager.cancel_order(order.exchange_order_id)
                    if not success:
                        logger.error(f"交易所取消订单失败: {order_id}")
                        return False
            
            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_time = datetime.now()
            self.stats['cancelled_orders'] += 1
            
            # 发送订单取消消息
            await message_bus.publish(Message(
                type=MessageType.ORDER_CANCELLED,
                data={'order': order},
                priority=MessagePriority.HIGH
            ))
            
            logger.info(f"订单取消成功: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {order_id} - {e}")
            return False
    
    async def modify_order(self, order_id: str, new_quantity: Optional[Decimal] = None, 
                          new_price: Optional[Decimal] = None) -> bool:
        """修改订单"""
        try:
            order = self.orders.get(order_id)
            if not order:
                logger.error(f"订单不存在: {order_id}")
                return False
            
            if order.status != OrderStatus.SUBMITTED:
                logger.warning(f"订单状态不支持修改: {order_id} - {order.status}")
                return False
            
            # 获取交易所管理器
            exchange_manager = self.exchange_managers.get(order.exchange)
            if not exchange_manager:
                logger.error(f"交易所管理器不存在: {order.exchange}")
                return False
            
            # 修改订单
            success = await exchange_manager.modify_order(
                order.exchange_order_id, 
                new_quantity, 
                new_price
            )
            
            if success:
                # 更新本地订单信息
                if new_quantity:
                    order.quantity = new_quantity
                    order.remaining_quantity = new_quantity - order.filled_quantity
                if new_price:
                    order.price = new_price
                
                order.updated_time = datetime.now()
                
                # 发送订单修改消息
                await message_bus.publish(Message(
                    type=MessageType.ORDER_MODIFIED,
                    data={'order': order},
                    priority=MessagePriority.HIGH
                ))
                
                logger.info(f"订单修改成功: {order_id}")
                return True
            else:
                logger.error(f"订单修改失败: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"修改订单失败: {order_id} - {e}")
            return False
    
    async def handle_fill(self, fill: Fill):
        """处理成交回报"""
        try:
            # 保存成交记录
            self.fills[fill.id] = fill
            
            # 查找对应订单
            order = None
            for o in self.orders.values():
                if o.exchange_order_id == fill.exchange_order_id:
                    order = o
                    break
            
            if not order:
                logger.warning(f"未找到对应订单: {fill.exchange_order_id}")
                return
            
            # 更新订单成交信息
            order.update_fill(fill.quantity, fill.price, fill.commission)
            
            # 更新统计信息
            self.stats['total_volume'] += fill.quantity
            self.stats['total_commission'] += fill.commission
            
            if order.status == OrderStatus.FILLED:
                self.stats['filled_orders'] += 1
            
            # 更新仓位
            if self.position_manager:
                await self.position_manager.update_position(fill)
            
            # 发送成交消息
            await message_bus.publish(Message(
                type=MessageType.ORDER_FILLED,
                data={'order': order, 'fill': fill},
                priority=MessagePriority.CRITICAL
            ))
            
            # 执行回调函数
            callbacks = self.order_callbacks.get(order.id, [])
            for callback in callbacks:
                try:
                    await callback(order, fill)
                except Exception as e:
                    logger.error(f"订单回调执行失败: {e}")
            
            logger.info(f"处理成交回报: {fill.id} - {fill.symbol} {fill.quantity}@{fill.price}")
            
        except Exception as e:
            logger.error(f"处理成交回报失败: {e}")
    
    def add_order_callback(self, order_id: str, callback: Callable):
        """添加订单回调函数"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append(callback)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """获取指定品种的订单"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        active_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]
        return [order for order in self.orders.values() if order.status in active_statuses]
    
    def get_filled_orders(self, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> List[Order]:
        """获取已成交订单"""
        filled_orders = [order for order in self.orders.values() if order.status == OrderStatus.FILLED]
        
        if start_time:
            filled_orders = [order for order in filled_orders if order.filled_time >= start_time]
        if end_time:
            filled_orders = [order for order in filled_orders if order.filled_time <= end_time]
        
        return filled_orders
    
    def get_fills_by_symbol(self, symbol: str) -> List[Fill]:
        """获取指定品种的成交记录"""
        return [fill for fill in self.fills.values() if fill.symbol == symbol]
    
    def get_order_stats(self) -> Dict[str, Any]:
        """获取订单统计信息"""
        active_orders = len(self.get_active_orders())
        
        return {
            **self.stats,
            'active_orders': active_orders,
            'fill_rate': (self.stats['filled_orders'] / max(self.stats['total_orders'], 1)) * 100,
            'cancel_rate': (self.stats['cancelled_orders'] / max(self.stats['total_orders'], 1)) * 100,
            'reject_rate': (self.stats['rejected_orders'] / max(self.stats['total_orders'], 1)) * 100
        }
    
    async def cleanup_expired_orders(self):
        """清理过期订单"""
        try:
            current_time = datetime.now()
            expired_orders = []
            
            for order in self.orders.values():
                if (order.time_in_force == TimeInForce.GTD and 
                    order.metadata.get('expire_time') and 
                    current_time > order.metadata['expire_time'] and
                    order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]):
                    expired_orders.append(order.id)
            
            for order_id in expired_orders:
                await self.cancel_order(order_id)
                order = self.orders[order_id]
                order.status = OrderStatus.EXPIRED
                logger.info(f"订单已过期: {order_id}")
            
        except Exception as e:
            logger.error(f"清理过期订单失败: {e}")
    
    async def start(self):
        """启动订单管理器"""
        try:
            # 启动定期清理任务
            asyncio.create_task(self._periodic_cleanup())
            logger.info("订单管理器启动成功")
            
        except Exception as e:
            logger.error(f"启动订单管理器失败: {e}")
            raise
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        while True:
            try:
                await self.cleanup_expired_orders()
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"定期清理任务失败: {e}")
                await asyncio.sleep(60)


# 全局订单管理器实例
order_manager = OrderManager()

