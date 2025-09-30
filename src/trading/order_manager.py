#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 订单管理系统
智能订单生命周期管理，支持复杂订单策略
专为史诗级AI量化交易设计，生产级实盘交易标准
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path

class OrderPriority(Enum):
    """订单优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class OrderStrategy(Enum):
    """订单策略"""
    IMMEDIATE = "immediate"  # 立即执行
    TWAP = "twap"           # 时间加权平均价格
    VWAP = "vwap"           # 成交量加权平均价格
    ICEBERG = "iceberg"     # 冰山订单
    SNIPER = "sniper"       # 狙击订单
    GRID = "grid"           # 网格订单

@dataclass
class OrderExecution:
    """订单执行记录"""
    execution_id: str
    order_id: str
    executed_amount: float
    execution_price: float
    execution_time: datetime
    fees: float
    exchange: str
    execution_type: str  # 'partial' or 'full'
    slippage: float = 0.0
    market_impact: float = 0.0

@dataclass
class OrderCondition:
    """订单条件"""
    condition_type: str  # 'price', 'time', 'volume', 'indicator'
    operator: str       # '>', '<', '>=', '<=', '=='
    value: float
    indicator: Optional[str] = None  # 技术指标名称
    is_met: bool = False

@dataclass
class SmartOrder:
    """智能订单"""
    order_id: str
    parent_order_id: Optional[str]
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str
    total_amount: float
    executed_amount: float
    remaining_amount: float
    target_price: Optional[float]
    limit_price: Optional[float]
    stop_price: Optional[float]
    
    # 智能特性
    strategy: OrderStrategy
    priority: OrderPriority
    conditions: List[OrderCondition]
    max_slippage: float
    time_in_force: str  # 'GTC', 'IOC', 'FOK', 'GTD'
    expire_time: Optional[datetime]
    
    # 执行控制
    min_fill_size: float
    max_fill_size: float
    execution_interval: float  # 秒
    adaptive_sizing: bool
    
    # 状态信息
    status: str
    created_at: datetime
    updated_at: datetime
    last_execution_time: Optional[datetime]
    executions: List[OrderExecution]
    
    # 性能统计
    average_price: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    success_rate: float = 0.0
    
    def __post_init__(self):
        self.remaining_amount = self.total_amount - self.executed_amount
        if not self.executions:
            self.executions = []

class OrderValidator:
    """订单验证器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_order_size = self.config.get('min_order_size', 0.001)
        self.max_order_size = self.config.get('max_order_size', 1000)
        self.max_price_deviation = self.config.get('max_price_deviation', 0.1)  # 10%
        
    def validate_order(self, order: SmartOrder, current_price: float = None) -> Tuple[bool, str]:
        """验证订单"""
        try:
            # 基础验证
            if order.total_amount <= 0:
                return False, "订单数量必须大于0"
            
            if order.total_amount < self.min_order_size:
                return False, f"订单数量不能小于 {self.min_order_size}"
            
            if order.total_amount > self.max_order_size:
                return False, f"订单数量不能大于 {self.max_order_size}"
            
            # 价格验证
            if current_price and order.limit_price:
                price_deviation = abs(order.limit_price - current_price) / current_price
                if price_deviation > self.max_price_deviation:
                    return False, f"价格偏离过大: {price_deviation:.2%}"
            
            # 止损价格验证
            if order.stop_price and order.limit_price:
                if order.side == 'buy':
                    if order.stop_price < order.limit_price:
                        return False, "买入止损价格不能低于限价"
                else:
                    if order.stop_price > order.limit_price:
                        return False, "卖出止损价格不能高于限价"
            
            # 时间验证
            if order.expire_time and order.expire_time <= datetime.now(timezone.utc):
                return False, "订单已过期"
            
            # 策略特定验证
            if order.strategy == OrderStrategy.ICEBERG:
                if order.max_fill_size >= order.total_amount:
                    return False, "冰山订单的最大执行量必须小于总量"
            
            return True, "验证通过"
            
        except Exception as e:
            logger.error(f"❌ 订单验证异常: {e}")
            return False, f"验证异常: {str(e)}"

class OrderExecutor:
    """订单执行器"""
    
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.execution_lock = threading.Lock()
        
    async def execute_order_slice(self, order: SmartOrder, slice_amount: float) -> Optional[OrderExecution]:
        """执行订单切片"""
        try:
            execution_id = f"exec_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            # 根据策略确定执行价格
            execution_price = await self._determine_execution_price(order, slice_amount)
            
            # 执行交易
            from src.trading.trading_engine import OrderSide, OrderType
            
            side = OrderSide.BUY if order.side == 'buy' else OrderSide.SELL
            order_type = OrderType.LIMIT if order.order_type == 'limit' else OrderType.MARKET
            
            result = await self.trading_engine.execute_order(
                symbol=order.symbol,
                side=side,
                amount=slice_amount,
                order_type=order_type,
                price=execution_price
            )
            
            if result.success:
                # 创建执行记录
                execution = OrderExecution(
                    execution_id=execution_id,
                    order_id=order.order_id,
                    executed_amount=slice_amount,
                    execution_price=execution_price,
                    execution_time=datetime.now(timezone.utc),
                    fees=result.fees,
                    exchange="auto",  # 由交易引擎自动选择
                    execution_type='partial' if slice_amount < order.remaining_amount else 'full',
                    slippage=result.slippage
                )
                
                logger.info(f"✅ 订单切片执行成功: {order.order_id} 数量: {slice_amount}")
                return execution
            else:
                logger.error(f"❌ 订单切片执行失败: {order.order_id} - {result.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 订单切片执行异常: {e}")
            return None
    
    async def _determine_execution_price(self, order: SmartOrder, amount: float) -> Optional[float]:
        """确定执行价格"""
        try:
            if order.order_type == 'market':
                return None  # 市价单不需要指定价格
            
            if order.strategy == OrderStrategy.TWAP:
                # TWAP策略：使用当前市价附近的价格
                # 这里简化处理，实际应该获取实时价格
                return order.target_price
            
            elif order.strategy == OrderStrategy.VWAP:
                # VWAP策略：基于成交量加权
                return order.target_price
            
            elif order.strategy == OrderStrategy.SNIPER:
                # 狙击策略：使用精确的限价
                return order.limit_price
            
            else:
                return order.limit_price or order.target_price
                
        except Exception as e:
            logger.error(f"❌ 价格确定失败: {e}")
            return order.target_price

class OrderScheduler:
    """订单调度器"""
    
    def __init__(self, order_executor: OrderExecutor):
        self.executor = order_executor
        self.scheduled_orders = {}  # order_id -> next_execution_time
        self.running = False
        self.scheduler_task = None
        
    async def start(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("📅 订单调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("📅 订单调度器已停止")
    
    def schedule_order(self, order: SmartOrder):
        """调度订单"""
        next_time = datetime.now(timezone.utc) + timedelta(seconds=order.execution_interval)
        self.scheduled_orders[order.order_id] = next_time
        logger.debug(f"📅 订单已调度: {order.order_id} 下次执行: {next_time}")
    
    def unschedule_order(self, order_id: str):
        """取消调度"""
        if order_id in self.scheduled_orders:
            del self.scheduled_orders[order_id]
            logger.debug(f"📅 订单调度已取消: {order_id}")
    
    async def _scheduler_loop(self):
        """调度循环"""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                ready_orders = []
                
                for order_id, next_time in self.scheduled_orders.items():
                    if now >= next_time:
                        ready_orders.append(order_id)
                
                # 处理就绪的订单
                for order_id in ready_orders:
                    # 这里需要从订单管理器获取订单并执行
                    # 简化处理
                    del self.scheduled_orders[order_id]
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"❌ 调度循环异常: {e}")
                await asyncio.sleep(5)

class OrderManager:
    """🦊 猎狐AI - 智能订单管理系统"""
    
    def __init__(self, trading_engine, config: Dict[str, Any] = None):
        self.trading_engine = trading_engine
        self.config = config or {}
        
        # 核心组件
        self.validator = OrderValidator(config)
        self.executor = OrderExecutor(trading_engine)
        self.scheduler = OrderScheduler(self.executor)
        
        # 订单存储
        self.active_orders = {}  # order_id -> SmartOrder
        self.completed_orders = {}  # order_id -> SmartOrder
        self.order_lock = threading.Lock()
        
        # 数据库
        self.db_path = Path(config.get('db_path', 'data/orders.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 统计信息
        self.stats = {
            'total_orders': 0,
            'active_orders': 0,
            'completed_orders': 0,
            'cancelled_orders': 0,
            'success_rate': 0.0,
            'average_execution_time': 0.0,
            'total_volume': 0.0,
            'total_fees': 0.0
        }
        
        logger.info("🦊 猎狐AI智能订单管理系统初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 订单表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    executed_amount REAL DEFAULT 0,
                    target_price REAL,
                    limit_price REAL,
                    stop_price REAL,
                    strategy TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    order_data TEXT  -- JSON格式的完整订单数据
                )
            ''')
            
            # 执行记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    executed_amount REAL NOT NULL,
                    execution_price REAL NOT NULL,
                    execution_time TIMESTAMP NOT NULL,
                    fees REAL NOT NULL,
                    exchange TEXT NOT NULL,
                    slippage REAL DEFAULT 0,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ 订单数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
    
    async def start(self):
        """启动订单管理器"""
        await self.scheduler.start()
        logger.success("🚀 智能订单管理系统已启动")
    
    async def stop(self):
        """停止订单管理器"""
        await self.scheduler.stop()
        
        # 取消所有活跃订单
        with self.order_lock:
            for order in list(self.active_orders.values()):
                await self.cancel_order(order.order_id)
        
        logger.info("🛑 智能订单管理系统已停止")
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = 'market',
                          target_price: Optional[float] = None,
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          strategy: OrderStrategy = OrderStrategy.IMMEDIATE,
                          priority: OrderPriority = OrderPriority.NORMAL,
                          conditions: List[OrderCondition] = None,
                          **kwargs) -> Optional[str]:
        """创建智能订单"""
        try:
            # 创建订单对象
            order = SmartOrder(
                order_id=f"smart_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                parent_order_id=kwargs.get('parent_order_id'),
                symbol=symbol,
                side=side,
                order_type=order_type,
                total_amount=amount,
                executed_amount=0.0,
                remaining_amount=amount,
                target_price=target_price,
                limit_price=limit_price,
                stop_price=stop_price,
                strategy=strategy,
                priority=priority,
                conditions=conditions or [],
                max_slippage=kwargs.get('max_slippage', 0.01),
                time_in_force=kwargs.get('time_in_force', 'GTC'),
                expire_time=kwargs.get('expire_time'),
                min_fill_size=kwargs.get('min_fill_size', amount * 0.01),
                max_fill_size=kwargs.get('max_fill_size', amount * 0.1),
                execution_interval=kwargs.get('execution_interval', 60.0),
                adaptive_sizing=kwargs.get('adaptive_sizing', True),
                status='pending',
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_execution_time=None,
                executions=[]
            )
            
            # 验证订单
            is_valid, error_msg = self.validator.validate_order(order)
            if not is_valid:
                logger.error(f"❌ 订单验证失败: {error_msg}")
                return None
            
            # 保存订单
            with self.order_lock:
                self.active_orders[order.order_id] = order
                self.stats['total_orders'] += 1
                self.stats['active_orders'] += 1
            
            # 保存到数据库
            await self._save_order_to_db(order)
            
            # 根据策略处理订单
            if strategy == OrderStrategy.IMMEDIATE:
                # 立即执行
                await self._execute_immediate_order(order)
            else:
                # 调度执行
                self.scheduler.schedule_order(order)
                order.status = 'scheduled'
            
            logger.info(f"✅ 智能订单已创建: {order.order_id} {symbol} {side} {amount}")
            return order.order_id
            
        except Exception as e:
            logger.error(f"❌ 创建订单失败: {e}")
            return None
    
    async def _execute_immediate_order(self, order: SmartOrder):
        """立即执行订单"""
        try:
            order.status = 'executing'
            
            if order.strategy == OrderStrategy.ICEBERG:
                # 冰山订单：分批执行
                await self._execute_iceberg_order(order)
            else:
                # 普通订单：一次性执行
                execution = await self.executor.execute_order_slice(order, order.total_amount)
                if execution:
                    await self._process_execution(order, execution)
                else:
                    order.status = 'failed'
                    
        except Exception as e:
            logger.error(f"❌ 立即执行订单失败: {e}")
            order.status = 'failed'
    
    async def _execute_iceberg_order(self, order: SmartOrder):
        """执行冰山订单"""
        try:
            while order.remaining_amount > 0:
                # 计算本次执行数量
                slice_amount = min(order.max_fill_size, order.remaining_amount)
                
                # 执行切片
                execution = await self.executor.execute_order_slice(order, slice_amount)
                if execution:
                    await self._process_execution(order, execution)
                    
                    # 如果还有剩余，等待一段时间再执行下一片
                    if order.remaining_amount > 0:
                        await asyncio.sleep(order.execution_interval)
                else:
                    # 执行失败，停止
                    order.status = 'failed'
                    break
                    
        except Exception as e:
            logger.error(f"❌ 冰山订单执行失败: {e}")
            order.status = 'failed'
    
    async def _process_execution(self, order: SmartOrder, execution: OrderExecution):
        """处理执行结果"""
        try:
            # 更新订单状态
            order.executed_amount += execution.executed_amount
            order.remaining_amount = order.total_amount - order.executed_amount
            order.executions.append(execution)
            order.last_execution_time = execution.execution_time
            order.updated_at = datetime.now(timezone.utc)
            
            # 更新统计信息
            order.total_fees += execution.fees
            order.total_slippage += execution.slippage
            
            # 计算平均价格
            total_value = sum(e.executed_amount * e.execution_price for e in order.executions)
            order.average_price = total_value / order.executed_amount if order.executed_amount > 0 else 0
            
            # 检查订单是否完成
            if order.remaining_amount <= 0.001:  # 考虑精度问题
                order.status = 'filled'
                await self._complete_order(order)
            else:
                order.status = 'partially_filled'
            
            # 保存执行记录到数据库
            await self._save_execution_to_db(execution)
            
            logger.info(f"📊 订单执行更新: {order.order_id} 已执行: {order.executed_amount}/{order.total_amount}")
            
        except Exception as e:
            logger.error(f"❌ 处理执行结果失败: {e}")
    
    async def _complete_order(self, order: SmartOrder):
        """完成订单"""
        try:
            with self.order_lock:
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                    self.completed_orders[order.order_id] = order
                    
                    self.stats['active_orders'] -= 1
                    self.stats['completed_orders'] += 1
                    self.stats['total_volume'] += order.executed_amount
                    self.stats['total_fees'] += order.total_fees
            
            # 取消调度
            self.scheduler.unschedule_order(order.order_id)
            
            logger.success(f"✅ 订单完成: {order.order_id} 平均价格: {order.average_price:.6f}")
            
        except Exception as e:
            logger.error(f"❌ 完成订单处理失败: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            with self.order_lock:
                if order_id not in self.active_orders:
                    logger.warning(f"⚠️ 订单不存在或已完成: {order_id}")
                    return False
                
                order = self.active_orders[order_id]
                order.status = 'cancelled'
                order.updated_at = datetime.now(timezone.utc)
                
                # 移到已完成订单
                del self.active_orders[order_id]
                self.completed_orders[order_id] = order
                
                self.stats['active_orders'] -= 1
                self.stats['cancelled_orders'] += 1
            
            # 取消调度
            self.scheduler.unschedule_order(order_id)
            
            logger.info(f"✅ 订单已取消: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 取消订单失败: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[SmartOrder]:
        """获取订单"""
        with self.order_lock:
            return self.active_orders.get(order_id) or self.completed_orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[SmartOrder]:
        """获取活跃订单"""
        with self.order_lock:
            orders = list(self.active_orders.values())
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[SmartOrder]:
        """获取订单历史"""
        with self.order_lock:
            orders = list(self.completed_orders.values())
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            # 按时间排序
            orders.sort(key=lambda x: x.updated_at, reverse=True)
            return orders[:limit]
    
    async def _save_order_to_db(self, order: SmartOrder):
        """保存订单到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 序列化订单数据
            order_data = {
                'conditions': [{'condition_type': c.condition_type, 'operator': c.operator, 
                              'value': c.value, 'indicator': c.indicator} for c in order.conditions],
                'max_slippage': order.max_slippage,
                'time_in_force': order.time_in_force,
                'expire_time': order.expire_time.isoformat() if order.expire_time else None,
                'min_fill_size': order.min_fill_size,
                'max_fill_size': order.max_fill_size,
                'execution_interval': order.execution_interval,
                'adaptive_sizing': order.adaptive_sizing
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO orders 
                (order_id, symbol, side, order_type, total_amount, executed_amount,
                 target_price, limit_price, stop_price, strategy, priority, status,
                 created_at, updated_at, order_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.order_id, order.symbol, order.side, order.order_type,
                order.total_amount, order.executed_amount, order.target_price,
                order.limit_price, order.stop_price, order.strategy.value,
                order.priority.value, order.status, order.created_at,
                order.updated_at, json.dumps(order_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存订单到数据库失败: {e}")
    
    async def _save_execution_to_db(self, execution: OrderExecution):
        """保存执行记录到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO executions 
                (execution_id, order_id, executed_amount, execution_price,
                 execution_time, fees, exchange, slippage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id, execution.order_id, execution.executed_amount,
                execution.execution_price, execution.execution_time, execution.fees,
                execution.exchange, execution.slippage
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存执行记录到数据库失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.order_lock:
            stats = self.stats.copy()
            
            # 计算成功率
            if stats['total_orders'] > 0:
                stats['success_rate'] = stats['completed_orders'] / stats['total_orders']
            
            return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            return {
                'status': 'healthy',
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders),
                'scheduler_running': self.scheduler.running,
                'database_accessible': self.db_path.exists()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
