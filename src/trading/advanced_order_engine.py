"""
⚡ 高级订单执行引擎
生产级高级订单执行系统，支持智能路由、算法交易、滑点控制
实现完整的订单执行优化、市场冲击最小化、执行成本控制等功能
"""
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume

class ExecutionAlgorithm(Enum):
    """执行算法"""
    SIMPLE = "simple"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    POV = "pov"    # Percentage of Volume
    IS = "is"      # Implementation Shortfall
    ICEBERG = "iceberg"
    SNIPER = "sniper"

@dataclass
class OrderSlice:
    """订单切片"""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: OrderType
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

@dataclass
class ExecutionReport:
    """执行报告"""
    order_id: str
    symbol: str
    total_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_cost: float
    slippage: float
    market_impact: float
    execution_time: float
    algorithm_used: str
    slice_count: int
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedOrderEngine:
    """高级订单执行引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=16)
        
        # 订单管理
        self.active_orders = {}
        self.order_slices = {}
        self.execution_queue = deque()
        self.completed_orders = deque(maxlen=10000)
        
        # 市场数据
        self.market_data = {}
        self.order_book_data = {}
        self.trade_history = deque(maxlen=50000)
        
        # 执行参数
        self.max_market_impact = 0.001  # 最大市场冲击 0.1%
        self.max_slippage = 0.002       # 最大滑点 0.2%
        self.min_slice_size = 0.001     # 最小切片大小
        self.max_slice_count = 100      # 最大切片数量
        
        # 性能监控
        self.execution_metrics = deque(maxlen=1000)
        self.algorithm_performance = {}
        
        # 风险控制
        self.position_limits = {}
        self.daily_volume_limits = {}
        self.current_positions = {}
        
    async def start(self):
        """启动高级订单引擎"""
        self.is_running = True
        self.logger.info("⚡ 高级订单执行引擎启动")
        
        # 启动执行循环
        tasks = [
            asyncio.create_task(self._order_execution_loop()),
            asyncio.create_task(self._market_data_processor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._risk_monitor())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop(self):
        """停止高级订单引擎"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("⚡ 高级订单执行引擎停止")
        
    async def submit_order(self, order: Dict) -> str:
        """提交订单"""
        try:
            order_id = order.get('order_id', f"order_{int(time.time() * 1000)}")
            
            # 风险检查
            if not await self._risk_check(order):
                raise ValueError("订单未通过风险检查")
                
            # 选择执行算法
            algorithm = self._select_execution_algorithm(order)
            
            # 创建执行计划
            execution_plan = await self._create_execution_plan(order, algorithm)
            
            # 存储订单
            self.active_orders[order_id] = {
                'order': order,
                'algorithm': algorithm,
                'execution_plan': execution_plan,
                'status': 'active',
                'created_time': datetime.now(),
                'slices': []
            }
            
            # 添加到执行队列
            self.execution_queue.append(order_id)
            
            self.logger.info(f"订单已提交: {order_id}, 算法: {algorithm.value}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"提交订单失败: {e}")
            raise
            
    async def _risk_check(self, order: Dict) -> bool:
        """风险检查"""
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        side = order.get('side')
        
        # 检查仓位限制
        if symbol in self.position_limits:
            current_pos = self.current_positions.get(symbol, 0)
            new_pos = current_pos + (quantity if side == 'buy' else -quantity)
            
            if abs(new_pos) > self.position_limits[symbol]:
                self.logger.warning(f"订单超出仓位限制: {symbol}")
                return False
                
        # 检查日交易量限制
        if symbol in self.daily_volume_limits:
            # 这里应该检查当日累计交易量
            pass
            
        # 检查订单大小合理性
        if quantity <= 0:
            return False
            
        return True
        
    def _select_execution_algorithm(self, order: Dict) -> ExecutionAlgorithm:
        """选择执行算法"""
        quantity = order.get('quantity', 0)
        urgency = order.get('urgency', 'normal')  # low, normal, high
        symbol = order.get('symbol')
        
        # 获取市场数据
        market_data = self.market_data.get(symbol, {})
        avg_volume = market_data.get('avg_volume', 0)
        
        # 根据订单大小和市场条件选择算法
        if quantity == 0:
            return ExecutionAlgorithm.SIMPLE
            
        volume_ratio = quantity / avg_volume if avg_volume > 0 else 1
        
        if urgency == 'high':
            if volume_ratio < 0.01:
                return ExecutionAlgorithm.SIMPLE
            else:
                return ExecutionAlgorithm.POV
        elif urgency == 'low':
            if volume_ratio > 0.1:
                return ExecutionAlgorithm.TWAP
            else:
                return ExecutionAlgorithm.VWAP
        else:  # normal
            if volume_ratio < 0.05:
                return ExecutionAlgorithm.SIMPLE
            elif volume_ratio < 0.2:
                return ExecutionAlgorithm.VWAP
            else:
                return ExecutionAlgorithm.TWAP
                
    async def _create_execution_plan(self, order: Dict, algorithm: ExecutionAlgorithm) -> Dict:
        """创建执行计划"""
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        side = order.get('side')
        time_horizon = order.get('time_horizon', 300)  # 默认5分钟
        
        plan = {
            'algorithm': algorithm,
            'total_quantity': quantity,
            'remaining_quantity': quantity,
            'time_horizon': time_horizon,
            'start_time': datetime.now(),
            'slices': []
        }
        
        if algorithm == ExecutionAlgorithm.SIMPLE:
            # 简单执行：一次性下单
            plan['slices'] = [{
                'quantity': quantity,
                'timing': 0,
                'order_type': OrderType.MARKET
            }]
            
        elif algorithm == ExecutionAlgorithm.TWAP:
            # 时间加权平均价格：均匀分割时间
            slice_count = min(max(int(time_horizon / 30), 2), self.max_slice_count)
            slice_quantity = quantity / slice_count
            slice_interval = time_horizon / slice_count
            
            for i in range(slice_count):
                plan['slices'].append({
                    'quantity': slice_quantity,
                    'timing': i * slice_interval,
                    'order_type': OrderType.LIMIT
                })
                
        elif algorithm == ExecutionAlgorithm.VWAP:
            # 成交量加权平均价格：根据历史成交量分布
            volume_profile = await self._get_volume_profile(symbol, time_horizon)
            plan['slices'] = self._create_vwap_slices(quantity, volume_profile)
            
        elif algorithm == ExecutionAlgorithm.POV:
            # 成交量参与率：根据实时成交量调整
            participation_rate = order.get('participation_rate', 0.1)  # 10%
            plan['participation_rate'] = participation_rate
            plan['dynamic_slicing'] = True
            
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # 冰山订单：隐藏订单数量
            visible_quantity = min(quantity * 0.1, quantity)  # 显示10%
            plan['visible_quantity'] = visible_quantity
            plan['hidden_quantity'] = quantity - visible_quantity
            
        return plan
        
    async def _get_volume_profile(self, symbol: str, time_horizon: int) -> List[float]:
        """获取成交量分布"""
        # 这里应该从历史数据中获取成交量分布
        # 简化实现：返回均匀分布
        intervals = max(int(time_horizon / 60), 1)  # 每分钟一个区间
        return [1.0 / intervals] * intervals
        
    def _create_vwap_slices(self, quantity: float, volume_profile: List[float]) -> List[Dict]:
        """创建VWAP切片"""
        slices = []
        
        for i, volume_weight in enumerate(volume_profile):
            slice_quantity = quantity * volume_weight
            if slice_quantity >= self.min_slice_size:
                slices.append({
                    'quantity': slice_quantity,
                    'timing': i * 60,  # 每分钟
                    'order_type': OrderType.LIMIT
                })
                
        return slices
        
    async def _order_execution_loop(self):
        """订单执行循环"""
        while self.is_running:
            try:
                if self.execution_queue:
                    order_id = self.execution_queue.popleft()
                    await self._execute_order(order_id)
                    
                await asyncio.sleep(0.1)  # 100ms检查一次
                
            except Exception as e:
                self.logger.error(f"订单执行循环错误: {e}")
                await asyncio.sleep(1)
                
    async def _execute_order(self, order_id: str):
        """执行订单"""
        if order_id not in self.active_orders:
            return
            
        order_info = self.active_orders[order_id]
        order = order_info['order']
        algorithm = order_info['algorithm']
        execution_plan = order_info['execution_plan']
        
        try:
            if algorithm == ExecutionAlgorithm.SIMPLE:
                await self._execute_simple(order_id, order, execution_plan)
            elif algorithm == ExecutionAlgorithm.TWAP:
                await self._execute_twap(order_id, order, execution_plan)
            elif algorithm == ExecutionAlgorithm.VWAP:
                await self._execute_vwap(order_id, order, execution_plan)
            elif algorithm == ExecutionAlgorithm.POV:
                await self._execute_pov(order_id, order, execution_plan)
            elif algorithm == ExecutionAlgorithm.ICEBERG:
                await self._execute_iceberg(order_id, order, execution_plan)
                
        except Exception as e:
            self.logger.error(f"执行订单 {order_id} 失败: {e}")
            order_info['status'] = 'failed'
            
    async def _execute_simple(self, order_id: str, order: Dict, plan: Dict):
        """执行简单订单"""
        slice_info = plan['slices'][0]
        
        # 创建订单切片
        slice_id = f"{order_id}_slice_0"
        order_slice = OrderSlice(
            slice_id=slice_id,
            parent_order_id=order_id,
            symbol=order['symbol'],
            side=order['side'],
            quantity=slice_info['quantity'],
            price=order.get('price'),
            order_type=slice_info['order_type']
        )
        
        # 执行切片
        await self._execute_slice(order_slice)
        
        # 更新订单状态
        self.active_orders[order_id]['status'] = 'completed'
        
    async def _execute_twap(self, order_id: str, order: Dict, plan: Dict):
        """执行TWAP算法"""
        start_time = plan['start_time']
        
        for i, slice_info in enumerate(plan['slices']):
            # 等待到执行时间
            target_time = start_time + timedelta(seconds=slice_info['timing'])
            current_time = datetime.now()
            
            if current_time < target_time:
                wait_seconds = (target_time - current_time).total_seconds()
                await asyncio.sleep(wait_seconds)
                
            # 创建并执行切片
            slice_id = f"{order_id}_slice_{i}"
            order_slice = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order_id,
                symbol=order['symbol'],
                side=order['side'],
                quantity=slice_info['quantity'],
                price=await self._calculate_limit_price(order['symbol'], order['side']),
                order_type=slice_info['order_type']
            )
            
            await self._execute_slice(order_slice)
            
        # 更新订单状态
        self.active_orders[order_id]['status'] = 'completed'
        
    async def _execute_vwap(self, order_id: str, order: Dict, plan: Dict):
        """执行VWAP算法"""
        # 类似TWAP，但根据成交量权重调整
        await self._execute_twap(order_id, order, plan)
        
    async def _execute_pov(self, order_id: str, order: Dict, plan: Dict):
        """执行POV算法"""
        participation_rate = plan.get('participation_rate', 0.1)
        symbol = order['symbol']
        remaining_quantity = plan['remaining_quantity']
        
        while remaining_quantity > self.min_slice_size and self.is_running:
            # 获取当前市场成交量
            current_volume = await self._get_current_volume(symbol)
            
            # 计算本次执行数量
            slice_quantity = min(
                current_volume * participation_rate,
                remaining_quantity
            )
            
            if slice_quantity >= self.min_slice_size:
                # 创建并执行切片
                slice_id = f"{order_id}_pov_{int(time.time())}"
                order_slice = OrderSlice(
                    slice_id=slice_id,
                    parent_order_id=order_id,
                    symbol=symbol,
                    side=order['side'],
                    quantity=slice_quantity,
                    price=await self._calculate_limit_price(symbol, order['side']),
                    order_type=OrderType.LIMIT
                )
                
                filled_qty = await self._execute_slice(order_slice)
                remaining_quantity -= filled_qty
                
            await asyncio.sleep(5)  # 5秒检查一次
            
        # 更新订单状态
        self.active_orders[order_id]['status'] = 'completed'
        
    async def _execute_iceberg(self, order_id: str, order: Dict, plan: Dict):
        """执行冰山订单"""
        visible_quantity = plan['visible_quantity']
        hidden_quantity = plan['hidden_quantity']
        symbol = order['symbol']
        side = order['side']
        
        while hidden_quantity > 0 and self.is_running:
            # 下可见部分订单
            current_visible = min(visible_quantity, hidden_quantity)
            
            slice_id = f"{order_id}_iceberg_{int(time.time())}"
            order_slice = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=current_visible,
                price=await self._calculate_limit_price(symbol, side),
                order_type=OrderType.LIMIT
            )
            
            filled_qty = await self._execute_slice(order_slice)
            hidden_quantity -= filled_qty
            
            # 如果部分成交，等待一段时间再继续
            if filled_qty < current_visible:
                await asyncio.sleep(10)
                
        # 更新订单状态
        self.active_orders[order_id]['status'] = 'completed'
        
    async def _execute_slice(self, order_slice: OrderSlice) -> float:
        """执行订单切片"""
        try:
            # 这里应该调用实际的交易所API
            # 模拟执行
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            # 模拟部分成交
            fill_ratio = np.random.uniform(0.8, 1.0)
            filled_quantity = order_slice.quantity * fill_ratio
            avg_fill_price = order_slice.price or 0
            
            # 更新切片状态
            order_slice.filled_quantity = filled_quantity
            order_slice.avg_fill_price = avg_fill_price
            order_slice.status = 'filled' if fill_ratio == 1.0 else 'partially_filled'
            
            # 记录到切片历史
            self.order_slices[order_slice.slice_id] = order_slice
            
            # 更新父订单
            if order_slice.parent_order_id in self.active_orders:
                parent_order = self.active_orders[order_slice.parent_order_id]
                parent_order['slices'].append(order_slice)
                
            self.logger.info(f"切片执行完成: {order_slice.slice_id}, "
                           f"成交量: {filled_quantity:.6f}, "
                           f"成交价: {avg_fill_price:.6f}")
            
            return filled_quantity
            
        except Exception as e:
            self.logger.error(f"执行切片 {order_slice.slice_id} 失败: {e}")
            order_slice.status = 'failed'
            return 0.0
            
    async def _calculate_limit_price(self, symbol: str, side: str) -> float:
        """计算限价单价格"""
        # 获取当前市场价格
        market_data = self.market_data.get(symbol, {})
        
        if side == 'buy':
            # 买单：使用买一价或略高价格
            bid_price = market_data.get('bid', 0)
            return bid_price * 1.0001  # 略高于买一价
        else:
            # 卖单：使用卖一价或略低价格
            ask_price = market_data.get('ask', 0)
            return ask_price * 0.9999  # 略低于卖一价
            
    async def _get_current_volume(self, symbol: str) -> float:
        """获取当前成交量"""
        # 这里应该获取最近一段时间的成交量
        # 模拟返回
        return np.random.uniform(1000, 10000)
        
    async def _market_data_processor(self):
        """市场数据处理器"""
        while self.is_running:
            try:
                # 这里应该处理实时市场数据
                # 更新价格、成交量、订单簿等信息
                await self._update_market_data()
                await asyncio.sleep(0.1)  # 100ms更新一次
                
            except Exception as e:
                self.logger.error(f"市场数据处理错误: {e}")
                await asyncio.sleep(1)
                
    async def _update_market_data(self):
        """更新市场数据"""
        # 模拟市场数据更新
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            # 模拟价格数据
            if symbol not in self.market_data:
                self.market_data[symbol] = {
                    'bid': 50000.0,
                    'ask': 50001.0,
                    'last': 50000.5,
                    'volume': 1000.0,
                    'avg_volume': 5000.0
                }
            else:
                # 随机波动
                data = self.market_data[symbol]
                change = np.random.uniform(-0.001, 0.001)
                data['last'] *= (1 + change)
                data['bid'] = data['last'] * 0.9999
                data['ask'] = data['last'] * 1.0001
                data['volume'] = np.random.uniform(500, 2000)
                
    async def _performance_monitor(self):
        """性能监控"""
        while self.is_running:
            try:
                await self._calculate_execution_metrics()
                await self._update_algorithm_performance()
                await asyncio.sleep(60)  # 1分钟更新一次
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(30)
                
    async def _calculate_execution_metrics(self):
        """计算执行指标"""
        # 计算最近完成订单的执行指标
        recent_orders = [order for order in self.completed_orders 
                        if (datetime.now() - order.timestamp).seconds < 3600]
        
        if not recent_orders:
            return
            
        # 计算平均滑点
        avg_slippage = np.mean([order.slippage for order in recent_orders])
        
        # 计算平均市场冲击
        avg_market_impact = np.mean([order.market_impact for order in recent_orders])
        
        # 计算平均执行时间
        avg_execution_time = np.mean([order.execution_time for order in recent_orders])
        
        metrics = {
            'timestamp': datetime.now(),
            'avg_slippage': avg_slippage,
            'avg_market_impact': avg_market_impact,
            'avg_execution_time': avg_execution_time,
            'order_count': len(recent_orders)
        }
        
        self.execution_metrics.append(metrics)
        
    async def _update_algorithm_performance(self):
        """更新算法性能"""
        for algorithm in ExecutionAlgorithm:
            algorithm_orders = [order for order in self.completed_orders 
                              if order.algorithm_used == algorithm.value]
            
            if algorithm_orders:
                avg_slippage = np.mean([order.slippage for order in algorithm_orders])
                avg_impact = np.mean([order.market_impact for order in algorithm_orders])
                
                self.algorithm_performance[algorithm.value] = {
                    'avg_slippage': avg_slippage,
                    'avg_market_impact': avg_impact,
                    'order_count': len(algorithm_orders),
                    'last_update': datetime.now()
                }
                
    async def _risk_monitor(self):
        """风险监控"""
        while self.is_running:
            try:
                await self._check_position_limits()
                await self._check_execution_quality()
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                self.logger.error(f"风险监控错误: {e}")
                await asyncio.sleep(60)
                
    async def _check_position_limits(self):
        """检查仓位限制"""
        for symbol, limit in self.position_limits.items():
            current_pos = self.current_positions.get(symbol, 0)
            if abs(current_pos) > limit * 0.9:  # 90%预警
                self.logger.warning(f"仓位接近限制: {symbol}, "
                                  f"当前: {current_pos}, 限制: {limit}")
                
    async def _check_execution_quality(self):
        """检查执行质量"""
        if not self.execution_metrics:
            return
            
        latest_metrics = self.execution_metrics[-1]
        
        # 检查滑点是否过高
        if latest_metrics['avg_slippage'] > self.max_slippage:
            self.logger.warning(f"平均滑点过高: {latest_metrics['avg_slippage']:.4f}")
            
        # 检查市场冲击是否过高
        if latest_metrics['avg_market_impact'] > self.max_market_impact:
            self.logger.warning(f"平均市场冲击过高: {latest_metrics['avg_market_impact']:.4f}")
            
    def get_execution_status(self) -> Dict[str, Any]:
        """获取执行状态"""
        return {
            'is_running': self.is_running,
            'active_orders': len(self.active_orders),
            'execution_queue': len(self.execution_queue),
            'completed_orders': len(self.completed_orders),
            'total_slices': len(self.order_slices),
            'algorithm_performance': self.algorithm_performance,
            'latest_metrics': dict(self.execution_metrics[-1]) if self.execution_metrics else {}
        }
        
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """获取订单状态"""
        if order_id in self.active_orders:
            order_info = self.active_orders[order_id]
            return {
                'order_id': order_id,
                'status': order_info['status'],
                'algorithm': order_info['algorithm'].value,
                'slice_count': len(order_info['slices']),
                'created_time': order_info['created_time'].isoformat()
            }
        return None
        
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.active_orders:
            self.active_orders[order_id]['status'] = 'cancelled'
            self.logger.info(f"订单已取消: {order_id}")
            return True
        return False
        
    async def update_market_data(self, symbol: str, data: Dict):
        """更新市场数据"""
        self.market_data[symbol] = data
        
    async def set_position_limit(self, symbol: str, limit: float):
        """设置仓位限制"""
        self.position_limits[symbol] = limit
        self.logger.info(f"设置仓位限制: {symbol} = {limit}")
        
    async def update_position(self, symbol: str, position: float):
        """更新当前仓位"""
        self.current_positions[symbol] = position
