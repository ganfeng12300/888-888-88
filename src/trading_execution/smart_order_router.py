"""
⚡ 智能订单路由系统
生产级多交易所订单路由，实现最优执行路径选择
支持实时流动性分析、订单分拆和交易成本最小化
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
import json

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.data_collection.exchange_connector import ExchangeConnector


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    ROUTING = "routing"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: str = ""
    max_slippage: float = 0.001  # 最大滑点 0.1%
    urgency: float = 0.5  # 紧急程度 0-1
    min_fill_size: float = 0.0  # 最小成交量
    max_participation_rate: float = 0.2  # 最大参与率
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExchangeLiquidity:
    """交易所流动性信息"""
    exchange: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    spread: float
    depth_score: float  # 深度评分
    latency_ms: float
    fee_rate: float
    last_update: float
    available_balance: float = 0.0


@dataclass
class RouteSegment:
    """路由片段"""
    exchange: str
    quantity: float
    price: float
    expected_cost: float
    expected_slippage: float
    priority: float
    estimated_fill_time: float


@dataclass
class ExecutionRoute:
    """执行路由"""
    order_id: str
    segments: List[RouteSegment]
    total_quantity: float
    total_cost: float
    expected_slippage: float
    expected_duration: float
    confidence_score: float
    created_time: float = field(default_factory=time.time)


class SmartOrderRouter:
    """智能订单路由器"""
    
    def __init__(self, exchange_connectors: Dict[str, ExchangeConnector]):
        self.exchange_connectors = exchange_connectors
        self.liquidity_cache: Dict[str, Dict[str, ExchangeLiquidity]] = {}
        self.active_routes: Dict[str, ExecutionRoute] = {}
        self.routing_history: deque = deque(maxlen=1000)
        
        # 路由配置
        self.max_route_segments = 5
        self.liquidity_refresh_interval = 1.0  # 秒
        self.route_cache_ttl = 5.0  # 路由缓存时间
        self.min_segment_size = 0.001  # 最小分片大小
        
        # 成本模型参数
        self.market_impact_factor = 0.0001  # 市场冲击系数
        self.latency_penalty = 0.00001  # 延迟惩罚
        self.spread_weight = 0.5  # 价差权重
        self.depth_weight = 0.3  # 深度权重
        self.fee_weight = 0.2  # 手续费权重
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.ORDER_ROUTING, [5, 6, 7, 8])
        
        # 启动流动性监控
        self.monitoring_active = False
        self._start_liquidity_monitoring()
        
        logger.info("智能订单路由器初始化完成")
    
    def _start_liquidity_monitoring(self):
        """启动流动性监控"""
        self.monitoring_active = True
        
        def monitor_liquidity():
            while self.monitoring_active:
                try:
                    asyncio.run(self._update_all_liquidity())
                    time.sleep(self.liquidity_refresh_interval)
                except Exception as e:
                    logger.error(f"流动性监控出错: {e}")
                    time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_liquidity, daemon=True)
        monitor_thread.start()
        
        logger.info("流动性监控已启动")
    
    async def _update_all_liquidity(self):
        """更新所有交易所流动性"""
        tasks = []
        
        for exchange_name, connector in self.exchange_connectors.items():
            # 获取该交易所的所有交易对
            symbols = await connector.get_trading_symbols()
            
            for symbol in symbols[:10]:  # 限制前10个主要交易对
                task = self._update_exchange_liquidity(exchange_name, symbol, connector)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _update_exchange_liquidity(self, exchange: str, symbol: str, connector: ExchangeConnector):
        """更新单个交易所流动性"""
        try:
            # 获取订单簿
            orderbook = await connector.get_orderbook(symbol, limit=20)
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return
            
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            if not bids or not asks:
                return
            
            # 计算流动性指标
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            bid_volume = sum(float(bid[1]) for bid in bids[:5])  # 前5档买量
            ask_volume = sum(float(ask[1]) for ask in asks[:5])  # 前5档卖量
            
            spread = (best_ask - best_bid) / best_bid
            
            # 计算深度评分 (基于前10档总量)
            total_bid_volume = sum(float(bid[1]) for bid in bids[:10])
            total_ask_volume = sum(float(ask[1]) for ask in asks[:10])
            depth_score = min(total_bid_volume, total_ask_volume)
            
            # 获取延迟信息
            latency_ms = await self._measure_exchange_latency(connector)
            
            # 获取手续费率
            fee_rate = await connector.get_trading_fee(symbol)
            
            # 获取可用余额
            balance = await connector.get_balance()
            available_balance = 0.0
            if balance and 'free' in balance:
                base_asset = symbol.split('/')[0]
                quote_asset = symbol.split('/')[1]
                available_balance = balance['free'].get(quote_asset, 0.0)
            
            # 创建流动性对象
            liquidity = ExchangeLiquidity(
                exchange=exchange,
                symbol=symbol,
                bid_price=best_bid,
                ask_price=best_ask,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                spread=spread,
                depth_score=depth_score,
                latency_ms=latency_ms,
                fee_rate=fee_rate,
                last_update=time.time(),
                available_balance=available_balance
            )
            
            # 更新缓存
            if exchange not in self.liquidity_cache:
                self.liquidity_cache[exchange] = {}
            
            self.liquidity_cache[exchange][symbol] = liquidity
            
        except Exception as e:
            logger.error(f"更新流动性失败 {exchange}-{symbol}: {e}")
    
    async def _measure_exchange_latency(self, connector: ExchangeConnector) -> float:
        """测量交易所延迟"""
        try:
            start_time = time.time()
            await connector.get_server_time()
            end_time = time.time()
            return (end_time - start_time) * 1000  # 转换为毫秒
        except:
            return 100.0  # 默认延迟
    
    def get_available_liquidity(self, symbol: str, side: OrderSide) -> List[ExchangeLiquidity]:
        """获取可用流动性"""
        available_liquidity = []
        
        for exchange, symbols_data in self.liquidity_cache.items():
            if symbol in symbols_data:
                liquidity = symbols_data[symbol]
                
                # 检查流动性是否新鲜
                if time.time() - liquidity.last_update < 10.0:  # 10秒内的数据
                    # 检查是否有足够的流动性
                    if side == OrderSide.BUY and liquidity.ask_volume > 0:
                        available_liquidity.append(liquidity)
                    elif side == OrderSide.SELL and liquidity.bid_volume > 0:
                        available_liquidity.append(liquidity)
        
        return available_liquidity
    
    def calculate_execution_cost(self, liquidity: ExchangeLiquidity, quantity: float, 
                                side: OrderSide) -> Tuple[float, float]:
        """计算执行成本和滑点"""
        try:
            if side == OrderSide.BUY:
                base_price = liquidity.ask_price
                available_volume = liquidity.ask_volume
            else:
                base_price = liquidity.bid_price
                available_volume = liquidity.bid_volume
            
            # 计算市场冲击
            participation_rate = min(quantity / available_volume, 1.0) if available_volume > 0 else 1.0
            market_impact = self.market_impact_factor * (participation_rate ** 0.5)
            
            # 计算滑点
            slippage = liquidity.spread / 2 + market_impact
            
            # 计算总成本
            execution_price = base_price * (1 + slippage if side == OrderSide.BUY else 1 - slippage)
            trading_fee = execution_price * quantity * liquidity.fee_rate
            latency_cost = liquidity.latency_ms * self.latency_penalty * quantity
            
            total_cost = execution_price * quantity + trading_fee + latency_cost
            
            return total_cost, slippage
            
        except Exception as e:
            logger.error(f"计算执行成本失败: {e}")
            return float('inf'), 1.0
    
    def calculate_route_priority(self, segment: RouteSegment, urgency: float) -> float:
        """计算路由优先级"""
        # 基础优先级 (成本越低优先级越高)
        cost_priority = 1.0 / (1.0 + segment.expected_cost)
        
        # 滑点优先级 (滑点越低优先级越高)
        slippage_priority = 1.0 / (1.0 + segment.expected_slippage * 100)
        
        # 时间优先级 (填充时间越短优先级越高)
        time_priority = 1.0 / (1.0 + segment.estimated_fill_time)
        
        # 紧急程度调整
        urgency_weight = 0.3 + urgency * 0.4  # 0.3-0.7
        cost_weight = 0.7 - urgency * 0.4     # 0.3-0.7
        
        # 综合优先级
        priority = (cost_priority * cost_weight + 
                   slippage_priority * 0.2 + 
                   time_priority * urgency_weight)
        
        return priority
    
    async def find_optimal_route(self, order: OrderRequest) -> Optional[ExecutionRoute]:
        """寻找最优执行路由"""
        try:
            # 获取可用流动性
            available_liquidity = self.get_available_liquidity(order.symbol, order.side)
            
            if not available_liquidity:
                logger.warning(f"没有可用流动性: {order.symbol}")
                return None
            
            # 生成路由片段
            route_segments = []
            remaining_quantity = order.quantity
            
            # 按优先级排序流动性
            liquidity_scores = []
            for liquidity in available_liquidity:
                cost, slippage = self.calculate_execution_cost(liquidity, remaining_quantity, order.side)
                
                # 检查余额是否足够
                if order.side == OrderSide.BUY:
                    required_balance = cost
                else:
                    required_balance = remaining_quantity
                
                if liquidity.available_balance < required_balance:
                    continue
                
                # 计算可执行数量
                max_quantity = min(
                    remaining_quantity,
                    liquidity.ask_volume if order.side == OrderSide.BUY else liquidity.bid_volume,
                    remaining_quantity * order.max_participation_rate
                )
                
                if max_quantity < self.min_segment_size:
                    continue
                
                # 创建路由片段
                segment = RouteSegment(
                    exchange=liquidity.exchange,
                    quantity=max_quantity,
                    price=liquidity.ask_price if order.side == OrderSide.BUY else liquidity.bid_price,
                    expected_cost=cost,
                    expected_slippage=slippage,
                    priority=0.0,  # 稍后计算
                    estimated_fill_time=liquidity.latency_ms / 1000.0
                )
                
                segment.priority = self.calculate_route_priority(segment, order.urgency)
                liquidity_scores.append((segment.priority, segment))
            
            # 按优先级排序
            liquidity_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 贪心算法选择最优路由
            for priority, segment in liquidity_scores:
                if remaining_quantity <= 0:
                    break
                
                if len(route_segments) >= self.max_route_segments:
                    break
                
                # 调整片段数量
                segment.quantity = min(segment.quantity, remaining_quantity)
                route_segments.append(segment)
                remaining_quantity -= segment.quantity
            
            if remaining_quantity > order.quantity * 0.01:  # 如果还有超过1%未分配
                logger.warning(f"无法完全路由订单，剩余: {remaining_quantity}")
            
            if not route_segments:
                return None
            
            # 创建执行路由
            total_cost = sum(seg.expected_cost for seg in route_segments)
            total_quantity = sum(seg.quantity for seg in route_segments)
            avg_slippage = sum(seg.expected_slippage * seg.quantity for seg in route_segments) / total_quantity
            max_duration = max(seg.estimated_fill_time for seg in route_segments)
            
            # 计算置信度分数
            confidence_score = min(1.0, total_quantity / order.quantity)
            
            route = ExecutionRoute(
                order_id=order.client_order_id or f"route_{int(time.time() * 1000)}",
                segments=route_segments,
                total_quantity=total_quantity,
                total_cost=total_cost,
                expected_slippage=avg_slippage,
                expected_duration=max_duration,
                confidence_score=confidence_score
            )
            
            # 缓存路由
            self.active_routes[route.order_id] = route
            
            logger.info(f"找到最优路由: {len(route_segments)}个片段, "
                       f"总量={total_quantity:.4f}, 预期滑点={avg_slippage:.4f}")
            
            return route
            
        except Exception as e:
            logger.error(f"寻找最优路由失败: {e}")
            return None
    
    async def optimize_route_timing(self, route: ExecutionRoute, 
                                   market_conditions: Dict[str, Any]) -> ExecutionRoute:
        """优化路由时机"""
        try:
            # 分析市场条件
            volatility = market_conditions.get('volatility', 0.02)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            volume_profile = market_conditions.get('volume_profile', 1.0)
            
            # 调整执行时机
            optimized_segments = []
            
            for segment in route.segments:
                # 根据市场条件调整执行时间
                if volatility > 0.05:  # 高波动性
                    segment.estimated_fill_time *= 0.8  # 加快执行
                elif volatility < 0.01:  # 低波动性
                    segment.estimated_fill_time *= 1.2  # 可以稍慢执行
                
                # 根据趋势调整
                if abs(trend_strength) > 0.5:  # 强趋势
                    if (trend_strength > 0 and route.segments[0].quantity > 0) or \
                       (trend_strength < 0 and route.segments[0].quantity < 0):
                        segment.priority *= 1.1  # 顺势交易提高优先级
                
                optimized_segments.append(segment)
            
            # 重新排序片段
            optimized_segments.sort(key=lambda x: x.priority, reverse=True)
            
            # 更新路由
            route.segments = optimized_segments
            route.expected_duration = max(seg.estimated_fill_time for seg in optimized_segments)
            
            logger.info(f"路由时机优化完成: {route.order_id}")
            
            return route
            
        except Exception as e:
            logger.error(f"优化路由时机失败: {e}")
            return route
    
    def validate_route(self, route: ExecutionRoute, order: OrderRequest) -> bool:
        """验证路由有效性"""
        try:
            # 检查总量
            if route.total_quantity < order.quantity * 0.95:  # 至少95%
                logger.warning(f"路由数量不足: {route.total_quantity} < {order.quantity}")
                return False
            
            # 检查滑点
            if route.expected_slippage > order.max_slippage:
                logger.warning(f"预期滑点过高: {route.expected_slippage} > {order.max_slippage}")
                return False
            
            # 检查片段有效性
            for segment in route.segments:
                if segment.quantity < self.min_segment_size:
                    logger.warning(f"片段数量过小: {segment.quantity}")
                    return False
                
                if segment.exchange not in self.exchange_connectors:
                    logger.warning(f"无效交易所: {segment.exchange}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证路由失败: {e}")
            return False
    
    async def route_order(self, order: OrderRequest) -> Optional[ExecutionRoute]:
        """路由订单"""
        try:
            logger.info(f"开始路由订单: {order.symbol} {order.side.value} {order.quantity}")
            
            # 寻找最优路由
            route = await self.find_optimal_route(order)
            if not route:
                logger.error("未找到可用路由")
                return None
            
            # 获取市场条件
            market_conditions = await self._analyze_market_conditions(order.symbol)
            
            # 优化路由时机
            route = await self.optimize_route_timing(route, market_conditions)
            
            # 验证路由
            if not self.validate_route(route, order):
                logger.error("路由验证失败")
                return None
            
            # 记录路由历史
            self.routing_history.append({
                'timestamp': time.time(),
                'order_id': route.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'segments': len(route.segments),
                'expected_slippage': route.expected_slippage,
                'confidence_score': route.confidence_score
            })
            
            logger.info(f"订单路由完成: {route.order_id}, "
                       f"片段数={len(route.segments)}, 置信度={route.confidence_score:.3f}")
            
            return route
            
        except Exception as e:
            logger.error(f"路由订单失败: {e}")
            return None
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """分析市场条件"""
        try:
            # 获取最近价格数据
            price_data = []
            volume_data = []
            
            for exchange, connector in self.exchange_connectors.items():
                try:
                    klines = await connector.get_klines(symbol, '1m', limit=20)
                    if klines:
                        prices = [float(k[4]) for k in klines]  # 收盘价
                        volumes = [float(k[5]) for k in klines]  # 成交量
                        price_data.extend(prices)
                        volume_data.extend(volumes)
                except:
                    continue
            
            if not price_data:
                return {'volatility': 0.02, 'trend_strength': 0.0, 'volume_profile': 1.0}
            
            # 计算波动率
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.02
            
            # 计算趋势强度
            if len(price_data) >= 10:
                trend_strength = (price_data[-1] - price_data[-10]) / price_data[-10]
            else:
                trend_strength = 0.0
            
            # 计算成交量概况
            if volume_data:
                recent_volume = np.mean(volume_data[-5:]) if len(volume_data) >= 5 else volume_data[-1]
                avg_volume = np.mean(volume_data)
                volume_profile = recent_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_profile = 1.0
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logger.error(f"分析市场条件失败: {e}")
            return {'volatility': 0.02, 'trend_strength': 0.0, 'volume_profile': 1.0}
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计"""
        if not self.routing_history:
            return {}
        
        history = list(self.routing_history)
        
        return {
            'total_routes': len(history),
            'avg_segments': np.mean([h['segments'] for h in history]),
            'avg_slippage': np.mean([h['expected_slippage'] for h in history]),
            'avg_confidence': np.mean([h['confidence_score'] for h in history]),
            'success_rate': len([h for h in history if h['confidence_score'] > 0.9]) / len(history),
            'active_routes': len(self.active_routes),
            'cached_liquidity': sum(len(symbols) for symbols in self.liquidity_cache.values())
        }
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info("订单路由监控已停止")


# 全局路由器实例
smart_router = None


def create_smart_router(exchange_connectors: Dict[str, ExchangeConnector]) -> SmartOrderRouter:
    """创建智能路由器"""
    global smart_router
    smart_router = SmartOrderRouter(exchange_connectors)
    return smart_router


async def main():
    """测试主函数"""
    logger.info("启动智能订单路由器测试...")
    
    # 模拟交易所连接器
    mock_connectors = {
        'binance': None,  # 实际应该是真实的连接器
        'okx': None,
        'bybit': None
    }
    
    # 创建路由器
    router = create_smart_router(mock_connectors)
    
    try:
        # 模拟订单
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            max_slippage=0.002,
            urgency=0.7,
            client_order_id="test_order_001"
        )
        
        # 路由订单
        route = await router.route_order(order)
        
        if route:
            logger.info(f"路由成功: {route.order_id}")
            logger.info(f"片段数: {len(route.segments)}")
            logger.info(f"预期滑点: {route.expected_slippage:.4f}")
            logger.info(f"置信度: {route.confidence_score:.3f}")
        else:
            logger.error("路由失败")
        
        # 获取统计信息
        stats = router.get_routing_statistics()
        logger.info(f"路由统计: {stats}")
        
        # 运行一段时间
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        router.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())

