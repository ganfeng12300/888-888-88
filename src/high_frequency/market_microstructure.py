"""
📊 市场微观结构分析器
生产级市场微观结构分析系统，实现订单簿分析、流动性预测、价格影响模型等完整功能
支持实时订单流分析、市场深度评估和交易成本预测
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings
from src.high_frequency.low_latency_engine import MarketData, OrderBook


class LiquidityType(Enum):
    """流动性类型"""
    HIGH = "high"           # 高流动性
    MEDIUM = "medium"       # 中等流动性
    LOW = "low"             # 低流动性
    ILLIQUID = "illiquid"   # 非流动性


class OrderFlowType(Enum):
    """订单流类型"""
    AGGRESSIVE_BUY = "aggressive_buy"       # 激进买入
    AGGRESSIVE_SELL = "aggressive_sell"     # 激进卖出
    PASSIVE_BUY = "passive_buy"             # 被动买入
    PASSIVE_SELL = "passive_sell"           # 被动卖出
    NEUTRAL = "neutral"                     # 中性


@dataclass
class OrderBookLevel:
    """订单簿档位"""
    price: float                    # 价格
    size: float                     # 数量
    orders: int                     # 订单数
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str                     # 交易对
    bids: List[OrderBookLevel]      # 买盘
    asks: List[OrderBookLevel]      # 卖盘
    mid_price: float                # 中间价
    spread: float                   # 价差
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiquidityMetrics:
    """流动性指标"""
    symbol: str                     # 交易对
    bid_ask_spread: float           # 买卖价差
    effective_spread: float         # 有效价差
    quoted_spread: float            # 报价价差
    depth_at_best: float            # 最优价位深度
    depth_5_levels: float           # 5档深度
    depth_10_levels: float          # 10档深度
    market_impact_1pct: float       # 1%市场影响
    market_impact_5pct: float       # 5%市场影响
    liquidity_ratio: float          # 流动性比率
    turnover_rate: float            # 换手率
    price_volatility: float         # 价格波动率
    liquidity_type: LiquidityType   # 流动性类型
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderFlowMetrics:
    """订单流指标"""
    symbol: str                     # 交易对
    order_flow_imbalance: float     # 订单流不平衡
    trade_size_ratio: float         # 交易规模比率
    aggressive_ratio: float         # 激进交易比率
    order_arrival_rate: float       # 订单到达率
    cancellation_rate: float        # 取消率
    fill_rate: float                # 成交率
    average_trade_size: float       # 平均交易规模
    large_trade_ratio: float        # 大单比率
    flow_toxicity: float            # 流动性毒性
    informed_trading_prob: float    # 知情交易概率
    timestamp: float = field(default_factory=time.time)


@dataclass
class PriceImpactModel:
    """价格影响模型"""
    symbol: str                     # 交易对
    linear_impact: float            # 线性影响系数
    sqrt_impact: float              # 平方根影响系数
    permanent_impact: float         # 永久影响
    temporary_impact: float         # 临时影响
    resilience_time: float          # 恢复时间
    impact_decay: float             # 影响衰减
    confidence_interval: Tuple[float, float]  # 置信区间
    r_squared: float                # 拟合优度
    timestamp: float = field(default_factory=time.time)


class MarketMicrostructureAnalyzer:
    """市场微观结构分析器"""
    
    def __init__(self):
        # 数据存储
        self.order_book_snapshots: Dict[str, deque] = {}
        self.trade_data: Dict[str, deque] = {}
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        self.order_flow_metrics: Dict[str, OrderFlowMetrics] = {}
        self.price_impact_models: Dict[str, PriceImpactModel] = {}
        
        # 分析参数
        self.snapshot_window = 1000     # 快照窗口大小
        self.trade_window = 5000        # 交易窗口大小
        self.analysis_interval = 1.0    # 分析间隔(秒)
        self.min_data_points = 100      # 最小数据点数
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MicrostructureAnalyzer")
        
        # 运行状态
        self.running = False
        self.analysis_thread = None
        
        logger.info("市场微观结构分析器初始化完成")
    
    async def update_order_book(self, symbol: str, order_book: OrderBook):
        """更新订单簿数据"""
        try:
            # 获取订单簿快照
            depth_data = order_book.get_depth(levels=20)
            
            # 创建快照
            bids = []
            asks = []
            
            for price, size in depth_data['bids']:
                bids.append(OrderBookLevel(price=price, size=size, orders=1))
            
            for price, size in depth_data['asks']:
                asks.append(OrderBookLevel(price=price, size=size, orders=1))
            
            if bids and asks:
                mid_price = (bids[0].price + asks[0].price) / 2
                spread = asks[0].price - bids[0].price
                
                snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    mid_price=mid_price,
                    spread=spread
                )
                
                # 存储快照
                if symbol not in self.order_book_snapshots:
                    self.order_book_snapshots[symbol] = deque(maxlen=self.snapshot_window)
                
                self.order_book_snapshots[symbol].append(snapshot)
                
                # 触发分析
                await self._analyze_liquidity(symbol)
                
        except Exception as e:
            logger.error(f"更新订单簿失败: {e}")
    
    async def update_trade_data(self, symbol: str, trade_data: Dict[str, Any]):
        """更新交易数据"""
        try:
            if symbol not in self.trade_data:
                self.trade_data[symbol] = deque(maxlen=self.trade_window)
            
            # 添加时间戳
            trade_data['timestamp'] = time.time()
            self.trade_data[symbol].append(trade_data)
            
            # 触发订单流分析
            await self._analyze_order_flow(symbol)
            
        except Exception as e:
            logger.error(f"更新交易数据失败: {e}")
    
    async def _analyze_liquidity(self, symbol: str):
        """分析流动性"""
        try:
            if symbol not in self.order_book_snapshots:
                return
            
            snapshots = list(self.order_book_snapshots[symbol])
            if len(snapshots) < self.min_data_points:
                return
            
            # 获取最新快照
            latest_snapshot = snapshots[-1]
            
            # 计算基本价差指标
            bid_ask_spread = latest_snapshot.spread
            relative_spread = bid_ask_spread / latest_snapshot.mid_price
            
            # 计算深度指标
            depth_at_best = latest_snapshot.bids[0].size + latest_snapshot.asks[0].size
            
            depth_5_levels = sum(level.size for level in latest_snapshot.bids[:5])
            depth_5_levels += sum(level.size for level in latest_snapshot.asks[:5])
            
            depth_10_levels = sum(level.size for level in latest_snapshot.bids[:10])
            depth_10_levels += sum(level.size for level in latest_snapshot.asks[:10])
            
            # 计算市场影响
            market_impact_1pct = self._calculate_market_impact(latest_snapshot, 0.01)
            market_impact_5pct = self._calculate_market_impact(latest_snapshot, 0.05)
            
            # 计算流动性比率
            total_depth = depth_10_levels
            liquidity_ratio = total_depth / latest_snapshot.mid_price if latest_snapshot.mid_price > 0 else 0
            
            # 计算价格波动率
            if len(snapshots) >= 100:
                mid_prices = [s.mid_price for s in snapshots[-100:]]
                returns = np.diff(np.log(mid_prices))
                price_volatility = np.std(returns) * np.sqrt(252 * 24 * 3600)  # 年化波动率
            else:
                price_volatility = 0.0
            
            # 计算换手率 (简化)
            turnover_rate = 0.0
            if symbol in self.trade_data and len(self.trade_data[symbol]) > 0:
                recent_trades = list(self.trade_data[symbol])[-100:]
                total_volume = sum(trade.get('volume', 0) for trade in recent_trades)
                turnover_rate = total_volume / (total_depth + 1e-8)
            
            # 确定流动性类型
            liquidity_type = self._classify_liquidity(
                relative_spread, depth_at_best, market_impact_1pct
            )
            
            # 创建流动性指标
            metrics = LiquidityMetrics(
                symbol=symbol,
                bid_ask_spread=bid_ask_spread,
                effective_spread=bid_ask_spread,  # 简化
                quoted_spread=bid_ask_spread,
                depth_at_best=depth_at_best,
                depth_5_levels=depth_5_levels,
                depth_10_levels=depth_10_levels,
                market_impact_1pct=market_impact_1pct,
                market_impact_5pct=market_impact_5pct,
                liquidity_ratio=liquidity_ratio,
                turnover_rate=turnover_rate,
                price_volatility=price_volatility,
                liquidity_type=liquidity_type
            )
            
            self.liquidity_metrics[symbol] = metrics
            
        except Exception as e:
            logger.error(f"流动性分析失败: {e}")
    
    def _calculate_market_impact(self, snapshot: OrderBookSnapshot, percentage: float) -> float:
        """计算市场影响"""
        try:
            target_volume = snapshot.mid_price * percentage
            
            # 计算买入影响
            buy_impact = 0.0
            remaining_volume = target_volume
            
            for ask in snapshot.asks:
                if remaining_volume <= 0:
                    break
                
                volume_at_level = min(remaining_volume, ask.size * ask.price)
                buy_impact += (ask.price - snapshot.mid_price) * volume_at_level
                remaining_volume -= volume_at_level
            
            # 计算卖出影响
            sell_impact = 0.0
            remaining_volume = target_volume
            
            for bid in snapshot.bids:
                if remaining_volume <= 0:
                    break
                
                volume_at_level = min(remaining_volume, bid.size * bid.price)
                sell_impact += (snapshot.mid_price - bid.price) * volume_at_level
                remaining_volume -= volume_at_level
            
            # 返回平均影响
            avg_impact = (buy_impact + sell_impact) / (2 * target_volume) if target_volume > 0 else 0
            return avg_impact
            
        except Exception as e:
            logger.error(f"计算市场影响失败: {e}")
            return 0.0
    
    def _classify_liquidity(
        self, 
        relative_spread: float, 
        depth_at_best: float, 
        market_impact: float
    ) -> LiquidityType:
        """分类流动性类型"""
        # 基于多个指标的综合评估
        score = 0
        
        # 价差评分
        if relative_spread < 0.001:
            score += 3
        elif relative_spread < 0.005:
            score += 2
        elif relative_spread < 0.01:
            score += 1
        
        # 深度评分
        if depth_at_best > 1000:
            score += 3
        elif depth_at_best > 100:
            score += 2
        elif depth_at_best > 10:
            score += 1
        
        # 市场影响评分
        if market_impact < 0.0001:
            score += 3
        elif market_impact < 0.001:
            score += 2
        elif market_impact < 0.01:
            score += 1
        
        # 分类
        if score >= 7:
            return LiquidityType.HIGH
        elif score >= 4:
            return LiquidityType.MEDIUM
        elif score >= 2:
            return LiquidityType.LOW
        else:
            return LiquidityType.ILLIQUID
    
    async def _analyze_order_flow(self, symbol: str):
        """分析订单流"""
        try:
            if symbol not in self.trade_data:
                return
            
            trades = list(self.trade_data[symbol])
            if len(trades) < self.min_data_points:
                return
            
            # 计算订单流不平衡
            buy_volume = sum(trade.get('volume', 0) for trade in trades if trade.get('side') == 'buy')
            sell_volume = sum(trade.get('volume', 0) for trade in trades if trade.get('side') == 'sell')
            total_volume = buy_volume + sell_volume
            
            order_flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # 计算交易规模统计
            trade_sizes = [trade.get('volume', 0) for trade in trades]
            average_trade_size = np.mean(trade_sizes) if trade_sizes else 0
            
            # 计算大单比率 (假设大单为平均规模的3倍以上)
            large_trade_threshold = average_trade_size * 3
            large_trades = [size for size in trade_sizes if size > large_trade_threshold]
            large_trade_ratio = len(large_trades) / len(trade_sizes) if trade_sizes else 0
            
            # 计算激进交易比率 (简化：假设市价单为激进交易)
            aggressive_trades = [trade for trade in trades if trade.get('type') == 'market']
            aggressive_ratio = len(aggressive_trades) / len(trades) if trades else 0
            
            # 计算订单到达率
            if len(trades) >= 2:
                time_span = trades[-1]['timestamp'] - trades[0]['timestamp']
                order_arrival_rate = len(trades) / time_span if time_span > 0 else 0
            else:
                order_arrival_rate = 0
            
            # 计算流动性毒性 (基于价格影响的持续性)
            flow_toxicity = self._calculate_flow_toxicity(trades)
            
            # 计算知情交易概率 (PIN模型简化版)
            informed_trading_prob = self._calculate_pin_probability(trades)
            
            # 创建订单流指标
            metrics = OrderFlowMetrics(
                symbol=symbol,
                order_flow_imbalance=order_flow_imbalance,
                trade_size_ratio=average_trade_size / np.median(trade_sizes) if trade_sizes else 1.0,
                aggressive_ratio=aggressive_ratio,
                order_arrival_rate=order_arrival_rate,
                cancellation_rate=0.0,  # 需要订单数据
                fill_rate=1.0,          # 简化
                average_trade_size=average_trade_size,
                large_trade_ratio=large_trade_ratio,
                flow_toxicity=flow_toxicity,
                informed_trading_prob=informed_trading_prob
            )
            
            self.order_flow_metrics[symbol] = metrics
            
        except Exception as e:
            logger.error(f"订单流分析失败: {e}")
    
    def _calculate_flow_toxicity(self, trades: List[Dict[str, Any]]) -> float:
        """计算流动性毒性"""
        try:
            if len(trades) < 10:
                return 0.0
            
            # 简化的毒性计算：基于交易后价格变化的持续性
            price_changes = []
            for i in range(1, len(trades)):
                if 'price' in trades[i] and 'price' in trades[i-1]:
                    change = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    price_changes.append(change)
            
            if len(price_changes) < 5:
                return 0.0
            
            # 计算价格变化的自相关性
            changes_array = np.array(price_changes)
            if len(changes_array) > 1:
                correlation = np.corrcoef(changes_array[:-1], changes_array[1:])[0, 1]
                toxicity = max(0, correlation)  # 正相关表示毒性
            else:
                toxicity = 0.0
            
            return toxicity
            
        except Exception as e:
            logger.error(f"计算流动性毒性失败: {e}")
            return 0.0
    
    def _calculate_pin_probability(self, trades: List[Dict[str, Any]]) -> float:
        """计算知情交易概率 (PIN模型简化版)"""
        try:
            if len(trades) < 20:
                return 0.0
            
            # 简化的PIN计算
            buy_trades = [t for t in trades if t.get('side') == 'buy']
            sell_trades = [t for t in trades if t.get('side') == 'sell']
            
            buy_rate = len(buy_trades) / len(trades)
            sell_rate = len(sell_trades) / len(trades)
            
            # 基于交易不平衡计算知情交易概率
            imbalance = abs(buy_rate - sell_rate)
            pin_probability = min(imbalance * 2, 1.0)  # 简化公式
            
            return pin_probability
            
        except Exception as e:
            logger.error(f"计算PIN概率失败: {e}")
            return 0.0
    
    async def build_price_impact_model(self, symbol: str) -> Optional[PriceImpactModel]:
        """构建价格影响模型"""
        try:
            if symbol not in self.trade_data or len(self.trade_data[symbol]) < 200:
                return None
            
            trades = list(self.trade_data[symbol])
            
            # 准备数据
            volumes = []
            price_impacts = []
            
            for i in range(1, len(trades)):
                if 'volume' in trades[i] and 'price' in trades[i] and 'price' in trades[i-1]:
                    volume = trades[i]['volume']
                    price_impact = abs(trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    
                    volumes.append(volume)
                    price_impacts.append(price_impact)
            
            if len(volumes) < 50:
                return None
            
            volumes = np.array(volumes)
            price_impacts = np.array(price_impacts)
            
            # 线性回归拟合
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # 线性模型: impact = a * volume
            X_linear = volumes.reshape(-1, 1)
            linear_model = LinearRegression().fit(X_linear, price_impacts)
            linear_impact = linear_model.coef_[0]
            linear_r2 = r2_score(price_impacts, linear_model.predict(X_linear))
            
            # 平方根模型: impact = a * sqrt(volume)
            X_sqrt = np.sqrt(volumes).reshape(-1, 1)
            sqrt_model = LinearRegression().fit(X_sqrt, price_impacts)
            sqrt_impact = sqrt_model.coef_[0]
            sqrt_r2 = r2_score(price_impacts, sqrt_model.predict(X_sqrt))
            
            # 选择更好的模型
            if sqrt_r2 > linear_r2:
                best_impact = sqrt_impact
                best_r2 = sqrt_r2
            else:
                best_impact = linear_impact
                best_r2 = linear_r2
            
            # 计算置信区间 (简化)
            impact_std = np.std(price_impacts)
            confidence_interval = (
                best_impact - 1.96 * impact_std,
                best_impact + 1.96 * impact_std
            )
            
            # 创建价格影响模型
            model = PriceImpactModel(
                symbol=symbol,
                linear_impact=linear_impact,
                sqrt_impact=sqrt_impact,
                permanent_impact=best_impact * 0.6,  # 假设60%为永久影响
                temporary_impact=best_impact * 0.4,  # 假设40%为临时影响
                resilience_time=300.0,  # 5分钟恢复时间
                impact_decay=0.1,       # 10%衰减率
                confidence_interval=confidence_interval,
                r_squared=best_r2
            )
            
            self.price_impact_models[symbol] = model
            return model
            
        except Exception as e:
            logger.error(f"构建价格影响模型失败: {e}")
            return None
    
    def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """获取流动性指标"""
        return self.liquidity_metrics.get(symbol)
    
    def get_order_flow_metrics(self, symbol: str) -> Optional[OrderFlowMetrics]:
        """获取订单流指标"""
        return self.order_flow_metrics.get(symbol)
    
    def get_price_impact_model(self, symbol: str) -> Optional[PriceImpactModel]:
        """获取价格影响模型"""
        return self.price_impact_models.get(symbol)
    
    def predict_market_impact(self, symbol: str, volume: float) -> float:
        """预测市场影响"""
        try:
            model = self.price_impact_models.get(symbol)
            if not model:
                return 0.0
            
            # 使用平方根模型预测
            predicted_impact = model.sqrt_impact * np.sqrt(volume)
            return predicted_impact
            
        except Exception as e:
            logger.error(f"预测市场影响失败: {e}")
            return 0.0
    
    def estimate_execution_cost(
        self, 
        symbol: str, 
        volume: float, 
        urgency: float = 0.5
    ) -> Dict[str, float]:
        """估算执行成本"""
        try:
            liquidity_metrics = self.liquidity_metrics.get(symbol)
            impact_model = self.price_impact_models.get(symbol)
            
            if not liquidity_metrics or not impact_model:
                return {'total_cost': 0.0, 'spread_cost': 0.0, 'impact_cost': 0.0}
            
            # 价差成本
            spread_cost = liquidity_metrics.bid_ask_spread / 2
            
            # 市场影响成本
            impact_cost = self.predict_market_impact(symbol, volume)
            
            # 紧急性调整
            urgency_multiplier = 1 + urgency
            impact_cost *= urgency_multiplier
            
            # 总成本
            total_cost = spread_cost + impact_cost
            
            return {
                'total_cost': total_cost,
                'spread_cost': spread_cost,
                'impact_cost': impact_cost,
                'urgency_multiplier': urgency_multiplier
            }
            
        except Exception as e:
            logger.error(f"估算执行成本失败: {e}")
            return {'total_cost': 0.0, 'spread_cost': 0.0, 'impact_cost': 0.0}
    
    async def get_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """获取分析摘要"""
        try:
            summary = {
                'symbol': symbol,
                'timestamp': time.time(),
                'liquidity_metrics': None,
                'order_flow_metrics': None,
                'price_impact_model': None,
                'data_quality': {}
            }
            
            # 流动性指标
            if symbol in self.liquidity_metrics:
                metrics = self.liquidity_metrics[symbol]
                summary['liquidity_metrics'] = {
                    'liquidity_type': metrics.liquidity_type.value,
                    'bid_ask_spread': metrics.bid_ask_spread,
                    'depth_at_best': metrics.depth_at_best,
                    'market_impact_1pct': metrics.market_impact_1pct,
                    'price_volatility': metrics.price_volatility
                }
            
            # 订单流指标
            if symbol in self.order_flow_metrics:
                metrics = self.order_flow_metrics[symbol]
                summary['order_flow_metrics'] = {
                    'order_flow_imbalance': metrics.order_flow_imbalance,
                    'aggressive_ratio': metrics.aggressive_ratio,
                    'flow_toxicity': metrics.flow_toxicity,
                    'informed_trading_prob': metrics.informed_trading_prob
                }
            
            # 价格影响模型
            if symbol in self.price_impact_models:
                model = self.price_impact_models[symbol]
                summary['price_impact_model'] = {
                    'sqrt_impact': model.sqrt_impact,
                    'permanent_impact': model.permanent_impact,
                    'r_squared': model.r_squared
                }
            
            # 数据质量
            summary['data_quality'] = {
                'order_book_snapshots': len(self.order_book_snapshots.get(symbol, [])),
                'trade_records': len(self.trade_data.get(symbol, [])),
                'analysis_coverage': time.time() - (
                    self.order_book_snapshots[symbol][0].timestamp 
                    if symbol in self.order_book_snapshots and self.order_book_snapshots[symbol]
                    else time.time()
                )
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取分析摘要失败: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """关闭分析器"""
        try:
            self.running = False
            self.executor.shutdown(wait=True)
            logger.info("市场微观结构分析器已关闭")
        except Exception as e:
            logger.error(f"关闭市场微观结构分析器失败: {e}")


# 全局市场微观结构分析器实例
microstructure_analyzer = MarketMicrostructureAnalyzer()
