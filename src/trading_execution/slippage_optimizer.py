"""
📊 滑点优化算法系统
生产级市场冲击模型和滑点最小化算法
实现动态订单大小调整、时间加权平均价格和实施缺口分析
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
import math

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores


class SlippageModel(Enum):
    """滑点模型类型"""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    ALMGREN_CHRISS = "almgren_chriss"
    ADAPTIVE = "adaptive"


class ExecutionStrategy(Enum):
    """执行策略"""
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"    # 成交量参与率
    IS = "is"      # 实施缺口
    ADAPTIVE = "adaptive"  # 自适应策略


@dataclass
class MarketImpactParams:
    """市场冲击参数"""
    permanent_impact_coeff: float = 0.1  # 永久冲击系数
    temporary_impact_coeff: float = 0.01  # 临时冲击系数
    volatility: float = 0.02  # 波动率
    daily_volume: float = 1000000.0  # 日均成交量
    bid_ask_spread: float = 0.001  # 买卖价差
    participation_rate: float = 0.1  # 参与率
    risk_aversion: float = 1e-6  # 风险厌恶系数


@dataclass
class SlippageMetrics:
    """滑点指标"""
    timestamp: float
    symbol: str
    order_size: float
    market_volume: float
    expected_slippage: float
    actual_slippage: float
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float


@dataclass
class OptimizationResult:
    """优化结果"""
    strategy: ExecutionStrategy
    optimal_chunks: List[float]  # 最优分片大小
    optimal_intervals: List[float]  # 最优时间间隔
    expected_slippage: float
    expected_cost: float
    risk_measure: float
    confidence_score: float


class SlippageOptimizer:
    """滑点优化器"""
    
    def __init__(self):
        # 市场冲击模型参数
        self.impact_params = MarketImpactParams()
        
        # 历史滑点数据
        self.slippage_history: deque = deque(maxlen=1000)
        self.market_data_cache: Dict[str, Dict] = {}
        
        # 优化配置
        self.max_chunks = 20  # 最大分片数
        self.min_chunk_size = 0.01  # 最小分片大小
        self.max_execution_time = 3600.0  # 最大执行时间(秒)
        self.optimization_window = 300.0  # 优化窗口(秒)
        
        # 模型参数学习
        self.model_learning_rate = 0.01
        self.model_decay_factor = 0.95
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.SLIPPAGE_OPTIMIZATION, [13, 14])
        
        logger.info("滑点优化器初始化完成")
    
    def update_market_impact_params(self, symbol: str, market_data: Dict[str, Any]):
        """更新市场冲击参数"""
        try:
            # 提取市场数据
            current_price = market_data.get('price', 0.0)
            volume_24h = market_data.get('volume_24h', 0.0)
            volatility = market_data.get('volatility', 0.02)
            bid_ask_spread = market_data.get('spread', 0.001)
            
            # 更新参数
            self.impact_params.daily_volume = volume_24h
            self.impact_params.volatility = volatility
            self.impact_params.bid_ask_spread = bid_ask_spread
            
            # 缓存市场数据
            self.market_data_cache[symbol] = {
                'price': current_price,
                'volume': volume_24h,
                'volatility': volatility,
                'spread': bid_ask_spread,
                'timestamp': time.time()
            }
            
            logger.debug(f"更新市场冲击参数: {symbol}, 波动率={volatility:.4f}")
            
        except Exception as e:
            logger.error(f"更新市场冲击参数失败: {e}")
    
    def calculate_market_impact(self, order_size: float, market_volume: float, 
                               model: SlippageModel = SlippageModel.SQUARE_ROOT) -> float:
        """计算市场冲击"""
        try:
            if market_volume <= 0:
                return 0.1  # 默认冲击
            
            # 参与率
            participation_rate = abs(order_size) / market_volume
            
            if model == SlippageModel.LINEAR:
                # 线性模型
                impact = self.impact_params.temporary_impact_coeff * participation_rate
            
            elif model == SlippageModel.SQUARE_ROOT:
                # 平方根模型 (最常用)
                impact = self.impact_params.temporary_impact_coeff * np.sqrt(participation_rate)
            
            elif model == SlippageModel.LOGARITHMIC:
                # 对数模型
                impact = self.impact_params.temporary_impact_coeff * np.log(1 + participation_rate)
            
            elif model == SlippageModel.ALMGREN_CHRISS:
                # Almgren-Chriss模型
                permanent_impact = self.impact_params.permanent_impact_coeff * participation_rate
                temporary_impact = self.impact_params.temporary_impact_coeff * np.sqrt(participation_rate)
                impact = permanent_impact + temporary_impact
            
            else:  # ADAPTIVE
                # 自适应模型 - 根据历史数据调整
                base_impact = self.impact_params.temporary_impact_coeff * np.sqrt(participation_rate)
                
                # 基于历史滑点调整
                if len(self.slippage_history) > 10:
                    recent_slippage = [s.actual_slippage for s in list(self.slippage_history)[-10:]]
                    avg_slippage = np.mean(recent_slippage)
                    adjustment_factor = avg_slippage / base_impact if base_impact > 0 else 1.0
                    impact = base_impact * adjustment_factor
                else:
                    impact = base_impact
            
            return max(0.0, impact)
            
        except Exception as e:
            logger.error(f"计算市场冲击失败: {e}")
            return 0.01  # 默认1%冲击
    
    def calculate_timing_cost(self, execution_time: float, volatility: float) -> float:
        """计算时机成本"""
        try:
            # 基于波动率的时机成本
            # 执行时间越长，面临的价格波动风险越大
            timing_cost = volatility * np.sqrt(execution_time / 86400.0)  # 年化波动率转日波动率
            
            return timing_cost
            
        except Exception as e:
            logger.error(f"计算时机成本失败: {e}")
            return 0.0
    
    def calculate_opportunity_cost(self, delay_time: float, expected_return: float) -> float:
        """计算机会成本"""
        try:
            # 延迟执行的机会成本
            opportunity_cost = expected_return * (delay_time / 86400.0)  # 日化收益
            
            return max(0.0, opportunity_cost)
            
        except Exception as e:
            logger.error(f"计算机会成本失败: {e}")
            return 0.0
    
    def optimize_twap_strategy(self, total_size: float, execution_time: float, 
                              market_data: Dict[str, Any]) -> OptimizationResult:
        """优化TWAP策略"""
        try:
            # 计算最优分片数量
            volatility = market_data.get('volatility', self.impact_params.volatility)
            daily_volume = market_data.get('volume', self.impact_params.daily_volume)
            
            # 基于Almgren-Chriss模型的最优分片
            # 平衡市场冲击和时机成本
            
            # 计算最优分片数
            temp_impact_coeff = self.impact_params.temporary_impact_coeff
            risk_aversion = self.impact_params.risk_aversion
            
            # 最优分片数公式
            optimal_chunks = int(np.sqrt(
                (temp_impact_coeff * total_size * volatility ** 2 * execution_time) /
                (2 * risk_aversion * daily_volume)
            ))
            
            optimal_chunks = max(1, min(optimal_chunks, self.max_chunks))
            
            # 计算分片大小和时间间隔
            chunk_size = total_size / optimal_chunks
            time_interval = execution_time / optimal_chunks
            
            chunks = [chunk_size] * optimal_chunks
            intervals = [time_interval] * optimal_chunks
            
            # 计算预期成本
            market_impact = self.calculate_market_impact(chunk_size, daily_volume)
            timing_cost = self.calculate_timing_cost(execution_time, volatility)
            
            expected_slippage = market_impact + timing_cost
            expected_cost = expected_slippage * total_size
            
            # 风险度量 (方差)
            risk_measure = volatility ** 2 * execution_time / optimal_chunks
            
            return OptimizationResult(
                strategy=ExecutionStrategy.TWAP,
                optimal_chunks=chunks,
                optimal_intervals=intervals,
                expected_slippage=expected_slippage,
                expected_cost=expected_cost,
                risk_measure=risk_measure,
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error(f"优化TWAP策略失败: {e}")
            return self._get_default_result(total_size, execution_time)
    
    def optimize_vwap_strategy(self, total_size: float, execution_time: float,
                              volume_profile: List[float]) -> OptimizationResult:
        """优化VWAP策略"""
        try:
            if not volume_profile:
                return self._get_default_result(total_size, execution_time)
            
            # 根据成交量分布调整分片大小
            total_volume = sum(volume_profile)
            if total_volume <= 0:
                return self._get_default_result(total_size, execution_time)
            
            # 按成交量比例分配订单
            chunks = []
            intervals = []
            
            time_per_interval = execution_time / len(volume_profile)
            
            for volume in volume_profile:
                volume_ratio = volume / total_volume
                chunk_size = total_size * volume_ratio
                
                if chunk_size >= self.min_chunk_size:
                    chunks.append(chunk_size)
                    intervals.append(time_per_interval)
            
            if not chunks:
                return self._get_default_result(total_size, execution_time)
            
            # 计算预期成本
            avg_chunk_size = np.mean(chunks)
            market_impact = self.calculate_market_impact(avg_chunk_size, np.mean(volume_profile))
            timing_cost = self.calculate_timing_cost(execution_time, self.impact_params.volatility)
            
            expected_slippage = market_impact + timing_cost
            expected_cost = expected_slippage * total_size
            
            # VWAP策略的风险较低
            risk_measure = self.impact_params.volatility ** 2 * execution_time / len(chunks) * 0.8
            
            return OptimizationResult(
                strategy=ExecutionStrategy.VWAP,
                optimal_chunks=chunks,
                optimal_intervals=intervals,
                expected_slippage=expected_slippage,
                expected_cost=expected_cost,
                risk_measure=risk_measure,
                confidence_score=0.85
            )
            
        except Exception as e:
            logger.error(f"优化VWAP策略失败: {e}")
            return self._get_default_result(total_size, execution_time)
    
    def optimize_pov_strategy(self, total_size: float, target_participation: float,
                             market_data: Dict[str, Any]) -> OptimizationResult:
        """优化POV(参与率)策略"""
        try:
            daily_volume = market_data.get('volume', self.impact_params.daily_volume)
            volatility = market_data.get('volatility', self.impact_params.volatility)
            
            # 基于目标参与率计算执行时间
            if target_participation <= 0:
                target_participation = 0.1  # 默认10%参与率
            
            # 估算执行时间
            execution_time = total_size / (daily_volume * target_participation / 86400.0)  # 秒
            execution_time = min(execution_time, self.max_execution_time)
            
            # 计算分片
            chunk_volume = daily_volume * target_participation / 86400.0  # 每秒的目标成交量
            time_interval = 60.0  # 1分钟间隔
            chunk_size = chunk_volume * time_interval
            
            num_chunks = int(np.ceil(total_size / chunk_size))
            num_chunks = min(num_chunks, self.max_chunks)
            
            # 调整分片大小
            actual_chunk_size = total_size / num_chunks
            chunks = [actual_chunk_size] * num_chunks
            intervals = [time_interval] * num_chunks
            
            # 计算预期成本
            market_impact = self.calculate_market_impact(actual_chunk_size, chunk_volume * time_interval)
            timing_cost = self.calculate_timing_cost(execution_time, volatility)
            
            expected_slippage = market_impact + timing_cost
            expected_cost = expected_slippage * total_size
            
            # POV策略风险中等
            risk_measure = volatility ** 2 * execution_time / num_chunks
            
            return OptimizationResult(
                strategy=ExecutionStrategy.POV,
                optimal_chunks=chunks,
                optimal_intervals=intervals,
                expected_slippage=expected_slippage,
                expected_cost=expected_cost,
                risk_measure=risk_measure,
                confidence_score=0.75
            )
            
        except Exception as e:
            logger.error(f"优化POV策略失败: {e}")
            return self._get_default_result(total_size, 3600.0)
    
    def optimize_implementation_shortfall(self, total_size: float, urgency: float,
                                        market_data: Dict[str, Any]) -> OptimizationResult:
        """优化实施缺口策略"""
        try:
            volatility = market_data.get('volatility', self.impact_params.volatility)
            daily_volume = market_data.get('volume', self.impact_params.daily_volume)
            
            # 基于紧急程度调整执行时间
            # urgency: 0(不急) -> 1(非常急)
            max_time = self.max_execution_time
            min_time = 60.0  # 最少1分钟
            
            execution_time = max_time * (1 - urgency) + min_time * urgency
            
            # 实施缺口最优化
            # 最小化: 市场冲击成本 + 时机成本 + 机会成本
            
            # 计算最优分片数 (基于实施缺口模型)
            temp_impact = self.impact_params.temporary_impact_coeff
            risk_aversion = self.impact_params.risk_aversion
            
            # 考虑紧急程度的风险厌恶调整
            adjusted_risk_aversion = risk_aversion * (1 + urgency)
            
            optimal_chunks = int(np.sqrt(
                (temp_impact * total_size * volatility ** 2 * execution_time) /
                (2 * adjusted_risk_aversion * daily_volume)
            ))
            
            optimal_chunks = max(1, min(optimal_chunks, self.max_chunks))
            
            # 非均匀分片 - 前面大后面小 (紧急订单)
            chunks = []
            total_weight = sum(1.0 / (i + 1) for i in range(optimal_chunks))
            
            for i in range(optimal_chunks):
                weight = (1.0 / (i + 1)) / total_weight
                chunk_size = total_size * weight
                chunks.append(chunk_size)
            
            # 时间间隔也非均匀 - 前面短后面长
            intervals = []
            total_time_weight = sum(1.0 + i * 0.1 for i in range(optimal_chunks))
            
            for i in range(optimal_chunks):
                time_weight = (1.0 + i * 0.1) / total_time_weight
                interval = execution_time * time_weight
                intervals.append(interval)
            
            # 计算预期成本
            avg_chunk_size = np.mean(chunks)
            market_impact = self.calculate_market_impact(avg_chunk_size, daily_volume)
            timing_cost = self.calculate_timing_cost(execution_time, volatility)
            opportunity_cost = self.calculate_opportunity_cost(execution_time, urgency * 0.01)
            
            expected_slippage = market_impact + timing_cost + opportunity_cost
            expected_cost = expected_slippage * total_size
            
            # 实施缺口策略风险较高但执行效率高
            risk_measure = volatility ** 2 * execution_time / optimal_chunks * (1 + urgency)
            
            return OptimizationResult(
                strategy=ExecutionStrategy.IS,
                optimal_chunks=chunks,
                optimal_intervals=intervals,
                expected_slippage=expected_slippage,
                expected_cost=expected_cost,
                risk_measure=risk_measure,
                confidence_score=0.9
            )
            
        except Exception as e:
            logger.error(f"优化实施缺口策略失败: {e}")
            return self._get_default_result(total_size, 1800.0)
    
    def select_optimal_strategy(self, total_size: float, market_data: Dict[str, Any],
                               constraints: Dict[str, Any]) -> OptimizationResult:
        """选择最优执行策略"""
        try:
            urgency = constraints.get('urgency', 0.5)
            max_execution_time = constraints.get('max_execution_time', self.max_execution_time)
            target_participation = constraints.get('target_participation', 0.1)
            volume_profile = constraints.get('volume_profile', [])
            
            # 评估所有策略
            strategies = []
            
            # TWAP策略
            twap_result = self.optimize_twap_strategy(total_size, max_execution_time, market_data)
            strategies.append(twap_result)
            
            # VWAP策略 (如果有成交量分布)
            if volume_profile:
                vwap_result = self.optimize_vwap_strategy(total_size, max_execution_time, volume_profile)
                strategies.append(vwap_result)
            
            # POV策略
            pov_result = self.optimize_pov_strategy(total_size, target_participation, market_data)
            strategies.append(pov_result)
            
            # 实施缺口策略
            is_result = self.optimize_implementation_shortfall(total_size, urgency, market_data)
            strategies.append(is_result)
            
            # 选择最优策略 (综合考虑成本、风险和置信度)
            best_strategy = None
            best_score = float('inf')
            
            for strategy in strategies:
                # 综合评分 = 成本 + 风险惩罚 - 置信度奖励
                risk_penalty = strategy.risk_measure * 1000  # 风险惩罚
                confidence_bonus = (1 - strategy.confidence_score) * strategy.expected_cost * 0.1
                
                score = strategy.expected_cost + risk_penalty + confidence_bonus
                
                if score < best_score:
                    best_score = score
                    best_strategy = strategy
            
            if best_strategy:
                logger.info(f"选择最优策略: {best_strategy.strategy.value}, "
                           f"预期滑点={best_strategy.expected_slippage:.4f}, "
                           f"置信度={best_strategy.confidence_score:.3f}")
                return best_strategy
            else:
                return self._get_default_result(total_size, max_execution_time)
            
        except Exception as e:
            logger.error(f"选择最优策略失败: {e}")
            return self._get_default_result(total_size, 3600.0)
    
    def _get_default_result(self, total_size: float, execution_time: float) -> OptimizationResult:
        """获取默认优化结果"""
        # 简单均匀分片
        num_chunks = min(10, max(1, int(total_size / self.min_chunk_size)))
        chunk_size = total_size / num_chunks
        time_interval = execution_time / num_chunks
        
        return OptimizationResult(
            strategy=ExecutionStrategy.TWAP,
            optimal_chunks=[chunk_size] * num_chunks,
            optimal_intervals=[time_interval] * num_chunks,
            expected_slippage=0.01,  # 默认1%滑点
            expected_cost=total_size * 0.01,
            risk_measure=0.001,
            confidence_score=0.5
        )
    
    def record_execution_result(self, symbol: str, order_size: float, 
                               expected_slippage: float, actual_slippage: float,
                               market_data: Dict[str, Any]):
        """记录执行结果"""
        try:
            market_volume = market_data.get('volume', 0.0)
            
            # 计算实施缺口
            implementation_shortfall = actual_slippage - expected_slippage
            
            # 分解滑点成分
            market_impact = self.calculate_market_impact(order_size, market_volume)
            timing_cost = actual_slippage - market_impact
            
            # 创建滑点指标
            metrics = SlippageMetrics(
                timestamp=time.time(),
                symbol=symbol,
                order_size=order_size,
                market_volume=market_volume,
                expected_slippage=expected_slippage,
                actual_slippage=actual_slippage,
                implementation_shortfall=implementation_shortfall,
                market_impact=market_impact,
                timing_cost=timing_cost,
                opportunity_cost=0.0  # 需要额外计算
            )
            
            # 添加到历史记录
            self.slippage_history.append(metrics)
            
            # 更新模型参数 (在线学习)
            self._update_model_parameters(metrics)
            
            logger.info(f"记录执行结果: {symbol}, 预期滑点={expected_slippage:.4f}, "
                       f"实际滑点={actual_slippage:.4f}, 实施缺口={implementation_shortfall:.4f}")
            
        except Exception as e:
            logger.error(f"记录执行结果失败: {e}")
    
    def _update_model_parameters(self, metrics: SlippageMetrics):
        """更新模型参数"""
        try:
            # 基于实际结果调整模型参数
            prediction_error = metrics.actual_slippage - metrics.expected_slippage
            
            # 调整临时冲击系数
            if abs(prediction_error) > 0.001:  # 误差超过0.1%
                adjustment = self.model_learning_rate * prediction_error
                self.impact_params.temporary_impact_coeff *= (1 + adjustment)
                self.impact_params.temporary_impact_coeff = max(0.001, 
                    min(0.1, self.impact_params.temporary_impact_coeff))
            
            # 衰减学习率
            self.model_learning_rate *= self.model_decay_factor
            
        except Exception as e:
            logger.error(f"更新模型参数失败: {e}")
    
    def get_slippage_statistics(self) -> Dict[str, Any]:
        """获取滑点统计"""
        if not self.slippage_history:
            return {}
        
        history = list(self.slippage_history)
        
        expected_slippages = [h.expected_slippage for h in history]
        actual_slippages = [h.actual_slippage for h in history]
        implementation_shortfalls = [h.implementation_shortfall for h in history]
        
        return {
            'total_executions': len(history),
            'avg_expected_slippage': np.mean(expected_slippages),
            'avg_actual_slippage': np.mean(actual_slippages),
            'avg_implementation_shortfall': np.mean(implementation_shortfalls),
            'slippage_std': np.std(actual_slippages),
            'prediction_accuracy': 1.0 - np.mean(np.abs(implementation_shortfalls)),
            'model_parameters': {
                'temporary_impact_coeff': self.impact_params.temporary_impact_coeff,
                'permanent_impact_coeff': self.impact_params.permanent_impact_coeff,
                'learning_rate': self.model_learning_rate
            }
        }


# 全局滑点优化器实例
slippage_optimizer = SlippageOptimizer()


async def main():
    """测试主函数"""
    logger.info("启动滑点优化器测试...")
    
    try:
        # 模拟市场数据
        market_data = {
            'price': 50000.0,
            'volume': 1000000.0,
            'volatility': 0.02,
            'spread': 0.001
        }
        
        # 模拟约束条件
        constraints = {
            'urgency': 0.7,
            'max_execution_time': 1800.0,  # 30分钟
            'target_participation': 0.15,
            'volume_profile': [100, 150, 200, 180, 120, 90, 110, 140]
        }
        
        # 优化执行策略
        result = slippage_optimizer.select_optimal_strategy(
            total_size=10.0,
            market_data=market_data,
            constraints=constraints
        )
        
        logger.info(f"最优策略: {result.strategy.value}")
        logger.info(f"分片数: {len(result.optimal_chunks)}")
        logger.info(f"预期滑点: {result.expected_slippage:.4f}")
        logger.info(f"预期成本: {result.expected_cost:.2f}")
        logger.info(f"置信度: {result.confidence_score:.3f}")
        
        # 模拟执行结果记录
        slippage_optimizer.record_execution_result(
            symbol="BTC/USDT",
            order_size=10.0,
            expected_slippage=result.expected_slippage,
            actual_slippage=result.expected_slippage * 1.1,  # 实际滑点稍高
            market_data=market_data
        )
        
        # 获取统计信息
        stats = slippage_optimizer.get_slippage_statistics()
        logger.info(f"滑点统计: {stats}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    except Exception as e:
        logger.error(f"测试出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
