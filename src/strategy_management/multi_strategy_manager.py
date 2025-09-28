"""
🎯 多策略管理器
生产级多策略组合管理系统，整合投资组合优化、相关性分析和动态权重分配
实现完整的策略生命周期管理、性能监控和风险控制
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings
from src.strategy_management.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationMethod, OptimizationConstraints, PortfolioMetrics
)
from src.strategy_management.strategy_correlation_analyzer import (
    StrategyCorrelationAnalyzer, CorrelationMethod, CorrelationMetrics
)
from src.strategy_management.dynamic_weight_allocator import (
    DynamicWeightAllocator, AllocationMethod, AllocationConstraints, AllocationResult
)


class StrategyStatus(Enum):
    """策略状态"""
    ACTIVE = "active"           # 活跃
    INACTIVE = "inactive"       # 非活跃
    PAUSED = "paused"          # 暂停
    DEPRECATED = "deprecated"   # 已弃用
    TESTING = "testing"        # 测试中


class RebalanceReason(Enum):
    """再平衡原因"""
    SCHEDULED = "scheduled"           # 定期再平衡
    DRIFT_THRESHOLD = "drift"         # 权重漂移
    CORRELATION_CHANGE = "correlation" # 相关性变化
    PERFORMANCE_TRIGGER = "performance" # 性能触发
    RISK_LIMIT = "risk_limit"         # 风险限制
    MANUAL = "manual"                 # 手动触发


@dataclass
class StrategyInfo:
    """策略信息"""
    name: str                           # 策略名称
    description: str                    # 策略描述
    status: StrategyStatus              # 策略状态
    inception_date: datetime            # 创建日期
    last_update: datetime               # 最后更新时间
    target_weight: float                # 目标权重
    current_weight: float               # 当前权重
    min_weight: float = 0.0             # 最小权重
    max_weight: float = 1.0             # 最大权重
    risk_budget: float = 0.0            # 风险预算
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RebalanceEvent:
    """再平衡事件"""
    timestamp: datetime                 # 时间戳
    reason: RebalanceReason            # 再平衡原因
    old_weights: np.ndarray            # 旧权重
    new_weights: np.ndarray            # 新权重
    turnover: float                    # 换手率
    expected_improvement: float        # 预期改善
    execution_cost: float = 0.0        # 执行成本
    success: bool = True               # 是否成功


@dataclass
class PortfolioSnapshot:
    """组合快照"""
    timestamp: datetime                 # 时间戳
    weights: Dict[str, float]          # 权重分布
    portfolio_metrics: PortfolioMetrics # 组合指标
    correlation_metrics: CorrelationMetrics # 相关性指标
    individual_performance: Dict[str, Dict[str, float]]  # 个别策略表现
    risk_attribution: Dict[str, float] # 风险归因
    total_value: float                 # 总价值
    cash_position: float = 0.0         # 现金仓位


class MultiStrategyManager:
    """多策略管理器"""
    
    def __init__(self):
        # 核心组件
        self.portfolio_optimizer = PortfolioOptimizer()
        self.correlation_analyzer = StrategyCorrelationAnalyzer()
        self.weight_allocator = DynamicWeightAllocator()
        
        # 策略管理
        self.strategies: Dict[str, StrategyInfo] = {}
        self.strategy_returns: pd.DataFrame = pd.DataFrame()
        
        # 组合状态
        self.current_weights: Optional[np.ndarray] = None
        self.target_weights: Optional[np.ndarray] = None
        self.last_rebalance: Optional[datetime] = None
        
        # 历史记录
        self.rebalance_history: List[RebalanceEvent] = []
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        
        # 配置参数
        self.rebalance_frequency = 5        # 再平衡频率(天)
        self.drift_threshold = 0.05         # 权重漂移阈值
        self.correlation_threshold = 0.1    # 相关性变化阈值
        self.min_rebalance_improvement = 0.01  # 最小再平衡改善
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("多策略管理器初始化完成")
    
    async def add_strategy(
        self,
        name: str,
        description: str,
        returns: pd.Series,
        target_weight: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        risk_budget: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加策略
        
        Args:
            name: 策略名称
            description: 策略描述
            returns: 策略收益率序列
            target_weight: 目标权重
            min_weight: 最小权重
            max_weight: 最大权重
            risk_budget: 风险预算
            metadata: 元数据
        
        Returns:
            bool: 是否成功添加
        """
        try:
            if name in self.strategies:
                logger.warning(f"策略 {name} 已存在，将更新信息")
            
            # 创建策略信息
            strategy_info = StrategyInfo(
                name=name,
                description=description,
                status=StrategyStatus.ACTIVE,
                inception_date=datetime.now(),
                last_update=datetime.now(),
                target_weight=target_weight,
                current_weight=0.0,
                min_weight=min_weight,
                max_weight=max_weight,
                risk_budget=risk_budget,
                metadata=metadata or {}
            )
            
            # 计算基本性能指标
            strategy_info.performance_metrics = self._calculate_strategy_metrics(returns)
            
            # 添加到策略字典
            self.strategies[name] = strategy_info
            
            # 更新收益率数据
            if name not in self.strategy_returns.columns:
                self.strategy_returns[name] = returns
            else:
                self.strategy_returns[name] = returns
            
            # 重新对齐数据
            self.strategy_returns = self.strategy_returns.dropna()
            
            logger.info(f"策略 {name} 添加成功，当前共有 {len(self.strategies)} 个策略")
            return True
            
        except Exception as e:
            logger.error(f"添加策略 {name} 失败: {e}")
            return False
    
    def _calculate_strategy_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算策略性能指标"""
        if len(returns) == 0:
            return {}
        
        # 基本统计量
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns),
            'avg_return': returns.mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    async def remove_strategy(self, name: str) -> bool:
        """移除策略"""
        try:
            if name not in self.strategies:
                logger.warning(f"策略 {name} 不存在")
                return False
            
            # 移除策略信息
            del self.strategies[name]
            
            # 移除收益率数据
            if name in self.strategy_returns.columns:
                self.strategy_returns = self.strategy_returns.drop(columns=[name])
            
            # 重置权重
            self.current_weights = None
            self.target_weights = None
            
            logger.info(f"策略 {name} 移除成功")
            return True
            
        except Exception as e:
            logger.error(f"移除策略 {name} 失败: {e}")
            return False
    
    async def update_strategy_status(self, name: str, status: StrategyStatus) -> bool:
        """更新策略状态"""
        try:
            if name not in self.strategies:
                logger.warning(f"策略 {name} 不存在")
                return False
            
            self.strategies[name].status = status
            self.strategies[name].last_update = datetime.now()
            
            logger.info(f"策略 {name} 状态更新为 {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"更新策略状态失败: {e}")
            return False
    
    async def optimize_portfolio(
        self,
        method: OptimizationMethod = OptimizationMethod.MARKOWITZ,
        constraints: Optional[OptimizationConstraints] = None,
        force_rebalance: bool = False
    ) -> Optional[PortfolioMetrics]:
        """
        优化投资组合
        
        Args:
            method: 优化方法
            constraints: 约束条件
            force_rebalance: 是否强制再平衡
        
        Returns:
            PortfolioMetrics: 优化结果
        """
        try:
            if len(self.strategies) == 0:
                logger.warning("没有可用策略进行优化")
                return None
            
            # 获取活跃策略的收益率数据
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            
            if not active_strategies:
                logger.warning("没有活跃策略")
                return None
            
            active_returns = self.strategy_returns[active_strategies].dropna()
            
            if len(active_returns) < 30:
                logger.warning("历史数据不足，无法进行优化")
                return None
            
            # 设置约束条件
            if constraints is None:
                constraints = OptimizationConstraints()
                
                # 从策略信息中获取权重限制
                for i, strategy_name in enumerate(active_strategies):
                    strategy_info = self.strategies[strategy_name]
                    constraints.min_weight = max(constraints.min_weight, strategy_info.min_weight)
                    constraints.max_weight = min(constraints.max_weight, strategy_info.max_weight)
            
            # 执行优化
            portfolio_metrics = await self.portfolio_optimizer.optimize_portfolio(
                active_returns, method, constraints, self.current_weights
            )
            
            # 检查是否需要再平衡
            should_rebalance = await self._should_rebalance(
                portfolio_metrics.weights, force_rebalance
            )
            
            if should_rebalance:
                await self._execute_rebalance(
                    portfolio_metrics.weights, 
                    RebalanceReason.SCHEDULED if not force_rebalance else RebalanceReason.MANUAL
                )
            
            # 创建组合快照
            await self._create_portfolio_snapshot(portfolio_metrics)
            
            logger.info(f"组合优化完成 - 夏普比率: {portfolio_metrics.sharpe_ratio:.3f}")
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            return None
    
    async def _should_rebalance(
        self, 
        target_weights: np.ndarray, 
        force_rebalance: bool = False
    ) -> bool:
        """判断是否需要再平衡"""
        if force_rebalance:
            return True
        
        # 如果没有当前权重，需要初始化
        if self.current_weights is None:
            return True
        
        # 检查时间间隔
        if self.last_rebalance is not None:
            days_since_rebalance = (datetime.now() - self.last_rebalance).days
            if days_since_rebalance < self.rebalance_frequency:
                return False
        
        # 检查权重漂移
        weight_drift = np.sum(np.abs(target_weights - self.current_weights))
        if weight_drift > self.drift_threshold:
            logger.info(f"权重漂移 {weight_drift:.3f} 超过阈值 {self.drift_threshold}")
            return True
        
        # 检查相关性变化
        try:
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            active_returns = self.strategy_returns[active_strategies].tail(60)
            
            current_corr = active_returns.corr().values
            
            # 与历史相关性比较
            if len(self.portfolio_snapshots) > 0:
                last_snapshot = self.portfolio_snapshots[-1]
                historical_corr = last_snapshot.correlation_metrics.correlation_matrix
                
                corr_change = np.mean(np.abs(current_corr - historical_corr))
                if corr_change > self.correlation_threshold:
                    logger.info(f"相关性变化 {corr_change:.3f} 超过阈值 {self.correlation_threshold}")
                    return True
        except Exception as e:
            logger.warning(f"相关性检查失败: {e}")
        
        return False
    
    async def _execute_rebalance(
        self, 
        new_weights: np.ndarray, 
        reason: RebalanceReason
    ) -> bool:
        """执行再平衡"""
        try:
            old_weights = self.current_weights.copy() if self.current_weights is not None else np.zeros(len(new_weights))
            
            # 计算换手率
            turnover = np.sum(np.abs(new_weights - old_weights)) / 2
            
            # 估算执行成本 (简化模型)
            execution_cost = turnover * 0.001  # 假设0.1%的交易成本
            
            # 估算预期改善
            expected_improvement = self._estimate_rebalance_improvement(old_weights, new_weights)
            
            # 检查是否值得再平衡
            if expected_improvement < self.min_rebalance_improvement and reason != RebalanceReason.MANUAL:
                logger.info(f"预期改善 {expected_improvement:.4f} 低于最小阈值，跳过再平衡")
                return False
            
            # 创建再平衡事件
            rebalance_event = RebalanceEvent(
                timestamp=datetime.now(),
                reason=reason,
                old_weights=old_weights,
                new_weights=new_weights,
                turnover=turnover,
                expected_improvement=expected_improvement,
                execution_cost=execution_cost,
                success=True
            )
            
            # 更新权重
            self.current_weights = new_weights.copy()
            self.target_weights = new_weights.copy()
            self.last_rebalance = datetime.now()
            
            # 更新策略权重
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            
            for i, strategy_name in enumerate(active_strategies):
                self.strategies[strategy_name].current_weight = new_weights[i]
            
            # 记录再平衡事件
            self.rebalance_history.append(rebalance_event)
            
            # 保持最近100次记录
            if len(self.rebalance_history) > 100:
                self.rebalance_history = self.rebalance_history[-100:]
            
            logger.info(f"再平衡执行成功 - 原因: {reason.value}, 换手率: {turnover:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"再平衡执行失败: {e}")
            return False
    
    def _estimate_rebalance_improvement(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray
    ) -> float:
        """估算再平衡改善"""
        try:
            # 简化的改善估算：基于权重变化和策略表现
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            
            if len(active_strategies) == 0:
                return 0.0
            
            # 获取策略近期表现
            recent_returns = self.strategy_returns[active_strategies].tail(21).mean()
            
            # 计算权重调整对预期收益的影响
            old_expected_return = np.dot(old_weights, recent_returns.values)
            new_expected_return = np.dot(new_weights, recent_returns.values)
            
            improvement = new_expected_return - old_expected_return
            return improvement
            
        except Exception as e:
            logger.warning(f"改善估算失败: {e}")
            return 0.0
    
    async def _create_portfolio_snapshot(self, portfolio_metrics: PortfolioMetrics):
        """创建组合快照"""
        try:
            # 获取活跃策略
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            
            if not active_strategies:
                return
            
            # 权重分布
            weights_dict = {
                strategy: weight 
                for strategy, weight in zip(active_strategies, portfolio_metrics.weights)
            }
            
            # 相关性分析
            active_returns = self.strategy_returns[active_strategies].dropna()
            correlation_metrics, _ = await self.correlation_analyzer.analyze_strategy_correlations(
                active_returns, rolling_analysis=False
            )
            
            # 个别策略表现
            individual_performance = {}
            for strategy_name in active_strategies:
                individual_performance[strategy_name] = self.strategies[strategy_name].performance_metrics
            
            # 风险归因
            risk_attribution = {
                strategy: contrib 
                for strategy, contrib in zip(active_strategies, portfolio_metrics.risk_contributions)
            }
            
            # 创建快照
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                weights=weights_dict,
                portfolio_metrics=portfolio_metrics,
                correlation_metrics=correlation_metrics,
                individual_performance=individual_performance,
                risk_attribution=risk_attribution,
                total_value=1.0,  # 标准化为1
                cash_position=0.0
            )
            
            self.portfolio_snapshots.append(snapshot)
            
            # 保持最近50个快照
            if len(self.portfolio_snapshots) > 50:
                self.portfolio_snapshots = self.portfolio_snapshots[-50:]
            
        except Exception as e:
            logger.error(f"创建组合快照失败: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        try:
            active_strategies = [
                name for name, info in self.strategies.items() 
                if info.status == StrategyStatus.ACTIVE
            ]
            
            summary = {
                'total_strategies': len(self.strategies),
                'active_strategies': len(active_strategies),
                'strategy_list': list(self.strategies.keys()),
                'active_strategy_list': active_strategies,
                'last_rebalance': self.last_rebalance,
                'rebalance_count': len(self.rebalance_history),
                'snapshot_count': len(self.portfolio_snapshots)
            }
            
            # 当前权重
            if self.current_weights is not None and active_strategies:
                summary['current_weights'] = {
                    strategy: weight 
                    for strategy, weight in zip(active_strategies, self.current_weights)
                }
            
            # 最新快照信息
            if self.portfolio_snapshots:
                latest_snapshot = self.portfolio_snapshots[-1]
                summary['latest_metrics'] = {
                    'expected_return': latest_snapshot.portfolio_metrics.expected_return,
                    'volatility': latest_snapshot.portfolio_metrics.volatility,
                    'sharpe_ratio': latest_snapshot.portfolio_metrics.sharpe_ratio,
                    'max_drawdown': latest_snapshot.portfolio_metrics.max_drawdown,
                    'diversification_ratio': latest_snapshot.portfolio_metrics.diversification_ratio
                }
            
            # 策略状态统计
            status_counts = {}
            for status in StrategyStatus:
                count = sum(1 for info in self.strategies.values() if info.status == status)
                if count > 0:
                    status_counts[status.value] = count
            summary['strategy_status'] = status_counts
            
            return summary
            
        except Exception as e:
            logger.error(f"获取组合摘要失败: {e}")
            return {}
    
    async def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 筛选时间范围内的快照
            recent_snapshots = [
                snapshot for snapshot in self.portfolio_snapshots 
                if snapshot.timestamp >= cutoff_date
            ]
            
            if not recent_snapshots:
                return {'error': '没有足够的历史数据'}
            
            # 计算组合表现
            portfolio_returns = []
            timestamps = []
            
            for i in range(1, len(recent_snapshots)):
                prev_snapshot = recent_snapshots[i-1]
                curr_snapshot = recent_snapshots[i]
                
                # 简化的收益率计算
                portfolio_return = (
                    curr_snapshot.portfolio_metrics.expected_return - 
                    prev_snapshot.portfolio_metrics.expected_return
                )
                
                portfolio_returns.append(portfolio_return)
                timestamps.append(curr_snapshot.timestamp)
            
            # 性能指标
            if portfolio_returns:
                total_return = sum(portfolio_returns)
                avg_return = np.mean(portfolio_returns)
                volatility = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0
                
                performance_metrics = {
                    'period_days': days,
                    'total_return': total_return,
                    'average_return': avg_return,
                    'volatility': volatility,
                    'sharpe_ratio': avg_return / volatility if volatility > 0 else 0,
                    'max_return': max(portfolio_returns) if portfolio_returns else 0,
                    'min_return': min(portfolio_returns) if portfolio_returns else 0
                }
            else:
                performance_metrics = {'error': '无法计算性能指标'}
            
            # 再平衡统计
            recent_rebalances = [
                event for event in self.rebalance_history 
                if event.timestamp >= cutoff_date
            ]
            
            rebalance_stats = {
                'total_rebalances': len(recent_rebalances),
                'avg_turnover': np.mean([event.turnover for event in recent_rebalances]) if recent_rebalances else 0,
                'total_cost': sum(event.execution_cost for event in recent_rebalances),
                'reasons': {}
            }
            
            # 再平衡原因统计
            for event in recent_rebalances:
                reason = event.reason.value
                rebalance_stats['reasons'][reason] = rebalance_stats['reasons'].get(reason, 0) + 1
            
            return {
                'performance_metrics': performance_metrics,
                'rebalance_stats': rebalance_stats,
                'snapshot_count': len(recent_snapshots),
                'data_period': {
                    'start_date': recent_snapshots[0].timestamp if recent_snapshots else None,
                    'end_date': recent_snapshots[-1].timestamp if recent_snapshots else None
                }
            }
            
        except Exception as e:
            logger.error(f"获取性能报告失败: {e}")
            return {'error': str(e)}
    
    async def export_data(self, format: str = 'json') -> Optional[str]:
        """导出数据"""
        try:
            export_data = {
                'strategies': {
                    name: {
                        'name': info.name,
                        'description': info.description,
                        'status': info.status.value,
                        'inception_date': info.inception_date.isoformat(),
                        'last_update': info.last_update.isoformat(),
                        'target_weight': info.target_weight,
                        'current_weight': info.current_weight,
                        'min_weight': info.min_weight,
                        'max_weight': info.max_weight,
                        'risk_budget': info.risk_budget,
                        'performance_metrics': info.performance_metrics,
                        'metadata': info.metadata
                    }
                    for name, info in self.strategies.items()
                },
                'current_weights': self.current_weights.tolist() if self.current_weights is not None else None,
                'target_weights': self.target_weights.tolist() if self.target_weights is not None else None,
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'rebalance_history': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'reason': event.reason.value,
                        'old_weights': event.old_weights.tolist(),
                        'new_weights': event.new_weights.tolist(),
                        'turnover': event.turnover,
                        'expected_improvement': event.expected_improvement,
                        'execution_cost': event.execution_cost,
                        'success': event.success
                    }
                    for event in self.rebalance_history
                ],
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                logger.warning(f"不支持的导出格式: {format}")
                return None
                
        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return None
    
    async def shutdown(self):
        """关闭管理器"""
        try:
            await self.portfolio_optimizer.shutdown()
            self.executor.shutdown(wait=True)
            logger.info("多策略管理器已关闭")
        except Exception as e:
            logger.error(f"关闭多策略管理器失败: {e}")


# 全局多策略管理器实例
multi_strategy_manager = MultiStrategyManager()
