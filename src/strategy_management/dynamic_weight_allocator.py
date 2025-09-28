"""
⚖️ 动态权重分配器
生产级动态权重分配系统，实现Kelly公式、风险预算、动态再平衡等完整算法
支持实时权重调整、风险控制和性能监控
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import scipy.optimize as sco
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings


class AllocationMethod(Enum):
    """权重分配方法"""
    KELLY_CRITERION = "kelly_criterion"      # Kelly公式
    RISK_BUDGETING = "risk_budgeting"        # 风险预算
    VOLATILITY_TARGETING = "vol_targeting"   # 波动率目标
    MOMENTUM_BASED = "momentum_based"        # 动量基础
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    ADAPTIVE_ALLOCATION = "adaptive"         # 自适应分配
    REGIME_BASED = "regime_based"            # 制度基础
    MACHINE_LEARNING = "ml_based"            # 机器学习


@dataclass
class AllocationConstraints:
    """分配约束条件"""
    min_weight: float = 0.0                  # 最小权重
    max_weight: float = 0.5                  # 最大权重
    max_turnover: float = 0.2                # 最大换手率
    rebalance_threshold: float = 0.05        # 再平衡阈值
    risk_budget: Optional[Dict[str, float]] = None  # 风险预算
    target_volatility: float = 0.15          # 目标波动率
    lookback_period: int = 252               # 回望期
    confidence_level: float = 0.95           # 置信水平


@dataclass
class AllocationResult:
    """分配结果"""
    weights: np.ndarray                      # 权重向量
    expected_return: float                   # 预期收益率
    expected_volatility: float               # 预期波动率
    expected_sharpe: float                   # 预期夏普比率
    turnover: float                          # 换手率
    risk_contributions: np.ndarray           # 风险贡献
    kelly_fractions: Optional[np.ndarray] = None  # Kelly分数
    confidence_scores: Optional[np.ndarray] = None  # 置信度得分
    allocation_rationale: str = ""           # 分配理由


class DynamicWeightAllocator:
    """动态权重分配器"""
    
    def __init__(self):
        self.risk_free_rate = 0.02           # 无风险利率
        self.rebalance_frequency = 5         # 再平衡频率(天)
        self.min_history_length = 60         # 最小历史长度
        
        # 分配历史记录
        self.allocation_history: List[Dict[str, Any]] = []
        
        # 性能监控
        self.performance_tracker = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'average_turnover': 0.0,
            'average_sharpe': 0.0
        }
        
        logger.info("动态权重分配器初始化完成")
    
    async def allocate_weights(
        self,
        strategy_returns: pd.DataFrame,
        method: AllocationMethod = AllocationMethod.KELLY_CRITERION,
        constraints: Optional[AllocationConstraints] = None,
        current_weights: Optional[np.ndarray] = None,
        market_regime: Optional[str] = None,
        additional_features: Optional[pd.DataFrame] = None
    ) -> AllocationResult:
        """
        动态分配权重
        
        Args:
            strategy_returns: 策略收益率数据
            method: 分配方法
            constraints: 约束条件
            current_weights: 当前权重
            market_regime: 市场制度
            additional_features: 额外特征
        
        Returns:
            AllocationResult: 分配结果
        """
        try:
            if constraints is None:
                constraints = AllocationConstraints()
            
            # 数据预处理
            returns_clean = self._preprocess_returns(strategy_returns)
            
            # 根据方法选择分配算法
            if method == AllocationMethod.KELLY_CRITERION:
                result = await self._kelly_allocation(returns_clean, constraints)
            elif method == AllocationMethod.RISK_BUDGETING:
                result = await self._risk_budgeting_allocation(returns_clean, constraints)
            elif method == AllocationMethod.VOLATILITY_TARGETING:
                result = await self._volatility_targeting_allocation(returns_clean, constraints)
            elif method == AllocationMethod.MOMENTUM_BASED:
                result = await self._momentum_based_allocation(returns_clean, constraints)
            elif method == AllocationMethod.MEAN_REVERSION:
                result = await self._mean_reversion_allocation(returns_clean, constraints)
            elif method == AllocationMethod.ADAPTIVE_ALLOCATION:
                result = await self._adaptive_allocation(returns_clean, constraints, current_weights)
            elif method == AllocationMethod.REGIME_BASED:
                result = await self._regime_based_allocation(returns_clean, constraints, market_regime)
            elif method == AllocationMethod.MACHINE_LEARNING:
                result = await self._ml_based_allocation(returns_clean, constraints, additional_features)
            else:
                raise ValueError(f"未知的分配方法: {method}")
            
            # 应用约束条件
            result = await self._apply_constraints(result, constraints, current_weights)
            
            # 记录分配历史
            self._record_allocation(method, constraints, result)
            
            # 更新性能跟踪
            self._update_performance_tracker(result)
            
            logger.info(f"权重分配完成 - 方法: {method.value}, 预期夏普: {result.expected_sharpe:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"权重分配失败: {e}")
            raise
    
    def _preprocess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """预处理收益率数据"""
        # 去除缺失值
        returns_clean = returns.dropna()
        
        # 确保有足够的数据
        if len(returns_clean) < self.min_history_length:
            raise ValueError(f"数据不足，至少需要{self.min_history_length}个观测值")
        
        # 去除异常值
        for col in returns_clean.columns:
            q1 = returns_clean[col].quantile(0.01)
            q99 = returns_clean[col].quantile(0.99)
            returns_clean[col] = returns_clean[col].clip(lower=q1, upper=q99)
        
        return returns_clean
    
    async def _kelly_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> AllocationResult:
        """Kelly公式分配"""
        n_strategies = len(returns.columns)
        
        # 计算Kelly分数
        kelly_fractions = np.zeros(n_strategies)
        confidence_scores = np.zeros(n_strategies)
        
        for i, col in enumerate(returns.columns):
            strategy_returns = returns[col].values
            
            # 计算统计量
            mean_return = np.mean(strategy_returns) * 252  # 年化
            variance = np.var(strategy_returns) * 252      # 年化
            
            # Kelly公式: f = (μ - r) / σ²
            if variance > 0:
                kelly_fraction = max(0, (mean_return - self.risk_free_rate) / variance)
                kelly_fractions[i] = kelly_fraction
                
                # 计算置信度 (基于t统计量)
                t_stat = mean_return / (np.sqrt(variance / len(strategy_returns)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(strategy_returns) - 1))
                confidence_scores[i] = 1 - p_value
        
        # 标准化权重
        if np.sum(kelly_fractions) > 0:
            weights = kelly_fractions / np.sum(kelly_fractions)
        else:
            weights = np.ones(n_strategies) / n_strategies
        
        # 计算预期指标
        expected_return = np.dot(weights, returns.mean().values * 252)
        cov_matrix = returns.cov().values * 252
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,  # 将在apply_constraints中计算
            risk_contributions=risk_contributions,
            kelly_fractions=kelly_fractions,
            confidence_scores=confidence_scores,
            allocation_rationale="基于Kelly公式的最优增长分配"
        )
    
    async def _risk_budgeting_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> AllocationResult:
        """风险预算分配"""
        n_strategies = len(returns.columns)
        cov_matrix = returns.cov().values * 252
        
        # 默认等风险预算
        if constraints.risk_budget is None:
            risk_budget = np.ones(n_strategies) / n_strategies
        else:
            risk_budget = np.array([
                constraints.risk_budget.get(col, 1/n_strategies) 
                for col in returns.columns
            ])
            risk_budget = risk_budget / np.sum(risk_budget)
        
        def risk_budget_objective(weights):
            """风险预算目标函数"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol == 0:
                return 1e6
            
            # 计算风险贡献
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # 目标：最小化实际风险贡献与目标风险预算的差异
            return np.sum((risk_contrib - risk_budget) ** 2)
        
        # 约束条件
        constraints_opt = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_strategies)]
        
        # 初始权重
        x0 = np.ones(n_strategies) / n_strategies
        
        # 优化
        result = sco.minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("风险预算优化失败，使用等权重")
            weights = np.ones(n_strategies) / n_strategies
        
        # 计算预期指标
        expected_return = np.dot(weights, returns.mean().values * 252)
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale="基于风险预算的分配"
        )
    
    async def _volatility_targeting_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> AllocationResult:
        """波动率目标分配"""
        n_strategies = len(returns.columns)
        cov_matrix = returns.cov().values * 252
        
        # 计算每个策略的波动率
        strategy_vols = np.sqrt(np.diag(cov_matrix))
        
        # 基于逆波动率加权
        inv_vol_weights = (1 / strategy_vols) / np.sum(1 / strategy_vols)
        
        # 调整到目标波动率
        portfolio_vol = np.sqrt(inv_vol_weights.T @ cov_matrix @ inv_vol_weights)
        vol_scaling = constraints.target_volatility / portfolio_vol
        
        # 如果需要降低波动率，按比例缩放权重
        if vol_scaling < 1:
            weights = inv_vol_weights * vol_scaling
            # 剩余权重分配给现金(无风险资产)
            cash_weight = 1 - np.sum(weights)
        else:
            weights = inv_vol_weights
            cash_weight = 0
        
        # 计算预期指标
        expected_return = np.dot(weights, returns.mean().values * 252) + cash_weight * self.risk_free_rate
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        # 计算风险贡献
        if expected_volatility > 0:
            marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
            risk_contributions = weights * marginal_contrib
        else:
            risk_contributions = np.zeros(n_strategies)
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale=f"目标波动率{constraints.target_volatility:.1%}的分配"
        )
    
    async def _momentum_based_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> AllocationResult:
        """动量基础分配"""
        n_strategies = len(returns.columns)
        lookback = min(constraints.lookback_period, len(returns))
        
        # 计算动量得分
        momentum_scores = np.zeros(n_strategies)
        
        for i, col in enumerate(returns.columns):
            recent_returns = returns[col].iloc[-lookback:]
            
            # 多时间框架动量
            momentum_1m = recent_returns.iloc[-21:].mean() if len(recent_returns) >= 21 else 0
            momentum_3m = recent_returns.iloc[-63:].mean() if len(recent_returns) >= 63 else 0
            momentum_6m = recent_returns.iloc[-126:].mean() if len(recent_returns) >= 126 else 0
            momentum_12m = recent_returns.mean()
            
            # 加权动量得分
            momentum_score = (
                0.4 * momentum_1m + 0.3 * momentum_3m + 
                0.2 * momentum_6m + 0.1 * momentum_12m
            )
            momentum_scores[i] = momentum_score
        
        # 转换为权重 (使用softmax)
        exp_scores = np.exp(momentum_scores * 10)  # 放大差异
        weights = exp_scores / np.sum(exp_scores)
        
        # 计算预期指标
        cov_matrix = returns.cov().values * 252
        expected_return = np.dot(weights, returns.mean().values * 252)
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale="基于多时间框架动量的分配"
        )
    
    async def _mean_reversion_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> AllocationResult:
        """均值回归分配"""
        n_strategies = len(returns.columns)
        lookback = min(constraints.lookback_period, len(returns))
        
        # 计算均值回归得分
        reversion_scores = np.zeros(n_strategies)
        
        for i, col in enumerate(returns.columns):
            recent_returns = returns[col].iloc[-lookback:]
            
            # 计算长期均值
            long_term_mean = recent_returns.mean()
            
            # 计算短期表现
            short_term_return = recent_returns.iloc[-21:].mean() if len(recent_returns) >= 21 else 0
            
            # 均值回归得分 = 长期均值 - 短期表现
            reversion_score = long_term_mean - short_term_return
            reversion_scores[i] = reversion_score
        
        # 转换为权重 (正得分表示低配，负得分表示超配)
        # 使用softmax处理负得分
        exp_scores = np.exp(reversion_scores * 5)
        weights = exp_scores / np.sum(exp_scores)
        
        # 计算预期指标
        cov_matrix = returns.cov().values * 252
        expected_return = np.dot(weights, returns.mean().values * 252)
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale="基于均值回归的分配"
        )
    
    async def _adaptive_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints,
        current_weights: Optional[np.ndarray]
    ) -> AllocationResult:
        """自适应分配"""
        # 结合多种方法
        methods = [
            AllocationMethod.KELLY_CRITERION,
            AllocationMethod.MOMENTUM_BASED,
            AllocationMethod.VOLATILITY_TARGETING
        ]
        
        results = []
        for method in methods:
            try:
                if method == AllocationMethod.KELLY_CRITERION:
                    result = await self._kelly_allocation(returns, constraints)
                elif method == AllocationMethod.MOMENTUM_BASED:
                    result = await self._momentum_based_allocation(returns, constraints)
                elif method == AllocationMethod.VOLATILITY_TARGETING:
                    result = await self._volatility_targeting_allocation(returns, constraints)
                
                results.append(result)
            except Exception as e:
                logger.warning(f"自适应分配中{method.value}失败: {e}")
        
        if not results:
            # 如果所有方法都失败，使用等权重
            n_strategies = len(returns.columns)
            weights = np.ones(n_strategies) / n_strategies
        else:
            # 基于夏普比率加权平均
            sharpe_ratios = np.array([r.expected_sharpe for r in results])
            sharpe_ratios = np.maximum(sharpe_ratios, 0)  # 确保非负
            
            if np.sum(sharpe_ratios) > 0:
                method_weights = sharpe_ratios / np.sum(sharpe_ratios)
            else:
                method_weights = np.ones(len(results)) / len(results)
            
            # 加权平均权重
            weights = np.zeros(len(returns.columns))
            for i, result in enumerate(results):
                weights += method_weights[i] * result.weights
        
        # 计算预期指标
        cov_matrix = returns.cov().values * 252
        expected_return = np.dot(weights, returns.mean().values * 252)
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale="多方法自适应组合分配"
        )
    
    async def _regime_based_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints,
        market_regime: Optional[str]
    ) -> AllocationResult:
        """制度基础分配"""
        # 根据市场制度调整分配策略
        if market_regime == "bull":
            # 牛市：偏向动量策略
            return await self._momentum_based_allocation(returns, constraints)
        elif market_regime == "bear":
            # 熊市：偏向防御性分配
            return await self._volatility_targeting_allocation(returns, constraints)
        elif market_regime == "sideways":
            # 震荡市：偏向均值回归
            return await self._mean_reversion_allocation(returns, constraints)
        else:
            # 未知制度：使用自适应分配
            return await self._adaptive_allocation(returns, constraints, None)
    
    async def _ml_based_allocation(
        self,
        returns: pd.DataFrame,
        constraints: AllocationConstraints,
        additional_features: Optional[pd.DataFrame]
    ) -> AllocationResult:
        """机器学习基础分配"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            n_strategies = len(returns.columns)
            
            # 构建特征
            features = []
            
            # 历史收益率特征
            for i in range(1, 22):  # 过去21天
                if len(returns) > i:
                    features.append(returns.iloc[-i:].mean().values)
            
            # 波动率特征
            features.append(returns.rolling(21).std().iloc[-1].values)
            features.append(returns.rolling(63).std().iloc[-1].values)
            
            # 相关性特征
            corr_matrix = returns.tail(63).corr().values
            features.append(np.mean(corr_matrix, axis=1))
            
            # 额外特征
            if additional_features is not None:
                features.append(additional_features.iloc[-1].values)
            
            # 组合特征矩阵
            X = np.column_stack(features)
            
            # 目标变量：未来收益率
            y = returns.iloc[-21:].mean().values  # 使用最近21天平均收益作为目标
            
            # 训练模型
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # 预测权重
            predicted_returns = model.predict(X_scaled)
            
            # 转换为权重
            exp_returns = np.exp(predicted_returns * 10)
            weights = exp_returns / np.sum(exp_returns)
            
        except Exception as e:
            logger.warning(f"机器学习分配失败: {e}，使用Kelly分配")
            return await self._kelly_allocation(returns, constraints)
        
        # 计算预期指标
        cov_matrix = returns.cov().values * 252
        expected_return = np.dot(weights, returns.mean().values * 252)
        expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        expected_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        # 计算风险贡献
        marginal_contrib = np.dot(cov_matrix, weights) / expected_volatility
        risk_contributions = weights * marginal_contrib
        
        return AllocationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            turnover=0.0,
            risk_contributions=risk_contributions,
            allocation_rationale="基于机器学习的预测分配"
        )
    
    async def _apply_constraints(
        self,
        result: AllocationResult,
        constraints: AllocationConstraints,
        current_weights: Optional[np.ndarray]
    ) -> AllocationResult:
        """应用约束条件"""
        weights = result.weights.copy()
        
        # 应用权重限制
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        
        # 重新标准化
        weights = weights / np.sum(weights)
        
        # 计算换手率
        turnover = 0.0
        if current_weights is not None:
            turnover = np.sum(np.abs(weights - current_weights)) / 2
            
            # 如果换手率超过限制，向当前权重调整
            if turnover > constraints.max_turnover:
                adjustment_factor = constraints.max_turnover / turnover
                weights = current_weights + adjustment_factor * (weights - current_weights)
                weights = weights / np.sum(weights)
                turnover = constraints.max_turnover
        
        # 更新结果
        result.weights = weights
        result.turnover = turnover
        
        return result
    
    def _record_allocation(
        self,
        method: AllocationMethod,
        constraints: AllocationConstraints,
        result: AllocationResult
    ):
        """记录分配历史"""
        record = {
            'timestamp': datetime.now(),
            'method': method.value,
            'weights': result.weights.tolist(),
            'expected_return': result.expected_return,
            'expected_volatility': result.expected_volatility,
            'expected_sharpe': result.expected_sharpe,
            'turnover': result.turnover,
            'allocation_rationale': result.allocation_rationale
        }
        
        self.allocation_history.append(record)
        
        # 保持最近100次记录
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]
    
    def _update_performance_tracker(self, result: AllocationResult):
        """更新性能跟踪"""
        self.performance_tracker['total_allocations'] += 1
        
        if result.expected_sharpe > 0:
            self.performance_tracker['successful_allocations'] += 1
        
        # 更新平均值
        n = self.performance_tracker['total_allocations']
        self.performance_tracker['average_turnover'] = (
            (self.performance_tracker['average_turnover'] * (n-1) + result.turnover) / n
        )
        self.performance_tracker['average_sharpe'] = (
            (self.performance_tracker['average_sharpe'] * (n-1) + result.expected_sharpe) / n
        )
    
    async def get_allocation_history(self) -> List[Dict[str, Any]]:
        """获取分配历史"""
        return self.allocation_history.copy()
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'performance_tracker': self.performance_tracker.copy(),
            'success_rate': (
                self.performance_tracker['successful_allocations'] / 
                max(1, self.performance_tracker['total_allocations'])
            ),
            'recent_allocations': len(self.allocation_history),
            'last_allocation': (
                self.allocation_history[-1]['timestamp'] 
                if self.allocation_history else None
            )
        }


# 全局权重分配器实例
weight_allocator = DynamicWeightAllocator()
