"""
🔧 投资组合优化器
生产级多策略组合优化系统，实现马科维茨优化、风险平价、Kelly公式等完整算法
支持实时权重调整、风险约束和性能监控
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.optimize as sco
from scipy import linalg
import cvxpy as cp
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings


class OptimizationMethod(Enum):
    """优化方法枚举"""
    MARKOWITZ = "markowitz"              # 马科维茨均值方差优化
    RISK_PARITY = "risk_parity"          # 风险平价
    KELLY_CRITERION = "kelly_criterion"   # Kelly公式
    BLACK_LITTERMAN = "black_litterman"   # Black-Litterman模型
    HIERARCHICAL_RISK_PARITY = "hrp"     # 层次风险平价
    MINIMUM_VARIANCE = "min_variance"     # 最小方差
    MAXIMUM_SHARPE = "max_sharpe"        # 最大夏普比率
    EQUAL_WEIGHT = "equal_weight"        # 等权重


@dataclass
class OptimizationConstraints:
    """优化约束条件"""
    min_weight: float = 0.0              # 最小权重
    max_weight: float = 1.0              # 最大权重
    max_concentration: float = 0.4       # 最大集中度
    target_return: Optional[float] = None # 目标收益率
    max_volatility: Optional[float] = None# 最大波动率
    max_drawdown: float = 0.2            # 最大回撤限制
    turnover_limit: float = 0.5          # 换手率限制
    sector_limits: Dict[str, float] = field(default_factory=dict)  # 行业限制


@dataclass
class PortfolioMetrics:
    """组合性能指标"""
    weights: np.ndarray                  # 权重向量
    expected_return: float               # 预期收益率
    volatility: float                    # 波动率
    sharpe_ratio: float                  # 夏普比率
    sortino_ratio: float                 # Sortino比率
    max_drawdown: float                  # 最大回撤
    var_95: float                        # 95% VaR
    cvar_95: float                       # 95% CVaR
    diversification_ratio: float         # 分散化比率
    concentration_index: float           # 集中度指数
    turnover: float                      # 换手率
    tracking_error: Optional[float] = None # 跟踪误差


class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 无风险利率2%
        self.lookback_period = 252  # 回望期252个交易日
        self.rebalance_frequency = 5  # 5天重新平衡一次
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 优化历史记录
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("投资组合优化器初始化完成")
    
    async def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MARKOWITZ,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None,
        market_views: Optional[Dict[str, float]] = None
    ) -> PortfolioMetrics:
        """
        优化投资组合
        
        Args:
            returns: 收益率数据 (DataFrame)
            method: 优化方法
            constraints: 约束条件
            current_weights: 当前权重
            market_views: 市场观点 (用于Black-Litterman)
        
        Returns:
            PortfolioMetrics: 优化后的组合指标
        """
        try:
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # 数据预处理
            returns_clean = self._preprocess_returns(returns)
            
            # 计算协方差矩阵和期望收益
            cov_matrix = self._calculate_covariance_matrix(returns_clean)
            expected_returns = self._calculate_expected_returns(returns_clean, method)
            
            # 根据方法选择优化算法
            if method == OptimizationMethod.MARKOWITZ:
                weights = await self._markowitz_optimization(
                    expected_returns, cov_matrix, constraints
                )
            elif method == OptimizationMethod.RISK_PARITY:
                weights = await self._risk_parity_optimization(
                    cov_matrix, constraints
                )
            elif method == OptimizationMethod.KELLY_CRITERION:
                weights = await self._kelly_optimization(
                    returns_clean, constraints
                )
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = await self._black_litterman_optimization(
                    returns_clean, cov_matrix, market_views, constraints
                )
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                weights = await self._hrp_optimization(
                    returns_clean, constraints
                )
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = await self._minimum_variance_optimization(
                    cov_matrix, constraints
                )
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                weights = await self._maximum_sharpe_optimization(
                    expected_returns, cov_matrix, constraints
                )
            else:  # EQUAL_WEIGHT
                weights = await self._equal_weight_optimization(
                    len(returns_clean.columns), constraints
                )
            
            # 计算组合性能指标
            metrics = self._calculate_portfolio_metrics(
                weights, expected_returns, cov_matrix, returns_clean, current_weights
            )
            
            # 记录优化历史
            self._record_optimization(method, constraints, metrics)
            
            logger.info(f"组合优化完成 - 方法: {method.value}, 夏普比率: {metrics.sharpe_ratio:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            raise
    
    def _preprocess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """预处理收益率数据"""
        # 去除缺失值
        returns_clean = returns.dropna()
        
        # 去除异常值 (3倍标准差)
        for col in returns_clean.columns:
            mean = returns_clean[col].mean()
            std = returns_clean[col].std()
            returns_clean[col] = returns_clean[col].clip(
                lower=mean - 3*std, upper=mean + 3*std
            )
        
        # 确保有足够的数据
        if len(returns_clean) < 30:
            raise ValueError("收益率数据不足，至少需要30个观测值")
        
        return returns_clean
    
    def _calculate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """计算协方差矩阵"""
        # 使用指数加权移动平均
        lambda_decay = 0.94
        weights = np.array([lambda_decay**i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        # 计算加权协方差矩阵
        returns_centered = returns - returns.mean()
        cov_matrix = np.cov(returns_centered.T, aweights=weights)
        
        # 确保协方差矩阵正定
        eigenvals, eigenvecs = linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return cov_matrix
    
    def _calculate_expected_returns(
        self, 
        returns: pd.DataFrame, 
        method: OptimizationMethod
    ) -> np.ndarray:
        """计算期望收益率"""
        if method in [OptimizationMethod.MINIMUM_VARIANCE, OptimizationMethod.RISK_PARITY]:
            # 最小方差和风险平价不需要期望收益
            return np.zeros(len(returns.columns))
        
        # 使用指数加权移动平均计算期望收益
        lambda_decay = 0.94
        weights = np.array([lambda_decay**i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        expected_returns = np.average(returns.values, axis=0, weights=weights)
        
        # 年化收益率
        expected_returns = expected_returns * 252
        
        return expected_returns
    
    async def _markowitz_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """马科维茨均值方差优化"""
        n_assets = len(expected_returns)
        
        # 定义优化变量
        w = cp.Variable(n_assets)
        
        # 目标函数：最大化效用 (收益 - 风险厌恶系数 * 方差)
        risk_aversion = 1.0
        portfolio_return = expected_returns.T @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        
        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,  # 权重和为1
            w >= constraints.min_weight,  # 最小权重
            w <= constraints.max_weight   # 最大权重
        ]
        
        # 目标收益约束
        if constraints.target_return is not None:
            constraints_list.append(portfolio_return >= constraints.target_return)
        
        # 最大波动率约束
        if constraints.max_volatility is not None:
            constraints_list.append(
                cp.sqrt(portfolio_variance) <= constraints.max_volatility
            )
        
        # 最大集中度约束
        if constraints.max_concentration < 1.0:
            constraints_list.append(w <= constraints.max_concentration)
        
        # 求解优化问题
        problem = cp.Problem(cp.Maximize(utility), constraints_list)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["infeasible", "unbounded"]:
            return w.value
        else:
            logger.warning("马科维茨优化失败，使用等权重")
            return np.ones(n_assets) / n_assets
    
    async def _risk_parity_optimization(
        self,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """风险平价优化"""
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            """风险平价目标函数"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # 计算边际风险贡献
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            # 目标：最小化风险贡献的差异
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints_opt = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = sco.minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("风险平价优化失败，使用等权重")
            return np.ones(n_assets) / n_assets
    
    async def _kelly_optimization(
        self,
        returns: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Kelly公式优化"""
        n_assets = len(returns.columns)
        
        def kelly_objective(weights):
            """Kelly公式目标函数"""
            weights = np.array(weights)
            
            # 计算组合收益率
            portfolio_returns = (returns.values @ weights)
            
            # Kelly公式：最大化对数期望收益
            # 避免负收益导致的对数问题
            portfolio_returns = np.maximum(portfolio_returns, -0.99)
            log_returns = np.log(1 + portfolio_returns)
            
            return -np.mean(log_returns)  # 负号因为要最大化
        
        # 约束条件
        constraints_opt = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = sco.minimize(
            kelly_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("Kelly优化失败，使用等权重")
            return np.ones(n_assets) / n_assets
    
    async def _black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray,
        market_views: Optional[Dict[str, float]],
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Black-Litterman模型优化"""
        n_assets = len(returns.columns)
        
        # 市场均衡收益率 (使用历史收益率)
        market_returns = returns.mean().values * 252
        
        # 如果没有市场观点，使用市场均衡
        if market_views is None:
            expected_returns = market_returns
        else:
            # 构建观点矩阵
            P = np.zeros((len(market_views), n_assets))
            Q = np.zeros(len(market_views))
            
            for i, (asset, view) in enumerate(market_views.items()):
                if asset in returns.columns:
                    asset_idx = returns.columns.get_loc(asset)
                    P[i, asset_idx] = 1
                    Q[i] = view
            
            # Black-Litterman公式
            tau = 0.025  # 不确定性参数
            omega = np.eye(len(market_views)) * 0.01  # 观点不确定性
            
            # 计算新的期望收益率
            M1 = linalg.inv(tau * cov_matrix)
            M2 = P.T @ linalg.inv(omega) @ P
            M3 = linalg.inv(M1 + M2)
            
            mu_bl = M3 @ (M1 @ market_returns + P.T @ linalg.inv(omega) @ Q)
            expected_returns = mu_bl
        
        # 使用马科维茨优化
        return await self._markowitz_optimization(expected_returns, cov_matrix, constraints)
    
    async def _hrp_optimization(
        self,
        returns: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """层次风险平价优化"""
        # 计算相关性矩阵
        corr_matrix = returns.corr().values
        
        # 计算距离矩阵
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # 层次聚类
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # 转换为距离向量
        distance_vector = squareform(distance_matrix, checks=False)
        
        # 执行聚类
        linkage_matrix = linkage(distance_vector, method='single')
        
        # 递归二分法分配权重
        def _get_cluster_var(cov_matrix, cluster_items):
            """计算聚类方差"""
            cluster_cov = cov_matrix[np.ix_(cluster_items, cluster_items)]
            inv_diag = 1 / np.diag(cluster_cov)
            weights = inv_diag / inv_diag.sum()
            cluster_var = weights.T @ cluster_cov @ weights
            return cluster_var
        
        def _get_quasi_diag(linkage_matrix):
            """获取准对角化排序"""
            link = linkage_matrix.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = sort_ix.append(df0).sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            
            return sort_ix.tolist()
        
        def _get_rec_bipart(cov_matrix, sort_ix):
            """递归二分法"""
            weights = pd.Series(1, index=sort_ix)
            cluster_items = [sort_ix]
            
            while len(cluster_items) > 0:
                cluster_items = [
                    i[j:k] for i in cluster_items
                    for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                    if len(i) > 1
                ]
                
                for i in range(0, len(cluster_items), 2):
                    cluster0 = cluster_items[i]
                    cluster1 = cluster_items[i + 1]
                    
                    cluster_var0 = _get_cluster_var(cov_matrix, cluster0)
                    cluster_var1 = _get_cluster_var(cov_matrix, cluster1)
                    
                    alpha = 1 - cluster_var0 / (cluster_var0 + cluster_var1)
                    
                    weights[cluster0] *= alpha
                    weights[cluster1] *= 1 - alpha
            
            return weights.values
        
        # 计算协方差矩阵
        cov_matrix = returns.cov().values * 252
        
        # 获取排序
        sort_ix = _get_quasi_diag(linkage_matrix)
        
        # 计算HRP权重
        hrp_weights = _get_rec_bipart(cov_matrix, sort_ix)
        
        # 重新排序到原始顺序
        weights = np.zeros(len(returns.columns))
        for i, idx in enumerate(sort_ix):
            weights[idx] = hrp_weights[i]
        
        return weights
    
    async def _minimum_variance_optimization(
        self,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """最小方差优化"""
        n_assets = len(cov_matrix)
        
        # 定义优化变量
        w = cp.Variable(n_assets)
        
        # 目标函数：最小化方差
        portfolio_variance = cp.quad_form(w, cov_matrix)
        
        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,  # 权重和为1
            w >= constraints.min_weight,  # 最小权重
            w <= constraints.max_weight   # 最大权重
        ]
        
        # 最大集中度约束
        if constraints.max_concentration < 1.0:
            constraints_list.append(w <= constraints.max_concentration)
        
        # 求解优化问题
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["infeasible", "unbounded"]:
            return w.value
        else:
            logger.warning("最小方差优化失败，使用等权重")
            return np.ones(n_assets) / n_assets
    
    async def _maximum_sharpe_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """最大夏普比率优化"""
        n_assets = len(expected_returns)
        
        # 定义优化变量
        w = cp.Variable(n_assets)
        
        # 目标函数：最大化夏普比率
        # 使用二次规划形式
        portfolio_return = expected_returns.T @ w - self.risk_free_rate
        portfolio_variance = cp.quad_form(w, cov_matrix)
        
        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,  # 权重和为1
            w >= constraints.min_weight,  # 最小权重
            w <= constraints.max_weight,  # 最大权重
            portfolio_return >= 0.01  # 最小超额收益
        ]
        
        # 最大集中度约束
        if constraints.max_concentration < 1.0:
            constraints_list.append(w <= constraints.max_concentration)
        
        # 求解优化问题 (最大化收益/风险比)
        problem = cp.Problem(
            cp.Maximize(portfolio_return / cp.sqrt(portfolio_variance)),
            constraints_list
        )
        
        try:
            problem.solve(solver=cp.ECOS)
            if problem.status not in ["infeasible", "unbounded"]:
                return w.value
        except:
            pass
        
        # 如果失败，使用替代方法
        logger.warning("最大夏普比率优化失败，使用马科维茨优化")
        return await self._markowitz_optimization(expected_returns, cov_matrix, constraints)
    
    async def _equal_weight_optimization(
        self,
        n_assets: int,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """等权重优化"""
        return np.ones(n_assets) / n_assets
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """计算组合性能指标"""
        # 基本指标
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 夏普比率
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # 计算历史组合收益率
        portfolio_returns = (returns.values @ weights)
        
        # Sortino比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
        sortino_ratio = (portfolio_return - self.risk_free_rate) / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # VaR和CVaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # 分散化比率
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        diversification_ratio = weighted_avg_vol / portfolio_volatility
        
        # 集中度指数 (HHI)
        concentration_index = np.sum(weights**2)
        
        # 换手率
        turnover = 0.0
        if current_weights is not None:
            turnover = np.sum(np.abs(weights - current_weights)) / 2
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            diversification_ratio=diversification_ratio,
            concentration_index=concentration_index,
            turnover=turnover
        )
    
    def _record_optimization(
        self,
        method: OptimizationMethod,
        constraints: OptimizationConstraints,
        metrics: PortfolioMetrics
    ):
        """记录优化历史"""
        record = {
            'timestamp': datetime.now(),
            'method': method.value,
            'constraints': constraints,
            'metrics': metrics,
            'weights': metrics.weights.tolist(),
            'sharpe_ratio': metrics.sharpe_ratio,
            'volatility': metrics.volatility,
            'expected_return': metrics.expected_return
        }
        
        self.optimization_history.append(record)
        
        # 保持最近100次记录
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    async def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_history.copy()
    
    async def compare_methods(
        self,
        returns: pd.DataFrame,
        methods: List[OptimizationMethod],
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, PortfolioMetrics]:
        """比较不同优化方法"""
        results = {}
        
        for method in methods:
            try:
                metrics = await self.optimize_portfolio(returns, method, constraints)
                results[method.value] = metrics
                logger.info(f"{method.value} - 夏普比率: {metrics.sharpe_ratio:.3f}")
            except Exception as e:
                logger.error(f"{method.value} 优化失败: {e}")
        
        return results
    
    async def shutdown(self):
        """关闭优化器"""
        self.executor.shutdown(wait=True)
        logger.info("投资组合优化器已关闭")


# 全局优化器实例
portfolio_optimizer = PortfolioOptimizer()

