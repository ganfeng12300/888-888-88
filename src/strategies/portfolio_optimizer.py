"""
📊 组合优化器 - 生产级实盘交易投资组合优化和风险管理系统
基于现代投资组合理论和机器学习的智能资产配置优化
提供马科维茨优化、风险平价、Black-Litterman等多种优化算法
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

try:
    from scipy.optimize import minimize
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available, some optimization features will be limited")

class OptimizationMethod(Enum):
    """优化方法"""
    MARKOWITZ = "markowitz"  # 马科维茨均值方差优化
    RISK_PARITY = "risk_parity"  # 风险平价
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman模型
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    MINIMUM_VARIANCE = "minimum_variance"  # 最小方差
    MAXIMUM_SHARPE = "maximum_sharpe"  # 最大夏普比率

class RebalanceFrequency(Enum):
    """再平衡频率"""
    DAILY = "daily"  # 每日
    WEEKLY = "weekly"  # 每周
    MONTHLY = "monthly"  # 每月
    QUARTERLY = "quarterly"  # 每季度

@dataclass
class AssetData:
    """资产数据"""
    symbol: str  # 资产符号
    returns: List[float]  # 收益率序列
    expected_return: float  # 期望收益率
    volatility: float  # 波动率
    last_price: float  # 最新价格
    market_cap: Optional[float] = None  # 市值
    sector: Optional[str] = None  # 行业

@dataclass
class OptimizationConstraints:
    """优化约束"""
    min_weight: float = 0.0  # 最小权重
    max_weight: float = 1.0  # 最大权重
    max_concentration: float = 0.4  # 最大集中度
    min_assets: int = 2  # 最少资产数量
    max_assets: int = 20  # 最多资产数量
    target_return: Optional[float] = None  # 目标收益率
    max_volatility: Optional[float] = None  # 最大波动率

@dataclass
class PortfolioWeights:
    """组合权重"""
    weights: Dict[str, float]  # 资产权重
    expected_return: float  # 期望收益率
    volatility: float  # 组合波动率
    sharpe_ratio: float  # 夏普比率
    optimization_method: OptimizationMethod  # 优化方法
    timestamp: float = field(default_factory=time.time)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

class CovarianceEstimator:
    """协方差矩阵估计器"""
    
    def __init__(self, method: str = "sample", window: int = 252):
        self.method = method
        self.window = window
        
        logger.info(f"协方差估计器初始化: {method}")
    
    def estimate_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """估计协方差矩阵"""
        try:
            if self.method == "sample":
                return self._sample_covariance(returns_data)
            elif self.method == "shrinkage":
                return self._shrinkage_covariance(returns_data)
            elif self.method == "exponential":
                return self._exponential_covariance(returns_data)
            else:
                logger.warning(f"未知的协方差估计方法: {self.method}, 使用样本协方差")
                return self._sample_covariance(returns_data)
        
        except Exception as e:
            logger.error(f"协方差矩阵估计失败: {e}")
            # 返回单位矩阵作为备选
            n = len(returns_data.columns)
            return np.eye(n) * 0.01
    
    def _sample_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """样本协方差矩阵"""
        return returns_data.cov().values * 252  # 年化
    
    def _shrinkage_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """收缩估计协方差矩阵"""
        try:
            sample_cov = returns_data.cov().values * 252
            
            # 简单的收缩目标：对角矩阵
            n = sample_cov.shape[0]
            shrinkage_target = np.eye(n) * np.trace(sample_cov) / n
            
            # 收缩强度
            shrinkage_intensity = 0.2
            
            shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target
            
            return shrunk_cov
        
        except Exception as e:
            logger.error(f"收缩协方差估计失败: {e}")
            return self._sample_covariance(returns_data)
    
    def _exponential_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """指数加权协方差矩阵"""
        try:
            # 指数衰减因子
            decay_factor = 0.94
            
            returns_array = returns_data.values
            n_assets = returns_array.shape[1]
            n_periods = returns_array.shape[0]
            
            # 计算权重
            weights = np.array([decay_factor ** i for i in range(n_periods)])
            weights = weights[::-1]  # 反转，使最新数据权重最大
            weights = weights / weights.sum()
            
            # 计算加权均值
            weighted_mean = np.average(returns_array, axis=0, weights=weights)
            
            # 计算加权协方差
            centered_returns = returns_array - weighted_mean
            weighted_cov = np.zeros((n_assets, n_assets))
            
            for i in range(n_periods):
                outer_product = np.outer(centered_returns[i], centered_returns[i])
                weighted_cov += weights[i] * outer_product
            
            return weighted_cov * 252  # 年化
        
        except Exception as e:
            logger.error(f"指数加权协方差估计失败: {e}")
            return self._sample_covariance(returns_data)

class MarkowitzOptimizer:
    """马科维茨优化器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
        logger.info("马科维茨优化器初始化完成")
    
    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """马科维茨优化"""
        try:
            if not SCIPY_AVAILABLE:
                logger.error("SciPy不可用，无法进行马科维茨优化")
                return None
            
            n_assets = len(expected_returns)
            
            # 目标函数：最小化组合方差
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # 约束条件
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
            ]
            
            # 如果有目标收益率约束
            if constraints.target_return is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.dot(x, expected_returns) - constraints.target_return
                })
            
            # 权重边界
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # 初始权重
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # 优化
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.error(f"马科维茨优化失败: {result.message}")
                return None
        
        except Exception as e:
            logger.error(f"马科维茨优化失败: {e}")
            return None
    
    def efficient_frontier(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                          constraints: OptimizationConstraints, n_points: int = 50) -> List[Tuple[float, float]]:
        """计算有效前沿"""
        try:
            if not SCIPY_AVAILABLE:
                return []
            
            min_ret = np.min(expected_returns)
            max_ret = np.max(expected_returns)
            
            target_returns = np.linspace(min_ret, max_ret, n_points)
            efficient_portfolios = []
            
            for target_ret in target_returns:
                constraints.target_return = target_ret
                weights = self.optimize(expected_returns, cov_matrix, constraints)
                
                if weights is not None:
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                    efficient_portfolios.append((portfolio_risk, portfolio_return))
            
            return efficient_portfolios
        
        except Exception as e:
            logger.error(f"计算有效前沿失败: {e}")
            return []

class RiskParityOptimizer:
    """风险平价优化器"""
    
    def __init__(self):
        logger.info("风险平价优化器初始化完成")
    
    def optimize(self, cov_matrix: np.ndarray, constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """风险平价优化"""
        try:
            if not SCIPY_AVAILABLE:
                logger.error("SciPy不可用，无法进行风险平价优化")
                return None
            
            n_assets = cov_matrix.shape[0]
            
            # 目标函数：最小化风险贡献的平方差
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # 约束条件
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
            ]
            
            # 权重边界
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # 初始权重
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # 优化
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.error(f"风险平价优化失败: {result.message}")
                return None
        
        except Exception as e:
            logger.error(f"风险平价优化失败: {e}")
            return None

class PortfolioOptimizer:
    """组合优化器主类"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.covariance_estimator = CovarianceEstimator()
        self.markowitz_optimizer = MarkowitzOptimizer(risk_free_rate)
        self.risk_parity_optimizer = RiskParityOptimizer()
        
        # 历史数据
        self.asset_data: Dict[str, AssetData] = {}
        self.optimization_history: List[PortfolioWeights] = []
        
        # 再平衡设置
        self.rebalance_frequency = RebalanceFrequency.MONTHLY
        self.last_rebalance_time = 0
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("组合优化器初始化完成")
    
    def add_asset_data(self, asset_data: AssetData):
        """添加资产数据"""
        try:
            with self.lock:
                self.asset_data[asset_data.symbol] = asset_data
                logger.info(f"添加资产数据: {asset_data.symbol}")
        
        except Exception as e:
            logger.error(f"添加资产数据失败: {e}")
    
    def update_asset_returns(self, symbol: str, new_return: float):
        """更新资产收益率"""
        try:
            with self.lock:
                if symbol in self.asset_data:
                    asset = self.asset_data[symbol]
                    asset.returns.append(new_return)
                    
                    # 保持收益率序列在合理长度
                    if len(asset.returns) > 1000:
                        asset.returns = asset.returns[-500:]
                    
                    # 更新统计数据
                    asset.expected_return = np.mean(asset.returns[-252:])  # 最近一年
                    asset.volatility = np.std(asset.returns[-252:]) * np.sqrt(252)
        
        except Exception as e:
            logger.error(f"更新资产收益率失败: {e}")
    
    def optimize_portfolio(self, method: OptimizationMethod,
                          constraints: OptimizationConstraints,
                          selected_assets: Optional[List[str]] = None) -> Optional[PortfolioWeights]:
        """优化投资组合"""
        try:
            with self.lock:
                # 选择资产
                if selected_assets is None:
                    selected_assets = list(self.asset_data.keys())
                
                if len(selected_assets) < constraints.min_assets:
                    logger.error(f"资产数量不足: {len(selected_assets)} < {constraints.min_assets}")
                    return None
                
                # 准备数据
                expected_returns = np.array([
                    self.asset_data[symbol].expected_return 
                    for symbol in selected_assets
                ])
                
                # 构建收益率矩阵
                returns_data = pd.DataFrame({
                    symbol: self.asset_data[symbol].returns[-252:]  # 最近一年
                    for symbol in selected_assets
                    if len(self.asset_data[symbol].returns) >= 30
                })
                
                if returns_data.empty:
                    logger.error("没有足够的收益率数据")
                    return None
                
                # 估计协方差矩阵
                cov_matrix = self.covariance_estimator.estimate_covariance(returns_data)
                
                # 执行优化
                weights = None
                
                if method == OptimizationMethod.MARKOWITZ:
                    weights = self.markowitz_optimizer.optimize(expected_returns, cov_matrix, constraints)
                elif method == OptimizationMethod.RISK_PARITY:
                    weights = self.risk_parity_optimizer.optimize(cov_matrix, constraints)
                elif method == OptimizationMethod.EQUAL_WEIGHT:
                    weights = self._equal_weight_optimization(len(selected_assets))
                elif method == OptimizationMethod.MINIMUM_VARIANCE:
                    weights = self._minimum_variance_optimization(cov_matrix, constraints)
                elif method == OptimizationMethod.MAXIMUM_SHARPE:
                    weights = self._maximum_sharpe_optimization(expected_returns, cov_matrix, constraints)
                
                if weights is None:
                    return None
                
                # 计算组合统计
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                # 创建权重字典
                weights_dict = {symbol: weight for symbol, weight in zip(selected_assets, weights)}
                
                # 创建组合权重对象
                portfolio_weights = PortfolioWeights(
                    weights=weights_dict,
                    expected_return=portfolio_return,
                    volatility=portfolio_volatility,
                    sharpe_ratio=sharpe_ratio,
                    optimization_method=method
                )
                
                # 添加到历史记录
                self.optimization_history.append(portfolio_weights)
                
                # 保持历史记录在合理范围内
                if len(self.optimization_history) > 1000:
                    self.optimization_history = self.optimization_history[-500:]
                
                logger.info(f"组合优化完成: {method.value} - 夏普比率 {sharpe_ratio:.3f}")
                
                return portfolio_weights
        
        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            return None
    
    def _equal_weight_optimization(self, n_assets: int) -> np.ndarray:
        """等权重优化"""
        return np.array([1.0 / n_assets] * n_assets)
    
    def _minimum_variance_optimization(self, cov_matrix: np.ndarray,
                                     constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """最小方差优化"""
        try:
            if not SCIPY_AVAILABLE:
                return None
            
            n_assets = cov_matrix.shape[0]
            
            # 目标函数：最小化组合方差
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # 约束条件
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
            ]
            
            # 权重边界
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # 初始权重
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # 优化
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            return result.x if result.success else None
        
        except Exception as e:
            logger.error(f"最小方差优化失败: {e}")
            return None
    
    def _maximum_sharpe_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                   constraints: OptimizationConstraints) -> Optional[np.ndarray]:
        """最大夏普比率优化"""
        try:
            if not SCIPY_AVAILABLE:
                return None
            
            n_assets = len(expected_returns)
            
            # 目标函数：最大化夏普比率（最小化负夏普比率）
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                if portfolio_volatility == 0:
                    return -np.inf
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # 最小化负值
            
            # 约束条件
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
            ]
            
            # 权重边界
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # 初始权重
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # 优化
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            return result.x if result.success else None
        
        except Exception as e:
            logger.error(f"最大夏普比率优化失败: {e}")
            return None
    
    def should_rebalance(self) -> bool:
        """检查是否需要再平衡"""
        try:
            current_time = time.time()
            
            if self.rebalance_frequency == RebalanceFrequency.DAILY:
                interval = 86400  # 1天
            elif self.rebalance_frequency == RebalanceFrequency.WEEKLY:
                interval = 86400 * 7  # 7天
            elif self.rebalance_frequency == RebalanceFrequency.MONTHLY:
                interval = 86400 * 30  # 30天
            elif self.rebalance_frequency == RebalanceFrequency.QUARTERLY:
                interval = 86400 * 90  # 90天
            else:
                interval = 86400 * 30  # 默认30天
            
            return current_time - self.last_rebalance_time > interval
        
        except Exception as e:
            logger.error(f"检查再平衡失败: {e}")
            return False
    
    def set_rebalance_frequency(self, frequency: RebalanceFrequency):
        """设置再平衡频率"""
        self.rebalance_frequency = frequency
        logger.info(f"设置再平衡频率: {frequency.value}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        try:
            with self.lock:
                if not self.optimization_history:
                    return {}
                
                latest_optimization = self.optimization_history[-1]
                
                # 统计不同优化方法的使用次数
                method_counts = {}
                for opt in self.optimization_history[-50:]:  # 最近50次
                    method = opt.optimization_method.value
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                return {
                    'total_optimizations': len(self.optimization_history),
                    'latest_optimization': {
                        'method': latest_optimization.optimization_method.value,
                        'expected_return': latest_optimization.expected_return,
                        'volatility': latest_optimization.volatility,
                        'sharpe_ratio': latest_optimization.sharpe_ratio,
                        'weights': latest_optimization.weights,
                        'timestamp': latest_optimization.timestamp
                    },
                    'method_usage': method_counts,
                    'total_assets': len(self.asset_data),
                    'rebalance_frequency': self.rebalance_frequency.value,
                    'should_rebalance': self.should_rebalance()
                }
        
        except Exception as e:
            logger.error(f"获取优化摘要失败: {e}")
            return {}
    
    def get_latest_weights(self) -> Optional[PortfolioWeights]:
        """获取最新的组合权重"""
        with self.lock:
            return self.optimization_history[-1] if self.optimization_history else None

# 全局组合优化器实例（需要在使用前初始化）
portfolio_optimizer = None

def initialize_portfolio_optimizer(risk_free_rate: float = 0.02):
    """初始化组合优化器"""
    global portfolio_optimizer
    portfolio_optimizer = PortfolioOptimizer(risk_free_rate)
    return portfolio_optimizer
