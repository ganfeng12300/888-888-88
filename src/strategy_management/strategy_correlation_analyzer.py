"""
📊 策略相关性分析器
生产级策略相关性分析系统，实现滚动相关性、协方差矩阵、因子分解等完整分析
支持实时相关性监控、策略聚类和风险分散度评估
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from src.core.config import settings


class CorrelationMethod(Enum):
    """相关性计算方法"""
    PEARSON = "pearson"          # 皮尔逊相关系数
    SPEARMAN = "spearman"        # 斯皮尔曼等级相关
    KENDALL = "kendall"          # 肯德尔τ相关
    DISTANCE = "distance"        # 距离相关性
    MUTUAL_INFO = "mutual_info"  # 互信息
    COPULA = "copula"           # Copula相关性


class ClusteringMethod(Enum):
    """聚类方法"""
    HIERARCHICAL = "hierarchical"  # 层次聚类
    KMEANS = "kmeans"             # K均值聚类
    DBSCAN = "dbscan"             # DBSCAN聚类
    SPECTRAL = "spectral"         # 谱聚类


@dataclass
class CorrelationMetrics:
    """相关性指标"""
    correlation_matrix: np.ndarray           # 相关性矩阵
    average_correlation: float               # 平均相关性
    max_correlation: float                   # 最大相关性
    min_correlation: float                   # 最小相关性
    correlation_stability: float             # 相关性稳定性
    diversification_ratio: float             # 分散化比率
    effective_strategies: float              # 有效策略数量
    correlation_clusters: Dict[int, List[str]]  # 相关性聚类
    eigenvalues: np.ndarray                  # 特征值
    explained_variance_ratio: np.ndarray     # 解释方差比
    factor_loadings: Optional[np.ndarray] = None  # 因子载荷


@dataclass
class RollingCorrelationResult:
    """滚动相关性结果"""
    timestamps: List[datetime]               # 时间戳
    correlations: np.ndarray                # 滚动相关性矩阵
    average_correlations: List[float]        # 平均相关性时间序列
    correlation_trends: Dict[str, List[float]]  # 相关性趋势
    regime_changes: List[datetime]           # 制度变化点
    stability_scores: List[float]            # 稳定性得分


class StrategyCorrelationAnalyzer:
    """策略相关性分析器"""
    
    def __init__(self):
        self.lookback_window = 252      # 回望窗口252个交易日
        self.rolling_window = 60        # 滚动窗口60个交易日
        self.min_periods = 30           # 最小观测期数
        self.correlation_threshold = 0.7 # 高相关性阈值
        
        # 分析历史记录
        self.analysis_history: List[Dict[str, Any]] = []
        
        logger.info("策略相关性分析器初始化完成")
    
    async def analyze_strategy_correlations(
        self,
        strategy_returns: pd.DataFrame,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        rolling_analysis: bool = True
    ) -> Tuple[CorrelationMetrics, Optional[RollingCorrelationResult]]:
        """
        分析策略相关性
        
        Args:
            strategy_returns: 策略收益率数据
            method: 相关性计算方法
            rolling_analysis: 是否进行滚动分析
        
        Returns:
            Tuple[CorrelationMetrics, RollingCorrelationResult]: 相关性指标和滚动分析结果
        """
        try:
            # 数据预处理
            returns_clean = self._preprocess_returns(strategy_returns)
            
            # 计算相关性矩阵
            correlation_matrix = await self._calculate_correlation_matrix(
                returns_clean, method
            )
            
            # 计算相关性指标
            metrics = await self._calculate_correlation_metrics(
                correlation_matrix, returns_clean
            )
            
            # 滚动相关性分析
            rolling_result = None
            if rolling_analysis and len(returns_clean) >= self.rolling_window * 2:
                rolling_result = await self._rolling_correlation_analysis(
                    returns_clean, method
                )
            
            # 记录分析历史
            self._record_analysis(method, metrics, rolling_result)
            
            logger.info(f"相关性分析完成 - 平均相关性: {metrics.average_correlation:.3f}")
            return metrics, rolling_result
            
        except Exception as e:
            logger.error(f"相关性分析失败: {e}")
            raise
    
    def _preprocess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """预处理收益率数据"""
        # 去除缺失值
        returns_clean = returns.dropna()
        
        # 确保有足够的数据
        if len(returns_clean) < self.min_periods:
            raise ValueError(f"数据不足，至少需要{self.min_periods}个观测值")
        
        # 去除异常值 (5倍标准差)
        for col in returns_clean.columns:
            mean = returns_clean[col].mean()
            std = returns_clean[col].std()
            returns_clean[col] = returns_clean[col].clip(
                lower=mean - 5*std, upper=mean + 5*std
            )
        
        return returns_clean
    
    async def _calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: CorrelationMethod
    ) -> np.ndarray:
        """计算相关性矩阵"""
        if method == CorrelationMethod.PEARSON:
            return returns.corr(method='pearson').values
        
        elif method == CorrelationMethod.SPEARMAN:
            return returns.corr(method='spearman').values
        
        elif method == CorrelationMethod.KENDALL:
            return returns.corr(method='kendall').values
        
        elif method == CorrelationMethod.DISTANCE:
            return await self._distance_correlation_matrix(returns)
        
        elif method == CorrelationMethod.MUTUAL_INFO:
            return await self._mutual_information_matrix(returns)
        
        elif method == CorrelationMethod.COPULA:
            return await self._copula_correlation_matrix(returns)
        
        else:
            return returns.corr(method='pearson').values
    
    async def _distance_correlation_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """计算距离相关性矩阵"""
        n_strategies = len(returns.columns)
        distance_corr_matrix = np.eye(n_strategies)
        
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                x = returns.iloc[:, i].values
                y = returns.iloc[:, j].values
                
                # 计算距离相关性
                dcorr = self._distance_correlation(x, y)
                distance_corr_matrix[i, j] = dcorr
                distance_corr_matrix[j, i] = dcorr
        
        return distance_corr_matrix
    
    def _distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算距离相关性"""
        n = len(x)
        
        # 计算距离矩阵
        a = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        b = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
        
        # 中心化距离矩阵
        A = a - a.mean(axis=0) - a.mean(axis=1)[:, np.newaxis] + a.mean()
        B = b - b.mean(axis=0) - b.mean(axis=1)[:, np.newaxis] + b.mean()
        
        # 计算距离协方差和方差
        dcov_xy = np.sqrt(np.mean(A * B))
        dcov_xx = np.sqrt(np.mean(A * A))
        dcov_yy = np.sqrt(np.mean(B * B))
        
        # 距离相关性
        if dcov_xx > 0 and dcov_yy > 0:
            return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            return 0.0
    
    async def _mutual_information_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """计算互信息矩阵"""
        from sklearn.feature_selection import mutual_info_regression
        
        n_strategies = len(returns.columns)
        mi_matrix = np.eye(n_strategies)
        
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                x = returns.iloc[:, i].values.reshape(-1, 1)
                y = returns.iloc[:, j].values
                
                # 计算互信息
                mi = mutual_info_regression(x, y, random_state=42)[0]
                
                # 标准化到[-1, 1]范围
                mi_normalized = 2 * mi / (np.var(returns.iloc[:, i]) + np.var(returns.iloc[:, j])) - 1
                mi_normalized = np.clip(mi_normalized, -1, 1)
                
                mi_matrix[i, j] = mi_normalized
                mi_matrix[j, i] = mi_normalized
        
        return mi_matrix
    
    async def _copula_correlation_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """计算Copula相关性矩阵"""
        from scipy.stats import rankdata
        
        # 转换为均匀分布 (经验分布函数)
        n_obs, n_strategies = returns.shape
        uniform_data = np.zeros_like(returns.values)
        
        for i in range(n_strategies):
            ranks = rankdata(returns.iloc[:, i])
            uniform_data[:, i] = ranks / (n_obs + 1)
        
        # 转换为标准正态分布
        from scipy.stats import norm
        normal_data = norm.ppf(uniform_data)
        
        # 计算正态Copula相关性
        copula_corr = np.corrcoef(normal_data.T)
        
        return copula_corr
    
    async def _calculate_correlation_metrics(
        self,
        correlation_matrix: np.ndarray,
        returns: pd.DataFrame
    ) -> CorrelationMetrics:
        """计算相关性指标"""
        # 去除对角线元素
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        correlations = correlation_matrix[mask]
        
        # 基本统计量
        average_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)
        min_correlation = np.min(correlations)
        
        # 相关性稳定性 (标准差的倒数)
        correlation_stability = 1 / (np.std(correlations) + 1e-8)
        
        # 分散化比率
        weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_vol = np.sqrt(weights.T @ (returns.cov().values * 252) @ weights)
        individual_vols = np.sqrt(np.diag(returns.cov().values * 252))
        weighted_avg_vol = np.dot(weights, individual_vols)
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        # 有效策略数量 (基于相关性)
        eigenvalues, _ = np.linalg.eigh(correlation_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        effective_strategies = np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-8)))
        
        # 相关性聚类
        correlation_clusters = await self._cluster_strategies(
            correlation_matrix, returns.columns.tolist()
        )
        
        # PCA分析
        pca = PCA()
        pca.fit(returns.values)
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # 因子分析
        try:
            n_factors = min(5, len(returns.columns) - 1)
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa.fit(returns.values)
            factor_loadings = fa.components_
        except:
            factor_loadings = None
        
        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            average_correlation=average_correlation,
            max_correlation=max_correlation,
            min_correlation=min_correlation,
            correlation_stability=correlation_stability,
            diversification_ratio=diversification_ratio,
            effective_strategies=effective_strategies,
            correlation_clusters=correlation_clusters,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained_variance_ratio,
            factor_loadings=factor_loadings
        )
    
    async def _cluster_strategies(
        self,
        correlation_matrix: np.ndarray,
        strategy_names: List[str],
        method: ClusteringMethod = ClusteringMethod.HIERARCHICAL
    ) -> Dict[int, List[str]]:
        """策略聚类"""
        if method == ClusteringMethod.HIERARCHICAL:
            # 转换相关性为距离
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
            
            # 层次聚类
            distance_vector = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_vector, method='ward')
            
            # 确定聚类数量 (基于相关性阈值)
            n_clusters = max(2, min(len(strategy_names) // 2, 5))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(strategy_names[i])
            
            return clusters
        
        else:
            # 其他聚类方法的实现
            from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
            
            if method == ClusteringMethod.KMEANS:
                n_clusters = max(2, min(len(strategy_names) // 2, 5))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(correlation_matrix)
            
            elif method == ClusteringMethod.DBSCAN:
                dbscan = DBSCAN(eps=0.3, min_samples=2)
                cluster_labels = dbscan.fit_predict(correlation_matrix)
            
            elif method == ClusteringMethod.SPECTRAL:
                n_clusters = max(2, min(len(strategy_names) // 2, 5))
                spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
                cluster_labels = spectral.fit_predict(correlation_matrix)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(strategy_names[i])
            
            return clusters
    
    async def _rolling_correlation_analysis(
        self,
        returns: pd.DataFrame,
        method: CorrelationMethod
    ) -> RollingCorrelationResult:
        """滚动相关性分析"""
        timestamps = []
        correlations = []
        average_correlations = []
        
        # 滚动计算相关性
        for i in range(self.rolling_window, len(returns)):
            window_data = returns.iloc[i-self.rolling_window:i]
            timestamp = returns.index[i]
            
            # 计算窗口相关性矩阵
            corr_matrix = await self._calculate_correlation_matrix(window_data, method)
            
            # 计算平均相关性
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.mean(corr_matrix[mask])
            
            timestamps.append(timestamp)
            correlations.append(corr_matrix)
            average_correlations.append(avg_corr)
        
        correlations = np.array(correlations)
        
        # 计算相关性趋势
        correlation_trends = {}
        n_strategies = len(returns.columns)
        
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                pair_name = f"{returns.columns[i]}-{returns.columns[j]}"
                correlation_trends[pair_name] = correlations[:, i, j].tolist()
        
        # 检测制度变化点
        regime_changes = await self._detect_regime_changes(
            np.array(average_correlations), timestamps
        )
        
        # 计算稳定性得分
        stability_scores = await self._calculate_stability_scores(correlations)
        
        return RollingCorrelationResult(
            timestamps=timestamps,
            correlations=correlations,
            average_correlations=average_correlations,
            correlation_trends=correlation_trends,
            regime_changes=regime_changes,
            stability_scores=stability_scores
        )
    
    async def _detect_regime_changes(
        self,
        correlation_series: np.ndarray,
        timestamps: List[datetime]
    ) -> List[datetime]:
        """检测相关性制度变化点"""
        from ruptures import Pelt
        
        try:
            # 使用PELT算法检测变化点
            algo = Pelt(model="rbf").fit(correlation_series.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # 转换为时间戳
            regime_changes = []
            for cp in change_points[:-1]:  # 最后一个是序列长度
                if cp < len(timestamps):
                    regime_changes.append(timestamps[cp])
            
            return regime_changes
        
        except Exception as e:
            logger.warning(f"制度变化检测失败: {e}")
            return []
    
    async def _calculate_stability_scores(
        self,
        correlations: np.ndarray
    ) -> List[float]:
        """计算相关性稳定性得分"""
        stability_scores = []
        window_size = 20  # 20期窗口
        
        for i in range(window_size, len(correlations)):
            # 计算窗口内相关性的变异系数
            window_corrs = correlations[i-window_size:i]
            
            # 计算每对策略相关性的标准差
            n_strategies = correlations.shape[1]
            pair_stds = []
            
            for j in range(n_strategies):
                for k in range(j+1, n_strategies):
                    pair_corrs = window_corrs[:, j, k]
                    pair_std = np.std(pair_corrs)
                    pair_stds.append(pair_std)
            
            # 稳定性得分 = 1 / (平均标准差 + 小常数)
            avg_std = np.mean(pair_stds)
            stability_score = 1 / (avg_std + 0.01)
            stability_scores.append(stability_score)
        
        return stability_scores
    
    def _record_analysis(
        self,
        method: CorrelationMethod,
        metrics: CorrelationMetrics,
        rolling_result: Optional[RollingCorrelationResult]
    ):
        """记录分析历史"""
        record = {
            'timestamp': datetime.now(),
            'method': method.value,
            'average_correlation': metrics.average_correlation,
            'max_correlation': metrics.max_correlation,
            'min_correlation': metrics.min_correlation,
            'diversification_ratio': metrics.diversification_ratio,
            'effective_strategies': metrics.effective_strategies,
            'n_clusters': len(metrics.correlation_clusters),
            'has_rolling_analysis': rolling_result is not None
        }
        
        self.analysis_history.append(record)
        
        # 保持最近50次记录
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
    
    async def get_correlation_heatmap_data(
        self,
        correlation_matrix: np.ndarray,
        strategy_names: List[str]
    ) -> Dict[str, Any]:
        """获取相关性热力图数据"""
        return {
            'matrix': correlation_matrix.tolist(),
            'labels': strategy_names,
            'title': '策略相关性热力图',
            'colorscale': 'RdBu',
            'zmin': -1,
            'zmax': 1
        }
    
    async def get_cluster_analysis_report(
        self,
        metrics: CorrelationMetrics,
        strategy_names: List[str]
    ) -> Dict[str, Any]:
        """获取聚类分析报告"""
        report = {
            'summary': {
                'n_strategies': len(strategy_names),
                'n_clusters': len(metrics.correlation_clusters),
                'average_correlation': metrics.average_correlation,
                'diversification_ratio': metrics.diversification_ratio,
                'effective_strategies': metrics.effective_strategies
            },
            'clusters': {},
            'high_correlation_pairs': [],
            'recommendations': []
        }
        
        # 聚类详情
        for cluster_id, strategies in metrics.correlation_clusters.items():
            report['clusters'][f'Cluster_{cluster_id}'] = {
                'strategies': strategies,
                'size': len(strategies),
                'description': f'包含{len(strategies)}个策略的聚类'
            }
        
        # 高相关性策略对
        n_strategies = len(strategy_names)
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                corr = metrics.correlation_matrix[i, j]
                if abs(corr) > self.correlation_threshold:
                    report['high_correlation_pairs'].append({
                        'strategy1': strategy_names[i],
                        'strategy2': strategy_names[j],
                        'correlation': corr,
                        'type': 'positive' if corr > 0 else 'negative'
                    })
        
        # 建议
        if metrics.average_correlation > 0.6:
            report['recommendations'].append("策略间相关性较高，建议增加策略多样性")
        
        if metrics.diversification_ratio < 1.2:
            report['recommendations'].append("分散化效果有限，建议优化策略组合")
        
        if len(metrics.correlation_clusters) < 3:
            report['recommendations'].append("策略聚类数量较少，建议增加不同类型的策略")
        
        return report
    
    async def monitor_correlation_changes(
        self,
        current_correlations: np.ndarray,
        historical_correlations: List[np.ndarray],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """监控相关性变化"""
        if not historical_correlations:
            return {'status': 'no_history', 'changes': []}
        
        # 计算与历史平均的差异
        historical_avg = np.mean(historical_correlations, axis=0)
        correlation_changes = np.abs(current_correlations - historical_avg)
        
        # 识别显著变化
        significant_changes = []
        n_strategies = current_correlations.shape[0]
        
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                change = correlation_changes[i, j]
                if change > threshold:
                    significant_changes.append({
                        'strategy_pair': (i, j),
                        'current_correlation': current_correlations[i, j],
                        'historical_correlation': historical_avg[i, j],
                        'change_magnitude': change,
                        'change_direction': 'increase' if current_correlations[i, j] > historical_avg[i, j] else 'decrease'
                    })
        
        return {
            'status': 'analyzed',
            'n_significant_changes': len(significant_changes),
            'changes': significant_changes,
            'overall_correlation_change': np.mean(correlation_changes),
            'max_change': np.max(correlation_changes)
        }
    
    async def get_analysis_history(self) -> List[Dict[str, Any]]:
        """获取分析历史"""
        return self.analysis_history.copy()


# 全局相关性分析器实例
correlation_analyzer = StrategyCorrelationAnalyzer()

