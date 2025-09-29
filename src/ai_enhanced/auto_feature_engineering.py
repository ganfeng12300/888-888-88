"""
🔧 自动特征工程模块 - 生产级实盘交易特征自动生成系统
基于机器学习的智能特征工程，自动发现和生成有效的交易特征
支持技术指标、统计特征、时间序列特征、交叉特征的自动生成和选择
"""
import asyncio
import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available, some feature engineering capabilities will be limited")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using custom technical indicators")

from loguru import logger

class FeatureType(Enum):
    """特征类型"""
    TECHNICAL = "technical"  # 技术指标
    STATISTICAL = "statistical"  # 统计特征
    TIME_SERIES = "time_series"  # 时间序列特征
    CROSS = "cross"  # 交叉特征
    DERIVED = "derived"  # 衍生特征
    TRANSFORMED = "transformed"  # 变换特征

class ScalingMethod(Enum):
    """缩放方法"""
    STANDARD = "standard"  # 标准化
    MINMAX = "minmax"  # 最小最大缩放
    ROBUST = "robust"  # 鲁棒缩放
    NONE = "none"  # 不缩放

@dataclass
class FeatureInfo:
    """特征信息"""
    name: str  # 特征名称
    feature_type: FeatureType  # 特征类型
    importance: float  # 特征重要性
    correlation: float  # 与目标的相关性
    stability: float  # 特征稳定性
    description: str  # 特征描述
    parameters: Dict[str, Any] = field(default_factory=dict)  # 特征参数
    created_at: float = field(default_factory=time.time)  # 创建时间

@dataclass
class FeatureSet:
    """特征集合"""
    features: pd.DataFrame  # 特征数据
    feature_info: Dict[str, FeatureInfo]  # 特征信息
    target: Optional[pd.Series] = None  # 目标变量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

class TechnicalIndicatorGenerator:
    """技术指标生成器"""
    
    def __init__(self):
        self.indicators = {}
        logger.info("技术指标生成器初始化完成")
    
    def generate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成基础技术指标"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # 价格相关指标
            features['price_change'] = data['close'].pct_change()
            features['price_change_abs'] = features['price_change'].abs()
            features['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # 移动平均线
            for period in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
                features[f'price_to_ema_{period}'] = data['close'] / features[f'ema_{period}']
            
            # 布林带
            for period in [20, 50]:
                sma = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma
                features[f'bb_position_{period}'] = (data['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
            
            # RSI
            for period in [14, 21, 30]:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # 随机指标
            for period in [14, 21]:
                low_min = data['low'].rolling(window=period).min()
                high_max = data['high'].rolling(window=period).max()
                features[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
                features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
            
            # 威廉指标
            for period in [14, 21]:
                high_max = data['high'].rolling(window=period).max()
                low_min = data['low'].rolling(window=period).min()
                features[f'williams_r_{period}'] = -100 * (high_max - data['close']) / (high_max - low_min)
            
            # 成交量指标
            features['volume_sma_10'] = data['volume'].rolling(window=10).mean()
            features['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma_20']
            features['price_volume'] = data['close'] * data['volume']
            
            # OBV (On Balance Volume)
            features['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            
            # 真实波动范围 (ATR)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features['atr_14'] = true_range.rolling(window=14).mean()
            features['atr_21'] = true_range.rolling(window=21).mean()
            
            logger.info(f"生成基础技术指标: {len(features.columns)}个")
            return features
        
        except Exception as e:
            logger.error(f"生成基础技术指标失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def generate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成高级技术指标"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # 商品通道指数 (CCI)
            for period in [14, 20]:
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                features[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # 动量指标
            for period in [10, 14, 20]:
                features[f'momentum_{period}'] = data['close'] / data['close'].shift(period)
                features[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100
            
            # 价格震荡指标
            for period in [14, 21]:
                features[f'price_oscillator_{period}'] = (data['close'] - data['close'].rolling(window=period).mean()) / data['close'].rolling(window=period).std()
            
            # 支撑阻力指标
            for period in [20, 50]:
                features[f'support_{period}'] = data['low'].rolling(window=period).min()
                features[f'resistance_{period}'] = data['high'].rolling(window=period).max()
                features[f'support_distance_{period}'] = (data['close'] - features[f'support_{period}']) / data['close']
                features[f'resistance_distance_{period}'] = (features[f'resistance_{period}'] - data['close']) / data['close']
            
            # 波动率指标
            for period in [10, 20, 30]:
                features[f'volatility_{period}'] = data['close'].rolling(window=period).std()
                features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(window=period).mean()
            
            # 趋势强度指标
            for period in [14, 21]:
                price_changes = data['close'].diff()
                up_moves = price_changes.where(price_changes > 0, 0)
                down_moves = -price_changes.where(price_changes < 0, 0)
                
                up_sum = up_moves.rolling(window=period).sum()
                down_sum = down_moves.rolling(window=period).sum()
                
                features[f'trend_strength_{period}'] = (up_sum - down_sum) / (up_sum + down_sum)
            
            # 市场效率比率
            for period in [10, 20]:
                price_change = np.abs(data['close'] - data['close'].shift(period))
                volatility_sum = np.abs(data['close'].diff()).rolling(window=period).sum()
                features[f'efficiency_ratio_{period}'] = price_change / volatility_sum
            
            logger.info(f"生成高级技术指标: {len(features.columns)}个")
            return features
        
        except Exception as e:
            logger.error(f"生成高级技术指标失败: {e}")
            return pd.DataFrame(index=data.index)

class StatisticalFeatureGenerator:
    """统计特征生成器"""
    
    def __init__(self):
        logger.info("统计特征生成器初始化完成")
    
    def generate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成统计特征"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # 基础统计特征
            for period in [5, 10, 20, 50]:
                # 均值和标准差
                features[f'mean_{period}'] = data['close'].rolling(window=period).mean()
                features[f'std_{period}'] = data['close'].rolling(window=period).std()
                features[f'cv_{period}'] = features[f'std_{period}'] / features[f'mean_{period}']  # 变异系数
                
                # 偏度和峰度
                features[f'skew_{period}'] = data['close'].rolling(window=period).skew()
                features[f'kurt_{period}'] = data['close'].rolling(window=period).kurt()
                
                # 分位数
                features[f'q25_{period}'] = data['close'].rolling(window=period).quantile(0.25)
                features[f'q75_{period}'] = data['close'].rolling(window=period).quantile(0.75)
                features[f'iqr_{period}'] = features[f'q75_{period}'] - features[f'q25_{period}']
                
                # 最大最小值
                features[f'max_{period}'] = data['close'].rolling(window=period).max()
                features[f'min_{period}'] = data['close'].rolling(window=period).min()
                features[f'range_{period}'] = features[f'max_{period}'] - features[f'min_{period}']
                
                # 相对位置
                features[f'position_in_range_{period}'] = (data['close'] - features[f'min_{period}']) / features[f'range_{period}']
            
            # 高级统计特征
            for period in [10, 20, 30]:
                # 自相关
                features[f'autocorr_{period}'] = data['close'].rolling(window=period).apply(
                    lambda x: x.autocorr() if len(x) > 1 else 0
                )
                
                # 趋势特征
                features[f'trend_slope_{period}'] = data['close'].rolling(window=period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
                # 线性回归R²
                features[f'r_squared_{period}'] = data['close'].rolling(window=period).apply(
                    lambda x: np.corrcoef(range(len(x)), x)[0, 1]**2 if len(x) > 1 else 0
                )
            
            # 价格分布特征
            for period in [20, 50]:
                # 价格在不同区间的分布
                rolling_data = data['close'].rolling(window=period)
                features[f'above_mean_ratio_{period}'] = rolling_data.apply(
                    lambda x: (x > x.mean()).sum() / len(x)
                )
                features[f'above_median_ratio_{period}'] = rolling_data.apply(
                    lambda x: (x > x.median()).sum() / len(x)
                )
            
            # 成交量统计特征
            for period in [10, 20]:
                features[f'volume_mean_{period}'] = data['volume'].rolling(window=period).mean()
                features[f'volume_std_{period}'] = data['volume'].rolling(window=period).std()
                features[f'volume_skew_{period}'] = data['volume'].rolling(window=period).skew()
                
                # 价格-成交量关系
                features[f'price_volume_corr_{period}'] = data['close'].rolling(window=period).corr(data['volume'])
            
            logger.info(f"生成统计特征: {len(features.columns)}个")
            return features
        
        except Exception as e:
            logger.error(f"生成统计特征失败: {e}")
            return pd.DataFrame(index=data.index)

class TimeSeriesFeatureGenerator:
    """时间序列特征生成器"""
    
    def __init__(self):
        logger.info("时间序列特征生成器初始化完成")
    
    def generate_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成时间序列特征"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # 滞后特征
            for lag in [1, 2, 3, 5, 10, 20]:
                features[f'close_lag_{lag}'] = data['close'].shift(lag)
                features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
                features[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
            
            # 差分特征
            for order in [1, 2]:
                features[f'close_diff_{order}'] = data['close'].diff(order)
                features[f'volume_diff_{order}'] = data['volume'].diff(order)
            
            # 季节性特征 (基于时间索引)
            if hasattr(data.index, 'hour'):
                features['hour'] = data.index.hour
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            
            if hasattr(data.index, 'dayofweek'):
                features['dayofweek'] = data.index.dayofweek
                features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
            
            if hasattr(data.index, 'month'):
                features['month'] = data.index.month
                features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
                features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # 周期性特征
            for period in [24, 168, 720]:  # 小时、周、月
                if len(data) > period:
                    features[f'seasonal_{period}'] = data['close'].shift(period)
                    features[f'seasonal_diff_{period}'] = data['close'] - data['close'].shift(period)
            
            # 趋势分解特征
            for window in [50, 100, 200]:
                if len(data) > window:
                    # 趋势
                    features[f'trend_{window}'] = data['close'].rolling(window=window, center=True).mean()
                    # 去趋势
                    features[f'detrended_{window}'] = data['close'] - features[f'trend_{window}']
            
            # 变化点检测特征
            for window in [20, 50]:
                # 均值变化
                mean_before = data['close'].rolling(window=window).mean()
                mean_after = data['close'].shift(-window).rolling(window=window).mean()
                features[f'mean_change_{window}'] = mean_after - mean_before
                
                # 方差变化
                var_before = data['close'].rolling(window=window).var()
                var_after = data['close'].shift(-window).rolling(window=window).var()
                features[f'var_change_{window}'] = var_after - var_before
            
            # 序列模式特征
            for window in [5, 10, 20]:
                # 连续上涨/下跌天数
                price_change = data['close'].diff()
                up_streak = (price_change > 0).astype(int)
                down_streak = (price_change < 0).astype(int)
                
                features[f'up_streak_{window}'] = up_streak.rolling(window=window).sum()
                features[f'down_streak_{window}'] = down_streak.rolling(window=window).sum()
            
            logger.info(f"生成时间序列特征: {len(features.columns)}个")
            return features
        
        except Exception as e:
            logger.error(f"生成时间序列特征失败: {e}")
            return pd.DataFrame(index=data.index)

class CrossFeatureGenerator:
    """交叉特征生成器"""
    
    def __init__(self):
        logger.info("交叉特征生成器初始化完成")
    
    def generate_cross_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """生成交叉特征"""
        try:
            cross_features = pd.DataFrame(index=features.index)
            
            # 选择重要的基础特征进行交叉
            important_features = [
                'close', 'volume', 'sma_20', 'ema_20', 'rsi_14', 'macd',
                'bb_position_20', 'atr_14', 'volatility_20'
            ]
            
            # 过滤存在的特征
            available_features = [f for f in important_features if f in features.columns]
            
            # 比率特征
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    if features[feat2].abs().min() > 1e-8:  # 避免除零
                        cross_features[f'{feat1}_to_{feat2}_ratio'] = features[feat1] / features[feat2]
            
            # 差值特征
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    cross_features[f'{feat1}_{feat2}_diff'] = features[feat1] - features[feat2]
            
            # 乘积特征
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    cross_features[f'{feat1}_{feat2}_product'] = features[feat1] * features[feat2]
            
            # 条件特征
            if 'rsi_14' in features.columns and 'bb_position_20' in features.columns:
                cross_features['rsi_bb_signal'] = (
                    (features['rsi_14'] < 30) & (features['bb_position_20'] < 0.2)
                ).astype(int)
                cross_features['rsi_bb_overbought'] = (
                    (features['rsi_14'] > 70) & (features['bb_position_20'] > 0.8)
                ).astype(int)
            
            # 动量交叉特征
            if 'macd' in features.columns and 'macd_signal' in features.columns:
                cross_features['macd_cross_up'] = (
                    (features['macd'] > features['macd_signal']) & 
                    (features['macd'].shift(1) <= features['macd_signal'].shift(1))
                ).astype(int)
                cross_features['macd_cross_down'] = (
                    (features['macd'] < features['macd_signal']) & 
                    (features['macd'].shift(1) >= features['macd_signal'].shift(1))
                ).astype(int)
            
            # 价格位置特征
            if 'close' in features.columns:
                for ma_col in [col for col in features.columns if 'sma_' in col or 'ema_' in col]:
                    cross_features[f'price_above_{ma_col}'] = (features['close'] > features[ma_col]).astype(int)
                    cross_features[f'price_distance_{ma_col}'] = (features['close'] - features[ma_col]) / features[ma_col]
            
            logger.info(f"生成交叉特征: {len(cross_features.columns)}个")
            return cross_features
        
        except Exception as e:
            logger.error(f"生成交叉特征失败: {e}")
            return pd.DataFrame(index=features.index)

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
        logger.info("特征选择器初始化完成")
    
    def select_features_by_importance(self, features: pd.DataFrame, target: pd.Series, 
                                    method: str = 'random_forest', k: int = 50) -> List[str]:
        """基于重要性选择特征"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn不可用，返回所有特征")
                return features.columns.tolist()
            
            # 移除无效特征
            valid_features = features.select_dtypes(include=[np.number]).dropna(axis=1)
            
            if len(valid_features.columns) == 0:
                return []
            
            # 对齐数据
            common_index = valid_features.index.intersection(target.index)
            X = valid_features.loc[common_index]
            y = target.loc[common_index]
            
            if len(X) == 0 or len(y) == 0:
                return []
            
            if method == 'random_forest':
                # 随机森林特征重要性
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                selected = feature_importance.head(k)['feature'].tolist()
                self.feature_scores = dict(zip(feature_importance['feature'], feature_importance['importance']))
            
            elif method == 'correlation':
                # 相关性选择
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected = correlations.head(k).index.tolist()
                self.feature_scores = correlations.to_dict()
            
            elif method == 'mutual_info':
                # 互信息选择
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_df = pd.DataFrame({
                    'feature': X.columns,
                    'mi_score': mi_scores
                }).sort_values('mi_score', ascending=False)
                
                selected = mi_df.head(k)['feature'].tolist()
                self.feature_scores = dict(zip(mi_df['feature'], mi_df['mi_score']))
            
            else:
                # 默认使用F统计量
                selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
                selector.fit(X, y)
                
                selected_mask = selector.get_support()
                selected = X.columns[selected_mask].tolist()
                
                scores = selector.scores_
                self.feature_scores = dict(zip(X.columns, scores))
            
            self.selected_features = selected
            logger.info(f"选择特征: {len(selected)}个 (方法: {method})")
            
            return selected
        
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return features.columns.tolist()[:k]  # 返回前k个特征作为备选
    
    def remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """移除高相关性特征"""
        try:
            # 计算相关性矩阵
            corr_matrix = features.corr().abs()
            
            # 找到高相关性特征对
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 找到需要删除的特征
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            # 保留的特征
            remaining_features = [col for col in features.columns if col not in to_drop]
            
            logger.info(f"移除高相关性特征: {len(to_drop)}个，保留: {len(remaining_features)}个")
            
            return remaining_features
        
        except Exception as e:
            logger.error(f"移除相关性特征失败: {e}")
            return features.columns.tolist()
    
    def select_stable_features(self, features: pd.DataFrame, stability_threshold: float = 0.1) -> List[str]:
        """选择稳定的特征"""
        try:
            stable_features = []
            
            for col in features.columns:
                if features[col].dtype in [np.float64, np.int64]:
                    # 计算变异系数作为稳定性指标
                    mean_val = features[col].mean()
                    std_val = features[col].std()
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        if cv > stability_threshold:  # 变异系数大于阈值认为是稳定的
                            stable_features.append(col)
            
            logger.info(f"选择稳定特征: {len(stable_features)}个")
            return stable_features
        
        except Exception as e:
            logger.error(f"选择稳定特征失败: {e}")
            return features.columns.tolist()

class FeatureTransformer:
    """特征变换器"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        logger.info("特征变换器初始化完成")
    
    def scale_features(self, features: pd.DataFrame, method: ScalingMethod = ScalingMethod.STANDARD) -> pd.DataFrame:
        """缩放特征"""
        try:
            if not SKLEARN_AVAILABLE or method == ScalingMethod.NONE:
                return features
            
            scaled_features = features.copy()
            
            # 选择缩放器
            if method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
            else:
                return features
            
            # 只缩放数值特征
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                scaled_data = scaler.fit_transform(features[numeric_columns])
                scaled_features[numeric_columns] = scaled_data
                
                # 保存缩放器
                self.scalers[method.value] = scaler
            
            logger.info(f"特征缩放完成: {method.value}")
            return scaled_features
        
        except Exception as e:
            logger.error(f"特征缩放失败: {e}")
            return features
    
    def apply_pca(self, features: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """应用PCA降维"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn不可用，跳过PCA")
                return features
            
            # 只对数值特征应用PCA
            numeric_features = features.select_dtypes(include=[np.number]).dropna(axis=1)
            
            if len(numeric_features.columns) == 0:
                return features
            
            # 应用PCA
            n_components = min(n_components, len(numeric_features.columns), len(numeric_features))
            pca = PCA(n_components=n_components, random_state=42)
            
            pca_features = pca.fit_transform(numeric_features)
            
            # 创建PCA特征DataFrame
            pca_columns = [f'pca_{i}' for i in range(n_components)]
            pca_df = pd.DataFrame(pca_features, index=features.index, columns=pca_columns)
            
            # 保存变换器
            self.transformers['pca'] = pca
            
            logger.info(f"PCA降维完成: {len(numeric_features.columns)} -> {n_components}")
            return pca_df
        
        except Exception as e:
            logger.error(f"PCA降维失败: {e}")
            return features
    
    def apply_polynomial_features(self, features: pd.DataFrame, degree: int = 2, 
                                 max_features: int = 100) -> pd.DataFrame:
        """生成多项式特征"""
        try:
            # 选择重要特征进行多项式变换
            important_cols = features.columns[:min(10, len(features.columns))]  # 限制特征数量
            selected_features = features[important_cols]
            
            poly_features = pd.DataFrame(index=features.index)
            
            # 生成二次项
            if degree >= 2:
                for i, col1 in enumerate(important_cols):
                    for col2 in important_cols[i:]:
                        if len(poly_features.columns) >= max_features:
                            break
                        
                        if col1 == col2:
                            poly_features[f'{col1}_squared'] = selected_features[col1] ** 2
                        else:
                            poly_features[f'{col1}_{col2}_interaction'] = selected_features[col1] * selected_features[col2]
            
            # 生成三次项（限制数量）
            if degree >= 3 and len(poly_features.columns) < max_features:
                for col in important_cols[:5]:  # 只对前5个特征生成三次项
                    if len(poly_features.columns) >= max_features:
                        break
                    poly_features[f'{col}_cubed'] = selected_features[col] ** 3
            
            logger.info(f"生成多项式特征: {len(poly_features.columns)}个")
            return poly_features
        
        except Exception as e:
            logger.error(f"生成多项式特征失败: {e}")
            return pd.DataFrame(index=features.index)

class AutoFeatureEngineer:
    """自动特征工程主类"""
    
    def __init__(self):
        self.technical_generator = TechnicalIndicatorGenerator()
        self.statistical_generator = StatisticalFeatureGenerator()
        self.time_series_generator = TimeSeriesFeatureGenerator()
        self.cross_generator = CrossFeatureGenerator()
        self.feature_selector = FeatureSelector()
        self.feature_transformer = FeatureTransformer()
        
        # 特征集合历史
        self.feature_sets = {}
        
        logger.info("自动特征工程系统初始化完成")
    
    async def generate_features(self, data: pd.DataFrame, target: pd.Series = None,
                              feature_types: List[FeatureType] = None,
                              max_features: int = 200) -> FeatureSet:
        """生成特征集合"""
        try:
            if feature_types is None:
                feature_types = [FeatureType.TECHNICAL, FeatureType.STATISTICAL, 
                               FeatureType.TIME_SERIES, FeatureType.CROSS]
            
            all_features = pd.DataFrame(index=data.index)
            feature_info = {}
            
            # 生成技术指标特征
            if FeatureType.TECHNICAL in feature_types:
                logger.info("生成技术指标特征...")
                basic_tech = self.technical_generator.generate_basic_indicators(data)
                advanced_tech = self.technical_generator.generate_advanced_indicators(data)
                
                tech_features = pd.concat([basic_tech, advanced_tech], axis=1)
                all_features = pd.concat([all_features, tech_features], axis=1)
                
                # 记录特征信息
                for col in tech_features.columns:
                    feature_info[col] = FeatureInfo(
                        name=col,
                        feature_type=FeatureType.TECHNICAL,
                        importance=0.0,
                        correlation=0.0,
                        stability=0.0,
                        description=f"技术指标: {col}"
                    )
            
            # 生成统计特征
            if FeatureType.STATISTICAL in feature_types:
                logger.info("生成统计特征...")
                stat_features = self.statistical_generator.generate_statistical_features(data)
                all_features = pd.concat([all_features, stat_features], axis=1)
                
                for col in stat_features.columns:
                    feature_info[col] = FeatureInfo(
                        name=col,
                        feature_type=FeatureType.STATISTICAL,
                        importance=0.0,
                        correlation=0.0,
                        stability=0.0,
                        description=f"统计特征: {col}"
                    )
            
            # 生成时间序列特征
            if FeatureType.TIME_SERIES in feature_types:
                logger.info("生成时间序列特征...")
                ts_features = self.time_series_generator.generate_time_series_features(data)
                all_features = pd.concat([all_features, ts_features], axis=1)
                
                for col in ts_features.columns:
                    feature_info[col] = FeatureInfo(
                        name=col,
                        feature_type=FeatureType.TIME_SERIES,
                        importance=0.0,
                        correlation=0.0,
                        stability=0.0,
                        description=f"时间序列特征: {col}"
                    )
            
            # 生成交叉特征
            if FeatureType.CROSS in feature_types and len(all_features.columns) > 0:
                logger.info("生成交叉特征...")
                cross_features = self.cross_generator.generate_cross_features(all_features)
                all_features = pd.concat([all_features, cross_features], axis=1)
                
                for col in cross_features.columns:
                    feature_info[col] = FeatureInfo(
                        name=col,
                        feature_type=FeatureType.CROSS,
                        importance=0.0,
                        correlation=0.0,
                        stability=0.0,
                        description=f"交叉特征: {col}"
                    )
            
            # 清理特征
            all_features = self._clean_features(all_features)
            
            # 特征选择
            if target is not None and len(all_features.columns) > max_features:
                logger.info("执行特征选择...")
                
                # 移除高相关性特征
                selected_cols = self.feature_selector.remove_correlated_features(all_features, threshold=0.95)
                all_features = all_features[selected_cols]
                
                # 基于重要性选择特征
                if len(all_features.columns) > max_features:
                    selected_cols = self.feature_selector.select_features_by_importance(
                        all_features, target, method='random_forest', k=max_features
                    )
                    all_features = all_features[selected_cols]
                
                # 更新特征重要性
                for col in selected_cols:
                    if col in feature_info and col in self.feature_selector.feature_scores:
                        feature_info[col].importance = self.feature_selector.feature_scores[col]
                        if target is not None:
                            feature_info[col].correlation = abs(all_features[col].corr(target))
            
            # 创建特征集合
            feature_set = FeatureSet(
                features=all_features,
                feature_info=feature_info,
                target=target,
                metadata={
                    'generation_time': time.time(),
                    'data_shape': data.shape,
                    'feature_types': [ft.value for ft in feature_types],
                    'total_features': len(all_features.columns)
                }
            )
            
            logger.info(f"特征生成完成: {len(all_features.columns)}个特征")
            return feature_set
        
        except Exception as e:
            logger.error(f"特征生成失败: {e}")
            return FeatureSet(
                features=pd.DataFrame(index=data.index),
                feature_info={},
                target=target
            )
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """清理特征"""
        try:
            # 移除无限值和NaN值过多的特征
            cleaned_features = features.copy()
            
            for col in features.columns:
                # 检查无限值
                if np.isinf(features[col]).any():
                    cleaned_features[col] = features[col].replace([np.inf, -np.inf], np.nan)
                
                # 移除NaN值过多的特征（超过50%）
                nan_ratio = cleaned_features[col].isna().sum() / len(cleaned_features)
                if nan_ratio > 0.5:
                    cleaned_features = cleaned_features.drop(columns=[col])
                    continue
                
                # 填充剩余的NaN值
                if cleaned_features[col].isna().any():
                    if cleaned_features[col].dtype in [np.float64, np.int64]:
                        cleaned_features[col] = cleaned_features[col].fillna(cleaned_features[col].median())
                    else:
                        cleaned_features[col] = cleaned_features[col].fillna(0)
            
            # 移除常数特征
            constant_features = []
            for col in cleaned_features.columns:
                if cleaned_features[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                cleaned_features = cleaned_features.drop(columns=constant_features)
                logger.info(f"移除常数特征: {len(constant_features)}个")
            
            return cleaned_features
        
        except Exception as e:
            logger.error(f"特征清理失败: {e}")
            return features
    
    def get_feature_importance_report(self, feature_set: FeatureSet) -> pd.DataFrame:
        """获取特征重要性报告"""
        try:
            report_data = []
            
            for name, info in feature_set.feature_info.items():
                report_data.append({
                    'feature_name': name,
                    'feature_type': info.feature_type.value,
                    'importance': info.importance,
                    'correlation': info.correlation,
                    'stability': info.stability,
                    'description': info.description
                })
            
            report_df = pd.DataFrame(report_data)
            report_df = report_df.sort_values('importance', ascending=False)
            
            return report_df
        
        except Exception as e:
            logger.error(f"生成特征重要性报告失败: {e}")
            return pd.DataFrame()
    
    def save_feature_set(self, feature_set: FeatureSet, filepath: str):
        """保存特征集合"""
        try:
            import pickle
            
            with open(filepath, 'wb') as f:
                pickle.dump(feature_set, f)
            
            logger.info(f"特征集合保存完成: {filepath}")
        
        except Exception as e:
            logger.error(f"保存特征集合失败: {e}")
    
    def load_feature_set(self, filepath: str) -> Optional[FeatureSet]:
        """加载特征集合"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                feature_set = pickle.load(f)
            
            logger.info(f"特征集合加载完成: {filepath}")
            return feature_set
        
        except Exception as e:
            logger.error(f"加载特征集合失败: {e}")
            return None

# 全局自动特征工程实例
auto_feature_engineer = AutoFeatureEngineer()


def initialize_auto_feature_engineering():
    """初始化自动特征工程系统"""
    from src.ai_enhanced.auto_feature_engineering import AutoFeatureEngineeringSystem
    system = AutoFeatureEngineeringSystem()
    logger.success("✅ 自动特征工程系统初始化完成")
    return system


# 创建全局实例供导入使用
auto_feature_engineering = auto_feature_engineer
