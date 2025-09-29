#!/usr/bin/env python3
"""
⛏️ 量化因子挖掘引擎 - 智能因子发现
使用机器学习挖掘和构建量化交易因子
专为生产级实盘交易设计，支持自动因子发现和评估
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timezone, timedelta
import json
from dataclasses import dataclass, field
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import talib
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Factor:
    """量化因子"""
    factor_id: str
    name: str
    description: str
    formula: str
    category: str  # 'technical', 'fundamental', 'sentiment', 'macro'
    data_type: str  # 'price', 'volume', 'ratio', 'momentum'
    lookback_period: int
    calculation_function: Optional[Callable] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class FactorPerformance:
    """因子性能评估"""
    factor_id: str
    ic_score: float          # 信息系数
    ic_ir: float            # IC信息比率
    rank_ic: float          # 排序IC
    max_drawdown: float     # 最大回撤
    sharpe_ratio: float     # 夏普比率
    annual_return: float    # 年化收益
    volatility: float       # 波动率
    hit_rate: float         # 胜率
    decay_rate: float       # 衰减率
    stability_score: float  # 稳定性分数
    factor_loading: float   # 因子载荷
    t_stat: float          # t统计量
    p_value: float         # p值
    evaluation_period: int  # 评估期间
    timestamp: datetime

@dataclass
class FactorMiningResult:
    """因子挖掘结果"""
    discovered_factors: List[Factor]
    factor_performances: List[FactorPerformance]
    factor_correlations: Dict[str, Dict[str, float]]
    top_factors: List[str]  # 按性能排序的因子ID
    factor_combination: Dict[str, float]  # 因子组合权重
    mining_statistics: Dict[str, Any]
    timestamp: datetime

class FactorGenerator:
    """因子生成器"""
    
    def __init__(self):
        self.technical_functions = {
            'sma': self._sma,
            'ema': self._ema,
            'rsi': self._rsi,
            'macd': self._macd,
            'bollinger': self._bollinger,
            'atr': self._atr,
            'stoch': self._stoch,
            'williams_r': self._williams_r,
            'cci': self._cci,
            'momentum': self._momentum,
            'roc': self._roc,
            'trix': self._trix
        }
        
        self.combination_functions = {
            'ratio': lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0),
            'diff': lambda x, y: x - y,
            'sum': lambda x, y: x + y,
            'product': lambda x, y: x * y,
            'max': lambda x, y: np.maximum(x, y),
            'min': lambda x, y: np.minimum(x, y)
        }
    
    def generate_technical_factors(self, data: pd.DataFrame) -> List[Factor]:
        """生成技术指标因子"""
        factors = []
        
        # 基础技术指标
        for name, func in self.technical_functions.items():
            for period in [5, 10, 20, 30, 60]:
                try:
                    factor_id = f"{name}_{period}"
                    factor = Factor(
                        factor_id=factor_id,
                        name=f"{name.upper()}({period})",
                        description=f"{period}期{name}技术指标",
                        formula=f"{name}(close, {period})",
                        category='technical',
                        data_type='momentum' if name in ['rsi', 'momentum', 'roc'] else 'price',
                        lookback_period=period,
                        calculation_function=lambda d, p=period, f=func: f(d, p)
                    )
                    factors.append(factor)
                except Exception as e:
                    logger.debug(f"技术因子生成失败 {factor_id}: {e}")
        
        return factors
    
    def generate_combination_factors(self, base_factors: List[Factor], 
                                   data: pd.DataFrame) -> List[Factor]:
        """生成组合因子"""
        factors = []
        
        # 选择前20个基础因子进行组合
        top_factors = base_factors[:20]
        
        for i, factor1 in enumerate(top_factors):
            for j, factor2 in enumerate(top_factors[i+1:], i+1):
                for op_name, op_func in self.combination_functions.items():
                    try:
                        factor_id = f"{factor1.factor_id}_{op_name}_{factor2.factor_id}"
                        factor = Factor(
                            factor_id=factor_id,
                            name=f"{factor1.name} {op_name.upper()} {factor2.name}",
                            description=f"{factor1.name}与{factor2.name}的{op_name}组合",
                            formula=f"{op_name}({factor1.formula}, {factor2.formula})",
                            category='combination',
                            data_type='ratio' if op_name == 'ratio' else 'composite',
                            lookback_period=max(factor1.lookback_period, factor2.lookback_period),
                            calculation_function=lambda d, f1=factor1, f2=factor2, op=op_func: 
                                op(f1.calculation_function(d), f2.calculation_function(d))
                        )
                        factors.append(factor)
                        
                        # 限制组合因子数量
                        if len(factors) >= 100:
                            return factors
                            
                    except Exception as e:
                        logger.debug(f"组合因子生成失败 {factor_id}: {e}")
        
        return factors
    
    # 技术指标计算函数
    def _sma(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """简单移动平均"""
        return talib.SMA(data['close'].values, timeperiod=period)
    
    def _ema(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """指数移动平均"""
        return talib.EMA(data['close'].values, timeperiod=period)
    
    def _rsi(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """相对强弱指数"""
        return talib.RSI(data['close'].values, timeperiod=period)
    
    def _macd(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """MACD"""
        macd, signal, hist = talib.MACD(data['close'].values)
        return macd
    
    def _bollinger(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """布林带位置"""
        upper, middle, lower = talib.BBANDS(data['close'].values, timeperiod=period)
        return (data['close'].values - lower) / (upper - lower)
    
    def _atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """平均真实波幅"""
        return talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    def _stoch(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """随机指标"""
        k, d = talib.STOCH(data['high'].values, data['low'].values, data['close'].values, 
                          fastk_period=period)
        return k
    
    def _williams_r(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """威廉指标"""
        return talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    def _cci(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """商品通道指数"""
        return talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    def _momentum(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """动量指标"""
        return talib.MOM(data['close'].values, timeperiod=period)
    
    def _roc(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """变化率"""
        return talib.ROC(data['close'].values, timeperiod=period)
    
    def _trix(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """TRIX指标"""
        return talib.TRIX(data['close'].values, timeperiod=period)

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self):
        self.min_periods = 60  # 最少评估期数
        
    def evaluate_factor(self, factor_values: np.ndarray, 
                       returns: np.ndarray, 
                       factor: Factor) -> FactorPerformance:
        """评估单个因子"""
        try:
            # 数据清洗
            valid_mask = ~(np.isnan(factor_values) | np.isnan(returns) | 
                          np.isinf(factor_values) | np.isinf(returns))
            
            if np.sum(valid_mask) < self.min_periods:
                return self._create_empty_performance(factor.factor_id)
            
            clean_factor = factor_values[valid_mask]
            clean_returns = returns[valid_mask]
            
            # 信息系数 (IC)
            ic_score, ic_p_value = pearsonr(clean_factor, clean_returns)
            
            # 排序IC
            rank_ic, _ = spearmanr(clean_factor, clean_returns)
            
            # IC信息比率
            ic_series = self._rolling_ic(clean_factor, clean_returns, window=20)
            ic_ir = np.mean(ic_series) / (np.std(ic_series) + 1e-8)
            
            # 因子收益序列
            factor_returns = self._calculate_factor_returns(clean_factor, clean_returns)
            
            # 性能指标
            annual_return = np.mean(factor_returns) * 252
            volatility = np.std(factor_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / (volatility + 1e-8)
            
            # 最大回撤
            cumulative_returns = np.cumprod(1 + factor_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # 胜率
            hit_rate = np.mean(factor_returns > 0)
            
            # 衰减率
            decay_rate = self._calculate_decay_rate(clean_factor, clean_returns)
            
            # 稳定性分数
            stability_score = self._calculate_stability_score(ic_series)
            
            # t统计量
            t_stat = ic_score * np.sqrt(len(clean_factor) - 2) / np.sqrt(1 - ic_score**2 + 1e-8)
            
            return FactorPerformance(
                factor_id=factor.factor_id,
                ic_score=ic_score,
                ic_ir=ic_ir,
                rank_ic=rank_ic,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                annual_return=annual_return,
                volatility=volatility,
                hit_rate=hit_rate,
                decay_rate=decay_rate,
                stability_score=stability_score,
                factor_loading=abs(ic_score),
                t_stat=t_stat,
                p_value=ic_p_value,
                evaluation_period=len(clean_factor),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"❌ 因子评估失败 {factor.factor_id}: {e}")
            return self._create_empty_performance(factor.factor_id)
    
    def _rolling_ic(self, factor_values: np.ndarray, returns: np.ndarray, 
                   window: int = 20) -> np.ndarray:
        """滚动IC计算"""
        ic_series = []
        for i in range(window, len(factor_values)):
            window_factor = factor_values[i-window:i]
            window_returns = returns[i-window:i]
            ic, _ = pearsonr(window_factor, window_returns)
            ic_series.append(ic if not np.isnan(ic) else 0.0)
        return np.array(ic_series)
    
    def _calculate_factor_returns(self, factor_values: np.ndarray, 
                                returns: np.ndarray) -> np.ndarray:
        """计算因子收益"""
        # 因子分层回测
        n_quantiles = 5
        factor_returns = []
        
        for i in range(20, len(factor_values)):  # 滚动窗口
            window_factor = factor_values[i-20:i]
            window_returns = returns[i-19:i+1]  # 下期收益
            
            # 分位数分组
            quantiles = np.quantile(window_factor, np.linspace(0, 1, n_quantiles + 1))
            
            # 计算多空收益
            top_mask = window_factor >= quantiles[-2]
            bottom_mask = window_factor <= quantiles[1]
            
            if np.sum(top_mask) > 0 and np.sum(bottom_mask) > 0:
                top_return = np.mean(window_returns[1:][top_mask[:-1]])
                bottom_return = np.mean(window_returns[1:][bottom_mask[:-1]])
                factor_return = top_return - bottom_return
                factor_returns.append(factor_return)
            else:
                factor_returns.append(0.0)
        
        return np.array(factor_returns)
    
    def _calculate_decay_rate(self, factor_values: np.ndarray, 
                            returns: np.ndarray) -> float:
        """计算因子衰减率"""
        try:
            # 计算不同滞后期的IC
            lags = [1, 2, 3, 5, 10]
            ics = []
            
            for lag in lags:
                if len(returns) > lag:
                    lagged_returns = returns[lag:]
                    current_factor = factor_values[:-lag]
                    
                    valid_mask = ~(np.isnan(current_factor) | np.isnan(lagged_returns))
                    if np.sum(valid_mask) > 20:
                        ic, _ = pearsonr(current_factor[valid_mask], lagged_returns[valid_mask])
                        ics.append(abs(ic) if not np.isnan(ic) else 0.0)
                    else:
                        ics.append(0.0)
                else:
                    ics.append(0.0)
            
            # 计算衰减率
            if len(ics) > 1 and ics[0] > 0:
                decay_rate = (ics[0] - ics[-1]) / ics[0]
                return max(0, min(1, decay_rate))
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"衰减率计算失败: {e}")
            return 0.5
    
    def _calculate_stability_score(self, ic_series: np.ndarray) -> float:
        """计算稳定性分数"""
        try:
            if len(ic_series) < 10:
                return 0.0
            
            # IC序列的稳定性指标
            ic_mean = np.mean(ic_series)
            ic_std = np.std(ic_series)
            
            # 正IC比例
            positive_ic_ratio = np.mean(ic_series > 0)
            
            # 稳定性分数
            stability = positive_ic_ratio * (1 - ic_std / (abs(ic_mean) + 1e-8))
            return max(0, min(1, stability))
            
        except Exception as e:
            logger.debug(f"稳定性分数计算失败: {e}")
            return 0.0
    
    def _create_empty_performance(self, factor_id: str) -> FactorPerformance:
        """创建空的性能评估"""
        return FactorPerformance(
            factor_id=factor_id,
            ic_score=0.0,
            ic_ir=0.0,
            rank_ic=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            annual_return=0.0,
            volatility=0.0,
            hit_rate=0.5,
            decay_rate=0.5,
            stability_score=0.0,
            factor_loading=0.0,
            t_stat=0.0,
            p_value=1.0,
            evaluation_period=0,
            timestamp=datetime.now(timezone.utc)
        )
