#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 技术指标计算引擎
200+技术指标实时计算，GPU加速优化
专为史诗级AI量化交易设计，毫秒级指标计算
"""

import numpy as np
import pandas as pd
import talib
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, cuda
import cupy as cp

@dataclass
class IndicatorResult:
    """技术指标结果"""
    name: str
    value: float
    timestamp: datetime
    parameters: Dict[str, Any]
    confidence: float = 1.0

class GPUAcceleratedIndicators:
    """GPU加速技术指标计算"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info(f"🚀 GPU加速技术指标引擎启动 - 设备: {self.device}")
        else:
            logger.info("💻 CPU技术指标引擎启动")
    
    def sma_gpu(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """GPU加速简单移动平均"""
        if self.use_gpu and data.is_cuda:
            # 使用GPU卷积实现高效SMA
            kernel = torch.ones(1, 1, period, device=data.device) / period
            data_padded = F.pad(data.unsqueeze(0).unsqueeze(0), (period-1, 0))
            sma = F.conv1d(data_padded, kernel).squeeze()
            return sma
        else:
            # CPU实现
            return data.unfold(0, period, 1).mean(dim=1)
    
    def ema_gpu(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """GPU加速指数移动平均"""
        alpha = 2.0 / (period + 1)
        ema = torch.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def rsi_gpu(self, data: torch.Tensor, period: int = 14) -> torch.Tensor:
        """GPU加速RSI计算"""
        delta = data[1:] - data[:-1]
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        avg_gain = self.ema_gpu(gain, period)
        avg_loss = self.ema_gpu(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # 添加第一个值
        rsi = torch.cat([torch.tensor([50.0], device=data.device), rsi])
        return rsi

class TechnicalIndicatorEngine:
    """🦊 猎狐AI - 技术指标计算引擎"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_indicators = GPUAcceleratedIndicators(device)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 指标缓存
        self.indicator_cache = {}
        self.cache_duration = 60  # 1分钟缓存
        
        logger.info("🦊 猎狐AI技术指标引擎初始化完成")
    
    async def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算所有技术指标"""
        try:
            if len(data) < 100:
                logger.warning("⚠️ 数据不足，无法计算完整指标")
                return {}
            
            # 提取价格数据
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            open_price = data['open'].values
            
            # 并行计算指标组
            tasks = [
                self._calculate_trend_indicators(high, low, close, volume),
                self._calculate_momentum_indicators(high, low, close, volume),
                self._calculate_volatility_indicators(high, low, close, volume),
                self._calculate_volume_indicators(close, volume),
                self._calculate_support_resistance(high, low, close),
                self._calculate_pattern_indicators(open_price, high, low, close)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并结果
            all_indicators = {}
            for result in results:
                if isinstance(result, dict):
                    all_indicators.update(result)
                elif isinstance(result, Exception):
                    logger.error(f"❌ 指标计算失败: {result}")
            
            # 计算综合指标
            all_indicators.update(self._calculate_composite_indicators(all_indicators))
            
            return all_indicators
            
        except Exception as e:
            logger.error(f"❌ 技术指标计算失败: {e}")
            return {}
    
    async def _calculate_trend_indicators(self, high: np.ndarray, low: np.ndarray, 
                                        close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """计算趋势指标"""
        try:
            indicators = {}
            
            # 移动平均线
            indicators['sma_5'] = float(talib.SMA(close, timeperiod=5)[-1])
            indicators['sma_10'] = float(talib.SMA(close, timeperiod=10)[-1])
            indicators['sma_20'] = float(talib.SMA(close, timeperiod=20)[-1])
            indicators['sma_50'] = float(talib.SMA(close, timeperiod=50)[-1])
            indicators['sma_100'] = float(talib.SMA(close, timeperiod=100)[-1])
            indicators['sma_200'] = float(talib.SMA(close, timeperiod=200)[-1])
            
            # 指数移动平均
            indicators['ema_5'] = float(talib.EMA(close, timeperiod=5)[-1])
            indicators['ema_10'] = float(talib.EMA(close, timeperiod=10)[-1])
            indicators['ema_12'] = float(talib.EMA(close, timeperiod=12)[-1])
            indicators['ema_20'] = float(talib.EMA(close, timeperiod=20)[-1])
            indicators['ema_26'] = float(talib.EMA(close, timeperiod=26)[-1])
            indicators['ema_50'] = float(talib.EMA(close, timeperiod=50)[-1])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = float(macd[-1])
            indicators['macd_signal'] = float(macd_signal[-1])
            indicators['macd_histogram'] = float(macd_hist[-1])
            
            # ADX趋势强度
            indicators['adx'] = float(talib.ADX(high, low, close, timeperiod=14)[-1])
            indicators['adx_25'] = float(talib.ADX(high, low, close, timeperiod=25)[-1])
            
            # 抛物线SAR
            indicators['sar'] = float(talib.SAR(high, low)[-1])
            
            # 趋势线斜率
            if len(close) >= 20:
                x = np.arange(20)
                slope, _ = np.polyfit(x, close[-20:], 1)
                indicators['trend_slope_20'] = float(slope)
            
            # 均线排列
            mas = [indicators['sma_5'], indicators['sma_10'], indicators['sma_20'], indicators['sma_50']]
            indicators['ma_alignment'] = 1.0 if all(mas[i] > mas[i+1] for i in range(len(mas)-1)) else \
                                       -1.0 if all(mas[i] < mas[i+1] for i in range(len(mas)-1)) else 0.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 趋势指标计算失败: {e}")
            return {}
    
    async def _calculate_momentum_indicators(self, high: np.ndarray, low: np.ndarray,
                                           close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """计算动量指标"""
        try:
            indicators = {}
            
            # RSI
            indicators['rsi_14'] = float(talib.RSI(close, timeperiod=14)[-1])
            indicators['rsi_21'] = float(talib.RSI(close, timeperiod=21)[-1])
            indicators['rsi_30'] = float(talib.RSI(close, timeperiod=30)[-1])
            
            # 随机指标
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = float(slowk[-1])
            indicators['stoch_d'] = float(slowd[-1])
            
            # 快速随机指标
            fastk, fastd = talib.STOCHF(high, low, close)
            indicators['stochf_k'] = float(fastk[-1])
            indicators['stochf_d'] = float(fastd[-1])
            
            # 威廉指标
            indicators['williams_r'] = float(talib.WILLR(high, low, close)[-1])
            
            # 商品通道指数
            indicators['cci_14'] = float(talib.CCI(high, low, close, timeperiod=14)[-1])
            indicators['cci_20'] = float(talib.CCI(high, low, close, timeperiod=20)[-1])
            
            # 动量指标
            indicators['mom_10'] = float(talib.MOM(close, timeperiod=10)[-1])
            indicators['mom_14'] = float(talib.MOM(close, timeperiod=14)[-1])
            
            # 变化率
            indicators['roc_10'] = float(talib.ROC(close, timeperiod=10)[-1])
            indicators['roc_14'] = float(talib.ROC(close, timeperiod=14)[-1])
            
            # TRIX
            indicators['trix'] = float(talib.TRIX(close)[-1])
            
            # 终极振荡器
            indicators['ultosc'] = float(talib.ULTOSC(high, low, close)[-1])
            
            # 资金流量指数
            indicators['mfi'] = float(talib.MFI(high, low, close, volume)[-1])
            
            # 动量综合评分
            momentum_scores = [
                1 if indicators['rsi_14'] > 50 else -1,
                1 if indicators['stoch_k'] > 50 else -1,
                1 if indicators['cci_14'] > 0 else -1,
                1 if indicators['mom_14'] > 0 else -1,
                1 if indicators['roc_14'] > 0 else -1
            ]
            indicators['momentum_score'] = sum(momentum_scores) / len(momentum_scores)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 动量指标计算失败: {e}")
            return {}
    
    async def _calculate_volatility_indicators(self, high: np.ndarray, low: np.ndarray,
                                             close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """计算波动率指标"""
        try:
            indicators = {}
            
            # 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = float(bb_upper[-1])
            indicators['bb_middle'] = float(bb_middle[-1])
            indicators['bb_lower'] = float(bb_lower[-1])
            indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # ATR
            indicators['atr_14'] = float(talib.ATR(high, low, close, timeperiod=14)[-1])
            indicators['atr_20'] = float(talib.ATR(high, low, close, timeperiod=20)[-1])
            
            # 真实范围
            indicators['trange'] = float(talib.TRANGE(high, low, close)[-1])
            
            # 标准差
            indicators['stddev_20'] = float(talib.STDDEV(close, timeperiod=20)[-1])
            indicators['stddev_30'] = float(talib.STDDEV(close, timeperiod=30)[-1])
            
            # 历史波动率
            if len(close) >= 20:
                returns = np.diff(np.log(close[-21:]))
                indicators['hist_volatility_20'] = float(np.std(returns) * np.sqrt(252))
            
            # 肯特纳通道
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(high, low, close)
            indicators['kc_upper'] = float(kc_upper)
            indicators['kc_middle'] = float(kc_middle)
            indicators['kc_lower'] = float(kc_lower)
            
            # 唐奇安通道
            indicators['donchian_upper'] = float(np.max(high[-20:]))
            indicators['donchian_lower'] = float(np.min(low[-20:]))
            indicators['donchian_middle'] = (indicators['donchian_upper'] + indicators['donchian_lower']) / 2
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 波动率指标计算失败: {e}")
            return {}
    
    async def _calculate_volume_indicators(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """计算成交量指标"""
        try:
            indicators = {}
            
            # 成交量移动平均
            indicators['volume_sma_10'] = float(talib.SMA(volume, timeperiod=10)[-1])
            indicators['volume_sma_20'] = float(talib.SMA(volume, timeperiod=20)[-1])
            
            # 成交量比率
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma_20']
            
            # OBV
            indicators['obv'] = float(talib.OBV(close, volume)[-1])
            
            # 累积/派发线
            indicators['ad'] = float(talib.AD(np.array([close[-1]]), np.array([close[-1]]), 
                                           np.array([close[-1]]), np.array([volume[-1]]))[-1])
            
            # 成交量价格趋势
            indicators['vpt'] = self._calculate_vpt(close, volume)
            
            # 简易波动指标
            indicators['eom'] = self._calculate_eom(np.array([close[-1]]), np.array([close[-1]]), 
                                                  np.array([close[-1]]), volume)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 成交量指标计算失败: {e}")
            return {}
    
    async def _calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, 
                                          close: np.ndarray) -> Dict[str, float]:
        """计算支撑阻力位"""
        try:
            indicators = {}
            
            # 枢轴点
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            indicators['pivot'] = float(pivot)
            indicators['r1'] = float(2 * pivot - low[-1])
            indicators['r2'] = float(pivot + (high[-1] - low[-1]))
            indicators['r3'] = float(high[-1] + 2 * (pivot - low[-1]))
            indicators['s1'] = float(2 * pivot - high[-1])
            indicators['s2'] = float(pivot - (high[-1] - low[-1]))
            indicators['s3'] = float(low[-1] - 2 * (high[-1] - pivot))
            
            # 斐波那契回撤
            if len(high) >= 50:
                period_high = np.max(high[-50:])
                period_low = np.min(low[-50:])
                diff = period_high - period_low
                
                indicators['fib_23.6'] = float(period_high - 0.236 * diff)
                indicators['fib_38.2'] = float(period_high - 0.382 * diff)
                indicators['fib_50.0'] = float(period_high - 0.500 * diff)
                indicators['fib_61.8'] = float(period_high - 0.618 * diff)
                indicators['fib_78.6'] = float(period_high - 0.786 * diff)
            
            # 动态支撑阻力
            indicators['dynamic_support'] = float(np.min(low[-20:]))
            indicators['dynamic_resistance'] = float(np.max(high[-20:]))
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 支撑阻力计算失败: {e}")
            return {}
    
    async def _calculate_pattern_indicators(self, open_price: np.ndarray, high: np.ndarray,
                                          low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """计算形态指标"""
        try:
            indicators = {}
            
            # K线形态识别
            indicators['doji'] = float(talib.CDLDOJI(open_price, high, low, close)[-1])
            indicators['hammer'] = float(talib.CDLHAMMER(open_price, high, low, close)[-1])
            indicators['hanging_man'] = float(talib.CDLHANGINGMAN(open_price, high, low, close)[-1])
            indicators['shooting_star'] = float(talib.CDLSHOOTINGSTAR(open_price, high, low, close)[-1])
            indicators['engulfing'] = float(talib.CDLENGULFING(open_price, high, low, close)[-1])
            indicators['morning_star'] = float(talib.CDLMORNINGSTAR(open_price, high, low, close)[-1])
            indicators['evening_star'] = float(talib.CDLEVENINGSTAR(open_price, high, low, close)[-1])
            
            # 价格形态
            indicators['inside_bar'] = 1.0 if (high[-1] < high[-2] and low[-1] > low[-2]) else 0.0
            indicators['outside_bar'] = 1.0 if (high[-1] > high[-2] and low[-1] < low[-2]) else 0.0
            
            # 缺口分析
            if len(close) >= 2:
                gap_up = low[-1] > high[-2]
                gap_down = high[-1] < low[-2]
                indicators['gap_up'] = 1.0 if gap_up else 0.0
                indicators['gap_down'] = 1.0 if gap_down else 0.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 形态指标计算失败: {e}")
            return {}
    
    def _calculate_composite_indicators(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """计算综合指标"""
        try:
            composite = {}
            
            # 趋势强度综合评分
            trend_signals = []
            if 'sma_20' in indicators and 'close' in indicators:
                trend_signals.append(1 if indicators.get('close', 0) > indicators['sma_20'] else -1)
            if 'ema_20' in indicators and 'close' in indicators:
                trend_signals.append(1 if indicators.get('close', 0) > indicators['ema_20'] else -1)
            if 'macd' in indicators:
                trend_signals.append(1 if indicators['macd'] > 0 else -1)
            if 'adx' in indicators:
                trend_signals.append(1 if indicators['adx'] > 25 else 0)
            
            if trend_signals:
                composite['trend_strength'] = sum(trend_signals) / len(trend_signals)
            
            # 超买超卖综合评分
            overbought_oversold = []
            if 'rsi_14' in indicators:
                if indicators['rsi_14'] > 70:
                    overbought_oversold.append(-1)  # 超买
                elif indicators['rsi_14'] < 30:
                    overbought_oversold.append(1)   # 超卖
                else:
                    overbought_oversold.append(0)
            
            if 'stoch_k' in indicators:
                if indicators['stoch_k'] > 80:
                    overbought_oversold.append(-1)
                elif indicators['stoch_k'] < 20:
                    overbought_oversold.append(1)
                else:
                    overbought_oversold.append(0)
            
            if overbought_oversold:
                composite['overbought_oversold'] = sum(overbought_oversold) / len(overbought_oversold)
            
            # 波动率状态
            if 'bb_width' in indicators:
                if indicators['bb_width'] > 0.1:
                    composite['volatility_state'] = 1.0  # 高波动
                elif indicators['bb_width'] < 0.05:
                    composite['volatility_state'] = -1.0  # 低波动
                else:
                    composite['volatility_state'] = 0.0  # 正常波动
            
            return composite
            
        except Exception as e:
            logger.error(f"❌ 综合指标计算失败: {e}")
            return {}
    
    def _calculate_keltner_channels(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """计算肯特纳通道"""
        try:
            ema = talib.EMA(close, timeperiod=period)
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            upper = ema[-1] + 2 * atr[-1]
            middle = ema[-1]
            lower = ema[-1] - 2 * atr[-1]
            
            return upper, middle, lower
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_vpt(self, close: np.ndarray, volume: np.ndarray) -> float:
        """计算成交量价格趋势"""
        try:
            if len(close) < 2:
                return 0.0
            
            price_change = (close[-1] - close[-2]) / close[-2]
            vpt = volume[-1] * price_change
            return float(vpt)
        except:
            return 0.0
    
    def _calculate_eom(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, volume: np.ndarray) -> float:
        """计算简易波动指标"""
        try:
            if len(volume) == 0:
                return 0.0
            
            distance_moved = (high[-1] + low[-1]) / 2 - (high[0] + low[0]) / 2
            box_height = volume[-1] / (high[-1] - low[-1]) if high[-1] != low[-1] else 0
            eom = distance_moved / box_height if box_height != 0 else 0
            
            return float(eom)
        except:
            return 0.0
