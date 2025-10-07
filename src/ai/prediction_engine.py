#!/usr/bin/env python3
"""
🔮 888-888-88 预测引擎
生产级AI预测和信号生成系统
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import time
from loguru import logger
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionSignal:
    """预测信号"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    confidence_level: ConfidenceLevel
    predicted_price: float
    current_price: float
    expected_return: float
    time_horizon: int  # 预测时间范围（分钟）
    features_used: List[str]
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketRegime:
    """市场状态"""
    regime_type: str  # trending, ranging, volatile
    volatility_level: str  # low, medium, high
    trend_direction: str  # up, down, sideways
    strength: float  # 0-1
    confidence: float  # 0-1


@dataclass
class EnsemblePrediction:
    """集成预测结果"""
    symbol: str
    timestamp: datetime
    predictions: List[float]
    weights: List[float]
    final_prediction: float
    uncertainty: float
    model_agreement: float


class PredictionEngine:
    """生产级预测引擎"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.feature_processors: Dict[str, Any] = {}
        
        # 实时数据缓存
        self.data_cache: Dict[str, deque] = {}
        self.prediction_cache: Dict[str, List[PredictionSignal]] = {}
        
        # 预测状态
        self.predicting = False
        self.prediction_tasks: List[asyncio.Task] = []
        
        # 性能统计
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'last_update': datetime.now()
        }
        
        # 市场状态分析
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        
        logger.info("🔮 预测引擎初始化完成")
    
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """添加预测模型"""
        self.models[name] = model
        self.model_weights[name] = weight
        logger.info(f"🧠 添加模型: {name} (权重: {weight})")
    
    def remove_model(self, name: str) -> None:
        """移除模型"""
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]
            logger.info(f"🗑️ 移除模型: {name}")
    
    def update_model_weight(self, name: str, weight: float) -> None:
        """更新模型权重"""
        if name in self.models:
            self.model_weights[name] = weight
            logger.info(f"⚖️ 更新模型权重: {name} -> {weight}")
    
    def add_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """添加实时数据"""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = deque(maxlen=1000)
        
        # 添加时间戳
        data['timestamp'] = datetime.now()
        self.data_cache[symbol].append(data)
    
    def get_latest_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """获取最新数据"""
        if symbol not in self.data_cache or len(self.data_cache[symbol]) == 0:
            return pd.DataFrame()
        
        # 转换为DataFrame
        data_list = list(self.data_cache[symbol])[-lookback:]
        df = pd.DataFrame(data_list)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    async def start_prediction(self, symbols: List[str], interval: int = 60) -> None:
        """开始预测"""
        if self.predicting:
            logger.warning("⚠️ 预测引擎已在运行")
            return
        
        self.predicting = True
        logger.info(f"🚀 开始预测: {symbols}, 间隔: {interval}秒")
        
        # 为每个交易对启动预测任务
        for symbol in symbols:
            task = asyncio.create_task(
                self._prediction_loop(symbol, interval)
            )
            self.prediction_tasks.append(task)
        
        # 启动性能监控任务
        monitor_task = asyncio.create_task(self._monitor_performance())
        self.prediction_tasks.append(monitor_task)
        
        logger.info("✅ 预测任务已启动")
    
    async def stop_prediction(self) -> None:
        """停止预测"""
        self.predicting = False
        
        # 取消所有任务
        for task in self.prediction_tasks:
            task.cancel()
        
        if self.prediction_tasks:
            await asyncio.gather(*self.prediction_tasks, return_exceptions=True)
        
        self.prediction_tasks.clear()
        logger.info("⏹️ 预测引擎已停止")
    
    async def _prediction_loop(self, symbol: str, interval: int) -> None:
        """预测循环"""
        while self.predicting:
            try:
                # 获取最新数据
                df = self.get_latest_data(symbol)
                
                if len(df) < 60:  # 需要足够的历史数据
                    await asyncio.sleep(interval)
                    continue
                
                # 生成预测
                prediction = await self._generate_prediction(symbol, df)
                
                if prediction:
                    # 缓存预测结果
                    if symbol not in self.prediction_cache:
                        self.prediction_cache[symbol] = []
                    
                    self.prediction_cache[symbol].append(prediction)
                    
                    # 保持缓存大小
                    if len(self.prediction_cache[symbol]) > 100:
                        self.prediction_cache[symbol] = self.prediction_cache[symbol][-100:]
                    
                    # 更新统计
                    self.performance_stats['total_predictions'] += 1
                    
                    logger.info(f"🔮 {symbol} 预测: {prediction.signal_type.value} "
                              f"(置信度: {prediction.confidence:.2f})")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ {symbol} 预测循环异常: {e}")
                await asyncio.sleep(interval)
    
    async def _generate_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[PredictionSignal]:
        """生成预测信号"""
        try:
            # 特征工程
            features_df = self._prepare_features(df)
            
            if features_df.empty:
                return None
            
            # 市场状态分析
            market_regime = self.market_regime_analyzer.analyze(df)
            
            # 集成预测
            ensemble_result = await self._ensemble_predict(symbol, features_df, market_regime)
            
            if not ensemble_result:
                return None
            
            # 生成交易信号
            signal = self._generate_trading_signal(
                symbol, ensemble_result, market_regime, df.iloc[-1]['close']
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ 生成预测失败: {e}")
            return None
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        try:
            features_df = df.copy()
            
            # 基础特征
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            features_df['volatility'] = features_df['returns'].rolling(20).std()
            
            # 技术指标
            features_df = self._add_technical_indicators(features_df)
            
            # 价格特征
            features_df['price_change'] = features_df['close'] - features_df['open']
            features_df['price_range'] = features_df['high'] - features_df['low']
            features_df['upper_shadow'] = features_df['high'] - features_df[['open', 'close']].max(axis=1)
            features_df['lower_shadow'] = features_df[['open', 'close']].min(axis=1) - features_df['low']
            
            # 成交量特征
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # 时间特征
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            
            # 删除NaN
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ 特征准备失败: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # 移动平均
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    async def _ensemble_predict(self, symbol: str, features_df: pd.DataFrame, 
                              market_regime: MarketRegime) -> Optional[EnsemblePrediction]:
        """集成预测"""
        try:
            predictions = []
            weights = []
            model_names = []
            
            # 获取最新特征
            latest_features = features_df.iloc[-60:].values  # 最近60个时间点
            
            for model_name, model in self.models.items():
                try:
                    # 根据市场状态调整模型权重
                    adjusted_weight = self._adjust_weight_for_regime(
                        model_name, self.model_weights[model_name], market_regime
                    )
                    
                    if adjusted_weight <= 0:
                        continue
                    
                    # 模型预测
                    if hasattr(model, 'predict'):
                        pred = model.predict(latest_features)
                        if isinstance(pred, np.ndarray):
                            pred = pred[0] if len(pred) > 0 else 0.0
                        
                        predictions.append(float(pred))
                        weights.append(adjusted_weight)
                        model_names.append(model_name)
                
                except Exception as e:
                    logger.error(f"❌ 模型 {model_name} 预测失败: {e}")
                    continue
            
            if not predictions:
                return None
            
            # 标准化权重
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # 加权平均预测
            final_prediction = sum(p * w for p, w in zip(predictions, weights))
            
            # 计算不确定性
            uncertainty = np.std(predictions) if len(predictions) > 1 else 0.0
            
            # 计算模型一致性
            if len(predictions) > 1:
                mean_pred = np.mean(predictions)
                agreement = 1.0 - (np.std(predictions) / abs(mean_pred)) if mean_pred != 0 else 0.0
                agreement = max(0.0, min(1.0, agreement))
            else:
                agreement = 1.0
            
            return EnsemblePrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                predictions=predictions,
                weights=weights,
                final_prediction=final_prediction,
                uncertainty=uncertainty,
                model_agreement=agreement
            )
            
        except Exception as e:
            logger.error(f"❌ 集成预测失败: {e}")
            return None
    
    def _adjust_weight_for_regime(self, model_name: str, base_weight: float, 
                                market_regime: MarketRegime) -> float:
        """根据市场状态调整模型权重"""
        # 根据市场状态调整权重的策略
        adjustment = 1.0
        
        # 根据波动率调整
        if market_regime.volatility_level == 'high':
            # 高波动时降低某些模型权重
            if 'lstm' in model_name.lower():
                adjustment *= 0.8
        elif market_regime.volatility_level == 'low':
            # 低波动时提高趋势模型权重
            if 'trend' in model_name.lower():
                adjustment *= 1.2
        
        # 根据趋势强度调整
        if market_regime.strength > 0.7:
            # 强趋势时提高趋势模型权重
            if 'trend' in model_name.lower() or 'momentum' in model_name.lower():
                adjustment *= 1.3
        
        return base_weight * adjustment
    
    def _generate_trading_signal(self, symbol: str, ensemble_result: EnsemblePrediction,
                               market_regime: MarketRegime, current_price: float) -> PredictionSignal:
        """生成交易信号"""
        predicted_return = ensemble_result.final_prediction
        confidence = ensemble_result.model_agreement
        
        # 调整置信度
        confidence *= market_regime.confidence
        
        # 确定信号类型
        if predicted_return > 0.02:  # 2%以上涨幅
            if confidence > 0.8:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
        elif predicted_return < -0.02:  # 2%以上跌幅
            if confidence > 0.8:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # 置信度等级
        if confidence >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        # 预测价格
        predicted_price = current_price * (1 + predicted_return)
        
        return PredictionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            confidence_level=confidence_level,
            predicted_price=predicted_price,
            current_price=current_price,
            expected_return=predicted_return,
            time_horizon=60,  # 1小时预测
            features_used=list(ensemble_result.predictions),
            model_version="ensemble_v1.0",
            metadata={
                'market_regime': market_regime.__dict__,
                'model_agreement': ensemble_result.model_agreement,
                'uncertainty': ensemble_result.uncertainty,
                'num_models': len(ensemble_result.predictions)
            }
        )
    
    async def _monitor_performance(self) -> None:
        """监控预测性能"""
        while self.predicting:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                
                # 计算准确率
                total_predictions = self.performance_stats['total_predictions']
                if total_predictions > 0:
                    accuracy = self.performance_stats['correct_predictions'] / total_predictions
                    self.performance_stats['accuracy'] = accuracy
                
                # 计算平均置信度
                all_confidences = []
                for symbol_predictions in self.prediction_cache.values():
                    for pred in symbol_predictions[-10:]:  # 最近10个预测
                        all_confidences.append(pred.confidence)
                
                if all_confidences:
                    self.performance_stats['avg_confidence'] = np.mean(all_confidences)
                
                self.performance_stats['last_update'] = datetime.now()
                
                logger.info(f"📊 预测性能 - 准确率: {self.performance_stats['accuracy']:.2%}, "
                          f"平均置信度: {self.performance_stats['avg_confidence']:.2f}")
                
            except Exception as e:
                logger.error(f"❌ 性能监控异常: {e}")
    
    def get_latest_prediction(self, symbol: str) -> Optional[PredictionSignal]:
        """获取最新预测"""
        if symbol in self.prediction_cache and self.prediction_cache[symbol]:
            return self.prediction_cache[symbol][-1]
        return None
    
    def get_prediction_history(self, symbol: str, limit: int = 50) -> List[PredictionSignal]:
        """获取预测历史"""
        if symbol in self.prediction_cache:
            return self.prediction_cache[symbol][-limit:]
        return []
    
    def validate_prediction(self, symbol: str, prediction_id: str, actual_outcome: bool) -> None:
        """验证预测结果"""
        if actual_outcome:
            self.performance_stats['correct_predictions'] += 1
        
        logger.info(f"✅ 预测验证: {symbol} - {'正确' if actual_outcome else '错误'}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'predicting': self.predicting,
            'active_models': len(self.models),
            'cached_symbols': len(self.prediction_cache),
            'performance': self.performance_stats.copy(),
            'model_weights': self.model_weights.copy()
        }


class MarketRegimeAnalyzer:
    """市场状态分析器"""
    
    def analyze(self, df: pd.DataFrame) -> MarketRegime:
        """分析市场状态"""
        try:
            # 计算趋势
            trend_direction, trend_strength = self._analyze_trend(df)
            
            # 计算波动率
            volatility_level = self._analyze_volatility(df)
            
            # 确定市场类型
            regime_type = self._determine_regime_type(df, trend_strength, volatility_level)
            
            # 计算置信度
            confidence = min(trend_strength, 0.9)
            
            return MarketRegime(
                regime_type=regime_type,
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                strength=trend_strength,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ 市场状态分析失败: {e}")
            return MarketRegime(
                regime_type="ranging",
                volatility_level="medium",
                trend_direction="sideways",
                strength=0.5,
                confidence=0.5
            )
    
    def _analyze_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """分析趋势"""
        if len(df) < 20:
            return "sideways", 0.5
        
        # 使用多个时间周期的移动平均
        sma_short = df['close'].rolling(10).mean()
        sma_long = df['close'].rolling(20).mean()
        
        # 趋势方向
        if sma_short.iloc[-1] > sma_long.iloc[-1]:
            direction = "up"
        elif sma_short.iloc[-1] < sma_long.iloc[-1]:
            direction = "down"
        else:
            direction = "sideways"
        
        # 趋势强度
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        strength = min(abs(price_change) * 10, 1.0)  # 标准化到0-1
        
        return direction, strength
    
    def _analyze_volatility(self, df: pd.DataFrame) -> str:
        """分析波动率"""
        if len(df) < 20:
            return "medium"
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        if volatility > 0.03:  # 3%
            return "high"
        elif volatility < 0.01:  # 1%
            return "low"
        else:
            return "medium"
    
    def _determine_regime_type(self, df: pd.DataFrame, trend_strength: float, 
                             volatility_level: str) -> str:
        """确定市场类型"""
        if trend_strength > 0.6:
            return "trending"
        elif volatility_level == "high":
            return "volatile"
        else:
            return "ranging"


if __name__ == "__main__":
    async def test_prediction_engine():
        # 测试预测引擎
        engine = PredictionEngine()
        
        # 模拟添加模型
        class MockModel:
            def predict(self, data):
                return np.random.randn() * 0.02  # 随机预测-2%到2%的收益率
        
        engine.add_model("mock_lstm", MockModel(), 0.6)
        engine.add_model("mock_transformer", MockModel(), 0.4)
        
        # 模拟添加数据
        for i in range(100):
            data = {
                'open': 50000 + np.random.randn() * 100,
                'high': 50100 + np.random.randn() * 100,
                'low': 49900 + np.random.randn() * 100,
                'close': 50000 + np.random.randn() * 100,
                'volume': 1000 + np.random.randn() * 100
            }
            engine.add_data("BTC/USDT", data)
        
        # 开始预测
        await engine.start_prediction(["BTC/USDT"], interval=5)
        
        # 运行一段时间
        await asyncio.sleep(30)
        
        # 获取预测结果
        latest_prediction = engine.get_latest_prediction("BTC/USDT")
        if latest_prediction:
            print(f"最新预测: {latest_prediction.signal_type.value} "
                  f"(置信度: {latest_prediction.confidence:.2f})")
        
        # 停止预测
        await engine.stop_prediction()
        
        # 生成报告
        report = engine.get_performance_report()
        print(f"性能报告: {report}")
    
    # asyncio.run(test_prediction_engine())

