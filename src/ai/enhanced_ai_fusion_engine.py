#!/usr/bin/env python3
"""
🧠 888-888-88 增强AI融合引擎
Production-Grade Enhanced AI Fusion Engine
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
from loguru import logger

# 导入错误处理系统
from src.core.error_handling_system import (
    handle_errors, ai_operation, critical_section,
    ErrorCategory, ErrorSeverity, error_handler
)

# 导入AI组件
from .ai_model_manager import AIModelManager, ModelPriority, ai_model_manager
from .ai_performance_monitor import AIPerformanceMonitor, ai_performance_monitor

class SignalType(Enum):
    """信号类型"""
    PRICE_PREDICTION = "price_prediction"
    TREND_ANALYSIS = "trend_analysis"
    VOLATILITY_FORECAST = "volatility_forecast"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_RECOGNITION = "pattern_recognition"
    SUPPORT_RESISTANCE = "support_resistance"

class SignalStrength(Enum):
    """信号强度"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class AISignal:
    """AI信号"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    value: Any
    confidence: float
    strength: SignalStrength
    model_id: str
    processing_time_ms: float
    features_used: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class FusionDecision:
    """融合决策"""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    risk_level: float
    expected_return: float
    time_horizon: str
    contributing_signals: List[str]
    fusion_method: str
    reasoning: str

class EnhancedAIFusionEngine:
    """增强AI融合引擎"""
    
    def __init__(self, max_signals: int = 10000):
        self.model_manager = ai_model_manager
        self.performance_monitor = ai_performance_monitor
        
        # 信号管理
        self.signals_history: deque = deque(maxlen=max_signals)
        self.active_signals: Dict[str, List[AISignal]] = defaultdict(list)
        
        # 决策历史
        self.decisions_history: deque = deque(maxlen=1000)
        
        # 融合配置
        self.fusion_config = {
            "min_signals_required": 3,
            "confidence_threshold": 0.6,
            "signal_timeout_minutes": 30,
            "model_weights": {},
            "signal_type_weights": {
                SignalType.PRICE_PREDICTION: 0.3,
                SignalType.TREND_ANALYSIS: 0.2,
                SignalType.VOLATILITY_FORECAST: 0.15,
                SignalType.SENTIMENT_ANALYSIS: 0.1,
                SignalType.RISK_ASSESSMENT: 0.1,
                SignalType.PATTERN_RECOGNITION: 0.1,
                SignalType.SUPPORT_RESISTANCE: 0.05
            }
        }
        
        # 性能统计
        self.total_decisions = 0
        self.successful_decisions = 0
        self.total_signals_processed = 0
        
        # 后台任务
        self._background_tasks = []
        self._is_running = False
        
        logger.info("🧠 增强AI融合引擎初始化完成")
    
    @critical_section
    async def initialize(self):
        """初始化融合引擎"""
        try:
            # 初始化AI模型管理器
            await self.model_manager.initialize()
            
            # 启动后台任务
            self._is_running = True
            self._background_tasks.append(
                asyncio.create_task(self._signal_cleanup_task())
            )
            self._background_tasks.append(
                asyncio.create_task(self._performance_monitoring_task())
            )
            
            logger.info("✅ AI融合引擎启动完成")
            
        except Exception as e:
            logger.error(f"❌ AI融合引擎初始化失败: {e}")
            raise
    
    @ai_operation
    async def process_market_data(self, symbol: str, market_data: Dict[str, Any]) -> List[AISignal]:
        """处理市场数据并生成AI信号"""
        try:
            signals = []
            
            # 获取活跃的AI模型
            active_models = self.model_manager.get_active_models()
            
            if not active_models:
                logger.warning("⚠️ 没有活跃的AI模型")
                return signals
            
            # 并行处理多个模型
            tasks = []
            for model_id in active_models:
                task = asyncio.create_task(
                    self._generate_model_signal(model_id, symbol, market_data)
                )
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集成功的信号
            for result in results:
                if isinstance(result, AISignal):
                    signals.append(result)
                    self.total_signals_processed += 1
                elif isinstance(result, Exception):
                    logger.warning(f"⚠️ 模型信号生成失败: {result}")
            
            # 存储信号
            for signal in signals:
                self.signals_history.append(signal)
                self.active_signals[symbol].append(signal)
                
                # 记录到性能监控器
                await self.performance_monitor.record_prediction(
                    model_id=signal.model_id,
                    features={"symbol": symbol, **market_data},
                    prediction=signal.value,
                    confidence=signal.confidence,
                    processing_time_ms=signal.processing_time_ms
                )
            
            logger.info(f"📊 处理 {symbol} 市场数据，生成 {len(signals)} 个信号")
            return signals
            
        except Exception as e:
            logger.error(f"❌ 处理市场数据失败 {symbol}: {e}")
            return []
    
    async def _generate_model_signal(self, model_id: str, symbol: str, market_data: Dict[str, Any]) -> Optional[AISignal]:
        """生成单个模型的信号"""
        try:
            start_time = datetime.now()
            
            # 准备特征数据
            features = self._prepare_features(symbol, market_data)
            
            # 执行预测
            result = await self.model_manager.predict(
                model_id=model_id,
                features=features,
                priority=ModelPriority.HIGH
            )
            
            if not result:
                return None
            
            # 确定信号类型
            signal_type = self._determine_signal_type(model_id)
            
            # 确定信号强度
            strength = self._determine_signal_strength(result.confidence)
            
            # 创建信号
            signal = AISignal(
                signal_id=f"{model_id}_{symbol}_{int(start_time.timestamp())}",
                timestamp=start_time,
                symbol=symbol,
                signal_type=signal_type,
                value=result.prediction,
                confidence=result.confidence,
                strength=strength,
                model_id=model_id,
                processing_time_ms=result.processing_time_ms,
                features_used=result.features_used,
                metadata={
                    "model_version": result.model_version,
                    "market_conditions": self._assess_market_conditions(market_data)
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ 生成模型信号失败 {model_id}: {e}")
            return None
    
    def _prepare_features(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备特征数据"""
        try:
            features = {
                "symbol": symbol,
                "timestamp": datetime.now().timestamp(),
            }
            
            # 价格特征
            if "price" in market_data:
                features["current_price"] = market_data["price"]
            
            if "ohlcv" in market_data:
                ohlcv = market_data["ohlcv"]
                features.update({
                    "open": ohlcv.get("open", 0),
                    "high": ohlcv.get("high", 0),
                    "low": ohlcv.get("low", 0),
                    "close": ohlcv.get("close", 0),
                    "volume": ohlcv.get("volume", 0)
                })
            
            # 技术指标
            if "indicators" in market_data:
                indicators = market_data["indicators"]
                features.update({
                    "rsi": indicators.get("rsi", 50),
                    "macd": indicators.get("macd", 0),
                    "bb_upper": indicators.get("bb_upper", 0),
                    "bb_lower": indicators.get("bb_lower", 0),
                    "sma_20": indicators.get("sma_20", 0),
                    "ema_12": indicators.get("ema_12", 0)
                })
            
            # 市场深度
            if "orderbook" in market_data:
                orderbook = market_data["orderbook"]
                features.update({
                    "bid_price": orderbook.get("bid", 0),
                    "ask_price": orderbook.get("ask", 0),
                    "spread": orderbook.get("spread", 0)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ 准备特征数据失败: {e}")
            return {"symbol": symbol}
    
    def _determine_signal_type(self, model_id: str) -> SignalType:
        """确定信号类型"""
        # 根据模型ID确定信号类型
        if "price" in model_id.lower() or "lstm" in model_id.lower():
            return SignalType.PRICE_PREDICTION
        elif "trend" in model_id.lower():
            return SignalType.TREND_ANALYSIS
        elif "volatility" in model_id.lower():
            return SignalType.VOLATILITY_FORECAST
        elif "sentiment" in model_id.lower():
            return SignalType.SENTIMENT_ANALYSIS
        elif "risk" in model_id.lower():
            return SignalType.RISK_ASSESSMENT
        elif "pattern" in model_id.lower():
            return SignalType.PATTERN_RECOGNITION
        elif "support" in model_id.lower() or "resistance" in model_id.lower():
            return SignalType.SUPPORT_RESISTANCE
        else:
            return SignalType.PRICE_PREDICTION
    
    def _determine_signal_strength(self, confidence: float) -> SignalStrength:
        """确定信号强度"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.75:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _assess_market_conditions(self, market_data: Dict[str, Any]) -> str:
        """评估市场条件"""
        try:
            # 简化的市场条件评估
            if "indicators" in market_data:
                rsi = market_data["indicators"].get("rsi", 50)
                if rsi > 70:
                    return "overbought"
                elif rsi < 30:
                    return "oversold"
                else:
                    return "neutral"
            return "unknown"
        except:
            return "unknown"
    
    @ai_operation
    async def generate_fusion_decision(self, symbol: str, current_price: float) -> Optional[FusionDecision]:
        """生成融合决策"""
        try:
            # 获取最近的信号
            recent_signals = self._get_recent_signals(symbol)
            
            if len(recent_signals) < self.fusion_config["min_signals_required"]:
                logger.info(f"ℹ️ 信号数量不足 {symbol}: {len(recent_signals)}")
                return None
            
            # 信号融合
            fusion_result = await self._fuse_signals(recent_signals, current_price)
            
            if fusion_result["confidence"] < self.fusion_config["confidence_threshold"]:
                logger.info(f"ℹ️ 融合置信度不足 {symbol}: {fusion_result['confidence']:.2f}")
                return None
            
            # 生成决策
            decision = FusionDecision(
                decision_id=f"decision_{symbol}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                symbol=symbol,
                action=fusion_result["action"],
                confidence=fusion_result["confidence"],
                price_target=fusion_result.get("price_target"),
                stop_loss=fusion_result.get("stop_loss"),
                take_profit=fusion_result.get("take_profit"),
                position_size=fusion_result.get("position_size", 0.1),
                risk_level=fusion_result.get("risk_level", 0.5),
                expected_return=fusion_result.get("expected_return", 0.0),
                time_horizon=fusion_result.get("time_horizon", "short"),
                contributing_signals=[s.signal_id for s in recent_signals],
                fusion_method="weighted_ensemble",
                reasoning=fusion_result.get("reasoning", "")
            )
            
            # 记录决策
            self.decisions_history.append(decision)
            self.total_decisions += 1
            
            logger.info(f"🎯 生成融合决策 {symbol}: {decision.action} (置信度: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"❌ 生成融合决策失败 {symbol}: {e}")
            return None
    
    def _get_recent_signals(self, symbol: str, minutes: int = 30) -> List[AISignal]:
        """获取最近的信号"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_signals = [
            signal for signal in self.active_signals[symbol]
            if signal.timestamp >= cutoff_time
        ]
        
        # 按时间排序
        recent_signals.sort(key=lambda s: s.timestamp, reverse=True)
        
        return recent_signals
    
    async def _fuse_signals(self, signals: List[AISignal], current_price: float) -> Dict[str, Any]:
        """融合信号"""
        try:
            # 按信号类型分组
            signals_by_type = defaultdict(list)
            for signal in signals:
                signals_by_type[signal.signal_type].append(signal)
            
            # 计算加权分数
            total_score = 0.0
            total_weight = 0.0
            
            for signal_type, type_signals in signals_by_type.items():
                type_weight = self.fusion_config["signal_type_weights"].get(signal_type, 0.1)
                
                for signal in type_signals:
                    # 信号权重 = 类型权重 × 置信度 × 强度权重
                    strength_multiplier = signal.strength.value / 4.0
                    signal_weight = type_weight * signal.confidence * strength_multiplier
                    
                    # 信号分数（标准化到-1到1）
                    signal_score = self._normalize_signal_value(signal)
                    
                    total_score += signal_score * signal_weight
                    total_weight += signal_weight
            
            # 计算最终分数
            final_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # 确定行动
            if final_score > 0.3:
                action = "buy"
            elif final_score < -0.3:
                action = "sell"
            else:
                action = "hold"
            
            # 计算置信度
            confidence = min(0.95, abs(final_score) + 0.1)
            
            # 计算价格目标
            price_target = self._calculate_price_target(signals, current_price, action)
            
            # 计算风险管理参数
            risk_params = self._calculate_risk_parameters(signals, current_price, action)
            
            return {
                "action": action,
                "confidence": confidence,
                "final_score": final_score,
                "price_target": price_target,
                "stop_loss": risk_params.get("stop_loss"),
                "take_profit": risk_params.get("take_profit"),
                "position_size": risk_params.get("position_size", 0.1),
                "risk_level": risk_params.get("risk_level", 0.5),
                "expected_return": risk_params.get("expected_return", 0.0),
                "time_horizon": "short",
                "reasoning": f"融合{len(signals)}个信号，最终分数: {final_score:.3f}"
            }
            
        except Exception as e:
            logger.error(f"❌ 融合信号失败: {e}")
            return {"action": "hold", "confidence": 0.0}
    
    def _normalize_signal_value(self, signal: AISignal) -> float:
        """标准化信号值到-1到1范围"""
        try:
            value = signal.value
            
            if signal.signal_type == SignalType.PRICE_PREDICTION:
                # 价格预测：转换为相对变化
                if isinstance(value, (int, float)):
                    return np.tanh(value * 0.01)  # 假设value是价格变化百分比
            
            elif signal.signal_type == SignalType.TREND_ANALYSIS:
                # 趋势分析：直接使用值
                if isinstance(value, (int, float)):
                    return np.tanh(value)
            
            elif signal.signal_type == SignalType.SENTIMENT_ANALYSIS:
                # 情绪分析：通常已经在-1到1范围
                if isinstance(value, (int, float)):
                    return max(-1, min(1, value))
            
            elif signal.signal_type == SignalType.RISK_ASSESSMENT:
                # 风险评估：转换为负值（高风险 = 负信号）
                if isinstance(value, (int, float)):
                    return -np.tanh(value)
            
            # 默认处理
            if isinstance(value, (int, float)):
                return np.tanh(value)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ 标准化信号值失败: {e}")
            return 0.0
    
    def _calculate_price_target(self, signals: List[AISignal], current_price: float, action: str) -> Optional[float]:
        """计算价格目标"""
        try:
            price_signals = [
                s for s in signals 
                if s.signal_type == SignalType.PRICE_PREDICTION
            ]
            
            if not price_signals:
                return None
            
            # 加权平均价格预测
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for signal in price_signals:
                weight = signal.confidence * signal.strength.value
                if isinstance(signal.value, (int, float)):
                    weighted_sum += signal.value * weight
                    weight_sum += weight
            
            if weight_sum == 0:
                return None
            
            predicted_change = weighted_sum / weight_sum
            price_target = current_price * (1 + predicted_change / 100)
            
            return price_target
            
        except Exception as e:
            logger.error(f"❌ 计算价格目标失败: {e}")
            return None
    
    def _calculate_risk_parameters(self, signals: List[AISignal], current_price: float, action: str) -> Dict[str, Any]:
        """计算风险管理参数"""
        try:
            # 获取风险信号
            risk_signals = [
                s for s in signals 
                if s.signal_type == SignalType.RISK_ASSESSMENT
            ]
            
            # 获取波动率信号
            volatility_signals = [
                s for s in signals 
                if s.signal_type == SignalType.VOLATILITY_FORECAST
            ]
            
            # 计算平均风险水平
            if risk_signals:
                risk_values = [float(s.value) for s in risk_signals if isinstance(s.value, (int, float))]
                avg_risk = sum(risk_values) / len(risk_values) if risk_values else 0.5
            else:
                avg_risk = 0.5
            
            # 计算预期波动率
            if volatility_signals:
                vol_values = [float(s.value) for s in volatility_signals if isinstance(s.value, (int, float))]
                expected_volatility = sum(vol_values) / len(vol_values) if vol_values else 0.02
            else:
                expected_volatility = 0.02
            
            # 基于风险和波动率计算参数
            risk_multiplier = 1.0 + avg_risk
            volatility_multiplier = 1.0 + expected_volatility
            
            # 止损距离（基于波动率）
            stop_loss_distance = expected_volatility * 2 * risk_multiplier
            
            # 止盈距离（风险回报比1:2）
            take_profit_distance = stop_loss_distance * 2
            
            # 仓位大小（基于风险）
            base_position_size = 0.1
            position_size = base_position_size * (1 - avg_risk)
            
            if action == "buy":
                stop_loss = current_price * (1 - stop_loss_distance)
                take_profit = current_price * (1 + take_profit_distance)
                expected_return = take_profit_distance
            elif action == "sell":
                stop_loss = current_price * (1 + stop_loss_distance)
                take_profit = current_price * (1 - take_profit_distance)
                expected_return = take_profit_distance
            else:
                stop_loss = None
                take_profit = None
                expected_return = 0.0
            
            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": max(0.01, min(0.2, position_size)),
                "risk_level": avg_risk,
                "expected_return": expected_return,
                "expected_volatility": expected_volatility
            }
            
        except Exception as e:
            logger.error(f"❌ 计算风险参数失败: {e}")
            return {
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.05,
                "risk_level": 0.5,
                "expected_return": 0.0
            }
    
    async def _signal_cleanup_task(self):
        """信号清理任务"""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # 每5分钟执行一次
                
                cutoff_time = datetime.now() - timedelta(hours=2)
                
                # 清理过期信号
                for symbol in list(self.active_signals.keys()):
                    original_count = len(self.active_signals[symbol])
                    self.active_signals[symbol] = [
                        signal for signal in self.active_signals[symbol]
                        if signal.timestamp >= cutoff_time
                    ]
                    
                    cleaned_count = original_count - len(self.active_signals[symbol])
                    if cleaned_count > 0:
                        logger.info(f"🧹 清理 {symbol} 的 {cleaned_count} 个过期信号")
                
            except Exception as e:
                logger.error(f"❌ 信号清理任务错误: {e}")
    
    async def _performance_monitoring_task(self):
        """性能监控任务"""
        while self._is_running:
            try:
                await asyncio.sleep(600)  # 每10分钟执行一次
                
                # 更新模型权重
                performance_data = await self._calculate_model_performance()
                await self._update_fusion_weights(performance_data)
                
            except Exception as e:
                logger.error(f"❌ 性能监控任务错误: {e}")
    
    async def _calculate_model_performance(self) -> Dict[str, float]:
        """计算模型性能"""
        try:
            performance_data = {}
            
            # 获取所有模型的性能数据
            all_performance = await self.performance_monitor.get_all_models_performance(hours=24)
            
            for model_id, perf in all_performance.items():
                # 综合评分：准确率 × 0.5 + (1 - 错误率) × 0.3 + 置信度 × 0.2
                accuracy_score = perf.get("accuracy", 0.0) * 0.5
                error_rate_score = (1 - perf.get("error_rate", 1.0)) * 0.3
                confidence_score = perf.get("avg_confidence", 0.0) * 0.2
                
                total_score = accuracy_score + error_rate_score + confidence_score
                performance_data[model_id] = total_score
            
            return performance_data
            
        except Exception as e:
            logger.error(f"❌ 计算模型性能失败: {e}")
            return {}
    
    async def _update_fusion_weights(self, performance_data: Dict[str, float]):
        """更新融合权重"""
        try:
            if not performance_data:
                return
            
            # 更新模型权重
            total_performance = sum(performance_data.values())
            
            if total_performance > 0:
                for model_id, performance in performance_data.items():
                    normalized_weight = performance / total_performance
                    self.fusion_config["model_weights"][model_id] = normalized_weight
            
            logger.info("🔄 融合权重已更新")
            
        except Exception as e:
            logger.error(f"❌ 更新融合权重失败: {e}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "total_signals_processed": self.total_signals_processed,
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": self.successful_decisions / max(1, self.total_decisions),
            "active_signals_count": sum(len(signals) for signals in self.active_signals.values()),
            "monitored_symbols": len(self.active_signals),
            "fusion_config": self.fusion_config,
            "is_running": self._is_running
        }
    
    async def shutdown(self):
        """关闭融合引擎"""
        try:
            self._is_running = False
            
            # 取消后台任务
            for task in self._background_tasks:
                task.cancel()
            
            # 关闭AI模型管理器
            await self.model_manager.shutdown()
            
            logger.info("🛑 AI融合引擎已关闭")
            
        except Exception as e:
            logger.error(f"❌ 关闭AI融合引擎失败: {e}")

# 全局AI融合引擎实例
enhanced_ai_fusion_engine = EnhancedAIFusionEngine()

# 导出主要组件
__all__ = [
    'EnhancedAIFusionEngine',
    'AISignal',
    'FusionDecision',
    'SignalType',
    'SignalStrength',
    'enhanced_ai_fusion_engine'
]
