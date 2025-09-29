#!/usr/bin/env python3
"""
📡 生产级交易信号生成器
基于多AI模型融合决策，生成高质量交易信号
专为实盘交易设计，无模拟数据，完整生产级代码
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import json

from loguru import logger
import pandas as pd
import numpy as np
import talib

from ..exchanges.multi_exchange_manager import TradingSignal, OrderSide, OrderType
from ..ai.ai_decision_fusion_engine import ai_decision_fusion_engine
from ..ai_enhanced.sentiment_analysis import sentiment_monitor
from ..security.risk_control_system import risk_control_system

class SignalStrength(Enum):
    """信号强度"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

class MarketCondition(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
@dataclass
class TechnicalIndicators:
    """技术指标"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    atr: float
    adx: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    cci: float
    momentum: float
    roc: float

@dataclass
class AISignal:
    """AI信号"""
    model_name: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    timestamp: datetime

@dataclass
class FusedSignal:
    """融合信号"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: List[str]
    ai_signals: List[AISignal]
    technical_score: float
    sentiment_score: float
    risk_score: float
    timestamp: datetime

class ProductionSignalGenerator:
    """生产级信号生成器"""
    
    def __init__(self):
        self.active = False
        self.signal_history: List[FusedSignal] = []
        self.market_data_cache: Dict[str, List[MarketData]] = {}
        self.technical_cache: Dict[str, TechnicalIndicators] = {}
        self.last_signals: Dict[str, FusedSignal] = {}
        
        # 配置参数
        self.min_confidence = 0.65  # 最小置信度
        self.max_position_size = 0.02  # 最大仓位比例
        self.stop_loss_pct = 0.02  # 止损比例
        self.take_profit_pct = 0.04  # 止盈比例
        self.signal_cooldown = 300  # 信号冷却时间（秒）
        
        # 支持的交易对
        self.supported_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
            'SOL/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT'
        ]
        
        self._lock = threading.Lock()
        
        logger.info("📡 生产级信号生成器初始化完成")
        
    def start_generation(self):
        """启动信号生成"""
        self.active = True
        logger.info("🚀 信号生成器启动")
        
    def stop_generation(self):
        """停止信号生成"""
        self.active = False
        logger.info("⏹️ 信号生成器停止")
        
    def update_market_data(self, symbol: str, data: List[MarketData]):
        """更新市场数据"""
        with self._lock:
            self.market_data_cache[symbol] = data[-200:]  # 保留最近200根K线
            
            # 计算技术指标
            if len(data) >= 50:  # 确保有足够数据计算指标
                self.technical_cache[symbol] = self._calculate_technical_indicators(data)
                
    def _calculate_technical_indicators(self, data: List[MarketData]) -> TechnicalIndicators:
        """计算技术指标"""
        # 转换为numpy数组
        closes = np.array([d.close for d in data])
        highs = np.array([d.high for d in data])
        lows = np.array([d.low for d in data])
        volumes = np.array([d.volume for d in data])
        
        # RSI
        rsi = talib.RSI(closes, timeperiod=14)[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # 移动平均线
        sma_20 = talib.SMA(closes, timeperiod=20)[-1]
        sma_50 = talib.SMA(closes, timeperiod=50)[-1]
        ema_12 = talib.EMA(closes, timeperiod=12)[-1]
        ema_26 = talib.EMA(closes, timeperiod=26)[-1]
        
        # ATR
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        
        # ADX
        adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
        
        # 随机指标
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # 威廉指标
        williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)[-1]
        
        # CCI
        cci = talib.CCI(highs, lows, closes, timeperiod=14)[-1]
        
        # 动量指标
        momentum = talib.MOM(closes, timeperiod=10)[-1]
        
        # ROC
        roc = talib.ROC(closes, timeperiod=10)[-1]
        
        return TechnicalIndicators(
            rsi=rsi,
            macd=macd[-1],
            macd_signal=macd_signal[-1],
            macd_histogram=macd_hist[-1],
            bb_upper=bb_upper[-1],
            bb_middle=bb_middle[-1],
            bb_lower=bb_lower[-1],
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            atr=atr,
            adx=adx,
            stoch_k=stoch_k[-1],
            stoch_d=stoch_d[-1],
            williams_r=williams_r,
            cci=cci,
            momentum=momentum,
            roc=roc
        )
        
    def _analyze_market_condition(self, symbol: str) -> MarketCondition:
        """分析市场状态"""
        if symbol not in self.technical_cache:
            return MarketCondition.SIDEWAYS
            
        indicators = self.technical_cache[symbol]
        data = self.market_data_cache.get(symbol, [])
        
        if len(data) < 20:
            return MarketCondition.SIDEWAYS
            
        # 趋势判断
        if indicators.sma_20 > indicators.sma_50 and indicators.adx > 25:
            return MarketCondition.TRENDING_UP
        elif indicators.sma_20 < indicators.sma_50 and indicators.adx > 25:
            return MarketCondition.TRENDING_DOWN
        elif indicators.atr > np.mean([d.close for d in data[-20:]]) * 0.03:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.SIDEWAYS
            
    def _get_ai_signals(self, symbol: str) -> List[AISignal]:
        """获取AI信号"""
        ai_signals = []
        current_time = datetime.now(timezone.utc)
        
        try:
            # 获取AI决策融合引擎的信号
            if hasattr(ai_decision_fusion_engine, 'get_trading_decision'):
                decision = ai_decision_fusion_engine.get_trading_decision(symbol)
                if decision:
                    ai_signals.append(AISignal(
                        model_name="AI融合引擎",
                        signal=decision.get('action', 'HOLD'),
                        confidence=decision.get('confidence', 0.5),
                        reasoning=decision.get('reasoning', ''),
                        timestamp=current_time
                    ))
        except Exception as e:
            logger.warning(f"获取AI融合引擎信号失败: {e}")
            
        try:
            # 获取情感分析信号
            if hasattr(sentiment_monitor, 'get_market_sentiment'):
                sentiment = sentiment_monitor.get_market_sentiment(symbol)
                if sentiment:
                    signal = "BUY" if sentiment > 0.6 else "SELL" if sentiment < 0.4 else "HOLD"
                    ai_signals.append(AISignal(
                        model_name="情感分析",
                        signal=signal,
                        confidence=abs(sentiment - 0.5) * 2,
                        reasoning=f"市场情感分数: {sentiment:.2f}",
                        timestamp=current_time
                    ))
        except Exception as e:
            logger.warning(f"获取情感分析信号失败: {e}")
            
        # 如果没有AI信号，创建默认信号
        if not ai_signals:
            ai_signals.append(AISignal(
                model_name="默认",
                signal="HOLD",
                confidence=0.5,
                reasoning="AI模型暂不可用",
                timestamp=current_time
            ))
            
        return ai_signals
        
    def _calculate_technical_score(self, symbol: str) -> float:
        """计算技术分析得分"""
        if symbol not in self.technical_cache:
            return 0.5
            
        indicators = self.technical_cache[symbol]
        data = self.market_data_cache.get(symbol, [])
        
        if not data:
            return 0.5
            
        current_price = data[-1].close
        score = 0.5  # 基础分数
        
        # RSI评分
        if indicators.rsi < 30:
            score += 0.1  # 超卖，看涨
        elif indicators.rsi > 70:
            score -= 0.1  # 超买，看跌
            
        # MACD评分
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            score += 0.1  # MACD金叉，看涨
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            score -= 0.1  # MACD死叉，看跌
            
        # 布林带评分
        if current_price < indicators.bb_lower:
            score += 0.1  # 价格在下轨下方，看涨
        elif current_price > indicators.bb_upper:
            score -= 0.1  # 价格在上轨上方，看跌
            
        # 移动平均线评分
        if indicators.ema_12 > indicators.ema_26:
            score += 0.05  # 短期均线在长期均线上方，看涨
        else:
            score -= 0.05  # 短期均线在长期均线下方，看跌
            
        # 随机指标评分
        if indicators.stoch_k < 20 and indicators.stoch_d < 20:
            score += 0.05  # 超卖，看涨
        elif indicators.stoch_k > 80 and indicators.stoch_d > 80:
            score -= 0.05  # 超买，看跌
            
        return max(0.0, min(1.0, score))
        
    def _calculate_position_size(self, symbol: str, confidence: float, risk_score: float) -> float:
        """计算仓位大小"""
        # 基础仓位
        base_size = self.max_position_size * confidence
        
        # 风险调整
        risk_adjusted_size = base_size * (1 - risk_score)
        
        # 波动率调整
        if symbol in self.technical_cache:
            atr = self.technical_cache[symbol].atr
            data = self.market_data_cache.get(symbol, [])
            if data:
                current_price = data[-1].close
                volatility = atr / current_price
                
                # 高波动率降低仓位
                if volatility > 0.05:
                    risk_adjusted_size *= 0.5
                elif volatility > 0.03:
                    risk_adjusted_size *= 0.7
                    
        return max(0.001, min(self.max_position_size, risk_adjusted_size))
        
    def _calculate_stop_loss_take_profit(self, symbol: str, action: str, entry_price: float) -> Tuple[float, float]:
        """计算止损止盈价格"""
        if symbol in self.technical_cache:
            atr = self.technical_cache[symbol].atr
            
            # 基于ATR的动态止损止盈
            if action == "BUY":
                stop_loss = entry_price - (atr * 2)
                take_profit = entry_price + (atr * 3)
            else:  # SELL
                stop_loss = entry_price + (atr * 2)
                take_profit = entry_price - (atr * 3)
        else:
            # 固定比例止损止盈
            if action == "BUY":
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
                
        return stop_loss, take_profit
        
    def generate_signal(self, symbol: str) -> Optional[FusedSignal]:
        """生成交易信号"""
        if not self.active:
            return None
            
        # 检查冷却时间
        if symbol in self.last_signals:
            last_signal_time = self.last_signals[symbol].timestamp
            if (datetime.now(timezone.utc) - last_signal_time).total_seconds() < self.signal_cooldown:
                return None
                
        # 检查数据完整性
        if symbol not in self.market_data_cache or symbol not in self.technical_cache:
            logger.warning(f"缺少 {symbol} 的市场数据或技术指标")
            return None
            
        data = self.market_data_cache[symbol]
        if len(data) < 50:
            logger.warning(f"{symbol} 数据不足，需要至少50根K线")
            return None
            
        current_time = datetime.now(timezone.utc)
        current_price = data[-1].close
        
        # 获取AI信号
        ai_signals = self._get_ai_signals(symbol)
        
        # 计算各项得分
        technical_score = self._calculate_technical_score(symbol)
        
        # 获取情感分析得分
        try:
            sentiment_score = sentiment_monitor.get_market_sentiment(symbol) if hasattr(sentiment_monitor, 'get_market_sentiment') else 0.5
        except:
            sentiment_score = 0.5
            
        # 获取风险得分
        try:
            risk_assessment = risk_control_system.assess_market_risk(symbol) if hasattr(risk_control_system, 'assess_market_risk') else {'risk_level': 0.3}
            risk_score = risk_assessment.get('risk_level', 0.3)
        except:
            risk_score = 0.3
            
        # 融合决策
        buy_votes = sum(1 for signal in ai_signals if signal.signal == "BUY")
        sell_votes = sum(1 for signal in ai_signals if signal.signal == "SELL")
        hold_votes = sum(1 for signal in ai_signals if signal.signal == "HOLD")
        
        # 加权投票
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in ai_signals:
            weight = signal.confidence
            if signal.signal == "BUY":
                weighted_score += weight
            elif signal.signal == "SELL":
                weighted_score -= weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_score /= total_weight
            
        # 技术分析权重
        final_score = (weighted_score * 0.4 + 
                      (technical_score - 0.5) * 0.3 + 
                      (sentiment_score - 0.5) * 0.2 + 
                      (0.5 - risk_score) * 0.1)
        
        # 决策逻辑
        action = "HOLD"
        confidence = abs(final_score)
        
        if final_score > 0.15 and confidence > self.min_confidence:
            action = "BUY"
        elif final_score < -0.15 and confidence > self.min_confidence:
            action = "SELL"
            
        # 信号强度
        if confidence > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence > 0.7:
            strength = SignalStrength.STRONG
        elif confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
            
        # 如果信号不够强，返回None
        if action == "HOLD" or confidence < self.min_confidence:
            return None
            
        # 计算仓位大小
        position_size = self._calculate_position_size(symbol, confidence, risk_score)
        
        # 计算止损止盈
        stop_loss, take_profit = self._calculate_stop_loss_take_profit(symbol, action, current_price)
        
        # 构建推理说明
        reasoning = []
        reasoning.append(f"技术分析得分: {technical_score:.2f}")
        reasoning.append(f"情感分析得分: {sentiment_score:.2f}")
        reasoning.append(f"风险评估得分: {risk_score:.2f}")
        reasoning.append(f"AI投票: 买入{buy_votes} 卖出{sell_votes} 持有{hold_votes}")
        reasoning.append(f"最终得分: {final_score:.2f}")
        
        # 创建融合信号
        fused_signal = FusedSignal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=reasoning,
            ai_signals=ai_signals,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            timestamp=current_time
        )
        
        # 记录信号
        with self._lock:
            self.signal_history.append(fused_signal)
            self.last_signals[symbol] = fused_signal
            
        logger.info(f"🎯 生成交易信号: {symbol} {action} 置信度:{confidence:.2f} 仓位:{position_size:.3f}")
        
        return fused_signal
        
    def convert_to_trading_signal(self, fused_signal: FusedSignal) -> TradingSignal:
        """转换为交易信号"""
        side = OrderSide.BUY if fused_signal.action == "BUY" else OrderSide.SELL
        
        return TradingSignal(
            symbol=fused_signal.symbol,
            side=side,
            order_type=OrderType.MARKET,  # 使用市价单确保成交
            quantity=fused_signal.position_size,
            timestamp=fused_signal.timestamp
        )
        
    def get_signal_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[FusedSignal]:
        """获取信号历史"""
        with self._lock:
            if symbol:
                signals = [s for s in self.signal_history if s.symbol == symbol]
            else:
                signals = self.signal_history.copy()
                
            return signals[-limit:]
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._lock:
            if not self.signal_history:
                return {}
                
            total_signals = len(self.signal_history)
            buy_signals = sum(1 for s in self.signal_history if s.action == "BUY")
            sell_signals = sum(1 for s in self.signal_history if s.action == "SELL")
            
            avg_confidence = np.mean([s.confidence for s in self.signal_history])
            
            # 按强度统计
            strength_stats = {}
            for strength in SignalStrength:
                count = sum(1 for s in self.signal_history if s.strength == strength)
                strength_stats[strength.name] = count
                
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'avg_confidence': avg_confidence,
                'strength_distribution': strength_stats,
                'active_symbols': len(set(s.symbol for s in self.signal_history)),
                'last_signal_time': self.signal_history[-1].timestamp if self.signal_history else None
            }

# 全局信号生成器实例
production_signal_generator = ProductionSignalGenerator()

def initialize_production_signal_generator():
    """初始化生产级信号生成器"""
    logger.success("✅ 生产级信号生成器初始化完成")
    return production_signal_generator
