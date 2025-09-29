#!/usr/bin/env python3
"""
🔮 时间序列预测先知 - 价格预测引擎
使用深度学习进行高精度价格预测
专为生产级实盘交易设计，支持多时间框架预测
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PredictionResult:
    """预测结果"""
    timestamp: datetime
    current_price: float
    predicted_prices: Dict[str, float]  # 不同时间框架的预测
    confidence_scores: Dict[str, float]  # 置信度分数
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0-1之间
    support_levels: List[float]
    resistance_levels: List[float]
    volatility_forecast: float
    risk_score: float

@dataclass
class MarketFeatures:
    """市场特征"""
    prices: List[float]
    volumes: List[float]
    timestamps: List[datetime]
    technical_indicators: Dict[str, List[float]]
    market_sentiment: List[float]
    news_scores: List[float]

class TransformerPredictor(nn.Module):
    """基于Transformer的价格预测模型"""
    
    def __init__(self, input_dim: int = 20, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, seq_len: int = 100, pred_len: int = 24):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, pred_len)
        )
        
        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len),
            nn.Sigmoid()
        )
        
        # 波动率预测头
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            predictions: [batch_size, pred_len]
            confidence: [batch_size, pred_len]
            volatility: [batch_size, 1]
        """
        # 输入嵌入和位置编码
        embedded = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        embedded = self.positional_encoding(embedded)
        
        # Transformer编码
        encoded = self.transformer_encoder(embedded)  # [batch_size, seq_len, d_model]
        
        # 全局平均池化
        pooled = torch.mean(encoded, dim=1)  # [batch_size, d_model]
        
        # 预测
        predictions = self.prediction_head(pooled)
        confidence = self.confidence_head(pooled)
        volatility = self.volatility_head(pooled)
        
        return predictions, confidence, volatility

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class TimeSeriesProphet:
    """时间序列预测先知"""
    
    def __init__(self, device: str = None, model_path: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or "models/time_series_prophet.pth"
        
        # 模型参数
        self.seq_len = 100  # 输入序列长度
        self.pred_len = 24  # 预测长度（小时）
        self.input_dim = 20  # 输入特征维度
        
        # 初始化模型
        self.model = TransformerPredictor(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # 数据预处理
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # 历史数据缓存
        self.price_history = []
        self.feature_history = []
        self.max_history = 1000
        
        # 预测缓存
        self.prediction_cache = {}
        self.cache_duration = 300  # 5分钟缓存
        
        # 性能统计
        self.performance_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'mape': 0.0,
            'accuracy': 0.0,
            'sharpe_ratio': 0.0,
            'hit_rate': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
        # 实时状态
        self.last_prediction = None
        self.last_confidence = 0.0
        self.performance_score = 0.5
        
        # 加载预训练模型
        self.load_model()
        
        logger.info(f"🔮 时间序列预测先知初始化完成 - 设备: {self.device}")
    
    async def predict_prices(self, market_data: Dict[str, Any], 
                           timeframes: List[str] = None) -> PredictionResult:
        """预测价格"""
        try:
            if timeframes is None:
                timeframes = ['1h', '4h', '1d']
            
            # 检查缓存
            cache_key = f"prediction_{int(time.time() // self.cache_duration)}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # 准备输入数据
            input_features = self._prepare_features(market_data)
            if input_features is None:
                return self._create_fallback_prediction(market_data)
            
            # GPU预测
            with torch.no_grad():
                self.model.eval()
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
                predictions, confidence, volatility = self.model(input_tensor)
                
                # 转换为numpy
                pred_prices = predictions.cpu().numpy()[0]
                conf_scores = confidence.cpu().numpy()[0]
                vol_forecast = float(volatility.cpu().numpy()[0][0])
            
            # 反标准化预测结果
            current_price = market_data.get('price', 0.0)
            predicted_prices = self._denormalize_predictions(pred_prices, current_price)
            
            # 计算不同时间框架的预测
            timeframe_predictions = {}
            timeframe_confidence = {}
            
            for i, tf in enumerate(timeframes):
                if i < len(predicted_prices):
                    timeframe_predictions[tf] = predicted_prices[i]
                    timeframe_confidence[tf] = conf_scores[i] if i < len(conf_scores) else 0.5
            
            # 趋势分析
            trend_direction, trend_strength = self._analyze_trend(predicted_prices)
            
            # 支撑阻力位
            support_levels, resistance_levels = self._calculate_support_resistance(
                market_data, predicted_prices
            )
            
            # 风险评估
            risk_score = self._calculate_risk_score(vol_forecast, conf_scores, market_data)
            
            # 创建预测结果
            result = PredictionResult(
                timestamp=datetime.now(timezone.utc),
                current_price=current_price,
                predicted_prices=timeframe_predictions,
                confidence_scores=timeframe_confidence,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volatility_forecast=vol_forecast,
                risk_score=risk_score
            )
            
            # 缓存结果
            self.prediction_cache[cache_key] = result
            
            # 清理旧缓存
            self._cleanup_cache()
            
            # 更新状态
            self.last_prediction = result
            self.last_confidence = np.mean(list(timeframe_confidence.values()))
            
            logger.info(f"🔮 价格预测完成 - 趋势: {trend_direction}, 强度: {trend_strength:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 价格预测失败: {e}")
            return self._create_fallback_prediction(market_data)
    
    def _prepare_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """准备输入特征"""
        try:
            # 检查历史数据是否足够
            if len(self.price_history) < self.seq_len:
                logger.warning("⚠️ 历史数据不足，无法进行预测")
                return None
            
            # 获取最近的价格和特征
            recent_prices = self.price_history[-self.seq_len:]
            recent_features = self.feature_history[-self.seq_len:]
            
            # 构建特征矩阵
            features = []
            for i in range(self.seq_len):
                feature_vector = [
                    recent_prices[i],  # 价格
                    market_data.get('volume', 0.0),  # 成交量
                    market_data.get('rsi', 50.0),  # RSI
                    market_data.get('macd', 0.0),  # MACD
                    market_data.get('bb_upper', 0.0),  # 布林带上轨
                    market_data.get('bb_lower', 0.0),  # 布林带下轨
                    market_data.get('atr', 0.0),  # ATR
                    market_data.get('ema_12', 0.0),  # EMA12
                    market_data.get('ema_26', 0.0),  # EMA26
                    market_data.get('sma_50', 0.0),  # SMA50
                    market_data.get('volume_sma', 0.0),  # 成交量均线
                    market_data.get('price_change', 0.0),  # 价格变化
                    market_data.get('volatility', 0.0),  # 波动率
                    market_data.get('sentiment', 0.0),  # 市场情绪
                    market_data.get('news_impact', 0.0),  # 新闻影响
                    market_data.get('time_of_day', 0.5),  # 时间特征
                    market_data.get('day_of_week', 0.5),  # 星期特征
                    market_data.get('support_level', 0.0),  # 支撑位
                    market_data.get('resistance_level', 0.0),  # 阻力位
                    market_data.get('trend_strength', 0.0)  # 趋势强度
                ]
                features.append(feature_vector)
            
            # 标准化特征
            features_array = np.array(features)
            if not self.is_fitted:
                self.feature_scaler.fit(features_array)
                self.is_fitted = True
            
            normalized_features = self.feature_scaler.transform(features_array)
            return normalized_features
            
        except Exception as e:
            logger.error(f"❌ 特征准备失败: {e}")
            return None
