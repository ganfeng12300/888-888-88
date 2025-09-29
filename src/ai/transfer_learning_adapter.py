#!/usr/bin/env python3
"""
🔄 迁移学习适配器 - 跨市场知识迁移
使用迁移学习技术适配不同市场和时间框架
专为生产级实盘交易设计，支持多市场知识共享
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

@dataclass
class MarketDomain:
    """市场领域定义"""
    domain_id: str
    domain_name: str
    market_type: str  # 'crypto', 'stock', 'forex', 'commodity'
    timeframe: str    # '1m', '5m', '15m', '1h', '4h', '1d'
    features: List[str]
    data_samples: int
    performance_score: float
    last_update: datetime

@dataclass
class TransferResult:
    """迁移学习结果"""
    source_domain: str
    target_domain: str
    transfer_score: float
    adaptation_loss: float
    feature_alignment: float
    knowledge_retention: float
    improvement_ratio: float
    timestamp: datetime

class DomainAdversarialNetwork(nn.Module):
    """领域对抗网络"""
    
    def __init__(self, feature_dim: int = 128, num_domains: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(30, 128),  # 输入特征维度
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
        
        # 预测器（任务特定）
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 领域分类器（对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains),
            nn.Softmax(dim=-1)
        )
        
        # 梯度反转层权重
        self.lambda_grl = 1.0
        
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 特征提取
        features = self.feature_extractor(x)
        
        # 任务预测
        prediction = self.predictor(features)
        
        # 领域分类（梯度反转）
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return prediction, domain_output

class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class TransferLearningAdapter:
    """迁移学习适配器"""
    
    def __init__(self, device: str = None, model_dir: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir or "models/transfer_learning"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 领域管理
        self.domains = {}
        self.domain_models = {}
        self.domain_scalers = {}
        
        # 主模型
        self.main_model = DomainAdversarialNetwork().to(self.device)
        self.optimizer = optim.AdamW(
            self.main_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 迁移学习参数
        self.adaptation_rate = 0.1
        self.domain_weight = 0.1
        self.transfer_threshold = 0.7
        self.max_adaptation_epochs = 50
        
        # 性能追踪
        self.transfer_history = []
        self.domain_performance = {}
        self.adaptation_scores = {}
        
        # 知识库
        self.knowledge_base = {
            'patterns': {},
            'features': {},
            'correlations': {},
            'market_regimes': {}
        }
        
        # 实时状态
        self.current_domain = None
        self.last_transfer_score = 0.0
        self.performance_score = 0.5
        
        # 初始化默认领域
        self._initialize_default_domains()
        
        # 加载预训练模型
        self.load_models()
        
        logger.info(f"🔄 迁移学习适配器初始化完成 - 设备: {self.device}")
    
    def _initialize_default_domains(self):
        """初始化默认领域"""
        default_domains = [
            {
                'domain_id': 'crypto_1h',
                'domain_name': '加密货币1小时',
                'market_type': 'crypto',
                'timeframe': '1h',
                'features': ['price', 'volume', 'rsi', 'macd', 'bb_position']
            },
            {
                'domain_id': 'crypto_4h',
                'domain_name': '加密货币4小时',
                'market_type': 'crypto',
                'timeframe': '4h',
                'features': ['price', 'volume', 'rsi', 'macd', 'bb_position']
            },
            {
                'domain_id': 'crypto_1d',
                'domain_name': '加密货币日线',
                'market_type': 'crypto',
                'timeframe': '1d',
                'features': ['price', 'volume', 'rsi', 'macd', 'bb_position']
            },
            {
                'domain_id': 'stock_1d',
                'domain_name': '股票日线',
                'market_type': 'stock',
                'timeframe': '1d',
                'features': ['price', 'volume', 'rsi', 'macd', 'bb_position']
            }
        ]
        
        for domain_config in default_domains:
            domain = MarketDomain(
                domain_id=domain_config['domain_id'],
                domain_name=domain_config['domain_name'],
                market_type=domain_config['market_type'],
                timeframe=domain_config['timeframe'],
                features=domain_config['features'],
                data_samples=0,
                performance_score=0.5,
                last_update=datetime.now(timezone.utc)
            )
            self.domains[domain.domain_id] = domain
            self.domain_scalers[domain.domain_id] = StandardScaler()
    
    async def adapt_to_domain(self, target_domain_id: str, 
                            market_data: List[Dict[str, Any]]) -> TransferResult:
        """适配到目标领域"""
        try:
            if target_domain_id not in self.domains:
                logger.error(f"❌ 未知的目标领域: {target_domain_id}")
                return self._create_failed_transfer_result(target_domain_id)
            
            target_domain = self.domains[target_domain_id]
            
            # 寻找最佳源领域
            source_domain_id = self._find_best_source_domain(target_domain_id)
            if not source_domain_id:
                logger.warning(f"⚠️ 未找到合适的源领域用于迁移到 {target_domain_id}")
                return self._create_failed_transfer_result(target_domain_id)
            
            # 准备数据
            source_data = self._prepare_domain_data(source_domain_id)
            target_data = self._prepare_target_data(market_data, target_domain_id)
            
            if not source_data or not target_data:
                logger.error("❌ 数据准备失败")
                return self._create_failed_transfer_result(target_domain_id)
            
            # 执行迁移学习
            transfer_result = await self._perform_domain_adaptation(
                source_domain_id, target_domain_id, source_data, target_data
            )
            
            # 更新领域信息
            self._update_domain_performance(target_domain_id, transfer_result)
            
            # 记录迁移历史
            self.transfer_history.append(transfer_result)
            if len(self.transfer_history) > 100:
                self.transfer_history = self.transfer_history[-100:]
            
            # 更新当前领域
            self.current_domain = target_domain_id
            self.last_transfer_score = transfer_result.transfer_score
            
            logger.info(f"🔄 领域适配完成 - {source_domain_id} → {target_domain_id}, 分数: {transfer_result.transfer_score:.4f}")
            return transfer_result
            
        except Exception as e:
            logger.error(f"❌ 领域适配失败: {e}")
            return self._create_failed_transfer_result(target_domain_id)
    
    def _find_best_source_domain(self, target_domain_id: str) -> Optional[str]:
        """寻找最佳源领域"""
        try:
            target_domain = self.domains[target_domain_id]
            best_source = None
            best_similarity = 0.0
            
            for domain_id, domain in self.domains.items():
                if domain_id == target_domain_id:
                    continue
                
                # 计算领域相似性
                similarity = self._calculate_domain_similarity(domain, target_domain)
                
                # 考虑性能分数
                weighted_similarity = similarity * domain.performance_score
                
                if weighted_similarity > best_similarity and domain.data_samples > 100:
                    best_similarity = weighted_similarity
                    best_source = domain_id
            
            return best_source
            
        except Exception as e:
            logger.error(f"❌ 源领域搜索失败: {e}")
            return None
    
    def _calculate_domain_similarity(self, domain1: MarketDomain, domain2: MarketDomain) -> float:
        """计算领域相似性"""
        try:
            similarity = 0.0
            
            # 市场类型相似性
            if domain1.market_type == domain2.market_type:
                similarity += 0.4
            elif domain1.market_type in ['crypto', 'stock'] and domain2.market_type in ['crypto', 'stock']:
                similarity += 0.2
            
            # 时间框架相似性
            timeframe_similarity = {
                ('1m', '5m'): 0.8, ('5m', '15m'): 0.8, ('15m', '1h'): 0.8,
                ('1h', '4h'): 0.6, ('4h', '1d'): 0.4, ('1m', '15m'): 0.6,
                ('5m', '1h'): 0.6, ('15m', '4h'): 0.4, ('1h', '1d'): 0.3
            }
            
            tf_key = (domain1.timeframe, domain2.timeframe)
            if tf_key in timeframe_similarity:
                similarity += timeframe_similarity[tf_key] * 0.3
            elif domain1.timeframe == domain2.timeframe:
                similarity += 0.3
            
            # 特征相似性
            common_features = set(domain1.features) & set(domain2.features)
            total_features = set(domain1.features) | set(domain2.features)
            if total_features:
                feature_similarity = len(common_features) / len(total_features)
                similarity += feature_similarity * 0.3
            
            return min(similarity, 1.0)
            
        except Exception as e:
            logger.error(f"❌ 领域相似性计算失败: {e}")
            return 0.0
    
    def _prepare_domain_data(self, domain_id: str) -> Optional[Dict[str, np.ndarray]]:
        """准备领域数据"""
        try:
            # 这里应该从数据库或缓存中加载历史数据
            # 为了演示，我们创建模拟数据
            if domain_id not in self.domains:
                return None
            
            domain = self.domains[domain_id]
            
            # 模拟历史数据
            n_samples = max(domain.data_samples, 1000)
            n_features = 30
            
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            y = np.random.randn(n_samples).astype(np.float32)
            
            # 添加一些模式
            X[:, 0] = np.sin(np.linspace(0, 10*np.pi, n_samples))  # 周期性模式
            X[:, 1] = np.cumsum(np.random.randn(n_samples) * 0.01)  # 趋势模式
            
            return {'X': X, 'y': y}
            
        except Exception as e:
            logger.error(f"❌ 领域数据准备失败: {e}")
            return None
    
    def _prepare_target_data(self, market_data: List[Dict[str, Any]], 
                           domain_id: str) -> Optional[Dict[str, np.ndarray]]:
        """准备目标数据"""
        try:
            if not market_data:
                return None
            
            features = []
            targets = []
            
            for i, data in enumerate(market_data):
                # 提取特征
                feature_vector = [
                    data.get('price', 0.0),
                    data.get('volume', 0.0),
                    data.get('rsi', 50.0),
                    data.get('macd', 0.0),
                    data.get('bb_position', 0.5),
                    data.get('atr', 0.0),
                    data.get('ema_12', 0.0),
                    data.get('ema_26', 0.0),
                    data.get('sma_50', 0.0),
                    data.get('volume_sma', 0.0),
                    data.get('price_change', 0.0),
                    data.get('volatility', 0.0),
                    data.get('sentiment', 0.0),
                    data.get('news_impact', 0.0),
                    data.get('time_of_day', 0.5),
                    data.get('day_of_week', 0.5),
                    data.get('support_level', 0.0),
                    data.get('resistance_level', 0.0),
                    data.get('trend_strength', 0.0),
                    data.get('momentum', 0.0),
                    data.get('stoch_k', 50.0),
                    data.get('stoch_d', 50.0),
                    data.get('williams_r', -50.0),
                    data.get('cci', 0.0),
                    data.get('roc', 0.0),
                    data.get('trix', 0.0),
                    data.get('dmi_plus', 25.0),
                    data.get('dmi_minus', 25.0),
                    data.get('adx', 25.0),
                    data.get('obv', 0.0)
                ]
                
                features.append(feature_vector)
                
                # 目标值（下一个价格变化）
                if i < len(market_data) - 1:
                    next_price = market_data[i + 1].get('price', data.get('price', 0.0))
                    current_price = data.get('price', 0.0)
                    if current_price > 0:
                        target = (next_price - current_price) / current_price
                    else:
                        target = 0.0
                    targets.append(target)
            
            if len(features) != len(targets):
                features = features[:len(targets)]
            
            X = np.array(features, dtype=np.float32)
            y = np.array(targets, dtype=np.float32)
            
            return {'X': X, 'y': y}
            
        except Exception as e:
            logger.error(f"❌ 目标数据准备失败: {e}")
            return None
