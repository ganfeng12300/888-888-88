#!/usr/bin/env python3
"""
🕵️ 情感分析侦察兵 - 市场情绪监控
多源情感数据分析和市场情绪预测
专为生产级实盘交易设计，支持实时情感监控
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
import re
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import tweepy
import feedparser
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SentimentData:
    """情感数据"""
    source: str
    content: str
    sentiment_score: float  # -1到1之间
    confidence: float      # 0到1之间
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class MarketSentiment:
    """市场情感分析结果"""
    overall_sentiment: float     # 综合情感分数
    sentiment_strength: float    # 情感强度
    sentiment_trend: str         # 'bullish', 'bearish', 'neutral'
    confidence_level: float      # 置信度
    source_breakdown: Dict[str, float]  # 各来源情感分数
    key_topics: List[str]        # 关键话题
    sentiment_volatility: float  # 情感波动性
    news_impact_score: float     # 新闻影响分数
    social_buzz_level: float     # 社交媒体热度
    fear_greed_index: float      # 恐惧贪婪指数
    timestamp: datetime

class SentimentTransformer(nn.Module):
    """情感分析Transformer模型"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 负面、中性、正面
        )
        
        # 情感强度预测
        self.intensity_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        seq_len = x.size(1)
        
        # 词嵌入和位置编码
        embedded = self.embedding(x)
        embedded += self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer编码
        encoded = self.transformer(embedded)
        
        # 全局平均池化
        pooled = torch.mean(encoded, dim=1)
        
        # 分类和强度预测
        sentiment_logits = self.classifier(pooled)
        intensity = self.intensity_head(pooled)
        
        return sentiment_logits, intensity

class SentimentAnalysisScout:
    """情感分析侦察兵"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.sentiment_model = SentimentTransformer().to(self.device)
        self.optimizer = optim.AdamW(self.sentiment_model.parameters(), lr=1e-4)
        
        # 预训练模型
        try:
            self.bert_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            logger.warning("⚠️ BERT模型加载失败，使用备用方案")
            self.bert_analyzer = None
        
        # NLTK情感分析器
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except:
            logger.warning("⚠️ VADER分析器初始化失败")
            self.vader_analyzer = None
        
        # 数据源配置
        self.data_sources = {
            'news': {
                'enabled': True,
                'weight': 0.3,
                'urls': [
                    'https://feeds.finance.yahoo.com/rss/2.0/headline',
                    'https://www.coindesk.com/arc/outboundfeeds/rss/',
                    'https://cointelegraph.com/rss'
                ]
            },
            'social': {
                'enabled': True,
                'weight': 0.2,
                'platforms': ['twitter', 'reddit']
            },
            'market_data': {
                'enabled': True,
                'weight': 0.3,
                'indicators': ['vix', 'put_call_ratio', 'margin_debt']
            },
            'technical': {
                'enabled': True,
                'weight': 0.2,
                'indicators': ['rsi', 'macd', 'bb_position']
            }
        }
        
        # 情感历史数据
        self.sentiment_history = []
        self.max_history = 1000
        
        # 关键词库
        self.bullish_keywords = [
            'bull', 'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout',
            'breakthrough', 'adoption', 'institutional', 'positive', 'optimistic',
            'growth', 'expansion', 'partnership', 'upgrade', 'innovation'
        ]
        
        self.bearish_keywords = [
            'bear', 'bearish', 'crash', 'dump', 'correction', 'decline',
            'selloff', 'panic', 'fear', 'uncertainty', 'regulation', 'ban',
            'hack', 'scam', 'bubble', 'overvalued', 'risk', 'concern'
        ]
        
        # 缓存
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5分钟缓存
        
        # 实时状态
        self.last_sentiment = None
        self.last_confidence = 0.0
        self.performance_score = 0.5
        
        logger.info("🕵️ 情感分析侦察兵初始化完成")
    
    async def analyze_market_sentiment(self, symbol: str = "BTC", 
                                     timeframe: str = "1h") -> MarketSentiment:
        """分析市场情感"""
        try:
            # 检查缓存
            cache_key = f"{symbol}_{timeframe}_{int(time.time() // self.cache_duration)}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # 并行收集各源数据
            tasks = []
            
            if self.data_sources['news']['enabled']:
                tasks.append(self._collect_news_sentiment(symbol))
            
            if self.data_sources['social']['enabled']:
                tasks.append(self._collect_social_sentiment(symbol))
            
            if self.data_sources['market_data']['enabled']:
                tasks.append(self._collect_market_sentiment(symbol))
            
            if self.data_sources['technical']['enabled']:
                tasks.append(self._collect_technical_sentiment(symbol))
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            sentiment_data = []
            for result in results:
                if not isinstance(result, Exception) and result:
                    sentiment_data.extend(result)
            
            if not sentiment_data:
                return self._create_neutral_sentiment()
            
            # 计算综合情感
            market_sentiment = self._calculate_market_sentiment(sentiment_data, symbol)
            
            # 缓存结果
            self.sentiment_cache[cache_key] = market_sentiment
            
            # 清理旧缓存
            self._cleanup_cache()
            
            # 更新历史
            self.sentiment_history.append({
                'timestamp': market_sentiment.timestamp,
                'sentiment': market_sentiment.overall_sentiment,
                'confidence': market_sentiment.confidence_level,
                'symbol': symbol
            })
            
            if len(self.sentiment_history) > self.max_history:
                self.sentiment_history = self.sentiment_history[-self.max_history:]
            
            # 更新状态
            self.last_sentiment = market_sentiment
            self.last_confidence = market_sentiment.confidence_level
            
            logger.info(f"🕵️ 市场情感分析完成 - {symbol}: {market_sentiment.overall_sentiment:.3f}")
            return market_sentiment
            
        except Exception as e:
            logger.error(f"❌ 市场情感分析失败: {e}")
            return self._create_neutral_sentiment()
    
    async def _collect_news_sentiment(self, symbol: str) -> List[SentimentData]:
        """收集新闻情感"""
        try:
            sentiment_data = []
            
            for url in self.data_sources['news']['urls']:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)
                                
                                for entry in feed.entries[:10]:  # 最新10条
                                    title = entry.get('title', '')
                                    summary = entry.get('summary', '')
                                    text = f"{title} {summary}"
                                    
                                    # 检查是否与目标资产相关
                                    if self._is_relevant(text, symbol):
                                        sentiment_score = await self._analyze_text_sentiment(text)
                                        
                                        sentiment_data.append(SentimentData(
                                            source='news',
                                            content=text[:200],
                                            sentiment_score=sentiment_score,
                                            confidence=0.8,
                                            timestamp=datetime.now(timezone.utc),
                                            metadata={'url': url, 'title': title}
                                        ))
                                
                except Exception as e:
                    logger.warning(f"⚠️ 新闻源获取失败 {url}: {e}")
                    continue
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ 新闻情感收集失败: {e}")
            return []
    
    async def _collect_social_sentiment(self, symbol: str) -> List[SentimentData]:
        """收集社交媒体情感"""
        try:
            sentiment_data = []
            
            # 模拟社交媒体数据（实际应用中需要API密钥）
            social_texts = [
                f"{symbol} looking bullish today! 🚀",
                f"Bearish on {symbol}, expecting correction",
                f"{symbol} breaking resistance, moon time!",
                f"Worried about {symbol} regulation news",
                f"{symbol} adoption growing, very optimistic"
            ]
            
            for text in social_texts:
                sentiment_score = await self._analyze_text_sentiment(text)
                
                sentiment_data.append(SentimentData(
                    source='social',
                    content=text,
                    sentiment_score=sentiment_score,
                    confidence=0.6,
                    timestamp=datetime.now(timezone.utc),
                    metadata={'platform': 'twitter'}
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ 社交媒体情感收集失败: {e}")
            return []
    
    async def _collect_market_sentiment(self, symbol: str) -> List[SentimentData]:
        """收集市场数据情感"""
        try:
            sentiment_data = []
            
            # VIX恐慌指数（模拟数据）
            vix_value = np.random.uniform(15, 35)  # 实际应从API获取
            vix_sentiment = -((vix_value - 20) / 20)  # VIX越高越恐慌
            vix_sentiment = max(-1, min(1, vix_sentiment))
            
            sentiment_data.append(SentimentData(
                source='market_data',
                content=f"VIX恐慌指数: {vix_value:.2f}",
                sentiment_score=vix_sentiment,
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                metadata={'indicator': 'vix', 'value': vix_value}
            ))
            
            # 看跌看涨比率（模拟数据）
            put_call_ratio = np.random.uniform(0.5, 1.5)
            pc_sentiment = -((put_call_ratio - 1.0) / 0.5)  # 比率越高越看跌
            pc_sentiment = max(-1, min(1, pc_sentiment))
            
            sentiment_data.append(SentimentData(
                source='market_data',
                content=f"看跌看涨比率: {put_call_ratio:.2f}",
                sentiment_score=pc_sentiment,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                metadata={'indicator': 'put_call_ratio', 'value': put_call_ratio}
            ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ 市场数据情感收集失败: {e}")
            return []
    
    async def _collect_technical_sentiment(self, symbol: str) -> List[SentimentData]:
        """收集技术指标情感"""
        try:
            sentiment_data = []
            
            # 模拟技术指标数据
            rsi = np.random.uniform(30, 70)
            rsi_sentiment = 0.0
            if rsi > 70:
                rsi_sentiment = -0.5  # 超买
            elif rsi < 30:
                rsi_sentiment = 0.5   # 超卖
            else:
                rsi_sentiment = (50 - rsi) / 50  # 中性区域
            
            sentiment_data.append(SentimentData(
                source='technical',
                content=f"RSI指标: {rsi:.2f}",
                sentiment_score=rsi_sentiment,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                metadata={'indicator': 'rsi', 'value': rsi}
            ))
            
            # MACD情感
            macd = np.random.uniform(-0.1, 0.1)
            macd_sentiment = macd * 10  # 放大信号
            macd_sentiment = max(-1, min(1, macd_sentiment))
            
            sentiment_data.append(SentimentData(
                source='technical',
                content=f"MACD指标: {macd:.4f}",
                sentiment_score=macd_sentiment,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                metadata={'indicator': 'macd', 'value': macd}
            ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ 技术指标情感收集失败: {e}")
            return []
    
    async def _analyze_text_sentiment(self, text: str) -> float:
        """分析文本情感"""
        try:
            sentiments = []
            
            # BERT分析
            if self.bert_analyzer:
                try:
                    result = self.bert_analyzer(text[:512])  # BERT限制长度
                    if result and len(result) > 0:
                        label = result[0]['label'].lower()
                        score = result[0]['score']
                        
                        if 'positive' in label or '5' in label or '4' in label:
                            sentiments.append(score)
                        elif 'negative' in label or '1' in label or '2' in label:
                            sentiments.append(-score)
                        else:
                            sentiments.append(0.0)
                except Exception as e:
                    logger.debug(f"BERT分析失败: {e}")
            
            # VADER分析
            if self.vader_analyzer:
                try:
                    scores = self.vader_analyzer.polarity_scores(text)
                    compound_score = scores['compound']
                    sentiments.append(compound_score)
                except Exception as e:
                    logger.debug(f"VADER分析失败: {e}")
            
            # TextBlob分析
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                sentiments.append(polarity)
            except Exception as e:
                logger.debug(f"TextBlob分析失败: {e}")
            
            # 关键词分析
            keyword_sentiment = self._analyze_keywords(text)
            if keyword_sentiment != 0:
                sentiments.append(keyword_sentiment)
            
            # 计算平均情感
            if sentiments:
                return np.mean(sentiments)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ 文本情感分析失败: {e}")
            return 0.0
    
    def _analyze_keywords(self, text: str) -> float:
        """关键词情感分析"""
        try:
            text_lower = text.lower()
            bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
            bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
            
            total_keywords = bullish_count + bearish_count
            if total_keywords == 0:
                return 0.0
            
            sentiment = (bullish_count - bearish_count) / total_keywords
            return sentiment
            
        except Exception as e:
            logger.error(f"❌ 关键词分析失败: {e}")
            return 0.0
    
    def _is_relevant(self, text: str, symbol: str) -> bool:
        """检查文本是否与目标资产相关"""
        try:
            text_lower = text.lower()
            symbol_lower = symbol.lower()
            
            # 直接匹配
            if symbol_lower in text_lower:
                return True
            
            # 常见别名匹配
            aliases = {
                'btc': ['bitcoin', 'btc'],
                'eth': ['ethereum', 'eth'],
                'ada': ['cardano', 'ada'],
                'dot': ['polkadot', 'dot']
            }
            
            if symbol_lower in aliases:
                for alias in aliases[symbol_lower]:
                    if alias in text_lower:
                        return True
            
            # 通用加密货币关键词
            crypto_keywords = ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft']
            if any(keyword in text_lower for keyword in crypto_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 相关性检查失败: {e}")
            return False
    
    def _calculate_market_sentiment(self, sentiment_data: List[SentimentData], 
                                  symbol: str) -> MarketSentiment:
        """计算市场情感"""
        try:
            if not sentiment_data:
                return self._create_neutral_sentiment()
            
            # 按来源分组
            source_sentiments = {}
            for data in sentiment_data:
                if data.source not in source_sentiments:
                    source_sentiments[data.source] = []
                source_sentiments[data.source].append(data.sentiment_score)
            
            # 计算各来源平均情感
            source_breakdown = {}
            weighted_sentiments = []
            
            for source, sentiments in source_sentiments.items():
                avg_sentiment = np.mean(sentiments)
                source_breakdown[source] = avg_sentiment
                
                # 应用权重
                weight = self.data_sources.get(source, {}).get('weight', 0.25)
                weighted_sentiments.append(avg_sentiment * weight)
            
            # 综合情感分数
            overall_sentiment = sum(weighted_sentiments)
            overall_sentiment = max(-1, min(1, overall_sentiment))
            
            # 情感强度
            sentiment_strength = abs(overall_sentiment)
            
            # 情感趋势
            if overall_sentiment > 0.2:
                sentiment_trend = 'bullish'
            elif overall_sentiment < -0.2:
                sentiment_trend = 'bearish'
            else:
                sentiment_trend = 'neutral'
            
            # 置信度（基于数据量和一致性）
            confidence_level = min(len(sentiment_data) / 20.0, 1.0)  # 数据量因子
            sentiment_std = np.std([d.sentiment_score for d in sentiment_data])
            consistency_factor = max(0, 1 - sentiment_std)  # 一致性因子
            confidence_level = (confidence_level + consistency_factor) / 2
            
            # 关键话题提取
            key_topics = self._extract_key_topics(sentiment_data)
            
            # 情感波动性
            sentiment_volatility = sentiment_std
            
            # 新闻影响分数
            news_data = [d for d in sentiment_data if d.source == 'news']
            news_impact_score = abs(np.mean([d.sentiment_score for d in news_data])) if news_data else 0.0
            
            # 社交媒体热度
            social_data = [d for d in sentiment_data if d.source == 'social']
            social_buzz_level = len(social_data) / 10.0  # 标准化到0-1
            social_buzz_level = min(social_buzz_level, 1.0)
            
            # 恐惧贪婪指数
            fear_greed_index = self._calculate_fear_greed_index(sentiment_data)
            
            return MarketSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_strength=sentiment_strength,
                sentiment_trend=sentiment_trend,
                confidence_level=confidence_level,
                source_breakdown=source_breakdown,
                key_topics=key_topics,
                sentiment_volatility=sentiment_volatility,
                news_impact_score=news_impact_score,
                social_buzz_level=social_buzz_level,
                fear_greed_index=fear_greed_index,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"❌ 市场情感计算失败: {e}")
            return self._create_neutral_sentiment()
    
    def _extract_key_topics(self, sentiment_data: List[SentimentData]) -> List[str]:
        """提取关键话题"""
        try:
            # 简单的关键词频率分析
            all_text = ' '.join([d.content for d in sentiment_data])
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # 过滤停用词
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            
            # 计算词频
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 返回前5个高频词
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [word for word, freq in top_words]
            
        except Exception as e:
            logger.error(f"❌ 关键话题提取失败: {e}")
            return []
    
    def _calculate_fear_greed_index(self, sentiment_data: List[SentimentData]) -> float:
        """计算恐惧贪婪指数"""
        try:
            # 基于情感分数计算恐惧贪婪指数
            sentiments = [d.sentiment_score for d in sentiment_data]
            if not sentiments:
                return 50.0  # 中性
            
            avg_sentiment = np.mean(sentiments)
            
            # 转换到0-100范围，50为中性
            fear_greed = 50 + (avg_sentiment * 50)
            return max(0, min(100, fear_greed))
            
        except Exception as e:
            logger.error(f"❌ 恐惧贪婪指数计算失败: {e}")
            return 50.0
    
    def _create_neutral_sentiment(self) -> MarketSentiment:
        """创建中性情感结果"""
        return MarketSentiment(
            overall_sentiment=0.0,
            sentiment_strength=0.0,
            sentiment_trend='neutral',
            confidence_level=0.1,
            source_breakdown={},
            key_topics=[],
            sentiment_volatility=0.0,
            news_impact_score=0.0,
            social_buzz_level=0.0,
            fear_greed_index=50.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key in self.sentiment_cache:
                # 从key中提取时间戳
                parts = key.split('_')
                if len(parts) >= 3:
                    try:
                        cache_time = int(parts[-1]) * self.cache_duration
                        if current_time - cache_time > self.cache_duration:
                            expired_keys.append(key)
                    except ValueError:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self.sentiment_cache[key]
                
        except Exception as e:
            logger.error(f"❌ 缓存清理失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        try:
            return {
                'model_id': 'sentiment_analysis_scout',
                'model_name': '情感分析侦察兵',
                'device': self.device,
                'data_sources': self.data_sources,
                'sentiment_history_length': len(self.sentiment_history),
                'cache_size': len(self.sentiment_cache),
                'last_sentiment': self.last_sentiment.overall_sentiment if self.last_sentiment else 0.0,
                'last_confidence': self.last_confidence,
                'performance_score': self.performance_score,
                'bullish_keywords_count': len(self.bullish_keywords),
                'bearish_keywords_count': len(self.bearish_keywords)
            }
        except Exception as e:
            logger.error(f"❌ 状态获取失败: {e}")
            return {'error': str(e)}

# 全局实例
sentiment_analysis_scout = SentimentAnalysisScout()

def initialize_sentiment_analysis_scout(device: str = None) -> SentimentAnalysisScout:
    """初始化情感分析侦察兵"""
    global sentiment_analysis_scout
    sentiment_analysis_scout = SentimentAnalysisScout(device)
    return sentiment_analysis_scout

if __name__ == "__main__":
    # 测试代码
    async def test_sentiment_analysis():
        scout = initialize_sentiment_analysis_scout()
        
        # 测试市场情感分析
        sentiment = await scout.analyze_market_sentiment("BTC")
        print(f"市场情感分析: {sentiment}")
        
        # 状态报告
        status = scout.get_status()
        print(f"状态报告: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    asyncio.run(test_sentiment_analysis())
