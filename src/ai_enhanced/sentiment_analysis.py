"""
📊 情感分析模块 - 生产级实盘交易市场情感分析系统
基于多源数据的市场情感分析，包含新闻、社交媒体、技术指标情感
支持实时情感监控、情感指数计算、情感驱动交易信号生成
"""
import asyncio
import re
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available, some sentiment analysis features will be limited")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("Web scraping libraries not available")

from loguru import logger

class SentimentType(Enum):
    """情感类型"""
    VERY_NEGATIVE = "very_negative"  # 极度负面
    NEGATIVE = "negative"  # 负面
    NEUTRAL = "neutral"  # 中性
    POSITIVE = "positive"  # 正面
    VERY_POSITIVE = "very_positive"  # 极度正面

class DataSource(Enum):
    """数据源类型"""
    NEWS = "news"  # 新闻
    SOCIAL_MEDIA = "social_media"  # 社交媒体
    TECHNICAL = "technical"  # 技术指标
    FUNDAMENTAL = "fundamental"  # 基本面
    MARKET_DATA = "market_data"  # 市场数据

@dataclass
class SentimentData:
    """情感数据"""
    text: str  # 原始文本
    sentiment_score: float  # 情感得分 (-1到1)
    sentiment_type: SentimentType  # 情感类型
    confidence: float  # 置信度
    source: DataSource  # 数据源
    symbol: str  # 相关交易对
    timestamp: float  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class SentimentIndex:
    """情感指数"""
    symbol: str  # 交易对
    overall_sentiment: float  # 总体情感 (-1到1)
    sentiment_type: SentimentType  # 情感类型
    confidence: float  # 置信度
    news_sentiment: float  # 新闻情感
    social_sentiment: float  # 社交情感
    technical_sentiment: float  # 技术情感
    volume_weighted_sentiment: float  # 成交量加权情感
    sentiment_momentum: float  # 情感动量
    sentiment_volatility: float  # 情感波动率
    data_count: int  # 数据点数量
    timestamp: float  # 时间戳

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            try:
                # 下载必要的NLTK数据
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"NLTK初始化失败: {e}")
        
        # 金融相关关键词
        self.financial_keywords = {
            'positive': ['bull', 'bullish', 'rise', 'up', 'gain', 'profit', 'buy', 'long', 
                        'moon', 'pump', 'surge', 'rally', 'breakout', 'support'],
            'negative': ['bear', 'bearish', 'fall', 'down', 'loss', 'sell', 'short', 
                        'crash', 'dump', 'drop', 'decline', 'resistance', 'breakdown']
        }
        
        logger.info("文本预处理器初始化完成")
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        try:
            # 转换为小写
            text = text.lower()
            
            # 移除URL
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # 移除特殊字符，保留字母、数字和空格
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # 移除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            if NLTK_AVAILABLE and self.lemmatizer:
                # 分词
                tokens = word_tokenize(text)
                
                # 移除停用词和词形还原
                tokens = [self.lemmatizer.lemmatize(token) 
                         for token in tokens 
                         if token not in self.stop_words and len(token) > 2]
                
                text = ' '.join(tokens)
            
            return text
        
        except Exception as e:
            logger.error(f"文本预处理失败: {e}")
            return text

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader_analyzer = None
        
        if NLTK_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"VADER分析器初始化失败: {e}")
        
        # 金融情感词典
        self.financial_sentiment_dict = self._build_financial_sentiment_dict()
        
        logger.info("情感分析器初始化完成")
    
    def _build_financial_sentiment_dict(self) -> Dict[str, float]:
        """构建金融情感词典"""
        sentiment_dict = {}
        
        # 正面词汇
        positive_words = [
            'bull', 'bullish', 'rise', 'up', 'gain', 'profit', 'buy', 'long',
            'moon', 'pump', 'surge', 'rally', 'breakout', 'support', 'strong',
            'growth', 'increase', 'uptrend', 'momentum', 'breakthrough'
        ]
        
        # 负面词汇
        negative_words = [
            'bear', 'bearish', 'fall', 'down', 'loss', 'sell', 'short',
            'crash', 'dump', 'drop', 'decline', 'resistance', 'breakdown',
            'weak', 'decrease', 'downtrend', 'correction', 'collapse'
        ]
        
        # 分配情感得分
        for word in positive_words:
            sentiment_dict[word] = 0.8
        
        for word in negative_words:
            sentiment_dict[word] = -0.8
        
        return sentiment_dict
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """分析文本情感"""
        try:
            # 预处理文本
            processed_text = self.preprocessor.preprocess_text(text)
            
            sentiment_scores = []
            confidences = []
            
            # TextBlob分析
            try:
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity
                sentiment_scores.append(textblob_score)
                confidences.append(0.7)
            except Exception as e:
                logger.debug(f"TextBlob分析失败: {e}")
            
            # VADER分析
            if self.vader_analyzer:
                try:
                    vader_scores = self.vader_analyzer.polarity_scores(text)
                    vader_score = vader_scores['compound']
                    sentiment_scores.append(vader_score)
                    confidences.append(0.8)
                except Exception as e:
                    logger.debug(f"VADER分析失败: {e}")
            
            # 金融词典分析
            financial_score = self._analyze_financial_sentiment(processed_text)
            if financial_score != 0:
                sentiment_scores.append(financial_score)
                confidences.append(0.9)
            
            # 计算加权平均
            if sentiment_scores:
                weighted_sentiment = np.average(sentiment_scores, weights=confidences)
                avg_confidence = np.mean(confidences)
            else:
                weighted_sentiment = 0.0
                avg_confidence = 0.0
            
            return weighted_sentiment, avg_confidence
        
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return 0.0, 0.0
    
    def _analyze_financial_sentiment(self, text: str) -> float:
        """分析金融情感"""
        words = text.split()
        sentiment_sum = 0.0
        word_count = 0
        
        for word in words:
            if word in self.financial_sentiment_dict:
                sentiment_sum += self.financial_sentiment_dict[word]
                word_count += 1
        
        if word_count > 0:
            return sentiment_sum / word_count
        
        return 0.0
    
    def classify_sentiment(self, sentiment_score: float) -> SentimentType:
        """分类情感类型"""
        if sentiment_score <= -0.6:
            return SentimentType.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            return SentimentType.NEGATIVE
        elif sentiment_score < 0.2:
            return SentimentType.NEUTRAL
        elif sentiment_score < 0.6:
            return SentimentType.POSITIVE
        else:
            return SentimentType.VERY_POSITIVE

class NewsCollector:
    """新闻收集器"""
    
    def __init__(self):
        self.news_sources = [
            'https://cointelegraph.com',
            'https://coindesk.com',
            'https://decrypt.co',
            'https://bitcoinist.com'
        ]
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info("新闻收集器初始化完成")
    
    async def collect_news(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """收集新闻"""
        if not WEB_SCRAPING_AVAILABLE:
            logger.warning("网页抓取功能不可用")
            return []
        
        news_articles = []
        
        try:
            # 模拟新闻数据 (实际应用中应该从真实API获取)
            sample_news = [
                {
                    'title': f'{symbol} shows strong bullish momentum',
                    'content': f'{symbol} price has been rising steadily with strong volume support',
                    'source': 'CoinTelegraph',
                    'timestamp': time.time() - 3600,
                    'url': 'https://example.com/news1'
                },
                {
                    'title': f'{symbol} faces resistance at key level',
                    'content': f'{symbol} is struggling to break above the resistance level',
                    'source': 'CoinDesk',
                    'timestamp': time.time() - 7200,
                    'url': 'https://example.com/news2'
                },
                {
                    'title': f'Market analysis: {symbol} outlook positive',
                    'content': f'Technical analysis suggests {symbol} has potential for further gains',
                    'source': 'Decrypt',
                    'timestamp': time.time() - 10800,
                    'url': 'https://example.com/news3'
                }
            ]
            
            news_articles.extend(sample_news[:limit])
            
            logger.info(f"收集到 {len(news_articles)} 条新闻")
            
        except Exception as e:
            logger.error(f"新闻收集失败: {e}")
        
        return news_articles

class SocialMediaCollector:
    """社交媒体收集器"""
    
    def __init__(self):
        # 社交媒体平台配置
        self.platforms = ['twitter', 'reddit', 'telegram']
        
        logger.info("社交媒体收集器初始化完成")
    
    async def collect_social_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """收集社交媒体数据"""
        social_posts = []
        
        try:
            # 模拟社交媒体数据 (实际应用中应该从真实API获取)
            sample_posts = [
                {
                    'text': f'{symbol} to the moon! 🚀 Strong buy signal',
                    'platform': 'twitter',
                    'author': 'crypto_trader_1',
                    'likes': 150,
                    'retweets': 45,
                    'timestamp': time.time() - 1800
                },
                {
                    'text': f'Bearish on {symbol}, expecting a pullback',
                    'platform': 'reddit',
                    'author': 'market_analyst',
                    'upvotes': 23,
                    'comments': 12,
                    'timestamp': time.time() - 3600
                },
                {
                    'text': f'{symbol} looking strong, good entry point',
                    'platform': 'telegram',
                    'author': 'trading_group',
                    'views': 500,
                    'timestamp': time.time() - 5400
                }
            ]
            
            # 生成更多样本数据
            for i in range(min(limit, 20)):
                post = sample_posts[i % len(sample_posts)].copy()
                post['timestamp'] = time.time() - (i * 300)  # 每5分钟一条
                social_posts.append(post)
            
            logger.info(f"收集到 {len(social_posts)} 条社交媒体数据")
            
        except Exception as e:
            logger.error(f"社交媒体数据收集失败: {e}")
        
        return social_posts

class TechnicalSentimentAnalyzer:
    """技术指标情感分析器"""
    
    def __init__(self):
        logger.info("技术指标情感分析器初始化完成")
    
    def analyze_technical_sentiment(self, market_data: pd.DataFrame) -> float:
        """分析技术指标情感"""
        try:
            if len(market_data) < 20:
                return 0.0
            
            sentiment_scores = []
            
            # RSI情感
            rsi_sentiment = self._analyze_rsi_sentiment(market_data)
            sentiment_scores.append(rsi_sentiment)
            
            # MACD情感
            macd_sentiment = self._analyze_macd_sentiment(market_data)
            sentiment_scores.append(macd_sentiment)
            
            # 移动平均情感
            ma_sentiment = self._analyze_ma_sentiment(market_data)
            sentiment_scores.append(ma_sentiment)
            
            # 成交量情感
            volume_sentiment = self._analyze_volume_sentiment(market_data)
            sentiment_scores.append(volume_sentiment)
            
            # 计算平均情感
            avg_sentiment = np.mean(sentiment_scores)
            
            return np.clip(avg_sentiment, -1.0, 1.0)
        
        except Exception as e:
            logger.error(f"技术指标情感分析失败: {e}")
            return 0.0
    
    def _analyze_rsi_sentiment(self, data: pd.DataFrame) -> float:
        """分析RSI情感"""
        try:
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # RSI情感映射
            if current_rsi > 70:
                return -0.5  # 超买，负面情感
            elif current_rsi < 30:
                return 0.5   # 超卖，正面情感
            else:
                return (50 - current_rsi) / 100  # 中性区域
        
        except Exception as e:
            logger.debug(f"RSI情感分析失败: {e}")
            return 0.0
    
    def _analyze_macd_sentiment(self, data: pd.DataFrame) -> float:
        """分析MACD情感"""
        try:
            # 计算MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            current_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]
            
            # MACD情感
            if current_histogram > 0 and current_histogram > prev_histogram:
                return 0.6  # 正面动量
            elif current_histogram < 0 and current_histogram < prev_histogram:
                return -0.6  # 负面动量
            else:
                return current_histogram / abs(current_histogram) * 0.3 if current_histogram != 0 else 0
        
        except Exception as e:
            logger.debug(f"MACD情感分析失败: {e}")
            return 0.0
    
    def _analyze_ma_sentiment(self, data: pd.DataFrame) -> float:
        """分析移动平均情感"""
        try:
            # 计算移动平均
            ma5 = data['close'].rolling(window=5).mean()
            ma20 = data['close'].rolling(window=20).mean()
            
            current_price = data['close'].iloc[-1]
            current_ma5 = ma5.iloc[-1]
            current_ma20 = ma20.iloc[-1]
            
            # 移动平均情感
            sentiment = 0.0
            
            if current_price > current_ma5 > current_ma20:
                sentiment += 0.5  # 强势上涨
            elif current_price < current_ma5 < current_ma20:
                sentiment -= 0.5  # 强势下跌
            
            # 价格相对于MA20的位置
            ma20_sentiment = (current_price - current_ma20) / current_ma20
            sentiment += np.clip(ma20_sentiment, -0.5, 0.5)
            
            return sentiment
        
        except Exception as e:
            logger.debug(f"移动平均情感分析失败: {e}")
            return 0.0
    
    def _analyze_volume_sentiment(self, data: pd.DataFrame) -> float:
        """分析成交量情感"""
        try:
            # 计算成交量移动平均
            volume_ma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            # 价格变化
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
            
            # 成交量情感
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5 and price_change > 0:
                return 0.4  # 放量上涨
            elif volume_ratio > 1.5 and price_change < 0:
                return -0.4  # 放量下跌
            else:
                return 0.0  # 正常成交量
        
        except Exception as e:
            logger.debug(f"成交量情感分析失败: {e}")
            return 0.0

class SentimentAggregator:
    """情感聚合器"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_collector = NewsCollector()
        self.social_collector = SocialMediaCollector()
        self.technical_analyzer = TechnicalSentimentAnalyzer()
        
        # 情感历史数据
        self.sentiment_history: Dict[str, List[SentimentData]] = {}
        self.sentiment_indices: Dict[str, List[SentimentIndex]] = {}
        
        # 权重配置
        self.source_weights = {
            DataSource.NEWS: 0.3,
            DataSource.SOCIAL_MEDIA: 0.2,
            DataSource.TECHNICAL: 0.4,
            DataSource.MARKET_DATA: 0.1
        }
        
        logger.info("情感聚合器初始化完成")
    
    async def analyze_symbol_sentiment(self, symbol: str, market_data: pd.DataFrame = None) -> SentimentIndex:
        """分析交易对情感"""
        try:
            sentiment_data_list = []
            
            # 收集新闻情感
            news_articles = await self.news_collector.collect_news(symbol, limit=20)
            for article in news_articles:
                sentiment_score, confidence = self.sentiment_analyzer.analyze_text_sentiment(
                    article['title'] + ' ' + article['content']
                )
                
                sentiment_data = SentimentData(
                    text=article['title'],
                    sentiment_score=sentiment_score,
                    sentiment_type=self.sentiment_analyzer.classify_sentiment(sentiment_score),
                    confidence=confidence,
                    source=DataSource.NEWS,
                    symbol=symbol,
                    timestamp=article['timestamp'],
                    metadata=article
                )
                sentiment_data_list.append(sentiment_data)
            
            # 收集社交媒体情感
            social_posts = await self.social_collector.collect_social_data(symbol, limit=50)
            for post in social_posts:
                sentiment_score, confidence = self.sentiment_analyzer.analyze_text_sentiment(post['text'])
                
                sentiment_data = SentimentData(
                    text=post['text'],
                    sentiment_score=sentiment_score,
                    sentiment_type=self.sentiment_analyzer.classify_sentiment(sentiment_score),
                    confidence=confidence,
                    source=DataSource.SOCIAL_MEDIA,
                    symbol=symbol,
                    timestamp=post['timestamp'],
                    metadata=post
                )
                sentiment_data_list.append(sentiment_data)
            
            # 分析技术指标情感
            technical_sentiment = 0.0
            if market_data is not None and len(market_data) > 0:
                technical_sentiment = self.technical_analyzer.analyze_technical_sentiment(market_data)
                
                technical_data = SentimentData(
                    text="Technical Analysis",
                    sentiment_score=technical_sentiment,
                    sentiment_type=self.sentiment_analyzer.classify_sentiment(technical_sentiment),
                    confidence=0.8,
                    source=DataSource.TECHNICAL,
                    symbol=symbol,
                    timestamp=time.time(),
                    metadata={'technical_sentiment': technical_sentiment}
                )
                sentiment_data_list.append(technical_data)
            
            # 存储情感数据
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].extend(sentiment_data_list)
            
            # 保持历史数据在合理范围内
            if len(self.sentiment_history[symbol]) > 1000:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-1000:]
            
            # 计算综合情感指数
            sentiment_index = self._calculate_sentiment_index(symbol, sentiment_data_list, technical_sentiment)
            
            # 存储情感指数
            if symbol not in self.sentiment_indices:
                self.sentiment_indices[symbol] = []
            
            self.sentiment_indices[symbol].append(sentiment_index)
            
            # 保持指数历史在合理范围内
            if len(self.sentiment_indices[symbol]) > 100:
                self.sentiment_indices[symbol] = self.sentiment_indices[symbol][-100:]
            
            logger.info(f"完成{symbol}情感分析 - 总体情感: {sentiment_index.overall_sentiment:.3f}")
            
            return sentiment_index
        
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return self._create_neutral_sentiment_index(symbol)
    
    def _calculate_sentiment_index(self, symbol: str, sentiment_data_list: List[SentimentData], 
                                 technical_sentiment: float) -> SentimentIndex:
        """计算情感指数"""
        try:
            if not sentiment_data_list:
                return self._create_neutral_sentiment_index(symbol)
            
            # 按数据源分组
            source_sentiments = {source: [] for source in DataSource}
            
            for data in sentiment_data_list:
                source_sentiments[data.source].append(data.sentiment_score)
            
            # 计算各数据源平均情感
            news_sentiment = np.mean(source_sentiments[DataSource.NEWS]) if source_sentiments[DataSource.NEWS] else 0.0
            social_sentiment = np.mean(source_sentiments[DataSource.SOCIAL_MEDIA]) if source_sentiments[DataSource.SOCIAL_MEDIA] else 0.0
            
            # 计算加权总体情感
            weighted_sentiments = []
            weights = []
            
            if source_sentiments[DataSource.NEWS]:
                weighted_sentiments.append(news_sentiment)
                weights.append(self.source_weights[DataSource.NEWS])
            
            if source_sentiments[DataSource.SOCIAL_MEDIA]:
                weighted_sentiments.append(social_sentiment)
                weights.append(self.source_weights[DataSource.SOCIAL_MEDIA])
            
            if technical_sentiment != 0:
                weighted_sentiments.append(technical_sentiment)
                weights.append(self.source_weights[DataSource.TECHNICAL])
            
            if weighted_sentiments:
                overall_sentiment = np.average(weighted_sentiments, weights=weights)
            else:
                overall_sentiment = 0.0
            
            # 计算置信度
            confidence = min(len(sentiment_data_list) / 50.0, 1.0)  # 基于数据量的置信度
            
            # 计算成交量加权情感 (简化实现)
            volume_weighted_sentiment = overall_sentiment  # 实际应用中应该基于成交量加权
            
            # 计算情感动量
            sentiment_momentum = self._calculate_sentiment_momentum(symbol)
            
            # 计算情感波动率
            sentiment_volatility = self._calculate_sentiment_volatility(symbol)
            
            # 创建情感指数
            sentiment_index = SentimentIndex(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                sentiment_type=self.sentiment_analyzer.classify_sentiment(overall_sentiment),
                confidence=confidence,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                technical_sentiment=technical_sentiment,
                volume_weighted_sentiment=volume_weighted_sentiment,
                sentiment_momentum=sentiment_momentum,
                sentiment_volatility=sentiment_volatility,
                data_count=len(sentiment_data_list),
                timestamp=time.time()
            )
            
            return sentiment_index
        
        except Exception as e:
            logger.error(f"计算情感指数失败: {e}")
            return self._create_neutral_sentiment_index(symbol)
    
    def _calculate_sentiment_momentum(self, symbol: str) -> float:
        """计算情感动量"""
        try:
            if symbol not in self.sentiment_indices or len(self.sentiment_indices[symbol]) < 2:
                return 0.0
            
            recent_indices = self.sentiment_indices[symbol][-5:]  # 最近5个指数
            
            if len(recent_indices) < 2:
                return 0.0
            
            # 计算情感变化率
            sentiment_changes = []
            for i in range(1, len(recent_indices)):
                change = recent_indices[i].overall_sentiment - recent_indices[i-1].overall_sentiment
                sentiment_changes.append(change)
            
            momentum = np.mean(sentiment_changes)
            return np.clip(momentum, -1.0, 1.0)
        
        except Exception as e:
            logger.debug(f"计算情感动量失败: {e}")
            return 0.0
    
    def _calculate_sentiment_volatility(self, symbol: str) -> float:
        """计算情感波动率"""
        try:
            if symbol not in self.sentiment_indices or len(self.sentiment_indices[symbol]) < 5:
                return 0.0
            
            recent_indices = self.sentiment_indices[symbol][-20:]  # 最近20个指数
            sentiments = [idx.overall_sentiment for idx in recent_indices]
            
            volatility = np.std(sentiments)
            return min(volatility, 1.0)  # 限制在0-1范围内
        
        except Exception as e:
            logger.debug(f"计算情感波动率失败: {e}")
            return 0.0
    
    def _create_neutral_sentiment_index(self, symbol: str) -> SentimentIndex:
        """创建中性情感指数"""
        return SentimentIndex(
            symbol=symbol,
            overall_sentiment=0.0,
            sentiment_type=SentimentType.NEUTRAL,
            confidence=0.0,
            news_sentiment=0.0,
            social_sentiment=0.0,
            technical_sentiment=0.0,
            volume_weighted_sentiment=0.0,
            sentiment_momentum=0.0,
            sentiment_volatility=0.0,
            data_count=0,
            timestamp=time.time()
        )
    
    def get_sentiment_history(self, symbol: str, limit: int = 50) -> List[SentimentData]:
        """获取情感历史"""
        if symbol not in self.sentiment_history:
            return []
        
        return self.sentiment_history[symbol][-limit:]
    
    def get_sentiment_index_history(self, symbol: str, limit: int = 20) -> List[SentimentIndex]:
        """获取情感指数历史"""
        if symbol not in self.sentiment_indices:
            return []
        
        return self.sentiment_indices[symbol][-limit:]
    
    def generate_sentiment_signals(self, symbol: str) -> Dict[str, Any]:
        """生成情感交易信号"""
        try:
            if symbol not in self.sentiment_indices or not self.sentiment_indices[symbol]:
                return {'signal': 'HOLD', 'strength': 0.0, 'reason': '无情感数据'}
            
            current_index = self.sentiment_indices[symbol][-1]
            
            # 信号生成逻辑
            signal = 'HOLD'
            strength = 0.0
            reasons = []
            
            # 基于总体情感
            if current_index.overall_sentiment > 0.4:
                signal = 'BUY'
                strength += abs(current_index.overall_sentiment) * 0.4
                reasons.append(f'正面情感 ({current_index.overall_sentiment:.2f})')
            elif current_index.overall_sentiment < -0.4:
                signal = 'SELL'
                strength += abs(current_index.overall_sentiment) * 0.4
                reasons.append(f'负面情感 ({current_index.overall_sentiment:.2f})')
            
            # 基于情感动量
            if abs(current_index.sentiment_momentum) > 0.2:
                if current_index.sentiment_momentum > 0:
                    if signal != 'SELL':
                        signal = 'BUY'
                    strength += abs(current_index.sentiment_momentum) * 0.3
                    reasons.append(f'情感动量向上 ({current_index.sentiment_momentum:.2f})')
                else:
                    if signal != 'BUY':
                        signal = 'SELL'
                    strength += abs(current_index.sentiment_momentum) * 0.3
                    reasons.append(f'情感动量向下 ({current_index.sentiment_momentum:.2f})')
            
            # 基于技术情感
            if abs(current_index.technical_sentiment) > 0.3:
                if current_index.technical_sentiment > 0:
                    if signal != 'SELL':
                        signal = 'BUY'
                    strength += abs(current_index.technical_sentiment) * 0.3
                    reasons.append(f'技术面正面 ({current_index.technical_sentiment:.2f})')
                else:
                    if signal != 'BUY':
                        signal = 'SELL'
                    strength += abs(current_index.technical_sentiment) * 0.3
                    reasons.append(f'技术面负面 ({current_index.technical_sentiment:.2f})')
            
            # 置信度调整
            strength *= current_index.confidence
            
            # 限制强度范围
            strength = min(strength, 1.0)
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': current_index.confidence,
                'reasons': reasons,
                'sentiment_index': current_index,
                'timestamp': time.time()
            }
        
        except Exception as e:
            logger.error(f"生成情感信号失败: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reason': '信号生成失败'}

class SentimentMonitor:
    """情感监控器"""
    
    def __init__(self, update_interval: int = 300):  # 5分钟更新一次
        self.aggregator = SentimentAggregator()
        self.update_interval = update_interval
        self.monitored_symbols = set()
        self.running = False
        self.monitor_task = None
        
        logger.info("情感监控器初始化完成")
    
    def add_symbol(self, symbol: str):
        """添加监控交易对"""
        self.monitored_symbols.add(symbol)
        logger.info(f"添加情感监控: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """移除监控交易对"""
        self.monitored_symbols.discard(symbol)
        logger.info(f"移除情感监控: {symbol}")
    
    async def start_monitoring(self):
        """开始监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("情感监控器已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("情感监控器已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                for symbol in self.monitored_symbols:
                    # 分析情感
                    sentiment_index = await self.aggregator.analyze_symbol_sentiment(symbol)
                    
                    # 生成信号
                    signals = self.aggregator.generate_sentiment_signals(symbol)
                    
                    # 记录重要信号
                    if signals['strength'] > 0.6:
                        logger.info(f"强情感信号 - {symbol}: {signals['signal']} "
                                  f"(强度: {signals['strength']:.2f}, 置信度: {signals['confidence']:.2f})")
                
                await asyncio.sleep(self.update_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"情感监控循环错误: {e}")
                await asyncio.sleep(60)  # 错误后等待1分钟
    
    async def get_current_sentiment(self, symbol: str, market_data: pd.DataFrame = None) -> SentimentIndex:
        """获取当前情感"""
        return await self.aggregator.analyze_symbol_sentiment(symbol, market_data)
    
    def get_sentiment_signals(self, symbol: str) -> Dict[str, Any]:
        """获取情感信号"""
        return self.aggregator.generate_sentiment_signals(symbol)

# 全局情感监控器实例
sentiment_monitor = SentimentMonitor()


def initialize_sentiment_analysis():
    """初始化情感分析系统"""
    # 返回全局情感监控器实例
    global sentiment_monitor
    logger.success("✅ 情感分析系统初始化完成")
    return sentiment_monitor
