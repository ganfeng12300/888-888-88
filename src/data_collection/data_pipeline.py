"""
🔄 数据处理管道
生产级实时数据处理、清洗、特征工程和存储系统
支持多数据源、实时流处理和批量处理
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import redis.asyncio as redis
from clickhouse_driver import Client as ClickHouseClient
import psycopg2
from psycopg2.extras import RealDictCursor
import kafka
from kafka import KafkaProducer, KafkaConsumer
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.hardware.storage_manager import DataType as StorageDataType, storage_manager
from src.data_collection.exchange_connector import MarketData, DataType


class ProcessingStage(Enum):
    """处理阶段"""
    RAW = "raw"
    CLEANED = "cleaned"
    FEATURED = "featured"
    AGGREGATED = "aggregated"
    STORED = "stored"


class DataQuality(Enum):
    """数据质量"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ProcessedData:
    """处理后的数据结构"""
    original_data: MarketData
    stage: ProcessingStage
    quality: DataQuality
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """管道配置"""
    enable_cleaning: bool = True
    enable_feature_engineering: bool = True
    enable_aggregation: bool = True
    enable_storage: bool = True
    batch_size: int = 1000
    max_queue_size: int = 10000
    processing_timeout: float = 5.0
    redis_url: str = "redis://localhost:6379/0"
    clickhouse_url: str = "clickhouse://localhost:8123"
    postgres_url: str = "postgresql://user:pass@localhost:5432/db"
    kafka_bootstrap_servers: str = "localhost:9092"


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.outlier_threshold = 3.0  # 标准差倍数
        self.price_change_threshold = 0.1  # 10%价格变化阈值
        
    async def clean_ticker_data(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], DataQuality]:
        """清洗ticker数据"""
        try:
            cleaned_data = data.copy()
            quality = DataQuality.EXCELLENT
            
            # 检查必要字段
            required_fields = ['last', 'bid', 'ask', 'volume']
            for field in required_fields:
                if field not in data or data[field] is None:
                    quality = DataQuality.POOR
                    cleaned_data[field] = 0.0
            
            # 价格合理性检查
            last_price = float(cleaned_data.get('last', 0))
            bid_price = float(cleaned_data.get('bid', 0))
            ask_price = float(cleaned_data.get('ask', 0))
            
            if last_price <= 0 or bid_price <= 0 or ask_price <= 0:
                quality = DataQuality.INVALID
                return cleaned_data, quality
            
            # 买卖价差检查
            spread = ask_price - bid_price
            if spread < 0 or spread / last_price > 0.05:  # 5%价差阈值
                quality = DataQuality.FAIR
            
            # 成交量检查
            volume = float(cleaned_data.get('volume', 0))
            if volume < 0:
                cleaned_data['volume'] = 0
                quality = DataQuality.FAIR
            
            return cleaned_data, quality
            
        except Exception as e:
            logger.error(f"清洗ticker数据失败: {e}")
            return data, DataQuality.INVALID
    
    async def clean_orderbook_data(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], DataQuality]:
        """清洗订单簿数据"""
        try:
            cleaned_data = data.copy()
            quality = DataQuality.EXCELLENT
            
            # 检查买卖盘数据
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return cleaned_data, DataQuality.INVALID
            
            # 清洗买盘数据
            cleaned_bids = []
            for bid in bids:
                if len(bid) >= 2 and bid[0] > 0 and bid[1] > 0:
                    cleaned_bids.append([float(bid[0]), float(bid[1])])
            
            # 清洗卖盘数据
            cleaned_asks = []
            for ask in asks:
                if len(ask) >= 2 and ask[0] > 0 and ask[1] > 0:
                    cleaned_asks.append([float(ask[0]), float(ask[1])])
            
            # 排序检查
            if cleaned_bids != sorted(cleaned_bids, key=lambda x: x[0], reverse=True):
                cleaned_bids = sorted(cleaned_bids, key=lambda x: x[0], reverse=True)
                quality = DataQuality.GOOD
            
            if cleaned_asks != sorted(cleaned_asks, key=lambda x: x[0]):
                cleaned_asks = sorted(cleaned_asks, key=lambda x: x[0])
                quality = DataQuality.GOOD
            
            # 价格交叉检查
            if cleaned_bids and cleaned_asks and cleaned_bids[0][0] >= cleaned_asks[0][0]:
                quality = DataQuality.POOR
            
            cleaned_data['bids'] = cleaned_bids
            cleaned_data['asks'] = cleaned_asks
            
            return cleaned_data, quality
            
        except Exception as e:
            logger.error(f"清洗订单簿数据失败: {e}")
            return data, DataQuality.INVALID
    
    async def clean_trade_data(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], DataQuality]:
        """清洗交易数据"""
        try:
            cleaned_data = data.copy()
            quality = DataQuality.EXCELLENT
            
            # 检查必要字段
            price = float(cleaned_data.get('price', 0))
            amount = float(cleaned_data.get('amount', 0))
            
            if price <= 0 or amount <= 0:
                return cleaned_data, DataQuality.INVALID
            
            # 时间戳检查
            timestamp = cleaned_data.get('timestamp')
            if not timestamp:
                cleaned_data['timestamp'] = time.time() * 1000
                quality = DataQuality.GOOD
            
            return cleaned_data, quality
            
        except Exception as e:
            logger.error(f"清洗交易数据失败: {e}")
            return data, DataQuality.INVALID


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.price_history = {}  # 价格历史缓存
        self.volume_history = {}  # 成交量历史缓存
        self.history_size = 100  # 历史数据大小
        
    async def extract_ticker_features(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """提取ticker特征"""
        try:
            features = {}
            
            # 基础特征
            last_price = float(data.get('last', 0))
            bid_price = float(data.get('bid', 0))
            ask_price = float(data.get('ask', 0))
            volume = float(data.get('volume', 0))
            
            # 价差特征
            spread = ask_price - bid_price
            features['spread'] = spread
            features['spread_pct'] = (spread / last_price) * 100 if last_price > 0 else 0
            features['mid_price'] = (bid_price + ask_price) / 2
            
            # 价格变化特征
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.history_size)
            
            price_hist = self.price_history[symbol]
            if price_hist:
                prev_price = price_hist[-1]
                price_change = last_price - prev_price
                features['price_change'] = price_change
                features['price_change_pct'] = (price_change / prev_price) * 100 if prev_price > 0 else 0
                
                # 价格波动率
                if len(price_hist) >= 20:
                    prices = np.array(list(price_hist))
                    returns = np.diff(prices) / prices[:-1]
                    features['volatility'] = np.std(returns) * np.sqrt(1440)  # 日化波动率
            
            price_hist.append(last_price)
            
            # 成交量特征
            if symbol not in self.volume_history:
                self.volume_history[symbol] = deque(maxlen=self.history_size)
            
            volume_hist = self.volume_history[symbol]
            if volume_hist:
                avg_volume = np.mean(list(volume_hist))
                features['volume_ratio'] = volume / avg_volume if avg_volume > 0 else 1
            
            volume_hist.append(volume)
            
            # 技术指标
            if len(price_hist) >= 20:
                prices = np.array(list(price_hist))
                
                # 移动平均
                features['sma_5'] = np.mean(prices[-5:])
                features['sma_20'] = np.mean(prices[-20:])
                features['price_vs_sma20'] = (last_price / features['sma_20'] - 1) * 100
                
                # RSI
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                if len(gains) >= 14:
                    avg_gain = np.mean(gains[-14:])
                    avg_loss = np.mean(losses[-14:])
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        features['rsi'] = 100 - (100 / (1 + rs))
            
            return features
            
        except Exception as e:
            logger.error(f"提取ticker特征失败: {e}")
            return {}
    
    async def extract_orderbook_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取订单簿特征"""
        try:
            features = {}
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return features
            
            # 最优买卖价
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            # 深度特征
            bid_depth_5 = sum([bid[1] for bid in bids[:5]])
            ask_depth_5 = sum([ask[1] for ask in asks[:5]])
            
            features['bid_depth_5'] = bid_depth_5
            features['ask_depth_5'] = ask_depth_5
            features['depth_imbalance'] = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5)
            
            # 价格分布特征
            bid_prices = [bid[0] for bid in bids[:10]]
            ask_prices = [ask[0] for ask in asks[:10]]
            
            features['bid_price_std'] = np.std(bid_prices)
            features['ask_price_std'] = np.std(ask_prices)
            
            # 订单簿斜率
            if len(bids) >= 5 and len(asks) >= 5:
                bid_slope = (bids[0][0] - bids[4][0]) / 5
                ask_slope = (asks[4][0] - asks[0][0]) / 5
                features['bid_slope'] = bid_slope
                features['ask_slope'] = ask_slope
            
            return features
            
        except Exception as e:
            logger.error(f"提取订单簿特征失败: {e}")
            return {}


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        self.aggregation_windows = [60, 300, 900, 3600]  # 1分钟、5分钟、15分钟、1小时
        self.aggregated_data = {}
        
    async def aggregate_ticker_data(self, data: List[Dict[str, Any]], 
                                  symbol: str, window: int) -> Dict[str, Any]:
        """聚合ticker数据"""
        try:
            if not data:
                return {}
            
            prices = [float(d.get('last', 0)) for d in data if d.get('last')]
            volumes = [float(d.get('volume', 0)) for d in data if d.get('volume')]
            
            if not prices:
                return {}
            
            aggregated = {
                'symbol': symbol,
                'window': window,
                'timestamp': time.time(),
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': sum(volumes) if volumes else 0,
                'count': len(prices),
                'vwap': 0  # 成交量加权平均价
            }
            
            # 计算VWAP
            if volumes and sum(volumes) > 0:
                total_value = sum(p * v for p, v in zip(prices, volumes))
                aggregated['vwap'] = total_value / sum(volumes)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"聚合ticker数据失败: {e}")
            return {}


class DataStorage:
    """数据存储器"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.redis_client = None
        self.clickhouse_client = None
        self.postgres_conn = None
        self.kafka_producer = None
        
    async def initialize(self):
        """初始化存储连接"""
        try:
            # Redis连接
            self.redis_client = redis.from_url(self.config.redis_url)
            
            # ClickHouse连接
            self.clickhouse_client = ClickHouseClient(host='localhost')
            
            # PostgreSQL连接
            self.postgres_conn = psycopg2.connect(self.config.postgres_url)
            
            # Kafka生产者
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            logger.info("数据存储连接初始化完成")
            
        except Exception as e:
            logger.error(f"初始化存储连接失败: {e}")
    
    async def store_realtime_data(self, processed_data: ProcessedData):
        """存储实时数据到Redis"""
        try:
            if not self.redis_client:
                return
            
            key = f"realtime:{processed_data.original_data.exchange}:{processed_data.original_data.symbol}:{processed_data.original_data.data_type.value}"
            
            data_to_store = {
                'data': processed_data.original_data.data,
                'features': processed_data.features,
                'quality': processed_data.quality.value,
                'timestamp': processed_data.original_data.timestamp
            }
            
            await self.redis_client.setex(key, 3600, json.dumps(data_to_store))  # 1小时过期
            
        except Exception as e:
            logger.error(f"存储实时数据失败: {e}")
    
    async def store_historical_data(self, processed_data: ProcessedData):
        """存储历史数据到ClickHouse"""
        try:
            if not self.clickhouse_client:
                return
            
            # 根据数据类型选择表
            table_name = f"market_data_{processed_data.original_data.data_type.value}"
            
            # 准备数据
            data_to_insert = {
                'timestamp': processed_data.original_data.timestamp,
                'exchange': processed_data.original_data.exchange,
                'symbol': processed_data.original_data.symbol,
                'data': json.dumps(processed_data.original_data.data),
                'features': json.dumps(processed_data.features),
                'quality': processed_data.quality.value,
                'processing_time_ms': processed_data.processing_time_ms
            }
            
            # 插入数据
            self.clickhouse_client.execute(
                f"INSERT INTO {table_name} VALUES",
                [data_to_insert]
            )
            
        except Exception as e:
            logger.error(f"存储历史数据失败: {e}")
    
    async def publish_to_kafka(self, processed_data: ProcessedData):
        """发布数据到Kafka"""
        try:
            if not self.kafka_producer:
                return
            
            topic = f"market_data_{processed_data.original_data.data_type.value}"
            
            message = {
                'exchange': processed_data.original_data.exchange,
                'symbol': processed_data.original_data.symbol,
                'data': processed_data.original_data.data,
                'features': processed_data.features,
                'quality': processed_data.quality.value,
                'timestamp': processed_data.original_data.timestamp
            }
            
            self.kafka_producer.send(topic, value=message)
            
        except Exception as e:
            logger.error(f"发布Kafka消息失败: {e}")


class DataPipeline:
    """数据处理管道"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.aggregator = DataAggregator()
        self.storage = DataStorage(config)
        
        # 处理队列
        self.processing_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.is_running = False
        
        # 统计信息
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'avg_processing_time': 0.0,
            'quality_distribution': {q.value: 0 for q in DataQuality}
        }
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.FEATURE_ENGINEERING, [5, 6, 7, 8])
        
        logger.info("数据处理管道初始化完成")
    
    async def start(self):
        """启动管道"""
        try:
            await self.storage.initialize()
            self.is_running = True
            
            # 启动处理任务
            for i in range(4):  # 4个处理协程
                asyncio.create_task(self._process_worker(f"worker-{i}"))
            
            logger.info("数据处理管道已启动")
            
        except Exception as e:
            logger.error(f"启动管道失败: {e}")
    
    async def stop(self):
        """停止管道"""
        self.is_running = False
        logger.info("数据处理管道已停止")
    
    async def process_data(self, market_data: MarketData) -> Optional[ProcessedData]:
        """处理单条数据"""
        try:
            if self.processing_queue.full():
                logger.warning("处理队列已满，丢弃数据")
                return None
            
            await self.processing_queue.put(market_data)
            return None
            
        except Exception as e:
            logger.error(f"添加数据到处理队列失败: {e}")
            return None
    
    async def _process_worker(self, worker_name: str):
        """处理工作协程"""
        logger.info(f"启动处理工作协程: {worker_name}")
        
        while self.is_running:
            try:
                # 获取数据
                market_data = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                # 处理数据
                processed_data = await self._process_single_data(market_data)
                
                if processed_data:
                    # 存储数据
                    await self._store_processed_data(processed_data)
                    
                    # 更新统计
                    self._update_stats(processed_data)
                
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"处理工作协程 {worker_name} 出错: {e}")
                self.stats['error_count'] += 1
    
    async def _process_single_data(self, market_data: MarketData) -> Optional[ProcessedData]:
        """处理单条数据"""
        start_time = time.time()
        
        try:
            processed_data = ProcessedData(
                original_data=market_data,
                stage=ProcessingStage.RAW,
                quality=DataQuality.EXCELLENT
            )
            
            # 数据清洗
            if self.config.enable_cleaning:
                cleaned_data, quality = await self._clean_data(market_data)
                processed_data.original_data.data = cleaned_data
                processed_data.quality = quality
                processed_data.stage = ProcessingStage.CLEANED
                
                if quality == DataQuality.INVALID:
                    processed_data.errors.append("数据质量无效")
                    return processed_data
            
            # 特征工程
            if self.config.enable_feature_engineering:
                features = await self._extract_features(market_data)
                processed_data.features = features
                processed_data.stage = ProcessingStage.FEATURED
            
            # 计算处理时间
            processing_time = (time.time() - start_time) * 1000
            processed_data.processing_time_ms = processing_time
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理数据失败: {e}")
            return None
    
    async def _clean_data(self, market_data: MarketData) -> tuple[Dict[str, Any], DataQuality]:
        """清洗数据"""
        try:
            if market_data.data_type == DataType.TICKER:
                return await self.cleaner.clean_ticker_data(market_data.data)
            elif market_data.data_type == DataType.ORDERBOOK:
                return await self.cleaner.clean_orderbook_data(market_data.data)
            elif market_data.data_type == DataType.TRADES:
                return await self.cleaner.clean_trade_data(market_data.data)
            else:
                return market_data.data, DataQuality.EXCELLENT
                
        except Exception as e:
            logger.error(f"清洗数据失败: {e}")
            return market_data.data, DataQuality.INVALID
    
    async def _extract_features(self, market_data: MarketData) -> Dict[str, Any]:
        """提取特征"""
        try:
            if market_data.data_type == DataType.TICKER:
                return await self.feature_engineer.extract_ticker_features(
                    market_data.data, market_data.symbol
                )
            elif market_data.data_type == DataType.ORDERBOOK:
                return await self.feature_engineer.extract_orderbook_features(market_data.data)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return {}
    
    async def _store_processed_data(self, processed_data: ProcessedData):
        """存储处理后的数据"""
        try:
            # 存储到Redis（实时数据）
            await self.storage.store_realtime_data(processed_data)
            
            # 存储到ClickHouse（历史数据）
            if self.config.enable_storage:
                await self.storage.store_historical_data(processed_data)
            
            # 发布到Kafka
            await self.storage.publish_to_kafka(processed_data)
            
        except Exception as e:
            logger.error(f"存储处理数据失败: {e}")
    
    def _update_stats(self, processed_data: ProcessedData):
        """更新统计信息"""
        self.stats['processed_count'] += 1
        self.stats['quality_distribution'][processed_data.quality.value] += 1
        
        # 更新平均处理时间
        current_avg = self.stats['avg_processing_time']
        count = self.stats['processed_count']
        new_avg = (current_avg * (count - 1) + processed_data.processing_time_ms) / count
        self.stats['avg_processing_time'] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# 全局数据管道实例
data_pipeline = None


def create_pipeline(config: PipelineConfig) -> DataPipeline:
    """创建数据管道"""
    global data_pipeline
    data_pipeline = DataPipeline(config)
    return data_pipeline


async def main():
    """测试主函数"""
    logger.info("启动数据处理管道测试...")
    
    # 创建配置
    config = PipelineConfig()
    
    # 创建管道
    pipeline = create_pipeline(config)
    
    try:
        # 启动管道
        await pipeline.start()
        
        # 模拟数据处理
        from src.data_collection.exchange_connector import MarketData, DataType
        
        test_data = MarketData(
            exchange="binance",
            symbol="BTC/USDT",
            data_type=DataType.TICKER,
            timestamp=time.time(),
            data={
                'last': 50000.0,
                'bid': 49999.0,
                'ask': 50001.0,
                'volume': 1000.0
            }
        )
        
        # 处理测试数据
        for i in range(100):
            test_data.data['last'] = 50000 + i * 10
            await pipeline.process_data(test_data)
            await asyncio.sleep(0.1)
        
        # 等待处理完成
        await asyncio.sleep(5)
        
        # 获取统计信息
        stats = pipeline.get_stats()
        logger.info(f"处理统计: {stats}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
