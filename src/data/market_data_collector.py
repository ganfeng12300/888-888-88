#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 市场数据采集器
实时数据采集系统，支持多交易所WebSocket连接
专为史诗级AI量化交易设计，毫秒级数据处理
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
import ccxt.async_support as ccxt
from concurrent.futures import ThreadPoolExecutor
import redis
import sqlite3
import gzip
import pickle

@dataclass
class MarketTick:
    """市场tick数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    exchange: str
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OHLCV:
    """OHLCV K线数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    timeframe: str

class ExchangeConnector:
    """交易所连接器基类"""
    
    def __init__(self, exchange_name: str, config: Dict[str, Any]):
        self.exchange_name = exchange_name
        self.config = config
        self.is_connected = False
        self.websocket = None
        self.callbacks = {}
        self.reconnect_count = 0
        self.max_reconnects = 10
        
    async def connect(self):
        """连接到交易所"""
        raise NotImplementedError
    
    async def subscribe_ticker(self, symbols: List[str], callback: Callable):
        """订阅ticker数据"""
        raise NotImplementedError
    
    async def subscribe_orderbook(self, symbols: List[str], callback: Callable):
        """订阅订单簿数据"""
        raise NotImplementedError
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False

class BinanceConnector(ExchangeConnector):
    """币安交易所连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("binance", config)
        self.base_url = "wss://stream.binance.com:9443/ws/"
        self.subscriptions = []
        
    async def connect(self):
        """连接到币安WebSocket"""
        try:
            # 构建订阅URL
            if self.subscriptions:
                stream_names = "/".join(self.subscriptions)
                url = f"{self.base_url}{stream_names}"
            else:
                url = self.base_url
            
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.reconnect_count = 0
            
            logger.info(f"🔗 币安WebSocket连接成功: {url}")
            
            # 启动消息处理
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"❌ 币安连接失败: {e}")
            await self._reconnect()
    
    async def subscribe_ticker(self, symbols: List[str], callback: Callable):
        """订阅ticker数据"""
        self.callbacks['ticker'] = callback
        
        for symbol in symbols:
            stream_name = f"{symbol.lower()}@ticker"
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
        
        if self.is_connected:
            await self._resubscribe()
    
    async def subscribe_orderbook(self, symbols: List[str], callback: Callable):
        """订阅订单簿数据"""
        self.callbacks['depth'] = callback
        
        for symbol in symbols:
            stream_name = f"{symbol.lower()}@depth20@100ms"
            if stream_name not in self.subscriptions:
                self.subscriptions.append(stream_name)
        
        if self.is_connected:
            await self._resubscribe()
    
    async def _handle_messages(self):
        """处理WebSocket消息"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ JSON解析失败: {e}")
                except Exception as e:
                    logger.error(f"❌ 消息处理失败: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ 币安WebSocket连接断开")
            self.is_connected = False
            await self._reconnect()
        except Exception as e:
            logger.error(f"❌ WebSocket消息处理异常: {e}")
            await self._reconnect()
    
    async def _process_message(self, data: Dict[str, Any]):
        """处理具体消息"""
        try:
            if 'stream' in data:
                stream = data['stream']
                payload = data['data']
                
                if '@ticker' in stream:
                    await self._handle_ticker(payload)
                elif '@depth' in stream:
                    await self._handle_depth(payload)
                    
        except Exception as e:
            logger.error(f"❌ 消息处理失败: {e}")
    
    async def _handle_ticker(self, data: Dict[str, Any]):
        """处理ticker数据"""
        try:
            tick = MarketTick(
                symbol=data['s'],
                timestamp=datetime.fromtimestamp(data['E'] / 1000, tz=timezone.utc),
                price=float(data['c']),
                volume=float(data['v']),
                bid=float(data['b']),
                ask=float(data['a']),
                bid_volume=float(data['B']),
                ask_volume=float(data['A']),
                exchange='binance',
                raw_data=data
            )
            
            if 'ticker' in self.callbacks:
                await self.callbacks['ticker'](tick)
                
        except Exception as e:
            logger.error(f"❌ Ticker处理失败: {e}")
    
    async def _handle_depth(self, data: Dict[str, Any]):
        """处理深度数据"""
        try:
            if 'depth' in self.callbacks:
                await self.callbacks['depth'](data)
        except Exception as e:
            logger.error(f"❌ 深度数据处理失败: {e}")
    
    async def _resubscribe(self):
        """重新订阅"""
        if self.websocket and self.is_connected:
            await self.websocket.close()
            await self.connect()
    
    async def _reconnect(self):
        """重连机制"""
        if self.reconnect_count >= self.max_reconnects:
            logger.error(f"❌ 币安重连次数超限: {self.reconnect_count}")
            return
        
        self.reconnect_count += 1
        wait_time = min(2 ** self.reconnect_count, 60)  # 指数退避
        
        logger.info(f"🔄 币安重连中... 第{self.reconnect_count}次，等待{wait_time}秒")
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"❌ 币安重连失败: {e}")
            await self._reconnect()

class OKXConnector(ExchangeConnector):
    """OKX交易所连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("okx", config)
        self.base_url = "wss://ws.okx.com:8443/ws/v5/public"
        
    async def connect(self):
        """连接到OKX WebSocket"""
        try:
            self.websocket = await websockets.connect(self.base_url)
            self.is_connected = True
            self.reconnect_count = 0
            
            logger.info(f"🔗 OKX WebSocket连接成功")
            
            # 启动消息处理
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"❌ OKX连接失败: {e}")
            await self._reconnect()
    
    async def subscribe_ticker(self, symbols: List[str], callback: Callable):
        """订阅ticker数据"""
        self.callbacks['ticker'] = callback
        
        args = []
        for symbol in symbols:
            args.append({
                "channel": "tickers",
                "instId": symbol
            })
        
        subscribe_msg = {
            "op": "subscribe",
            "args": args
        }
        
        if self.websocket and self.is_connected:
            await self.websocket.send(json.dumps(subscribe_msg))
    
    async def _handle_messages(self):
        """处理WebSocket消息"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except Exception as e:
                    logger.error(f"❌ OKX消息处理失败: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ OKX WebSocket连接断开")
            self.is_connected = False
            await self._reconnect()
    
    async def _process_message(self, data: Dict[str, Any]):
        """处理具体消息"""
        try:
            if 'data' in data and data.get('arg', {}).get('channel') == 'tickers':
                for item in data['data']:
                    await self._handle_ticker(item)
        except Exception as e:
            logger.error(f"❌ OKX消息处理失败: {e}")
    
    async def _handle_ticker(self, data: Dict[str, Any]):
        """处理ticker数据"""
        try:
            tick = MarketTick(
                symbol=data['instId'],
                timestamp=datetime.fromtimestamp(int(data['ts']) / 1000, tz=timezone.utc),
                price=float(data['last']),
                volume=float(data['vol24h']),
                bid=float(data['bidPx']),
                ask=float(data['askPx']),
                bid_volume=float(data['bidSz']),
                ask_volume=float(data['askSz']),
                exchange='okx',
                raw_data=data
            )
            
            if 'ticker' in self.callbacks:
                await self.callbacks['ticker'](tick)
                
        except Exception as e:
            logger.error(f"❌ OKX Ticker处理失败: {e}")
    
    async def _reconnect(self):
        """重连机制"""
        if self.reconnect_count >= self.max_reconnects:
            logger.error(f"❌ OKX重连次数超限: {self.reconnect_count}")
            return
        
        self.reconnect_count += 1
        wait_time = min(2 ** self.reconnect_count, 60)
        
        logger.info(f"🔄 OKX重连中... 第{self.reconnect_count}次，等待{wait_time}秒")
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"❌ OKX重连失败: {e}")
            await self._reconnect()

class MarketDataCollector:
    """🦊 猎狐AI - 市场数据采集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges = {}
        self.data_handlers = []
        self.is_running = False
        
        # 数据存储
        self.redis_client = None
        self.db_connection = None
        
        # 性能统计
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
        # 初始化存储
        self._init_storage()
        
        logger.info("🦊 猎狐AI市场数据采集器初始化完成")
    
    def _init_storage(self):
        """初始化存储系统"""
        try:
            # Redis连接（实时数据缓存）
            if self.config.get('redis', {}).get('enabled', False):
                import redis
                self.redis_client = redis.Redis(
                    host=self.config['redis'].get('host', 'localhost'),
                    port=self.config['redis'].get('port', 6379),
                    db=self.config['redis'].get('db', 0),
                    decode_responses=True
                )
                logger.info("✅ Redis连接成功")
            
            # SQLite连接（历史数据存储）
            db_path = self.config.get('database', {}).get('path', 'data/market_data.db')
            self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
            
            # 创建表
            self._create_tables()
            logger.info("✅ SQLite数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 存储系统初始化失败: {e}")
    
    def _create_tables(self):
        """创建数据库表"""
        cursor = self.db_connection.cursor()
        
        # Tick数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                bid_volume REAL NOT NULL,
                ask_volume REAL NOT NULL,
                exchange TEXT NOT NULL,
                raw_data TEXT
            )
        ''')
        
        # OHLCV数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                exchange TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                UNIQUE(symbol, timestamp, exchange, timeframe)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, timestamp)')
        
        self.db_connection.commit()
    
    async def add_exchange(self, exchange_name: str, config: Dict[str, Any]):
        """添加交易所连接"""
        try:
            if exchange_name.lower() == 'binance':
                connector = BinanceConnector(config)
            elif exchange_name.lower() == 'okx':
                connector = OKXConnector(config)
            else:
                logger.error(f"❌ 不支持的交易所: {exchange_name}")
                return False
            
            self.exchanges[exchange_name] = connector
            await connector.connect()
            
            logger.info(f"✅ 交易所 {exchange_name} 添加成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 添加交易所失败 {exchange_name}: {e}")
            return False
    
    async def subscribe_symbols(self, symbols: List[str], exchanges: List[str] = None):
        """订阅交易对数据"""
        if exchanges is None:
            exchanges = list(self.exchanges.keys())
        
        for exchange_name in exchanges:
            if exchange_name in self.exchanges:
                connector = self.exchanges[exchange_name]
                
                # 订阅ticker数据
                await connector.subscribe_ticker(symbols, self._handle_tick_data)
                
                # 订阅订单簿数据
                await connector.subscribe_orderbook(symbols, self._handle_orderbook_data)
                
                logger.info(f"✅ {exchange_name} 订阅完成: {symbols}")
    
    async def _handle_tick_data(self, tick: MarketTick):
        """处理tick数据"""
        try:
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = datetime.now(timezone.utc)
            
            # 存储到Redis（实时缓存）
            if self.redis_client:
                key = f"tick:{tick.exchange}:{tick.symbol}"
                data = {
                    'price': tick.price,
                    'volume': tick.volume,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'timestamp': tick.timestamp.isoformat()
                }
                self.redis_client.hset(key, mapping=data)
                self.redis_client.expire(key, 3600)  # 1小时过期
            
            # 存储到SQLite（历史数据）
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    INSERT INTO ticks (symbol, timestamp, price, volume, bid, ask, 
                                     bid_volume, ask_volume, exchange, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tick.symbol, tick.timestamp, tick.price, tick.volume,
                    tick.bid, tick.ask, tick.bid_volume, tick.ask_volume,
                    tick.exchange, json.dumps(tick.raw_data)
                ))
                self.db_connection.commit()
            
            # 调用数据处理器
            for handler in self.data_handlers:
                try:
                    await handler(tick)
                except Exception as e:
                    logger.error(f"❌ 数据处理器错误: {e}")
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Tick数据处理失败: {e}")
            self.stats['errors'] += 1
    
    async def _handle_orderbook_data(self, data: Dict[str, Any]):
        """处理订单簿数据"""
        try:
            # 存储到Redis
            if self.redis_client:
                symbol = data.get('s', 'unknown')
                exchange = 'binance'  # 根据实际情况调整
                key = f"orderbook:{exchange}:{symbol}"
                
                orderbook_data = {
                    'bids': json.dumps(data.get('b', [])),
                    'asks': json.dumps(data.get('a', [])),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                self.redis_client.hset(key, mapping=orderbook_data)
                self.redis_client.expire(key, 300)  # 5分钟过期
            
        except Exception as e:
            logger.error(f"❌ 订单簿数据处理失败: {e}")
    
    def add_data_handler(self, handler: Callable):
        """添加数据处理器"""
        self.data_handlers.append(handler)
        logger.info(f"✅ 数据处理器已添加: {handler.__name__}")
    
    async def start(self):
        """启动数据采集"""
        if self.is_running:
            logger.warning("⚠️ 数据采集器已在运行")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        logger.info("🚀 猎狐AI数据采集器启动")
        
        # 启动性能监控
        asyncio.create_task(self._performance_monitor())
    
    async def stop(self):
        """停止数据采集"""
        self.is_running = False
        
        # 断开所有交易所连接
        for connector in self.exchanges.values():
            await connector.disconnect()
        
        # 关闭数据库连接
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("🛑 猎狐AI数据采集器已停止")
    
    async def _performance_monitor(self):
        """性能监控"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟统计一次
                
                if self.stats['start_time']:
                    runtime = datetime.now(timezone.utc) - self.stats['start_time']
                    runtime_seconds = runtime.total_seconds()
                    
                    msg_per_sec = self.stats['messages_processed'] / max(runtime_seconds, 1)
                    error_rate = self.stats['errors'] / max(self.stats['messages_received'], 1) * 100
                    
                    logger.info(f"📊 性能统计 - 消息/秒: {msg_per_sec:.2f}, 错误率: {error_rate:.2f}%")
                
            except Exception as e:
                logger.error(f"❌ 性能监控错误: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime = datetime.now(timezone.utc) - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()
            stats['messages_per_second'] = stats['messages_processed'] / max(stats['runtime_seconds'], 1)
            stats['error_rate'] = stats['errors'] / max(stats['messages_received'], 1) * 100
        
        return stats

# 全局实例
market_data_collector = MarketDataCollector({})

def initialize_market_data_collector(config: Dict[str, Any]) -> MarketDataCollector:
    """初始化市场数据采集器"""
    global market_data_collector
    market_data_collector = MarketDataCollector(config)
    return market_data_collector

if __name__ == "__main__":
    # 测试代码
    async def test_collector():
        config = {
            'redis': {'enabled': False},
            'database': {'path': 'test_market_data.db'}
        }
        
        collector = initialize_market_data_collector(config)
        
        # 添加币安交易所
        await collector.add_exchange('binance', {})
        
        # 订阅BTC和ETH
        await collector.subscribe_symbols(['BTCUSDT', 'ETHUSDT'])
        
        # 启动采集
        await collector.start()
        
        # 运行10秒
        await asyncio.sleep(10)
        
        # 显示统计
        stats = collector.get_stats()
        print(f"统计信息: {json.dumps(stats, indent=2, default=str)}")
        
        # 停止采集
        await collector.stop()
    
    asyncio.run(test_collector())
