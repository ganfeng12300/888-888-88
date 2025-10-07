#!/usr/bin/env python3
"""
📊 888-888-88 市场数据管理系统
生产级多交易所数据收集和管理中心
"""

import asyncio
import aiohttp
import websockets
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import sqlite3
import ccxt.async_support as ccxt
from collections import defaultdict, deque
import gzip
import pickle


class DataType(Enum):
    """数据类型"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINE = "kline"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"


class TimeFrame(Enum):
    """时间周期"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class MarketTicker:
    """市场行情数据"""
    symbol: str
    exchange: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    change_24h: float = 0.0
    change_24h_pct: float = 0.0


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    size: float


@dataclass
class OrderBook:
    """订单簿数据"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    checksum: Optional[str] = None


@dataclass
class Trade:
    """交易数据"""
    symbol: str
    exchange: str
    timestamp: datetime
    price: float
    size: float
    side: str  # buy/sell
    trade_id: Optional[str] = None


@dataclass
class Kline:
    """K线数据"""
    symbol: str
    exchange: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: int = 0
    taker_buy_volume: float = 0.0


class DataStorage:
    """数据存储管理"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.init_database()
    
    def init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建表结构
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                bid REAL,
                ask REAL,
                bid_volume REAL,
                ask_volume REAL,
                change_24h REAL,
                change_24h_pct REAL,
                UNIQUE(symbol, exchange, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trades_count INTEGER,
                taker_buy_volume REAL,
                UNIQUE(symbol, exchange, timeframe, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                side TEXT NOT NULL,
                trade_id TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orderbooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL,
                UNIQUE(symbol, exchange, timestamp)
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickers_symbol_time ON tickers(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_klines_symbol_time ON klines(symbol, timeframe, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)")
        
        conn.commit()
        conn.close()
        logger.info("📊 数据库初始化完成")
    
    def save_ticker(self, ticker: MarketTicker) -> None:
        """保存行情数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO tickers 
                (symbol, exchange, timestamp, open, high, low, close, volume,
                 bid, ask, bid_volume, ask_volume, change_24h, change_24h_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker.symbol, ticker.exchange, int(ticker.timestamp.timestamp()),
                ticker.open, ticker.high, ticker.low, ticker.close, ticker.volume,
                ticker.bid, ticker.ask, ticker.bid_volume, ticker.ask_volume,
                ticker.change_24h, ticker.change_24h_pct
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存行情数据失败: {e}")
        finally:
            conn.close()
    
    def save_kline(self, kline: Kline) -> None:
        """保存K线数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO klines 
                (symbol, exchange, timeframe, timestamp, open, high, low, close, 
                 volume, trades_count, taker_buy_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kline.symbol, kline.exchange, kline.timeframe,
                int(kline.timestamp.timestamp()),
                kline.open, kline.high, kline.low, kline.close,
                kline.volume, kline.trades_count, kline.taker_buy_volume
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存K线数据失败: {e}")
        finally:
            conn.close()
    
    def save_trade(self, trade: Trade) -> None:
        """保存交易数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trades 
                (symbol, exchange, timestamp, price, size, side, trade_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.exchange, int(trade.timestamp.timestamp()),
                trade.price, trade.size, trade.side, trade.trade_id
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存交易数据失败: {e}")
        finally:
            conn.close()
    
    def save_orderbook(self, orderbook: OrderBook) -> None:
        """保存订单簿数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 压缩订单簿数据
            data = {
                'bids': [(level.price, level.size) for level in orderbook.bids],
                'asks': [(level.price, level.size) for level in orderbook.asks]
            }
            compressed_data = gzip.compress(pickle.dumps(data))
            
            cursor.execute("""
                INSERT OR REPLACE INTO orderbooks 
                (symbol, exchange, timestamp, data)
                VALUES (?, ?, ?, ?)
            """, (
                orderbook.symbol, orderbook.exchange,
                int(orderbook.timestamp.timestamp()), compressed_data
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存订单簿数据失败: {e}")
        finally:
            conn.close()
    
    def get_klines(self, symbol: str, exchange: str, timeframe: str,
                   start_time: datetime, end_time: datetime) -> List[Kline]:
        """获取K线数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT symbol, exchange, timeframe, timestamp, open, high, low, close,
                       volume, trades_count, taker_buy_volume
                FROM klines
                WHERE symbol = ? AND exchange = ? AND timeframe = ?
                  AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (
                symbol, exchange, timeframe,
                int(start_time.timestamp()), int(end_time.timestamp())
            ))
            
            klines = []
            for row in cursor.fetchall():
                klines.append(Kline(
                    symbol=row[0],
                    exchange=row[1],
                    timeframe=row[2],
                    timestamp=datetime.fromtimestamp(row[3]),
                    open=row[4],
                    high=row[5],
                    low=row[6],
                    close=row[7],
                    volume=row[8],
                    trades_count=row[9] or 0,
                    taker_buy_volume=row[10] or 0.0
                ))
            
            return klines
            
        except Exception as e:
            logger.error(f"❌ 获取K线数据失败: {e}")
            return []
        finally:
            conn.close()


class MarketDataManager:
    """生产级市场数据管理器"""
    
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.storage = DataStorage()
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 实时数据缓存
        self.ticker_cache: Dict[str, MarketTicker] = {}
        self.orderbook_cache: Dict[str, OrderBook] = {}
        self.recent_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # WebSocket连接
        self.ws_connections: Dict[str, Any] = {}
        self.ws_tasks: List[asyncio.Task] = []
        
        # 数据收集状态
        self.collecting = False
        self.collection_tasks: List[asyncio.Task] = []
        
        # 统计信息
        self.stats = {
            'tickers_received': 0,
            'trades_received': 0,
            'orderbooks_received': 0,
            'klines_received': 0,
            'last_update': datetime.now()
        }
        
        logger.info("📊 市场数据管理器初始化完成")
    
    def add_exchange(self, name: str, exchange: ccxt.Exchange) -> None:
        """添加交易所"""
        self.exchanges[name] = exchange
        logger.info(f"🏢 添加交易所: {name}")
    
    def subscribe(self, data_type: str, callback: Callable) -> None:
        """订阅数据更新"""
        self.subscribers[data_type].append(callback)
        logger.info(f"📡 订阅数据类型: {data_type}")
    
    def unsubscribe(self, data_type: str, callback: Callable) -> None:
        """取消订阅"""
        if callback in self.subscribers[data_type]:
            self.subscribers[data_type].remove(callback)
    
    async def start_collection(self, symbols: List[str]) -> None:
        """开始数据收集"""
        if self.collecting:
            logger.warning("⚠️ 数据收集已在运行")
            return
        
        self.collecting = True
        logger.info(f"🚀 开始收集数据: {symbols}")
        
        # 启动各种数据收集任务
        for exchange_name, exchange in self.exchanges.items():
            # 行情数据收集
            task = asyncio.create_task(
                self._collect_tickers(exchange_name, exchange, symbols)
            )
            self.collection_tasks.append(task)
            
            # K线数据收集
            for timeframe in [TimeFrame.M1, TimeFrame.M5, TimeFrame.H1]:
                task = asyncio.create_task(
                    self._collect_klines(exchange_name, exchange, symbols, timeframe.value)
                )
                self.collection_tasks.append(task)
            
            # WebSocket数据收集
            if hasattr(exchange, 'watch_ticker'):
                task = asyncio.create_task(
                    self._collect_websocket_data(exchange_name, exchange, symbols)
                )
                self.collection_tasks.append(task)
        
        # 启动数据清理任务
        cleanup_task = asyncio.create_task(self._cleanup_old_data())
        self.collection_tasks.append(cleanup_task)
        
        logger.info("✅ 数据收集任务已启动")
    
    async def stop_collection(self) -> None:
        """停止数据收集"""
        self.collecting = False
        
        # 取消所有任务
        for task in self.collection_tasks:
            task.cancel()
        
        # 等待任务完成
        if self.collection_tasks:
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        self.collection_tasks.clear()
        logger.info("⏹️ 数据收集已停止")
    
    async def _collect_tickers(self, exchange_name: str, exchange: ccxt.Exchange,
                             symbols: List[str]) -> None:
        """收集行情数据"""
        while self.collecting:
            try:
                tickers = await exchange.fetch_tickers(symbols)
                
                for symbol, ticker_data in tickers.items():
                    ticker = MarketTicker(
                        symbol=symbol,
                        exchange=exchange_name,
                        timestamp=datetime.now(),
                        open=ticker_data.get('open', 0.0),
                        high=ticker_data.get('high', 0.0),
                        low=ticker_data.get('low', 0.0),
                        close=ticker_data.get('close', 0.0),
                        volume=ticker_data.get('baseVolume', 0.0),
                        bid=ticker_data.get('bid', 0.0),
                        ask=ticker_data.get('ask', 0.0),
                        bid_volume=ticker_data.get('bidVolume', 0.0),
                        ask_volume=ticker_data.get('askVolume', 0.0),
                        change_24h=ticker_data.get('change', 0.0),
                        change_24h_pct=ticker_data.get('percentage', 0.0)
                    )
                    
                    # 更新缓存
                    cache_key = f"{exchange_name}:{symbol}"
                    self.ticker_cache[cache_key] = ticker
                    
                    # 保存到数据库
                    self.storage.save_ticker(ticker)
                    
                    # 通知订阅者
                    await self._notify_subscribers('ticker', ticker)
                    
                    self.stats['tickers_received'] += 1
                
                self.stats['last_update'] = datetime.now()
                await asyncio.sleep(5)  # 5秒更新一次
                
            except Exception as e:
                logger.error(f"❌ 收集{exchange_name}行情数据失败: {e}")
                await asyncio.sleep(10)
    
    async def _collect_klines(self, exchange_name: str, exchange: ccxt.Exchange,
                            symbols: List[str], timeframe: str) -> None:
        """收集K线数据"""
        while self.collecting:
            try:
                for symbol in symbols:
                    try:
                        # 获取最近的K线数据
                        ohlcv = await exchange.fetch_ohlcv(
                            symbol, timeframe, limit=100
                        )
                        
                        for candle in ohlcv:
                            kline = Kline(
                                symbol=symbol,
                                exchange=exchange_name,
                                timeframe=timeframe,
                                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                                open=candle[1],
                                high=candle[2],
                                low=candle[3],
                                close=candle[4],
                                volume=candle[5]
                            )
                            
                            # 保存到数据库
                            self.storage.save_kline(kline)
                            
                            # 通知订阅者
                            await self._notify_subscribers('kline', kline)
                            
                            self.stats['klines_received'] += 1
                    
                    except Exception as e:
                        logger.error(f"❌ 收集{symbol} K线数据失败: {e}")
                        continue
                
                # 根据时间周期调整更新频率
                if timeframe == '1m':
                    await asyncio.sleep(60)
                elif timeframe == '5m':
                    await asyncio.sleep(300)
                else:
                    await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ 收集K线数据失败: {e}")
                await asyncio.sleep(60)
    
    async def _collect_websocket_data(self, exchange_name: str, exchange: ccxt.Exchange,
                                    symbols: List[str]) -> None:
        """收集WebSocket实时数据"""
        while self.collecting:
            try:
                # 订阅实时行情
                for symbol in symbols:
                    try:
                        if hasattr(exchange, 'watch_ticker'):
                            ticker_data = await exchange.watch_ticker(symbol)
                            
                            ticker = MarketTicker(
                                symbol=symbol,
                                exchange=exchange_name,
                                timestamp=datetime.now(),
                                open=ticker_data.get('open', 0.0),
                                high=ticker_data.get('high', 0.0),
                                low=ticker_data.get('low', 0.0),
                                close=ticker_data.get('close', 0.0),
                                volume=ticker_data.get('baseVolume', 0.0),
                                bid=ticker_data.get('bid', 0.0),
                                ask=ticker_data.get('ask', 0.0),
                                bid_volume=ticker_data.get('bidVolume', 0.0),
                                ask_volume=ticker_data.get('askVolume', 0.0)
                            )
                            
                            # 更新缓存
                            cache_key = f"{exchange_name}:{symbol}"
                            self.ticker_cache[cache_key] = ticker
                            
                            # 通知订阅者
                            await self._notify_subscribers('ticker_realtime', ticker)
                        
                        # 订阅交易数据
                        if hasattr(exchange, 'watch_trades'):
                            trades_data = await exchange.watch_trades(symbol)
                            
                            for trade_data in trades_data:
                                trade = Trade(
                                    symbol=symbol,
                                    exchange=exchange_name,
                                    timestamp=datetime.fromtimestamp(
                                        trade_data.get('timestamp', time.time() * 1000) / 1000
                                    ),
                                    price=trade_data.get('price', 0.0),
                                    size=trade_data.get('amount', 0.0),
                                    side=trade_data.get('side', 'unknown'),
                                    trade_id=trade_data.get('id')
                                )
                                
                                # 添加到最近交易缓存
                                cache_key = f"{exchange_name}:{symbol}"
                                self.recent_trades[cache_key].append(trade)
                                
                                # 保存到数据库
                                self.storage.save_trade(trade)
                                
                                # 通知订阅者
                                await self._notify_subscribers('trade', trade)
                                
                                self.stats['trades_received'] += 1
                    
                    except Exception as e:
                        logger.error(f"❌ WebSocket收集{symbol}数据失败: {e}")
                        continue
                
                await asyncio.sleep(0.1)  # 高频更新
                
            except Exception as e:
                logger.error(f"❌ WebSocket数据收集失败: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_data(self) -> None:
        """清理旧数据"""
        while self.collecting:
            try:
                # 每小时清理一次
                await asyncio.sleep(3600)
                
                # 清理超过7天的交易数据
                cutoff_time = datetime.now() - timedelta(days=7)
                
                conn = sqlite3.connect(self.storage.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM trades 
                    WHERE timestamp < ?
                """, (int(cutoff_time.timestamp()),))
                
                cursor.execute("""
                    DELETE FROM orderbooks 
                    WHERE timestamp < ?
                """, (int(cutoff_time.timestamp()),))
                
                conn.commit()
                conn.close()
                
                logger.info("🧹 旧数据清理完成")
                
            except Exception as e:
                logger.error(f"❌ 数据清理失败: {e}")
    
    async def _notify_subscribers(self, data_type: str, data: Any) -> None:
        """通知订阅者"""
        for callback in self.subscribers[data_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"❌ 通知订阅者失败: {e}")
    
    def get_latest_ticker(self, exchange: str, symbol: str) -> Optional[MarketTicker]:
        """获取最新行情"""
        cache_key = f"{exchange}:{symbol}"
        return self.ticker_cache.get(cache_key)
    
    def get_recent_trades(self, exchange: str, symbol: str, limit: int = 100) -> List[Trade]:
        """获取最近交易"""
        cache_key = f"{exchange}:{symbol}"
        trades = list(self.recent_trades[cache_key])
        return trades[-limit:] if len(trades) > limit else trades
    
    def get_klines_dataframe(self, symbol: str, exchange: str, timeframe: str,
                           start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """获取K线数据DataFrame"""
        klines = self.storage.get_klines(symbol, exchange, timeframe, start_time, end_time)
        
        if not klines:
            return pd.DataFrame()
        
        data = []
        for kline in klines:
            data.append({
                'timestamp': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df.empty:
            return df
        
        # 移动平均线
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场概况"""
        summary = {
            'total_symbols': len(self.ticker_cache),
            'exchanges': list(self.exchanges.keys()),
            'stats': self.stats.copy(),
            'top_gainers': [],
            'top_losers': [],
            'highest_volume': []
        }
        
        # 分析涨跌幅
        tickers = list(self.ticker_cache.values())
        if tickers:
            # 涨幅榜
            gainers = sorted(tickers, key=lambda x: x.change_24h_pct, reverse=True)[:10]
            summary['top_gainers'] = [
                {'symbol': t.symbol, 'change': t.change_24h_pct} for t in gainers
            ]
            
            # 跌幅榜
            losers = sorted(tickers, key=lambda x: x.change_24h_pct)[:10]
            summary['top_losers'] = [
                {'symbol': t.symbol, 'change': t.change_24h_pct} for t in losers
            ]
            
            # 成交量榜
            volume_leaders = sorted(tickers, key=lambda x: x.volume, reverse=True)[:10]
            summary['highest_volume'] = [
                {'symbol': t.symbol, 'volume': t.volume} for t in volume_leaders
            ]
        
        return summary
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'collection_status': self.collecting,
            'active_tasks': len(self.collection_tasks),
            'cache_status': {
                'tickers': len(self.ticker_cache),
                'recent_trades': sum(len(trades) for trades in self.recent_trades.values())
            },
            'stats': self.stats.copy(),
            'data_freshness': {}
        }
        
        # 检查数据新鲜度
        now = datetime.now()
        for key, ticker in self.ticker_cache.items():
            age = (now - ticker.timestamp).total_seconds()
            if key not in report['data_freshness']:
                report['data_freshness'][key] = age
        
        return report


# 全局市场数据管理器实例
market_data_manager = None

def get_market_data_manager() -> MarketDataManager:
    """获取市场数据管理器实例"""
    global market_data_manager
    if market_data_manager is None:
        market_data_manager = MarketDataManager()
    return market_data_manager


if __name__ == "__main__":
    async def test_market_data():
        # 测试市场数据管理器
        manager = MarketDataManager()
        
        # 添加交易所（需要实际的API配置）
        # exchange = ccxt.binance({'enableRateLimit': True})
        # manager.add_exchange('binance', exchange)
        
        # 订阅数据更新
        def on_ticker_update(ticker):
            print(f"行情更新: {ticker.symbol} @ {ticker.close}")
        
        manager.subscribe('ticker', on_ticker_update)
        
        # 开始数据收集
        # await manager.start_collection(['BTC/USDT', 'ETH/USDT'])
        
        # 等待一段时间
        # await asyncio.sleep(60)
        
        # 停止收集
        # await manager.stop_collection()
        
        # 生成报告
        summary = manager.get_market_summary()
        print(f"市场概况: {summary}")
    
    # asyncio.run(test_market_data())

