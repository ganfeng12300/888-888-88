#!/usr/bin/env python3
"""
🟡 888-888-88 Binance数据收集器
专门针对Binance交易所的高性能数据收集模块
"""

import asyncio
import websockets
import json
import time
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import ccxt.async_support as ccxt
from loguru import logger
from dataclasses import dataclass
import numpy as np


@dataclass
class BinanceConfig:
    """Binance配置"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rate_limit: bool = True
    timeout: int = 30000
    
    # WebSocket配置
    ws_base_url: str = "wss://stream.binance.com:9443/ws/"
    ws_testnet_url: str = "wss://testnet.binance.vision/ws/"
    
    # REST API配置
    rest_base_url: str = "https://api.binance.com"
    rest_testnet_url: str = "https://testnet.binance.vision"


class BinanceCollector:
    """Binance专用数据收集器"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.exchange = None
        self.ws_connections: Dict[str, Any] = {}
        self.collecting = False
        self.collection_tasks: List[asyncio.Task] = []
        
        # 数据回调
        self.callbacks: Dict[str, List[Callable]] = {
            'ticker': [],
            'kline': [],
            'trade': [],
            'orderbook': [],
            'depth': []
        }
        
        # 统计信息
        self.stats = {
            'messages_received': 0,
            'reconnections': 0,
            'last_message_time': None,
            'connection_status': {}
        }
        
        logger.info("🟡 Binance数据收集器初始化完成")
    
    async def initialize(self) -> bool:
        """初始化交易所连接"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'sandbox': self.config.testnet,
                'enableRateLimit': self.config.rate_limit,
                'timeout': self.config.timeout,
                'options': {
                    'defaultType': 'spot'  # spot, future, delivery
                }
            })
            
            # 测试连接
            await self.exchange.load_markets()
            logger.info("✅ Binance连接初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Binance连接初始化失败: {e}")
            return False
    
    def add_callback(self, data_type: str, callback: Callable) -> None:
        """添加数据回调"""
        if data_type in self.callbacks:
            self.callbacks[data_type].append(callback)
            logger.info(f"📡 添加{data_type}回调")
    
    async def start_collection(self, symbols: List[str]) -> None:
        """开始数据收集"""
        if self.collecting:
            logger.warning("⚠️ Binance数据收集已在运行")
            return
        
        if not self.exchange:
            if not await self.initialize():
                return
        
        self.collecting = True
        logger.info(f"🚀 开始Binance数据收集: {symbols}")
        
        # 启动WebSocket连接
        for symbol in symbols:
            # 行情数据流
            task = asyncio.create_task(
                self._start_ticker_stream(symbol)
            )
            self.collection_tasks.append(task)
            
            # K线数据流
            for interval in ['1m', '5m', '1h']:
                task = asyncio.create_task(
                    self._start_kline_stream(symbol, interval)
                )
                self.collection_tasks.append(task)
            
            # 交易数据流
            task = asyncio.create_task(
                self._start_trade_stream(symbol)
            )
            self.collection_tasks.append(task)
            
            # 深度数据流
            task = asyncio.create_task(
                self._start_depth_stream(symbol)
            )
            self.collection_tasks.append(task)
        
        # 启动连接监控
        monitor_task = asyncio.create_task(self._monitor_connections())
        self.collection_tasks.append(monitor_task)
        
        logger.info("✅ Binance数据收集任务已启动")
    
    async def stop_collection(self) -> None:
        """停止数据收集"""
        self.collecting = False
        
        # 关闭WebSocket连接
        for ws in self.ws_connections.values():
            if ws and not ws.closed:
                await ws.close()
        
        # 取消所有任务
        for task in self.collection_tasks:
            task.cancel()
        
        if self.collection_tasks:
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        self.collection_tasks.clear()
        self.ws_connections.clear()
        
        if self.exchange:
            await self.exchange.close()
        
        logger.info("⏹️ Binance数据收集已停止")
    
    async def _start_ticker_stream(self, symbol: str) -> None:
        """启动行情数据流"""
        stream_name = f"{symbol.lower().replace('/', '')}@ticker"
        ws_url = self._get_ws_url() + stream_name
        
        while self.collecting:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[f"ticker_{symbol}"] = websocket
                    self.stats['connection_status'][f"ticker_{symbol}"] = 'connected'
                    
                    logger.info(f"📡 {symbol} 行情流已连接")
                    
                    async for message in websocket:
                        if not self.collecting:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_ticker_data(symbol, data)
                            
                            self.stats['messages_received'] += 1
                            self.stats['last_message_time'] = datetime.now()
                            
                        except Exception as e:
                            logger.error(f"❌ 处理{symbol}行情数据失败: {e}")
            
            except Exception as e:
                logger.error(f"❌ {symbol}行情流连接失败: {e}")
                self.stats['reconnections'] += 1
                self.stats['connection_status'][f"ticker_{symbol}"] = 'disconnected'
                
                if self.collecting:
                    await asyncio.sleep(5)  # 重连延迟
    
    async def _start_kline_stream(self, symbol: str, interval: str) -> None:
        """启动K线数据流"""
        stream_name = f"{symbol.lower().replace('/', '')}@kline_{interval}"
        ws_url = self._get_ws_url() + stream_name
        
        while self.collecting:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[f"kline_{symbol}_{interval}"] = websocket
                    self.stats['connection_status'][f"kline_{symbol}_{interval}"] = 'connected'
                    
                    logger.info(f"📊 {symbol} {interval} K线流已连接")
                    
                    async for message in websocket:
                        if not self.collecting:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_kline_data(symbol, interval, data)
                            
                            self.stats['messages_received'] += 1
                            self.stats['last_message_time'] = datetime.now()
                            
                        except Exception as e:
                            logger.error(f"❌ 处理{symbol} K线数据失败: {e}")
            
            except Exception as e:
                logger.error(f"❌ {symbol} K线流连接失败: {e}")
                self.stats['reconnections'] += 1
                self.stats['connection_status'][f"kline_{symbol}_{interval}"] = 'disconnected'
                
                if self.collecting:
                    await asyncio.sleep(5)
    
    async def _start_trade_stream(self, symbol: str) -> None:
        """启动交易数据流"""
        stream_name = f"{symbol.lower().replace('/', '')}@trade"
        ws_url = self._get_ws_url() + stream_name
        
        while self.collecting:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[f"trade_{symbol}"] = websocket
                    self.stats['connection_status'][f"trade_{symbol}"] = 'connected'
                    
                    logger.info(f"💱 {symbol} 交易流已连接")
                    
                    async for message in websocket:
                        if not self.collecting:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_trade_data(symbol, data)
                            
                            self.stats['messages_received'] += 1
                            self.stats['last_message_time'] = datetime.now()
                            
                        except Exception as e:
                            logger.error(f"❌ 处理{symbol}交易数据失败: {e}")
            
            except Exception as e:
                logger.error(f"❌ {symbol}交易流连接失败: {e}")
                self.stats['reconnections'] += 1
                self.stats['connection_status'][f"trade_{symbol}"] = 'disconnected'
                
                if self.collecting:
                    await asyncio.sleep(5)
    
    async def _start_depth_stream(self, symbol: str, levels: int = 20) -> None:
        """启动深度数据流"""
        stream_name = f"{symbol.lower().replace('/', '')}@depth{levels}@100ms"
        ws_url = self._get_ws_url() + stream_name
        
        while self.collecting:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[f"depth_{symbol}"] = websocket
                    self.stats['connection_status'][f"depth_{symbol}"] = 'connected'
                    
                    logger.info(f"📈 {symbol} 深度流已连接")
                    
                    async for message in websocket:
                        if not self.collecting:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._process_depth_data(symbol, data)
                            
                            self.stats['messages_received'] += 1
                            self.stats['last_message_time'] = datetime.now()
                            
                        except Exception as e:
                            logger.error(f"❌ 处理{symbol}深度数据失败: {e}")
            
            except Exception as e:
                logger.error(f"❌ {symbol}深度流连接失败: {e}")
                self.stats['reconnections'] += 1
                self.stats['connection_status'][f"depth_{symbol}"] = 'disconnected'
                
                if self.collecting:
                    await asyncio.sleep(5)
    
    async def _process_ticker_data(self, symbol: str, data: Dict) -> None:
        """处理行情数据"""
        try:
            ticker_info = {
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': datetime.fromtimestamp(data['E'] / 1000),
                'open': float(data['o']),
                'high': float(data['h']),
                'low': float(data['l']),
                'close': float(data['c']),
                'volume': float(data['v']),
                'bid': float(data['b']),
                'ask': float(data['a']),
                'bid_volume': float(data['B']),
                'ask_volume': float(data['A']),
                'change_24h': float(data['P']),
                'change_24h_pct': float(data['p'])
            }
            
            # 调用回调函数
            for callback in self.callbacks['ticker']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(ticker_info)
                    else:
                        callback(ticker_info)
                except Exception as e:
                    logger.error(f"❌ 行情回调失败: {e}")
        
        except Exception as e:
            logger.error(f"❌ 处理行情数据失败: {e}")
    
    async def _process_kline_data(self, symbol: str, interval: str, data: Dict) -> None:
        """处理K线数据"""
        try:
            kline_data = data['k']
            
            kline_info = {
                'symbol': symbol,
                'exchange': 'binance',
                'timeframe': interval,
                'timestamp': datetime.fromtimestamp(kline_data['t'] / 1000),
                'open': float(kline_data['o']),
                'high': float(kline_data['h']),
                'low': float(kline_data['l']),
                'close': float(kline_data['c']),
                'volume': float(kline_data['v']),
                'trades_count': int(kline_data['n']),
                'taker_buy_volume': float(kline_data['V']),
                'is_closed': kline_data['x']  # K线是否完结
            }
            
            # 只处理完结的K线
            if kline_info['is_closed']:
                for callback in self.callbacks['kline']:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(kline_info)
                        else:
                            callback(kline_info)
                    except Exception as e:
                        logger.error(f"❌ K线回调失败: {e}")
        
        except Exception as e:
            logger.error(f"❌ 处理K线数据失败: {e}")
    
    async def _process_trade_data(self, symbol: str, data: Dict) -> None:
        """处理交易数据"""
        try:
            trade_info = {
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                'price': float(data['p']),
                'size': float(data['q']),
                'side': 'buy' if data['m'] else 'sell',  # m=true表示买方是maker
                'trade_id': str(data['t'])
            }
            
            for callback in self.callbacks['trade']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trade_info)
                    else:
                        callback(trade_info)
                except Exception as e:
                    logger.error(f"❌ 交易回调失败: {e}")
        
        except Exception as e:
            logger.error(f"❌ 处理交易数据失败: {e}")
    
    async def _process_depth_data(self, symbol: str, data: Dict) -> None:
        """处理深度数据"""
        try:
            depth_info = {
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': datetime.now(),
                'bids': [(float(bid[0]), float(bid[1])) for bid in data['bids']],
                'asks': [(float(ask[0]), float(ask[1])) for ask in data['asks']],
                'last_update_id': data['lastUpdateId']
            }
            
            for callback in self.callbacks['depth']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(depth_info)
                    else:
                        callback(depth_info)
                except Exception as e:
                    logger.error(f"❌ 深度回调失败: {e}")
        
        except Exception as e:
            logger.error(f"❌ 处理深度数据失败: {e}")
    
    async def _monitor_connections(self) -> None:
        """监控连接状态"""
        while self.collecting:
            try:
                # 检查连接健康状态
                current_time = datetime.now()
                
                # 检查是否有消息超时
                if (self.stats['last_message_time'] and 
                    (current_time - self.stats['last_message_time']).seconds > 60):
                    logger.warning("⚠️ 消息接收超时，可能需要重连")
                
                # 检查连接状态
                disconnected_streams = []
                for stream_name, status in self.stats['connection_status'].items():
                    if status == 'disconnected':
                        disconnected_streams.append(stream_name)
                
                if disconnected_streams:
                    logger.warning(f"⚠️ 断开的流: {disconnected_streams}")
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"❌ 连接监控失败: {e}")
                await asyncio.sleep(30)
    
    def _get_ws_url(self) -> str:
        """获取WebSocket URL"""
        if self.config.testnet:
            return self.config.ws_testnet_url
        return self.config.ws_base_url
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                  start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取历史K线数据"""
        try:
            if not self.exchange:
                await self.initialize()
            
            # 转换时间格式
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)
            
            # 获取数据
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, interval, since=since, limit=1000
            )
            
            klines = []
            for candle in ohlcv:
                if candle[0] <= until:
                    klines.append({
                        'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
            
            return klines
            
        except Exception as e:
            logger.error(f"❌ 获取历史K线失败: {e}")
            return []
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """获取订单簿"""
        try:
            if not self.exchange:
                await self.initialize()
            
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': datetime.now(),
                'bids': orderbook['bids'],
                'asks': orderbook['asks']
            }
            
        except Exception as e:
            logger.error(f"❌ 获取订单簿失败: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'collecting': self.collecting,
            'active_connections': len([s for s in self.stats['connection_status'].values() 
                                     if s == 'connected']),
            'total_connections': len(self.stats['connection_status']),
            'messages_received': self.stats['messages_received'],
            'reconnections': self.stats['reconnections'],
            'last_message_time': self.stats['last_message_time'].isoformat() 
                               if self.stats['last_message_time'] else None,
            'connection_details': self.stats['connection_status'].copy()
        }


if __name__ == "__main__":
    async def test_binance_collector():
        # 测试Binance收集器
        config = BinanceConfig(testnet=True)  # 使用测试网
        collector = BinanceCollector(config)
        
        # 添加回调
        def on_ticker(data):
            print(f"行情: {data['symbol']} @ {data['close']}")
        
        def on_trade(data):
            print(f"交易: {data['symbol']} {data['side']} {data['size']} @ {data['price']}")
        
        collector.add_callback('ticker', on_ticker)
        collector.add_callback('trade', on_trade)
        
        # 开始收集
        await collector.start_collection(['BTC/USDT'])
        
        # 运行一段时间
        await asyncio.sleep(60)
        
        # 停止收集
        await collector.stop_collection()
        
        # 显示统计
        stats = collector.get_stats()
        print(f"统计信息: {stats}")
    
    # asyncio.run(test_binance_collector())

