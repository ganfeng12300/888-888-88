"""
🔗 交易所连接器
生产级多交易所统一接口，支持实时数据采集和交易执行
支持Binance、OKX、Bybit等主流交易所
"""

import asyncio
import time
import json
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import aiohttp
import ccxt.async_support as ccxt
from urllib.parse import urlencode

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores


class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    HUOBI = "huobi"
    KUCOIN = "kucoin"


class DataType(Enum):
    """数据类型"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINE = "kline"
    ACCOUNT = "account"
    ORDER = "order"
    POSITION = "position"


@dataclass
class MarketData:
    """市场数据结构"""
    exchange: str
    symbol: str
    data_type: DataType
    timestamp: float
    data: Dict[str, Any]
    latency_ms: float = 0.0


@dataclass
class ExchangeConfig:
    """交易所配置"""
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox: bool = False
    rate_limit: int = 1200
    timeout: int = 30
    enable_rate_limit: bool = True


class ExchangeConnector:
    """交易所连接器基类"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange_type = config.exchange_type
        self.is_connected = False
        self.websocket_connections: Dict[str, Any] = {}
        self.data_callbacks: Dict[DataType, List[Callable]] = {}
        self.last_heartbeat = time.time()
        
        # 初始化CCXT交易所实例
        self._init_ccxt_exchange()
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.DATA_COLLECTION, [1, 2, 3, 4])
        
        logger.info(f"初始化交易所连接器: {self.exchange_type.value}")
    
    def _init_ccxt_exchange(self):
        """初始化CCXT交易所实例"""
        try:
            exchange_class = getattr(ccxt, self.exchange_type.value)
            self.ccxt_exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.passphrase,
                'sandbox': self.config.sandbox,
                'rateLimit': self.config.rate_limit,
                'timeout': self.config.timeout * 1000,
                'enableRateLimit': self.config.enable_rate_limit,
                'options': {
                    'defaultType': 'spot',  # spot, future, option
                    'adjustForTimeDifference': True,
                }
            })
            
            # 设置代理（如果需要）
            if hasattr(self.ccxt_exchange, 'proxies'):
                self.ccxt_exchange.proxies = {
                    'http': None,
                    'https': None,
                }
            
        except Exception as e:
            logger.error(f"初始化CCXT交易所失败: {e}")
            raise
    
    async def connect(self) -> bool:
        """连接到交易所"""
        try:
            # 测试API连接
            await self.ccxt_exchange.load_markets()
            
            # 获取服务器时间
            server_time = await self.ccxt_exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            if time_diff > 5000:  # 5秒时间差
                logger.warning(f"服务器时间差异: {time_diff}ms")
            
            self.is_connected = True
            self.last_heartbeat = time.time()
            
            logger.info(f"成功连接到 {self.exchange_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"连接交易所失败: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        try:
            # 关闭WebSocket连接
            for ws_conn in self.websocket_connections.values():
                if ws_conn and not ws_conn.closed:
                    await ws_conn.close()
            
            self.websocket_connections.clear()
            
            # 关闭CCXT连接
            if self.ccxt_exchange:
                await self.ccxt_exchange.close()
            
            self.is_connected = False
            logger.info(f"已断开 {self.exchange_type.value} 连接")
            
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
    
    def register_callback(self, data_type: DataType, callback: Callable):
        """注册数据回调函数"""
        if data_type not in self.data_callbacks:
            self.data_callbacks[data_type] = []
        
        self.data_callbacks[data_type].append(callback)
        logger.info(f"注册 {data_type.value} 数据回调")
    
    async def _notify_callbacks(self, market_data: MarketData):
        """通知回调函数"""
        callbacks = self.data_callbacks.get(market_data.data_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(market_data)
                else:
                    callback(market_data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取ticker数据"""
        try:
            start_time = time.time()
            ticker = await self.ccxt_exchange.fetch_ticker(symbol)
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol=symbol,
                data_type=DataType.TICKER,
                timestamp=time.time(),
                data=ticker,
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            return ticker
            
        except Exception as e:
            logger.error(f"获取ticker失败 {symbol}: {e}")
            return None
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """获取订单簿数据"""
        try:
            start_time = time.time()
            orderbook = await self.ccxt_exchange.fetch_order_book(symbol, limit)
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol=symbol,
                data_type=DataType.ORDERBOOK,
                timestamp=time.time(),
                data=orderbook,
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            return orderbook
            
        except Exception as e:
            logger.error(f"获取订单簿失败 {symbol}: {e}")
            return None
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """获取最近交易数据"""
        try:
            start_time = time.time()
            trades = await self.ccxt_exchange.fetch_trades(symbol, limit=limit)
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol=symbol,
                data_type=DataType.TRADES,
                timestamp=time.time(),
                data=trades,
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            return trades
            
        except Exception as e:
            logger.error(f"获取交易数据失败 {symbol}: {e}")
            return None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                         limit: int = 100) -> Optional[List[List[float]]]:
        """获取K线数据"""
        try:
            start_time = time.time()
            ohlcv = await self.ccxt_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol=symbol,
                data_type=DataType.KLINE,
                timestamp=time.time(),
                data={
                    'timeframe': timeframe,
                    'ohlcv': ohlcv
                },
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            return ohlcv
            
        except Exception as e:
            logger.error(f"获取K线数据失败 {symbol}: {e}")
            return None
    
    async def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """获取账户余额"""
        try:
            start_time = time.time()
            balance = await self.ccxt_exchange.fetch_balance()
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol="",
                data_type=DataType.ACCOUNT,
                timestamp=time.time(),
                data=balance,
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            return balance
            
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            return None
    
    async def create_order(self, symbol: str, order_type: str, side: str,
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """创建订单"""
        try:
            start_time = time.time()
            
            order = await self.ccxt_exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            
            latency = (time.time() - start_time) * 1000
            
            market_data = MarketData(
                exchange=self.exchange_type.value,
                symbol=symbol,
                data_type=DataType.ORDER,
                timestamp=time.time(),
                data=order,
                latency_ms=latency
            )
            
            await self._notify_callbacks(market_data)
            
            logger.info(f"订单创建成功: {order['id']} {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """取消订单"""
        try:
            start_time = time.time()
            result = await self.ccxt_exchange.cancel_order(order_id, symbol)
            latency = (time.time() - start_time) * 1000
            
            logger.info(f"订单取消成功: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return None
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        try:
            order = await self.ccxt_exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            logger.error(f"获取订单状态失败: {e}")
            return None
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """获取未完成订单"""
        try:
            orders = await self.ccxt_exchange.fetch_open_orders(symbol)
            return orders
            
        except Exception as e:
            logger.error(f"获取未完成订单失败: {e}")
            return None
    
    async def start_websocket_streams(self, symbols: List[str], 
                                    data_types: List[DataType]):
        """启动WebSocket数据流"""
        try:
            # 根据交易所类型启动相应的WebSocket连接
            if self.exchange_type == ExchangeType.BINANCE:
                await self._start_binance_websocket(symbols, data_types)
            elif self.exchange_type == ExchangeType.OKX:
                await self._start_okx_websocket(symbols, data_types)
            elif self.exchange_type == ExchangeType.BYBIT:
                await self._start_bybit_websocket(symbols, data_types)
            
            logger.info(f"WebSocket数据流已启动: {symbols}")
            
        except Exception as e:
            logger.error(f"启动WebSocket失败: {e}")
    
    async def _start_binance_websocket(self, symbols: List[str], data_types: List[DataType]):
        """启动Binance WebSocket连接"""
        base_url = "wss://stream.binance.com:9443/ws/"
        
        # 构建订阅流
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.lower().replace('/', '')
            
            for data_type in data_types:
                if data_type == DataType.TICKER:
                    streams.append(f"{symbol_lower}@ticker")
                elif data_type == DataType.ORDERBOOK:
                    streams.append(f"{symbol_lower}@depth20@100ms")
                elif data_type == DataType.TRADES:
                    streams.append(f"{symbol_lower}@trade")
                elif data_type == DataType.KLINE:
                    streams.append(f"{symbol_lower}@kline_1m")
        
        if not streams:
            return
        
        ws_url = base_url + "/".join(streams)
        
        try:
            websocket = await websockets.connect(ws_url)
            self.websocket_connections['market_data'] = websocket
            
            # 启动消息处理任务
            asyncio.create_task(self._handle_binance_messages(websocket))
            
        except Exception as e:
            logger.error(f"Binance WebSocket连接失败: {e}")
    
    async def _handle_binance_messages(self, websocket):
        """处理Binance WebSocket消息"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_binance_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}")
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket连接已关闭")
        except Exception as e:
            logger.error(f"WebSocket消息处理失败: {e}")
    
    async def _process_binance_message(self, data: Dict[str, Any]):
        """处理Binance消息"""
        try:
            stream = data.get('stream', '')
            event_data = data.get('data', {})
            
            if '@ticker' in stream:
                symbol = event_data.get('s', '').replace('USDT', '/USDT')
                market_data = MarketData(
                    exchange=self.exchange_type.value,
                    symbol=symbol,
                    data_type=DataType.TICKER,
                    timestamp=time.time(),
                    data=event_data
                )
                await self._notify_callbacks(market_data)
                
            elif '@depth' in stream:
                symbol = event_data.get('s', '').replace('USDT', '/USDT')
                market_data = MarketData(
                    exchange=self.exchange_type.value,
                    symbol=symbol,
                    data_type=DataType.ORDERBOOK,
                    timestamp=time.time(),
                    data=event_data
                )
                await self._notify_callbacks(market_data)
                
            elif '@trade' in stream:
                symbol = event_data.get('s', '').replace('USDT', '/USDT')
                market_data = MarketData(
                    exchange=self.exchange_type.value,
                    symbol=symbol,
                    data_type=DataType.TRADES,
                    timestamp=time.time(),
                    data=event_data
                )
                await self._notify_callbacks(market_data)
                
            elif '@kline' in stream:
                symbol = event_data.get('s', '').replace('USDT', '/USDT')
                kline_data = event_data.get('k', {})
                market_data = MarketData(
                    exchange=self.exchange_type.value,
                    symbol=symbol,
                    data_type=DataType.KLINE,
                    timestamp=time.time(),
                    data=kline_data
                )
                await self._notify_callbacks(market_data)
                
        except Exception as e:
            logger.error(f"处理Binance消息失败: {e}")
    
    async def _start_okx_websocket(self, symbols: List[str], data_types: List[DataType]):
        """启动OKX WebSocket连接"""
        # OKX WebSocket实现
        pass
    
    async def _start_bybit_websocket(self, symbols: List[str], data_types: List[DataType]):
        """启动Bybit WebSocket连接"""
        # Bybit WebSocket实现
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查API连接
            server_time = await self.ccxt_exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            # 检查WebSocket连接
            ws_status = {}
            for name, ws in self.websocket_connections.items():
                ws_status[name] = not ws.closed if ws else False
            
            return {
                'exchange': self.exchange_type.value,
                'api_connected': self.is_connected,
                'time_diff_ms': time_diff,
                'websocket_status': ws_status,
                'last_heartbeat': self.last_heartbeat,
                'callbacks_registered': {
                    dt.value: len(callbacks) 
                    for dt, callbacks in self.data_callbacks.items()
                }
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'exchange': self.exchange_type.value,
                'api_connected': False,
                'error': str(e)
            }


class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self):
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.is_running = False
        
    def add_exchange(self, name: str, config: ExchangeConfig) -> ExchangeConnector:
        """添加交易所连接器"""
        connector = ExchangeConnector(config)
        self.connectors[name] = connector
        
        logger.info(f"添加交易所连接器: {name}")
        return connector
    
    async def connect_all(self) -> Dict[str, bool]:
        """连接所有交易所"""
        results = {}
        
        for name, connector in self.connectors.items():
            try:
                result = await connector.connect()
                results[name] = result
            except Exception as e:
                logger.error(f"连接 {name} 失败: {e}")
                results[name] = False
        
        return results
    
    async def disconnect_all(self):
        """断开所有连接"""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.error(f"断开 {name} 连接失败: {e}")
    
    def get_connector(self, name: str) -> Optional[ExchangeConnector]:
        """获取交易所连接器"""
        return self.connectors.get(name)
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """所有交易所健康检查"""
        results = {}
        
        for name, connector in self.connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results


# 全局交易所管理器实例
exchange_manager = ExchangeManager()


async def main():
    """测试主函数"""
    logger.info("启动交易所连接器测试...")
    
    # 配置测试交易所（使用测试网）
    binance_config = ExchangeConfig(
        exchange_type=ExchangeType.BINANCE,
        api_key="test_api_key",
        api_secret="test_api_secret",
        sandbox=True
    )
    
    # 添加交易所
    binance_connector = exchange_manager.add_exchange("binance", binance_config)
    
    try:
        # 连接交易所
        results = await exchange_manager.connect_all()
        logger.info(f"连接结果: {results}")
        
        if results.get("binance"):
            # 注册数据回调
            async def ticker_callback(market_data: MarketData):
                logger.info(f"收到ticker数据: {market_data.symbol} - {market_data.data.get('last', 0)}")
            
            binance_connector.register_callback(DataType.TICKER, ticker_callback)
            
            # 获取市场数据
            ticker = await binance_connector.fetch_ticker('BTC/USDT')
            logger.info(f"BTC/USDT ticker: {ticker}")
            
            # 启动WebSocket数据流
            await binance_connector.start_websocket_streams(
                ['BTC/USDT', 'ETH/USDT'], 
                [DataType.TICKER, DataType.TRADES]
            )
            
            # 运行30秒
            await asyncio.sleep(30)
        
        # 健康检查
        health = await exchange_manager.health_check_all()
        logger.info(f"健康检查: {health}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        await exchange_manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
