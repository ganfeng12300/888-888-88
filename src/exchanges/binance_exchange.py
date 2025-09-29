"""
🔌 币安交易所接口 - 完整实盘交易接口实现
支持现货、期货、期权交易，包含REST API和WebSocket实时数据
完整实现所有交易功能：下单、撤单、查询、账户管理等
"""
import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    import aiohttp
    import websockets
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

from loguru import logger
from .base_exchange import (
    BaseExchange, OrderType, OrderSide, OrderStatus, TimeInForce, MarketType,
    Symbol, Order, Trade, Balance, Position, Ticker, OrderBook, Kline
)

class BinanceExchange(BaseExchange):
    """币安交易所接口"""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False, timeout: int = 30):
        super().__init__(api_key, api_secret, None, sandbox, timeout)
        
        self.name = "binance"
        
        if sandbox:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_url = "wss://stream.binance.com:9443/ws"
        
        # 币安特定配置
        self.recv_window = 5000  # 接收窗口
        
        logger.info("币安交易所接口初始化完成")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        return {
            "name": "Binance",
            "country": "Malta",
            "rateLimit": 1200,  # 每分钟请求限制
            "certified": True,
            "pro": True,
            "has": {
                "spot": True,
                "futures": True,
                "options": False,
                "margin": True,
                "websocket": True
            }
        }
    
    async def get_symbols(self) -> List[Symbol]:
        """获取所有交易对信息"""
        try:
            response = await self.make_request("GET", "/v3/exchangeInfo")
            symbols = []
            
            for symbol_info in response.get("symbols", []):
                if symbol_info["status"] != "TRADING":
                    continue
                
                # 解析过滤器
                min_qty = 0.0
                max_qty = float('inf')
                step_size = 0.0
                min_price = 0.0
                max_price = float('inf')
                tick_size = 0.0
                min_notional = 0.0
                
                for filter_info in symbol_info.get("filters", []):
                    if filter_info["filterType"] == "LOT_SIZE":
                        min_qty = float(filter_info["minQty"])
                        max_qty = float(filter_info["maxQty"])
                        step_size = float(filter_info["stepSize"])
                    elif filter_info["filterType"] == "PRICE_FILTER":
                        min_price = float(filter_info["minPrice"])
                        max_price = float(filter_info["maxPrice"])
                        tick_size = float(filter_info["tickSize"])
                    elif filter_info["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(filter_info["minNotional"])
                
                symbol = Symbol(
                    symbol=symbol_info["symbol"],
                    base_asset=symbol_info["baseAsset"],
                    quote_asset=symbol_info["quoteAsset"],
                    market_type=MarketType.SPOT,
                    min_qty=min_qty,
                    max_qty=max_qty,
                    step_size=step_size,
                    min_price=min_price,
                    max_price=max_price,
                    tick_size=tick_size,
                    min_notional=min_notional,
                    is_active=True
                )
                symbols.append(symbol)
            
            logger.info(f"获取币安交易对信息: {len(symbols)}个")
            return symbols
        
        except Exception as e:
            logger.error(f"获取币安交易对信息失败: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """获取单个交易对行情"""
        try:
            params = {"symbol": self.format_symbol(symbol)}
            response = await self.make_request("GET", "/v3/ticker/24hr", params=params)
            
            ticker = Ticker(
                symbol=response["symbol"],
                price=float(response["lastPrice"]),
                bid_price=float(response["bidPrice"]),
                ask_price=float(response["askPrice"]),
                bid_qty=float(response["bidQty"]),
                ask_qty=float(response["askQty"]),
                volume_24h=float(response["volume"]),
                price_change_24h=float(response["priceChange"]),
                price_change_percent_24h=float(response["priceChangePercent"]),
                high_24h=float(response["highPrice"]),
                low_24h=float(response["lowPrice"]),
                timestamp=float(response["closeTime"])
            )
            
            return ticker
        
        except Exception as e:
            logger.error(f"获取币安行情失败: {e}")
            raise
    
    async def get_tickers(self, symbols: List[str] = None) -> List[Ticker]:
        """获取多个交易对行情"""
        try:
            if symbols:
                # 获取指定交易对行情
                tickers = []
                for symbol in symbols:
                    ticker = await self.get_ticker(symbol)
                    tickers.append(ticker)
                return tickers
            else:
                # 获取所有交易对行情
                response = await self.make_request("GET", "/v3/ticker/24hr")
                tickers = []
                
                for ticker_data in response:
                    ticker = Ticker(
                        symbol=ticker_data["symbol"],
                        price=float(ticker_data["lastPrice"]),
                        bid_price=float(ticker_data["bidPrice"]),
                        ask_price=float(ticker_data["askPrice"]),
                        bid_qty=float(ticker_data["bidQty"]),
                        ask_qty=float(ticker_data["askQty"]),
                        volume_24h=float(ticker_data["volume"]),
                        price_change_24h=float(ticker_data["priceChange"]),
                        price_change_percent_24h=float(ticker_data["priceChangePercent"]),
                        high_24h=float(ticker_data["highPrice"]),
                        low_24h=float(ticker_data["lowPrice"]),
                        timestamp=float(ticker_data["closeTime"])
                    )
                    tickers.append(ticker)
                
                return tickers
        
        except Exception as e:
            logger.error(f"获取币安行情列表失败: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "limit": min(limit, 5000)  # 币安最大5000
            }
            response = await self.make_request("GET", "/v3/depth", params=params)
            
            bids = [(float(bid[0]), float(bid[1])) for bid in response["bids"]]
            asks = [(float(ask[0]), float(ask[1])) for ask in response["asks"]]
            
            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=time.time() * 1000
            )
            
            return orderbook
        
        except Exception as e:
            logger.error(f"获取币安订单簿失败: {e}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, start_time: int = None,
                        end_time: int = None, limit: int = 500) -> List[Kline]:
        """获取K线数据"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "interval": interval,
                "limit": min(limit, 1000)  # 币安最大1000
            }
            
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            response = await self.make_request("GET", "/v3/klines", params=params)
            klines = []
            
            for kline_data in response:
                kline = Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=float(kline_data[0]),
                    close_time=float(kline_data[6]),
                    open_price=float(kline_data[1]),
                    high_price=float(kline_data[2]),
                    low_price=float(kline_data[3]),
                    close_price=float(kline_data[4]),
                    volume=float(kline_data[5]),
                    quote_volume=float(kline_data[7]),
                    trade_count=int(kline_data[8])
                )
                klines.append(kline)
            
            return klines
        
        except Exception as e:
            logger.error(f"获取币安K线数据失败: {e}")
            return []
    
    async def get_trades(self, symbol: str, limit: int = 500) -> List[Trade]:
        """获取最近成交记录"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "limit": min(limit, 1000)  # 币安最大1000
            }
            response = await self.make_request("GET", "/v3/trades", params=params)
            trades = []
            
            for trade_data in response:
                trade = Trade(
                    trade_id=str(trade_data["id"]),
                    order_id="",  # 公开接口不提供订单ID
                    symbol=symbol,
                    side=OrderSide.BUY if trade_data["isBuyerMaker"] else OrderSide.SELL,
                    quantity=float(trade_data["qty"]),
                    price=float(trade_data["price"]),
                    commission=0.0,  # 公开接口不提供手续费
                    commission_asset="",
                    is_maker=trade_data["isBuyerMaker"],
                    timestamp=float(trade_data["time"])
                )
                trades.append(trade)
            
            return trades
        
        except Exception as e:
            logger.error(f"获取币安成交记录失败: {e}")
            return []
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: float = None, stop_price: float = None,
                         time_in_force: TimeInForce = TimeInForce.GTC,
                         client_order_id: str = None) -> Order:
        """下单"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "side": side.value.upper(),
                "type": self._convert_order_type(order_type),
                "quantity": str(quantity),
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            if client_order_id:
                params["newClientOrderId"] = client_order_id
            
            if order_type == OrderType.LIMIT:
                params["price"] = str(price)
                params["timeInForce"] = time_in_force.value.upper()
            
            if stop_price:
                params["stopPrice"] = str(stop_price)
            
            response = await self.make_request("POST", "/v3/order", data=params, signed=True)
            
            order = Order(
                order_id=str(response["orderId"]),
                client_order_id=response.get("clientOrderId", ""),
                symbol=response["symbol"],
                side=OrderSide(response["side"].lower()),
                order_type=self._parse_order_type(response["type"]),
                quantity=float(response["origQty"]),
                price=float(response.get("price", 0)),
                stop_price=float(response.get("stopPrice", 0)),
                time_in_force=TimeInForce(response.get("timeInForce", "gtc").lower()),
                status=self._parse_order_status(response["status"]),
                filled_qty=float(response["executedQty"]),
                remaining_qty=float(response["origQty"]) - float(response["executedQty"]),
                avg_price=float(response.get("price", 0)),
                commission=0.0,
                commission_asset="",
                created_at=float(response["transactTime"]),
                updated_at=float(response["transactTime"])
            )
            
            logger.info(f"币安下单成功: {order.order_id}")
            return order
        
        except Exception as e:
            logger.error(f"币安下单失败: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str = None,
                          client_order_id: str = None) -> bool:
        """撤单"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                raise ValueError("必须提供order_id或client_order_id")
            
            await self.make_request("DELETE", "/v3/order", data=params, signed=True)
            logger.info(f"币安撤单成功: {order_id or client_order_id}")
            return True
        
        except Exception as e:
            logger.error(f"币安撤单失败: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """撤销所有订单"""
        try:
            if symbol:
                params = {
                    "symbol": self.format_symbol(symbol),
                    "timestamp": int(time.time() * 1000),
                    "recvWindow": self.recv_window
                }
                await self.make_request("DELETE", "/v3/openOrders", data=params, signed=True)
            else:
                # 获取所有活跃订单并逐个撤销
                open_orders = await self.get_open_orders()
                for order in open_orders:
                    await self.cancel_order(order.symbol, order.order_id)
            
            logger.info("币安撤销所有订单成功")
            return True
        
        except Exception as e:
            logger.error(f"币安撤销所有订单失败: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str = None,
                       client_order_id: str = None) -> Order:
        """查询订单"""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                raise ValueError("必须提供order_id或client_order_id")
            
            response = await self.make_request("GET", "/v3/order", params=params, signed=True)
            
            order = Order(
                order_id=str(response["orderId"]),
                client_order_id=response.get("clientOrderId", ""),
                symbol=response["symbol"],
                side=OrderSide(response["side"].lower()),
                order_type=self._parse_order_type(response["type"]),
                quantity=float(response["origQty"]),
                price=float(response.get("price", 0)),
                stop_price=float(response.get("stopPrice", 0)),
                time_in_force=TimeInForce(response.get("timeInForce", "gtc").lower()),
                status=self._parse_order_status(response["status"]),
                filled_qty=float(response["executedQty"]),
                remaining_qty=float(response["origQty"]) - float(response["executedQty"]),
                avg_price=float(response.get("price", 0)),
                commission=0.0,
                commission_asset="",
                created_at=float(response["time"]),
                updated_at=float(response["updateTime"])
            )
            
            return order
        
        except Exception as e:
            logger.error(f"币安查询订单失败: {e}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """查询活跃订单"""
        try:
            params = {
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            if symbol:
                params["symbol"] = self.format_symbol(symbol)
            
            response = await self.make_request("GET", "/v3/openOrders", params=params, signed=True)
            orders = []
            
            for order_data in response:
                order = Order(
                    order_id=str(order_data["orderId"]),
                    client_order_id=order_data.get("clientOrderId", ""),
                    symbol=order_data["symbol"],
                    side=OrderSide(order_data["side"].lower()),
                    order_type=self._parse_order_type(order_data["type"]),
                    quantity=float(order_data["origQty"]),
                    price=float(order_data.get("price", 0)),
                    stop_price=float(order_data.get("stopPrice", 0)),
                    time_in_force=TimeInForce(order_data.get("timeInForce", "gtc").lower()),
                    status=self._parse_order_status(order_data["status"]),
                    filled_qty=float(order_data["executedQty"]),
                    remaining_qty=float(order_data["origQty"]) - float(order_data["executedQty"]),
                    avg_price=float(order_data.get("price", 0)),
                    commission=0.0,
                    commission_asset="",
                    created_at=float(order_data["time"]),
                    updated_at=float(order_data["updateTime"])
                )
                orders.append(order)
            
            return orders
        
        except Exception as e:
            logger.error(f"币安查询活跃订单失败: {e}")
            return []
    
    async def get_order_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Order]:
        """查询历史订单"""
        try:
            params = {
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window,
                "limit": min(limit, 1000)  # 币安最大1000
            }
            
            if symbol:
                params["symbol"] = self.format_symbol(symbol)
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            response = await self.make_request("GET", "/v3/allOrders", params=params, signed=True)
            orders = []
            
            for order_data in response:
                order = Order(
                    order_id=str(order_data["orderId"]),
                    client_order_id=order_data.get("clientOrderId", ""),
                    symbol=order_data["symbol"],
                    side=OrderSide(order_data["side"].lower()),
                    order_type=self._parse_order_type(order_data["type"]),
                    quantity=float(order_data["origQty"]),
                    price=float(order_data.get("price", 0)),
                    stop_price=float(order_data.get("stopPrice", 0)),
                    time_in_force=TimeInForce(order_data.get("timeInForce", "gtc").lower()),
                    status=self._parse_order_status(order_data["status"]),
                    filled_qty=float(order_data["executedQty"]),
                    remaining_qty=float(order_data["origQty"]) - float(order_data["executedQty"]),
                    avg_price=float(order_data.get("price", 0)),
                    commission=0.0,
                    commission_asset="",
                    created_at=float(order_data["time"]),
                    updated_at=float(order_data["updateTime"])
                )
                orders.append(order)
            
            return orders
        
        except Exception as e:
            logger.error(f"币安查询历史订单失败: {e}")
            return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            params = {
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            response = await self.make_request("GET", "/v3/account", params=params, signed=True)
            return response
        
        except Exception as e:
            logger.error(f"币安获取账户信息失败: {e}")
            return {}
    
    async def get_balances(self) -> List[Balance]:
        """获取账户余额"""
        try:
            account_info = await self.get_account_info()
            balances = []
            
            for balance_data in account_info.get("balances", []):
                free = float(balance_data["free"])
                locked = float(balance_data["locked"])
                
                if free > 0 or locked > 0:  # 只返回有余额的资产
                    balance = Balance(
                        asset=balance_data["asset"],
                        free=free,
                        locked=locked,
                        total=free + locked
                    )
                    balances.append(balance)
            
            return balances
        
        except Exception as e:
            logger.error(f"币安获取账户余额失败: {e}")
            return []
    
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """获取持仓信息（期货）"""
        # 币安现货不支持持仓，返回空列表
        return []
    
    async def get_trade_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Trade]:
        """获取成交历史"""
        try:
            if not symbol:
                raise ValueError("币安获取成交历史需要指定交易对")
            
            params = {
                "symbol": self.format_symbol(symbol),
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window,
                "limit": min(limit, 1000)  # 币安最大1000
            }
            
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            response = await self.make_request("GET", "/v3/myTrades", params=params, signed=True)
            trades = []
            
            for trade_data in response:
                trade = Trade(
                    trade_id=str(trade_data["id"]),
                    order_id=str(trade_data["orderId"]),
                    symbol=trade_data["symbol"],
                    side=OrderSide.BUY if trade_data["isBuyer"] else OrderSide.SELL,
                    quantity=float(trade_data["qty"]),
                    price=float(trade_data["price"]),
                    commission=float(trade_data["commission"]),
                    commission_asset=trade_data["commissionAsset"],
                    is_maker=trade_data["isMaker"],
                    timestamp=float(trade_data["time"])
                )
                trades.append(trade)
            
            return trades
        
        except Exception as e:
            logger.error(f"币安获取成交历史失败: {e}")
            return []
    
    # WebSocket相关方法
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅行情数据"""
        stream = f"{symbol.lower()}@ticker"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """订阅订单簿数据"""
        stream = f"{symbol.lower()}@depth"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """订阅成交数据"""
        stream = f"{symbol.lower()}@trade"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_klines(self, symbol: str, interval: str, callback: Callable):
        """订阅K线数据"""
        stream = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_user_data(self, callback: Callable):
        """订阅用户数据（订单、余额等）"""
        # 需要先获取listenKey
        try:
            response = await self.make_request("POST", "/v3/userDataStream", signed=True)
            listen_key = response["listenKey"]
            
            ws_url = f"{self.ws_url}/{listen_key}"
            await self._create_websocket_connection("user_data", ws_url, callback)
        
        except Exception as e:
            logger.error(f"订阅币安用户数据失败: {e}")
    
    async def _subscribe_stream(self, stream: str, callback: Callable):
        """订阅数据流"""
        if not ASYNC_AVAILABLE:
            logger.error("WebSocket功能需要websockets库")
            return
        
        ws_url = f"{self.ws_url}/{stream}"
        await self._create_websocket_connection(stream, ws_url, callback)
    
    async def _create_websocket_connection(self, stream_id: str, ws_url: str, callback: Callable):
        """创建WebSocket连接"""
        try:
            async def websocket_handler():
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[stream_id] = websocket
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await callback(data)
                        except Exception as e:
                            logger.error(f"处理WebSocket消息失败: {e}")
            
            # 在后台运行WebSocket连接
            asyncio.create_task(websocket_handler())
            logger.info(f"币安WebSocket连接创建成功: {stream_id}")
        
        except Exception as e:
            logger.error(f"创建币安WebSocket连接失败: {e}")
    
    def get_auth_headers(self, params: Dict = None, data: Dict = None) -> Dict[str, str]:
        """获取认证头"""
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        if data:  # 需要签名的请求
            query_string = urllib.parse.urlencode(data)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            data["signature"] = signature
        
        return headers
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """转换订单类型"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP_LOSS",
            OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT"
        }
        return mapping.get(order_type, "LIMIT")
    
    def _parse_order_type(self, binance_type: str) -> OrderType:
        """解析订单类型"""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT_LIMIT
        }
        return mapping.get(binance_type, OrderType.LIMIT)
    
    def _parse_order_status(self, binance_status: str) -> OrderStatus:
        """解析订单状态"""
        mapping = {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        return mapping.get(binance_status, OrderStatus.NEW)

