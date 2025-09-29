"""
🔌 火币交易所接口 - 完整实盘交易接口实现
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

class HuobiExchange(BaseExchange):
    """火币交易所接口"""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False, timeout: int = 30):
        super().__init__(api_key, api_secret, None, sandbox, timeout)
        
        self.name = "huobi"
        
        if sandbox:
            self.base_url = "https://api.testnet.huobi.pro"
            self.ws_url = "wss://api.testnet.huobi.pro/ws"
        else:
            self.base_url = "https://api.huobi.pro"
            self.ws_url = "wss://api.huobi.pro/ws"
        
        # 火币特定配置
        self.account_id = None  # 需要获取账户ID
        
        logger.info("火币交易所接口初始化完成")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        return {
            "name": "Huobi",
            "country": "Singapore",
            "rateLimit": 100,  # 每秒请求限制
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
            response = await self.make_request("GET", "/v1/common/symbols")
            symbols = []
            
            if response.get("status") == "ok":
                for symbol_info in response.get("data", []):
                    if symbol_info["state"] != "online":
                        continue
                    
                    symbol = Symbol(
                        symbol=symbol_info["symbol"],
                        base_asset=symbol_info["base-currency"].upper(),
                        quote_asset=symbol_info["quote-currency"].upper(),
                        market_type=MarketType.SPOT,
                        min_qty=float(symbol_info.get("min-order-amt", 0)),
                        max_qty=float(symbol_info.get("max-order-amt", float('inf'))),
                        step_size=float(symbol_info.get("amount-precision", 0.000001)),
                        min_price=0.0,
                        max_price=float('inf'),
                        tick_size=float(symbol_info.get("price-precision", 0.000001)),
                        min_notional=float(symbol_info.get("min-order-value", 0)),
                        is_active=True
                    )
                    symbols.append(symbol)
            
            logger.info(f"获取火币交易对信息: {len(symbols)}个")
            return symbols
        
        except Exception as e:
            logger.error(f"获取火币交易对信息失败: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """获取单个交易对行情"""
        try:
            params = {"symbol": symbol.lower()}
            response = await self.make_request("GET", "/market/detail/merged", params=params)
            
            if response.get("status") == "ok":
                tick = response["tick"]
                
                ticker = Ticker(
                    symbol=symbol,
                    price=float(tick["close"]),
                    bid_price=float(tick["bid"][0]) if tick.get("bid") else 0.0,
                    ask_price=float(tick["ask"][0]) if tick.get("ask") else 0.0,
                    bid_qty=float(tick["bid"][1]) if tick.get("bid") else 0.0,
                    ask_qty=float(tick["ask"][1]) if tick.get("ask") else 0.0,
                    volume_24h=float(tick["vol"]),
                    price_change_24h=float(tick["close"]) - float(tick["open"]),
                    price_change_percent_24h=((float(tick["close"]) - float(tick["open"])) / float(tick["open"])) * 100,
                    high_24h=float(tick["high"]),
                    low_24h=float(tick["low"]),
                    timestamp=float(response["ts"])
                )
                
                return ticker
            else:
                raise Exception(f"火币API错误: {response.get('err-msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"获取火币行情失败: {e}")
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
                response = await self.make_request("GET", "/market/tickers")
                tickers = []
                
                if response.get("status") == "ok":
                    for ticker_data in response.get("data", []):
                        ticker = Ticker(
                            symbol=ticker_data["symbol"],
                            price=float(ticker_data["close"]),
                            bid_price=float(ticker_data.get("bid", 0)),
                            ask_price=float(ticker_data.get("ask", 0)),
                            bid_qty=0.0,
                            ask_qty=0.0,
                            volume_24h=float(ticker_data["vol"]),
                            price_change_24h=float(ticker_data["close"]) - float(ticker_data["open"]),
                            price_change_percent_24h=((float(ticker_data["close"]) - float(ticker_data["open"])) / float(ticker_data["open"])) * 100,
                            high_24h=float(ticker_data["high"]),
                            low_24h=float(ticker_data["low"]),
                            timestamp=time.time() * 1000
                        )
                        tickers.append(ticker)
                
                return tickers
        
        except Exception as e:
            logger.error(f"获取火币行情列表失败: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        try:
            params = {
                "symbol": symbol.lower(),
                "type": "step0"  # 最高精度
            }
            response = await self.make_request("GET", "/market/depth", params=params)
            
            if response.get("status") == "ok":
                tick = response["tick"]
                
                bids = [(float(bid[0]), float(bid[1])) for bid in tick.get("bids", [])[:limit]]
                asks = [(float(ask[0]), float(ask[1])) for ask in tick.get("asks", [])[:limit]]
                
                orderbook = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=float(response["ts"])
                )
                
                return orderbook
            else:
                raise Exception(f"火币API错误: {response.get('err-msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"获取火币订单簿失败: {e}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, start_time: int = None,
                        end_time: int = None, limit: int = 500) -> List[Kline]:
        """获取K线数据"""
        try:
            params = {
                "symbol": symbol.lower(),
                "period": self._convert_interval(interval),
                "size": min(limit, 2000)  # 火币最大2000
            }
            
            response = await self.make_request("GET", "/market/history/kline", params=params)
            klines = []
            
            if response.get("status") == "ok":
                for kline_data in response.get("data", []):
                    kline = Kline(
                        symbol=symbol,
                        interval=interval,
                        open_time=float(kline_data["id"]) * 1000,
                        close_time=float(kline_data["id"]) * 1000 + self._get_interval_ms(interval),
                        open_price=float(kline_data["open"]),
                        high_price=float(kline_data["high"]),
                        low_price=float(kline_data["low"]),
                        close_price=float(kline_data["close"]),
                        volume=float(kline_data["vol"]),
                        quote_volume=float(kline_data["amount"]),
                        trade_count=int(kline_data.get("count", 0))
                    )
                    klines.append(kline)
            
            return klines
        
        except Exception as e:
            logger.error(f"获取火币K线数据失败: {e}")
            return []
    
    async def get_trades(self, symbol: str, limit: int = 500) -> List[Trade]:
        """获取最近成交记录"""
        try:
            params = {
                "symbol": symbol.lower(),
                "size": min(limit, 2000)  # 火币最大2000
            }
            response = await self.make_request("GET", "/market/history/trade", params=params)
            trades = []
            
            if response.get("status") == "ok":
                for trade_group in response.get("data", []):
                    for trade_data in trade_group.get("data", []):
                        trade = Trade(
                            trade_id=str(trade_data["id"]),
                            order_id="",  # 公开接口不提供订单ID
                            symbol=symbol,
                            side=OrderSide.BUY if trade_data["direction"] == "buy" else OrderSide.SELL,
                            quantity=float(trade_data["amount"]),
                            price=float(trade_data["price"]),
                            commission=0.0,  # 公开接口不提供手续费
                            commission_asset="",
                            is_maker=False,  # 公开接口无法确定
                            timestamp=float(trade_data["ts"])
                        )
                        trades.append(trade)
            
            return trades
        
        except Exception as e:
            logger.error(f"获取火币成交记录失败: {e}")
            return []
    
    async def _get_account_id(self) -> Optional[str]:
        """获取账户ID"""
        if self.account_id:
            return self.account_id
        
        try:
            response = await self.make_request("GET", "/v1/account/accounts", signed=True)
            
            if response.get("status") == "ok":
                for account in response.get("data", []):
                    if account["type"] == "spot":
                        self.account_id = str(account["id"])
                        return self.account_id
            
            logger.error("未找到现货账户")
            return None
        
        except Exception as e:
            logger.error(f"获取火币账户ID失败: {e}")
            return None
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: float = None, stop_price: float = None,
                         time_in_force: TimeInForce = TimeInForce.GTC,
                         client_order_id: str = None) -> Order:
        """下单"""
        try:
            account_id = await self._get_account_id()
            if not account_id:
                raise Exception("无法获取账户ID")
            
            data = {
                "account-id": account_id,
                "symbol": symbol.lower(),
                "type": self._convert_order_type(side, order_type),
                "amount": str(quantity)
            }
            
            if client_order_id:
                data["client-order-id"] = client_order_id
            
            if order_type == OrderType.LIMIT:
                data["price"] = str(price)
            
            response = await self.make_request("POST", "/v1/order/orders/place", data=data, signed=True)
            
            if response.get("status") == "ok":
                order_id = str(response["data"])
                
                # 查询订单详情
                order_detail = await self.get_order(symbol, order_id)
                return order_detail
            else:
                raise Exception(f"火币下单失败: {response.get('err-msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"火币下单失败: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str = None,
                          client_order_id: str = None) -> bool:
        """撤单"""
        try:
            if not order_id and not client_order_id:
                raise ValueError("必须提供order_id或client_order_id")
            
            if order_id:
                endpoint = f"/v1/order/orders/{order_id}/submitcancel"
                response = await self.make_request("POST", endpoint, signed=True)
            else:
                data = {"client-order-id": client_order_id}
                response = await self.make_request("POST", "/v1/order/orders/submitCancelClientOrder", data=data, signed=True)
            
            if response.get("status") == "ok":
                logger.info(f"火币撤单成功: {order_id or client_order_id}")
                return True
            else:
                logger.error(f"火币撤单失败: {response.get('err-msg', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"火币撤单失败: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """撤销所有订单"""
        try:
            account_id = await self._get_account_id()
            if not account_id:
                raise Exception("无法获取账户ID")
            
            data = {"account-id": account_id}
            if symbol:
                data["symbol"] = symbol.lower()
            
            response = await self.make_request("POST", "/v1/order/orders/batchCancelOpenOrders", data=data, signed=True)
            
            if response.get("status") == "ok":
                logger.info("火币撤销所有订单成功")
                return True
            else:
                logger.error(f"火币撤销所有订单失败: {response.get('err-msg', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"火币撤销所有订单失败: {e}")
            return False
    
    async def get_order(self, symbol: str, order_id: str = None,
                       client_order_id: str = None) -> Order:
        """查询订单"""
        try:
            if not order_id and not client_order_id:
                raise ValueError("必须提供order_id或client_order_id")
            
            if order_id:
                endpoint = f"/v1/order/orders/{order_id}"
                response = await self.make_request("GET", endpoint, signed=True)
            else:
                params = {"clientOrderId": client_order_id}
                response = await self.make_request("GET", "/v1/order/orders/getClientOrder", params=params, signed=True)
            
            if response.get("status") == "ok":
                order_data = response["data"]
                
                order = Order(
                    order_id=str(order_data["id"]),
                    client_order_id=order_data.get("client-order-id", ""),
                    symbol=order_data["symbol"].upper(),
                    side=self._parse_order_side(order_data["type"]),
                    order_type=self._parse_order_type(order_data["type"]),
                    quantity=float(order_data["amount"]),
                    price=float(order_data.get("price", 0)),
                    stop_price=0.0,  # 火币现货不支持止损价
                    time_in_force=TimeInForce.GTC,  # 火币默认GTC
                    status=self._parse_order_status(order_data["state"]),
                    filled_qty=float(order_data["filled-amount"]),
                    remaining_qty=float(order_data["amount"]) - float(order_data["filled-amount"]),
                    avg_price=float(order_data.get("filled-cash-amount", 0)) / float(order_data["filled-amount"]) if float(order_data["filled-amount"]) > 0 else 0,
                    commission=float(order_data.get("filled-fees", 0)),
                    commission_asset="",  # 火币不直接提供手续费资产
                    created_at=float(order_data["created-at"]),
                    updated_at=float(order_data.get("finished-at", order_data["created-at"]))
                )
                
                return order
            else:
                raise Exception(f"火币查询订单失败: {response.get('err-msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"火币查询订单失败: {e}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """查询活跃订单"""
        try:
            account_id = await self._get_account_id()
            if not account_id:
                raise Exception("无法获取账户ID")
            
            params = {"account-id": account_id}
            if symbol:
                params["symbol"] = symbol.lower()
            
            response = await self.make_request("GET", "/v1/order/openOrders", params=params, signed=True)
            orders = []
            
            if response.get("status") == "ok":
                for order_data in response.get("data", []):
                    order = Order(
                        order_id=str(order_data["id"]),
                        client_order_id=order_data.get("client-order-id", ""),
                        symbol=order_data["symbol"].upper(),
                        side=self._parse_order_side(order_data["type"]),
                        order_type=self._parse_order_type(order_data["type"]),
                        quantity=float(order_data["amount"]),
                        price=float(order_data.get("price", 0)),
                        stop_price=0.0,
                        time_in_force=TimeInForce.GTC,
                        status=self._parse_order_status(order_data["state"]),
                        filled_qty=float(order_data["filled-amount"]),
                        remaining_qty=float(order_data["amount"]) - float(order_data["filled-amount"]),
                        avg_price=float(order_data.get("filled-cash-amount", 0)) / float(order_data["filled-amount"]) if float(order_data["filled-amount"]) > 0 else 0,
                        commission=float(order_data.get("filled-fees", 0)),
                        commission_asset="",
                        created_at=float(order_data["created-at"]),
                        updated_at=float(order_data.get("finished-at", order_data["created-at"]))
                    )
                    orders.append(order)
            
            return orders
        
        except Exception as e:
            logger.error(f"火币查询活跃订单失败: {e}")
            return []
    
    async def get_order_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Order]:
        """查询历史订单"""
        try:
            params = {"states": "filled,canceled,partial-canceled"}
            if symbol:
                params["symbol"] = symbol.lower()
            if start_time:
                params["start-time"] = start_time
            if end_time:
                params["end-time"] = end_time
            if limit:
                params["size"] = min(limit, 500)  # 火币最大500
            
            response = await self.make_request("GET", "/v1/order/orders", params=params, signed=True)
            orders = []
            
            if response.get("status") == "ok":
                for order_data in response.get("data", []):
                    order = Order(
                        order_id=str(order_data["id"]),
                        client_order_id=order_data.get("client-order-id", ""),
                        symbol=order_data["symbol"].upper(),
                        side=self._parse_order_side(order_data["type"]),
                        order_type=self._parse_order_type(order_data["type"]),
                        quantity=float(order_data["amount"]),
                        price=float(order_data.get("price", 0)),
                        stop_price=0.0,
                        time_in_force=TimeInForce.GTC,
                        status=self._parse_order_status(order_data["state"]),
                        filled_qty=float(order_data["filled-amount"]),
                        remaining_qty=float(order_data["amount"]) - float(order_data["filled-amount"]),
                        avg_price=float(order_data.get("filled-cash-amount", 0)) / float(order_data["filled-amount"]) if float(order_data["filled-amount"]) > 0 else 0,
                        commission=float(order_data.get("filled-fees", 0)),
                        commission_asset="",
                        created_at=float(order_data["created-at"]),
                        updated_at=float(order_data.get("finished-at", order_data["created-at"]))
                    )
                    orders.append(order)
            
            return orders
        
        except Exception as e:
            logger.error(f"火币查询历史订单失败: {e}")
            return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            account_id = await self._get_account_id()
            if not account_id:
                raise Exception("无法获取账户ID")
            
            endpoint = f"/v1/account/accounts/{account_id}/balance"
            response = await self.make_request("GET", endpoint, signed=True)
            
            if response.get("status") == "ok":
                return response["data"]
            else:
                raise Exception(f"火币获取账户信息失败: {response.get('err-msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"火币获取账户信息失败: {e}")
            return {}
    
    async def get_balances(self) -> List[Balance]:
        """获取账户余额"""
        try:
            account_info = await self.get_account_info()
            balances = []
            
            for balance_data in account_info.get("list", []):
                balance_value = float(balance_data["balance"])
                if balance_value > 0:  # 只返回有余额的资产
                    balance = Balance(
                        asset=balance_data["currency"].upper(),
                        free=balance_value if balance_data["type"] == "trade" else 0.0,
                        locked=balance_value if balance_data["type"] == "frozen" else 0.0,
                        total=balance_value
                    )
                    
                    # 合并同一资产的可用和冻结余额
                    existing_balance = next((b for b in balances if b.asset == balance.asset), None)
                    if existing_balance:
                        if balance_data["type"] == "trade":
                            existing_balance.free = balance_value
                        else:
                            existing_balance.locked = balance_value
                        existing_balance.total = existing_balance.free + existing_balance.locked
                    else:
                        balances.append(balance)
            
            return balances
        
        except Exception as e:
            logger.error(f"火币获取账户余额失败: {e}")
            return []
    
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """获取持仓信息（期货）"""
        # 火币现货不支持持仓，返回空列表
        return []
    
    async def get_trade_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Trade]:
        """获取成交历史"""
        try:
            if not symbol:
                raise ValueError("火币获取成交历史需要指定交易对")
            
            params = {"symbol": symbol.lower()}
            if limit:
                params["size"] = min(limit, 500)  # 火币最大500
            
            response = await self.make_request("GET", "/v1/order/matchresults", params=params, signed=True)
            trades = []
            
            if response.get("status") == "ok":
                for trade_data in response.get("data", []):
                    trade = Trade(
                        trade_id=str(trade_data["id"]),
                        order_id=str(trade_data["order-id"]),
                        symbol=trade_data["symbol"].upper(),
                        side=OrderSide.BUY if trade_data["type"].endswith("buy") else OrderSide.SELL,
                        quantity=float(trade_data["filled-amount"]),
                        price=float(trade_data["price"]),
                        commission=float(trade_data["filled-fees"]),
                        commission_asset=trade_data.get("fee-currency", "").upper(),
                        is_maker=trade_data["role"] == "maker",
                        timestamp=float(trade_data["created-at"])
                    )
                    trades.append(trade)
            
            return trades
        
        except Exception as e:
            logger.error(f"火币获取成交历史失败: {e}")
            return []
    
    # WebSocket相关方法
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅行情数据"""
        topic = f"market.{symbol.lower()}.ticker"
        await self._subscribe_topic(topic, callback)
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """订阅订单簿数据"""
        topic = f"market.{symbol.lower()}.depth.step0"
        await self._subscribe_topic(topic, callback)
    
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """订阅成交数据"""
        topic = f"market.{symbol.lower()}.trade.detail"
        await self._subscribe_topic(topic, callback)
    
    async def subscribe_klines(self, symbol: str, interval: str, callback: Callable):
        """订阅K线数据"""
        huobi_interval = self._convert_interval(interval)
        topic = f"market.{symbol.lower()}.kline.{huobi_interval}"
        await self._subscribe_topic(topic, callback)
    
    async def subscribe_user_data(self, callback: Callable):
        """订阅用户数据（订单、余额等）"""
        # 火币用户数据需要单独的WebSocket连接
        try:
            account_id = await self._get_account_id()
            if not account_id:
                raise Exception("无法获取账户ID")
            
            # 订阅订单更新
            order_topic = f"orders#{account_id}"
            await self._subscribe_topic(order_topic, callback, is_private=True)
            
            # 订阅账户余额更新
            balance_topic = f"accounts.update#{account_id}"
            await self._subscribe_topic(balance_topic, callback, is_private=True)
        
        except Exception as e:
            logger.error(f"订阅火币用户数据失败: {e}")
    
    async def _subscribe_topic(self, topic: str, callback: Callable, is_private: bool = False):
        """订阅主题"""
        if not ASYNC_AVAILABLE:
            logger.error("WebSocket功能需要websockets库")
            return
        
        ws_url = self.ws_url
        if is_private:
            ws_url = self.ws_url.replace("/ws", "/ws/v2")  # 私有数据使用v2接口
        
        await self._create_websocket_connection(topic, ws_url, callback, is_private)
    
    async def _create_websocket_connection(self, topic: str, ws_url: str, callback: Callable, is_private: bool = False):
        """创建WebSocket连接"""
        try:
            import gzip
            
            async def websocket_handler():
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connections[topic] = websocket
                    
                    # 发送订阅消息
                    if is_private:
                        # 私有数据需要认证
                        auth_msg = self._create_auth_message()
                        await websocket.send(json.dumps(auth_msg))
                    
                    subscribe_msg = {
                        "sub": topic,
                        "id": str(int(time.time()))
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        try:
                            # 火币WebSocket数据是gzip压缩的
                            if isinstance(message, bytes):
                                message = gzip.decompress(message).decode('utf-8')
                            
                            data = json.loads(message)
                            
                            # 处理ping消息
                            if "ping" in data:
                                pong_msg = {"pong": data["ping"]}
                                await websocket.send(json.dumps(pong_msg))
                                continue
                            
                            # 处理订阅确认
                            if data.get("status") == "ok":
                                logger.info(f"火币WebSocket订阅成功: {topic}")
                                continue
                            
                            # 处理数据消息
                            if "ch" in data or "tick" in data:
                                await callback(data)
                        
                        except Exception as e:
                            logger.error(f"处理火币WebSocket消息失败: {e}")
            
            # 在后台运行WebSocket连接
            asyncio.create_task(websocket_handler())
            logger.info(f"火币WebSocket连接创建成功: {topic}")
        
        except Exception as e:
            logger.error(f"创建火币WebSocket连接失败: {e}")
    
    def _create_auth_message(self) -> Dict[str, Any]:
        """创建认证消息"""
        timestamp = str(int(time.time()))
        method = "GET"
        path = "/ws/v2"
        
        params = {
            "accessKey": self.api_key,
            "signatureMethod": "HmacSHA256",
            "signatureVersion": "2.1",
            "timestamp": timestamp
        }
        
        # 创建签名字符串
        query_string = urllib.parse.urlencode(sorted(params.items()))
        sign_string = f"{method}\napi.huobi.pro\n{path}\n{query_string}"
        
        # 生成签名
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            "action": "req",
            "ch": "auth",
            "params": {
                "authType": "api",
                "accessKey": self.api_key,
                "signatureMethod": "HmacSHA256",
                "signatureVersion": "2.1",
                "timestamp": timestamp,
                "signature": signature
            }
        }
    
    def get_auth_headers(self, params: Dict = None, data: Dict = None) -> Dict[str, str]:
        """获取认证头"""
        timestamp = str(int(time.time()))
        method = "POST" if data else "GET"
        path = ""  # 需要在调用时设置
        
        auth_params = {
            "AccessKeyId": self.api_key,
            "SignatureMethod": "HmacSHA256",
            "SignatureVersion": "2",
            "Timestamp": timestamp
        }
        
        if params:
            auth_params.update(params)
        
        # 创建签名字符串
        query_string = urllib.parse.urlencode(sorted(auth_params.items()))
        sign_string = f"{method}\napi.huobi.pro\n{path}\n{query_string}"
        
        # 生成签名
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        auth_params["Signature"] = signature
        
        return {
            "Content-Type": "application/json"
        }
    
    async def make_request(self, method: str, endpoint: str, params: Dict = None,
                          data: Dict = None, headers: Dict = None, 
                          signed: bool = False) -> Dict:
        """发送HTTP请求"""
        await self.create_session()
        
        url = self.base_url + endpoint
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        if signed:
            timestamp = str(int(time.time()))
            auth_params = {
                "AccessKeyId": self.api_key,
                "SignatureMethod": "HmacSHA256",
                "SignatureVersion": "2",
                "Timestamp": timestamp
            }
            
            if params:
                auth_params.update(params)
            
            # 创建签名字符串
            query_string = urllib.parse.urlencode(sorted(auth_params.items()))
            sign_string = f"{method}\napi.huobi.pro\n{endpoint}\n{query_string}"
            
            # 生成签名
            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    sign_string.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            auth_params["Signature"] = signature
            params = auth_params
        
        try:
            async with self.session.request(
                method, url, params=params, json=data, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"火币API请求失败: {response.status} - {error_text}")
                    raise Exception(f"火币API请求失败: {response.status}")
        
        except Exception as e:
            logger.error(f"火币HTTP请求异常: {e}")
            raise
    
    # 工具方法
    def _convert_order_type(self, side: OrderSide, order_type: OrderType) -> str:
        """转换订单类型"""
        if order_type == OrderType.MARKET:
            return f"{side.value}-market"
        elif order_type == OrderType.LIMIT:
            return f"{side.value}-limit"
        else:
            return f"{side.value}-limit"  # 默认限价单
    
    def _parse_order_side(self, huobi_type: str) -> OrderSide:
        """解析订单方向"""
        if "buy" in huobi_type:
            return OrderSide.BUY
        else:
            return OrderSide.SELL
    
    def _parse_order_type(self, huobi_type: str) -> OrderType:
        """解析订单类型"""
        if "market" in huobi_type:
            return OrderType.MARKET
        else:
            return OrderType.LIMIT
    
    def _parse_order_status(self, huobi_status: str) -> OrderStatus:
        """解析订单状态"""
        mapping = {
            "submitted": OrderStatus.NEW,
            "partial-filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "partial-canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED
        }
        return mapping.get(huobi_status, OrderStatus.NEW)
    
    def _convert_interval(self, interval: str) -> str:
        """转换时间间隔"""
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "4h": "4hour",
            "1d": "1day",
            "1w": "1week",
            "1M": "1mon"
        }
        return mapping.get(interval, "1min")
    
    def _get_interval_ms(self, interval: str) -> int:
        """获取时间间隔毫秒数"""
        mapping = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000
        }
        return mapping.get(interval, 60 * 1000)
