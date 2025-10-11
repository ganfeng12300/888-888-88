"""
Bybit交易所适配器
支持现货和期货交易
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
import hmac
import hashlib
import time
from decimal import Decimal

from .base_exchange import BaseExchange

class BybitAdapter(BaseExchange):
    """Bybit交易所适配器"""
    
    def __init__(self):
        super().__init__("bybit")
        self.base_url = "https://api.bybit.com"
        self.session = None
        
    async def initialize(self):
        """初始化连接"""
        self.session = aiohttp.ClientSession()
        self.logger.info("✅ Bybit适配器初始化完成")
        
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, params: Dict, timestamp: str) -> str:
        """生成签名"""
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        payload = timestamp + self.api_key + "5000" + param_str  # recv_window = 5000
        
        return hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """发送请求"""
        if params is None:
            params = {}
            
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json'
        }
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            params['api_key'] = self.api_key
            params['timestamp'] = timestamp
            params['recv_window'] = '5000'
            
            signature = self._generate_signature(params, timestamp)
            params['sign'] = signature
            
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
            else:
                async with self.session.post(url, json=params, headers=headers) as response:
                    data = await response.json()
                    
            if data.get('ret_code') != 0:
                raise Exception(f"Bybit API错误: {data}")
            return data.get('result', {})
            
        except Exception as e:
            self.logger.error(f"Bybit请求失败: {e}")
            raise
            
    async def get_ticker(self, symbol: str) -> Dict:
        """获取价格信息"""
        endpoint = "/v5/market/tickers"
        params = {"category": "spot", "symbol": symbol.upper()}
        data = await self._make_request("GET", endpoint, params)
        return data.get('list', [{}])[0] if data.get('list') else {}
        
    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """获取订单簿"""
        endpoint = "/v5/market/orderbook"
        params = {"category": "spot", "symbol": symbol.upper(), "limit": limit}
        return await self._make_request("GET", endpoint, params)
        
    async def get_balance(self) -> Dict:
        """获取账户余额"""
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "SPOT"}
        data = await self._make_request("GET", endpoint, params, signed=True)
        
        balances = {}
        if data.get('list'):
            for account in data['list']:
                for coin in account.get('coin', []):
                    asset = coin['coin']
                    available = float(coin['availableToWithdraw'])
                    locked = float(coin['locked'])
                    if available > 0 or locked > 0:
                        balances[asset] = {
                            'free': available,
                            'locked': locked,
                            'total': available + locked
                        }
        return balances
        
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: float = None) -> Dict:
        """下单"""
        endpoint = "/v5/order/create"
        
        params = {
            'category': 'spot',
            'symbol': symbol.upper(),
            'side': side.capitalize(),
            'orderType': order_type.capitalize(),
            'qty': str(quantity)
        }
        
        if price and order_type.upper() == 'LIMIT':
            params['price'] = str(price)
            
        return await self._make_request("POST", endpoint, params, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """取消订单"""
        endpoint = "/v5/order/cancel"
        params = {
            'category': 'spot',
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._make_request("POST", endpoint, params, signed=True)
        
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """获取订单状态"""
        endpoint = "/v5/order/realtime"
        params = {
            'category': 'spot',
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        data = await self._make_request("GET", endpoint, params, signed=True)
        return data.get('list', [{}])[0] if data.get('list') else {}
        
    async def get_trading_fees(self, symbol: str = None) -> Dict:
        """获取交易手续费"""
        endpoint = "/v5/account/fee-rate"
        params = {"category": "spot"}
        if symbol:
            params['symbol'] = symbol.upper()
        return await self._make_request("GET", endpoint, params, signed=True)
        
    async def get_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 获取主要交易对的价格信息
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
            market_data = {}
            
            for symbol in symbols:
                try:
                    ticker = await self.get_ticker(symbol)
                    orderbook = await self.get_orderbook(symbol, 20)
                    
                    if ticker:
                        market_data[symbol] = {
                            'price': float(ticker.get('lastPrice', 0)),
                            'bid': float(orderbook.get('b', [[0]])[0][0]) if orderbook.get('b') else 0,
                            'ask': float(orderbook.get('a', [[0]])[0][0]) if orderbook.get('a') else 0,
                            'volume': float(ticker.get('volume24h', 0)),
                            'change': float(ticker.get('price24hPcnt', 0)) * 100,
                            'timestamp': int(time.time() * 1000)
                        }
                except Exception as e:
                    self.logger.warning(f"获取{symbol}数据失败: {e}")
                    continue
                    
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取Bybit市场数据失败: {e}")
            return {}
            
    async def calculate_arbitrage_opportunity(self, symbol: str, other_exchange_price: float) -> Optional[Dict]:
        """计算套利机会"""
        try:
            ticker = await self.get_ticker(symbol)
            our_price = float(ticker.get('lastPrice', 0))
            
            if our_price == 0:
                return None
            
            # 计算价差
            price_diff = abs(our_price - other_exchange_price)
            price_diff_pct = (price_diff / our_price) * 100
            
            # 获取交易费用（Bybit一般是0.1%）
            trading_fee = 0.001  # 0.1%
            
            # 计算净利润（扣除手续费）
            net_profit_pct = price_diff_pct - (trading_fee * 2 * 100)  # 买卖两次手续费
            
            if net_profit_pct > 0.1:  # 最小0.1%利润
                return {
                    'symbol': symbol,
                    'exchange': 'bybit',
                    'our_price': our_price,
                    'other_price': other_exchange_price,
                    'price_diff': price_diff,
                    'price_diff_pct': price_diff_pct,
                    'net_profit_pct': net_profit_pct,
                    'trading_fee': trading_fee,
                    'direction': 'buy' if our_price < other_exchange_price else 'sell',
                    'timestamp': int(time.time() * 1000)
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"计算套利机会失败: {e}")
            return None
            
    async def execute_arbitrage_trade(self, opportunity: Dict, amount: float) -> Dict:
        """执行套利交易"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            price = opportunity['our_price']
            
            # 执行交易
            if direction == 'buy':
                order = await self.place_order(symbol, 'Buy', 'Market', amount)
            else:
                order = await self.place_order(symbol, 'Sell', 'Market', amount)
                
            self.logger.info(f"✅ Bybit套利交易执行成功: {symbol} {direction} {amount}")
            
            return {
                'success': True,
                'order_id': order.get('orderId'),
                'symbol': symbol,
                'side': direction,
                'amount': amount,
                'price': price,
                'exchange': 'bybit'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bybit套利交易执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'exchange': 'bybit'
            }
            
    async def get_funding_rate(self, symbol: str) -> Dict:
        """获取资金费率（期货）"""
        try:
            endpoint = "/v5/market/funding/history"
            params = {"category": "linear", "symbol": symbol.upper(), "limit": 1}
            data = await self._make_request("GET", endpoint, params)
            return data.get('list', [{}])[0] if data.get('list') else {}
        except Exception as e:
            self.logger.error(f"获取资金费率失败: {e}")
            return {}
            
    async def get_perpetual_info(self, symbol: str) -> Dict:
        """获取永续合约信息"""
        try:
            endpoint = "/v5/market/instruments-info"
            params = {"category": "linear", "symbol": symbol.upper()}
            data = await self._make_request("GET", endpoint, params)
            return data.get('list', [{}])[0] if data.get('list') else {}
        except Exception as e:
            self.logger.error(f"获取永续合约信息失败: {e}")
            return {}
            
    async def get_kline_data(self, symbol: str, interval: str = "1", limit: int = 200) -> List[Dict]:
        """获取K线数据"""
        try:
            endpoint = "/v5/market/kline"
            params = {
                "category": "spot",
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit
            }
            data = await self._make_request("GET", endpoint, params)
            
            klines = []
            for kline in data.get('list', []):
                klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            return klines
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            return []

