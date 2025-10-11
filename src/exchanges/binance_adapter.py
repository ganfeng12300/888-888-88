"""
Binance交易所适配器
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

class BinanceAdapter(BaseExchange):
    """Binance交易所适配器"""
    
    def __init__(self):
        super().__init__("binance")
        self.base_url = "https://api.binance.com"
        self.futures_url = "https://fapi.binance.com"
        self.session = None
        
    async def initialize(self):
        """初始化连接"""
        self.session = aiohttp.ClientSession()
        self.logger.info("✅ Binance适配器初始化完成")
        
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, params: Dict) -> str:
        """生成签名"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """发送请求"""
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
            
        headers = {
            'X-MBX-APIKEY': self.api_key
        } if signed else {}
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                data = await response.json()
                if response.status != 200:
                    raise Exception(f"Binance API错误: {data}")
                return data
        except Exception as e:
            self.logger.error(f"Binance请求失败: {e}")
            raise
            
    async def get_ticker(self, symbol: str) -> Dict:
        """获取价格信息"""
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": symbol.upper()}
        return await self._make_request("GET", endpoint, params)
        
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿"""
        endpoint = "/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": limit}
        return await self._make_request("GET", endpoint, params)
        
    async def get_balance(self) -> Dict:
        """获取账户余额"""
        endpoint = "/api/v3/account"
        data = await self._make_request("GET", endpoint, signed=True)
        
        balances = {}
        for balance in data.get('balances', []):
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                balances[asset] = {
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                }
        return balances
        
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: float = None) -> Dict:
        """下单"""
        endpoint = "/api/v3/order"
        
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        
        if price and order_type.upper() in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'
            
        return await self._make_request("POST", endpoint, params, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """取消订单"""
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._make_request("DELETE", endpoint, params, signed=True)
        
    async def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """获取订单状态"""
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._make_request("GET", endpoint, params, signed=True)
        
    async def get_trading_fees(self, symbol: str = None) -> Dict:
        """获取交易手续费"""
        endpoint = "/api/v3/tradeFee"
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return await self._make_request("GET", endpoint, params, signed=True)
        
    async def get_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 获取主要交易对的价格信息
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
            market_data = {}
            
            for symbol in symbols:
                ticker = await self.get_ticker(symbol)
                orderbook = await self.get_orderbook(symbol, 20)
                
                market_data[symbol] = {
                    'price': float(ticker['lastPrice']),
                    'bid': float(orderbook['bids'][0][0]) if orderbook['bids'] else 0,
                    'ask': float(orderbook['asks'][0][0]) if orderbook['asks'] else 0,
                    'volume': float(ticker['volume']),
                    'change': float(ticker['priceChangePercent']),
                    'timestamp': int(time.time() * 1000)
                }
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取Binance市场数据失败: {e}")
            return {}
            
    async def calculate_arbitrage_opportunity(self, symbol: str, other_exchange_price: float) -> Optional[Dict]:
        """计算套利机会"""
        try:
            ticker = await self.get_ticker(symbol)
            our_price = float(ticker['lastPrice'])
            
            # 计算价差
            price_diff = abs(our_price - other_exchange_price)
            price_diff_pct = (price_diff / our_price) * 100
            
            # 获取交易费用
            fees = await self.get_trading_fees(symbol)
            trading_fee = float(fees[0]['takerCommission']) if fees else 0.001
            
            # 计算净利润（扣除手续费）
            net_profit_pct = price_diff_pct - (trading_fee * 2 * 100)  # 买卖两次手续费
            
            if net_profit_pct > 0.1:  # 最小0.1%利润
                return {
                    'symbol': symbol,
                    'exchange': 'binance',
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
                order = await self.place_order(symbol, 'BUY', 'MARKET', amount)
            else:
                order = await self.place_order(symbol, 'SELL', 'MARKET', amount)
                
            self.logger.info(f"✅ Binance套利交易执行成功: {symbol} {direction} {amount}")
            
            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': direction,
                'amount': amount,
                'price': price,
                'exchange': 'binance'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Binance套利交易执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'exchange': 'binance'
            }

