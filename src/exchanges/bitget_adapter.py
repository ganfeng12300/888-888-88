"""
Bitget交易所适配器
支持现货和期货交易
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
import hmac
import hashlib
import time
import base64
from decimal import Decimal

from .base_exchange import BaseExchange

class BitgetAdapter(BaseExchange):
    """Bitget交易所适配器"""
    
    def __init__(self):
        super().__init__("bitget")
        self.base_url = "https://api.bitget.com"
        self.session = None
        
    async def initialize(self):
        """初始化连接"""
        self.session = aiohttp.ClientSession()
        self.logger.info("✅ Bitget适配器初始化完成")
        
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """生成签名"""
        message = timestamp + method.upper() + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
        
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """发送请求"""
        if params is None:
            params = {}
            
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'locale': 'en-US'
        }
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            body = ''
            if method.upper() == 'POST' and params:
                import json
                body = json.dumps(params)
                
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            headers.update({
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase  # Bitget需要passphrase
            })
            
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
            else:
                async with self.session.post(url, json=params, headers=headers) as response:
                    data = await response.json()
                    
            if data.get('code') != '00000':
                raise Exception(f"Bitget API错误: {data}")
            return data.get('data', {})
            
        except Exception as e:
            self.logger.error(f"Bitget请求失败: {e}")
            raise
            
    async def get_ticker(self, symbol: str) -> Dict:
        """获取价格信息"""
        endpoint = "/api/spot/v1/market/ticker"
        params = {"symbol": symbol.upper()}
        return await self._make_request("GET", endpoint, params)
        
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿"""
        endpoint = "/api/spot/v1/market/depth"
        params = {"symbol": symbol.upper(), "limit": str(limit), "type": "step0"}
        return await self._make_request("GET", endpoint, params)
        
    async def get_balance(self) -> Dict:
        """获取账户余额"""
        endpoint = "/api/spot/v1/account/assets"
        data = await self._make_request("GET", endpoint, signed=True)
        
        balances = {}
        for balance in data:
            asset = balance['coinName']
            available = float(balance['available'])
            frozen = float(balance['frozen'])
            if available > 0 or frozen > 0:
                balances[asset] = {
                    'free': available,
                    'locked': frozen,
                    'total': available + frozen
                }
        return balances
        
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: float = None) -> Dict:
        """下单"""
        endpoint = "/api/spot/v1/trade/orders"
        
        params = {
            'symbol': symbol.upper(),
            'side': side.lower(),
            'orderType': order_type.lower(),
            'size': str(quantity)
        }
        
        if price and order_type.lower() == 'limit':
            params['price'] = str(price)
            
        return await self._make_request("POST", endpoint, params, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """取消订单"""
        endpoint = "/api/spot/v1/trade/cancel-order"
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._make_request("POST", endpoint, params, signed=True)
        
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """获取订单状态"""
        endpoint = "/api/spot/v1/trade/orderInfo"
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._make_request("GET", endpoint, params, signed=True)
        
    async def get_trading_fees(self, symbol: str = None) -> Dict:
        """获取交易手续费"""
        # Bitget的手续费查询接口
        endpoint = "/api/spot/v1/account/tradeFee"
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return await self._make_request("GET", endpoint, params, signed=True)
        
    async def get_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 获取主要交易对的价格信息
            symbols = ['BTCUSDT', 'ETHUSDT', 'BGBUSDT', 'ADAUSDT', 'DOTUSDT']
            market_data = {}
            
            for symbol in symbols:
                try:
                    ticker = await self.get_ticker(symbol)
                    orderbook = await self.get_orderbook(symbol, 20)
                    
                    market_data[symbol] = {
                        'price': float(ticker['close']),
                        'bid': float(orderbook['bids'][0][0]) if orderbook['bids'] else 0,
                        'ask': float(orderbook['asks'][0][0]) if orderbook['asks'] else 0,
                        'volume': float(ticker['baseVol']),
                        'change': float(ticker['change']),
                        'timestamp': int(time.time() * 1000)
                    }
                except Exception as e:
                    self.logger.warning(f"获取{symbol}数据失败: {e}")
                    continue
                    
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取Bitget市场数据失败: {e}")
            return {}
            
    async def calculate_arbitrage_opportunity(self, symbol: str, other_exchange_price: float) -> Optional[Dict]:
        """计算套利机会"""
        try:
            ticker = await self.get_ticker(symbol)
            our_price = float(ticker['close'])
            
            # 计算价差
            price_diff = abs(our_price - other_exchange_price)
            price_diff_pct = (price_diff / our_price) * 100
            
            # 获取交易费用（Bitget一般是0.1%）
            trading_fee = 0.001  # 0.1%
            
            # 计算净利润（扣除手续费）
            net_profit_pct = price_diff_pct - (trading_fee * 2 * 100)  # 买卖两次手续费
            
            if net_profit_pct > 0.1:  # 最小0.1%利润
                return {
                    'symbol': symbol,
                    'exchange': 'bitget',
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
                order = await self.place_order(symbol, 'buy', 'market', amount)
            else:
                order = await self.place_order(symbol, 'sell', 'market', amount)
                
            self.logger.info(f"✅ Bitget套利交易执行成功: {symbol} {direction} {amount}")
            
            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': direction,
                'amount': amount,
                'price': price,
                'exchange': 'bitget'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bitget套利交易执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'exchange': 'bitget'
            }
            
    async def get_funding_rate(self, symbol: str) -> Dict:
        """获取资金费率（期货）"""
        try:
            endpoint = "/api/mix/v1/market/current-fundRate"
            params = {"symbol": symbol.upper()}
            return await self._make_request("GET", endpoint, params)
        except Exception as e:
            self.logger.error(f"获取资金费率失败: {e}")
            return {}
            
    async def get_perpetual_info(self, symbol: str) -> Dict:
        """获取永续合约信息"""
        try:
            endpoint = "/api/mix/v1/market/contracts"
            params = {"productType": "umcbl"}  # USDT永续
            data = await self._make_request("GET", endpoint, params)
            
            for contract in data:
                if contract['symbol'] == symbol.upper():
                    return contract
            return {}
        except Exception as e:
            self.logger.error(f"获取永续合约信息失败: {e}")
            return {}

