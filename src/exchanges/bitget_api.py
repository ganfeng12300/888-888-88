#!/usr/bin/env python3
"""
🔗 Bitget API接口 - 生产级实盘交易接口
Bitget API Interface - Production-Grade Live Trading Interface

生产级特性：
- 完整的Bitget API封装
- 实时行情数据获取
- 实盘订单执行
- 账户资产管理
- WebSocket实时推送
"""

import hmac
import hashlib
import base64
import time
import json
import requests
import websocket
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import urllib.parse

from ..monitoring.unified_logging_system import UnifiedLogger

@dataclass
class BitgetConfig:
    """Bitget配置"""
    api_key: str
    secret_key: str
    passphrase: str
    sandbox: bool = False
    timeout: int = 10

class BitgetAPI:
    """Bitget API主类"""
    
    def __init__(self, config: BitgetConfig):
        self.logger = UnifiedLogger("BitgetAPI")
        self.config = config
        
        # API端点
        if config.sandbox:
            self.base_url = "https://api.bitget.com"  # Bitget没有沙盒环境，使用正式环境
        else:
            self.base_url = "https://api.bitget.com"
        
        self.ws_url = "wss://ws.bitget.com/spot/v1/stream"
        
        # WebSocket连接
        self.ws = None
        self.ws_connected = False
        self.ws_callbacks = {}
        
        # 会话管理
        self.session = requests.Session()
        self.session.timeout = config.timeout
        
        self.logger.info("Bitget API初始化完成")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """生成API签名"""
        try:
            # 构建签名字符串
            message = timestamp + method.upper() + request_path + body
            
            # 生成签名
            signature = base64.b64encode(
                hmac.new(
                    self.config.secret_key.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            return signature
            
        except Exception as e:
            self.logger.error(f"生成签名失败: {e}")
            return ""
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """获取请求头"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        headers = {
            'ACCESS-KEY': self.config.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.config.passphrase,
            'Content-Type': 'application/json',
            'locale': 'en-US'
        }
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """发送API请求"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            # 处理查询参数
            if params:
                query_string = urllib.parse.urlencode(params)
                url += f"?{query_string}"
                request_path = f"{endpoint}?{query_string}"
            else:
                request_path = endpoint
            
            # 处理请求体
            body = ""
            if data:
                body = json.dumps(data)
            
            # 获取请求头
            headers = self._get_headers(method, request_path, body)
            
            # 发送请求
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, data=body)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers, data=body)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == '00000':  # Bitget成功代码
                    return result.get('data')
                else:
                    self.logger.error(f"API错误: {result.get('msg', 'Unknown error')}")
                    return None
            else:
                self.logger.error(f"HTTP错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"API请求失败: {e}")
            return None
    
    # 账户相关接口
    def get_account_info(self) -> Optional[Dict]:
        """获取账户信息"""
        return self._make_request('GET', '/api/spot/v1/account/assets')
    
    def get_balance(self, coin: str = None) -> Optional[Dict]:
        """获取余额"""
        if coin:
            return self._make_request('GET', f'/api/spot/v1/account/assets', {'coin': coin})
        else:
            return self._make_request('GET', '/api/spot/v1/account/assets')
    
    # 市场数据接口
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """获取ticker数据"""
        return self._make_request('GET', '/api/spot/v1/market/ticker', {'symbol': symbol})
    
    def get_tickers(self) -> Optional[List[Dict]]:
        """获取所有ticker数据"""
        return self._make_request('GET', '/api/spot/v1/market/tickers')
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """获取订单簿"""
        return self._make_request('GET', '/api/spot/v1/market/depth', {
            'symbol': symbol,
            'limit': limit
        })
    
    def get_klines(self, symbol: str, granularity: str, start_time: str = None, end_time: str = None) -> Optional[List[List]]:
        """获取K线数据"""
        params = {
            'symbol': symbol,
            'granularity': granularity
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._make_request('GET', '/api/spot/v1/market/candles', params)
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """获取最近成交"""
        return self._make_request('GET', '/api/spot/v1/market/fills', {
            'symbol': symbol,
            'limit': limit
        })
    
    # 交易接口
    def place_order(self, symbol: str, side: str, order_type: str, size: str, 
                   price: str = None, client_order_id: str = None) -> Optional[Dict]:
        """下单"""
        data = {
            'symbol': symbol,
            'side': side.lower(),  # buy/sell
            'orderType': order_type.lower(),  # limit/market
            'size': size
        }
        
        if price and order_type.lower() == 'limit':
            data['price'] = price
            
        if client_order_id:
            data['clientOrderId'] = client_order_id
        
        return self._make_request('POST', '/api/spot/v1/trade/orders', data=data)
    
    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """撤单"""
        data = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('POST', '/api/spot/v1/trade/cancel-order', data=data)
    
    def cancel_all_orders(self, symbol: str) -> Optional[Dict]:
        """撤销所有订单"""
        data = {'symbol': symbol}
        return self._make_request('POST', '/api/spot/v1/trade/cancel-symbol-order', data=data)
    
    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """获取订单详情"""
        return self._make_request('GET', '/api/spot/v1/trade/orderInfo', {
            'symbol': symbol,
            'orderId': order_id
        })
    
    def get_open_orders(self, symbol: str = None) -> Optional[List[Dict]]:
        """获取未成交订单"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/spot/v1/trade/open-orders', params)
    
    def get_order_history(self, symbol: str = None, start_time: str = None, 
                         end_time: str = None, limit: int = 100) -> Optional[List[Dict]]:
        """获取历史订单"""
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._make_request('GET', '/api/spot/v1/trade/history', params)
    
    def get_fills(self, symbol: str = None, order_id: str = None, 
                 start_time: str = None, end_time: str = None, limit: int = 100) -> Optional[List[Dict]]:
        """获取成交记录"""
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = order_id
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._make_request('GET', '/api/spot/v1/trade/fills', params)
    
    # WebSocket相关
    def connect_websocket(self):
        """连接WebSocket"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._handle_ws_message(data)
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息失败: {e}")
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket错误: {error}")
                self.ws_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket连接关闭")
                self.ws_connected = False
            
            def on_open(ws):
                self.logger.info("WebSocket连接成功")
                self.ws_connected = True
                # 发送登录消息
                self._ws_login()
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # 在新线程中运行WebSocket
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
            # 等待连接建立
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
    
    def _ws_login(self):
        """WebSocket登录"""
        try:
            timestamp = str(int(time.time()))
            sign = self._generate_signature(timestamp, 'GET', '/user/verify', '')
            
            login_msg = {
                'op': 'login',
                'args': [{
                    'apiKey': self.config.api_key,
                    'passphrase': self.config.passphrase,
                    'timestamp': timestamp,
                    'sign': sign
                }]
            }
            
            self.ws.send(json.dumps(login_msg))
            
        except Exception as e:
            self.logger.error(f"WebSocket登录失败: {e}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable = None):
        """订阅ticker数据"""
        if not self.ws_connected:
            self.connect_websocket()
            time.sleep(2)
        
        channel = f"spot/ticker:{symbol}"
        self.ws_callbacks[channel] = callback
        
        subscribe_msg = {
            'op': 'subscribe',
            'args': [channel]
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"订阅ticker: {symbol}")
    
    def subscribe_orderbook(self, symbol: str, callback: Callable = None):
        """订阅订单簿数据"""
        if not self.ws_connected:
            self.connect_websocket()
            time.sleep(2)
        
        channel = f"spot/depth5:{symbol}"
        self.ws_callbacks[channel] = callback
        
        subscribe_msg = {
            'op': 'subscribe',
            'args': [channel]
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"订阅订单簿: {symbol}")
    
    def subscribe_trades(self, symbol: str, callback: Callable = None):
        """订阅成交数据"""
        if not self.ws_connected:
            self.connect_websocket()
            time.sleep(2)
        
        channel = f"spot/trade:{symbol}"
        self.ws_callbacks[channel] = callback
        
        subscribe_msg = {
            'op': 'subscribe',
            'args': [channel]
        }
        
        self.ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"订阅成交数据: {symbol}")
    
    def _handle_ws_message(self, data: Dict):
        """处理WebSocket消息"""
        try:
            if 'event' in data:
                # 处理事件消息
                if data['event'] == 'login':
                    if data['code'] == '0':
                        self.logger.info("WebSocket登录成功")
                    else:
                        self.logger.error(f"WebSocket登录失败: {data.get('msg')}")
                elif data['event'] == 'subscribe':
                    self.logger.info(f"订阅成功: {data.get('arg')}")
                elif data['event'] == 'error':
                    self.logger.error(f"WebSocket错误: {data.get('msg')}")
            
            elif 'arg' in data and 'data' in data:
                # 处理数据消息
                channel = data['arg']['channel']
                callback = self.ws_callbacks.get(channel)
                
                if callback:
                    callback(data['data'])
                else:
                    self.logger.debug(f"收到数据: {channel}")
                    
        except Exception as e:
            self.logger.error(f"处理WebSocket消息异常: {e}")
    
    def close_websocket(self):
        """关闭WebSocket连接"""
        if self.ws:
            self.ws.close()
            self.ws_connected = False
            self.logger.info("WebSocket连接已关闭")
    
    # 工具方法
    def get_symbols(self) -> Optional[List[Dict]]:
        """获取所有交易对"""
        return self._make_request('GET', '/api/spot/v1/public/symbols')
    
    def get_server_time(self) -> Optional[Dict]:
        """获取服务器时间"""
        return self._make_request('GET', '/api/spot/v1/public/time')
    
    def test_connectivity(self) -> bool:
        """测试连接"""
        try:
            result = self.get_server_time()
            if result:
                self.logger.info("Bitget API连接测试成功")
                return True
            else:
                self.logger.error("Bitget API连接测试失败")
                return False
        except Exception as e:
            self.logger.error(f"连接测试异常: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 配置API
    config = BitgetConfig(
        api_key="your_api_key",
        secret_key="your_secret_key",
        passphrase="your_passphrase",
        sandbox=False
    )
    
    # 创建API实例
    api = BitgetAPI(config)
    
    try:
        # 测试连接
        if api.test_connectivity():
            print("API连接成功")
            
            # 获取账户信息
            account = api.get_account_info()
            print(f"账户信息: {account}")
            
            # 获取ticker
            ticker = api.get_ticker('BTCUSDT')
            print(f"BTC价格: {ticker}")
            
            # WebSocket测试
            def on_ticker(data):
                print(f"实时价格: {data}")
            
            api.subscribe_ticker('BTCUSDT', on_ticker)
            time.sleep(10)
            
        else:
            print("API连接失败")
            
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        api.close_websocket()
