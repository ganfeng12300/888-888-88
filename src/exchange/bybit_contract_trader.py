#!/usr/bin/env python3
"""
💰 Bybit合约交易器 - 小资金高频策略
Bybit Contract Trader - Small Capital High Frequency Strategy

专注小资金高频交易策略：
- 实时行情数据接入
- 高频订单执行优化
- 智能仓位管理
- 风险控制系统
"""

import asyncio
import json
import time
import hmac
import hashlib
import requests
import websocket
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

from loguru import logger


@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    symbol: str
    side: str  # Buy/Sell
    order_type: str  # Market/Limit
    qty: float
    price: float
    status: str
    filled_qty: float
    avg_price: float
    create_time: datetime
    update_time: datetime


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    side: str  # Buy/Sell/None
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    margin: float
    timestamp: datetime


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    change_24h: float
    funding_rate: float
    open_interest: float
    timestamp: datetime


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: str  # BUY/SELL/HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: datetime


class BybitContractTrader:
    """Bybit合约交易器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化交易器"""
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        
        # API端点
        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        # 交易配置
        self.symbols = config.get('symbols', ['BTCUSDT'])
        self.max_position_size = config.get('max_position_size', 0.1)  # 最大仓位比例
        self.leverage = config.get('leverage', 10)
        self.min_order_size = config.get('min_order_size', 0.001)
        
        # 风控配置
        self.max_daily_loss = config.get('max_daily_loss', 0.03)  # 3%日亏损限制
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%止损
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4%止盈
        
        # 状态管理
        self.is_running = False
        self.positions = {}
        self.orders = {}
        self.market_data = {}
        self.account_balance = 0.0
        self.daily_pnl = 0.0
        self.lock = threading.Lock()
        
        # WebSocket连接
        self.ws = None
        self.ws_thread = None
        
        logger.info("💰 Bybit合约交易器初始化完成")
    
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            # 检查必需的配置项
            required_fields = ['api_key', 'api_secret']
            for field in required_fields:
                if not getattr(self, field, None):
                    logger.error(f"缺少必需配置: {field}")
                    return False
            
            # 检查交易对配置
            if not self.symbols or not isinstance(self.symbols, list):
                logger.error("交易对配置无效")
                return False
            
            # 检查数值配置
            if self.max_position_size <= 0 or self.max_position_size > 1:
                logger.error("最大仓位比例配置无效")
                return False
            
            if self.leverage <= 0 or self.leverage > 100:
                logger.error("杠杆配置无效")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证异常: {str(e)}")
            return False
    
    def connect(self) -> bool:
        """连接到交易所"""
        try:
            # 测试API连接
            test_result = self._make_request("GET", "/v5/market/time")
            
            if test_result.get("retCode") == 0:
                logger.info("✅ Bybit API连接成功")
                self.is_running = True
                
                # 启动WebSocket连接
                self._start_websocket()
                
                return True
            else:
                logger.error(f"❌ Bybit API连接失败: {test_result.get('retMsg', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"连接异常: {str(e)}")
            return False
    
    def check_connection_status(self) -> Dict[str, Any]:
        """检查连接状态"""
        try:
            # 测试API连接
            test_result = self._make_request("GET", "/v5/market/time")
            
            api_connected = test_result.get("retCode") == 0
            
            # 检查WebSocket连接状态
            ws_connected = self.ws is not None and hasattr(self.ws, 'sock') and self.ws.sock is not None
            
            return {
                'api_connected': api_connected,
                'websocket_connected': ws_connected,
                'is_running': self.is_running,
                'last_check': datetime.now().isoformat(),
                'server_time': test_result.get('result', {}).get('timeSecond', 0) if api_connected else 0
            }
            
        except Exception as e:
            return {
                'api_connected': False,
                'websocket_connected': False,
                'is_running': False,
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def validate_order_params(self, order: Dict[str, Any]) -> bool:
        """验证订单参数"""
        try:
            # 检查必需字段
            required_fields = ['symbol', 'side', 'order_type', 'qty']
            for field in required_fields:
                if field not in order:
                    logger.error(f"订单缺少必需字段: {field}")
                    return False
            
            # 检查交易对
            if order['symbol'] not in self.symbols:
                logger.error(f"不支持的交易对: {order['symbol']}")
                return False
            
            # 检查方向
            if order['side'] not in ['Buy', 'Sell']:
                logger.error(f"无效的订单方向: {order['side']}")
                return False
            
            # 检查订单类型
            if order['order_type'] not in ['Market', 'Limit']:
                logger.error(f"无效的订单类型: {order['order_type']}")
                return False
            
            # 检查数量
            qty = float(order['qty'])
            if qty <= 0:
                logger.error(f"无效的订单数量: {qty}")
                return False
            
            if qty < self.min_order_size:
                logger.error(f"订单数量小于最小值: {qty} < {self.min_order_size}")
                return False
            
            # 检查限价单价格
            if order['order_type'] == 'Limit':
                if 'price' not in order or float(order['price']) <= 0:
                    logger.error("限价单缺少有效价格")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"订单参数验证异常: {str(e)}")
            return False
    
    def check_order_risk(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """检查订单风险"""
        try:
            symbol = order['symbol']
            side = order['side']
            qty = float(order['qty'])
            
            # 获取当前仓位
            current_position = self.positions.get(symbol, {})
            current_size = float(current_position.get('size', 0))
            
            # 计算新仓位大小
            if side == 'Buy':
                new_size = current_size + qty
            else:
                new_size = current_size - qty
            
            # 检查仓位限制
            position_ratio = abs(new_size) * float(order.get('price', 50000)) / max(self.account_balance, 1000)
            
            risk_checks = {
                'position_size_ok': position_ratio <= self.max_position_size,
                'daily_loss_ok': self.daily_pnl > -self.max_daily_loss * self.account_balance,
                'balance_sufficient': self.account_balance > 100,  # 最小余额检查
                'position_ratio': position_ratio,
                'current_daily_pnl': self.daily_pnl,
                'account_balance': self.account_balance
            }
            
            # 综合风险评估
            all_checks_passed = all([
                risk_checks['position_size_ok'],
                risk_checks['daily_loss_ok'],
                risk_checks['balance_sufficient']
            ])
            
            return {
                'allowed': all_checks_passed,
                'risk_level': 'low' if all_checks_passed else 'high',
                'checks': risk_checks,
                'warnings': self._generate_risk_warnings(risk_checks)
            }
            
        except Exception as e:
            logger.error(f"风险检查异常: {str(e)}")
            return {
                'allowed': False,
                'risk_level': 'high',
                'error': str(e),
                'checks': {},
                'warnings': ['风险检查系统异常']
            }
    
    def _generate_risk_warnings(self, risk_checks: Dict[str, Any]) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        if not risk_checks['position_size_ok']:
            warnings.append(f"仓位比例过大: {risk_checks['position_ratio']:.2%} > {self.max_position_size:.2%}")
        
        if not risk_checks['daily_loss_ok']:
            warnings.append(f"日亏损超限: {risk_checks['current_daily_pnl']:.2f}")
        
        if not risk_checks['balance_sufficient']:
            warnings.append(f"账户余额不足: {risk_checks['account_balance']:.2f}")
        
        return warnings
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场数据"""
        try:
            symbol = market_data.get('symbol')
            if not symbol:
                return {'status': 'failed', 'error': '缺少交易对信息'}
            
            # 更新市场数据缓存
            self.market_data[symbol] = {
                'price': float(market_data.get('price', 0)),
                'volume': float(market_data.get('volume', 0)),
                'timestamp': market_data.get('timestamp', time.time()),
                'last_update': datetime.now().isoformat()
            }
            
            # 计算技术指标
            processed_data = {
                'symbol': symbol,
                'current_price': self.market_data[symbol]['price'],
                'volume': self.market_data[symbol]['volume'],
                'price_change': self._calculate_price_change(symbol),
                'volatility': self._calculate_volatility(symbol),
                'trend': self._determine_trend(symbol),
                'liquidity_score': self._calculate_liquidity_score(symbol),
                'timestamp': self.market_data[symbol]['timestamp']
            }
            
            return {
                'status': 'success',
                'data': processed_data,
                'raw_data': market_data
            }
            
        except Exception as e:
            logger.error(f"市场数据处理异常: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'raw_data': market_data
            }
    
    def _calculate_price_change(self, symbol: str) -> float:
        """计算价格变化"""
        # 简化版本，实际应该基于历史数据
        current_data = self.market_data.get(symbol, {})
        current_price = current_data.get('price', 0)
        
        # 模拟价格变化计算
        return 0.0  # 需要历史数据支持
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算波动率"""
        # 简化版本，实际应该基于历史价格数据
        return 0.02  # 2%的默认波动率
    
    def _determine_trend(self, symbol: str) -> str:
        """判断趋势"""
        # 简化版本，实际应该基于技术指标
        return 'neutral'
    
    def _calculate_liquidity_score(self, symbol: str) -> float:
        """计算流动性评分"""
        current_data = self.market_data.get(symbol, {})
        volume = current_data.get('volume', 0)
        
        # 基于成交量的简单流动性评分
        if volume > 1000000:
            return 0.9
        elif volume > 100000:
            return 0.7
        elif volume > 10000:
            return 0.5
        else:
            return 0.3
    
    def analyze_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """分析仓位"""
        try:
            symbol = position.get('symbol')
            size = float(position.get('size', 0))
            entry_price = float(position.get('entry_price', 0))
            current_price = self.market_data.get(symbol, {}).get('price', entry_price)
            
            # 计算盈亏
            if size > 0:  # 多头仓位
                unrealized_pnl = (current_price - entry_price) * size
                pnl_percentage = (current_price - entry_price) / entry_price * 100
            elif size < 0:  # 空头仓位
                unrealized_pnl = (entry_price - current_price) * abs(size)
                pnl_percentage = (entry_price - current_price) / entry_price * 100
            else:
                unrealized_pnl = 0
                pnl_percentage = 0
            
            # 计算风险指标
            position_value = abs(size) * current_price
            account_ratio = position_value / max(self.account_balance, 1000)
            
            # 计算止损止盈价格
            if size > 0:  # 多头
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                take_profit_price = entry_price * (1 + self.take_profit_pct)
            else:  # 空头
                stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                take_profit_price = entry_price * (1 - self.take_profit_pct)
            
            analysis = {
                'symbol': symbol,
                'position_size': size,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'pnl_percentage': pnl_percentage,
                'position_value': position_value,
                'account_ratio': account_ratio,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_level': 'high' if account_ratio > 0.5 else 'medium' if account_ratio > 0.2 else 'low',
                'recommendations': self._generate_position_recommendations(
                    pnl_percentage, account_ratio, symbol
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"仓位分析异常: {str(e)}")
            return {
                'symbol': position.get('symbol', 'unknown'),
                'error': str(e),
                'analysis_failed': True
            }
    
    def _generate_position_recommendations(self, pnl_pct: float, account_ratio: float, symbol: str) -> List[str]:
        """生成仓位建议"""
        recommendations = []
        
        if pnl_pct < -self.stop_loss_pct * 100:
            recommendations.append("建议执行止损，避免进一步亏损")
        elif pnl_pct > self.take_profit_pct * 100:
            recommendations.append("建议部分止盈，锁定利润")
        
        if account_ratio > 0.5:
            recommendations.append("仓位过重，建议减仓降低风险")
        elif account_ratio < 0.1:
            recommendations.append("仓位较轻，可考虑适当加仓")
        
        # 基于市场数据的建议
        market_data = self.market_data.get(symbol, {})
        if market_data.get('volume', 0) < 10000:
            recommendations.append("流动性较低，注意滑点风险")
        
        return recommendations if recommendations else ["仓位状态正常，继续监控"]
    
    def _start_websocket(self):
        """启动WebSocket连接"""
        try:
            if self.ws_thread and self.ws_thread.is_alive():
                return
            
            self.ws_thread = threading.Thread(
                target=self._websocket_worker,
                daemon=True,
                name="BybitWebSocketThread"
            )
            self.ws_thread.start()
            
            logger.info("WebSocket连接线程已启动")
            
        except Exception as e:
            logger.error(f"启动WebSocket失败: {str(e)}")
    
    def _websocket_worker(self):
        """WebSocket工作线程"""
        try:
            import websocket
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._handle_websocket_message(data)
                except Exception as e:
                    logger.error(f"WebSocket消息处理失败: {str(e)}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket错误: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.warning("WebSocket连接已关闭")
            
            def on_open(ws):
                logger.info("WebSocket连接已建立")
                # 订阅市场数据
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"orderbook.1.{symbol}" for symbol in self.symbols]
                }
                ws.send(json.dumps(subscribe_msg))
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            self.ws.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket工作线程异常: {str(e)}")
    
    def _handle_websocket_message(self, data: Dict[str, Any]):
        """处理WebSocket消息"""
        try:
            topic = data.get('topic', '')
            
            if 'orderbook' in topic:
                # 处理订单簿数据
                symbol = topic.split('.')[-1]
                orderbook_data = data.get('data', {})
                
                if 'b' in orderbook_data and 'a' in orderbook_data:
                    bids = orderbook_data['b']
                    asks = orderbook_data['a']
                    
                    if bids and asks:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        mid_price = (best_bid + best_ask) / 2
                        
                        # 更新市场数据
                        self.market_data[symbol] = {
                            'price': mid_price,
                            'bid': best_bid,
                            'ask': best_ask,
                            'spread': best_ask - best_bid,
                            'timestamp': time.time(),
                            'last_update': datetime.now().isoformat()
                        }
            
        except Exception as e:
            logger.error(f"WebSocket消息处理异常: {str(e)}")
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """生成API签名"""
        param_str = timestamp + self.api_key + "5000" + params
        return hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """发送API请求"""
        if params is None:
            params = {}
        
        timestamp = str(int(time.time() * 1000))
        
        if method == "GET":
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        else:
            param_str = json.dumps(params) if params else ""
        
        signature = self._generate_signature(param_str, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"API请求失败: {method} {endpoint} - {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            result = self._make_request("GET", "/v5/account/wallet-balance", {
                "accountType": "UNIFIED"
            })
            
            if result.get("retCode") == 0:
                wallet_list = result.get("result", {}).get("list", [])
                if wallet_list:
                    wallet = wallet_list[0]
                    coins = wallet.get("coin", [])
                    
                    for coin in coins:
                        if coin.get("coin") == "USDT":
                            self.account_balance = float(coin.get("walletBalance", 0))
                            break
                    
                    return {
                        "balance": self.account_balance,
                        "available": float(wallet.get("totalAvailableBalance", 0)),
                        "margin": float(wallet.get("totalMarginBalance", 0)),
                        "unrealized_pnl": float(wallet.get("totalUnrealisedPnl", 0))
                    }
            
            logger.error(f"获取账户信息失败: {result}")
            return {}
            
        except Exception as e:
            logger.error(f"获取账户信息异常: {e}")
            return {}
    
    async def get_positions(self) -> List[PositionInfo]:
        """获取持仓信息"""
        try:
            result = self._make_request("GET", "/v5/position/list", {
                "category": "linear",
                "settleCoin": "USDT"
            })
            
            positions = []
            
            if result.get("retCode") == 0:
                position_list = result.get("result", {}).get("list", [])
                
                for pos in position_list:
                    if float(pos.get("size", 0)) > 0:  # 只返回有持仓的
                        position = PositionInfo(
                            symbol=pos.get("symbol", ""),
                            side=pos.get("side", ""),
                            size=float(pos.get("size", 0)),
                            entry_price=float(pos.get("avgPrice", 0)),
                            mark_price=float(pos.get("markPrice", 0)),
                            unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                            realized_pnl=float(pos.get("cumRealisedPnl", 0)),
                            leverage=int(pos.get("leverage", 1)),
                            margin=float(pos.get("positionIM", 0)),
                            timestamp=datetime.now()
                        )
                        positions.append(position)
                        
                        # 更新本地持仓记录
                        with self.lock:
                            self.positions[position.symbol] = position
            
            return positions
            
        except Exception as e:
            logger.error(f"获取持仓信息异常: {e}")
            return []
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取市场数据"""
        try:
            # 获取ticker数据
            result = self._make_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": symbol
            })
            
            if result.get("retCode") == 0:
                ticker_list = result.get("result", {}).get("list", [])
                if ticker_list:
                    ticker = ticker_list[0]
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(ticker.get("lastPrice", 0)),
                        bid_price=float(ticker.get("bid1Price", 0)),
                        ask_price=float(ticker.get("ask1Price", 0)),
                        volume_24h=float(ticker.get("volume24h", 0)),
                        change_24h=float(ticker.get("price24hPcnt", 0)) * 100,
                        funding_rate=0.0,  # 需要单独获取
                        open_interest=0.0,  # 需要单独获取
                        timestamp=datetime.now()
                    )
                    
                    # 更新本地市场数据
                    with self.lock:
                        self.market_data[symbol] = market_data
                    
                    return market_data
            
            logger.error(f"获取市场数据失败: {result}")
            return None
            
        except Exception as e:
            logger.error(f"获取市场数据异常: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, qty: float, 
                         order_type: str = "Market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> Optional[str]:
        """下单"""
        try:
            # 构建订单参数
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": "GTC"
            }
            
            if order_type == "Limit" and price:
                order_params["price"] = str(price)
            
            # 设置止损止盈
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
            
            result = self._make_request("POST", "/v5/order/create", order_params)
            
            if result.get("retCode") == 0:
                order_id = result.get("result", {}).get("orderId", "")
                logger.info(f"订单创建成功: {symbol} {side} {qty} @ {price or 'Market'}")
                return order_id
            else:
                logger.error(f"订单创建失败: {result}")
                return None
                
        except Exception as e:
            logger.error(f"下单异常: {e}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        try:
            result = self._make_request("POST", "/v5/order/cancel", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            
            if result.get("retCode") == 0:
                logger.info(f"订单取消成功: {order_id}")
                return True
            else:
                logger.error(f"订单取消失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"取消订单异常: {e}")
            return False
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """设置杠杆"""
        try:
            result = self._make_request("POST", "/v5/position/set-leverage", {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            })
            
            if result.get("retCode") == 0:
                logger.info(f"杠杆设置成功: {symbol} {leverage}x")
                return True
            else:
                logger.error(f"杠杆设置失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"设置杠杆异常: {e}")
            return False
    
    def _start_websocket(self):
        """启动WebSocket连接"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_websocket_message(data)
            except Exception as e:
                logger.error(f"WebSocket消息处理失败: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning("WebSocket连接关闭")
        
        def on_open(ws):
            logger.info("WebSocket连接已建立")
            # 订阅市场数据
            for symbol in self.symbols:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"tickers.{symbol}"]
                }
                ws.send(json.dumps(subscribe_msg))
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws.run_forever()
    
    def _handle_websocket_message(self, data: Dict):
        """处理WebSocket消息"""
        try:
            topic = data.get("topic", "")
            
            if topic.startswith("tickers."):
                # 处理ticker数据
                ticker_data = data.get("data", {})
                if ticker_data:
                    symbol = ticker_data.get("symbol", "")
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(ticker_data.get("lastPrice", 0)),
                        bid_price=float(ticker_data.get("bid1Price", 0)),
                        ask_price=float(ticker_data.get("ask1Price", 0)),
                        volume_24h=float(ticker_data.get("volume24h", 0)),
                        change_24h=float(ticker_data.get("price24hPcnt", 0)) * 100,
                        funding_rate=0.0,
                        open_interest=0.0,
                        timestamp=datetime.now()
                    )
                    
                    with self.lock:
                        self.market_data[symbol] = market_data
                    
                    logger.debug(f"更新市场数据: {symbol} @ {market_data.price}")
            
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")
    
    async def execute_trading_signal(self, signal: TradingSignal) -> bool:
        """执行交易信号"""
        try:
            if signal.action == "HOLD":
                return True
            
            # 检查风控
            if not self._check_risk_limits():
                logger.warning("风控检查未通过，跳过交易信号")
                return False
            
            # 计算订单数量
            order_qty = self._calculate_order_size(signal)
            if order_qty < self.min_order_size:
                logger.warning(f"订单数量过小: {order_qty}")
                return False
            
            # 执行订单
            side = "Buy" if signal.action == "BUY" else "Sell"
            
            order_id = await self.place_order(
                symbol=signal.symbol,
                side=side,
                qty=order_qty,
                order_type="Market",
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_id:
                logger.info(f"交易信号执行成功: {signal.symbol} {signal.action} "
                           f"数量={order_qty} 置信度={signal.confidence:.2f}")
                return True
            else:
                logger.error(f"交易信号执行失败: {signal.symbol} {signal.action}")
                return False
                
        except Exception as e:
            logger.error(f"执行交易信号异常: {e}")
            return False
    
    def _check_risk_limits(self) -> bool:
        """检查风控限制"""
        # 检查日亏损限制
        if self.daily_pnl < -self.max_daily_loss * self.account_balance:
            logger.warning(f"达到日亏损限制: {self.daily_pnl:.2f}")
            return False
        
        # 检查账户余额
        if self.account_balance <= 0:
            logger.warning("账户余额不足")
            return False
        
        return True
    
    def _calculate_order_size(self, signal: TradingSignal) -> float:
        """计算订单数量"""
        try:
            # 基于信号强度和账户余额计算
            base_size = signal.position_size * self.account_balance
            
            # 应用最大仓位限制
            max_size = self.max_position_size * self.account_balance
            order_size = min(base_size, max_size)
            
            # 转换为合约数量（简化处理）
            market_data = self.market_data.get(signal.symbol)
            if market_data:
                contract_qty = order_size / market_data.price
                return round(contract_qty, 6)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算订单数量失败: {e}")
            return 0.0
    
    async def start_trading(self):
        """启动交易"""
        self.is_running = True
        
        # 设置杠杆
        for symbol in self.symbols:
            await self.set_leverage(symbol, self.leverage)
        
        # 启动WebSocket
        self.ws_thread = threading.Thread(target=self._start_websocket, daemon=True)
        self.ws_thread.start()
        
        # 启动监控循环
        await self._trading_loop()
    
    async def _trading_loop(self):
        """交易主循环"""
        while self.is_running:
            try:
                # 更新账户信息
                await self.get_account_info()
                
                # 更新持仓信息
                await self.get_positions()
                
                # 更新市场数据
                for symbol in self.symbols:
                    await self.get_market_data(symbol)
                
                # 等待下一次循环
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                await asyncio.sleep(5)
    
    def stop_trading(self):
        """停止交易"""
        logger.info("正在停止交易...")
        self.is_running = False
        
        if self.ws:
            self.ws.close()
        
        logger.info("交易已停止")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """获取交易状态"""
        with self.lock:
            return {
                "is_running": self.is_running,
                "account_balance": self.account_balance,
                "daily_pnl": self.daily_pnl,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "market_data": {k: asdict(v) for k, v in self.market_data.items()},
                "symbols": self.symbols,
                "leverage": self.leverage,
                "max_position_size": self.max_position_size
            }


# 全局实例
_trader = None

def get_bybit_trader(config: Dict[str, Any] = None) -> BybitContractTrader:
    """获取Bybit交易器实例"""
    global _trader
    if _trader is None and config:
        _trader = BybitContractTrader(config)
    return _trader


if __name__ == "__main__":
    # 测试代码
    async def test_bybit_trader():
        """测试Bybit交易器"""
        config = {
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "leverage": 10,
            "max_position_size": 0.1
        }
        
        trader = get_bybit_trader(config)
        
        # 获取账户信息
        account_info = await trader.get_account_info()
        print(f"账户信息: {account_info}")
        
        # 获取市场数据
        market_data = await trader.get_market_data("BTCUSDT")
        print(f"市场数据: {market_data}")
        
        # 获取交易状态
        status = trader.get_trading_status()
        print(f"交易状态: {json.dumps(status, indent=2, default=str)}")
    
    # 运行测试
    asyncio.run(test_bybit_trader())
