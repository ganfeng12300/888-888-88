"""
🔌 交易所基础接口 - 统一交易所接口抽象基类
定义所有交易所必须实现的标准接口，支持现货、期货、期权交易
包含订单管理、账户查询、市场数据、WebSocket连接等核心功能
"""
import asyncio
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import aiohttp
    import websockets
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("aiohttp or websockets not available, async features will be limited")

from loguru import logger

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TAKE_PROFIT = "take_profit"  # 止盈单
    TAKE_PROFIT_LIMIT = "take_profit_limit"  # 止盈限价单
    TRAILING_STOP = "trailing_stop"  # 跟踪止损单

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出

class OrderStatus(Enum):
    """订单状态"""
    NEW = "new"  # 新建
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELED = "canceled"  # 已取消
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期

class TimeInForce(Enum):
    """订单有效期"""
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date

class MarketType(Enum):
    """市场类型"""
    SPOT = "spot"  # 现货
    FUTURES = "futures"  # 期货
    OPTIONS = "options"  # 期权
    MARGIN = "margin"  # 杠杆

@dataclass
class Symbol:
    """交易对信息"""
    symbol: str  # 交易对符号
    base_asset: str  # 基础资产
    quote_asset: str  # 计价资产
    market_type: MarketType  # 市场类型
    min_qty: float  # 最小数量
    max_qty: float  # 最大数量
    step_size: float  # 数量步长
    min_price: float  # 最小价格
    max_price: float  # 最大价格
    tick_size: float  # 价格步长
    min_notional: float  # 最小名义价值
    is_active: bool  # 是否活跃

@dataclass
class Order:
    """订单信息"""
    order_id: str  # 订单ID
    client_order_id: str  # 客户端订单ID
    symbol: str  # 交易对
    side: OrderSide  # 订单方向
    order_type: OrderType  # 订单类型
    quantity: float  # 数量
    price: Optional[float]  # 价格
    stop_price: Optional[float]  # 止损价格
    time_in_force: TimeInForce  # 有效期
    status: OrderStatus  # 订单状态
    filled_qty: float  # 已成交数量
    remaining_qty: float  # 剩余数量
    avg_price: float  # 平均成交价格
    commission: float  # 手续费
    commission_asset: str  # 手续费资产
    created_at: float  # 创建时间
    updated_at: float  # 更新时间

@dataclass
class Trade:
    """成交记录"""
    trade_id: str  # 成交ID
    order_id: str  # 订单ID
    symbol: str  # 交易对
    side: OrderSide  # 方向
    quantity: float  # 数量
    price: float  # 价格
    commission: float  # 手续费
    commission_asset: str  # 手续费资产
    is_maker: bool  # 是否为挂单方
    timestamp: float  # 时间戳

@dataclass
class Balance:
    """账户余额"""
    asset: str  # 资产
    free: float  # 可用余额
    locked: float  # 冻结余额
    total: float  # 总余额

@dataclass
class Position:
    """持仓信息"""
    symbol: str  # 交易对
    side: str  # 方向 (long/short)
    size: float  # 持仓数量
    entry_price: float  # 开仓价格
    mark_price: float  # 标记价格
    unrealized_pnl: float  # 未实现盈亏
    realized_pnl: float  # 已实现盈亏
    margin: float  # 保证金
    leverage: float  # 杠杆倍数
    liquidation_price: float  # 强平价格

@dataclass
class Ticker:
    """行情数据"""
    symbol: str  # 交易对
    price: float  # 最新价格
    bid_price: float  # 买一价
    ask_price: float  # 卖一价
    bid_qty: float  # 买一量
    ask_qty: float  # 卖一量
    volume_24h: float  # 24小时成交量
    price_change_24h: float  # 24小时价格变化
    price_change_percent_24h: float  # 24小时价格变化百分比
    high_24h: float  # 24小时最高价
    low_24h: float  # 24小时最低价
    timestamp: float  # 时间戳

@dataclass
class OrderBook:
    """订单簿"""
    symbol: str  # 交易对
    bids: List[Tuple[float, float]]  # 买单 [(价格, 数量)]
    asks: List[Tuple[float, float]]  # 卖单 [(价格, 数量)]
    timestamp: float  # 时间戳

@dataclass
class Kline:
    """K线数据"""
    symbol: str  # 交易对
    interval: str  # 时间间隔
    open_time: float  # 开盘时间
    close_time: float  # 收盘时间
    open_price: float  # 开盘价
    high_price: float  # 最高价
    low_price: float  # 最低价
    close_price: float  # 收盘价
    volume: float  # 成交量
    quote_volume: float  # 成交额
    trade_count: int  # 成交笔数

class BaseExchange(ABC):
    """交易所基础接口"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str = None, 
                 sandbox: bool = False, timeout: int = 30):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self.timeout = timeout
        
        # 基础配置
        self.base_url = ""
        self.ws_url = ""
        self.name = ""
        
        # 连接管理
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, Any] = {}
        
        # 限流管理
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # 订阅管理
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        logger.info(f"初始化交易所接口: {self.name}")
    
    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易所信息"""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[Symbol]:
        """获取所有交易对信息"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """获取单个交易对行情"""
        pass
    
    @abstractmethod
    async def get_tickers(self, symbols: List[str] = None) -> List[Ticker]:
        """获取多个交易对行情"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, start_time: int = None, 
                        end_time: int = None, limit: int = 500) -> List[Kline]:
        """获取K线数据"""
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 500) -> List[Trade]:
        """获取最近成交记录"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: float = None, stop_price: float = None,
                         time_in_force: TimeInForce = TimeInForce.GTC,
                         client_order_id: str = None) -> Order:
        """下单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str = None, 
                          client_order_id: str = None) -> bool:
        """撤单"""
        pass
    
    @abstractmethod
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """撤销所有订单"""
        pass
    
    @abstractmethod
    async def get_order(self, symbol: str, order_id: str = None, 
                       client_order_id: str = None) -> Order:
        """查询订单"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """查询活跃订单"""
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Order]:
        """查询历史订单"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        pass
    
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """获取账户余额"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """获取持仓信息（期货）"""
        pass
    
    @abstractmethod
    async def get_trade_history(self, symbol: str = None, start_time: int = None,
                               end_time: int = None, limit: int = 500) -> List[Trade]:
        """获取成交历史"""
        pass
    
    # WebSocket相关方法
    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅行情数据"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """订阅订单簿数据"""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """订阅成交数据"""
        pass
    
    @abstractmethod
    async def subscribe_klines(self, symbol: str, interval: str, callback: Callable):
        """订阅K线数据"""
        pass
    
    @abstractmethod
    async def subscribe_user_data(self, callback: Callable):
        """订阅用户数据（订单、余额等）"""
        pass
    
    # 工具方法
    def generate_signature(self, params: str, timestamp: str = None) -> str:
        """生成签名"""
        if timestamp is None:
            timestamp = str(int(time.time() * 1000))
        
        message = timestamp + params
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def generate_client_order_id(self) -> str:
        """生成客户端订单ID"""
        return f"{self.name}_{int(time.time() * 1000000)}"
    
    async def create_session(self):
        """创建HTTP会话"""
        if not ASYNC_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, params: Dict = None,
                          data: Dict = None, headers: Dict = None, 
                          signed: bool = False) -> Dict:
        """发送HTTP请求"""
        await self.create_session()
        
        url = self.base_url + endpoint
        
        if headers is None:
            headers = {}
        
        if signed:
            headers.update(self.get_auth_headers(params, data))
        
        try:
            async with self.session.request(
                method, url, params=params, json=data, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API请求失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
        
        except Exception as e:
            logger.error(f"HTTP请求异常: {e}")
            raise
    
    @abstractmethod
    def get_auth_headers(self, params: Dict = None, data: Dict = None) -> Dict[str, str]:
        """获取认证头"""
        pass
    
    def format_symbol(self, symbol: str) -> str:
        """格式化交易对符号"""
        return symbol.upper()
    
    def parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """解析交易对符号"""
        # 默认实现，子类可以重写
        if 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            quote = 'USDT'
        elif 'BTC' in symbol:
            base = symbol.replace('BTC', '')
            quote = 'BTC'
        elif 'ETH' in symbol:
            base = symbol.replace('ETH', '')
            quote = 'ETH'
        else:
            # 简单分割
            if len(symbol) >= 6:
                base = symbol[:3]
                quote = symbol[3:]
            else:
                base = symbol
                quote = 'USDT'
        
        return base, quote
    
    def normalize_price(self, price: float, tick_size: float) -> float:
        """标准化价格"""
        if tick_size == 0:
            return price
        return round(price / tick_size) * tick_size
    
    def normalize_quantity(self, quantity: float, step_size: float) -> float:
        """标准化数量"""
        if step_size == 0:
            return quantity
        return round(quantity / step_size) * step_size
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 关闭WebSocket连接
            for ws in self.ws_connections.values():
                if hasattr(ws, 'close'):
                    await ws.close()
            self.ws_connections.clear()
            
            # 关闭HTTP会话
            await self.close_session()
            
            logger.info(f"交易所接口清理完成: {self.name}")
        except Exception as e:
            logger.error(f"清理交易所接口失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # 在事件循环中清理
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
            except:
                pass

