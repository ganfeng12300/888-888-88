"""
🔌 交易所管理器 - 统一多交易所接口管理
支持币安、火币、OKX等多个交易所的统一管理和调度
提供统一的API接口，自动路由到对应的交易所
"""
import asyncio
from typing import Dict, List, Optional, Any, Type
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from .base_exchange import BaseExchange, Symbol, Order, Trade, Balance, Position, Ticker, OrderBook, Kline
from .binance_exchange import BinanceExchange
from .huobi_exchange import HuobiExchange
from .okx_exchange import OKXExchange

class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    HUOBI = "huobi"
    OKX = "okx"

class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self):
        self.exchanges: Dict[str, BaseExchange] = {}
        self.exchange_classes: Dict[ExchangeType, Type[BaseExchange]] = {
            ExchangeType.BINANCE: BinanceExchange,
            ExchangeType.HUOBI: HuobiExchange,
            ExchangeType.OKX: OKXExchange
        }
        
        logger.info("交易所管理器初始化完成")
    
    def add_exchange(self, exchange_type: ExchangeType, api_key: str, 
                    api_secret: str, passphrase: str = None, 
                    sandbox: bool = False, timeout: int = 30) -> bool:
        """添加交易所"""
        try:
            exchange_class = self.exchange_classes.get(exchange_type)
            if not exchange_class:
                logger.error(f"不支持的交易所类型: {exchange_type}")
                return False
            
            if exchange_type == ExchangeType.OKX:
                if not passphrase:
                    logger.error("OKX交易所需要提供passphrase")
                    return False
                exchange = exchange_class(api_key, api_secret, passphrase, sandbox, timeout)
            else:
                exchange = exchange_class(api_key, api_secret, sandbox, timeout)
            
            self.exchanges[exchange_type.value] = exchange
            logger.info(f"添加交易所成功: {exchange_type.value}")
            return True
        
        except Exception as e:
            logger.error(f"添加交易所失败: {e}")
            return False
    
    def get_exchange(self, exchange_type: ExchangeType) -> Optional[BaseExchange]:
        """获取交易所实例"""
        return self.exchanges.get(exchange_type.value)
    
    def remove_exchange(self, exchange_type: ExchangeType) -> bool:
        """移除交易所"""
        try:
            if exchange_type.value in self.exchanges:
                del self.exchanges[exchange_type.value]
                logger.info(f"移除交易所成功: {exchange_type.value}")
                return True
            else:
                logger.warning(f"交易所不存在: {exchange_type.value}")
                return False
        
        except Exception as e:
            logger.error(f"移除交易所失败: {e}")
            return False
    
    def get_available_exchanges(self) -> List[str]:
        """获取可用的交易所列表"""
        return list(self.exchanges.keys())
    
    async def get_all_symbols(self) -> Dict[str, List[Symbol]]:
        """获取所有交易所的交易对信息"""
        results = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                symbols = await exchange.get_symbols()
                results[exchange_name] = symbols
                logger.info(f"获取{exchange_name}交易对: {len(symbols)}个")
            except Exception as e:
                logger.error(f"获取{exchange_name}交易对失败: {e}")
                results[exchange_name] = []
        
        return results
    
    async def get_all_tickers(self, symbols: List[str] = None) -> Dict[str, List[Ticker]]:
        """获取所有交易所的行情数据"""
        results = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                tickers = await exchange.get_tickers(symbols)
                results[exchange_name] = tickers
                logger.info(f"获取{exchange_name}行情: {len(tickers)}个")
            except Exception as e:
                logger.error(f"获取{exchange_name}行情失败: {e}")
                results[exchange_name] = []
        
        return results
    
    async def get_best_price(self, symbol: str, side: str) -> Optional[Dict[str, Any]]:
        """获取最优价格"""
        best_price = None
        best_exchange = None
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.get_ticker(symbol)
                
                if side.lower() == "buy":
                    price = ticker.ask_price  # 买入时看卖一价
                    if best_price is None or price < best_price:
                        best_price = price
                        best_exchange = exchange_name
                else:
                    price = ticker.bid_price  # 卖出时看买一价
                    if best_price is None or price > best_price:
                        best_price = price
                        best_exchange = exchange_name
            
            except Exception as e:
                logger.error(f"获取{exchange_name}价格失败: {e}")
        
        if best_price and best_exchange:
            return {
                "exchange": best_exchange,
                "price": best_price,
                "symbol": symbol,
                "side": side
            }
        
        return None
    
    async def get_aggregated_orderbook(self, symbol: str, limit: int = 100) -> Optional[OrderBook]:
        """获取聚合订单簿"""
        all_bids = []
        all_asks = []
        latest_timestamp = 0
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                orderbook = await exchange.get_orderbook(symbol, limit)
                all_bids.extend(orderbook.bids)
                all_asks.extend(orderbook.asks)
                latest_timestamp = max(latest_timestamp, orderbook.timestamp)
            
            except Exception as e:
                logger.error(f"获取{exchange_name}订单簿失败: {e}")
        
        if all_bids or all_asks:
            # 按价格排序并合并相同价格的订单
            all_bids.sort(key=lambda x: x[0], reverse=True)  # 买单按价格降序
            all_asks.sort(key=lambda x: x[0])  # 卖单按价格升序
            
            # 合并相同价格的订单
            merged_bids = self._merge_orders(all_bids)
            merged_asks = self._merge_orders(all_asks)
            
            return OrderBook(
                symbol=symbol,
                bids=merged_bids[:limit],
                asks=merged_asks[:limit],
                timestamp=latest_timestamp
            )
        
        return None
    
    def _merge_orders(self, orders: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并相同价格的订单"""
        merged = {}
        
        for price, quantity in orders:
            if price in merged:
                merged[price] += quantity
            else:
                merged[price] = quantity
        
        return [(price, quantity) for price, quantity in merged.items()]
    
    async def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有交易所状态"""
        status = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # 测试连接
                symbols = await exchange.get_symbols()
                status[exchange_name] = {
                    "status": "online",
                    "symbol_count": len(symbols),
                    "info": exchange.get_exchange_info()
                }
            except Exception as e:
                status[exchange_name] = {
                    "status": "offline",
                    "error": str(e),
                    "info": exchange.get_exchange_info()
                }
        
        return status
    
    async def cleanup(self):
        """清理所有交易所连接"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.cleanup()
                logger.info(f"清理交易所连接: {exchange_name}")
            except Exception as e:
                logger.error(f"清理交易所连接失败: {exchange_name} - {e}")
        
        self.exchanges.clear()
        logger.info("交易所管理器清理完成")

# 全局交易所管理器实例
exchange_manager = ExchangeManager()

