#!/usr/bin/env python3
"""
💰 真实交易管理器
管理真实实盘交易数据、持仓、历史记录
"""

import asyncio
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from loguru import logger
import sqlite3
from pathlib import Path

from src.config.api_config_manager import APIConfigManager


@dataclass
class RealPosition:
    """真实持仓信息"""
    symbol: str
    side: str  # long/short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    percentage: float  # 持仓占比
    leverage: float
    margin_used: float
    liquidation_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RealTrade:
    """真实交易记录"""
    id: str
    symbol: str
    side: str  # buy/sell
    amount: float
    price: float
    cost: float
    fee: float
    timestamp: datetime
    order_type: str = "market"
    status: str = "closed"


@dataclass
class AccountInfo:
    """账户信息"""
    total_balance: float
    available_balance: float
    used_balance: float
    total_pnl: float
    daily_pnl: float
    positions_count: int
    leverage_ratio: float
    margin_ratio: float
    currencies: Dict[str, float] = field(default_factory=dict)


class RealTradingManager:
    """真实交易管理器"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self.api_config = APIConfigManager()
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.positions: Dict[str, RealPosition] = {}
        self.trades_history: List[RealTrade] = []
        self.account_info: Optional[AccountInfo] = None
        
        # 初始化数据库
        self.init_database()
        
        logger.info("💰 真实交易管理器初始化完成")
    
    def init_database(self) -> None:
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建持仓表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                percentage REAL NOT NULL,
                leverage REAL NOT NULL,
                margin_used REAL NOT NULL,
                liquidation_price REAL,
                timestamp INTEGER NOT NULL,
                UNIQUE(symbol, side)
            )
        """)
        
        # 创建交易历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                cost REAL NOT NULL,
                fee REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                order_type TEXT DEFAULT 'market',
                status TEXT DEFAULT 'closed'
            )
        """)
        
        # 创建账户信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_balance REAL NOT NULL,
                available_balance REAL NOT NULL,
                used_balance REAL NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                positions_count INTEGER NOT NULL,
                leverage_ratio REAL NOT NULL,
                margin_ratio REAL NOT NULL,
                currencies TEXT,
                timestamp INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def initialize_exchanges(self) -> bool:
        """初始化交易所连接"""
        try:
            # 获取配置的交易所
            exchanges = self.api_config.list_configured_exchanges()
            
            for exchange_name in exchanges:
                config = self.api_config.get_exchange_config(exchange_name)
                if not config:
                    continue
                
                if exchange_name == "bitget":
                    exchange = ccxt.bitget({
                        'apiKey': config.api_key,
                        'secret': config.api_secret,
                        'password': config.passphrase,
                        'sandbox': config.sandbox,
                        'enableRateLimit': True,
                    })
                elif exchange_name == "binance":
                    exchange = ccxt.binance({
                        'apiKey': config.api_key,
                        'secret': config.api_secret,
                        'sandbox': config.sandbox,
                        'enableRateLimit': True,
                    })
                else:
                    continue
                
                # 测试连接
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, exchange.fetch_balance
                    )
                    self.exchanges[exchange_name] = exchange
                    logger.info(f"✅ {exchange_name} 连接成功")
                except Exception as e:
                    logger.error(f"❌ {exchange_name} 连接失败: {e}")
            
            return len(self.exchanges) > 0
            
        except Exception as e:
            logger.error(f"❌ 初始化交易所失败: {e}")
            return False
    
    async def fetch_account_info(self) -> Optional[AccountInfo]:
        """获取账户信息"""
        try:
            if not self.exchanges:
                return None
            
            # 使用第一个可用的交易所
            exchange_name = list(self.exchanges.keys())[0]
            exchange = self.exchanges[exchange_name]
            
            # 获取余额信息
            balance = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_balance
            )
            
            total_balance = balance.get('total', {}).get('USDT', 0.0)
            free_balance = balance.get('free', {}).get('USDT', 0.0)
            used_balance = balance.get('used', {}).get('USDT', 0.0)
            
            # 计算持仓相关数据
            positions_count = len(self.positions)
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
            
            # 计算杠杆比率
            leverage_ratio = used_balance / total_balance if total_balance > 0 else 0.0
            margin_ratio = used_balance / total_balance if total_balance > 0 else 0.0
            
            # 获取所有币种余额
            currencies = {}
            for currency, amounts in balance.get('total', {}).items():
                if amounts > 0:
                    currencies[currency] = amounts
            
            self.account_info = AccountInfo(
                total_balance=total_balance,
                available_balance=free_balance,
                used_balance=used_balance,
                total_pnl=total_pnl,
                daily_pnl=0.0,  # 需要计算当日盈亏
                positions_count=positions_count,
                leverage_ratio=leverage_ratio,
                margin_ratio=margin_ratio,
                currencies=currencies
            )
            
            # 保存到数据库
            self._save_account_snapshot()
            
            logger.info(f"💰 账户信息更新: 总余额 ${total_balance:.2f}")
            return self.account_info
            
        except Exception as e:
            logger.error(f"❌ 获取账户信息失败: {e}")
            return None
    
    async def fetch_positions(self) -> Dict[str, RealPosition]:
        """获取真实持仓"""
        try:
            if not self.exchanges:
                return {}
            
            exchange_name = list(self.exchanges.keys())[0]
            exchange = self.exchanges[exchange_name]
            
            # 获取持仓信息
            positions = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_positions
            )
            
            self.positions.clear()
            
            for pos in positions:
                if pos['contracts'] == 0:  # 跳过空持仓
                    continue
                
                symbol = pos['symbol']
                side = 'long' if pos['side'] == 'long' else 'short'
                size = abs(pos['contracts'])
                entry_price = pos['entryPrice'] or 0.0
                current_price = pos['markPrice'] or 0.0
                unrealized_pnl = pos['unrealizedPnl'] or 0.0
                percentage = pos['percentage'] or 0.0
                leverage = pos['leverage'] or 1.0
                margin_used = pos['initialMargin'] or 0.0
                liquidation_price = pos['liquidationPrice']
                
                real_position = RealPosition(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0,
                    percentage=percentage,
                    leverage=leverage,
                    margin_used=margin_used,
                    liquidation_price=liquidation_price
                )
                
                self.positions[f"{symbol}_{side}"] = real_position
            
            # 保存到数据库
            self._save_positions()
            
            logger.info(f"📊 持仓信息更新: {len(self.positions)} 个持仓")
            return self.positions
            
        except Exception as e:
            logger.error(f"❌ 获取持仓信息失败: {e}")
            return {}
    
    async def fetch_trades_history(self, days: int = 7) -> List[RealTrade]:
        """获取交易历史"""
        try:
            if not self.exchanges:
                return []
            
            exchange_name = list(self.exchanges.keys())[0]
            exchange = self.exchanges[exchange_name]
            
            # 获取最近的交易记录
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            trades = await asyncio.get_event_loop().run_in_executor(
                None, lambda: exchange.fetch_my_trades(since=since)
            )
            
            self.trades_history.clear()
            
            for trade in trades:
                real_trade = RealTrade(
                    id=trade['id'],
                    symbol=trade['symbol'],
                    side=trade['side'],
                    amount=trade['amount'],
                    price=trade['price'],
                    cost=trade['cost'],
                    fee=trade['fee']['cost'] if trade['fee'] else 0.0,
                    timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                    order_type=trade.get('type', 'market'),
                    status='closed'
                )
                
                self.trades_history.append(real_trade)
            
            # 保存到数据库
            self._save_trades()
            
            logger.info(f"📈 交易历史更新: {len(self.trades_history)} 笔交易")
            return self.trades_history
            
        except Exception as e:
            logger.error(f"❌ 获取交易历史失败: {e}")
            return []
    
    def _save_positions(self) -> None:
        """保存持仓到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 清空现有持仓
            cursor.execute("DELETE FROM positions")
            
            # 插入新持仓
            for position in self.positions.values():
                cursor.execute("""
                    INSERT INTO positions 
                    (symbol, side, size, entry_price, current_price, unrealized_pnl, 
                     realized_pnl, percentage, leverage, margin_used, liquidation_price, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.symbol, position.side, position.size, position.entry_price,
                    position.current_price, position.unrealized_pnl, position.realized_pnl,
                    position.percentage, position.leverage, position.margin_used,
                    position.liquidation_price, int(position.timestamp.timestamp())
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存持仓失败: {e}")
    
    def _save_trades(self) -> None:
        """保存交易历史到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for trade in self.trades_history:
                cursor.execute("""
                    INSERT OR REPLACE INTO trades 
                    (id, symbol, side, amount, price, cost, fee, timestamp, order_type, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.symbol, trade.side, trade.amount, trade.price,
                    trade.cost, trade.fee, int(trade.timestamp.timestamp()),
                    trade.order_type, trade.status
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存交易历史失败: {e}")
    
    def _save_account_snapshot(self) -> None:
        """保存账户快照"""
        try:
            if not self.account_info:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO account_snapshots 
                (total_balance, available_balance, used_balance, total_pnl, daily_pnl,
                 positions_count, leverage_ratio, margin_ratio, currencies, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.account_info.total_balance, self.account_info.available_balance,
                self.account_info.used_balance, self.account_info.total_pnl,
                self.account_info.daily_pnl, self.account_info.positions_count,
                self.account_info.leverage_ratio, self.account_info.margin_ratio,
                json.dumps(self.account_info.currencies), int(datetime.now().timestamp())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存账户快照失败: {e}")
    
    async def update_all_data(self) -> Dict[str, Any]:
        """更新所有数据"""
        try:
            # 并发获取所有数据
            tasks = [
                self.fetch_account_info(),
                self.fetch_positions(),
                self.fetch_trades_history()
            ]
            
            account_info, positions, trades = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                'account_info': account_info if not isinstance(account_info, Exception) else None,
                'positions': positions if not isinstance(positions, Exception) else {},
                'trades': trades if not isinstance(trades, Exception) else [],
                'update_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 更新数据失败: {e}")
            return {}
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """获取交易摘要"""
        try:
            # 计算统计数据
            total_trades = len(self.trades_history)
            winning_trades = sum(1 for trade in self.trades_history 
                               if trade.side == 'sell' and trade.cost > 0)  # 简化的盈利判断
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_volume = sum(trade.cost for trade in self.trades_history)
            total_fees = sum(trade.fee for trade in self.trades_history)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_volume': total_volume,
                'total_fees': total_fees,
                'active_positions': len(self.positions),
                'account_balance': self.account_info.total_balance if self.account_info else 0.0,
                'total_pnl': self.account_info.total_pnl if self.account_info else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ 获取交易摘要失败: {e}")
            return {}


# 全局实例
_real_trading_manager = None

def get_real_trading_manager() -> RealTradingManager:
    """获取真实交易管理器实例"""
    global _real_trading_manager
    if _real_trading_manager is None:
        _real_trading_manager = RealTradingManager()
    return _real_trading_manager


if __name__ == "__main__":
    async def test_real_trading():
        manager = RealTradingManager()
        
        # 初始化交易所
        if await manager.initialize_exchanges():
            print("✅ 交易所初始化成功")
            
            # 更新数据
            data = await manager.update_all_data()
            print(f"📊 数据更新完成: {data}")
            
            # 获取摘要
            summary = manager.get_trading_summary()
            print(f"📋 交易摘要: {summary}")
        else:
            print("❌ 交易所初始化失败")
    
    asyncio.run(test_real_trading())

