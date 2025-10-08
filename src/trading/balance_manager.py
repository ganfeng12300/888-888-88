#!/usr/bin/env python3
"""
💰 余额管理器 - 合约和现货余额管理
Balance Manager - Futures and Spot Balance Management
"""
import os
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import ccxt
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Balance:
    """余额信息"""
    currency: str
    free: float
    used: float
    total: float
    usd_value: float
    timestamp: datetime

@dataclass
class AccountInfo:
    """账户信息"""
    account_type: str  # spot, futures
    balances: Dict[str, Balance]
    total_usd_value: float
    margin_level: Optional[float] = None
    available_margin: Optional[float] = None
    used_margin: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    timestamp: datetime = None

class BalanceManager:
    """余额管理器"""
    
    def __init__(self):
        self.exchanges = {}
        self.balances_cache = {}
        self.last_update = {}
        self.update_interval = 30  # 30秒更新一次
        
        # 初始化交易所
        self.init_exchanges()
        
        logger.info("💰 余额管理器初始化完成")
    
    def init_exchanges(self):
        """初始化交易所连接"""
        try:
            # Bitget现货
            self.exchanges['bitget_spot'] = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': os.getenv('BITGET_SANDBOX', 'false').lower() == 'true',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Bitget合约
            self.exchanges['bitget_futures'] = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': os.getenv('BITGET_SANDBOX', 'false').lower() == 'true',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'  # 永续合约
                }
            })
            
            logger.info("✅ 交易所连接初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 交易所初始化失败: {e}")
    
    async def get_spot_balance(self) -> Optional[AccountInfo]:
        """获取现货余额"""
        try:
            exchange = self.exchanges.get('bitget_spot')
            if not exchange:
                return None
            
            # 获取余额
            balance_data = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_balance
            )
            
            balances = {}
            total_usd_value = 0.0
            
            # 获取价格信息用于USD估值
            tickers = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_tickers
            )
            
            for currency, balance_info in balance_data.items():
                if currency == 'info':
                    continue
                
                free = float(balance_info.get('free', 0))
                used = float(balance_info.get('used', 0))
                total = float(balance_info.get('total', 0))
                
                if total > 0:
                    # 计算USD价值
                    usd_value = 0.0
                    if currency == 'USDT' or currency == 'USD':
                        usd_value = total
                    else:
                        # 尝试获取对USDT的价格
                        symbol = f"{currency}/USDT"
                        if symbol in tickers:
                            price = float(tickers[symbol]['last'])
                            usd_value = total * price
                    
                    balances[currency] = Balance(
                        currency=currency,
                        free=free,
                        used=used,
                        total=total,
                        usd_value=usd_value,
                        timestamp=datetime.now()
                    )
                    
                    total_usd_value += usd_value
            
            account_info = AccountInfo(
                account_type="spot",
                balances=balances,
                total_usd_value=total_usd_value,
                timestamp=datetime.now()
            )
            
            self.balances_cache['spot'] = account_info
            self.last_update['spot'] = datetime.now()
            
            logger.info(f"💰 现货余额更新: ${total_usd_value:.2f}")
            return account_info
            
        except Exception as e:
            logger.error(f"❌ 获取现货余额失败: {e}")
            return None
    
    async def get_futures_balance(self) -> Optional[AccountInfo]:
        """获取合约余额"""
        try:
            exchange = self.exchanges.get('bitget_futures')
            if not exchange:
                return None
            
            # 获取余额
            balance_data = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_balance
            )
            
            balances = {}
            total_usd_value = 0.0
            margin_level = None
            available_margin = None
            used_margin = None
            unrealized_pnl = None
            
            # 解析余额信息
            if 'info' in balance_data:
                info = balance_data['info']
                if isinstance(info, list) and len(info) > 0:
                    account_data = info[0]
                    
                    # 保证金信息
                    available_margin = float(account_data.get('available', 0))
                    used_margin = float(account_data.get('frozen', 0))
                    unrealized_pnl = float(account_data.get('unrealizedPL', 0))
                    
                    # 计算保证金水平
                    if used_margin > 0:
                        margin_level = available_margin / used_margin
            
            # 处理余额
            for currency, balance_info in balance_data.items():
                if currency == 'info':
                    continue
                
                free = float(balance_info.get('free', 0))
                used = float(balance_info.get('used', 0))
                total = float(balance_info.get('total', 0))
                
                if total > 0:
                    # 合约账户通常以USDT计价
                    usd_value = total if currency in ['USDT', 'USD'] else 0
                    
                    balances[currency] = Balance(
                        currency=currency,
                        free=free,
                        used=used,
                        total=total,
                        usd_value=usd_value,
                        timestamp=datetime.now()
                    )
                    
                    total_usd_value += usd_value
            
            account_info = AccountInfo(
                account_type="futures",
                balances=balances,
                total_usd_value=total_usd_value,
                margin_level=margin_level,
                available_margin=available_margin,
                used_margin=used_margin,
                unrealized_pnl=unrealized_pnl,
                timestamp=datetime.now()
            )
            
            self.balances_cache['futures'] = account_info
            self.last_update['futures'] = datetime.now()
            
            logger.info(f"💰 合约余额更新: ${total_usd_value:.2f}, 未实现盈亏: ${unrealized_pnl:.2f}")
            return account_info
            
        except Exception as e:
            logger.error(f"❌ 获取合约余额失败: {e}")
            return None
    
    async def get_all_balances(self) -> Dict[str, AccountInfo]:
        """获取所有账户余额"""
        results = {}
        
        # 并发获取现货和合约余额
        tasks = [
            self.get_spot_balance(),
            self.get_futures_balance()
        ]
        
        spot_balance, futures_balance = await asyncio.gather(*tasks, return_exceptions=True)
        
        if isinstance(spot_balance, AccountInfo):
            results['spot'] = spot_balance
        
        if isinstance(futures_balance, AccountInfo):
            results['futures'] = futures_balance
        
        return results
    
    def get_cached_balance(self, account_type: str) -> Optional[AccountInfo]:
        """获取缓存的余额信息"""
        if account_type not in self.balances_cache:
            return None
        
        last_update = self.last_update.get(account_type)
        if not last_update:
            return None
        
        # 检查缓存是否过期
        if (datetime.now() - last_update).seconds > self.update_interval:
            return None
        
        return self.balances_cache[account_type]
    
    def get_total_portfolio_value(self) -> float:
        """获取总投资组合价值"""
        total_value = 0.0
        
        for account_type in ['spot', 'futures']:
            account_info = self.get_cached_balance(account_type)
            if account_info:
                total_value += account_info.total_usd_value
        
        return total_value
    
    def get_balance_summary(self) -> Dict[str, Any]:
        """获取余额摘要"""
        summary = {
            'total_portfolio_value': 0.0,
            'spot_value': 0.0,
            'futures_value': 0.0,
            'unrealized_pnl': 0.0,
            'margin_level': None,
            'available_margin': 0.0,
            'used_margin': 0.0,
            'top_holdings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        all_holdings = []
        
        # 现货账户
        spot_account = self.get_cached_balance('spot')
        if spot_account:
            summary['spot_value'] = spot_account.total_usd_value
            for currency, balance in spot_account.balances.items():
                if balance.usd_value > 1:  # 只显示价值超过1美元的持仓
                    all_holdings.append({
                        'currency': currency,
                        'amount': balance.total,
                        'usd_value': balance.usd_value,
                        'account_type': 'spot'
                    })
        
        # 合约账户
        futures_account = self.get_cached_balance('futures')
        if futures_account:
            summary['futures_value'] = futures_account.total_usd_value
            summary['unrealized_pnl'] = futures_account.unrealized_pnl or 0.0
            summary['margin_level'] = futures_account.margin_level
            summary['available_margin'] = futures_account.available_margin or 0.0
            summary['used_margin'] = futures_account.used_margin or 0.0
            
            for currency, balance in futures_account.balances.items():
                if balance.usd_value > 1:
                    all_holdings.append({
                        'currency': currency,
                        'amount': balance.total,
                        'usd_value': balance.usd_value,
                        'account_type': 'futures'
                    })
        
        # 总价值
        summary['total_portfolio_value'] = summary['spot_value'] + summary['futures_value']
        
        # 按价值排序，取前10
        all_holdings.sort(key=lambda x: x['usd_value'], reverse=True)
        summary['top_holdings'] = all_holdings[:10]
        
        return summary
    
    def check_balance_alerts(self) -> List[Dict[str, Any]]:
        """检查余额警报"""
        alerts = []
        
        # 检查合约保证金水平
        futures_account = self.get_cached_balance('futures')
        if futures_account and futures_account.margin_level:
            if futures_account.margin_level < 1.5:  # 保证金水平低于150%
                alerts.append({
                    'type': 'margin_warning',
                    'message': f'保证金水平过低: {futures_account.margin_level:.2f}',
                    'severity': 'high' if futures_account.margin_level < 1.2 else 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        # 检查总资产变化
        total_value = self.get_total_portfolio_value()
        if hasattr(self, 'last_total_value'):
            change_pct = (total_value - self.last_total_value) / self.last_total_value * 100
            if abs(change_pct) > 5:  # 资产变化超过5%
                alerts.append({
                    'type': 'portfolio_change',
                    'message': f'投资组合价值变化: {change_pct:+.2f}%',
                    'severity': 'high' if abs(change_pct) > 10 else 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        self.last_total_value = total_value
        return alerts
    
    async def start_monitoring(self):
        """启动余额监控"""
        logger.info("🚀 启动余额监控")
        
        while True:
            try:
                # 更新余额
                await self.get_all_balances()
                
                # 检查警报
                alerts = self.check_balance_alerts()
                for alert in alerts:
                    logger.warning(f"⚠️ 余额警报: {alert['message']}")
                
                # 等待下次更新
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ 余额监控错误: {e}")
                await asyncio.sleep(10)  # 错误时短暂等待
    
    def get_balance_for_trading(self, account_type: str, currency: str) -> float:
        """获取可用于交易的余额"""
        account_info = self.get_cached_balance(account_type)
        if not account_info:
            return 0.0
        
        if currency not in account_info.balances:
            return 0.0
        
        balance = account_info.balances[currency]
        
        # 为安全起见，只使用90%的可用余额
        return balance.free * 0.9
    
    def calculate_position_size(self, account_type: str, currency: str, 
                              risk_percentage: float = 0.02) -> float:
        """计算仓位大小"""
        account_info = self.get_cached_balance(account_type)
        if not account_info:
            return 0.0
        
        # 基于风险百分比计算仓位
        risk_amount = account_info.total_usd_value * risk_percentage
        
        if currency in account_info.balances:
            balance = account_info.balances[currency]
            if balance.usd_value > 0:
                # 计算可以承受风险的仓位大小
                position_size = risk_amount / balance.usd_value * balance.total
                return min(position_size, balance.free * 0.9)  # 不超过可用余额的90%
        
        return 0.0

# 全局实例
balance_manager = BalanceManager()

if __name__ == "__main__":
    async def main():
        # 测试余额管理器
        balances = await balance_manager.get_all_balances()
        
        print("=== 余额信息 ===")
        for account_type, account_info in balances.items():
            print(f"\n{account_type.upper()} 账户:")
            print(f"总价值: ${account_info.total_usd_value:.2f}")
            
            if account_info.unrealized_pnl is not None:
                print(f"未实现盈亏: ${account_info.unrealized_pnl:.2f}")
            
            if account_info.margin_level is not None:
                print(f"保证金水平: {account_info.margin_level:.2f}")
            
            print("主要持仓:")
            for currency, balance in account_info.balances.items():
                if balance.usd_value > 1:
                    print(f"  {currency}: {balance.total:.4f} (${balance.usd_value:.2f})")
        
        # 显示摘要
        summary = balance_manager.get_balance_summary()
        print(f"\n=== 投资组合摘要 ===")
        print(f"总价值: ${summary['total_portfolio_value']:.2f}")
        print(f"现货价值: ${summary['spot_value']:.2f}")
        print(f"合约价值: ${summary['futures_value']:.2f}")
        
        if summary['unrealized_pnl']:
            print(f"未实现盈亏: ${summary['unrealized_pnl']:.2f}")
    
    asyncio.run(main())

