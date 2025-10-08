#!/usr/bin/env python3
"""
🔧 增强的API配置管理器
Enhanced API Configuration Manager
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
import ccxt
from datetime import datetime

class EnhancedAPIConfigManager:
    """增强的API配置管理器"""
    
    def __init__(self):
        self.config_path = Path("config/exchanges.json")
        self.env_path = Path(".env")
        self.exchanges = {}
        self.connected_exchanges = {}
        
        logger.info("🔧 初始化增强API配置管理器")
    
    async def load_config(self) -> Dict[str, Any]:
        """加载API配置"""
        try:
            if not self.config_path.exists():
                logger.error(f"❌ 配置文件不存在: {self.config_path}")
                return {}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"✅ 成功加载API配置: {len(config)}个交易所")
            return config
            
        except Exception as e:
            logger.error(f"❌ 加载API配置失败: {e}")
            return {}
    
    async def validate_api_keys(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """验证API密钥"""
        validation_results = {}
        
        for exchange_name, exchange_config in config.items():
            try:
                logger.info(f"🔍 验证 {exchange_name} API密钥...")
                
                # 检查必要的配置项
                api_key = exchange_config.get('api_key', '')
                secret = exchange_config.get('secret', '')
                
                if not api_key or not secret:
                    validation_results[exchange_name] = False
                    logger.warning(f"⚠️ {exchange_name} API密钥为空")
                    continue
                
                # 创建交易所实例进行测试
                exchange_class = getattr(ccxt, exchange_name, None)
                if not exchange_class:
                    validation_results[exchange_name] = False
                    logger.error(f"❌ 不支持的交易所: {exchange_name}")
                    continue
                
                # 配置交易所参数
                exchange_params = {
                    'apiKey': api_key,
                    'secret': secret,
                    'sandbox': exchange_config.get('sandbox', True),
                    'enableRateLimit': exchange_config.get('enable_rate_limit', True),
                    'timeout': exchange_config.get('timeout', 30000),
                }
                
                # 添加passphrase（如果需要）
                if exchange_config.get('passphrase'):
                    exchange_params['password'] = exchange_config['passphrase']
                
                # 创建交易所实例
                exchange = exchange_class(exchange_params)
                
                # 测试连接（获取账户信息）
                try:
                    # 对于演示模式，我们只检查配置是否正确
                    if api_key.startswith('demo_'):
                        validation_results[exchange_name] = True
                        logger.info(f"✅ {exchange_name} 演示模式配置正确")
                        self.connected_exchanges[exchange_name] = exchange
                    else:
                        # 真实API测试
                        balance = await exchange.fetch_balance()
                        validation_results[exchange_name] = True
                        logger.info(f"✅ {exchange_name} API连接成功")
                        self.connected_exchanges[exchange_name] = exchange
                        
                except Exception as api_error:
                    validation_results[exchange_name] = False
                    logger.error(f"❌ {exchange_name} API连接失败: {api_error}")
                
            except Exception as e:
                validation_results[exchange_name] = False
                logger.error(f"❌ 验证 {exchange_name} 时出错: {e}")
        
        return validation_results
    
    async def get_market_data(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        try:
            if exchange_name not in self.connected_exchanges:
                logger.error(f"❌ 交易所 {exchange_name} 未连接")
                return None
            
            exchange = self.connected_exchanges[exchange_name]
            
            # 对于演示模式，返回模拟数据
            if exchange.apiKey.startswith('demo_'):
                return self._generate_demo_market_data(symbol)
            
            # 获取真实市场数据
            ticker = await exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker['percentage'],
                'volume_24h': ticker['quoteVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            logger.error(f"❌ 获取 {exchange_name} {symbol} 市场数据失败: {e}")
            return self._generate_demo_market_data(symbol)
    
    def _generate_demo_market_data(self, symbol: str) -> Dict[str, Any]:
        """生成演示市场数据"""
        import random
        
        # 基础价格
        base_prices = {
            'BTC/USDT': 43890.50,
            'ETH/USDT': 2678.90,
            'SOL/USDT': 143.20,
            'BNB/USDT': 315.45,
            'ADA/USDT': 0.4520,
            'DOGE/USDT': 0.0875,
            'XRP/USDT': 0.6234,
            'MATIC/USDT': 0.8456,
            'DOT/USDT': 5.67,
            'AVAX/USDT': 23.45
        }
        
        base_price = base_prices.get(symbol, 100.0)
        price_change = random.uniform(-0.05, 0.05)  # ±5%变化
        current_price = base_price * (1 + price_change)
        
        return {
            'symbol': symbol,
            'price': round(current_price, 4),
            'change_24h': round(price_change * 100, 2),
            'volume_24h': random.uniform(1000000, 50000000),
            'high_24h': round(current_price * 1.03, 4),
            'low_24h': round(current_price * 0.97, 4),
            'timestamp': datetime.now().timestamp() * 1000
        }
    
    async def get_account_balance(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """获取账户余额"""
        try:
            if exchange_name not in self.connected_exchanges:
                logger.error(f"❌ 交易所 {exchange_name} 未连接")
                return None
            
            exchange = self.connected_exchanges[exchange_name]
            
            # 对于演示模式，返回模拟余额
            if exchange.apiKey.startswith('demo_'):
                return {
                    'USDT': {'free': 50000.0, 'used': 11543.22, 'total': 61543.22},
                    'BTC': {'free': 0.5, 'used': 0.3, 'total': 0.8},
                    'ETH': {'free': 2.0, 'used': 1.5, 'total': 3.5},
                    'SOL': {'free': 10.0, 'used': 5.0, 'total': 15.0}
                }
            
            # 获取真实余额
            balance = await exchange.fetch_balance()
            return balance
            
        except Exception as e:
            logger.error(f"❌ 获取 {exchange_name} 账户余额失败: {e}")
            return None
    
    async def get_open_positions(self, exchange_name: str) -> List[Dict[str, Any]]:
        """获取开放持仓"""
        try:
            if exchange_name not in self.connected_exchanges:
                logger.error(f"❌ 交易所 {exchange_name} 未连接")
                return []
            
            exchange = self.connected_exchanges[exchange_name]
            
            # 对于演示模式，返回模拟持仓
            if exchange.apiKey.startswith('demo_'):
                return [
                    {
                        'id': 'pos_001',
                        'symbol': 'BTC/USDT',
                        'side': 'long',
                        'size': 0.5,
                        'entry_price': 43250.00,
                        'current_price': 43890.50,
                        'leverage': 3.0,
                        'margin': 7208.33,
                        'unrealized_pnl': 320.25,
                        'unrealized_pnl_pct': 4.44,
                        'open_time': '2025-10-07T15:30:00',
                        'stop_loss': 42100.00,
                        'take_profit': 45000.00,
                        'status': 'open'
                    },
                    {
                        'id': 'pos_002',
                        'symbol': 'ETH/USDT',
                        'side': 'long',
                        'size': 2.0,
                        'entry_price': 2650.00,
                        'current_price': 2678.90,
                        'leverage': 2.0,
                        'margin': 2650.00,
                        'unrealized_pnl': 57.80,
                        'unrealized_pnl_pct': 2.18,
                        'open_time': '2025-10-07T16:15:00',
                        'stop_loss': 2580.00,
                        'take_profit': 2750.00,
                        'status': 'open'
                    }
                ]
            
            # 获取真实持仓
            positions = await exchange.fetch_positions()
            return [pos for pos in positions if pos['contracts'] > 0]
            
        except Exception as e:
            logger.error(f"❌ 获取 {exchange_name} 持仓失败: {e}")
            return []
    
    async def get_trade_history(self, exchange_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取交易历史"""
        try:
            if exchange_name not in self.connected_exchanges:
                logger.error(f"❌ 交易所 {exchange_name} 未连接")
                return []
            
            exchange = self.connected_exchanges[exchange_name]
            
            # 对于演示模式，返回模拟交易历史
            if exchange.apiKey.startswith('demo_'):
                return [
                    {
                        'id': 'trade_001',
                        'symbol': 'BTC/USDT',
                        'side': 'long',
                        'size': 0.3,
                        'entry_price': 42800.00,
                        'exit_price': 43500.00,
                        'leverage': 2.0,
                        'profit': 210.00,
                        'profit_pct': 3.27,
                        'open_time': '2025-10-07T10:30:00',
                        'close_time': '2025-10-07T14:45:00',
                        'duration': '4小时15分钟',
                        'status': 'closed',
                        'result': 'win'
                    },
                    {
                        'id': 'trade_002',
                        'symbol': 'ETH/USDT',
                        'side': 'short',
                        'size': 1.5,
                        'entry_price': 2680.00,
                        'exit_price': 2645.00,
                        'leverage': 3.0,
                        'profit': 52.50,
                        'profit_pct': 1.96,
                        'open_time': '2025-10-07T09:15:00',
                        'close_time': '2025-10-07T11:30:00',
                        'duration': '2小时15分钟',
                        'status': 'closed',
                        'result': 'win'
                    }
                ]
            
            # 获取真实交易历史
            trades = await exchange.fetch_my_trades(limit=limit)
            return trades
            
        except Exception as e:
            logger.error(f"❌ 获取 {exchange_name} 交易历史失败: {e}")
            return []
    
    async def test_all_connections(self) -> Dict[str, Any]:
        """测试所有连接"""
        logger.info("🧪 开始测试所有API连接...")
        
        config = await self.load_config()
        if not config:
            return {'status': 'error', 'message': '无法加载配置'}
        
        validation_results = await self.validate_api_keys(config)
        
        # 统计结果
        total_exchanges = len(config)
        connected_exchanges = sum(1 for result in validation_results.values() if result)
        
        test_results = {
            'status': 'success' if connected_exchanges > 0 else 'error',
            'total_exchanges': total_exchanges,
            'connected_exchanges': connected_exchanges,
            'connection_rate': round((connected_exchanges / total_exchanges) * 100, 1) if total_exchanges > 0 else 0,
            'results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ API连接测试完成: {connected_exchanges}/{total_exchanges} 成功")
        return test_results

# 创建全局实例
api_config_manager = EnhancedAPIConfigManager()

async def main():
    """测试函数"""
    results = await api_config_manager.test_all_connections()
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
