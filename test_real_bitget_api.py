#!/usr/bin/env python3
"""
🔧 真实Bitget API连接测试器
Real Bitget API Connection Tester
"""

import asyncio
import ccxt
import json
from datetime import datetime
from loguru import logger

def test_bitget_connection():
    """测试真实Bitget API连接"""
    try:
        logger.info("🔧 开始测试真实Bitget API连接...")
        
        # 配置Bitget交易所
        bitget = ccxt.bitget({
            'apiKey': 'bg_361f925c6f2139ad15bff1e662995fdd',
            'secret': '6b9f6868b5c6e90b4a866d1a626c3722a169e557dfcfd2175fbeb5fa84085c43',
            'password': 'Ganfeng321',  # Bitget使用password而不是passphrase
            'sandbox': False,  # 真实环境
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        print("🔍 测试API连接...")
        
        # 1. 测试基本连接 - 获取服务器时间
        try:
            server_time = bitget.fetch_time()
            print(f"✅ 服务器连接成功")
            print(f"   服务器时间: {datetime.fromtimestamp(server_time/1000)}")
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
        
        # 2. 测试账户权限 - 获取账户余额
        try:
            balance = bitget.fetch_balance()
            print(f"✅ 账户权限验证成功")
            
            # 显示主要余额
            main_currencies = ['USDT', 'BTC', 'ETH']
            for currency in main_currencies:
                if currency in balance and balance[currency]['total'] > 0:
                    print(f"   {currency}: {balance[currency]['total']:.8f} (可用: {balance[currency]['free']:.8f})")
            
        except Exception as e:
            print(f"❌ 账户权限验证失败: {e}")
            return False
        
        # 3. 测试市场数据 - 获取交易对信息
        try:
            markets = bitget.load_markets()
            print(f"✅ 市场数据获取成功")
            print(f"   支持的交易对数量: {len(markets)}")
            
            # 显示主要交易对
            main_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            for symbol in main_symbols:
                if symbol in markets:
                    print(f"   {symbol}: 支持 ✓")
                else:
                    print(f"   {symbol}: 不支持 ✗")
                    
        except Exception as e:
            print(f"❌ 市场数据获取失败: {e}")
            return False
        
        # 4. 测试实时价格数据
        try:
            ticker = bitget.fetch_ticker('BTC/USDT')
            print(f"✅ 实时价格数据获取成功")
            print(f"   BTC/USDT 价格: ${ticker['last']:,.2f}")
            print(f"   24h涨跌: {ticker['percentage']:.2f}%")
            print(f"   24h成交量: {ticker['quoteVolume']:,.2f} USDT")
            
        except Exception as e:
            print(f"❌ 实时价格数据获取失败: {e}")
            return False
        
        # 5. 测试订单历史（如果有）
        try:
            orders = bitget.fetch_orders('BTC/USDT', limit=5)
            print(f"✅ 订单历史获取成功")
            print(f"   历史订单数量: {len(orders)}")
            
        except Exception as e:
            print(f"⚠️ 订单历史获取失败（可能没有历史订单）: {e}")
        
        # 6. 测试持仓信息（期货）
        try:
            positions = bitget.fetch_positions()
            open_positions = [pos for pos in positions if pos['contracts'] > 0]
            print(f"✅ 持仓信息获取成功")
            print(f"   当前持仓数量: {len(open_positions)}")
            
            for pos in open_positions[:3]:  # 显示前3个持仓
                print(f"   {pos['symbol']}: {pos['side']} {pos['contracts']} (盈亏: {pos['unrealizedPnl']:.2f})")
                
        except Exception as e:
            print(f"⚠️ 持仓信息获取失败（可能是现货账户）: {e}")
        
        print("\n🎉 Bitget API连接测试完成！")
        print("✅ 所有基本功能正常，可以进行实盘交易")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bitget API连接测试失败: {e}")
        return False

def main():
    """主函数"""
    success = test_bitget_connection()
    if success:
        print("\n✅ API配置正确，系统可以启动")
    else:
        print("\n❌ API配置有问题，请检查配置")

if __name__ == "__main__":
    main()
