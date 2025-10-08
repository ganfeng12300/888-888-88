#!/usr/bin/env python3
"""
API连接测试脚本
"""
import ccxt
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_bitget_api():
    """测试Bitget API连接"""
    print("🔄 测试Bitget API连接...")
    
    try:
        # 创建交易所实例
        exchange = ccxt.bitget({
            'apiKey': os.getenv('BITGET_API_KEY'),
            'secret': os.getenv('BITGET_SECRET_KEY'),
            'password': os.getenv('BITGET_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # 测试连接 - 获取账户余额
        print("📊 获取账户余额...")
        balance = exchange.fetch_balance()
        print(f"✅ Bitget API连接成功!")
        print(f"📈 账户信息: {len(balance)} 个币种")
        
        # 显示主要余额
        main_balances = {}
        for currency, info in balance.items():
            if currency != 'info' and info.get('total', 0) > 0:
                main_balances[currency] = info['total']
        
        if main_balances:
            print("💰 主要持仓:")
            for currency, amount in main_balances.items():
                print(f"   {currency}: {amount}")
        
        # 测试市场数据获取
        print("\n📊 获取市场数据...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ 市场数据获取成功: BTC/USDT = ${ticker['last']}")
        
        # 测试现货和合约账户
        print("\n🔍 测试账户类型...")
        
        # 现货账户
        try:
            exchange.options['defaultType'] = 'spot'
            spot_balance = exchange.fetch_balance()
            print("✅ 现货账户连接成功")
        except Exception as e:
            print(f"❌ 现货账户连接失败: {e}")
        
        # 合约账户
        try:
            exchange.options['defaultType'] = 'swap'
            futures_balance = exchange.fetch_balance()
            print("✅ 合约账户连接成功")
        except Exception as e:
            print(f"❌ 合约账户连接失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bitget API连接失败: {e}")
        return False

def test_system_requirements():
    """测试系统要求"""
    print("\n🔧 检查系统要求...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 8):
        print("✅ Python版本符合要求 (>=3.8)")
    else:
        print("❌ Python版本过低，需要3.8+")
        return False
    
    # 检查必要的包
    required_packages = [
        'ccxt', 'pandas', 'numpy', 'sklearn', 
        'flask', 'flask_socketio', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少包: {missing_packages}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始系统API连接测试...")
    print("=" * 50)
    
    # 测试系统要求
    if not test_system_requirements():
        print("❌ 系统要求检查失败")
        return False
    
    print("\n" + "=" * 50)
    
    # 测试API连接
    if not test_bitget_api():
        print("❌ API连接测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！系统准备就绪！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

