#!/usr/bin/env python3
"""
🔐 真实实盘API配置脚本
配置Bitget等交易所的真实API密钥
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config.api_config_manager import APIConfigManager

def setup_bitget_real_api():
    """配置Bitget真实实盘API"""
    print("🔐 配置Bitget真实实盘API")
    print("=" * 50)
    
    # 获取API密钥信息
    print("请输入您的Bitget实盘API信息：")
    print("⚠️  请确保API密钥具有交易权限且来自实盘账户")
    print()
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key不能为空")
        return False
    
    api_secret = input("API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret不能为空")
        return False
    
    passphrase = input("Passphrase: ").strip()
    if not passphrase:
        print("❌ Passphrase不能为空")
        return False
    
    # 确认是否为实盘
    print()
    print("⚠️  重要确认：")
    print("1. 这些API密钥来自Bitget实盘账户（非模拟盘）？")
    print("2. API密钥已开启现货交易权限？")
    print("3. 您已充分了解实盘交易风险？")
    
    confirm = input("确认以上所有问题 (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ 配置已取消")
        return False
    
    try:
        # 初始化配置管理器
        config_manager = APIConfigManager()
        
        # 设置主密码
        master_password = input("设置主密码（用于加密API密钥）: ").strip()
        if not master_password:
            master_password = "Ganfeng888"  # 默认密码
        
        # 初始化配置
        if not config_manager.initialize_config(master_password):
            print("❌ 配置管理器初始化失败")
            return False
        
        # 添加Bitget配置
        success = config_manager.add_exchange_config(
            exchange_name="bitget",
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=False  # 实盘模式
        )
        
        if success:
            print("✅ Bitget实盘API配置成功")
            
            # 测试连接
            print("🔍 测试API连接...")
            if config_manager.test_exchange_connection("bitget"):
                print("✅ API连接测试成功")
                
                # 获取账户信息
                try:
                    import ccxt
                    exchange_config = config_manager.get_exchange_config("bitget")
                    
                    exchange = ccxt.bitget({
                        'apiKey': exchange_config.api_key,
                        'secret': exchange_config.api_secret,
                        'password': exchange_config.passphrase,
                        'sandbox': False,
                        'enableRateLimit': True,
                    })
                    
                    balance = exchange.fetch_balance()
                    total_balance = balance['total']
                    
                    print("💰 账户余额信息：")
                    for currency, amount in total_balance.items():
                        if amount > 0:
                            print(f"  {currency}: {amount}")
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️ 获取账户信息失败: {e}")
                    return True  # API配置成功，但获取余额失败
            else:
                print("❌ API连接测试失败")
                return False
        else:
            print("❌ Bitget API配置失败")
            return False
            
    except Exception as e:
        print(f"❌ 配置过程出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 888-888-88 真实实盘API配置")
    print("=" * 50)
    
    if setup_bitget_real_api():
        print()
        print("🎉 真实实盘API配置完成！")
        print("=" * 50)
        print("📋 下一步：")
        print("1. 运行 python auto_start_system.py 启动系统")
        print("2. 访问 http://localhost:8888 查看实时数据")
        print("3. 检查真实账户余额和交易状态")
        print("=" * 50)
        return True
    else:
        print()
        print("❌ API配置失败，请检查输入信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

