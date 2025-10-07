#!/usr/bin/env python3
"""
🔐 Bitget API 配置脚本
安全配置Bitget交易所API密钥
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config.api_config_manager import APIConfigManager
from loguru import logger

def setup_bitget_api():
    """配置Bitget API"""
    try:
        print("🔐 开始配置Bitget API...")
        
        # 创建API配置管理器
        config_manager = APIConfigManager()
        
        # 初始化配置 (使用默认密码)
        if not config_manager.initialize_config("Ganfeng888"):
            logger.error("❌ 配置管理器初始化失败")
            return False
        
        # Bitget API配置
        bitget_config = {
            'api_key': 'bg_361f925c6f2139ad15bff1e662995fdd',
            'secret': '6b9f6868b5c6e90b4a866d1a626c3722a169e557dfcfd2175fbeb5fa84085c43',
            'passphrase': 'Ganfeng321',
            'sandbox': False  # 实盘模式
        }
        
        print("📝 配置Bitget API密钥...")
        
        # 配置Bitget
        success = config_manager.add_exchange_credentials(
            'bitget', 
            bitget_config['api_key'],
            bitget_config['secret'],
            bitget_config['passphrase'],
            bitget_config['sandbox']
        )
        
        if success:
            print("✅ Bitget API配置成功")
            
            # 测试连接
            print("🧪 测试Bitget连接...")
            connection_test = config_manager.test_exchange_connection('bitget')
            
            if connection_test:
                print("✅ Bitget连接测试成功")
                print("🎉 Bitget API配置完成，可以开始实盘交易！")
                return True
            else:
                print("❌ Bitget连接测试失败")
                return False
        else:
            print("❌ Bitget API配置失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 配置过程出错: {e}")
        return False

if __name__ == "__main__":
    success = setup_bitget_api()
    if success:
        print("\n🚀 配置完成！现在可以运行:")
        print("python launch_production_system.py")
    else:
        print("\n❌ 配置失败，请检查API密钥")
        sys.exit(1)
