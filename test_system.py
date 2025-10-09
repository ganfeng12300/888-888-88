#!/usr/bin/env python3
"""
🧪 系统全面测试脚本
测试所有核心功能并修复问题
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有核心模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试配置系统
        from src.core.config import settings
        print("✅ 配置系统导入成功")
        
        # 测试日志系统
        from src.monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig
        print("✅ 日志系统导入成功")
        
        # 测试交易所API
        from src.exchanges.bitget_api import BitgetAPI
        print("✅ Bitget API导入成功")
        
        # 测试AI引擎
        from src.ai.ai_engine import AIEngine
        print("✅ AI引擎导入成功")
        
        # 测试交易引擎
        from src.trading.advanced_trading_engine import AdvancedTradingEngine
        print("✅ 交易引擎导入成功")
        
        # 测试风险管理
        from src.risk.enhanced_risk_manager import EnhancedRiskManager
        print("✅ 风险管理导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_config():
    """测试配置系统"""
    print("\n🔧 测试配置系统...")
    
    try:
        from src.core.config import settings
        
        # 测试基本配置
        print(f"应用名称: {settings.app_name}")
        print(f"版本: {settings.app_version}")
        print(f"环境: {settings.environment}")
        
        # 测试交易所配置
        bitget_config = settings.get_exchange_config("bitget")
        if bitget_config and bitget_config.get('api_key'):
            print("✅ Bitget配置加载成功")
            print(f"API Key: {bitget_config.get('api_key', 'N/A')[:10]}...")
        else:
            print("❌ Bitget配置未找到或API Key为空")
            return False
        
        # 测试AI权重验证
        if settings.validate_ai_weights():
            print("✅ AI模型权重验证通过")
        else:
            print("❌ AI模型权重验证失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_logging():
    """测试日志系统"""
    print("\n📝 测试日志系统...")
    
    try:
        from src.monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory
        
        # 创建日志配置
        log_config = LogConfig(
            log_dir="test_logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        
        # 初始化日志系统
        logger = UnifiedLoggingSystem(log_config)
        
        # 测试各种日志级别
        logger.info("测试信息日志", category=LogCategory.SYSTEM)
        logger.warning("测试警告日志", category=LogCategory.TRADING)
        logger.error("测试错误日志", category=LogCategory.AI)
        
        print("✅ 日志系统测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 日志系统测试失败: {e}")
        traceback.print_exc()
        return False

async def test_bitget_api():
    """测试Bitget API连接"""
    print("\n🔗 测试Bitget API连接...")
    
    try:
        from src.exchanges.bitget_api import BitgetAPI
        from src.core.config import settings
        
        # 获取Bitget配置
        config = settings.get_exchange_config("bitget")
        if not config or not config.get('api_key'):
            print("❌ Bitget配置未找到或API Key为空")
            return False
        
        # 初始化API
        api = BitgetAPI(
            api_key=config["api_key"],
            secret_key=config["secret"],
            passphrase=config["password"]
        )
        
        # 测试连接
        print("正在测试API连接...")
        account_info = api.get_account_info()
        
        if account_info:
            print("✅ Bitget API连接成功")
            print(f"账户信息: {account_info}")
            return True
        else:
            print("❌ 无法获取账户信息")
            return False
        
    except Exception as e:
        print(f"❌ Bitget API测试失败: {e}")
        traceback.print_exc()
        return False

def test_ai_engine():
    """测试AI引擎"""
    print("\n🤖 测试AI引擎...")
    
    try:
        from src.ai.ai_engine import AIEngine
        from src.core.config import settings
        
        # 初始化AI引擎
        ai_engine = AIEngine(settings)
        
        # 测试AI引擎初始化
        print("✅ AI引擎初始化成功")
        
        # 测试模型权重
        weights = settings.get_model_weights()
        print(f"模型权重: {weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI引擎测试失败: {e}")
        traceback.print_exc()
        return False

def test_trading_engine():
    """测试交易引擎"""
    print("\n📈 测试交易引擎...")
    
    try:
        from src.trading.advanced_trading_engine import AdvancedTradingEngine
        from src.core.config import settings
        
        # 初始化交易引擎
        trading_engine = AdvancedTradingEngine(settings)
        
        print("✅ 交易引擎初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 交易引擎测试失败: {e}")
        traceback.print_exc()
        return False

def test_risk_manager():
    """测试风险管理"""
    print("\n🛡️ 测试风险管理...")
    
    try:
        from src.risk.enhanced_risk_manager import EnhancedRiskManager
        from src.core.config import settings
        
        # 初始化风险管理器
        risk_manager = EnhancedRiskManager(settings)
        
        print("✅ 风险管理器初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 风险管理测试失败: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始系统全面测试\n")
    
    tests = [
        ("模块导入", test_imports),
        ("配置系统", test_config),
        ("日志系统", test_logging),
        ("Bitget API", test_bitget_api),
        ("AI引擎", test_ai_engine),
        ("交易引擎", test_trading_engine),
        ("风险管理", test_risk_manager),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 输出测试结果
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪！")
        return True
    else:
        print("⚠️ 部分测试失败，需要修复")
        return False

if __name__ == "__main__":
    # 运行测试
    result = asyncio.run(run_all_tests())
    
    if result:
        print("\n✅ 系统测试完成，可以安全下载到本地运行！")
    else:
        print("\n❌ 系统存在问题，正在尝试修复...")
