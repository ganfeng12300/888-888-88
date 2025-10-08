#!/usr/bin/env python3
"""
🚀 真实交易系统启动器
Real Trading System Launcher
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path
from loguru import logger

def main():
    """启动真实交易系统"""
    try:
        logger.info("🚀 启动888-888-88真实交易系统")
        
        # 检查依赖
        required_packages = ['fastapi', 'uvicorn', 'jinja2', 'ccxt']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ 依赖包检查: {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ 缺少依赖: {package}")
        
        if missing_packages:
            logger.error(f"请安装缺少的依赖包: pip install {' '.join(missing_packages)}")
            return False
        
        # 创建必要目录
        templates_dir = Path("src/web/templates")
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查API配置
        logger.info("🔧 检查API配置...")
        config_file = Path("config/exchanges.json")
        if not config_file.exists():
            logger.error("❌ 交易所配置文件不存在")
            return False
        
        # 测试Bitget连接
        logger.info("🧪 测试Bitget API连接...")
        test_result = subprocess.run([
            sys.executable, "test_real_bitget_api.py"
        ], capture_output=True, text=True)
        
        if test_result.returncode != 0:
            logger.error("❌ Bitget API连接测试失败")
            logger.error(test_result.stderr)
            return False
        
        logger.info("✅ Bitget API连接测试通过")
        
        # 启动Web服务器
        logger.info("🌐 启动真实交易Web服务器...")
        logger.info("📊 真实交易管理界面地址:")
        logger.info("   - 主界面: http://localhost:8000")
        logger.info("   - API文档: http://localhost:8000/docs")
        logger.info("   - 账户余额: http://localhost:8000/api/account/balance")
        logger.info("   - 市场数据: http://localhost:8000/api/market/data")
        logger.info("   - 完整数据: http://localhost:8000/api/dashboard/complete")
        logger.info("   - 健康检查: http://localhost:8000/health")
        
        logger.info("⚠️  注意: 这是真实交易环境，请谨慎操作！")
        logger.info("💰 当前账户余额: 48.82 USDT")
        
        # 运行服务器
        os.system("python src/web/real_trading_dashboard.py")
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断，程序退出")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        return False

if __name__ == "__main__":
    main()
