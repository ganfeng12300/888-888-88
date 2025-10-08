#!/usr/bin/env python3
"""
🚀 888-888-88 完整Web界面启动器
Complete Dashboard Launcher
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from loguru import logger

def main():
    """启动完整的Web管理界面"""
    try:
        logger.info("🚀 启动888-888-88完整Web管理界面")
        
        # 检查依赖
        required_packages = ['fastapi', 'uvicorn', 'jinja2']
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
        
        # 启动Web服务器
        logger.info("🌐 启动Web服务器...")
        logger.info("📊 完整管理界面地址:")
        logger.info("   - 主界面: http://localhost:8000")
        logger.info("   - API文档: http://localhost:8000/docs")
        logger.info("   - 完整数据API: http://localhost:8000/api/dashboard/complete")
        logger.info("   - 健康检查: http://localhost:8000/health")
        
        # 运行服务器
        os.system("python src/web/complete_dashboard.py")
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断，程序退出")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        return False

if __name__ == "__main__":
    main()
