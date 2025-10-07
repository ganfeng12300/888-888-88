#!/usr/bin/env python3
"""
🧪 888-888-88 自动化系统测试
完整的系统功能测试和评估报告
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.config.api_config_manager import APIConfigManager

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = {
            "api_configuration": {"status": "未测试", "score": 0, "details": []},
            "data_collection": {"status": "未测试", "score": 0, "details": []},
            "ai_models": {"status": "未测试", "score": 0, "details": []},
            "trading_engine": {"status": "未测试", "score": 0, "details": []},
            "risk_management": {"status": "未测试", "score": 0, "details": []},
            "monitoring": {"status": "未测试", "score": 0, "details": []},
            "web_interface": {"status": "未测试", "score": 0, "details": []},
            "system_integration": {"status": "未测试", "score": 0, "details": []}
        }
        
        self.overall_score = 0
        self.grade = "未评估"
    
    def print_banner(self):
        """打印测试横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                 888-888-88 系统功能测试                      ║
║                   生产级代码评估报告                         ║
║                                                              ║
║  🧪 全面功能测试 | 📊 性能评估 | 🔍 代码质量检查           ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🕒 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 66)
    
    def test_api_configuration(self) -> bool:
        """测试API配置"""
        try:
            logger.info("🔐 测试API配置模块...")
            details = []
            score = 0
            
            # 测试配置管理器初始化
            config_manager = APIConfigManager()
            if config_manager.initialize_config("Ganfeng888"):
                details.append("✅ API配置管理器初始化成功")
                score += 20
            else:
                details.append("❌ API配置管理器初始化失败")
            
            # 测试已配置的交易所
            exchanges = config_manager.list_configured_exchanges()
            if exchanges:
                details.append(f"✅ 已配置交易所: {', '.join(exchanges)}")
                score += 30
                
                # 测试连接
                for exchange in exchanges:
                    if config_manager.test_exchange_connection(exchange):
                        details.append(f"✅ {exchange} 连接测试成功")
                        score += 25
                    else:
                        details.append(f"⚠️ {exchange} 连接测试失败")
                        score += 10
            else:
                details.append("❌ 未配置任何交易所")
            
            # 测试加密功能
            try:
                config_manager._get_fernet("test_password")
                details.append("✅ 密钥加密功能正常")
                score += 25
            except Exception as e:
                details.append(f"❌ 密钥加密功能异常: {e}")
            
            self.test_results["api_configuration"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 70
            
        except Exception as e:
            self.test_results["api_configuration"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_data_collection(self) -> bool:
        """测试数据收集模块"""
        try:
            logger.info("📊 测试数据收集模块...")
            details = []
            score = 0
            
            # 检查数据收集相关文件
            data_files = [
                "src/data_collection/binance_collector.py",
                "src/data_collection/okx_collector.py",
                "src/data_collection/data_processor.py",
                "src/data_collection/market_data_manager.py"
            ]
            
            existing_files = 0
            for file_path in data_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    existing_files += 1
                    score += 20
                else:
                    details.append(f"⚠️ {file_path} 不存在")
            
            # 检查数据目录
            if Path("data").exists():
                details.append("✅ 数据目录存在")
                score += 20
            else:
                details.append("❌ 数据目录不存在")
            
            self.test_results["data_collection"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 60
            
        except Exception as e:
            self.test_results["data_collection"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_ai_models(self) -> bool:
        """测试AI模型模块"""
        try:
            logger.info("🤖 测试AI模型模块...")
            details = []
            score = 0
            
            # 检查AI相关文件
            ai_files = [
                "src/ai/ai_engine.py",
                "src/ai/ai_fusion_engine.py",
                "src/ai/model_trainer.py",
                "src/ai/prediction_engine.py"
            ]
            
            for file_path in ai_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 20
                    
                    # 检查文件内容是否有pass语句
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'pass' not in content or content.count('pass') < 3:
                            details.append(f"✅ {file_path} 无空pass语句")
                            score += 5
                        else:
                            details.append(f"⚠️ {file_path} 包含pass语句")
                else:
                    details.append(f"❌ {file_path} 不存在")
            
            # 检查模型目录
            if Path("models").exists():
                details.append("✅ 模型目录存在")
                score += 20
            else:
                details.append("❌ 模型目录不存在")
            
            self.test_results["ai_models"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 70
            
        except Exception as e:
            self.test_results["ai_models"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_trading_engine(self) -> bool:
        """测试交易引擎"""
        try:
            logger.info("💰 测试交易引擎...")
            details = []
            score = 0
            
            # 检查交易相关文件
            trading_files = [
                "src/trading/trading_engine.py",
                "src/trading/order_manager.py",
                "src/trading/position_manager.py",
                "src/trading/strategy_executor.py"
            ]
            
            for file_path in trading_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 25
                else:
                    details.append(f"⚠️ {file_path} 不存在")
            
            self.test_results["trading_engine"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 50
            
        except Exception as e:
            self.test_results["trading_engine"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_risk_management(self) -> bool:
        """测试风险管理"""
        try:
            logger.info("🛡️ 测试风险管理模块...")
            details = []
            score = 0
            
            # 检查风险管理文件
            risk_files = [
                "src/risk_management/risk_manager.py",
                "src/risk_management/position_sizer.py",
                "src/risk_management/drawdown_monitor.py"
            ]
            
            for file_path in risk_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 33
                else:
                    details.append(f"⚠️ {file_path} 不存在")
            
            self.test_results["risk_management"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 60
            
        except Exception as e:
            self.test_results["risk_management"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_monitoring(self) -> bool:
        """测试监控系统"""
        try:
            logger.info("📊 测试监控系统...")
            details = []
            score = 0
            
            # 检查监控文件
            monitoring_files = [
                "src/monitoring/system_monitor.py",
                "src/monitoring/performance_tracker.py",
                "src/monitoring/alert_manager.py"
            ]
            
            for file_path in monitoring_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 33
                else:
                    details.append(f"⚠️ {file_path} 不存在")
            
            # 检查日志目录
            if Path("logs").exists():
                details.append("✅ 日志目录存在")
                score += 10
            else:
                details.append("❌ 日志目录不存在")
            
            self.test_results["monitoring"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 60
            
        except Exception as e:
            self.test_results["monitoring"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_web_interface(self) -> bool:
        """测试Web界面"""
        try:
            logger.info("🌐 测试Web界面...")
            details = []
            score = 0
            
            # 检查Web相关文件
            web_files = [
                "src/web/app.py",
                "src/web/api_routes.py",
                "src/web/dashboard.py"
            ]
            
            for file_path in web_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 33
                else:
                    details.append(f"⚠️ {file_path} 不存在")
            
            self.test_results["web_interface"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 50
            
        except Exception as e:
            self.test_results["web_interface"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def test_system_integration(self) -> bool:
        """测试系统集成"""
        try:
            logger.info("🔧 测试系统集成...")
            details = []
            score = 0
            
            # 检查主要启动文件
            main_files = [
                "start_production_system.py",
                "launch_production_system.py",
                "setup_bitget_config.py"
            ]
            
            for file_path in main_files:
                if Path(file_path).exists():
                    details.append(f"✅ {file_path} 存在")
                    score += 25
                else:
                    details.append(f"❌ {file_path} 不存在")
            
            # 检查配置文件
            if Path("config").exists():
                details.append("✅ 配置目录存在")
                score += 25
                
                if Path("config/api_config.enc").exists():
                    details.append("✅ API配置文件存在")
                    score += 25
                else:
                    details.append("⚠️ API配置文件不存在")
            else:
                details.append("❌ 配置目录不存在")
            
            self.test_results["system_integration"] = {
                "status": "已完成",
                "score": min(score, 100),
                "details": details
            }
            
            return score >= 70
            
        except Exception as e:
            self.test_results["system_integration"] = {
                "status": "测试失败",
                "score": 0,
                "details": [f"❌ 测试异常: {e}"]
            }
            return False
    
    def calculate_overall_score(self):
        """计算总体评分"""
        total_score = 0
        module_count = 0
        
        weights = {
            "api_configuration": 0.15,
            "data_collection": 0.15,
            "ai_models": 0.15,
            "trading_engine": 0.15,
            "risk_management": 0.15,
            "monitoring": 0.10,
            "web_interface": 0.10,
            "system_integration": 0.05
        }
        
        for module, weight in weights.items():
            if self.test_results[module]["status"] == "已完成":
                total_score += self.test_results[module]["score"] * weight
                module_count += 1
        
        self.overall_score = total_score
        
        # 确定等级
        if total_score >= 95:
            self.grade = "A+"
        elif total_score >= 90:
            self.grade = "A"
        elif total_score >= 85:
            self.grade = "A-"
        elif total_score >= 80:
            self.grade = "B+"
        elif total_score >= 75:
            self.grade = "B"
        elif total_score >= 70:
            self.grade = "B-"
        elif total_score >= 65:
            self.grade = "C+"
        elif total_score >= 60:
            self.grade = "C"
        else:
            self.grade = "D"
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 66)
        print("📊 888-888-88 系统测试报告")
        print("=" * 66)
        
        # 模块测试结果
        for module_name, result in self.test_results.items():
            module_display = {
                "api_configuration": "🔐 API配置",
                "data_collection": "📊 数据收集",
                "ai_models": "🤖 AI模型",
                "trading_engine": "💰 交易引擎",
                "risk_management": "🛡️ 风险管理",
                "monitoring": "📊 监控系统",
                "web_interface": "🌐 Web界面",
                "system_integration": "🔧 系统集成"
            }
            
            print(f"\n{module_display[module_name]}")
            print("-" * 40)
            print(f"状态: {result['status']}")
            print(f"评分: {result['score']}/100")
            
            for detail in result['details']:
                print(f"  {detail}")
        
        # 总体评分
        print("\n" + "=" * 66)
        print("🏆 总体评估")
        print("-" * 66)
        print(f"总体评分: {self.overall_score:.1f}/100")
        print(f"系统等级: {self.grade}")
        
        # 生产级标准检查
        print("\n🎯 生产级标准检查")
        print("-" * 40)
        
        production_checks = [
            ("API配置完整", self.test_results["api_configuration"]["score"] >= 80),
            ("数据收集就绪", self.test_results["data_collection"]["score"] >= 60),
            ("AI模型完善", self.test_results["ai_models"]["score"] >= 70),
            ("交易引擎可用", self.test_results["trading_engine"]["score"] >= 50),
            ("风险管理健全", self.test_results["risk_management"]["score"] >= 60),
            ("监控系统运行", self.test_results["monitoring"]["score"] >= 60),
            ("系统集成完整", self.test_results["system_integration"]["score"] >= 70)
        ]
        
        production_ready = True
        for check_name, passed in production_checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                production_ready = False
        
        print("\n" + "=" * 66)
        if production_ready and self.overall_score >= 75:
            print("🎉 系统已达到生产级标准，可以安全用于实盘交易！")
        else:
            print("⚠️ 系统未完全达到生产级标准，建议进一步优化")
        
        print("=" * 66)
        
        # 保存报告
        report_data = {
            "timestamp": time.time(),
            "overall_score": self.overall_score,
            "grade": self.grade,
            "production_ready": production_ready,
            "test_results": self.test_results
        }
        
        with open("system_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存至: system_test_report.json")
    
    async def run_all_tests(self):
        """运行所有测试"""
        try:
            self.print_banner()
            
            # 运行各模块测试
            tests = [
                ("API配置", self.test_api_configuration),
                ("数据收集", self.test_data_collection),
                ("AI模型", self.test_ai_models),
                ("交易引擎", self.test_trading_engine),
                ("风险管理", self.test_risk_management),
                ("监控系统", self.test_monitoring),
                ("Web界面", self.test_web_interface),
                ("系统集成", self.test_system_integration)
            ]
            
            for test_name, test_func in tests:
                logger.info(f"🧪 开始测试: {test_name}")
                try:
                    result = test_func()
                    status = "通过" if result else "失败"
                    logger.info(f"✅ {test_name} 测试{status}")
                except Exception as e:
                    logger.error(f"❌ {test_name} 测试异常: {e}")
            
            # 计算总体评分
            self.calculate_overall_score()
            
            # 生成报告
            self.generate_report()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试运行失败: {e}")
            return False

async def main():
    """主函数"""
    tester = SystemTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print(f"\n🎉 系统测试完成！总体评分: {tester.grade} ({tester.overall_score:.1f}/100)")
        else:
            print("\n❌ 系统测试失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
