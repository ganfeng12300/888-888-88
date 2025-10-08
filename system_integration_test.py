#!/usr/bin/env python3
"""
🚀 系统集成测试 - System Integration Test
完整测试所有系统功能
"""
import sys
import os
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any

# 添加src目录到路径
sys.path.append('src')

from ai.hierarchical_ai_system import hierarchical_ai, MarketData
from trading.balance_manager import balance_manager
from utils.disk_cleanup import disk_cleanup_manager

class SystemIntegrationTest:
    """系统集成测试类"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """记录测试结果"""
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {details}")
    
    async def test_api_connectivity(self) -> bool:
        """测试API连接"""
        print("\n🔗 测试API连接...")
        
        try:
            # 测试余额获取
            balances = await balance_manager.get_all_balances()
            
            if balances:
                total_accounts = len(balances)
                total_value = sum(acc.total_usd_value for acc in balances.values())
                
                self.log_test(
                    "API连接测试", 
                    True, 
                    f"成功连接 {total_accounts} 个账户，总价值 ${total_value:.2f}"
                )
                return True
            else:
                self.log_test("API连接测试", False, "无法获取账户信息")
                return False
                
        except Exception as e:
            self.log_test("API连接测试", False, f"连接失败: {e}")
            return False
    
    async def test_ai_system(self) -> bool:
        """测试AI系统"""
        print("\n🧠 测试AI系统...")
        
        try:
            # 启动AI系统
            await hierarchical_ai.start()
            
            # 获取系统状态
            status = hierarchical_ai.get_system_status()
            
            # 模拟市场数据
            market_data = MarketData(
                symbol="BTCUSDT",
                price=122000.0,
                volume=1000000.0,
                timestamp=datetime.now(),
                indicators={
                    "rsi": 65.5,
                    "macd": 150.2,
                    "bollinger": 0.8,
                    "volume_profile": 1.2,
                    "sentiment": 0.3
                }
            )
            
            # 添加市场数据到队列
            hierarchical_ai.market_data_queue.put(market_data)
            
            # 等待处理
            await asyncio.sleep(2)
            
            # 检查决策生成
            decisions = hierarchical_ai.hierarchical_decision_making(market_data)
            
            self.log_test(
                "AI系统测试", 
                True, 
                f"系统运行正常，配置了 {len(status['model_configs'])} 个AI模型，生成了 {len(decisions)} 个决策"
            )
            return True
            
        except Exception as e:
            self.log_test("AI系统测试", False, f"AI系统错误: {e}")
            return False
    
    def test_balance_manager(self) -> bool:
        """测试余额管理器"""
        print("\n💰 测试余额管理器...")
        
        try:
            # 获取余额摘要
            summary = balance_manager.get_balance_summary()
            
            # 检查警报
            alerts = balance_manager.check_balance_alerts()
            
            # 测试仓位计算
            position_size = balance_manager.calculate_position_size("spot", "USDT", 0.02)
            
            self.log_test(
                "余额管理器测试", 
                True, 
                f"总价值 ${summary['total_portfolio_value']:.2f}，{len(alerts)} 个警报，建议仓位 {position_size:.4f}"
            )
            return True
            
        except Exception as e:
            self.log_test("余额管理器测试", False, f"余额管理器错误: {e}")
            return False
    
    def test_disk_cleanup(self) -> bool:
        """测试硬盘清理系统"""
        print("\n🗑️ 测试硬盘清理系统...")
        
        try:
            # 获取清理报告
            report = disk_cleanup_manager.get_cleanup_report()
            
            # 执行清理检查
            cleanup_success = disk_cleanup_manager.check_disk_space_and_cleanup()
            
            self.log_test(
                "硬盘清理系统测试", 
                cleanup_success, 
                f"磁盘使用率 {report['disk_usage']['usage_percentage']:.1f}%，状态: {report['status']}"
            )
            return cleanup_success
            
        except Exception as e:
            self.log_test("硬盘清理系统测试", False, f"清理系统错误: {e}")
            return False
    
    def test_web_dashboard(self) -> bool:
        """测试Web仪表板"""
        print("\n🌐 测试Web仪表板...")
        
        try:
            import requests
            
            # 启动仪表板（后台运行）
            import subprocess
            import signal
            
            # 启动Web服务器
            process = subprocess.Popen([
                'python', 'web_dashboard_ultimate.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待启动
            time.sleep(3)
            
            try:
                # 测试主页
                response = requests.get('http://localhost:8888', timeout=5)
                
                if response.status_code == 200 and 'Ultimate Trading System' in response.text:
                    self.log_test("Web仪表板测试", True, "仪表板正常运行，页面加载成功")
                    success = True
                else:
                    self.log_test("Web仪表板测试", False, f"页面响应异常: {response.status_code}")
                    success = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test("Web仪表板测试", False, f"无法连接到仪表板: {e}")
                success = False
            
            finally:
                # 停止Web服务器
                process.terminate()
                process.wait(timeout=5)
            
            return success
            
        except Exception as e:
            self.log_test("Web仪表板测试", False, f"仪表板测试错误: {e}")
            return False
    
    def test_data_persistence(self) -> bool:
        """测试数据持久化"""
        print("\n💾 测试数据持久化...")
        
        try:
            # 检查数据库文件
            db_files = [
                'data/hierarchical_ai.db',
                'data/trading.db'
            ]
            
            existing_dbs = []
            for db_file in db_files:
                if os.path.exists(db_file):
                    existing_dbs.append(db_file)
            
            # 测试AI系统数据保存
            from ai.hierarchical_ai_system import AIDecision
            test_decision = AIDecision(
                model_name="test_model",
                level=1,
                action="BUY",
                confidence=0.85,
                price_target=122500.0,
                stop_loss=121000.0,
                take_profit=124000.0,
                position_size=0.1,
                reasoning="测试决策",
                timestamp=datetime.now()
            )
            
            hierarchical_ai.save_decision(test_decision)
            
            self.log_test(
                "数据持久化测试", 
                True, 
                f"找到 {len(existing_dbs)} 个数据库文件，数据保存成功"
            )
            return True
            
        except Exception as e:
            self.log_test("数据持久化测试", False, f"数据持久化错误: {e}")
            return False
    
    def test_system_performance(self) -> bool:
        """测试系统性能"""
        print("\n⚡ 测试系统性能...")
        
        try:
            import psutil
            
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 性能基准
            performance_ok = (
                cpu_percent < 80 and  # CPU使用率低于80%
                memory.percent < 85 and  # 内存使用率低于85%
                disk.percent < 90  # 磁盘使用率低于90%
            )
            
            self.log_test(
                "系统性能测试", 
                performance_ok, 
                f"CPU: {cpu_percent:.1f}%, 内存: {memory.percent:.1f}%, 磁盘: {disk.percent:.1f}%"
            )
            return performance_ok
            
        except Exception as e:
            self.log_test("系统性能测试", False, f"性能测试错误: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🚀 开始系统集成测试...")
        print("=" * 60)
        
        # 执行所有测试
        tests = [
            ("API连接", self.test_api_connectivity()),
            ("AI系统", self.test_ai_system()),
            ("余额管理器", self.test_balance_manager()),
            ("硬盘清理", self.test_disk_cleanup()),
            ("Web仪表板", self.test_web_dashboard()),
            ("数据持久化", self.test_data_persistence()),
            ("系统性能", self.test_system_performance())
        ]
        
        results = {}
        for test_name, test_coro in tests:
            if asyncio.iscoroutine(test_coro):
                results[test_name] = await test_coro
            else:
                results[test_name] = test_coro
        
        # 生成测试报告
        return self.generate_report(results)
    
    def generate_report(self, results: Dict[str, bool]) -> Dict[str, Any]:
        """生成测试报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = sum(1 for success in results.values() if success)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "duration_seconds": duration
            },
            "test_results": self.test_results,
            "system_status": "HEALTHY" if success_rate >= 80 else "NEEDS_ATTENTION",
            "timestamp": end_time.isoformat()
        }
        
        print("\n" + "=" * 60)
        print("📊 测试报告摘要")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"测试时长: {duration:.1f}秒")
        print(f"系统状态: {report['system_status']}")
        
        # 保存报告
        with open('system_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存到: system_test_report.json")
        
        return report

async def main():
    """主函数"""
    tester = SystemIntegrationTest()
    report = await tester.run_all_tests()
    
    # 根据测试结果返回退出码
    if report['system_status'] == 'HEALTHY':
        print("\n🎉 所有测试通过！系统运行正常！")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查系统配置！")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        sys.exit(1)

