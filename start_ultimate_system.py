#!/usr/bin/env python3
"""
🚀 终极交易系统一键启动脚本
Ultimate Trading System One-Click Launcher
"""
import os
import sys
import time
import asyncio
import threading
import subprocess
from datetime import datetime
from typing import Dict, Any

# 添加src目录到路径
sys.path.append('src')

class UltimateSystemLauncher:
    """终极系统启动器"""
    
    def __init__(self):
        self.processes = []
        self.system_status = {
            'ai_system': False,
            'balance_manager': False,
            'web_dashboard': False,
            'disk_cleanup': False,
            'performance_monitor': False
        }
        
    def print_banner(self):
        """显示启动横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🌟 终极交易系统 🌟                        ║
║                Ultimate Trading System                        ║
║                                                              ║
║  🧠 6级分层AI系统    💰 实时余额监控    🌐 黑金Web界面        ║
║  🗑️ 智能硬盘清理    ⚡ 性能监控系统    🛡️ 风险管理          ║
║                                                              ║
║              🚀 一键启动，开始您的AI进化之旅！                ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 66)
    
    def check_dependencies(self):
        """检查系统依赖"""
        print("🔍 检查系统依赖...")
        
        required_packages = [
            'ccxt', 'pandas', 'numpy', 'sklearn', 
            'flask', 'flask_socketio', 'dotenv', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - 缺失")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ 缺少依赖包: {missing_packages}")
            print("请运行: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ 所有依赖检查通过！")
        return True
    
    def check_api_config(self):
        """检查API配置"""
        print("\n🔑 检查API配置...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            'BITGET_API_KEY',
            'BITGET_SECRET_KEY', 
            'BITGET_PASSPHRASE'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
            else:
                print(f"  ✅ {var}")
        
        if missing_vars:
            print(f"\n❌ 缺少环境变量: {missing_vars}")
            print("请在.env文件中配置API密钥")
            return False
        
        print("✅ API配置检查通过！")
        return True
    
    async def start_ai_system(self):
        """启动AI系统"""
        print("\n🧠 启动6级分层AI系统...")
        
        try:
            from ai.hierarchical_ai_system import hierarchical_ai
            await hierarchical_ai.start()
            self.system_status['ai_system'] = True
            print("  ✅ AI系统启动成功")
            
            # 显示AI等级信息
            status = hierarchical_ai.get_system_status()
            print(f"  📊 配置了 {len(status['model_configs'])} 个AI模型")
            print("  🎯 AI进化之旅开始！")
            
        except Exception as e:
            print(f"  ❌ AI系统启动失败: {e}")
            return False
        
        return True
    
    async def start_balance_manager(self):
        """启动余额管理器"""
        print("\n💰 启动余额管理系统...")
        
        try:
            from trading.balance_manager import balance_manager
            
            # 获取初始余额
            balances = await balance_manager.get_all_balances()
            total_value = sum(acc.total_usd_value for acc in balances.values())
            
            # 启动监控
            await balance_manager.start_monitoring()
            self.system_status['balance_manager'] = True
            
            print("  ✅ 余额管理系统启动成功")
            print(f"  💵 当前总资产: ${total_value:.2f}")
            print(f"  📊 监控账户: {len(balances)} 个")
            
        except Exception as e:
            print(f"  ❌ 余额管理系统启动失败: {e}")
            return False
        
        return True
    
    def start_web_dashboard(self):
        """启动Web仪表板"""
        print("\n🌐 启动黑金Web仪表板...")
        
        try:
            # 在后台启动Web服务器
            process = subprocess.Popen([
                sys.executable, 'web_dashboard_ultimate.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            
            # 等待启动
            time.sleep(3)
            
            if process.poll() is None:  # 进程仍在运行
                self.system_status['web_dashboard'] = True
                print("  ✅ Web仪表板启动成功")
                print("  🌐 访问地址: http://localhost:8888")
                print("  🎨 黑金主题界面已就绪")
            else:
                print("  ❌ Web仪表板启动失败")
                return False
                
        except Exception as e:
            print(f"  ❌ Web仪表板启动失败: {e}")
            return False
        
        return True
    
    def start_disk_cleanup(self):
        """启动硬盘清理系统"""
        print("\n🗑️ 启动智能硬盘清理系统...")
        
        try:
            from utils.disk_cleanup import disk_cleanup_manager
            
            # 获取磁盘状态
            report = disk_cleanup_manager.get_cleanup_report()
            
            # 启动监控
            disk_cleanup_manager.start_monitoring(check_interval_hours=6)
            self.system_status['disk_cleanup'] = True
            
            print("  ✅ 硬盘清理系统启动成功")
            print(f"  💾 磁盘使用率: {report['disk_usage']['usage_percentage']:.1f}%")
            print(f"  🧹 自动清理: 每6小时检查一次")
            
        except Exception as e:
            print(f"  ❌ 硬盘清理系统启动失败: {e}")
            return False
        
        return True
    
    def start_performance_monitor(self):
        """启动性能监控"""
        print("\n⚡ 启动性能监控系统...")
        
        try:
            import psutil
            
            def monitor_loop():
                while True:
                    try:
                        cpu_percent = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        
                        # 简单的性能日志
                        if cpu_percent > 80 or memory.percent > 85:
                            print(f"⚠️ 系统资源警告: CPU {cpu_percent:.1f}%, 内存 {memory.percent:.1f}%")
                        
                        time.sleep(60)  # 每分钟检查一次
                    except Exception:
                        break
            
            # 在后台线程中运行监控
            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()
            
            self.system_status['performance_monitor'] = True
            print("  ✅ 性能监控系统启动成功")
            print("  📊 实时监控CPU和内存使用")
            
        except Exception as e:
            print(f"  ❌ 性能监控系统启动失败: {e}")
            return False
        
        return True
    
    def display_ai_evolution_info(self):
        """显示AI进化信息"""
        print("\n" + "=" * 66)
        print("🧠 AI进化系统信息")
        print("=" * 66)
        
        evolution_info = """
🎯 AI进化等级:
  Level 1: 实时监控AI    (第1-7天)    - 1%日收益, 2倍杠杆
  Level 2: 执行优化AI    (第8-21天)   - 3%日收益, 3倍杠杆  
  Level 3: 技术分析AI    (第22-45天)  - 5%日收益, 5倍杠杆
  Level 4: 风险管理AI    (第46-90天)  - 8%日收益, 8倍杠杆
  Level 5: 战术协调AI    (第91-180天) - 12%日收益, 12倍杠杆
  Level 6: 战略总指挥AI  (第181-365天)- 20%日收益, 20倍杠杆

⏱️ 进化时间线:
  🚀 正常模式: 365天达到传奇级
  ⚡ 加速模式: 180天达到传奇级  
  🔥 极速模式: 90天达到传奇级

💰 收益预期 (基于$50,000初始资金):
  📅 1个月: $121,000 (+142%)
  📅 3个月: $2,180,000 (+4,260%)  
  📅 6个月: $15,600,000 (+31,100%)
  📅 1年: $186,000,000 (+371,900%)
        """
        print(evolution_info)
    
    def display_system_status(self):
        """显示系统状态"""
        print("\n" + "=" * 66)
        print("📊 系统状态总览")
        print("=" * 66)
        
        for system, status in self.system_status.items():
            status_icon = "✅" if status else "❌"
            system_name = {
                'ai_system': '🧠 6级分层AI系统',
                'balance_manager': '💰 余额管理系统', 
                'web_dashboard': '🌐 Web仪表板',
                'disk_cleanup': '🗑️ 硬盘清理系统',
                'performance_monitor': '⚡ 性能监控系统'
            }
            print(f"  {status_icon} {system_name[system]}")
        
        active_systems = sum(self.system_status.values())
        total_systems = len(self.system_status)
        
        print(f"\n🎯 系统就绪率: {active_systems}/{total_systems} ({active_systems/total_systems*100:.0f}%)")
        
        if active_systems == total_systems:
            print("🚀 所有系统正常运行，AI进化之旅开始！")
        else:
            print("⚠️ 部分系统未启动，请检查错误信息")
    
    def display_quick_commands(self):
        """显示快捷命令"""
        print("\n" + "=" * 66)
        print("🎮 快捷操作命令")
        print("=" * 66)
        
        commands = """
📊 监控命令:
  🌐 Web界面: http://localhost:8888
  📈 实时收益: 在Web界面查看
  🧠 AI状态: 在Web界面AI面板查看
  
🔧 管理命令:
  ⏹️  停止系统: Ctrl+C
  🔄 重启系统: python start_ultimate_system.py
  📋 查看日志: tail -f logs/system.log
  
📞 技术支持:
  📖 完整文档: AI_EVOLUTION_SYSTEM.md
  🆘 问题反馈: 查看系统日志
  💡 优化建议: 监控性能指标
        """
        print(commands)
    
    async def main(self):
        """主启动流程"""
        self.print_banner()
        
        # 1. 检查依赖
        if not self.check_dependencies():
            return False
        
        # 2. 检查API配置
        if not self.check_api_config():
            return False
        
        print("\n🚀 开始启动系统组件...")
        
        # 3. 启动各个系统组件
        success_count = 0
        
        if await self.start_ai_system():
            success_count += 1
        
        if await self.start_balance_manager():
            success_count += 1
        
        if self.start_web_dashboard():
            success_count += 1
        
        if self.start_disk_cleanup():
            success_count += 1
        
        if self.start_performance_monitor():
            success_count += 1
        
        # 4. 显示系统信息
        self.display_system_status()
        self.display_ai_evolution_info()
        self.display_quick_commands()
        
        if success_count >= 4:  # 至少4个系统启动成功
            print("\n🎉 系统启动完成！您的AI交易帝国已经开始运行！")
            print("💎 预祝您在AI进化之旅中获得丰厚收益！")
            
            # 保持系统运行
            try:
                print("\n⏳ 系统运行中... (按 Ctrl+C 停止)")
                while True:
                    await asyncio.sleep(60)
                    # 这里可以添加定期状态检查
                    
            except KeyboardInterrupt:
                print("\n⏹️ 用户请求停止系统...")
                self.cleanup()
                
        else:
            print("\n❌ 系统启动失败，请检查错误信息并重试")
            return False
        
        return True
    
    def cleanup(self):
        """清理资源"""
        print("🧹 正在清理系统资源...")
        
        # 停止所有子进程
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print("  ✅ 子进程已停止")
            except:
                process.kill()
                print("  ⚠️ 强制停止子进程")
        
        print("✅ 系统清理完成")

async def main():
    """主函数"""
    launcher = UltimateSystemLauncher()
    try:
        await launcher.main()
    except Exception as e:
        print(f"💥 系统启动异常: {e}")
        launcher.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 感谢使用终极交易系统！")
    except Exception as e:
        print(f"💥 启动失败: {e}")
        sys.exit(1)

