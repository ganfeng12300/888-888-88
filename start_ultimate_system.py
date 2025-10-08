#!/usr/bin/env python3
"""
🚀 终极合约交易系统启动器
Ultimate Contract Trading System Launcher

一键启动完整的AI交易系统：
- 硬件性能优化系统
- Bybit合约交易集成
- 严格风控系统
- 时区智能调度
- 六大智能体融合系统
"""

import asyncio
import json
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入各个系统模块
try:
    from src.hardware.gpu_performance_optimizer import get_gpu_optimizer
    from src.exchange.bybit_contract_trader import get_bybit_trader
    from src.risk.advanced_risk_controller import get_risk_controller
    from src.scheduler.timezone_scheduler import get_timezone_scheduler
    from src.ai.six_agents_fusion_system import get_fusion_system
    
    logger.info("✅ 所有系统模块导入成功")
except ImportError as e:
    logger.error(f"❌ 模块导入失败: {e}")
    sys.exit(1)


class UltimateSystemLauncher:
    """终极系统启动器"""
    
    def __init__(self, config_file: str = "config.json"):
        """初始化启动器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.systems = {}
        self.is_running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 终极系统启动器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"✅ 配置文件加载成功: {self.config_file}")
                return config
            else:
                logger.warning(f"⚠️ 配置文件不存在，使用默认配置: {self.config_file}")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "gpu_optimizer": {
                "target_gpu_utilization": 85.0,
                "max_memory_usage": 90.0,
                "monitoring_interval": 5,
                "optimization_interval": 30
            },
            "bybit_trader": {
                "api_key": "",
                "api_secret": "",
                "testnet": True,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "leverage": 10,
                "max_position_size": 0.1,
                "max_daily_loss": 0.03,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            },
            "risk_controller": {
                "max_daily_drawdown": 0.03,
                "max_total_drawdown": 0.15,
                "max_position_size": 0.25,
                "max_total_exposure": 0.80,
                "volatility_threshold": 0.05,
                "hard_stop_loss": 0.03,
                "monitoring_interval": 1
            },
            "timezone_scheduler": {
                "local_timezone": "Asia/Shanghai",
                "check_interval": 60,
                "activity_window": 300
            },
            "fusion_system": {
                "max_decision_history": 10000,
                "performance_window": 100,
                "weight_update_interval": 60,
                "min_confidence_threshold": 0.3
            },
            "system": {
                "web_port": 8888,
                "log_level": "INFO",
                "auto_start_trading": False,
                "status_update_interval": 10
            }
        }
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"🛑 收到信号 {signum}，正在关闭系统...")
        asyncio.create_task(self.shutdown())
    
    async def initialize_systems(self):
        """初始化所有系统"""
        logger.info("🔧 开始初始化各个系统...")
        
        try:
            # 1. 初始化GPU性能优化器
            logger.info("1️⃣ 初始化GPU性能优化器...")
            self.systems['gpu_optimizer'] = get_gpu_optimizer(
                self.config.get('gpu_optimizer', {})
            )
            logger.info("✅ GPU性能优化器初始化完成")
            
            # 2. 初始化风险控制器
            logger.info("2️⃣ 初始化风险控制器...")
            self.systems['risk_controller'] = get_risk_controller(
                self.config.get('risk_controller', {})
            )
            logger.info("✅ 风险控制器初始化完成")
            
            # 3. 初始化时区调度器
            logger.info("3️⃣ 初始化时区调度器...")
            self.systems['timezone_scheduler'] = get_timezone_scheduler(
                self.config.get('timezone_scheduler', {})
            )
            logger.info("✅ 时区调度器初始化完成")
            
            # 4. 初始化六大智能体融合系统
            logger.info("4️⃣ 初始化六大智能体融合系统...")
            self.systems['fusion_system'] = get_fusion_system(
                self.config.get('fusion_system', {})
            )
            logger.info("✅ 六大智能体融合系统初始化完成")
            
            # 5. 初始化Bybit交易器（需要API密钥）
            bybit_config = self.config.get('bybit_trader', {})
            if bybit_config.get('api_key') and bybit_config.get('api_secret'):
                logger.info("5️⃣ 初始化Bybit交易器...")
                self.systems['bybit_trader'] = get_bybit_trader(bybit_config)
                logger.info("✅ Bybit交易器初始化完成")
            else:
                logger.warning("⚠️ Bybit API密钥未配置，跳过交易器初始化")
            
            logger.info("🎉 所有系统初始化完成！")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            raise
    
    async def start_systems(self):
        """启动所有系统"""
        logger.info("🚀 开始启动各个系统...")
        
        try:
            # 启动交易器（如果已初始化）
            if 'bybit_trader' in self.systems:
                logger.info("📈 启动Bybit交易器...")
                # 注意：这里不直接调用start_trading，因为它是阻塞的
                # 实际使用时需要在单独的任务中运行
                if self.config.get('system', {}).get('auto_start_trading', False):
                    asyncio.create_task(self.systems['bybit_trader'].start_trading())
                    logger.info("✅ Bybit交易器已启动")
                else:
                    logger.info("ℹ️ 自动交易未启用，交易器处于待机状态")
            
            # 其他系统已经在初始化时自动启动了后台线程
            logger.info("🎉 所有系统启动完成！")
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            raise
    
    async def run_main_loop(self):
        """运行主循环"""
        logger.info("🔄 进入主循环...")
        self.is_running = True
        
        status_interval = self.config.get('system', {}).get('status_update_interval', 10)
        
        while self.is_running:
            try:
                # 定期输出系统状态
                await self._update_system_status()
                
                # 等待下一次更新
                await asyncio.sleep(status_interval)
                
            except Exception as e:
                logger.error(f"❌ 主循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _update_system_status(self):
        """更新系统状态"""
        try:
            status_info = []
            
            # GPU优化器状态
            if 'gpu_optimizer' in self.systems:
                gpu_report = self.systems['gpu_optimizer'].get_optimization_report()
                if 'error' not in gpu_report:
                    gpu_info = gpu_report.get('gpu_info', {})
                    if gpu_info.get('available'):
                        current_status = gpu_info.get('current_status', {})
                        memory_usage = current_status.get('memory_usage_percent', 0)
                        status_info.append(f"GPU: {memory_usage:.1f}%内存")
                    else:
                        status_info.append("GPU: 不可用")
                else:
                    status_info.append("GPU: 错误")
            
            # 风险控制器状态
            if 'risk_controller' in self.systems:
                risk_report = self.systems['risk_controller'].get_risk_report()
                if 'error' not in risk_report:
                    emergency_stop = risk_report.get('emergency_stop', False)
                    current_metrics = risk_report.get('current_metrics', {})
                    daily_drawdown = current_metrics.get('daily_drawdown', 0)
                    status_info.append(f"风控: {'🚨停止' if emergency_stop else '✅正常'} 回撤{daily_drawdown:.1%}")
                else:
                    status_info.append("风控: 错误")
            
            # 时区调度器状态
            if 'timezone_scheduler' in self.systems:
                scheduler_status = self.systems['timezone_scheduler'].get_scheduler_status()
                if 'error' not in scheduler_status:
                    current_session = scheduler_status.get('current_session', 'unknown')
                    current_mode = scheduler_status.get('current_mode', 'unknown')
                    status_info.append(f"调度: {current_session}-{current_mode}")
                else:
                    status_info.append("调度: 错误")
            
            # 融合系统状态
            if 'fusion_system' in self.systems:
                fusion_status = self.systems['fusion_system'].get_system_status()
                active_agents = fusion_status.get('active_agents', 0)
                decision_count = fusion_status.get('decision_count', 0)
                status_info.append(f"AI: {active_agents}个智能体 {decision_count}决策")
            
            # 交易器状态
            if 'bybit_trader' in self.systems:
                trading_status = self.systems['bybit_trader'].get_trading_status()
                is_running = trading_status.get('is_running', False)
                account_balance = trading_status.get('account_balance', 0)
                daily_pnl = trading_status.get('daily_pnl', 0)
                status_info.append(f"交易: {'🟢运行' if is_running else '🔴停止'} "
                                 f"余额${account_balance:.0f} PnL${daily_pnl:.0f}")
            
            # 输出状态信息
            if status_info:
                logger.info(f"📊 系统状态: {' | '.join(status_info)}")
            
        except Exception as e:
            logger.error(f"❌ 更新系统状态失败: {e}")
    
    async def shutdown(self):
        """关闭所有系统"""
        logger.info("🛑 开始关闭所有系统...")
        self.is_running = False
        
        try:
            # 关闭交易器
            if 'bybit_trader' in self.systems:
                logger.info("📈 关闭Bybit交易器...")
                self.systems['bybit_trader'].stop_trading()
            
            # 关闭其他系统
            for name, system in self.systems.items():
                if hasattr(system, 'shutdown'):
                    logger.info(f"🔧 关闭{name}...")
                    system.shutdown()
            
            logger.info("✅ 所有系统已关闭")
            
        except Exception as e:
            logger.error(f"❌ 关闭系统时出错: {e}")
    
    def save_config(self):
        """保存当前配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"❌ 保存配置失败: {e}")


async def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                🚀 终极合约交易系统 🚀                        ║
║              Ultimate Contract Trading System                ║
╚══════════════════════════════════════════════════════════════╝

🎯 系统功能:
├── 🖥️  GPU性能优化 - 20核CPU+GTX3060加速
├── 💰 Bybit合约交易 - 小资金高频策略  
├── 🛡️  严格风控系统 - 日回撤<3%保护
├── 🌍 时区智能调度 - 24/7全球优化
├── 🧠 六大智能体融合 - AI决策引擎
└── 📊 实时监控面板 - 全方位状态监控

🚀 正在启动系统...
    """)
    
    launcher = UltimateSystemLauncher()
    
    try:
        # 初始化系统
        await launcher.initialize_systems()
        
        # 启动系统
        await launcher.start_systems()
        
        print("""
✅ 系统启动完成！

📊 监控面板: http://localhost:8888
🔧 配置文件: config.json
📝 日志输出: 实时显示

💡 使用提示:
- Ctrl+C 安全关闭系统
- 修改config.json配置参数
- 查看日志了解系统状态

🎉 开始您的AI交易之旅！
        """)
        
        # 运行主循环
        await launcher.run_main_loop()
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭系统...")
    except Exception as e:
        logger.error(f"❌ 系统运行异常: {e}")
    finally:
        await launcher.shutdown()
        logger.info("👋 系统已完全关闭，再见！")


if __name__ == "__main__":
    # 设置日志格式
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO"
    )
    
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        logger.error(f"❌ 程序异常退出: {e}")
        sys.exit(1)
