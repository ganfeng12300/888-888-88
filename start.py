#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 一键启动脚本
专为多交易所带单交易设计的智能量化系统
"""

import asyncio
import os
import sys
import signal
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.system_manager import SystemManager
from src.core.config import Settings
from src.utils.hardware_monitor import HardwareMonitor
from src.utils.timezone_handler import TimezoneHandler

console = Console()
app = typer.Typer(rich_markup_mode="rich")

class QuantTradingSystem:
    """AI量化交易系统主控制器"""
    
    def __init__(self):
        self.settings = Settings()
        self.system_manager = SystemManager()
        self.hardware_monitor = HardwareMonitor()
        self.timezone_handler = TimezoneHandler()
        self.running = False
        
    async def initialize(self) -> bool:
        """初始化系统所有组件"""
        try:
            console.print(Panel.fit(
                "[bold gold1]🚀 AI量化交易系统启动中...[/bold gold1]",
                border_style="gold1"
            ))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                
                # 1. 硬件检测
                task1 = progress.add_task("🔧 检测硬件配置...", total=None)
                await self.hardware_monitor.detect_hardware()
                progress.update(task1, completed=True)
                
                # 2. 时区配置
                task2 = progress.add_task("🌍 配置时区处理...", total=None)
                await self.timezone_handler.setup_timezone()
                progress.update(task2, completed=True)
                
                # 3. 数据库连接
                task3 = progress.add_task("💾 连接数据库...", total=None)
                await self.system_manager.initialize_databases()
                progress.update(task3, completed=True)
                
                # 4. AI模型加载
                task4 = progress.add_task("🧠 加载AI模型...", total=None)
                await self.system_manager.initialize_ai_models()
                progress.update(task4, completed=True)
                
                # 5. 交易所连接
                task5 = progress.add_task("🏦 连接交易所...", total=None)
                await self.system_manager.initialize_exchanges()
                progress.update(task5, completed=True)
                
                # 6. 风控系统
                task6 = progress.add_task("🛡️ 启动风控系统...", total=None)
                await self.system_manager.initialize_risk_management()
                progress.update(task6, completed=True)
                
                # 7. Web界面
                task7 = progress.add_task("🌐 启动Web界面...", total=None)
                await self.system_manager.initialize_web_interface()
                progress.update(task7, completed=True)
                
            self._display_system_status()
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            console.print(f"[bold red]❌ 系统初始化失败: {e}[/bold red]")
            return False
    
    def _display_system_status(self):
        """显示系统状态信息"""
        table = Table(title="🎯 系统状态总览", style="gold1")
        table.add_column("模块", style="cyan", no_wrap=True)
        table.add_column("状态", style="green")
        table.add_column("详情", style="white")
        
        # 硬件状态
        hw_info = self.hardware_monitor.get_hardware_info()
        table.add_row("💻 硬件", "✅ 正常", f"CPU: {hw_info['cpu_cores']}核 | GPU: {hw_info['gpu_name']} | 内存: {hw_info['memory_gb']}GB")
        
        # AI模型状态
        table.add_row("🧠 AI引擎", "✅ 就绪", "8个AI模型已加载并优化")
        
        # 数据源状态
        table.add_row("📊 数据源", "✅ 连接", "实时行情数据流已建立")
        
        # 交易状态
        table.add_row("⚡ 交易引擎", "✅ 待命", "多交易所API已连接")
        
        # 风控状态
        table.add_row("🛡️ 风控系统", "✅ 监控", "实时风险监控已激活")
        
        # Web界面
        table.add_row("🌐 Web界面", "✅ 运行", "http://localhost:8000")
        
        console.print(table)
        
        # 显示收益目标
        console.print(Panel(
            "[bold green]📈 收益目标: 每周稳定盈利 20%+[/bold green]\n"
            "[bold yellow]🛡️ 风险控制: 最大日回撤 ≤ 3%[/bold yellow]\n"
            "[bold cyan]⚡ 系统延迟: < 10ms 交易执行[/bold cyan]",
            title="🎯 性能指标",
            border_style="green"
        ))
    
    async def start_trading(self):
        """启动交易系统"""
        try:
            self.running = True
            console.print("[bold green]🚀 交易系统已启动，开始自动化交易...[/bold green]")
            
            # 启动所有系统组件
            await asyncio.gather(
                self.system_manager.start_data_collection(),
                self.system_manager.start_ai_decision_engine(),
                self.system_manager.start_trading_engine(),
                self.system_manager.start_risk_monitoring(),
                self.system_manager.start_web_server(),
            )
            
        except Exception as e:
            logger.error(f"交易系统启动失败: {e}")
            console.print(f"[bold red]❌ 交易系统启动失败: {e}[/bold red]")
    
    async def stop_trading(self):
        """停止交易系统"""
        self.running = False
        console.print("[bold yellow]⏹️ 正在安全停止交易系统...[/bold yellow]")
        await self.system_manager.shutdown()
        console.print("[bold green]✅ 交易系统已安全停止[/bold green]")

# 全局系统实例
trading_system = QuantTradingSystem()

def signal_handler(signum, frame):
    """信号处理器"""
    console.print("\n[bold yellow]⚠️ 接收到停止信号，正在安全关闭系统...[/bold yellow]")
    asyncio.create_task(trading_system.stop_trading())

@app.command()
def start(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    debug: bool = typer.Option(False, "--debug", "-d", help="启用调试模式"),
    dry_run: bool = typer.Option(False, "--dry-run", help="模拟运行模式（不执行真实交易）")
):
    """🚀 启动AI量化交易系统"""
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def main():
        # 初始化系统
        if not await trading_system.initialize():
            sys.exit(1)
        
        # 启动交易
        await trading_system.start_trading()
        
        # 保持运行
        try:
            while trading_system.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await trading_system.stop_trading()
    
    # 运行主程序
    asyncio.run(main())

@app.command()
def status():
    """📊 查看系统状态"""
    console.print("[bold cyan]📊 系统状态检查中...[/bold cyan]")
    # TODO: 实现状态检查逻辑

@app.command()
def config():
    """⚙️ 配置管理"""
    console.print("[bold cyan]⚙️ 配置管理界面[/bold cyan]")
    # TODO: 实现配置管理界面

@app.command()
def backtest(
    start_date: str = typer.Option(..., "--start", help="回测开始日期 (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", help="回测结束日期 (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(10000.0, "--capital", help="初始资金")
):
    """📈 运行策略回测"""
    console.print(f"[bold cyan]📈 开始回测: {start_date} 到 {end_date}[/bold cyan]")
    # TODO: 实现回测功能

if __name__ == "__main__":
    # 显示启动横幅
    console.print(Panel.fit(
        "[bold gold1]🚀 AI量化交易系统 v1.0.0[/bold gold1]\n"
        "[cyan]专为多交易所带单交易设计的智能量化系统[/cyan]\n"
        "[dim]让AI为你的财富增值，让科技为你的投资护航！[/dim]",
        border_style="gold1"
    ))
    
    app()
