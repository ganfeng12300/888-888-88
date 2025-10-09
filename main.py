#!/usr/bin/env python3
"""
🚀 量化交易系统主程序 - 生产级实盘交易系统
Quantitative Trading System Main Program - Production-Grade Live Trading System

生产级特性：
- 完整的系统集成
- 实盘交易执行
- 多策略并行运行
- 实时风险控制
- 性能监控优化
"""

import os
import sys
import time
import signal
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from src.monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory
from src.exchanges.bitget_api import BitgetAPI, BitgetConfig
from src.trading.advanced_trading_engine import AdvancedTradingEngine
from src.trading.strategy_manager import StrategyManager, StrategyConfig
from src.risk.enhanced_risk_manager import EnhancedRiskManager
from src.ai.ai_engine import AIEngine
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.business.license_manager import LicenseManager
from src.ui.web_dashboard import WebDashboard
from src.deployment.docker_manager import DockerManager

class TradingSystemMain:
    """交易系统主类"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config)
        
        # 系统组件
        self.bitget_api = None
        self.trading_engine = None
        self.strategy_manager = None
        self.risk_manager = None
        self.ai_engine = None
        self.performance_optimizer = None
        self.license_manager = None
        self.web_dashboard = None
        
        # 运行状态
        self.running = False
        self.shutdown_event = threading.Event()
        
        # 配置
        self.config = self._load_config()
        
        self.logger.info("交易系统主程序初始化完成")
    
    def _load_config(self) -> Dict:
        """加载配置"""
        return {
            # Bitget API配置
            'bitget': {
                'api_key': 'bg_361f925c6f2139ad15bff1e662995fdd',
                'secret_key': '6b9f6868b5c6e90b4a866d1a626c3722a169e557dfcfd2175fbeb5fa84085c43',
                'passphrase': 'Ganfeng321',
                'sandbox': False
            },
            
            # 交易配置
            'trading': {
                'default_symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT'],
                'max_position_size': 10000,  # USDT
                'max_daily_loss': 1000,      # USDT
                'risk_level': 'medium'
            },
            
            # 策略配置
            'strategies': {
                'ma_cross': {
                    'enabled': True,
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'short_window': 10,
                    'long_window': 30,
                    'position_size': 1000
                },
                'rsi_reversal': {
                    'enabled': True,
                    'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                    'rsi_period': 14,
                    'oversold': 30,
                    'overbought': 70,
                    'position_size': 800
                }
            },
            
            # Web界面配置
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            }
        }
    
    def initialize_components(self):
        """初始化系统组件"""
        try:
            self.logger.info("开始初始化系统组件...")
            
            # 1. 初始化许可证管理器
            self.logger.info("初始化许可证管理器...")
            self.license_manager = LicenseManager()
            self.license_manager.start_validation_service()
            
            # 检查许可证
            if not self.license_manager.check_feature_access('basic_trading'):
                self.logger.error("许可证验证失败，无法启动交易功能")
                return False
            
            # 2. 初始化Bitget API
            self.logger.info("初始化Bitget API...")
            bitget_config = BitgetConfig(
                api_key=self.config['bitget']['api_key'],
                secret_key=self.config['bitget']['secret_key'],
                passphrase=self.config['bitget']['passphrase'],
                sandbox=self.config['bitget']['sandbox']
            )
            
            self.bitget_api = BitgetAPI(bitget_config)
            
            # 测试API连接
            if not self.bitget_api.test_connectivity():
                self.logger.error("Bitget API连接失败")
                return False
            
            # 3. 初始化风险管理器
            self.logger.info("初始化风险管理器...")
            self.risk_manager = EnhancedRiskManager()
            self.risk_manager.start_monitoring()
            
            # 4. 初始化AI系统
            self.logger.info("初始化AI系统...")
            self.ai_system = AISystem()
            self.ai_system.start_services()
            
            # 5. 初始化交易引擎
            self.logger.info("初始化交易引擎...")
            self.trading_engine = AdvancedTradingEngine()
            
            # 将Bitget API集成到交易引擎
            self._integrate_bitget_api()
            
            self.trading_engine.start()
            
            # 6. 初始化策略管理器
            self.logger.info("初始化策略管理器...")
            self.strategy_manager = StrategyManager()
            
            # 注册策略
            self._register_strategies()
            
            # 7. 初始化性能优化器
            self.logger.info("初始化性能优化器...")
            self.performance_optimizer = PerformanceOptimizer()
            self.performance_optimizer.start_monitoring()
            
            # 8. 初始化Web仪表板
            self.logger.info("初始化Web仪表板...")
            self.web_dashboard = WebDashboard(
                trading_engine=self.trading_engine,
                strategy_manager=self.strategy_manager,
                risk_manager=self.risk_manager
            )
            
            self.logger.info("所有系统组件初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化系统组件失败: {e}")
            return False
    
    def _integrate_bitget_api(self):
        """集成Bitget API到交易引擎"""
        try:
            # 替换交易引擎的市场数据管理器
            original_get_current_price = self.trading_engine.market_data.get_current_price
            
            def get_current_price_from_bitget(symbol):
                try:
                    # 转换符号格式
                    bitget_symbol = symbol.replace('/', '')
                    ticker = self.bitget_api.get_ticker(bitget_symbol)
                    if ticker and 'close' in ticker:
                        return float(ticker['close'])
                    else:
                        # 回退到原始方法
                        return original_get_current_price(symbol)
                except Exception as e:
                    self.logger.error(f"从Bitget获取价格失败: {e}")
                    return original_get_current_price(symbol)
            
            # 替换方法
            self.trading_engine.market_data.get_current_price = get_current_price_from_bitget
            
            # 替换下单方法
            original_place_order = self.trading_engine.place_order
            
            def place_order_on_bitget(symbol, side, order_type, quantity, price=None, **kwargs):
                try:
                    # 检查许可证
                    if not self.license_manager.check_feature_access('basic_trading'):
                        self.logger.error("许可证不允许交易")
                        return None
                    
                    # 风险检查
                    if not self._pre_trade_risk_check(symbol, side, quantity, price):
                        self.logger.warning("风险检查失败，拒绝下单")
                        return None
                    
                    # 转换符号格式
                    bitget_symbol = symbol.replace('/', '')
                    
                    # 在Bitget上下单
                    result = self.bitget_api.place_order(
                        symbol=bitget_symbol,
                        side=side,
                        order_type=order_type,
                        size=str(quantity),
                        price=str(price) if price else None
                    )
                    
                    if result and 'orderId' in result:
                        self.logger.info(f"Bitget下单成功: {result['orderId']}")
                        return result['orderId']
                    else:
                        self.logger.error("Bitget下单失败")
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Bitget下单异常: {e}")
                    return None
            
            # 替换方法
            self.trading_engine.place_order = place_order_on_bitget
            
            self.logger.info("Bitget API集成完成")
            
        except Exception as e:
            self.logger.error(f"集成Bitget API失败: {e}")
    
    def _pre_trade_risk_check(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """交易前风险检查"""
        try:
            # 获取账户余额
            balance = self.bitget_api.get_balance()
            if not balance:
                return False
            
            # 计算交易金额
            current_price = price or self.bitget_api.get_ticker(symbol.replace('/', '')).get('close', 0)
            if not current_price:
                return False
            
            trade_amount = quantity * float(current_price)
            
            # 检查最大持仓限制
            if trade_amount > self.config['trading']['max_position_size']:
                self.logger.warning(f"交易金额超过最大持仓限制: {trade_amount}")
                return False
            
            # 使用风险管理器进行检查
            risk_assessment = self.risk_manager.assess_trade_risk({
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'amount': trade_amount
            })
            
            return risk_assessment.get('approved', False)
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return False
    
    def _register_strategies(self):
        """注册交易策略"""
        try:
            # 注册移动平均策略
            if self.config['strategies']['ma_cross']['enabled']:
                ma_config = StrategyConfig(
                    strategy_id="ma_cross_001",
                    name="移动平均交叉策略",
                    description="基于短期和长期移动平均线交叉的交易策略",
                    parameters=self.config['strategies']['ma_cross'],
                    risk_limits={
                        'max_position_size': self.config['strategies']['ma_cross']['position_size'],
                        'max_daily_loss': 500
                    }
                )
                self.strategy_manager.register_strategy(ma_config)
            
            # 注册RSI策略
            if self.config['strategies']['rsi_reversal']['enabled']:
                rsi_config = StrategyConfig(
                    strategy_id="rsi_reversal_001",
                    name="RSI反转策略",
                    description="基于RSI指标的超买超卖反转策略",
                    parameters=self.config['strategies']['rsi_reversal'],
                    risk_limits={
                        'max_position_size': self.config['strategies']['rsi_reversal']['position_size'],
                        'max_daily_loss': 400
                    }
                )
                self.strategy_manager.register_strategy(rsi_config)
            
            self.logger.info("交易策略注册完成")
            
        except Exception as e:
            self.logger.error(f"注册策略失败: {e}")
    
    def start_market_data_feed(self):
        """启动市场数据推送"""
        try:
            self.logger.info("启动市场数据推送...")
            
            def on_ticker_update(data):
                try:
                    if isinstance(data, list) and len(data) > 0:
                        ticker_data = data[0]
                        symbol = ticker_data.get('instId', '')
                        price = float(ticker_data.get('last', 0))
                        
                        if symbol and price > 0:
                            # 转换为标准格式
                            standard_symbol = symbol
                            
                            # 推送给策略管理器
                            market_data = {
                                'symbol': standard_symbol,
                                'price': price,
                                'timestamp': datetime.now()
                            }
                            
                            self.strategy_manager.process_market_data(market_data)
                            
                except Exception as e:
                    self.logger.error(f"处理ticker数据失败: {e}")
            
            # 订阅主要交易对的实时数据
            for symbol in self.config['trading']['default_symbols']:
                self.bitget_api.subscribe_ticker(symbol, on_ticker_update)
                time.sleep(0.1)  # 避免频率限制
            
            self.logger.info("市场数据推送启动完成")
            
        except Exception as e:
            self.logger.error(f"启动市场数据推送失败: {e}")
    
    def start_web_interface(self):
        """启动Web界面"""
        try:
            self.logger.info("启动Web界面...")
            
            # 在新线程中启动Web服务器
            web_thread = threading.Thread(
                target=self.web_dashboard.run,
                kwargs={
                    'host': self.config['web']['host'],
                    'port': self.config['web']['port'],
                    'debug': self.config['web']['debug']
                },
                daemon=True
            )
            web_thread.start()
            
            self.logger.info(f"Web界面已启动: http://{self.config['web']['host']}:{self.config['web']['port']}")
            
        except Exception as e:
            self.logger.error(f"启动Web界面失败: {e}")
    
    def run(self):
        """运行交易系统"""
        try:
            self.logger.info("🚀 启动量化交易系统...")
            
            # 初始化组件
            if not self.initialize_components():
                self.logger.error("系统初始化失败")
                return False
            
            # 启动市场数据推送
            self.start_market_data_feed()
            
            # 启动Web界面
            self.start_web_interface()
            
            # 设置信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.running = True
            
            self.logger.info("✅ 量化交易系统启动成功！")
            self.logger.info("=" * 60)
            self.logger.info("🎯 系统状态:")
            self.logger.info(f"  📊 Web界面: http://localhost:{self.config['web']['port']}")
            self.logger.info(f"  🔗 Bitget API: 已连接")
            self.logger.info(f"  📈 交易引擎: 运行中")
            self.logger.info(f"  🎯 策略管理: 运行中")
            self.logger.info(f"  🛡️ 风险控制: 运行中")
            self.logger.info(f"  🤖 AI系统: 运行中")
            self.logger.info("=" * 60)
            
            # 主循环
            while self.running and not self.shutdown_event.is_set():
                try:
                    # 系统健康检查
                    self._health_check()
                    
                    # 等待
                    self.shutdown_event.wait(30)  # 每30秒检查一次
                    
                except Exception as e:
                    self.logger.error(f"主循环异常: {e}")
                    time.sleep(5)
            
            self.logger.info("交易系统正在关闭...")
            return True
            
        except Exception as e:
            self.logger.error(f"运行交易系统失败: {e}")
            return False
        
        finally:
            self.shutdown()
    
    def _health_check(self):
        """系统健康检查"""
        try:
            # 检查API连接
            if not self.bitget_api.test_connectivity():
                self.logger.warning("Bitget API连接异常")
            
            # 检查许可证状态
            license_status = self.license_manager.get_license_status()
            if license_status.get('status') != 'active':
                self.logger.warning(f"许可证状态异常: {license_status.get('status')}")
            
            # 获取系统性能报告
            performance_report = self.performance_optimizer.get_performance_report()
            if 'error' not in performance_report:
                health_score = performance_report.get('system_health', {}).get('score', 0)
                if health_score < 60:
                    self.logger.warning(f"系统健康评分较低: {health_score}")
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，准备关闭系统...")
        self.running = False
        self.shutdown_event.set()
    
    def shutdown(self):
        """关闭系统"""
        try:
            self.logger.info("开始关闭系统组件...")
            
            # 关闭各个组件
            if self.performance_optimizer:
                self.performance_optimizer.stop_monitoring()
            
            if self.strategy_manager:
                # 停止所有策略
                for strategy_id in self.strategy_manager.strategies.keys():
                    self.strategy_manager.stop_strategy(strategy_id)
            
            if self.trading_engine:
                self.trading_engine.stop()
            
            if self.ai_system:
                self.ai_system.stop_services()
            
            if self.risk_manager:
                self.risk_manager.stop_monitoring()
            
            if self.bitget_api:
                self.bitget_api.close_websocket()
            
            if self.license_manager:
                self.license_manager.stop_validation_service()
            
            self.logger.info("系统关闭完成")
            
        except Exception as e:
            self.logger.error(f"关闭系统失败: {e}")

def main():
    """主函数"""
    print("🚀 量化交易系统 v1.0")
    print("=" * 50)
    
    # 创建交易系统实例
    trading_system = TradingSystemMain()
    
    try:
        # 运行系统
        success = trading_system.run()
        
        if success:
            print("✅ 系统运行完成")
        else:
            print("❌ 系统运行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，正在关闭系统...")
        trading_system.shutdown()
    except Exception as e:
        print(f"❌ 系统异常: {e}")
        trading_system.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
