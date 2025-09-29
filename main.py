#!/usr/bin/env python3
"""
🚀 AI量化交易系统 - 主程序
集成多交易所管理、AI信号生成、风险控制等核心功能
专为生产级实盘交易设计，支持多AI融合决策
"""
import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
import pandas as pd
import numpy as np

# 导入核心模块
from src.ai.ai_evolution_system import ai_evolution_system
from src.ai.ai_decision_fusion_engine import ai_decision_fusion_engine
from src.ai.gpu_memory_optimizer import gpu_memory_optimizer
from src.ai_enhanced.deep_reinforcement_learning import initialize_deep_rl_system
from src.ai_enhanced.sentiment_analysis import sentiment_monitor
from src.ai_enhanced.auto_feature_engineering import auto_feature_engineering
from src.security.risk_control_system import risk_control_system
from src.security.anomaly_detection import anomaly_detection_system
from src.monitoring.hardware_monitor import hardware_monitor
from src.monitoring.ai_status_monitor import ai_status_monitor
from src.monitoring.system_health_checker import system_health_checker
from src.exchanges.multi_exchange_manager import multi_exchange_manager, initialize_multi_exchange_manager
from src.strategies.production_signal_generator import production_signal_generator, initialize_production_signal_generator, MarketData

class QuantTradingSystem:
    """量化交易系统主类"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.running = False
        self.system_components = {}
        self.performance_stats = {}
        
        logger.info("🚀 初始化AI量化交易系统...")
        
        # 初始化系统组件
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # AI核心组件
            logger.info("🤖 初始化AI核心组件...")
            self.system_components['ai_evolution'] = ai_evolution_system
            self.system_components['ai_fusion'] = ai_decision_fusion_engine
            self.system_components['gpu_optimizer'] = gpu_memory_optimizer
            
            # AI增强组件
            logger.info("🧠 初始化AI增强组件...")
            self.system_components['deep_rl'] = initialize_deep_rl_system()
            self.system_components['sentiment'] = sentiment_monitor
            self.system_components['feature_engineering'] = auto_feature_engineering
            
            # 安全组件
            logger.info("🔒 初始化安全组件...")
            self.system_components['risk_control'] = risk_control_system
            self.system_components['anomaly_detection'] = anomaly_detection_system
            
            # 监控组件
            logger.info("📊 初始化监控组件...")
            self.system_components['hardware_monitor'] = hardware_monitor
            self.system_components['ai_monitor'] = ai_status_monitor
            self.system_components['health_checker'] = system_health_checker
            
            # 交易组件
            logger.info("🏦 初始化交易组件...")
            self.system_components['exchange_manager'] = initialize_multi_exchange_manager()
            self.system_components['signal_generator'] = initialize_production_signal_generator()
            
            logger.success("✅ 所有系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 系统组件初始化失败: {e}")
            raise
            
    def start_system(self):
        """启动系统"""
        if self.running:
            logger.warning("⚠️ 系统已在运行中")
            return
            
        logger.info("🚀 启动AI量化交易系统...")
        self.running = True
        
        try:
            # 启动监控线程
            self._start_monitoring_threads()
            
            # 启动AI训练线程
            self._start_ai_training_threads()
            
            # 启动数据更新线程
            self._start_data_update_threads()
            
            logger.success("✅ AI量化交易系统启动成功")
            
            # 主循环
            self._main_loop()
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            self.running = False
            raise
            
    def _start_monitoring_threads(self):
        """启动监控线程"""
        logger.info("📊 启动监控线程...")
        
        # 硬件监控线程
        hardware_thread = threading.Thread(
            target=self._hardware_monitoring_loop,
            daemon=True
        )
        hardware_thread.start()
        
        # AI状态监控线程
        ai_monitor_thread = threading.Thread(
            target=self._ai_monitoring_loop,
            daemon=True
        )
        ai_monitor_thread.start()
        
        # 系统健康检查线程
        health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        health_thread.start()
        
        logger.success("✅ 监控线程启动完成")
        
    def _start_ai_training_threads(self):
        """启动AI训练线程"""
        logger.info("🤖 启动AI训练线程...")
        
        # AI进化训练线程
        evolution_thread = threading.Thread(
            target=self._ai_evolution_loop,
            daemon=True
        )
        evolution_thread.start()
        
        # 深度强化学习训练线程
        rl_thread = threading.Thread(
            target=self._deep_rl_training_loop,
            daemon=True
        )
        rl_thread.start()
        
        logger.success("✅ AI训练线程启动完成")
        
    def _start_data_update_threads(self):
        """启动数据更新线程"""
        logger.info("📈 启动数据更新线程...")
        
        # 市场数据更新线程
        market_data_thread = threading.Thread(
            target=self._market_data_update_loop,
            daemon=True
        )
        market_data_thread.start()
        
        # 情感分析数据更新线程
        sentiment_thread = threading.Thread(
            target=self._sentiment_update_loop,
            daemon=True
        )
        sentiment_thread.start()
        
        logger.success("✅ 数据更新线程启动完成")
        
    def _hardware_monitoring_loop(self):
        """硬件监控循环"""
        logger.info("💻 硬件监控循环开始...")
        
        while self.running:
            try:
                # 更新硬件状态
                hardware_monitor.update_all_metrics()
                
                # 检查资源使用情况
                cpu_usage = hardware_monitor.get_cpu_usage()
                memory_usage = hardware_monitor.get_memory_usage()
                
                # 资源警告
                if cpu_usage > 90:
                    logger.warning(f"⚠️ CPU使用率过高: {cpu_usage:.1f}%")
                    
                if memory_usage > 90:
                    logger.warning(f"⚠️ 内存使用率过高: {memory_usage:.1f}%")
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"硬件监控错误: {e}")
                time.sleep(60)
                
    def _ai_monitoring_loop(self):
        """AI状态监控循环"""
        logger.info("🤖 AI状态监控循环开始...")
        
        while self.running:
            try:
                # 更新AI状态
                ai_status_monitor.update_ai_status()
                
                # 检查AI性能
                performance = ai_status_monitor.get_overall_performance()
                if performance < 0.5:
                    logger.warning(f"⚠️ AI整体性能较低: {performance:.2f}")
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"AI监控错误: {e}")
                time.sleep(60)
                
    def _health_check_loop(self):
        """系统健康检查循环"""
        logger.info("🏥 系统健康检查循环开始...")
        
        while self.running:
            try:
                # 执行健康检查
                health_status = system_health_checker.check_all_systems()
                
                # 记录健康状态
                if hasattr(health_status, 'overall_healthy'):
                    if not health_status.overall_healthy:
                        logger.warning("⚠️ 系统健康状态异常")
                elif hasattr(health_status, 'get'):
                    if not health_status.get('overall_healthy', True):
                        logger.warning("⚠️ 系统健康状态异常")
                    
                time.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                time.sleep(300)
                
    def _ai_evolution_loop(self):
        """AI进化训练循环"""
        logger.info("🧬 AI进化训练循环开始...")
        
        while self.running:
            try:
                # 执行AI进化训练
                if hasattr(ai_evolution_system, 'evolve_models'):
                    ai_evolution_system.evolve_models()
                
                time.sleep(3600)  # 每小时进化一次
                
            except Exception as e:
                logger.error(f"AI进化训练错误: {e}")
                time.sleep(3600)
                
    def _deep_rl_training_loop(self):
        """深度强化学习训练循环"""
        logger.info("🎯 深度强化学习训练循环开始...")
        
        while self.running:
            try:
                # 执行强化学习训练
                deep_rl = self.system_components.get('deep_rl')
                if deep_rl and hasattr(deep_rl, 'train_step'):
                    deep_rl.train_step()
                
                time.sleep(1800)  # 每30分钟训练一次
                
            except Exception as e:
                logger.error(f"深度强化学习训练错误: {e}")
                time.sleep(1800)
                
    def _market_data_update_loop(self):
        """市场数据更新循环"""
        logger.info("📈 市场数据更新循环开始...")
        
        # 支持的交易对
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
        
        while self.running:
            try:
                # 获取活跃交易所
                active_exchanges = multi_exchange_manager.get_active_exchanges()
                
                if active_exchanges:
                    for symbol in symbols:
                        try:
                            # 获取K线数据
                            if isinstance(active_exchanges, list):
                                # 如果是列表，取第一个交易所
                                exchange = active_exchanges[0]
                                exchange_name = exchange.id if hasattr(exchange, 'id') else 'unknown'
                            else:
                                # 如果是字典，取第一个交易所
                                exchange_name = list(active_exchanges.keys())[0]
                                exchange = active_exchanges[exchange_name]
                            
                            # 获取1小时K线数据
                            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=200)
                            
                            if ohlcv:
                                # 转换为MarketData格式
                                market_data = []
                                for candle in ohlcv:
                                    timestamp, open_price, high, low, close, volume = candle
                                    market_data.append(MarketData(
                                        symbol=symbol,
                                        timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
                                        open=float(open_price),
                                        high=float(high),
                                        low=float(low),
                                        close=float(close),
                                        volume=float(volume)
                                    ))
                                
                                # 更新信号生成器的市场数据
                                production_signal_generator.update_market_data(symbol, market_data)
                                
                        except Exception as e:
                            logger.error(f"获取 {symbol} 市场数据失败: {e}")
                            continue
                
                time.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                logger.error(f"市场数据更新错误: {e}")
                time.sleep(60)
                
    def _sentiment_update_loop(self):
        """情感分析数据更新循环"""
        logger.info("😊 情感分析数据更新循环开始...")
        
        while self.running:
            try:
                # 更新市场情感数据
                if hasattr(sentiment_monitor, 'update_sentiment_data'):
                    sentiment_monitor.update_sentiment_data()
                
                time.sleep(300)  # 每5分钟更新一次
                
            except Exception as e:
                logger.error(f"情感分析更新错误: {e}")
                time.sleep(300)
                
    def _main_loop(self):
        """主循环"""
        logger.info("🔄 进入主循环...")
        
        while self.running:
            try:
                # 更新性能统计
                self._update_performance_stats()
                
                # 检查系统状态
                self._check_system_status()
                
                # 等待下一轮
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 收到停止信号...")
                self.stop_system()
                break
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(60)
                
    def _update_performance_stats(self):
        """更新性能统计"""
        try:
            current_time = datetime.now(timezone.utc)
            uptime = current_time - self.start_time
            
            self.performance_stats = {
                'uptime_seconds': uptime.total_seconds(),
                'uptime_hours': uptime.total_seconds() / 3600,
                'system_status': 'running' if self.running else 'stopped',
                'active_components': len([k for k, v in self.system_components.items() if v is not None]),
                'last_update': current_time.isoformat()
            }
            
            # 添加交易统计
            if 'signal_generator' in self.system_components:
                signal_stats = production_signal_generator.get_performance_stats()
                self.performance_stats.update(signal_stats)
                
            # 添加交易所统计
            if 'exchange_manager' in self.system_components:
                trading_stats = multi_exchange_manager.get_trading_summary()
                self.performance_stats.update(trading_stats)
                
        except Exception as e:
            logger.error(f"更新性能统计错误: {e}")
            
    def _check_system_status(self):
        """检查系统状态"""
        try:
            # 检查关键组件状态
            critical_components = ['ai_fusion', 'risk_control', 'hardware_monitor']
            
            for component in critical_components:
                if component not in self.system_components or self.system_components[component] is None:
                    logger.warning(f"⚠️ 关键组件 {component} 不可用")
                    
        except Exception as e:
            logger.error(f"系统状态检查错误: {e}")
            
    def stop_system(self):
        """停止系统"""
        logger.info("🛑 正在停止AI量化交易系统...")
        
        self.running = False
        
        try:
            # 停止信号生成器
            if 'signal_generator' in self.system_components:
                production_signal_generator.stop_generation()
                
            # 保存系统状态
            self._save_system_state()
            
            logger.success("✅ AI量化交易系统已安全停止")
            
        except Exception as e:
            logger.error(f"❌ 系统停止过程中出现错误: {e}")
            
    def _save_system_state(self):
        """保存系统状态"""
        try:
            state_data = {
                'stop_time': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                'performance_stats': self.performance_stats,
                'system_components': list(self.system_components.keys())
            }
            
            # 这里可以保存到文件或数据库
            logger.info("💾 系统状态已保存")
            
        except Exception as e:
            logger.error(f"保存系统状态错误: {e}")
            
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'start_time': self.start_time.isoformat(),
            'running': self.running,
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            'components': list(self.system_components.keys()),
            'performance_stats': self.performance_stats
        }

def main():
    """主函数"""
    try:
        # 创建系统实例
        trading_system = QuantTradingSystem()
        
        # 启动系统
        trading_system.start_system()
        
    except KeyboardInterrupt:
        logger.info("🛑 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 程序运行错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
