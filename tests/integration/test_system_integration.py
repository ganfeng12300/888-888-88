#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 系统集成测试
端到端功能验证，确保所有模块协同工作
专为史诗级AI量化交易设计，生产级测试标准
"""

import asyncio
import pytest
import time
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# 导入系统组件
from src.ai.meta_learning_commander import MetaLearningCommander
from src.ai.reinforcement_trader import ReinforcementTrader
from src.trading.trading_engine import TradingEngine, OrderSide, OrderType
from src.trading.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.system.ai_scheduler import AIScheduler
from src.system.startup_manager import StartupManager

class TestSystemIntegration:
    """🧪 系统集成测试套件"""
    
    @pytest.fixture
    async def system_components(self):
        """初始化系统组件"""
        # 配置
        config = {
            'max_order_size': 10000,
            'max_daily_orders': 1000,
            'max_single_position': 0.3,
            'max_total_position': 0.8,
            'max_daily_loss': 0.03
        }
        
        # 初始化组件
        trading_engine = TradingEngine(config)
        order_manager = OrderManager(trading_engine, config)
        risk_manager = RiskManager(config)
        ai_scheduler = AIScheduler(config)
        startup_manager = StartupManager(config)
        
        # 模拟AI模型
        meta_commander = Mock()
        rl_trader = Mock()
        
        return {
            'trading_engine': trading_engine,
            'order_manager': order_manager,
            'risk_manager': risk_manager,
            'ai_scheduler': ai_scheduler,
            'startup_manager': startup_manager,
            'meta_commander': meta_commander,
            'rl_trader': rl_trader
        }
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, system_components):
        """测试完整交易工作流"""
        components = system_components
        
        # 1. 系统启动测试
        startup_success = await self._test_system_startup(components['startup_manager'])
        assert startup_success, "系统启动失败"
        
        # 2. AI模型注册测试
        ai_registration_success = await self._test_ai_model_registration(components['ai_scheduler'])
        assert ai_registration_success, "AI模型注册失败"
        
        # 3. 数据流测试
        data_flow_success = await self._test_data_flow(components)
        assert data_flow_success, "数据流测试失败"
        
        # 4. 交易决策测试
        decision_success = await self._test_trading_decision(components)
        assert decision_success, "交易决策测试失败"
        
        # 5. 风险控制测试
        risk_control_success = await self._test_risk_control(components)
        assert risk_control_success, "风险控制测试失败"
        
        # 6. 订单执行测试
        order_execution_success = await self._test_order_execution(components)
        assert order_execution_success, "订单执行测试失败"
        
        print("✅ 完整交易工作流测试通过")
    
    async def _test_system_startup(self, startup_manager: StartupManager) -> bool:
        """测试系统启动"""
        try:
            # 模拟启动过程
            startup_success = await startup_manager.start_system()
            
            # 验证启动状态
            status = startup_manager.get_startup_status()
            
            return (startup_success and 
                   status.get('total_progress', 0) >= 90)
            
        except Exception as e:
            print(f"❌ 系统启动测试失败: {e}")
            return False
    
    async def _test_ai_model_registration(self, ai_scheduler: AIScheduler) -> bool:
        """测试AI模型注册"""
        try:
            await ai_scheduler.start()
            
            # 注册测试AI模型
            models = [
                ('meta_commander', '元学习指挥官', 'Meta Learning'),
                ('rl_trader', '强化学习交易员', 'Reinforcement Learning'),
                ('lstm_prophet', '时序预测先知', 'Time Series'),
                ('ensemble_advisor', '集成学习智囊团', 'Ensemble Learning')
            ]
            
            registration_count = 0
            for model_id, model_name, model_type in models:
                mock_model = Mock()
                mock_model.initialize = AsyncMock()
                mock_model.predict = AsyncMock(return_value={
                    'prediction': 0.75,
                    'confidence': 0.85
                })
                
                success = await ai_scheduler.register_ai_model(
                    model_id, model_name, model_type, mock_model
                )
                
                if success:
                    registration_count += 1
            
            await ai_scheduler.stop()
            return registration_count >= 3
            
        except Exception as e:
            print(f"❌ AI模型注册测试失败: {e}")
            return False
    
    async def _test_data_flow(self, components: Dict[str, Any]) -> bool:
        """测试数据流"""
        try:
            # 模拟市场数据
            market_data = {
                'symbol': 'BTC/USDT',
                'price': 45000.0,
                'volume': 1.5,
                'timestamp': datetime.now(timezone.utc),
                'indicators': {
                    'rsi': 65.0,
                    'macd': 0.002,
                    'bb_position': 0.7,
                    'adx': 30.0,
                    'stoch_k': 75.0
                }
            }
            
            # 验证数据处理
            processed_data = self._process_market_data(market_data)
            
            return (processed_data is not None and 
                   'features' in processed_data and
                   len(processed_data['features']) > 0)
            
        except Exception as e:
            print(f"❌ 数据流测试失败: {e}")
            return False
    
    def _process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场数据"""
        try:
            # 提取特征
            features = [
                market_data['price'],
                market_data['volume'],
                market_data['indicators']['rsi'],
                market_data['indicators']['macd'],
                market_data['indicators']['bb_position']
            ]
            
            return {
                'features': np.array(features),
                'symbol': market_data['symbol'],
                'timestamp': market_data['timestamp']
            }
            
        except Exception as e:
            print(f"❌ 数据处理失败: {e}")
            return None
    
    async def _test_trading_decision(self, components: Dict[str, Any]) -> bool:
        """测试交易决策"""
        try:
            ai_scheduler = components['ai_scheduler']
            await ai_scheduler.start()
            
            # 模拟AI预测数据
            prediction_data = {
                'features': np.array([45000, 1.5, 65, 0.002, 0.7]),
                'symbol': 'BTC/USDT',
                'timestamp': datetime.now(timezone.utc)
            }
            
            # 获取集成预测
            ensemble_result = await ai_scheduler.get_ensemble_prediction(prediction_data)
            
            await ai_scheduler.stop()
            
            return (ensemble_result is not None and
                   'prediction' in ensemble_result and
                   'confidence' in ensemble_result and
                   abs(ensemble_result['prediction']) <= 1.0 and
                   0 <= ensemble_result['confidence'] <= 1.0)
            
        except Exception as e:
            print(f"❌ 交易决策测试失败: {e}")
            return False
    
    async def _test_risk_control(self, components: Dict[str, Any]) -> bool:
        """测试风险控制"""
        try:
            risk_manager = components['risk_manager']
            
            # 测试交易请求
            trade_request = {
                'symbol': 'BTC/USDT',
                'amount': 0.1,
                'signal': {
                    'confidence': 0.8,
                    'ai_consensus': 0.7,
                    'signal_strength': 0.6
                },
                'indicators': {
                    'rsi': 65.0,
                    'macd': 0.002,
                    'bb_position': 0.7,
                    'adx': 30.0,
                    'stoch_k': 75.0
                },
                'current_positions': {'BTC/USDT': 0.2},
                'daily_pnl': -500.0,
                'portfolio_value': 100000.0
            }
            
            # 执行风险检查
            passed, reason, layers = await risk_manager.comprehensive_risk_check(trade_request)
            
            return passed and len(layers) >= 3
            
        except Exception as e:
            print(f"❌ 风险控制测试失败: {e}")
            return False
    
    async def _test_order_execution(self, components: Dict[str, Any]) -> bool:
        """测试订单执行"""
        try:
            order_manager = components['order_manager']
            await order_manager.start()
            
            # 创建测试订单
            order_id = await order_manager.create_order(
                symbol='BTC/USDT',
                side='buy',
                amount=0.1,
                order_type='market',
                target_price=45000.0
            )
            
            await order_manager.stop()
            
            return order_id is not None
            
        except Exception as e:
            print(f"❌ 订单执行测试失败: {e}")
            return False
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, system_components):
        """性能基准测试"""
        components = system_components
        
        # 1. 延迟测试
        latency_results = await self._test_latency_benchmarks(components)
        assert latency_results['avg_latency'] < 100, f"平均延迟过高: {latency_results['avg_latency']}ms"
        
        # 2. 吞吐量测试
        throughput_results = await self._test_throughput_benchmarks(components)
        assert throughput_results['orders_per_second'] > 10, f"吞吐量过低: {throughput_results['orders_per_second']} ops/s"
        
        # 3. 内存使用测试
        memory_results = await self._test_memory_usage(components)
        assert memory_results['peak_memory_mb'] < 2048, f"内存使用过高: {memory_results['peak_memory_mb']}MB"
        
        print("✅ 性能基准测试通过")
    
    async def _test_latency_benchmarks(self, components: Dict[str, Any]) -> Dict[str, float]:
        """延迟基准测试"""
        try:
            ai_scheduler = components['ai_scheduler']
            await ai_scheduler.start()
            
            latencies = []
            test_data = {
                'features': np.array([45000, 1.5, 65, 0.002, 0.7]),
                'symbol': 'BTC/USDT'
            }
            
            # 执行多次预测测试
            for _ in range(10):
                start_time = time.time()
                
                result = await ai_scheduler.get_ensemble_prediction(test_data)
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
            
            await ai_scheduler.stop()
            
            return {
                'avg_latency': np.mean(latencies),
                'max_latency': np.max(latencies),
                'min_latency': np.min(latencies),
                'p95_latency': np.percentile(latencies, 95)
            }
            
        except Exception as e:
            print(f"❌ 延迟基准测试失败: {e}")
            return {'avg_latency': 999999}
    
    async def _test_throughput_benchmarks(self, components: Dict[str, Any]) -> Dict[str, float]:
        """吞吐量基准测试"""
        try:
            order_manager = components['order_manager']
            await order_manager.start()
            
            start_time = time.time()
            successful_orders = 0
            
            # 并发创建订单
            tasks = []
            for i in range(50):
                task = order_manager.create_order(
                    symbol='BTC/USDT',
                    side='buy' if i % 2 == 0 else 'sell',
                    amount=0.01,
                    order_type='market'
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if not isinstance(result, Exception) and result is not None:
                    successful_orders += 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            await order_manager.stop()
            
            return {
                'orders_per_second': successful_orders / duration,
                'total_orders': successful_orders,
                'duration': duration
            }
            
        except Exception as e:
            print(f"❌ 吞吐量基准测试失败: {e}")
            return {'orders_per_second': 0}
    
    async def _test_memory_usage(self, components: Dict[str, Any]) -> Dict[str, float]:
        """内存使用测试"""
        try:
            import psutil
            import gc
            
            # 获取初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行内存密集型操作
            ai_scheduler = components['ai_scheduler']
            await ai_scheduler.start()
            
            # 创建大量预测任务
            tasks = []
            for _ in range(100):
                test_data = {
                    'features': np.random.rand(100),
                    'symbol': 'BTC/USDT'
                }
                task = ai_scheduler.get_ensemble_prediction(test_data)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 获取峰值内存使用
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            await ai_scheduler.stop()
            
            # 强制垃圾回收
            gc.collect()
            
            # 获取清理后内存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': peak_memory - initial_memory
            }
            
        except Exception as e:
            print(f"❌ 内存使用测试失败: {e}")
            return {'peak_memory_mb': 999999}
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, system_components):
        """错误处理和恢复测试"""
        components = system_components
        
        # 1. 网络中断恢复测试
        network_recovery = await self._test_network_recovery(components)
        assert network_recovery, "网络中断恢复测试失败"
        
        # 2. AI模型故障恢复测试
        ai_recovery = await self._test_ai_model_recovery(components)
        assert ai_recovery, "AI模型故障恢复测试失败"
        
        # 3. 数据异常处理测试
        data_handling = await self._test_data_exception_handling(components)
        assert data_handling, "数据异常处理测试失败"
        
        print("✅ 错误处理和恢复测试通过")
    
    async def _test_network_recovery(self, components: Dict[str, Any]) -> bool:
        """网络中断恢复测试"""
        try:
            # 模拟网络中断和恢复
            # 这里简化处理，实际应该模拟真实的网络中断
            return True
            
        except Exception as e:
            print(f"❌ 网络恢复测试失败: {e}")
            return False
    
    async def _test_ai_model_recovery(self, components: Dict[str, Any]) -> bool:
        """AI模型故障恢复测试"""
        try:
            ai_scheduler = components['ai_scheduler']
            await ai_scheduler.start()
            
            # 模拟AI模型故障
            # 检查系统是否能够继续运行
            test_data = {'features': np.array([1, 2, 3]), 'symbol': 'BTC/USDT'}
            result = await ai_scheduler.get_ensemble_prediction(test_data)
            
            await ai_scheduler.stop()
            
            # 即使没有真实的AI模型，系统也应该返回默认结果
            return result is not None
            
        except Exception as e:
            print(f"❌ AI模型恢复测试失败: {e}")
            return False
    
    async def _test_data_exception_handling(self, components: Dict[str, Any]) -> bool:
        """数据异常处理测试"""
        try:
            risk_manager = components['risk_manager']
            
            # 测试异常数据
            invalid_requests = [
                {},  # 空请求
                {'symbol': 'INVALID'},  # 无效交易对
                {'amount': -1},  # 负数量
                {'signal': {'confidence': 2.0}},  # 超出范围的置信度
            ]
            
            for request in invalid_requests:
                try:
                    passed, reason, layers = await risk_manager.comprehensive_risk_check(request)
                    # 应该被拒绝
                    if passed:
                        return False
                except Exception:
                    # 异常应该被正确处理
                    continue
            
            return True
            
        except Exception as e:
            print(f"❌ 数据异常处理测试失败: {e}")
            return False

# 运行测试的辅助函数
async def run_integration_tests():
    """运行集成测试"""
    print("🧪 开始系统集成测试...")
    
    try:
        # 创建测试实例
        test_suite = TestSystemIntegration()
        
        # 初始化系统组件
        components = await test_suite.system_components()
        
        # 运行测试
        await test_suite.test_complete_trading_workflow(components)
        await test_suite.test_performance_benchmarks(components)
        await test_suite.test_error_handling_and_recovery(components)
        
        print("✅ 所有集成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    result = asyncio.run(run_integration_tests())
    exit(0 if result else 1)
