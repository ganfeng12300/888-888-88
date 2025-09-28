"""
🔗 端到端集成测试系统
生产级完整链路测试，从AI决策到订单执行的全流程验证
实现真实市场数据测试、异常处理和系统稳定性验证
"""

import asyncio
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.ai_models.ai_evolution_system import AIEvolutionSystem, ModelType
from src.risk_management.risk_controller import AILevelRiskController
from src.trading_execution.smart_order_router import SmartOrderRouter, OrderRequest, OrderSide, OrderType
from src.trading_execution.low_latency_engine import LowLatencyExecutionEngine, ExecutionPriority
from src.data_collection.exchange_connector import ExchangeConnector


class TestType(Enum):
    """测试类型"""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    STRESS_TEST = "stress_test"
    CHAOS_TEST = "chaos_test"
    PERFORMANCE_TEST = "performance_test"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """测试用例"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    test_function: Callable
    timeout_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    expected_result: Any = None
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    status: TestStatus
    start_time: float
    end_time: float
    duration: float
    result_data: Any = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    assertions_passed: int = 0
    assertions_failed: int = 0


@dataclass
class SystemHealthMetrics:
    """系统健康指标"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    queue_sizes: Dict[str, int]
    error_rate: float
    throughput: float


class EndToEndTester:
    """端到端集成测试器"""
    
    def __init__(self):
        # 系统组件
        self.ai_evolution_system: Optional[AIEvolutionSystem] = None
        self.risk_controller: Optional[AILevelRiskController] = None
        self.order_router: Optional[SmartOrderRouter] = None
        self.execution_engine: Optional[LowLatencyExecutionEngine] = None
        self.exchange_connectors: Dict[str, ExchangeConnector] = {}
        
        # 测试管理
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.test_queue: deque = deque()
        self.running_tests: Dict[str, asyncio.Task] = {}
        
        # 性能监控
        self.health_metrics: deque = deque(maxlen=1000)
        self.performance_baseline: Dict[str, float] = {}
        
        # 测试配置
        self.max_concurrent_tests = 5
        self.test_timeout = 300.0  # 5分钟
        self.health_check_interval = 10.0  # 10秒
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.INTEGRATION_TESTING, [15, 16])
        
        # 启动健康监控
        self.monitoring_active = False
        self._start_health_monitoring()
        
        logger.info("端到端集成测试器初始化完成")
    
    def initialize_system_components(self, components: Dict[str, Any]):
        """初始化系统组件"""
        try:
            self.ai_evolution_system = components.get('ai_evolution_system')
            self.risk_controller = components.get('risk_controller')
            self.order_router = components.get('order_router')
            self.execution_engine = components.get('execution_engine')
            self.exchange_connectors = components.get('exchange_connectors', {})
            
            logger.info("系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化系统组件失败: {e}")
            raise
    
    def register_test_case(self, test_case: TestCase):
        """注册测试用例"""
        self.test_cases[test_case.test_id] = test_case
        logger.info(f"注册测试用例: {test_case.test_id} - {test_case.name}")
    
    def _start_health_monitoring(self):
        """启动健康监控"""
        self.monitoring_active = True
        
        def monitor_health():
            while self.monitoring_active:
                try:
                    asyncio.run(self._collect_health_metrics())
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"健康监控出错: {e}")
                    time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_health, daemon=True)
        monitor_thread.start()
        
        logger.info("系统健康监控已启动")
    
    async def _collect_health_metrics(self):
        """收集健康指标"""
        try:
            import psutil
            
            # 系统资源使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU使用率 (简化)
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass
            
            # 网络延迟 (简化)
            network_latency = 0.0
            
            # 队列大小
            queue_sizes = {}
            if self.execution_engine:
                queue_sizes = {
                    'execution_queue': sum(
                        q.qsize() for q in self.execution_engine.execution_queues.values()
                    )
                }
            
            # 错误率和吞吐量
            error_rate = self._calculate_error_rate()
            throughput = self._calculate_throughput()
            
            # 创建健康指标
            metrics = SystemHealthMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                gpu_usage=gpu_usage,
                disk_usage=disk.percent,
                network_latency=network_latency,
                active_connections=len(self.running_tests),
                queue_sizes=queue_sizes,
                error_rate=error_rate,
                throughput=throughput
            )
            
            self.health_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"收集健康指标失败: {e}")
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        if not self.test_results:
            return 0.0
        
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result.status in [TestStatus.FAILED, TestStatus.ERROR])
        total_tests = len(self.test_results)
        
        return failed_tests / total_tests if total_tests > 0 else 0.0
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量"""
        if len(self.test_results) < 2:
            return 0.0
        
        recent_results = list(self.test_results.values())[-10:]  # 最近10个测试
        if not recent_results:
            return 0.0
        
        time_span = recent_results[-1].end_time - recent_results[0].start_time
        return len(recent_results) / time_span if time_span > 0 else 0.0
    
    async def run_single_test(self, test_case: TestCase) -> TestResult:
        """运行单个测试用例"""
        start_time = time.time()
        
        try:
            logger.info(f"开始测试: {test_case.test_id} - {test_case.name}")
            
            # 执行setup
            if test_case.setup_function:
                await test_case.setup_function()
            
            # 执行测试
            result_data = await asyncio.wait_for(
                test_case.test_function(),
                timeout=test_case.timeout_seconds
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 验证结果
            status = TestStatus.PASSED
            if test_case.expected_result is not None:
                if result_data != test_case.expected_result:
                    status = TestStatus.FAILED
            
            # 创建测试结果
            test_result = TestResult(
                test_id=test_case.test_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                result_data=result_data,
                performance_metrics=self._extract_performance_metrics(result_data)
            )
            
            logger.info(f"测试完成: {test_case.test_id}, 状态={status.value}, 耗时={duration:.2f}s")
            
            return test_result
            
        except asyncio.TimeoutError:
            end_time = time.time()
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                error_message="测试超时"
            )
            
        except Exception as e:
            end_time = time.time()
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )
            
        finally:
            # 执行teardown
            if test_case.teardown_function:
                try:
                    await test_case.teardown_function()
                except Exception as e:
                    logger.error(f"Teardown失败: {e}")
    
    def _extract_performance_metrics(self, result_data: Any) -> Dict[str, float]:
        """提取性能指标"""
        metrics = {}
        
        if isinstance(result_data, dict):
            # 提取延迟指标
            if 'latency_ms' in result_data:
                metrics['latency_ms'] = float(result_data['latency_ms'])
            
            # 提取吞吐量指标
            if 'throughput' in result_data:
                metrics['throughput'] = float(result_data['throughput'])
            
            # 提取成功率指标
            if 'success_rate' in result_data:
                metrics['success_rate'] = float(result_data['success_rate'])
        
        return metrics
    
    async def run_test_suite(self, test_ids: List[str] = None) -> Dict[str, TestResult]:
        """运行测试套件"""
        if test_ids is None:
            test_ids = list(self.test_cases.keys())
        
        logger.info(f"开始运行测试套件，共{len(test_ids)}个测试")
        
        # 按依赖关系排序测试
        sorted_test_ids = self._sort_tests_by_dependencies(test_ids)
        
        # 并发执行测试
        semaphore = asyncio.Semaphore(self.max_concurrent_tests)
        
        async def run_with_semaphore(test_id: str):
            async with semaphore:
                test_case = self.test_cases[test_id]
                result = await self.run_single_test(test_case)
                self.test_results[test_id] = result
                return result
        
        # 创建任务
        tasks = [run_with_semaphore(test_id) for test_id in sorted_test_ids]
        
        # 等待所有测试完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        passed = sum(1 for r in results if isinstance(r, TestResult) and r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if isinstance(r, TestResult) and r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if isinstance(r, TestResult) and r.status == TestStatus.ERROR)
        
        logger.info(f"测试套件完成: 通过={passed}, 失败={failed}, 错误={errors}")
        
        return self.test_results
    
    def _sort_tests_by_dependencies(self, test_ids: List[str]) -> List[str]:
        """按依赖关系排序测试"""
        # 简化的拓扑排序
        sorted_ids = []
        remaining_ids = set(test_ids)
        
        while remaining_ids:
            # 找到没有未满足依赖的测试
            ready_tests = []
            for test_id in remaining_ids:
                test_case = self.test_cases[test_id]
                dependencies_satisfied = all(
                    dep in sorted_ids or dep not in test_ids 
                    for dep in test_case.dependencies
                )
                if dependencies_satisfied:
                    ready_tests.append(test_id)
            
            if not ready_tests:
                # 如果没有就绪的测试，可能存在循环依赖
                logger.warning("检测到可能的循环依赖，按原顺序执行剩余测试")
                ready_tests = list(remaining_ids)
            
            # 添加就绪的测试
            for test_id in ready_tests:
                sorted_ids.append(test_id)
                remaining_ids.remove(test_id)
        
        return sorted_ids
    
    # 具体的测试用例实现
    async def test_ai_decision_making(self) -> Dict[str, Any]:
        """测试AI决策制定"""
        if not self.ai_evolution_system:
            raise Exception("AI进化系统未初始化")
        
        start_time = time.time()
        
        # 模拟市场数据
        market_data = {
            'price': 50000.0,
            'volume': 1000000.0,
            'volatility': 0.02
        }
        
        # 获取AI决策
        decision = self.ai_evolution_system.get_fusion_decision(market_data)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        # 验证决策结果
        assert decision is not None, "AI决策不能为空"
        assert 'final_action' in decision, "决策必须包含最终动作"
        assert decision['final_action'] in ['buy', 'sell', 'hold'], "无效的决策动作"
        assert 'confidence' in decision, "决策必须包含置信度"
        assert 0 <= decision['confidence'] <= 1, "置信度必须在0-1之间"
        
        return {
            'decision': decision,
            'latency_ms': latency,
            'success': True
        }
    
    async def test_risk_control_integration(self) -> Dict[str, Any]:
        """测试风险控制集成"""
        if not self.risk_controller:
            raise Exception("风险控制器未初始化")
        
        start_time = time.time()
        
        # 模拟持仓更新
        symbol = "BTC/USDT"
        position_size = 1.0
        current_price = 50000.0
        entry_price = 49500.0
        
        self.risk_controller.update_position(symbol, position_size, current_price, entry_price)
        
        # 计算投资组合风险
        risk_metrics = self.risk_controller.calculate_portfolio_risk()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        # 验证风险指标
        assert risk_metrics is not None, "风险指标不能为空"
        assert risk_metrics.portfolio_value > 0, "投资组合价值必须大于0"
        assert risk_metrics.leverage_ratio >= 0, "杠杆比率不能为负"
        
        return {
            'risk_metrics': {
                'portfolio_value': risk_metrics.portfolio_value,
                'leverage_ratio': risk_metrics.leverage_ratio,
                'max_drawdown': risk_metrics.max_drawdown
            },
            'latency_ms': latency,
            'success': True
        }
    
    async def test_order_routing_execution(self) -> Dict[str, Any]:
        """测试订单路由执行"""
        if not self.order_router or not self.execution_engine:
            raise Exception("订单路由器或执行引擎未初始化")
        
        start_time = time.time()
        
        # 创建测试订单
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            max_slippage=0.002,
            urgency=0.7,
            client_order_id="test_order_e2e"
        )
        
        # 路由订单
        route = await self.order_router.route_order(order)
        
        routing_time = time.time()
        routing_latency = (routing_time - start_time) * 1000
        
        # 验证路由结果
        assert route is not None, "路由结果不能为空"
        assert len(route.segments) > 0, "路由必须包含执行片段"
        assert route.confidence_score > 0, "路由置信度必须大于0"
        
        # 执行路由
        task_ids = await self.execution_engine.execute_route(route, ExecutionPriority.HIGH)
        
        execution_time = time.time()
        execution_latency = (execution_time - routing_time) * 1000
        total_latency = (execution_time - start_time) * 1000
        
        # 验证执行结果
        assert len(task_ids) > 0, "执行任务ID不能为空"
        
        return {
            'route': {
                'segments': len(route.segments),
                'confidence_score': route.confidence_score,
                'expected_slippage': route.expected_slippage
            },
            'execution': {
                'task_count': len(task_ids)
            },
            'latency_ms': total_latency,
            'routing_latency_ms': routing_latency,
            'execution_latency_ms': execution_latency,
            'success': True
        }
    
    async def test_end_to_end_trading_flow(self) -> Dict[str, Any]:
        """测试端到端交易流程"""
        start_time = time.time()
        
        # 1. AI决策
        ai_decision = await self.test_ai_decision_making()
        decision_time = time.time()
        
        # 2. 风险控制
        risk_result = await self.test_risk_control_integration()
        risk_time = time.time()
        
        # 3. 订单执行 (如果决策是买入或卖出)
        execution_result = None
        if ai_decision['decision']['final_action'] in ['buy', 'sell']:
            execution_result = await self.test_order_routing_execution()
        
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000
        
        return {
            'ai_decision': ai_decision,
            'risk_control': risk_result,
            'execution': execution_result,
            'total_latency_ms': total_latency,
            'decision_latency_ms': (decision_time - start_time) * 1000,
            'risk_latency_ms': (risk_time - decision_time) * 1000,
            'success': True
        }
    
    async def test_system_stress(self) -> Dict[str, Any]:
        """系统压力测试"""
        start_time = time.time()
        
        # 并发执行多个交易流程
        concurrent_flows = 10
        tasks = [self.test_end_to_end_trading_flow() for _ in range(concurrent_flows)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 统计结果
        successful_flows = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed_flows = concurrent_flows - successful_flows
        
        # 计算性能指标
        if successful_flows > 0:
            avg_latency = np.mean([
                r['total_latency_ms'] for r in results 
                if isinstance(r, dict) and 'total_latency_ms' in r
            ])
        else:
            avg_latency = 0
        
        throughput = successful_flows / duration
        
        return {
            'concurrent_flows': concurrent_flows,
            'successful_flows': successful_flows,
            'failed_flows': failed_flows,
            'success_rate': successful_flows / concurrent_flows,
            'avg_latency_ms': avg_latency,
            'throughput': throughput,
            'duration_seconds': duration,
            'success': failed_flows == 0
        }
    
    def setup_standard_test_suite(self):
        """设置标准测试套件"""
        # AI决策测试
        self.register_test_case(TestCase(
            test_id="ai_decision_test",
            name="AI决策制定测试",
            description="测试AI系统的决策制定能力",
            test_type=TestType.INTEGRATION_TEST,
            test_function=self.test_ai_decision_making,
            timeout_seconds=30.0,
            tags=["ai", "decision"]
        ))
        
        # 风险控制测试
        self.register_test_case(TestCase(
            test_id="risk_control_test",
            name="风险控制集成测试",
            description="测试风险控制系统的集成功能",
            test_type=TestType.INTEGRATION_TEST,
            test_function=self.test_risk_control_integration,
            timeout_seconds=30.0,
            tags=["risk", "control"]
        ))
        
        # 订单路由执行测试
        self.register_test_case(TestCase(
            test_id="order_routing_test",
            name="订单路由执行测试",
            description="测试订单路由和执行的完整流程",
            test_type=TestType.INTEGRATION_TEST,
            test_function=self.test_order_routing_execution,
            timeout_seconds=60.0,
            dependencies=["ai_decision_test"],
            tags=["routing", "execution"]
        ))
        
        # 端到端流程测试
        self.register_test_case(TestCase(
            test_id="e2e_trading_flow_test",
            name="端到端交易流程测试",
            description="测试完整的交易决策到执行流程",
            test_type=TestType.INTEGRATION_TEST,
            test_function=self.test_end_to_end_trading_flow,
            timeout_seconds=120.0,
            dependencies=["ai_decision_test", "risk_control_test", "order_routing_test"],
            tags=["e2e", "trading"]
        ))
        
        # 压力测试
        self.register_test_case(TestCase(
            test_id="system_stress_test",
            name="系统压力测试",
            description="测试系统在高并发下的性能表现",
            test_type=TestType.STRESS_TEST,
            test_function=self.test_system_stress,
            timeout_seconds=300.0,
            dependencies=["e2e_trading_flow_test"],
            tags=["stress", "performance"]
        ))
        
        logger.info("标准测试套件设置完成")
    
    def get_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        if not self.test_results:
            return {"message": "没有测试结果"}
        
        results = list(self.test_results.values())
        
        # 统计信息
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        # 性能统计
        durations = [r.duration for r in results if r.duration > 0]
        avg_duration = np.mean(durations) if durations else 0
        max_duration = np.max(durations) if durations else 0
        
        # 健康指标
        latest_health = self.health_metrics[-1] if self.health_metrics else None
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'avg_duration': avg_duration,
                'max_duration': max_duration
            },
            'test_results': {
                test_id: {
                    'status': result.status.value,
                    'duration': result.duration,
                    'error_message': result.error_message,
                    'performance_metrics': result.performance_metrics
                }
                for test_id, result in self.test_results.items()
            },
            'system_health': {
                'cpu_usage': latest_health.cpu_usage if latest_health else 0,
                'memory_usage': latest_health.memory_usage if latest_health else 0,
                'error_rate': latest_health.error_rate if latest_health else 0,
                'throughput': latest_health.throughput if latest_health else 0
            } if latest_health else {},
            'timestamp': time.time()
        }
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info("端到端测试监控已停止")


# 全局测试器实例
e2e_tester = EndToEndTester()


async def main():
    """测试主函数"""
    logger.info("启动端到端集成测试...")
    
    try:
        # 设置标准测试套件
        e2e_tester.setup_standard_test_suite()
        
        # 模拟系统组件初始化
        components = {
            'ai_evolution_system': None,  # 实际应该是真实的组件
            'risk_controller': None,
            'order_router': None,
            'execution_engine': None,
            'exchange_connectors': {}
        }
        
        e2e_tester.initialize_system_components(components)
        
        # 运行测试套件
        results = await e2e_tester.run_test_suite()
        
        # 生成测试报告
        report = e2e_tester.get_test_report()
        
        logger.info("测试报告:")
        logger.info(json.dumps(report, indent=2))
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        e2e_tester.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
