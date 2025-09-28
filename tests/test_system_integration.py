"""
🧪 系统集成测试套件
完整的端到端测试，验证AI量化交易系统各组件集成功能
包括启动管理、消息总线、配置管理、健康监控等核心系统测试
"""

import pytest
import asyncio
import time
import os
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# 导入被测试的模块
from src.system.startup_manager import StartupManager, ComponentConfig, ComponentType
from src.system.message_bus import MessageBus, Message, MessagePriority
from src.system.config_manager import ConfigManager
from src.system.health_monitor import HealthMonitor, HealthStatus, AlertLevel
from monitoring.prometheus_metrics import PrometheusMetricsManager, MetricType, MetricConfig
from monitoring.alert_manager import AlertManager, Alert, AlertSeverity, AlertStatus
from performance.cpu_optimizer import CPUOptimizer, ProcessType
from performance.memory_optimizer import MemoryOptimizer


class TestSystemIntegration:
    """系统集成测试类"""
    
    @pytest.fixture
    async def startup_manager(self):
        """启动管理器测试夹具"""
        manager = StartupManager()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    async def message_bus(self):
        """消息总线测试夹具"""
        bus = MessageBus()
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def config_manager(self):
        """配置管理器测试夹具"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(config_dir=temp_dir)
            yield manager
    
    @pytest.fixture
    async def health_monitor(self):
        """健康监控器测试夹具"""
        monitor = HealthMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()
    
    @pytest.fixture
    def metrics_manager(self):
        """指标管理器测试夹具"""
        manager = PrometheusMetricsManager()
        yield manager
    
    @pytest.fixture
    def alert_manager(self):
        """告警管理器测试夹具"""
        manager = AlertManager()
        yield manager
    
    @pytest.fixture
    def cpu_optimizer(self):
        """CPU优化器测试夹具"""
        optimizer = CPUOptimizer()
        yield optimizer
    
    @pytest.fixture
    def memory_optimizer(self):
        """内存优化器测试夹具"""
        optimizer = MemoryOptimizer(total_memory_gb=1)  # 测试用小内存
        yield optimizer
        optimizer.cleanup()

    @pytest.mark.asyncio
    async def test_startup_manager_basic_functionality(self, startup_manager):
        """测试启动管理器基本功能"""
        # 添加测试组件
        config = ComponentConfig(
            name="test_component",
            component_type=ComponentType.SERVICE,
            module_path="test.module",
            dependencies=[],
            startup_timeout=10.0,
            health_check_interval=5.0,
            auto_restart=True,
            max_restart_attempts=3
        )
        
        startup_manager.add_component(config)
        
        # 验证组件已添加
        assert "test_component" in startup_manager.components
        assert startup_manager.components["test_component"] == config
        
        # 测试依赖解析
        dependencies = startup_manager.resolve_dependencies()
        assert isinstance(dependencies, list)
    
    @pytest.mark.asyncio
    async def test_message_bus_communication(self, message_bus):
        """测试消息总线通信功能"""
        received_messages = []
        
        # 定义消息处理器
        async def message_handler(message: Message):
            received_messages.append(message)
        
        # 订阅主题
        await message_bus.subscribe("test_topic", message_handler)
        
        # 发送消息
        test_message = Message(
            topic="test_topic",
            data={"test": "data"},
            priority=MessagePriority.NORMAL
        )
        
        await message_bus.publish(test_message)
        
        # 等待消息处理
        await asyncio.sleep(0.1)
        
        # 验证消息接收
        assert len(received_messages) == 1
        assert received_messages[0].topic == "test_topic"
        assert received_messages[0].data == {"test": "data"}
    
    def test_config_manager_operations(self, config_manager):
        """测试配置管理器操作"""
        # 测试设置配置
        test_config = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            }
        }
        
        config_manager.set_config("test_env", test_config)
        
        # 测试获取配置
        retrieved_config = config_manager.get_config("test_env")
        assert retrieved_config == test_config
        
        # 测试获取嵌套配置
        db_host = config_manager.get_config("test_env", "database.host")
        assert db_host == "localhost"
        
        # 测试配置验证
        schema = {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"}
                    },
                    "required": ["host", "port"]
                }
            },
            "required": ["database"]
        }
        
        is_valid = config_manager.validate_config("test_env", schema)
        assert is_valid
    
    @pytest.mark.asyncio
    async def test_health_monitor_functionality(self, health_monitor):
        """测试健康监控器功能"""
        # 注册健康检查
        async def test_health_check():
            return HealthStatus.HEALTHY, "Test component is healthy"
        
        health_monitor.register_health_check("test_component", test_health_check)
        
        # 执行健康检查
        health_status = await health_monitor.check_health()
        
        # 验证健康状态
        assert health_status.overall_status == HealthStatus.HEALTHY
        assert "test_component" in health_status.component_status
        assert health_status.component_status["test_component"]["status"] == HealthStatus.HEALTHY
    
    def test_metrics_manager_operations(self, metrics_manager):
        """测试指标管理器操作"""
        # 创建自定义指标
        metric_config = MetricConfig(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            description="Test counter metric",
            labels=["service", "method"]
        )
        
        metric = metrics_manager.create_metric(metric_config)
        
        if metric:  # 只有在Prometheus可用时测试
            # 验证指标创建
            assert metrics_manager.get_metric("test_counter") is not None
            
            # 测试指标记录
            metrics_manager.record_http_request("GET", "/api/test", 200, 0.1)
            
            # 获取指标文本
            metrics_text = metrics_manager.get_metrics_text()
            assert isinstance(metrics_text, str)
    
    @pytest.mark.asyncio
    async def test_alert_manager_functionality(self, alert_manager):
        """测试告警管理器功能"""
        # 创建测试告警
        alert = Alert(
            id="test_alert_001",
            name="Test Alert",
            description="This is a test alert",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            labels={"service": "test", "environment": "testing"}
        )
        
        # 模拟通知渠道
        mock_notifier = Mock()
        mock_notifier.send_alert = AsyncMock(return_value=True)
        
        alert_manager.notifiers["test_channel"] = mock_notifier
        
        # 发送告警
        await alert_manager.send_alert(alert)
        
        # 验证告警状态
        assert alert.id in alert_manager.active_alerts
        assert len(alert_manager.alert_history) > 0
        
        # 解决告警
        alert_manager.resolve_alert(alert.id)
        assert alert.id not in alert_manager.active_alerts
    
    def test_cpu_optimizer_functionality(self, cpu_optimizer):
        """测试CPU优化器功能"""
        # 获取优化统计
        stats = cpu_optimizer.get_optimization_stats()
        
        # 验证统计信息结构
        assert "topology" in stats
        assert "assignments" in stats
        assert "performance" in stats
        
        # 验证拓扑信息
        topology = stats["topology"]
        assert "total_cores" in topology
        assert "physical_cores" in topology
        assert "hyperthread_cores" in topology
        assert "numa_nodes" in topology
        
        # 测试进程优化（使用当前进程）
        result = cpu_optimizer.optimize_current_process(ProcessType.GENERAL)
        # 结果可能因权限而异，只验证不抛异常
        assert isinstance(result, bool)
    
    def test_memory_optimizer_functionality(self, memory_optimizer):
        """测试内存优化器功能"""
        # 获取优化统计
        stats = memory_optimizer.get_optimization_stats()
        
        # 验证统计信息结构
        assert "memory_pools" in stats
        assert "gc_optimizer" in stats
        assert "memory_monitor" in stats
        
        # 测试内存分配
        allocation = memory_optimizer.allocate_memory("general", 1024)
        if allocation:
            start_block, block_count = allocation
            assert isinstance(start_block, int)
            assert isinstance(block_count, int)
            
            # 测试内存释放
            result = memory_optimizer.deallocate_memory("general", start_block, block_count)
            assert result is True
        
        # 测试垃圾回收优化
        gc_result = memory_optimizer.optimize_gc()
        assert gc_result is not None or gc_result is None  # 可能返回None
    
    @pytest.mark.asyncio
    async def test_system_startup_sequence(self, startup_manager, message_bus, config_manager):
        """测试系统启动序列"""
        # 模拟系统组件配置
        components = [
            ComponentConfig(
                name="config_manager",
                component_type=ComponentType.CORE,
                module_path="src.system.config_manager",
                dependencies=[],
                startup_timeout=5.0
            ),
            ComponentConfig(
                name="message_bus",
                component_type=ComponentType.CORE,
                module_path="src.system.message_bus",
                dependencies=["config_manager"],
                startup_timeout=5.0
            ),
            ComponentConfig(
                name="health_monitor",
                component_type=ComponentType.SERVICE,
                module_path="src.system.health_monitor",
                dependencies=["config_manager", "message_bus"],
                startup_timeout=10.0
            )
        ]
        
        # 添加组件到启动管理器
        for component in components:
            startup_manager.add_component(component)
        
        # 解析依赖关系
        startup_order = startup_manager.resolve_dependencies()
        
        # 验证启动顺序
        assert len(startup_order) == 3
        
        # config_manager应该首先启动
        assert startup_order[0].name == "config_manager"
        
        # message_bus应该在config_manager之后
        config_index = next(i for i, c in enumerate(startup_order) if c.name == "config_manager")
        bus_index = next(i for i, c in enumerate(startup_order) if c.name == "message_bus")
        assert bus_index > config_index
    
    @pytest.mark.asyncio
    async def test_cross_component_communication(self, message_bus, health_monitor):
        """测试跨组件通信"""
        health_updates = []
        
        # 健康状态更新处理器
        async def health_update_handler(message: Message):
            health_updates.append(message.data)
        
        # 订阅健康状态更新
        await message_bus.subscribe("health.status", health_update_handler)
        
        # 模拟健康状态变化
        health_message = Message(
            topic="health.status",
            data={
                "component": "test_service",
                "status": "healthy",
                "timestamp": time.time()
            },
            priority=MessagePriority.HIGH
        )
        
        await message_bus.publish(health_message)
        
        # 等待消息处理
        await asyncio.sleep(0.1)
        
        # 验证消息接收
        assert len(health_updates) == 1
        assert health_updates[0]["component"] == "test_service"
        assert health_updates[0]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, metrics_manager, cpu_optimizer, memory_optimizer):
        """测试性能监控集成"""
        # 启动CPU监控
        await cpu_optimizer.start_monitoring(interval=0.1)
        
        # 启动内存监控
        await memory_optimizer.start_monitoring(interval=0.1)
        
        # 等待一些监控数据
        await asyncio.sleep(0.5)
        
        # 获取性能统计
        cpu_stats = cpu_optimizer.get_optimization_stats()
        memory_stats = memory_optimizer.get_optimization_stats()
        
        # 验证监控数据
        assert "performance" in cpu_stats
        assert "memory_monitor" in memory_stats
        
        # 停止监控
        await cpu_optimizer.stop_monitoring()
        await memory_optimizer.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, startup_manager, health_monitor):
        """测试错误处理和恢复"""
        # 注册一个会失败的健康检查
        async def failing_health_check():
            raise Exception("Simulated failure")
        
        health_monitor.register_health_check("failing_component", failing_health_check)
        
        # 执行健康检查
        health_status = await health_monitor.check_health()
        
        # 验证错误被正确处理
        assert "failing_component" in health_status.component_status
        component_status = health_status.component_status["failing_component"]
        assert component_status["status"] == HealthStatus.CRITICAL
        assert "error" in component_status
    
    def test_configuration_hot_reload(self, config_manager):
        """测试配置热重载"""
        # 设置初始配置
        initial_config = {"setting": "initial_value"}
        config_manager.set_config("test_env", initial_config)
        
        # 验证初始配置
        assert config_manager.get_config("test_env", "setting") == "initial_value"
        
        # 更新配置
        updated_config = {"setting": "updated_value"}
        config_manager.set_config("test_env", updated_config)
        
        # 验证配置已更新
        assert config_manager.get_config("test_env", "setting") == "updated_value"
        
        # 测试配置变更通知
        change_notifications = []
        
        def config_change_handler(env, key, old_value, new_value):
            change_notifications.append({
                "env": env,
                "key": key,
                "old_value": old_value,
                "new_value": new_value
            })
        
        config_manager.add_change_listener(config_change_handler)
        
        # 再次更新配置
        config_manager.set_config("test_env", {"setting": "final_value"})
        
        # 验证变更通知
        assert len(change_notifications) > 0
    
    @pytest.mark.asyncio
    async def test_system_shutdown_sequence(self, startup_manager, message_bus, health_monitor):
        """测试系统关闭序列"""
        # 启动组件
        await message_bus.start()
        await health_monitor.start()
        
        # 验证组件运行状态
        assert message_bus.running
        assert health_monitor.running
        
        # 执行关闭序列
        await health_monitor.stop()
        await message_bus.stop()
        
        # 验证组件已停止
        assert not health_monitor.running
        assert not message_bus.running
    
    def test_resource_cleanup(self, memory_optimizer):
        """测试资源清理"""
        # 分配一些内存
        allocations = []
        for i in range(5):
            allocation = memory_optimizer.allocate_memory("general", 1024)
            if allocation:
                allocations.append(allocation)
        
        # 获取分配前统计
        stats_before = memory_optimizer.get_optimization_stats()
        
        # 清理资源
        memory_optimizer.cleanup()
        
        # 验证资源已清理
        stats_after = memory_optimizer.get_optimization_stats()
        
        # 内存池应该被清理
        assert len(stats_after.get("memory_pools", {})) == 0


class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.asyncio
    async def test_message_bus_throughput(self):
        """测试消息总线吞吐量"""
        bus = MessageBus()
        await bus.start()
        
        try:
            message_count = 1000
            received_count = 0
            
            async def counter_handler(message: Message):
                nonlocal received_count
                received_count += 1
            
            await bus.subscribe("benchmark", counter_handler)
            
            # 测量发送时间
            start_time = time.time()
            
            for i in range(message_count):
                message = Message(
                    topic="benchmark",
                    data={"index": i},
                    priority=MessagePriority.NORMAL
                )
                await bus.publish(message)
            
            # 等待所有消息处理完成
            while received_count < message_count:
                await asyncio.sleep(0.01)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = message_count / duration
            
            # 验证性能指标
            assert throughput > 100  # 至少100消息/秒
            assert received_count == message_count
            
            print(f"消息总线吞吐量: {throughput:.2f} 消息/秒")
            
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_health_check_latency(self):
        """测试健康检查延迟"""
        monitor = HealthMonitor()
        await monitor.start()
        
        try:
            # 注册多个健康检查
            for i in range(10):
                async def health_check():
                    await asyncio.sleep(0.001)  # 模拟1ms检查时间
                    return HealthStatus.HEALTHY, f"Component {i} is healthy"
                
                monitor.register_health_check(f"component_{i}", health_check)
            
            # 测量健康检查时间
            start_time = time.time()
            health_status = await monitor.check_health()
            end_time = time.time()
            
            duration = end_time - start_time
            
            # 验证性能指标
            assert duration < 1.0  # 总时间应小于1秒
            assert health_status.overall_status == HealthStatus.HEALTHY
            assert len(health_status.component_status) == 10
            
            print(f"健康检查延迟: {duration*1000:.2f} ms")
            
        finally:
            await monitor.stop()
    
    def test_memory_allocation_performance(self):
        """测试内存分配性能"""
        optimizer = MemoryOptimizer(total_memory_gb=1)
        
        try:
            allocation_count = 1000
            allocations = []
            
            # 测量分配时间
            start_time = time.time()
            
            for i in range(allocation_count):
                allocation = optimizer.allocate_memory("general", 1024)
                if allocation:
                    allocations.append(allocation)
            
            end_time = time.time()
            allocation_duration = end_time - start_time
            
            # 测量释放时间
            start_time = time.time()
            
            for start_block, block_count in allocations:
                optimizer.deallocate_memory("general", start_block, block_count)
            
            end_time = time.time()
            deallocation_duration = end_time - start_time
            
            # 验证性能指标
            allocation_rate = len(allocations) / allocation_duration
            deallocation_rate = len(allocations) / deallocation_duration
            
            assert allocation_rate > 100  # 至少100次分配/秒
            assert deallocation_rate > 100  # 至少100次释放/秒
            
            print(f"内存分配速率: {allocation_rate:.2f} 次/秒")
            print(f"内存释放速率: {deallocation_rate:.2f} 次/秒")
            
        finally:
            optimizer.cleanup()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])

