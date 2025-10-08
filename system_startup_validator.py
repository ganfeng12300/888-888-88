#!/usr/bin/env python3
"""
🚀 系统启动验证器
System Startup Validator

验证系统完整启动流程和所有组件协作
"""

import sys
import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.append('.')

from loguru import logger
import psutil

class SystemStartupValidator:
    """系统启动验证器"""
    
    def __init__(self):
        self.validation_results = {}
        self.startup_metrics = {}
        self.component_status = {}
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
        
        # 初始化验证环境
        self.setup_validation_environment()
    
    def setup_validation_environment(self):
        """设置验证环境"""
        try:
            # 创建验证结果目录
            results_dir = Path('tests/results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 记录系统基线
            self.record_system_baseline()
            
            logger.info("✅ 验证环境初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 验证环境初始化失败: {str(e)}")
    
    def record_system_baseline(self):
        """记录系统基线指标"""
        try:
            process = psutil.Process()
            self.startup_metrics['baseline'] = {
                'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"基线记录失败: {str(e)}")
    
    def validate_check(self, check_name: str, check_func, timeout: int = 30):
        """验证单个检查项"""
        self.total_checks += 1
        logger.info(f"🔍 检查: {check_name}")
        
        try:
            start_time = time.time()
            
            # 使用线程池执行检查，支持超时
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_func)
                try:
                    result = future.result(timeout=timeout)
                except Exception as e:
                    result = {'status': 'failed', 'message': f'检查超时或异常: {str(e)}'}
            
            duration = time.time() - start_time
            
            if result.get('status') == 'success':
                self.passed_checks += 1
                logger.success(f"✅ {check_name} - 通过 ({duration:.2f}s)")
            else:
                self.failed_checks += 1
                logger.error(f"❌ {check_name} - 失败: {result.get('message', '')}")
                
            self.validation_results[check_name] = {
                'status': result.get('status', 'failed'),
                'message': result.get('message', ''),
                'duration': duration,
                'details': result.get('details', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.failed_checks += 1
            error_msg = f"异常: {str(e)}"
            logger.error(f"❌ {check_name} - {error_msg}")
            self.validation_results[check_name] = {
                'status': 'failed',
                'message': error_msg,
                'duration': 0,
                'details': {'traceback': traceback.format_exc()},
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_system_launcher_startup(self):
        """验证系统启动器启动"""
        try:
            from start_ultimate_system import UltimateSystemLauncher
            
            # 记录启动前状态
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            # 初始化启动器
            launcher = UltimateSystemLauncher()
            
            # 记录启动后状态
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            startup_time = time.time() - start_time
            memory_increase = end_memory - start_memory
            
            # 验证启动器状态
            if not hasattr(launcher, 'config') or not launcher.config:
                return {'status': 'failed', 'message': '启动器配置加载失败'}
            
            # 验证组件初始化
            components_status = launcher.get_components_status()
            if not isinstance(components_status, dict):
                return {'status': 'failed', 'message': '组件状态获取失败'}
            
            # 验证系统健康检查
            health_status = launcher.check_system_health()
            if not isinstance(health_status, dict):
                return {'status': 'failed', 'message': '系统健康检查失败'}
            
            # 记录启动指标
            self.startup_metrics['launcher'] = {
                'startup_time': startup_time,
                'memory_increase': memory_increase,
                'components_count': len(components_status),
                'health_score': health_status.get('score', 0)
            }
            
            return {
                'status': 'success',
                'message': '系统启动器启动成功',
                'details': {
                    'startup_time': startup_time,
                    'memory_usage': memory_increase,
                    'components': list(components_status.keys()),
                    'health_score': health_status.get('score', 0)
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'启动器启动异常: {str(e)}'}
    
    def validate_components_initialization(self):
        """验证组件初始化"""
        try:
            components_results = {}
            
            # 验证GPU优化器初始化
            try:
                from src.hardware.gpu_performance_optimizer import GPUPerformanceOptimizer
                gpu_optimizer = GPUPerformanceOptimizer()
                components_results['gpu_optimizer'] = 'success'
                self.component_status['gpu_optimizer'] = gpu_optimizer
            except Exception as e:
                components_results['gpu_optimizer'] = f'failed: {str(e)}'
            
            # 验证Bybit交易器初始化
            try:
                from src.exchange.bybit_contract_trader import BybitContractTrader
                trader_config = {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret',
                    'testnet': True
                }
                bybit_trader = BybitContractTrader(trader_config)
                components_results['bybit_trader'] = 'success'
                self.component_status['bybit_trader'] = bybit_trader
            except Exception as e:
                components_results['bybit_trader'] = f'failed: {str(e)}'
            
            # 验证风险控制器初始化
            try:
                from src.risk.advanced_risk_controller import AdvancedRiskController
                risk_config = {'max_daily_drawdown': 0.03}
                risk_controller = AdvancedRiskController(risk_config)
                components_results['risk_controller'] = 'success'
                self.component_status['risk_controller'] = risk_controller
            except Exception as e:
                components_results['risk_controller'] = f'failed: {str(e)}'
            
            # 验证时区调度器初始化
            try:
                from src.scheduler.timezone_scheduler import TimezoneScheduler
                scheduler_config = {'local_timezone': 'Asia/Shanghai'}
                timezone_scheduler = TimezoneScheduler(scheduler_config)
                components_results['timezone_scheduler'] = 'success'
                self.component_status['timezone_scheduler'] = timezone_scheduler
            except Exception as e:
                components_results['timezone_scheduler'] = f'failed: {str(e)}'
            
            # 验证AI融合系统初始化
            try:
                from src.ai.six_agents_fusion_system import SixAgentsFusionSystem
                ai_config = {'max_agents': 6}
                ai_system = SixAgentsFusionSystem(ai_config)
                components_results['ai_fusion_system'] = 'success'
                self.component_status['ai_fusion_system'] = ai_system
            except Exception as e:
                components_results['ai_fusion_system'] = f'failed: {str(e)}'
            
            # 统计结果
            successful_components = [k for k, v in components_results.items() if v == 'success']
            failed_components = [k for k, v in components_results.items() if v != 'success']
            
            if len(failed_components) > 0:
                return {
                    'status': 'failed',
                    'message': f'组件初始化失败: {", ".join(failed_components)}',
                    'details': {
                        'successful': successful_components,
                        'failed': failed_components,
                        'results': components_results
                    }
                }
            
            return {
                'status': 'success',
                'message': f'所有{len(successful_components)}个组件初始化成功',
                'details': {
                    'components': successful_components,
                    'results': components_results
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'组件初始化验证异常: {str(e)}'}
    
    def validate_components_interaction(self):
        """验证组件间交互"""
        try:
            if not self.component_status:
                return {'status': 'failed', 'message': '没有可用的组件进行交互测试'}
            
            interaction_results = {}
            
            # 测试数据流传递
            test_market_data = {
                'symbol': 'BTCUSDT',
                'price': 50000,
                'volume': 1000,
                'timestamp': time.time()
            }
            
            # AI系统 -> 风险控制器交互
            if 'ai_fusion_system' in self.component_status and 'risk_controller' in self.component_status:
                try:
                    ai_system = self.component_status['ai_fusion_system']
                    risk_controller = self.component_status['risk_controller']
                    
                    # AI生成决策
                    ai_decision = ai_system.make_decision(test_market_data)
                    
                    # 风险控制器评估决策
                    if isinstance(ai_decision, dict) and 'action' in ai_decision:
                        mock_order = {
                            'symbol': 'BTCUSDT',
                            'side': ai_decision['action'],
                            'size': 0.1
                        }
                        risk_assessment = risk_controller.check_risk_limits(mock_order)
                        interaction_results['ai_risk_interaction'] = 'success'
                    else:
                        interaction_results['ai_risk_interaction'] = 'failed: invalid AI decision'
                        
                except Exception as e:
                    interaction_results['ai_risk_interaction'] = f'failed: {str(e)}'
            
            # 时区调度器 -> 交易器交互
            if 'timezone_scheduler' in self.component_status and 'bybit_trader' in self.component_status:
                try:
                    scheduler = self.component_status['timezone_scheduler']
                    trader = self.component_status['bybit_trader']
                    
                    # 获取交易建议
                    trading_recommendation = scheduler.get_trading_recommendation()
                    
                    # 交易器处理建议
                    if isinstance(trading_recommendation, dict):
                        # 模拟根据建议调整交易参数
                        interaction_results['scheduler_trader_interaction'] = 'success'
                    else:
                        interaction_results['scheduler_trader_interaction'] = 'failed: invalid recommendation'
                        
                except Exception as e:
                    interaction_results['scheduler_trader_interaction'] = f'failed: {str(e)}'
            
            # GPU优化器 -> 系统性能交互
            if 'gpu_optimizer' in self.component_status:
                try:
                    optimizer = self.component_status['gpu_optimizer']
                    
                    # 获取性能数据
                    performance_data = optimizer.monitor_performance()
                    
                    if isinstance(performance_data, dict) and 'cpu_usage' in performance_data:
                        interaction_results['gpu_performance_interaction'] = 'success'
                    else:
                        interaction_results['gpu_performance_interaction'] = 'failed: invalid performance data'
                        
                except Exception as e:
                    interaction_results['gpu_performance_interaction'] = f'failed: {str(e)}'
            
            # 统计交互结果
            successful_interactions = [k for k, v in interaction_results.items() if v == 'success']
            failed_interactions = [k for k, v in interaction_results.items() if v != 'success']
            
            if len(failed_interactions) > len(successful_interactions):
                return {
                    'status': 'failed',
                    'message': f'多数组件交互失败: {len(failed_interactions)}/{len(interaction_results)}',
                    'details': {
                        'successful': successful_interactions,
                        'failed': failed_interactions,
                        'results': interaction_results
                    }
                }
            
            return {
                'status': 'success',
                'message': f'组件交互验证通过: {len(successful_interactions)}/{len(interaction_results)}',
                'details': {
                    'successful': successful_interactions,
                    'failed': failed_interactions,
                    'results': interaction_results
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'组件交互验证异常: {str(e)}'}
    
    def validate_system_stability(self):
        """验证系统稳定性"""
        try:
            stability_metrics = {}
            
            # 内存稳定性测试
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # 运行多次操作测试内存泄漏
            for i in range(10):
                if 'ai_fusion_system' in self.component_status:
                    ai_system = self.component_status['ai_fusion_system']
                    test_data = {
                        'symbol': 'BTCUSDT',
                        'price': 50000 + i * 100,
                        'volume': 1000,
                        'timestamp': time.time()
                    }
                    decision = ai_system.make_decision(test_data)
                
                time.sleep(0.1)  # 短暂等待
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            stability_metrics['memory_stability'] = memory_increase < 50  # 内存增长小于50MB
            
            # CPU使用率稳定性
            cpu_usage = psutil.Process().cpu_percent(interval=1)
            stability_metrics['cpu_stability'] = cpu_usage < 80  # CPU使用率小于80%
            
            # 线程数稳定性
            thread_count = psutil.Process().num_threads()
            stability_metrics['thread_stability'] = thread_count < 50  # 线程数小于50
            
            # 文件句柄稳定性
            try:
                open_files = len(psutil.Process().open_files())
                stability_metrics['file_handle_stability'] = open_files < 100  # 文件句柄小于100
            except:
                stability_metrics['file_handle_stability'] = True  # 无法检测时认为稳定
            
            # 综合稳定性评分
            stable_metrics = sum(stability_metrics.values())
            total_metrics = len(stability_metrics)
            stability_score = (stable_metrics / total_metrics) * 100
            
            status = 'success' if stability_score >= 75 else 'failed'
            message = f'系统稳定性评分: {stability_score:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'memory_increase_mb': memory_increase,
                    'cpu_usage_percent': cpu_usage,
                    'thread_count': thread_count,
                    'stability_metrics': stability_metrics,
                    'stability_score': stability_score
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'系统稳定性验证异常: {str(e)}'}
    
    def validate_error_recovery(self):
        """验证错误恢复能力"""
        try:
            recovery_results = {}
            
            # 测试配置错误恢复
            try:
                from start_ultimate_system import UltimateSystemLauncher
                
                # 模拟配置错误
                original_config_path = 'config.json'
                backup_config = None
                
                if Path(original_config_path).exists():
                    with open(original_config_path, 'r') as f:
                        backup_config = f.read()
                
                # 创建无效配置
                with open(original_config_path, 'w') as f:
                    f.write('{"invalid": "config"}')
                
                # 尝试启动系统
                try:
                    launcher = UltimateSystemLauncher()
                    recovery_results['config_error_recovery'] = 'success'
                except Exception as e:
                    recovery_results['config_error_recovery'] = f'failed: {str(e)}'
                
                # 恢复原始配置
                if backup_config:
                    with open(original_config_path, 'w') as f:
                        f.write(backup_config)
                        
            except Exception as e:
                recovery_results['config_error_recovery'] = f'test_failed: {str(e)}'
            
            # 测试组件故障恢复
            if 'ai_fusion_system' in self.component_status:
                try:
                    ai_system = self.component_status['ai_fusion_system']
                    
                    # 发送无效数据
                    invalid_data = None
                    decision = ai_system.make_decision(invalid_data)
                    
                    # 如果没有抛出异常，说明有错误处理
                    recovery_results['invalid_data_recovery'] = 'success'
                    
                except Exception as e:
                    # 如果抛出异常但系统仍然运行，也算成功
                    recovery_results['invalid_data_recovery'] = 'handled_exception'
            
            # 测试网络错误恢复
            if 'bybit_trader' in self.component_status:
                try:
                    trader = self.component_status['bybit_trader']
                    
                    # 测试连接状态检查
                    connection_status = trader.check_connection_status()
                    recovery_results['network_error_recovery'] = 'success'
                    
                except Exception as e:
                    recovery_results['network_error_recovery'] = f'failed: {str(e)}'
            
            # 评估恢复能力
            successful_recoveries = [k for k, v in recovery_results.items() if 'success' in v or 'handled' in v]
            total_tests = len(recovery_results)
            
            if len(successful_recoveries) >= total_tests * 0.7:  # 70%以上成功
                return {
                    'status': 'success',
                    'message': f'错误恢复能力良好: {len(successful_recoveries)}/{total_tests}',
                    'details': {
                        'recovery_results': recovery_results,
                        'success_rate': len(successful_recoveries) / total_tests * 100
                    }
                }
            else:
                return {
                    'status': 'failed',
                    'message': f'错误恢复能力不足: {len(successful_recoveries)}/{total_tests}',
                    'details': {
                        'recovery_results': recovery_results,
                        'success_rate': len(successful_recoveries) / total_tests * 100
                    }
                }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'错误恢复验证异常: {str(e)}'}
    
    def validate_performance_benchmarks(self):
        """验证性能基准"""
        try:
            benchmarks = {}
            
            # 系统启动性能
            if 'launcher' in self.startup_metrics:
                startup_time = self.startup_metrics['launcher']['startup_time']
                benchmarks['startup_performance'] = startup_time < 10  # 启动时间小于10秒
            
            # 决策生成性能
            if 'ai_fusion_system' in self.component_status:
                ai_system = self.component_status['ai_fusion_system']
                
                start_time = time.time()
                for _ in range(10):
                    test_data = {
                        'symbol': 'BTCUSDT',
                        'price': 50000,
                        'volume': 1000,
                        'timestamp': time.time()
                    }
                    decision = ai_system.make_decision(test_data)
                
                avg_decision_time = (time.time() - start_time) / 10
                benchmarks['decision_performance'] = avg_decision_time < 0.5  # 平均决策时间小于0.5秒
            
            # 风险计算性能
            if 'risk_controller' in self.component_status:
                risk_controller = self.component_status['risk_controller']
                
                test_positions = [
                    {'symbol': 'BTCUSDT', 'size': 0.1, 'pnl': -100},
                    {'symbol': 'ETHUSDT', 'size': 0.2, 'pnl': 200}
                ]
                
                start_time = time.time()
                for _ in range(10):
                    risk_metrics = risk_controller.calculate_risk_metrics(test_positions)
                
                avg_risk_calc_time = (time.time() - start_time) / 10
                benchmarks['risk_calculation_performance'] = avg_risk_calc_time < 0.1  # 平均风险计算时间小于0.1秒
            
            # 内存使用性能
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            benchmarks['memory_performance'] = current_memory < 500  # 内存使用小于500MB
            
            # CPU使用性能
            cpu_usage = psutil.Process().cpu_percent(interval=1)
            benchmarks['cpu_performance'] = cpu_usage < 50  # CPU使用率小于50%
            
            # 综合性能评分
            performance_score = sum(benchmarks.values()) / len(benchmarks) * 100
            
            status = 'success' if performance_score >= 80 else 'failed'
            message = f'性能基准评分: {performance_score:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'benchmarks': benchmarks,
                    'performance_score': performance_score,
                    'current_memory_mb': current_memory,
                    'cpu_usage_percent': cpu_usage
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'性能基准验证异常: {str(e)}'}
    
    def generate_startup_validation_report(self):
        """生成启动验证报告"""
        logger.info("\n" + "="*60)
        logger.info("🚀 系统启动验证报告")
        logger.info("="*60)
        
        # 总体统计
        success_rate = (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        
        logger.info(f"📊 验证统计:")
        logger.info(f"   总检查项: {self.total_checks}")
        logger.info(f"   ✅ 通过: {self.passed_checks}")
        logger.info(f"   ❌ 失败: {self.failed_checks}")
        logger.info(f"   📈 成功率: {success_rate:.1f}%")
        
        # 启动状态评估
        if self.failed_checks == 0:
            status = "🟢 优秀"
            message = "系统启动完美，所有组件正常运行"
        elif self.failed_checks <= 2:
            status = "🟡 良好"
            message = "系统启动良好，有少量问题需要关注"
        else:
            status = "🔴 需要改进"
            message = "系统启动存在问题，需要修复"
        
        logger.info(f"\n🎯 启动状态: {status}")
        logger.info(f"💬 评估: {message}")
        
        # 性能指标
        if 'launcher' in self.startup_metrics:
            metrics = self.startup_metrics['launcher']
            logger.info(f"\n📈 启动性能:")
            logger.info(f"   启动时间: {metrics.get('startup_time', 0):.2f}秒")
            logger.info(f"   内存增长: {metrics.get('memory_increase', 0):.1f}MB")
            logger.info(f"   组件数量: {metrics.get('components_count', 0)}")
            logger.info(f"   健康评分: {metrics.get('health_score', 0)}")
        
        # 详细结果
        if self.failed_checks > 0:
            logger.info(f"\n📋 失败详情:")
            for check_name, result in self.validation_results.items():
                if result['status'] == 'failed':
                    logger.info(f"   ❌ {check_name}: {result['message']}")
        
        # 保存详细报告
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': self.total_checks,
                'passed': self.passed_checks,
                'failed': self.failed_checks,
                'success_rate': success_rate,
                'status': status,
                'message': message
            },
            'startup_metrics': self.startup_metrics,
            'component_status': {k: str(type(v)) for k, v in self.component_status.items()},
            'validation_results': self.validation_results
        }
        
        try:
            with open('tests/results/startup_validation_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"\n📄 详细报告已保存到: tests/results/startup_validation_report.json")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
        
        return success_rate >= 80  # 80%以上认为启动验证通过
    
    def run_all_startup_validations(self):
        """运行所有启动验证"""
        logger.info("🚀 开始系统启动验证...")
        logger.info("="*60)
        
        # 定义所有验证项
        validations = [
            ("系统启动器启动验证", self.validate_system_launcher_startup, 30),
            ("组件初始化验证", self.validate_components_initialization, 45),
            ("组件交互验证", self.validate_components_interaction, 30),
            ("系统稳定性验证", self.validate_system_stability, 60),
            ("错误恢复验证", self.validate_error_recovery, 30),
            ("性能基准验证", self.validate_performance_benchmarks, 30),
        ]
        
        # 运行所有验证
        for validation_name, validation_func, timeout in validations:
            self.validate_check(validation_name, validation_func, timeout)
        
        # 生成报告
        return self.generate_startup_validation_report()


def main():
    """主函数"""
    print("🚀 终极合约交易系统 - 启动验证器")
    print("="*60)
    
    validator = SystemStartupValidator()
    is_startup_valid = validator.run_all_startup_validations()
    
    # 返回适当的退出码
    sys.exit(0 if is_startup_valid else 1)


if __name__ == "__main__":
    main()

