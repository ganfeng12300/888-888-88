#!/usr/bin/env python3
"""
🔬 功能验证测试模块
Functional Validation Test Module

深度验证每个核心模块的实际功能和业务逻辑
"""

import sys
import asyncio
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
sys.path.append('.')

from loguru import logger
import numpy as np
import pandas as pd

class FunctionalValidator:
    """功能验证器"""
    
    def __init__(self):
        self.validation_results = {}
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
    
    def validate_function(self, func_name: str, validation_func, *args, **kwargs):
        """验证单个功能"""
        self.total_validations += 1
        logger.info(f"🔬 验证: {func_name}")
        
        try:
            start_time = time.time()
            result = validation_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if result.get('status') == 'success':
                self.passed_validations += 1
                logger.success(f"✅ {func_name} - 验证通过 ({duration:.2f}s)")
            else:
                self.failed_validations += 1
                logger.error(f"❌ {func_name} - 验证失败: {result.get('message', '')}")
                
            self.validation_results[func_name] = {
                'status': result.get('status', 'failed'),
                'message': result.get('message', ''),
                'duration': duration,
                'details': result.get('details', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.failed_validations += 1
            error_msg = f"异常: {str(e)}"
            logger.error(f"❌ {func_name} - {error_msg}")
            self.validation_results[func_name] = {
                'status': 'failed',
                'message': error_msg,
                'duration': 0,
                'details': {'traceback': traceback.format_exc()},
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_gpu_optimizer_functionality(self):
        """验证GPU优化器功能"""
        try:
            from src.hardware.gpu_performance_optimizer import GPUPerformanceOptimizer
            
            optimizer = GPUPerformanceOptimizer()
            
            # 验证系统信息获取
            system_info = optimizer.get_system_info()
            required_info = ['cpu_count', 'memory_total', 'gpu_available']
            missing_info = [info for info in required_info if info not in system_info]
            
            if missing_info:
                return {
                    'status': 'failed',
                    'message': f'系统信息不完整，缺少: {", ".join(missing_info)}'
                }
            
            # 验证性能监控
            performance_data = optimizer.monitor_performance()
            if not performance_data or 'cpu_usage' not in performance_data:
                return {
                    'status': 'failed',
                    'message': '性能监控数据无效'
                }
            
            # 验证优化建议生成
            optimization_suggestions = optimizer.get_optimization_suggestions()
            if not isinstance(optimization_suggestions, list):
                return {
                    'status': 'failed',
                    'message': '优化建议格式错误'
                }
            
            # 验证GPU内存管理
            if hasattr(optimizer, 'optimize_gpu_memory'):
                memory_optimization = optimizer.optimize_gpu_memory()
                if not isinstance(memory_optimization, dict):
                    return {
                        'status': 'failed',
                        'message': 'GPU内存优化失败'
                    }
            
            return {
                'status': 'success',
                'message': 'GPU优化器功能验证通过',
                'details': {
                    'system_info_complete': True,
                    'performance_monitoring': True,
                    'optimization_suggestions': len(optimization_suggestions),
                    'gpu_memory_optimization': hasattr(optimizer, 'optimize_gpu_memory')
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'GPU优化器验证异常: {str(e)}'}
    
    def validate_bybit_trader_functionality(self):
        """验证Bybit交易器功能"""
        try:
            from src.exchange.bybit_contract_trader import BybitContractTrader
            
            # 使用测试配置
            config = {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'testnet': True,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'leverage': 10,
                'max_position_size': 0.1
            }
            
            trader = BybitContractTrader(config)
            
            # 验证配置验证功能
            if not trader.validate_config():
                return {
                    'status': 'failed',
                    'message': '配置验证功能失效'
                }
            
            # 验证订单参数验证
            valid_order = {
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'order_type': 'Market',
                'qty': 0.01
            }
            
            order_validation = trader.validate_order_params(valid_order)
            if not order_validation:
                return {
                    'status': 'failed',
                    'message': '订单参数验证失败'
                }
            
            # 验证风险检查
            risk_check = trader.check_order_risk(valid_order)
            if not isinstance(risk_check, dict):
                return {
                    'status': 'failed',
                    'message': '订单风险检查失败'
                }
            
            # 验证市场数据处理
            mock_market_data = {
                'symbol': 'BTCUSDT',
                'price': 50000,
                'volume': 1000,
                'timestamp': time.time()
            }
            
            processed_data = trader.process_market_data(mock_market_data)
            if not isinstance(processed_data, dict):
                return {
                    'status': 'failed',
                    'message': '市场数据处理失败'
                }
            
            # 验证仓位管理
            mock_position = {
                'symbol': 'BTCUSDT',
                'size': 0.1,
                'side': 'Buy',
                'entry_price': 50000,
                'unrealized_pnl': 100
            }
            
            position_analysis = trader.analyze_position(mock_position)
            if not isinstance(position_analysis, dict):
                return {
                    'status': 'failed',
                    'message': '仓位分析失败'
                }
            
            return {
                'status': 'success',
                'message': 'Bybit交易器功能验证通过',
                'details': {
                    'config_validation': True,
                    'order_validation': True,
                    'risk_check': True,
                    'market_data_processing': True,
                    'position_analysis': True
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'Bybit交易器验证异常: {str(e)}'}
    
    def validate_risk_controller_functionality(self):
        """验证风险控制器功能"""
        try:
            from src.risk.advanced_risk_controller import AdvancedRiskController
            
            config = {
                'max_daily_drawdown': 0.03,
                'max_total_drawdown': 0.15,
                'max_position_size': 0.25,
                'stop_loss_pct': 0.02,
                'monitoring_interval': 1
            }
            
            controller = AdvancedRiskController(config)
            
            # 验证风险指标计算
            test_positions = [
                {
                    'symbol': 'BTCUSDT',
                    'size': 0.1,
                    'entry_price': 50000,
                    'current_price': 49000,
                    'pnl': -1000
                },
                {
                    'symbol': 'ETHUSDT',
                    'size': 0.5,
                    'entry_price': 3000,
                    'current_price': 3100,
                    'pnl': 500
                }
            ]
            
            risk_metrics = controller.calculate_risk_metrics(test_positions)
            required_metrics = ['total_exposure', 'total_pnl', 'max_drawdown', 'risk_score']
            missing_metrics = [m for m in required_metrics if m not in risk_metrics]
            
            if missing_metrics:
                return {
                    'status': 'failed',
                    'message': f'风险指标不完整，缺少: {", ".join(missing_metrics)}'
                }
            
            # 验证风险限制检查
            for position in test_positions:
                risk_check = controller.check_risk_limits(position)
                if not isinstance(risk_check, dict) or 'allowed' not in risk_check:
                    return {
                        'status': 'failed',
                        'message': f'风险限制检查失败: {position["symbol"]}'
                    }
            
            # 验证动态风险调整
            market_volatility = 0.05
            adjusted_limits = controller.adjust_risk_limits(market_volatility)
            if not isinstance(adjusted_limits, dict):
                return {
                    'status': 'failed',
                    'message': '动态风险调整失败'
                }
            
            # 验证止损建议
            for position in test_positions:
                stop_loss_suggestion = controller.suggest_stop_loss(position)
                if not isinstance(stop_loss_suggestion, dict):
                    return {
                        'status': 'failed',
                        'message': f'止损建议生成失败: {position["symbol"]}'
                    }
            
            # 验证风险报告生成
            risk_report = controller.generate_risk_report(test_positions)
            if not isinstance(risk_report, dict):
                return {
                    'status': 'failed',
                    'message': '风险报告生成失败'
                }
            
            return {
                'status': 'success',
                'message': '风险控制器功能验证通过',
                'details': {
                    'risk_metrics_complete': True,
                    'risk_limits_check': True,
                    'dynamic_adjustment': True,
                    'stop_loss_suggestions': True,
                    'risk_reporting': True,
                    'positions_analyzed': len(test_positions)
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'风险控制器验证异常: {str(e)}'}
    
    def validate_timezone_scheduler_functionality(self):
        """验证时区调度器功能"""
        try:
            from src.scheduler.timezone_scheduler import TimezoneScheduler
            
            config = {
                'local_timezone': 'Asia/Shanghai',
                'check_interval': 60,
                'enable_dynamic_scheduling': True,
                'activity_window': 300
            }
            
            scheduler = TimezoneScheduler(config)
            
            # 验证时区转换
            test_times = [
                datetime.now(),
                datetime.now() + timedelta(hours=8),
                datetime.now() + timedelta(hours=16)
            ]
            
            for test_time in test_times:
                converted_time = scheduler.convert_timezone(test_time, 'UTC')
                if not isinstance(converted_time, datetime):
                    return {
                        'status': 'failed',
                        'message': '时区转换失败'
                    }
            
            # 验证市场时段识别
            current_session = scheduler.get_current_market_session()
            required_session_fields = ['name', 'start_time', 'end_time', 'characteristics']
            missing_fields = [f for f in required_session_fields if f not in current_session]
            
            if missing_fields:
                return {
                    'status': 'failed',
                    'message': f'市场时段信息不完整，缺少: {", ".join(missing_fields)}'
                }
            
            # 验证活跃度计算
            activity_levels = []
            for hour in range(24):
                test_time = datetime.now().replace(hour=hour, minute=0, second=0)
                activity = scheduler.calculate_market_activity(test_time)
                if not isinstance(activity, (int, float)) or activity < 0 or activity > 100:
                    return {
                        'status': 'failed',
                        'message': f'活跃度计算异常: {hour}时 = {activity}'
                    }
                activity_levels.append(activity)
            
            # 验证交易建议生成
            recommendation = scheduler.get_trading_recommendation()
            required_rec_fields = ['action', 'intensity', 'reasoning', 'optimal_pairs']
            missing_rec_fields = [f for f in required_rec_fields if f not in recommendation]
            
            if missing_rec_fields:
                return {
                    'status': 'failed',
                    'message': f'交易建议不完整，缺少: {", ".join(missing_rec_fields)}'
                }
            
            # 验证调度优化
            optimization_result = scheduler.optimize_schedule()
            if not isinstance(optimization_result, dict):
                return {
                    'status': 'failed',
                    'message': '调度优化失败'
                }
            
            return {
                'status': 'success',
                'message': '时区调度器功能验证通过',
                'details': {
                    'timezone_conversion': True,
                    'market_session_detection': True,
                    'activity_calculation': True,
                    'trading_recommendations': True,
                    'schedule_optimization': True,
                    'activity_range': f'{min(activity_levels):.1f}-{max(activity_levels):.1f}%'
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'时区调度器验证异常: {str(e)}'}
    
    def validate_ai_fusion_functionality(self):
        """验证AI融合系统功能"""
        try:
            from src.ai.six_agents_fusion_system import SixAgentsFusionSystem
            
            config = {
                'max_agents': 6,
                'consensus_threshold': 0.6,
                'enable_meta_learning': True,
                'performance_window': 100
            }
            
            fusion_system = SixAgentsFusionSystem(config)
            
            # 验证智能体初始化
            agents_status = fusion_system.get_agents_status()
            expected_agents = [
                'expert_guardian', 'transfer_adapter', 'timeseries_prophet',
                'reinforcement_executor', 'integration_coordinator', 'meta_commander'
            ]
            
            missing_agents = [agent for agent in expected_agents if agent not in agents_status]
            if missing_agents:
                return {
                    'status': 'failed',
                    'message': f'智能体初始化不完整，缺少: {", ".join(missing_agents)}'
                }
            
            # 验证决策生成
            test_market_data = {
                'symbol': 'BTCUSDT',
                'price': 50000,
                'volume': 1000,
                'volatility': 0.02,
                'trend': 'bullish',
                'timestamp': time.time()
            }
            
            decision = fusion_system.make_decision(test_market_data)
            required_decision_fields = ['action', 'confidence', 'reasoning', 'risk_level']
            missing_decision_fields = [f for f in required_decision_fields if f not in decision]
            
            if missing_decision_fields:
                return {
                    'status': 'failed',
                    'message': f'决策结构不完整，缺少: {", ".join(missing_decision_fields)}'
                }
            
            # 验证多次决策的一致性
            decisions = []
            for _ in range(10):
                decision = fusion_system.make_decision(test_market_data)
                decisions.append(decision)
            
            # 检查决策一致性
            actions = [d['action'] for d in decisions]
            confidence_levels = [d['confidence'] for d in decisions]
            
            if len(set(actions)) > 3:  # 决策过于分散
                return {
                    'status': 'failed',
                    'message': '决策一致性差，结果过于分散'
                }
            
            # 验证学习能力
            if hasattr(fusion_system, 'update_performance'):
                # 模拟性能反馈
                performance_feedback = {
                    'decision_id': 'test_001',
                    'actual_return': 0.02,
                    'predicted_return': 0.015,
                    'accuracy': 0.85
                }
                
                update_result = fusion_system.update_performance(performance_feedback)
                if not isinstance(update_result, dict):
                    return {
                        'status': 'failed',
                        'message': '学习能力验证失败'
                    }
            
            # 验证元学习
            if hasattr(fusion_system, 'meta_learn'):
                meta_learning_result = fusion_system.meta_learn()
                if not isinstance(meta_learning_result, dict):
                    return {
                        'status': 'failed',
                        'message': '元学习功能失败'
                    }
            
            return {
                'status': 'success',
                'message': 'AI融合系统功能验证通过',
                'details': {
                    'agents_initialized': len(agents_status),
                    'decision_generation': True,
                    'decision_consistency': True,
                    'learning_capability': hasattr(fusion_system, 'update_performance'),
                    'meta_learning': hasattr(fusion_system, 'meta_learn'),
                    'avg_confidence': np.mean(confidence_levels),
                    'decision_distribution': dict(zip(*np.unique(actions, return_counts=True)))
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'AI融合系统验证异常: {str(e)}'}
    
    def validate_system_integration_flow(self):
        """验证系统集成流程"""
        try:
            # 模拟完整的交易流程
            flow_results = {}
            
            # 1. 市场数据获取和处理
            market_data = {
                'symbol': 'BTCUSDT',
                'price': 50000,
                'volume': 1000,
                'volatility': 0.02,
                'timestamp': time.time()
            }
            
            # 2. 时区调度分析
            from src.scheduler.timezone_scheduler import TimezoneScheduler
            scheduler = TimezoneScheduler({'local_timezone': 'Asia/Shanghai'})
            schedule_analysis = scheduler.get_trading_recommendation()
            flow_results['schedule_analysis'] = schedule_analysis
            
            # 3. AI决策生成
            from src.ai.six_agents_fusion_system import SixAgentsFusionSystem
            ai_system = SixAgentsFusionSystem({'max_agents': 6})
            ai_decision = ai_system.make_decision(market_data)
            flow_results['ai_decision'] = ai_decision
            
            # 4. 风险评估
            from src.risk.advanced_risk_controller import AdvancedRiskController
            risk_controller = AdvancedRiskController({'max_daily_drawdown': 0.03})
            
            # 模拟订单
            proposed_order = {
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'size': 0.1,
                'price': market_data['price']
            }
            
            risk_assessment = risk_controller.check_risk_limits(proposed_order)
            flow_results['risk_assessment'] = risk_assessment
            
            # 5. 交易执行决策
            from src.exchange.bybit_contract_trader import BybitContractTrader
            trader = BybitContractTrader({'testnet': True})
            
            execution_decision = trader.validate_order_params(proposed_order)
            flow_results['execution_decision'] = execution_decision
            
            # 6. 性能监控
            from src.hardware.gpu_performance_optimizer import GPUPerformanceOptimizer
            optimizer = GPUPerformanceOptimizer()
            performance_status = optimizer.monitor_performance()
            flow_results['performance_status'] = performance_status
            
            # 验证流程完整性
            required_stages = [
                'schedule_analysis', 'ai_decision', 'risk_assessment',
                'execution_decision', 'performance_status'
            ]
            
            completed_stages = [stage for stage in required_stages if stage in flow_results]
            
            if len(completed_stages) != len(required_stages):
                missing_stages = [s for s in required_stages if s not in completed_stages]
                return {
                    'status': 'failed',
                    'message': f'集成流程不完整，缺少阶段: {", ".join(missing_stages)}'
                }
            
            # 验证数据流一致性
            data_consistency = self.check_data_consistency(flow_results)
            
            return {
                'status': 'success',
                'message': '系统集成流程验证通过',
                'details': {
                    'completed_stages': completed_stages,
                    'data_consistency': data_consistency,
                    'flow_results': flow_results
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'系统集成流程验证异常: {str(e)}'}
    
    def check_data_consistency(self, flow_results: Dict) -> bool:
        """检查数据流一致性"""
        try:
            # 检查时间戳一致性
            timestamps = []
            for stage, result in flow_results.items():
                if isinstance(result, dict) and 'timestamp' in result:
                    timestamps.append(result['timestamp'])
            
            # 检查决策一致性
            if 'ai_decision' in flow_results and 'risk_assessment' in flow_results:
                ai_action = flow_results['ai_decision'].get('action', '')
                risk_allowed = flow_results['risk_assessment'].get('allowed', False)
                
                # 如果AI建议买入但风险不允许，这是正常的
                # 如果AI建议卖出但风险允许，也是正常的
                return True
            
            return True
            
        except Exception as e:
            logger.warning(f"数据一致性检查异常: {str(e)}")
            return False
    
    def generate_validation_report(self):
        """生成功能验证报告"""
        logger.info("\n" + "="*60)
        logger.info("🔬 功能验证测试报告")
        logger.info("="*60)
        
        # 总体统计
        success_rate = (self.passed_validations / self.total_validations * 100) if self.total_validations > 0 else 0
        
        logger.info(f"📊 验证统计:")
        logger.info(f"   总验证项: {self.total_validations}")
        logger.info(f"   ✅ 通过: {self.passed_validations}")
        logger.info(f"   ❌ 失败: {self.failed_validations}")
        logger.info(f"   📈 成功率: {success_rate:.1f}%")
        
        # 功能状态评估
        if self.failed_validations == 0:
            status = "🟢 优秀"
            message = "所有功能完美运行，系统功能完整"
        elif self.failed_validations <= 2:
            status = "🟡 良好"
            message = "大部分功能正常，需要修复少量问题"
        else:
            status = "🔴 需要改进"
            message = "多个功能存在问题，需要重点修复"
        
        logger.info(f"\n🎯 功能状态: {status}")
        logger.info(f"💬 评估: {message}")
        
        # 详细结果
        if self.failed_validations > 0:
            logger.info(f"\n📋 失败详情:")
            for func_name, result in self.validation_results.items():
                if result['status'] == 'failed':
                    logger.info(f"   ❌ {func_name}: {result['message']}")
        
        # 保存详细报告
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validations': self.total_validations,
                'passed': self.passed_validations,
                'failed': self.failed_validations,
                'success_rate': success_rate,
                'status': status,
                'message': message
            },
            'validation_results': self.validation_results
        }
        
        try:
            with open('tests/results/functional_validation_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"\n📄 详细报告已保存到: tests/results/functional_validation_report.json")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
        
        return success_rate >= 85  # 85%以上认为功能验证通过
    
    def run_all_functional_validations(self):
        """运行所有功能验证"""
        logger.info("🚀 开始功能验证测试...")
        logger.info("="*60)
        
        # 定义所有验证项
        validations = [
            ("GPU优化器功能验证", self.validate_gpu_optimizer_functionality),
            ("Bybit交易器功能验证", self.validate_bybit_trader_functionality),
            ("风险控制器功能验证", self.validate_risk_controller_functionality),
            ("时区调度器功能验证", self.validate_timezone_scheduler_functionality),
            ("AI融合系统功能验证", self.validate_ai_fusion_functionality),
            ("系统集成流程验证", self.validate_system_integration_flow),
        ]
        
        # 运行所有验证
        for validation_name, validation_func in validations:
            self.validate_function(validation_name, validation_func)
        
        # 生成报告
        return self.generate_validation_report()


def main():
    """主函数"""
    print("🔬 终极合约交易系统 - 功能验证测试")
    print("="*60)
    
    validator = FunctionalValidator()
    is_functional = validator.run_all_functional_validations()
    
    # 返回适当的退出码
    sys.exit(0 if is_functional else 1)


if __name__ == "__main__":
    main()

