#!/usr/bin/env python3
"""
🛡️ 专家系统守护者 - 规则引擎风险控制
基于专家知识的规则引擎系统
专为生产级实盘交易设计，提供多层风险控制
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timezone, timedelta
import json
from dataclasses import dataclass, field
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import re

class RiskLevel(Enum):
    """风险等级"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    CRITICAL = 6

class ActionType(Enum):
    """动作类型"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class Rule:
    """交易规则"""
    rule_id: str
    name: str
    description: str
    condition: str  # 条件表达式
    action: ActionType
    priority: int  # 优先级 1-10
    risk_level: RiskLevel
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class RuleEvaluation:
    """规则评估结果"""
    rule_id: str
    rule_name: str
    is_triggered: bool
    condition_result: bool
    action: ActionType
    risk_level: RiskLevel
    message: str
    parameters: Dict[str, Any]
    evaluation_time: float
    timestamp: datetime

@dataclass
class GuardianDecision:
    """守护者决策"""
    final_action: ActionType
    risk_score: float
    triggered_rules: List[RuleEvaluation]
    warnings: List[str]
    recommendations: List[str]
    position_adjustment: float  # 仓位调整建议 0-1
    max_leverage: float
    stop_loss_required: bool
    take_profit_required: bool
    timestamp: datetime

class ExpertSystemGuardian:
    """专家系统守护者"""
    
    def __init__(self):
        # 规则库
        self.rules = {}
        self.rule_categories = {
            'risk_management': [],
            'position_sizing': [],
            'market_conditions': [],
            'technical_analysis': [],
            'fundamental_analysis': [],
            'portfolio_management': [],
            'emergency_controls': []
        }
        
        # 市场状态
        self.market_state = {
            'volatility': 0.0,
            'trend_strength': 0.0,
            'volume_profile': 0.0,
            'sentiment': 0.0,
            'news_impact': 0.0,
            'correlation_risk': 0.0
        }
        
        # 账户状态
        self.account_state = {
            'balance': 0.0,
            'equity': 0.0,
            'margin_used': 0.0,
            'margin_available': 0.0,
            'unrealized_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'open_positions': 0
        }
        
        # 风险参数
        self.risk_parameters = {
            'max_position_size': 0.05,  # 最大单笔仓位5%
            'max_total_exposure': 0.20,  # 最大总敞口20%
            'max_leverage': 10.0,  # 最大杠杆10倍
            'max_daily_loss': 0.02,  # 最大日亏损2%
            'max_drawdown': 0.10,  # 最大回撤10%
            'min_win_rate': 0.40,  # 最小胜率40%
            'max_correlation': 0.70,  # 最大相关性70%
            'volatility_threshold': 0.30,  # 波动率阈值30%
            'news_impact_threshold': 0.80  # 新闻影响阈值80%
        }
        
        # 性能统计
        self.performance_stats = {
            'total_evaluations': 0,
            'rules_triggered': 0,
            'blocks_issued': 0,
            'warnings_issued': 0,
            'emergency_stops': 0,
            'avg_evaluation_time': 0.0,
            'last_evaluation': None
        }
        
        # 实时状态
        self.last_decision = None
        self.last_risk_score = 0.0
        self.performance_score = 0.5
        
        # 初始化默认规则
        self._initialize_default_rules()
        
        logger.info("🛡️ 专家系统守护者初始化完成")
    
    def _initialize_default_rules(self):
        """初始化默认规则"""
        
        # 风险管理规则
        self.add_rule(Rule(
            rule_id="RM001",
            name="最大单笔仓位限制",
            description="单笔交易仓位不得超过账户资金的5%",
            condition="position_size > risk_parameters['max_position_size']",
            action=ActionType.BLOCK,
            priority=9,
            risk_level=RiskLevel.HIGH,
            parameters={'max_size': 0.05}
        ), 'risk_management')
        
        self.add_rule(Rule(
            rule_id="RM002",
            name="最大总敞口限制",
            description="总敞口不得超过账户资金的20%",
            condition="total_exposure > risk_parameters['max_total_exposure']",
            action=ActionType.BLOCK,
            priority=9,
            risk_level=RiskLevel.HIGH,
            parameters={'max_exposure': 0.20}
        ), 'risk_management')
        
        self.add_rule(Rule(
            rule_id="RM003",
            name="最大日亏损限制",
            description="日亏损超过2%时停止交易",
            condition="daily_pnl < -risk_parameters['max_daily_loss']",
            action=ActionType.EMERGENCY_STOP,
            priority=10,
            risk_level=RiskLevel.CRITICAL,
            parameters={'max_daily_loss': 0.02}
        ), 'risk_management')
        
        self.add_rule(Rule(
            rule_id="RM004",
            name="最大回撤限制",
            description="回撤超过10%时紧急停止",
            condition="max_drawdown > risk_parameters['max_drawdown']",
            action=ActionType.EMERGENCY_STOP,
            priority=10,
            risk_level=RiskLevel.CRITICAL,
            parameters={'max_drawdown': 0.10}
        ), 'risk_management')
        
        # 市场条件规则
        self.add_rule(Rule(
            rule_id="MC001",
            name="高波动率警告",
            description="市场波动率过高时发出警告",
            condition="volatility > risk_parameters['volatility_threshold']",
            action=ActionType.WARN,
            priority=6,
            risk_level=RiskLevel.MEDIUM,
            parameters={'volatility_threshold': 0.30}
        ), 'market_conditions')
        
        self.add_rule(Rule(
            rule_id="MC002",
            name="重大新闻影响",
            description="重大新闻影响时减少仓位",
            condition="news_impact > risk_parameters['news_impact_threshold']",
            action=ActionType.REDUCE,
            priority=7,
            risk_level=RiskLevel.HIGH,
            parameters={'news_threshold': 0.80, 'reduction_factor': 0.5}
        ), 'market_conditions')
        
        # 技术分析规则
        self.add_rule(Rule(
            rule_id="TA001",
            name="趋势强度检查",
            description="趋势强度不足时警告",
            condition="trend_strength < 0.3",
            action=ActionType.WARN,
            priority=4,
            risk_level=RiskLevel.LOW,
            parameters={'min_trend_strength': 0.3}
        ), 'technical_analysis')
        
        self.add_rule(Rule(
            rule_id="TA002",
            name="支撑阻力位检查",
            description="价格接近关键支撑阻力位时警告",
            condition="near_support_resistance",
            action=ActionType.WARN,
            priority=5,
            risk_level=RiskLevel.MEDIUM,
            parameters={'distance_threshold': 0.02}
        ), 'technical_analysis')
        
        # 组合管理规则
        self.add_rule(Rule(
            rule_id="PM001",
            name="相关性风险控制",
            description="持仓相关性过高时限制新仓位",
            condition="correlation_risk > risk_parameters['max_correlation']",
            action=ActionType.BLOCK,
            priority=7,
            risk_level=RiskLevel.HIGH,
            parameters={'max_correlation': 0.70}
        ), 'portfolio_management')
        
        self.add_rule(Rule(
            rule_id="PM002",
            name="胜率监控",
            description="胜率过低时发出警告",
            condition="win_rate < risk_parameters['min_win_rate'] and total_trades > 20",
            action=ActionType.WARN,
            priority=5,
            risk_level=RiskLevel.MEDIUM,
            parameters={'min_win_rate': 0.40, 'min_trades': 20}
        ), 'portfolio_management')
        
        # 紧急控制规则
        self.add_rule(Rule(
            rule_id="EC001",
            name="系统异常检测",
            description="检测到系统异常时紧急停止",
            condition="system_error or connection_lost",
            action=ActionType.EMERGENCY_STOP,
            priority=10,
            risk_level=RiskLevel.CRITICAL,
            parameters={}
        ), 'emergency_controls')
        
        logger.info(f"✅ 已加载 {len(self.rules)} 条默认规则")
    
    def add_rule(self, rule: Rule, category: str = 'risk_management'):
        """添加规则"""
        try:
            self.rules[rule.rule_id] = rule
            if category in self.rule_categories:
                self.rule_categories[category].append(rule.rule_id)
            logger.debug(f"✅ 规则已添加: {rule.name} ({rule.rule_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 规则添加失败: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        try:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                del self.rules[rule_id]
                
                # 从分类中移除
                for category, rule_ids in self.rule_categories.items():
                    if rule_id in rule_ids:
                        rule_ids.remove(rule_id)
                
                logger.debug(f"✅ 规则已移除: {rule.name} ({rule_id})")
                return True
            else:
                logger.warning(f"⚠️ 规则不存在: {rule_id}")
                return False
        except Exception as e:
            logger.error(f"❌ 规则移除失败: {e}")
            return False
    
    def update_rule(self, rule_id: str, **kwargs) -> bool:
        """更新规则"""
        try:
            if rule_id not in self.rules:
                logger.warning(f"⚠️ 规则不存在: {rule_id}")
                return False
            
            rule = self.rules[rule_id]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.debug(f"✅ 规则已更新: {rule.name} ({rule_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 规则更新失败: {e}")
            return False
    
    async def evaluate_trading_decision(self, trading_request: Dict[str, Any]) -> GuardianDecision:
        """评估交易决策"""
        try:
            start_time = time.time()
            
            # 更新市场和账户状态
            self._update_states(trading_request)
            
            # 评估所有规则
            rule_evaluations = []
            triggered_rules = []
            
            for rule_id, rule in self.rules.items():
                if not rule.is_active:
                    continue
                
                evaluation = await self._evaluate_rule(rule, trading_request)
                rule_evaluations.append(evaluation)
                
                if evaluation.is_triggered:
                    triggered_rules.append(evaluation)
                    rule.last_triggered = datetime.now(timezone.utc)
                    rule.trigger_count += 1
            
            # 计算最终决策
            decision = self._make_final_decision(triggered_rules, trading_request)
            
            # 更新性能统计
            evaluation_time = time.time() - start_time
            self._update_performance_stats(evaluation_time, len(triggered_rules))
            
            # 更新状态
            self.last_decision = decision
            self.last_risk_score = decision.risk_score
            
            logger.info(f"🛡️ 交易决策评估完成 - 动作: {decision.final_action.value}, 风险分数: {decision.risk_score:.3f}")
            return decision
            
        except Exception as e:
            logger.error(f"❌ 交易决策评估失败: {e}")
            return self._create_emergency_decision()
    
    async def _evaluate_rule(self, rule: Rule, context: Dict[str, Any]) -> RuleEvaluation:
        """评估单个规则"""
        try:
            start_time = time.time()
            
            # 准备评估上下文
            eval_context = {
                **context,
                'market_state': self.market_state,
                'account_state': self.account_state,
                'risk_parameters': self.risk_parameters,
                **rule.parameters
            }
            
            # 评估条件
            condition_result = self._evaluate_condition(rule.condition, eval_context)
            
            # 创建评估结果
            evaluation = RuleEvaluation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                is_triggered=condition_result,
                condition_result=condition_result,
                action=rule.action,
                risk_level=rule.risk_level,
                message=self._generate_rule_message(rule, condition_result, eval_context),
                parameters=rule.parameters.copy(),
                evaluation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc)
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ 规则评估失败 {rule.rule_id}: {e}")
            return RuleEvaluation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                is_triggered=False,
                condition_result=False,
                action=ActionType.ALLOW,
                risk_level=RiskLevel.LOW,
                message=f"规则评估错误: {str(e)}",
                parameters={},
                evaluation_time=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        try:
            # 安全的表达式评估
            allowed_names = {
                'position_size', 'total_exposure', 'daily_pnl', 'max_drawdown',
                'volatility', 'trend_strength', 'news_impact', 'correlation_risk',
                'win_rate', 'total_trades', 'market_state', 'account_state',
                'risk_parameters', 'near_support_resistance', 'system_error',
                'connection_lost', 'abs', 'max', 'min', 'and', 'or', 'not'
            }
            
            # 创建安全的评估环境
            safe_dict = {name: context.get(name, 0) for name in allowed_names}
            safe_dict.update({
                'abs': abs, 'max': max, 'min': min,
                'True': True, 'False': False
            })
            
            # 特殊条件处理
            if 'near_support_resistance' in condition:
                safe_dict['near_support_resistance'] = self._check_support_resistance(context)
            
            if 'system_error' in condition:
                safe_dict['system_error'] = context.get('system_error', False)
            
            if 'connection_lost' in condition:
                safe_dict['connection_lost'] = context.get('connection_lost', False)
            
            # 评估表达式
            result = eval(condition, {"__builtins__": {}}, safe_dict)
            return bool(result)
            
        except Exception as e:
            logger.error(f"❌ 条件评估失败: {condition}, 错误: {e}")
            return False
    
    def _check_support_resistance(self, context: Dict[str, Any]) -> bool:
        """检查是否接近支撑阻力位"""
        try:
            current_price = context.get('price', 0.0)
            support_level = context.get('support_level', 0.0)
            resistance_level = context.get('resistance_level', 0.0)
            threshold = context.get('distance_threshold', 0.02)
            
            if current_price <= 0:
                return False
            
            # 检查是否接近支撑位
            if support_level > 0:
                distance_to_support = abs(current_price - support_level) / current_price
                if distance_to_support <= threshold:
                    return True
            
            # 检查是否接近阻力位
            if resistance_level > 0:
                distance_to_resistance = abs(current_price - resistance_level) / current_price
                if distance_to_resistance <= threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 支撑阻力位检查失败: {e}")
            return False
    
    def _generate_rule_message(self, rule: Rule, triggered: bool, context: Dict[str, Any]) -> str:
        """生成规则消息"""
        try:
            if not triggered:
                return f"规则 {rule.name} 未触发"
            
            message = f"规则触发: {rule.name} - {rule.description}"
            
            # 添加具体数值
            if 'position_size' in rule.condition:
                position_size = context.get('position_size', 0.0)
                message += f" (当前仓位: {position_size:.2%})"
            
            if 'daily_pnl' in rule.condition:
                daily_pnl = context.get('daily_pnl', 0.0)
                message += f" (日损益: {daily_pnl:.2%})"
            
            if 'volatility' in rule.condition:
                volatility = context.get('volatility', 0.0)
                message += f" (波动率: {volatility:.2%})"
            
            return message
            
        except Exception as e:
            logger.error(f"❌ 规则消息生成失败: {e}")
            return f"规则 {rule.name} 已触发"
