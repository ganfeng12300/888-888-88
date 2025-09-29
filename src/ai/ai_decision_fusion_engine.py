#!/usr/bin/env python3
"""
AI决策融合引擎 - 生产级多AI模型决策融合系统
实现元学习、集成学习、强化学习等多AI智能融合决策
"""
import asyncio
import os
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import queue
import pickle

class AIDecision:
    """AI决策类"""
    
    def __init__(self, model_type: str, confidence: float, action: str, 
                 reasoning: str, metadata: Dict = None):
        self.model_type = model_type
        self.confidence = confidence
        self.action = action
        self.reasoning = reasoning
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.weight = 1.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'confidence': self.confidence,
            'action': self.action,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'weight': self.weight
        }

class ProductionAIDecisionFusionEngine:
    """生产级AI决策融合引擎"""
    
    def __init__(self):
        # AI模型配置
        self.ai_models = {
            'reinforcement_learning': {
                'weight': 0.25,
                'confidence_threshold': 0.7,
                'specialization': ['trend_following', 'momentum'],
                'performance_history': [],
                'current_accuracy': 0.0
            },
            'deep_learning': {
                'weight': 0.20,
                'confidence_threshold': 0.65,
                'specialization': ['pattern_recognition', 'time_series'],
                'performance_history': [],
                'current_accuracy': 0.0
            },
            'ensemble_learning': {
                'weight': 0.15,
                'confidence_threshold': 0.6,
                'specialization': ['feature_combination', 'voting'],
                'performance_history': [],
                'current_accuracy': 0.0
            },
            'expert_system': {
                'weight': 0.10,
                'confidence_threshold': 0.8,
                'specialization': ['rule_based', 'risk_management'],
                'performance_history': [],
                'current_accuracy': 0.0
            },
            'meta_learning': {
                'weight': 0.20,
                'confidence_threshold': 0.75,
                'specialization': ['adaptation', 'learning_to_learn'],
                'performance_history': [],
                'current_accuracy': 0.0
            },
            'transfer_learning': {
                'weight': 0.10,
                'confidence_threshold': 0.65,
                'specialization': ['cross_domain', 'knowledge_transfer'],
                'performance_history': [],
                'current_accuracy': 0.0
            }
        }
        
        # 融合策略
        self.fusion_strategies = {
            'weighted_voting': self._weighted_voting_fusion,
            'confidence_based': self._confidence_based_fusion,
            'performance_weighted': self._performance_weighted_fusion,
            'dynamic_ensemble': self._dynamic_ensemble_fusion,
            'meta_fusion': self._meta_fusion
        }
        
        # 决策历史
        self.decision_history = []
        self.fusion_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'accuracy': 0.0,
            'average_confidence': 0.0,
            'fusion_effectiveness': 0.0
        }
        
        # 实时决策队列
        self.decision_queue = queue.Queue()
        self.fusion_results = {}
        
        # 控制参数
        self.is_running = False
        self.fusion_thread = None
        self.min_decisions_for_fusion = 2
        self.decision_timeout = 5.0  # 5秒决策超时
        
        # 动态权重调整
        self.weight_adjustment_factor = 0.1
        self.performance_window = 100  # 性能评估窗口
        
        logger.info("🧠 AI决策融合引擎初始化完成")
    
    def start_fusion_engine(self):
        """启动决策融合引擎"""
        if self.is_running:
            logger.warning("AI决策融合引擎已在运行中")
            return
        
        self.is_running = True
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()
        
        logger.info("🚀 AI决策融合引擎启动")
    
    def _fusion_loop(self):
        """融合主循环"""
        while self.is_running:
            try:
                # 处理决策队列
                self._process_decision_queue()
                
                # 更新模型权重
                self._update_model_weights()
                
                # 清理历史数据
                self._cleanup_history()
                
                time.sleep(0.1)  # 100ms循环间隔
                
            except Exception as e:
                logger.error(f"融合循环错误: {e}")
                time.sleep(1)
    
    def submit_ai_decision(self, model_type: str, confidence: float, action: str, 
                          reasoning: str, metadata: Dict = None) -> str:
        """提交AI决策"""
        try:
            if model_type not in self.ai_models:
                logger.warning(f"未知的AI模型类型: {model_type}")
                return None
            
            decision = AIDecision(model_type, confidence, action, reasoning, metadata)
            decision_id = f"{model_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"
            
            # 添加到决策队列
            self.decision_queue.put((decision_id, decision))
            
            logger.debug(f"📝 AI决策提交 - {model_type}: {action} (置信度: {confidence:.2f})")
            return decision_id
            
        except Exception as e:
            logger.error(f"AI决策提交错误: {e}")
            return None
    
    def _process_decision_queue(self):
        """处理决策队列"""
        try:
            current_decisions = []
            
            # 收集当前时间窗口内的决策
            start_time = time.time()
            while not self.decision_queue.empty() and (time.time() - start_time) < self.decision_timeout:
                try:
                    decision_id, decision = self.decision_queue.get_nowait()
                    current_decisions.append((decision_id, decision))
                except queue.Empty:
                    break
            
            # 如果有足够的决策，进行融合
            if len(current_decisions) >= self.min_decisions_for_fusion:
                fusion_result = self._fuse_decisions(current_decisions)
                if fusion_result:
                    self._store_fusion_result(fusion_result)
                    
        except Exception as e:
            logger.error(f"决策队列处理错误: {e}")
    
    def _fuse_decisions(self, decisions: List[Tuple[str, AIDecision]]) -> Optional[Dict[str, Any]]:
        """融合多个AI决策"""
        try:
            decision_list = [decision for _, decision in decisions]
            
            # 使用多种融合策略
            fusion_results = {}
            for strategy_name, strategy_func in self.fusion_strategies.items():
                try:
                    result = strategy_func(decision_list)
                    fusion_results[strategy_name] = result
                except Exception as e:
                    logger.error(f"融合策略 {strategy_name} 错误: {e}")
            
            # 选择最佳融合结果
            final_result = self._select_best_fusion_result(fusion_results)
            
            if final_result:
                final_result['input_decisions'] = [d.to_dict() for d in decision_list]
                final_result['fusion_timestamp'] = time.time()
                final_result['decision_count'] = len(decision_list)
                
                logger.info(f"🔀 决策融合完成 - 最终行动: {final_result['action']} "
                          f"(置信度: {final_result['confidence']:.2f})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"决策融合错误: {e}")
            return None
    
    def _weighted_voting_fusion(self, decisions: List[AIDecision]) -> Dict[str, Any]:
        """加权投票融合"""
        try:
            action_votes = {}
            total_weight = 0
            
            for decision in decisions:
                model_config = self.ai_models[decision.model_type]
                weight = model_config['weight'] * decision.confidence
                
                if decision.action not in action_votes:
                    action_votes[decision.action] = {
                        'weight': 0,
                        'confidence_sum': 0,
                        'count': 0,
                        'decisions': []
                    }
                
                action_votes[decision.action]['weight'] += weight
                action_votes[decision.action]['confidence_sum'] += decision.confidence
                action_votes[decision.action]['count'] += 1
                action_votes[decision.action]['decisions'].append(decision)
                total_weight += weight
            
            # 选择权重最高的行动
            best_action = max(action_votes.keys(), key=lambda x: action_votes[x]['weight'])
            best_vote = action_votes[best_action]
            
            return {
                'strategy': 'weighted_voting',
                'action': best_action,
                'confidence': best_vote['weight'] / total_weight,
                'support_ratio': best_vote['count'] / len(decisions),
                'average_confidence': best_vote['confidence_sum'] / best_vote['count'],
                'reasoning': f"加权投票选择，{best_vote['count']}/{len(decisions)}个模型支持"
            }
            
        except Exception as e:
            logger.error(f"加权投票融合错误: {e}")
            return None
    
    def _confidence_based_fusion(self, decisions: List[AIDecision]) -> Dict[str, Any]:
        """基于置信度的融合"""
        try:
            # 按置信度排序
            sorted_decisions = sorted(decisions, key=lambda x: x.confidence, reverse=True)
            
            # 选择置信度最高的决策
            best_decision = sorted_decisions[0]
            
            # 计算支持度
            same_action_count = sum(1 for d in decisions if d.action == best_decision.action)
            support_ratio = same_action_count / len(decisions)
            
            return {
                'strategy': 'confidence_based',
                'action': best_decision.action,
                'confidence': best_decision.confidence * support_ratio,
                'support_ratio': support_ratio,
                'reasoning': f"选择最高置信度决策 ({best_decision.model_type}): {best_decision.reasoning}"
            }
            
        except Exception as e:
            logger.error(f"置信度融合错误: {e}")
            return None
    
    def _performance_weighted_fusion(self, decisions: List[AIDecision]) -> Dict[str, Any]:
        """基于性能的加权融合"""
        try:
            action_scores = {}
            
            for decision in decisions:
                model_config = self.ai_models[decision.model_type]
                performance_weight = model_config['current_accuracy']
                
                score = decision.confidence * performance_weight
                
                if decision.action not in action_scores:
                    action_scores[decision.action] = {
                        'score': 0,
                        'count': 0,
                        'decisions': []
                    }
                
                action_scores[decision.action]['score'] += score
                action_scores[decision.action]['count'] += 1
                action_scores[decision.action]['decisions'].append(decision)
            
            # 选择得分最高的行动
            best_action = max(action_scores.keys(), key=lambda x: action_scores[x]['score'])
            best_score = action_scores[best_action]
            
            return {
                'strategy': 'performance_weighted',
                'action': best_action,
                'confidence': best_score['score'] / best_score['count'],
                'support_ratio': best_score['count'] / len(decisions),
                'reasoning': f"基于性能加权选择，平均得分: {best_score['score']/best_score['count']:.3f}"
            }
            
        except Exception as e:
            logger.error(f"性能加权融合错误: {e}")
            return None
    
    def _dynamic_ensemble_fusion(self, decisions: List[AIDecision]) -> Dict[str, Any]:
        """动态集成融合"""
        try:
            # 根据市场条件和模型专长动态调整权重
            context_weights = self._calculate_context_weights(decisions)
            
            action_scores = {}
            total_weight = 0
            
            for decision in decisions:
                context_weight = context_weights.get(decision.model_type, 1.0)
                model_weight = self.ai_models[decision.model_type]['weight']
                
                final_weight = decision.confidence * model_weight * context_weight
                
                if decision.action not in action_scores:
                    action_scores[decision.action] = 0
                
                action_scores[decision.action] += final_weight
                total_weight += final_weight
            
            # 选择得分最高的行动
            best_action = max(action_scores.keys(), key=lambda x: action_scores[x])
            
            return {
                'strategy': 'dynamic_ensemble',
                'action': best_action,
                'confidence': action_scores[best_action] / total_weight,
                'reasoning': f"动态集成选择，考虑上下文权重调整"
            }
            
        except Exception as e:
            logger.error(f"动态集成融合错误: {e}")
            return None
    
    def _meta_fusion(self, decisions: List[AIDecision]) -> Dict[str, Any]:
        """元学习融合"""
        try:
            # 使用历史融合效果来指导当前融合
            historical_performance = self._analyze_historical_performance()
            
            # 基于历史表现调整策略选择
            strategy_weights = {
                'weighted_voting': historical_performance.get('weighted_voting', 0.25),
                'confidence_based': historical_performance.get('confidence_based', 0.25),
                'performance_weighted': historical_performance.get('performance_weighted', 0.25),
                'dynamic_ensemble': historical_performance.get('dynamic_ensemble', 0.25)
            }
            
            # 执行各种策略并加权组合
            strategy_results = {}
            for strategy_name in ['weighted_voting', 'confidence_based', 'performance_weighted', 'dynamic_ensemble']:
                if strategy_name in self.fusion_strategies:
                    result = self.fusion_strategies[strategy_name](decisions)
                    if result:
                        strategy_results[strategy_name] = result
            
            # 元融合：基于策略权重选择最佳结果
            best_strategy = max(strategy_weights.keys(), key=lambda x: strategy_weights[x])
            
            if best_strategy in strategy_results:
                result = strategy_results[best_strategy].copy()
                result['strategy'] = 'meta_fusion'
                result['meta_strategy'] = best_strategy
                result['reasoning'] = f"元学习选择策略: {best_strategy}"
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"元学习融合错误: {e}")
            return None
    
    def _calculate_context_weights(self, decisions: List[AIDecision]) -> Dict[str, float]:
        """计算上下文权重"""
        try:
            context_weights = {}
            
            # 基于决策一致性调整权重
            action_counts = {}
            for decision in decisions:
                if decision.action not in action_counts:
                    action_counts[decision.action] = 0
                action_counts[decision.action] += 1
            
            # 如果市场趋势明确，增强趋势跟踪模型权重
            if len(action_counts) == 1:  # 所有模型一致
                for decision in decisions:
                    if 'trend_following' in self.ai_models[decision.model_type]['specialization']:
                        context_weights[decision.model_type] = 1.2
                    else:
                        context_weights[decision.model_type] = 1.0
            else:  # 模型分歧，增强风险管理模型权重
                for decision in decisions:
                    if 'risk_management' in self.ai_models[decision.model_type]['specialization']:
                        context_weights[decision.model_type] = 1.3
                    else:
                        context_weights[decision.model_type] = 1.0
            
            return context_weights
            
        except Exception as e:
            logger.error(f"上下文权重计算错误: {e}")
            return {}
    
    def _select_best_fusion_result(self, fusion_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """选择最佳融合结果"""
        try:
            if not fusion_results:
                return None
            
            # 基于置信度和策略历史表现选择
            best_result = None
            best_score = -1
            
            for strategy_name, result in fusion_results.items():
                if result is None:
                    continue
                
                # 计算综合得分
                confidence_score = result.get('confidence', 0)
                support_score = result.get('support_ratio', 0.5)
                
                # 历史表现权重
                historical_performance = self._get_strategy_performance(strategy_name)
                
                total_score = (confidence_score * 0.4 + 
                             support_score * 0.3 + 
                             historical_performance * 0.3)
                
                if total_score > best_score:
                    best_score = total_score
                    best_result = result
            
            return best_result
            
        except Exception as e:
            logger.error(f"最佳融合结果选择错误: {e}")
            return None
    
    def _get_strategy_performance(self, strategy_name: str) -> float:
        """获取策略历史表现"""
        try:
            # 从融合历史中计算策略表现
            strategy_results = [r for r in self.fusion_history 
                              if r.get('strategy') == strategy_name]
            
            if not strategy_results:
                return 0.5  # 默认中等表现
            
            # 计算平均准确率
            total_accuracy = sum(r.get('accuracy', 0.5) for r in strategy_results[-50:])
            return total_accuracy / min(len(strategy_results), 50)
            
        except Exception as e:
            logger.error(f"策略表现获取错误: {e}")
            return 0.5
    
    def _analyze_historical_performance(self) -> Dict[str, float]:
        """分析历史表现"""
        try:
            performance = {}
            
            for strategy in self.fusion_strategies.keys():
                performance[strategy] = self._get_strategy_performance(strategy)
            
            # 归一化权重
            total = sum(performance.values())
            if total > 0:
                performance = {k: v/total for k, v in performance.items()}
            else:
                # 如果没有历史数据，使用均等权重
                equal_weight = 1.0 / len(performance)
                performance = {k: equal_weight for k in performance.keys()}
            
            return performance
            
        except Exception as e:
            logger.error(f"历史表现分析错误: {e}")
            return {}
    
    def _store_fusion_result(self, fusion_result: Dict[str, Any]):
        """存储融合结果"""
        try:
            self.fusion_results[fusion_result['fusion_timestamp']] = fusion_result
            self.fusion_history.append(fusion_result)
            
            # 更新性能指标
            self.performance_metrics['total_decisions'] += 1
            
            logger.debug(f"💾 融合结果已存储 - {fusion_result['action']}")
            
        except Exception as e:
            logger.error(f"融合结果存储错误: {e}")
    
    def _update_model_weights(self):
        """更新模型权重"""
        try:
            # 基于最近表现动态调整模型权重
            for model_type, config in self.ai_models.items():
                recent_performance = self._calculate_recent_performance(model_type)
                
                # 动态调整权重
                if recent_performance > 0.7:  # 表现良好
                    config['weight'] = min(0.4, config['weight'] * (1 + self.weight_adjustment_factor))
                elif recent_performance < 0.3:  # 表现不佳
                    config['weight'] = max(0.05, config['weight'] * (1 - self.weight_adjustment_factor))
                
                config['current_accuracy'] = recent_performance
            
            # 归一化权重
            total_weight = sum(config['weight'] for config in self.ai_models.values())
            if total_weight > 0:
                for config in self.ai_models.values():
                    config['weight'] /= total_weight
                    
        except Exception as e:
            logger.error(f"模型权重更新错误: {e}")
    
    def _calculate_recent_performance(self, model_type: str) -> float:
        """计算模型最近表现"""
        try:
            # 从决策历史中计算模型表现
            recent_decisions = [d for d in self.decision_history[-self.performance_window:] 
                              if d.get('model_type') == model_type]
            
            if not recent_decisions:
                return 0.5  # 默认中等表现
            
            # 计算准确率（这里需要实际的交易结果反馈）
            correct_count = sum(1 for d in recent_decisions if d.get('was_correct', False))
            return correct_count / len(recent_decisions)
            
        except Exception as e:
            logger.error(f"最近表现计算错误: {e}")
            return 0.5
    
    def _cleanup_history(self):
        """清理历史数据"""
        try:
            # 保持历史记录在合理范围内
            max_history = 1000
            
            if len(self.decision_history) > max_history:
                self.decision_history = self.decision_history[-max_history//2:]
            
            if len(self.fusion_history) > max_history:
                self.fusion_history = self.fusion_history[-max_history//2:]
            
            # 清理旧的融合结果
            current_time = time.time()
            old_results = [ts for ts in self.fusion_results.keys() 
                          if current_time - ts > 3600]  # 1小时前的结果
            
            for ts in old_results:
                del self.fusion_results[ts]
                
        except Exception as e:
            logger.error(f"历史数据清理错误: {e}")
    
    def get_latest_fusion_result(self) -> Optional[Dict[str, Any]]:
        """获取最新融合结果"""
        try:
            if not self.fusion_results:
                return None
            
            latest_timestamp = max(self.fusion_results.keys())
            return self.fusion_results[latest_timestamp]
            
        except Exception as e:
            logger.error(f"最新融合结果获取错误: {e}")
            return None
    
    def get_fusion_status(self) -> Dict[str, Any]:
        """获取融合引擎状态"""
        try:
            return {
                'is_running': self.is_running,
                'ai_models': {k: {
                    'weight': v['weight'],
                    'current_accuracy': v['current_accuracy'],
                    'specialization': v['specialization']
                } for k, v in self.ai_models.items()},
                'performance_metrics': self.performance_metrics.copy(),
                'decision_queue_size': self.decision_queue.qsize(),
                'fusion_results_count': len(self.fusion_results),
                'history_length': len(self.fusion_history)
            }
            
        except Exception as e:
            logger.error(f"融合状态获取错误: {e}")
            return {'error': str(e)}
    
    def update_decision_feedback(self, decision_timestamp: float, was_correct: bool):
        """更新决策反馈"""
        try:
            # 找到对应的融合结果并更新反馈
            if decision_timestamp in self.fusion_results:
                result = self.fusion_results[decision_timestamp]
                result['was_correct'] = was_correct
                
                # 更新性能指标
                if was_correct:
                    self.performance_metrics['correct_decisions'] += 1
                
                self.performance_metrics['accuracy'] = (
                    self.performance_metrics['correct_decisions'] / 
                    self.performance_metrics['total_decisions']
                )
                
                logger.debug(f"📊 决策反馈更新 - 正确: {was_correct}")
                
        except Exception as e:
            logger.error(f"决策反馈更新错误: {e}")
    
    def stop_fusion_engine(self):
        """停止融合引擎"""
        self.is_running = False
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=5)
        
        logger.info("🛑 AI决策融合引擎已停止")

# 全局AI决策融合引擎实例
_ai_decision_fusion_engine = None

def initialize_ai_decision_fusion_engine() -> ProductionAIDecisionFusionEngine:
    """初始化AI决策融合引擎"""
    global _ai_decision_fusion_engine
    
    if _ai_decision_fusion_engine is None:
        _ai_decision_fusion_engine = ProductionAIDecisionFusionEngine()
        _ai_decision_fusion_engine.start_fusion_engine()
        logger.success("✅ AI决策融合引擎初始化完成")
    
    return _ai_decision_fusion_engine

def get_ai_decision_fusion_engine() -> Optional[ProductionAIDecisionFusionEngine]:
    """获取AI决策融合引擎实例"""
    return _ai_decision_fusion_engine

if __name__ == "__main__":
    # 测试AI决策融合引擎
    engine = initialize_ai_decision_fusion_engine()
    
    # 模拟AI决策提交
    test_decisions = [
        ('reinforcement_learning', 0.85, 'BUY', '强化学习模型检测到上涨趋势'),
        ('deep_learning', 0.78, 'BUY', '深度学习识别出看涨模式'),
        ('ensemble_learning', 0.72, 'HOLD', '集成学习建议观望'),
        ('expert_system', 0.90, 'BUY', '专家系统规则触发买入信号'),
        ('meta_learning', 0.80, 'BUY', '元学习适应当前市场条件')
    ]
    
    for model_type, confidence, action, reasoning in test_decisions:
        engine.submit_ai_decision(model_type, confidence, action, reasoning)
    
    # 等待融合完成
    time.sleep(2)
    
    # 获取融合结果
    result = engine.get_latest_fusion_result()
    if result:
        print(f"融合结果: {result['action']} (置信度: {result['confidence']:.2f})")
        print(f"策略: {result['strategy']}")
        print(f"推理: {result['reasoning']}")
    
    # 获取状态
    status = engine.get_fusion_status()
    print(f"融合引擎状态: {status}")
    
    engine.stop_fusion_engine()
