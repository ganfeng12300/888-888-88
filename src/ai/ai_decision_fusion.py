"""
🧠 AI决策融合引擎 - 多模型智能决策融合系统
基于AI等级进化的多模型决策融合，支持动态权重调整和实时学习优化
"""

import asyncio
import json
import math
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from .ai_level_evolution import AIEvolutionEngine, AIPerformanceData, AILevelInfo


class DecisionType(Enum):
    """决策类型"""
    BUY = "buy"                 # 买入
    SELL = "sell"               # 卖出
    HOLD = "hold"               # 持有
    CLOSE = "close"             # 平仓


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = "very_low"       # 极低 (0-20%)
    LOW = "low"                 # 低 (20-40%)
    MEDIUM = "medium"           # 中等 (40-60%)
    HIGH = "high"               # 高 (60-80%)
    VERY_HIGH = "very_high"     # 极高 (80-100%)


@dataclass
class AIDecision:
    """AI决策"""
    model_id: str                       # 模型ID
    decision_type: DecisionType         # 决策类型
    confidence: float                   # 置信度 (0-1)
    confidence_level: ConfidenceLevel   # 置信度等级
    price_target: Optional[float]       # 目标价格
    stop_loss: Optional[float]          # 止损价格
    take_profit: Optional[float]        # 止盈价格
    position_size: float                # 仓位大小
    reasoning: str                      # 决策理由
    features: Dict[str, float]          # 特征数据
    timestamp: float                    # 时间戳
    model_level: int                    # 模型等级
    model_tier: str                     # 模型段位


@dataclass
class FusedDecision:
    """融合决策"""
    final_decision: DecisionType        # 最终决策
    final_confidence: float             # 最终置信度
    final_position_size: float          # 最终仓位大小
    final_price_target: Optional[float] # 最终目标价格
    final_stop_loss: Optional[float]    # 最终止损价格
    final_take_profit: Optional[float]  # 最终止盈价格
    contributing_models: List[str]      # 参与决策的模型
    model_weights: Dict[str, float]     # 模型权重
    individual_decisions: List[AIDecision] # 个体决策
    fusion_reasoning: str               # 融合理由
    timestamp: float                    # 时间戳
    consensus_score: float              # 共识得分


class ModelWeightCalculator:
    """模型权重计算器"""
    
    def __init__(self):
        # 权重计算配置
        self.level_weight_factor = 0.4      # 等级权重因子
        self.performance_weight_factor = 0.3 # 性能权重因子
        self.confidence_weight_factor = 0.2  # 置信度权重因子
        self.recency_weight_factor = 0.1     # 时效性权重因子
        
        # 等级权重映射
        self.tier_weights = {
            "novice": 0.1,      # 新手级
            "bronze": 0.2,      # 青铜级
            "silver": 0.4,      # 白银级
            "gold": 0.6,        # 黄金级
            "platinum": 0.8,    # 铂金级
            "diamond": 0.9,     # 钻石级
            "legendary": 1.0    # 传奇级
        }
        
        logger.info("模型权重计算器初始化完成")
    
    def calculate_model_weight(self, 
                             model_id: str,
                             level_info: AILevelInfo,
                             decision: AIDecision,
                             recent_performance: Optional[AIPerformanceData] = None) -> float:
        """计算模型权重"""
        try:
            # 1. 等级权重 (基于AI等级)
            level_weight = (level_info.level / 100.0) * self.level_weight_factor
            
            # 2. 性能权重 (基于最近性能)
            if recent_performance:
                performance_weight = (recent_performance.accuracy * 0.5 + 
                                    max(0, recent_performance.profit_rate) * 0.3 +
                                    max(0, recent_performance.sharpe_ratio / 3.0) * 0.2) * self.performance_weight_factor
            else:
                performance_weight = level_info.performance_score / 100.0 * self.performance_weight_factor
            
            # 3. 置信度权重 (基于决策置信度)
            confidence_weight = decision.confidence * self.confidence_weight_factor
            
            # 4. 时效性权重 (基于最近活跃度)
            current_time = time.time()
            time_since_last_upgrade = current_time - level_info.last_upgrade_time
            recency_factor = max(0.1, 1.0 - (time_since_last_upgrade / 86400))  # 24小时衰减
            recency_weight = recency_factor * self.recency_weight_factor
            
            # 计算总权重
            total_weight = level_weight + performance_weight + confidence_weight + recency_weight
            
            # 应用段位加成
            tier_bonus = self.tier_weights.get(level_info.tier.value, 0.1)
            final_weight = total_weight * (1 + tier_bonus)
            
            return max(0.01, min(final_weight, 2.0))  # 限制在0.01-2.0范围内
            
        except Exception as e:
            logger.error(f"计算模型权重失败: {e}")
            return 0.1  # 默认权重


class DecisionFusionEngine:
    """决策融合引擎"""
    
    def __init__(self, evolution_engine: AIEvolutionEngine):
        self.evolution_engine = evolution_engine
        self.weight_calculator = ModelWeightCalculator()
        self.fusion_history: List[FusedDecision] = []
        self.fusion_lock = threading.RLock()
        
        # 融合配置
        self.min_models_for_fusion = 2     # 最少融合模型数
        self.max_fusion_history = 1000     # 最大融合历史记录
        self.consensus_threshold = 0.6     # 共识阈值
        
        logger.info("决策融合引擎初始化完成")
    
    def fuse_decisions(self, decisions: List[AIDecision]) -> Optional[FusedDecision]:
        """融合多个AI决策"""
        try:
            with self.fusion_lock:
                if len(decisions) < self.min_models_for_fusion:
                    logger.warning(f"决策数量不足，需要至少{self.min_models_for_fusion}个决策")
                    return None
                
                # 过滤有效决策
                valid_decisions = [d for d in decisions if d.confidence > 0.1]
                if not valid_decisions:
                    logger.warning("没有有效的决策")
                    return None
                
                # 计算模型权重
                model_weights = {}
                for decision in valid_decisions:
                    level_info = self.evolution_engine.get_ai_level_info(decision.model_id)
                    if level_info:
                        weight = self.weight_calculator.calculate_model_weight(
                            decision.model_id, level_info, decision
                        )
                        model_weights[decision.model_id] = weight
                    else:
                        model_weights[decision.model_id] = 0.1  # 默认权重
                
                # 标准化权重
                total_weight = sum(model_weights.values())
                if total_weight > 0:
                    model_weights = {k: v/total_weight for k, v in model_weights.items()}
                
                # 执行决策融合
                fused_decision = self._perform_fusion(valid_decisions, model_weights)
                
                # 记录融合历史
                self.fusion_history.append(fused_decision)
                if len(self.fusion_history) > self.max_fusion_history:
                    self.fusion_history.pop(0)
                
                return fused_decision
                
        except Exception as e:
            logger.error(f"决策融合失败: {e}")
            return None
    
    def _perform_fusion(self, decisions: List[AIDecision], weights: Dict[str, float]) -> FusedDecision:
        """执行决策融合"""
        try:
            # 1. 决策类型融合 (加权投票)
            decision_votes = {}
            for decision in decisions:
                weight = weights.get(decision.model_id, 0.1)
                if decision.decision_type not in decision_votes:
                    decision_votes[decision.decision_type] = 0
                decision_votes[decision.decision_type] += weight * decision.confidence
            
            # 选择得票最高的决策类型
            final_decision = max(decision_votes.items(), key=lambda x: x[1])[0]
            
            # 2. 置信度融合 (加权平均)
            final_confidence = 0.0
            for decision in decisions:
                weight = weights.get(decision.model_id, 0.1)
                final_confidence += weight * decision.confidence
            
            # 3. 仓位大小融合 (加权平均)
            final_position_size = 0.0
            for decision in decisions:
                weight = weights.get(decision.model_id, 0.1)
                final_position_size += weight * decision.position_size
            
            # 4. 价格目标融合 (加权平均，排除None值)
            price_targets = [d.price_target for d in decisions if d.price_target is not None]
            if price_targets:
                weighted_price_sum = 0.0
                weight_sum = 0.0
                for decision in decisions:
                    if decision.price_target is not None:
                        weight = weights.get(decision.model_id, 0.1)
                        weighted_price_sum += weight * decision.price_target
                        weight_sum += weight
                final_price_target = weighted_price_sum / weight_sum if weight_sum > 0 else None
            else:
                final_price_target = None
            
            # 5. 止损价格融合
            stop_losses = [d.stop_loss for d in decisions if d.stop_loss is not None]
            if stop_losses:
                weighted_sl_sum = 0.0
                weight_sum = 0.0
                for decision in decisions:
                    if decision.stop_loss is not None:
                        weight = weights.get(decision.model_id, 0.1)
                        weighted_sl_sum += weight * decision.stop_loss
                        weight_sum += weight
                final_stop_loss = weighted_sl_sum / weight_sum if weight_sum > 0 else None
            else:
                final_stop_loss = None
            
            # 6. 止盈价格融合
            take_profits = [d.take_profit for d in decisions if d.take_profit is not None]
            if take_profits:
                weighted_tp_sum = 0.0
                weight_sum = 0.0
                for decision in decisions:
                    if decision.take_profit is not None:
                        weight = weights.get(decision.model_id, 0.1)
                        weighted_tp_sum += weight * decision.take_profit
                        weight_sum += weight
                final_take_profit = weighted_tp_sum / weight_sum if weight_sum > 0 else None
            else:
                final_take_profit = None
            
            # 7. 计算共识得分
            consensus_score = self._calculate_consensus_score(decisions, final_decision)
            
            # 8. 生成融合理由
            fusion_reasoning = self._generate_fusion_reasoning(
                decisions, weights, final_decision, consensus_score
            )
            
            # 创建融合决策
            fused_decision = FusedDecision(
                final_decision=final_decision,
                final_confidence=final_confidence,
                final_position_size=final_position_size,
                final_price_target=final_price_target,
                final_stop_loss=final_stop_loss,
                final_take_profit=final_take_profit,
                contributing_models=[d.model_id for d in decisions],
                model_weights=weights,
                individual_decisions=decisions,
                fusion_reasoning=fusion_reasoning,
                timestamp=time.time(),
                consensus_score=consensus_score
            )
            
            logger.info(f"决策融合完成: {final_decision.value} (置信度: {final_confidence:.2f}, 共识: {consensus_score:.2f})")
            
            return fused_decision
            
        except Exception as e:
            logger.error(f"执行决策融合失败: {e}")
            raise
    
    def _calculate_consensus_score(self, decisions: List[AIDecision], final_decision: DecisionType) -> float:
        """计算共识得分"""
        try:
            if not decisions:
                return 0.0
            
            # 计算支持最终决策的模型比例
            supporting_count = sum(1 for d in decisions if d.decision_type == final_decision)
            consensus_ratio = supporting_count / len(decisions)
            
            # 计算置信度加权共识
            total_confidence = sum(d.confidence for d in decisions)
            supporting_confidence = sum(d.confidence for d in decisions if d.decision_type == final_decision)
            confidence_consensus = supporting_confidence / total_confidence if total_confidence > 0 else 0
            
            # 综合共识得分
            consensus_score = (consensus_ratio * 0.6 + confidence_consensus * 0.4)
            
            return max(0.0, min(consensus_score, 1.0))
            
        except Exception as e:
            logger.error(f"计算共识得分失败: {e}")
            return 0.0
    
    def _generate_fusion_reasoning(self, 
                                 decisions: List[AIDecision], 
                                 weights: Dict[str, float],
                                 final_decision: DecisionType,
                                 consensus_score: float) -> str:
        """生成融合理由"""
        try:
            reasoning_parts = []
            
            # 基本信息
            reasoning_parts.append(f"融合了{len(decisions)}个AI模型的决策")
            reasoning_parts.append(f"最终决策: {final_decision.value}")
            reasoning_parts.append(f"共识得分: {consensus_score:.2f}")
            
            # 模型权重信息
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            top_models = sorted_weights[:3]  # 前3个权重最高的模型
            
            reasoning_parts.append("主要贡献模型:")
            for model_id, weight in top_models:
                model_decision = next((d for d in decisions if d.model_id == model_id), None)
                if model_decision:
                    level_info = self.evolution_engine.get_ai_level_info(model_id)
                    level_str = f"Lv.{level_info.level}" if level_info else "未知等级"
                    reasoning_parts.append(
                        f"- {model_id} ({level_str}): {model_decision.decision_type.value} "
                        f"(权重: {weight:.2f}, 置信度: {model_decision.confidence:.2f})"
                    )
            
            # 共识分析
            if consensus_score >= 0.8:
                reasoning_parts.append("高度共识决策，模型意见高度一致")
            elif consensus_score >= 0.6:
                reasoning_parts.append("中等共识决策，大部分模型意见一致")
            else:
                reasoning_parts.append("低共识决策，模型意见分歧较大，需谨慎执行")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"生成融合理由失败: {e}")
            return "融合理由生成失败"
    
    def get_fusion_history(self, limit: int = 50) -> List[FusedDecision]:
        """获取融合历史"""
        return self.fusion_history[-limit:] if limit > 0 else self.fusion_history
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        try:
            if not self.fusion_history:
                return {}
            
            stats = {
                "total_fusions": len(self.fusion_history),
                "decision_distribution": {},
                "avg_confidence": 0.0,
                "avg_consensus": 0.0,
                "high_consensus_ratio": 0.0,
                "recent_performance": {}
            }
            
            # 决策类型分布
            for decision_type in DecisionType:
                stats["decision_distribution"][decision_type.value] = 0
            
            total_confidence = 0.0
            total_consensus = 0.0
            high_consensus_count = 0
            
            for fusion in self.fusion_history:
                # 决策分布
                stats["decision_distribution"][fusion.final_decision.value] += 1
                
                # 置信度和共识统计
                total_confidence += fusion.final_confidence
                total_consensus += fusion.consensus_score
                
                if fusion.consensus_score >= 0.8:
                    high_consensus_count += 1
            
            # 计算平均值
            stats["avg_confidence"] = total_confidence / len(self.fusion_history)
            stats["avg_consensus"] = total_consensus / len(self.fusion_history)
            stats["high_consensus_ratio"] = high_consensus_count / len(self.fusion_history)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取融合统计失败: {e}")
            return {}
    
    def evaluate_fusion_quality(self, fusion: FusedDecision, actual_outcome: float) -> Dict[str, float]:
        """评估融合质量"""
        try:
            evaluation = {
                "accuracy": 0.0,
                "confidence_calibration": 0.0,
                "consensus_effectiveness": 0.0,
                "overall_quality": 0.0
            }
            
            # 1. 准确性评估
            if fusion.final_decision in [DecisionType.BUY, DecisionType.SELL]:
                expected_direction = 1 if fusion.final_decision == DecisionType.BUY else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                evaluation["accuracy"] = 1.0 if expected_direction == actual_direction else 0.0
            
            # 2. 置信度校准
            # 高置信度应该对应高准确性
            confidence_accuracy_diff = abs(fusion.final_confidence - evaluation["accuracy"])
            evaluation["confidence_calibration"] = max(0.0, 1.0 - confidence_accuracy_diff)
            
            # 3. 共识有效性
            # 高共识应该对应更好的结果
            if evaluation["accuracy"] > 0.5:
                evaluation["consensus_effectiveness"] = fusion.consensus_score
            else:
                evaluation["consensus_effectiveness"] = 1.0 - fusion.consensus_score
            
            # 4. 综合质量
            evaluation["overall_quality"] = (
                evaluation["accuracy"] * 0.4 +
                evaluation["confidence_calibration"] * 0.3 +
                evaluation["consensus_effectiveness"] * 0.3
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"评估融合质量失败: {e}")
            return {"accuracy": 0.0, "confidence_calibration": 0.0, 
                   "consensus_effectiveness": 0.0, "overall_quality": 0.0}


# 全局决策融合引擎实例
decision_fusion_engine = None

def initialize_decision_fusion(evolution_engine: AIEvolutionEngine):
    """初始化决策融合引擎"""
    global decision_fusion_engine
    decision_fusion_engine = DecisionFusionEngine(evolution_engine)
    return decision_fusion_engine
