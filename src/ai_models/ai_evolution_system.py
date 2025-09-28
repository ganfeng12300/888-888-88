"""
🧬 AI等级进化系统
生产级AI模型等级管理和进化机制，支持1-100级AI进化
实现动态升级降级、模型融合决策和实时学习优化
"""

import asyncio
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from src.hardware.gpu_manager import GPUTaskType, allocate_gpu_memory
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores


class AILevel(Enum):
    """AI等级分类"""
    BRONZE = "bronze"      # 1-20级
    SILVER = "silver"      # 21-40级
    GOLD = "gold"          # 41-60级
    PLATINUM = "platinum"  # 61-80级
    DIAMOND = "diamond"    # 81-95级
    EPIC = "epic"          # 96-100级


class ModelType(Enum):
    """AI模型类型"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES_DEEP = "time_series_deep"
    ENSEMBLE_LEARNING = "ensemble_learning"
    EXPERT_SYSTEM = "expert_system"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"


@dataclass
class AIModelStats:
    """AI模型统计数据"""
    model_type: ModelType
    level: int = 1
    experience: float = 0.0
    accuracy: float = 0.5
    win_rate: float = 0.5
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    training_hours: float = 0.0
    last_updated: float = field(default_factory=time.time)
    performance_history: List[float] = field(default_factory=list)


@dataclass
class EvolutionConfig:
    """进化配置"""
    max_level: int = 100
    experience_per_level: float = 1000.0
    level_multiplier: float = 1.2
    accuracy_threshold: float = 0.6
    performance_window: int = 100
    upgrade_threshold: float = 0.8
    downgrade_threshold: float = 0.4
    fusion_threshold: float = 0.9
    learning_rate_decay: float = 0.95


class AIEvolutionSystem:
    """AI进化系统核心"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.models: Dict[ModelType, AIModelStats] = {}
        self.fusion_weights: Dict[ModelType, float] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        
        # 初始化所有AI模型
        self._initialize_models()
        
        # 分配硬件资源
        self.gpu_memory = allocate_gpu_memory(GPUTaskType.AI_EVOLUTION, "evolution_system", 4096)
        assign_cpu_cores(CPUTaskType.AI_TRAINING_HEAVY, [13, 14, 15, 16])
        
        # 进化统计
        self.evolution_stats = {
            'total_evolutions': 0,
            'upgrades': 0,
            'downgrades': 0,
            'fusions': 0,
            'avg_level': 1.0,
            'best_performer': None
        }
        
        logger.info("AI进化系统初始化完成")
    
    def _initialize_models(self):
        """初始化所有AI模型"""
        for model_type in ModelType:
            self.models[model_type] = AIModelStats(
                model_type=model_type,
                level=1,
                experience=0.0,
                accuracy=np.random.uniform(0.45, 0.55),  # 初始随机准确率
                performance_history=[]
            )
            self.fusion_weights[model_type] = 1.0 / len(ModelType)
        
        logger.info(f"初始化了 {len(ModelType)} 个AI模型")
    
    def get_ai_level_category(self, level: int) -> AILevel:
        """获取AI等级分类"""
        if level <= 20:
            return AILevel.BRONZE
        elif level <= 40:
            return AILevel.SILVER
        elif level <= 60:
            return AILevel.GOLD
        elif level <= 80:
            return AILevel.PLATINUM
        elif level <= 95:
            return AILevel.DIAMOND
        else:
            return AILevel.EPIC
    
    def calculate_experience_needed(self, current_level: int) -> float:
        """计算升级所需经验"""
        return self.config.experience_per_level * (self.config.level_multiplier ** (current_level - 1))
    
    def update_model_performance(self, model_type: ModelType, 
                                performance_data: Dict[str, float]):
        """更新模型性能数据"""
        if model_type not in self.models:
            logger.error(f"未知模型类型: {model_type}")
            return
        
        model = self.models[model_type]
        
        # 更新基础统计
        model.accuracy = performance_data.get('accuracy', model.accuracy)
        model.win_rate = performance_data.get('win_rate', model.win_rate)
        model.sharpe_ratio = performance_data.get('sharpe_ratio', model.sharpe_ratio)
        model.max_drawdown = performance_data.get('max_drawdown', model.max_drawdown)
        model.total_trades += performance_data.get('new_trades', 0)
        model.profitable_trades += performance_data.get('profitable_trades', 0)
        model.training_hours += performance_data.get('training_time', 0)
        
        # 计算综合性能分数
        performance_score = self._calculate_performance_score(model)
        model.performance_history.append(performance_score)
        
        # 限制历史记录长度
        if len(model.performance_history) > self.config.performance_window:
            model.performance_history.pop(0)
        
        # 计算经验增长
        experience_gain = self._calculate_experience_gain(performance_score)
        model.experience += experience_gain
        
        model.last_updated = time.time()
        
        # 检查是否需要进化
        self._check_evolution(model_type)
        
        logger.info(f"{model_type.value} 性能更新: 准确率={model.accuracy:.3f}, "
                   f"等级={model.level}, 经验={model.experience:.1f}")
    
    def _calculate_performance_score(self, model: AIModelStats) -> float:
        """计算综合性能分数"""
        # 基础分数 (0-1)
        base_score = model.accuracy
        
        # 胜率加成
        win_rate_bonus = (model.win_rate - 0.5) * 0.2
        
        # 夏普比率加成
        sharpe_bonus = min(model.sharpe_ratio / 2.0, 0.3) if model.sharpe_ratio > 0 else 0
        
        # 回撤惩罚
        drawdown_penalty = abs(model.max_drawdown) * 0.1
        
        # 交易频率调整
        trade_frequency = model.total_trades / max(model.training_hours, 1)
        frequency_bonus = min(trade_frequency / 10.0, 0.1)
        
        total_score = base_score + win_rate_bonus + sharpe_bonus - drawdown_penalty + frequency_bonus
        return max(0.0, min(1.0, total_score))
    
    def _calculate_experience_gain(self, performance_score: float) -> float:
        """计算经验增长"""
        base_exp = 10.0
        performance_multiplier = performance_score * 2.0
        return base_exp * performance_multiplier
    
    def _check_evolution(self, model_type: ModelType):
        """检查模型是否需要进化"""
        model = self.models[model_type]
        
        # 检查升级
        experience_needed = self.calculate_experience_needed(model.level)
        if model.experience >= experience_needed and model.level < self.config.max_level:
            self._upgrade_model(model_type)
        
        # 检查降级 (性能持续低下)
        if len(model.performance_history) >= 20:
            recent_avg = np.mean(model.performance_history[-20:])
            if recent_avg < self.config.downgrade_threshold and model.level > 1:
                self._downgrade_model(model_type)
    
    def _upgrade_model(self, model_type: ModelType):
        """升级模型"""
        model = self.models[model_type]
        old_level = model.level
        
        # 升级
        model.level += 1
        model.experience = 0.0  # 重置经验
        
        # 记录进化历史
        evolution_record = {
            'timestamp': time.time(),
            'model_type': model_type.value,
            'action': 'upgrade',
            'old_level': old_level,
            'new_level': model.level,
            'performance': model.performance_history[-1] if model.performance_history else 0.0
        }
        self.evolution_history.append(evolution_record)
        
        # 更新统计
        self.evolution_stats['total_evolutions'] += 1
        self.evolution_stats['upgrades'] += 1
        self._update_fusion_weights()
        
        level_category = self.get_ai_level_category(model.level)
        logger.info(f"🎉 {model_type.value} 升级! {old_level} → {model.level} "
                   f"({level_category.value.upper()})")
    
    def _downgrade_model(self, model_type: ModelType):
        """降级模型"""
        model = self.models[model_type]
        old_level = model.level
        
        # 降级
        model.level = max(1, model.level - 1)
        model.experience = 0.0  # 重置经验
        
        # 记录进化历史
        evolution_record = {
            'timestamp': time.time(),
            'model_type': model_type.value,
            'action': 'downgrade',
            'old_level': old_level,
            'new_level': model.level,
            'performance': np.mean(model.performance_history[-20:])
        }
        self.evolution_history.append(evolution_record)
        
        # 更新统计
        self.evolution_stats['total_evolutions'] += 1
        self.evolution_stats['downgrades'] += 1
        self._update_fusion_weights()
        
        logger.warning(f"⬇️ {model_type.value} 降级! {old_level} → {model.level} "
                      f"(性能不佳)")
    
    def _update_fusion_weights(self):
        """更新模型融合权重"""
        total_score = 0.0
        model_scores = {}
        
        # 计算每个模型的综合分数
        for model_type, model in self.models.items():
            # 等级权重
            level_weight = model.level / self.config.max_level
            
            # 性能权重
            performance_weight = model.accuracy if model.accuracy > 0 else 0.1
            
            # 最近表现权重
            recent_performance = 0.5
            if model.performance_history:
                recent_performance = np.mean(model.performance_history[-10:])
            
            # 综合分数
            combined_score = (level_weight * 0.4 + 
                            performance_weight * 0.4 + 
                            recent_performance * 0.2)
            
            model_scores[model_type] = combined_score
            total_score += combined_score
        
        # 归一化权重
        if total_score > 0:
            for model_type in ModelType:
                self.fusion_weights[model_type] = model_scores[model_type] / total_score
        
        # 更新平均等级
        avg_level = np.mean([model.level for model in self.models.values()])
        self.evolution_stats['avg_level'] = avg_level
        
        # 找出最佳表现者
        best_model = max(self.models.items(), 
                        key=lambda x: x[1].accuracy * x[1].level)
        self.evolution_stats['best_performer'] = best_model[0].value
    
    def get_fusion_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取融合决策"""
        decisions = {}
        confidence_scores = {}
        
        # 收集各模型的决策 (模拟)
        for model_type, weight in self.fusion_weights.items():
            model = self.models[model_type]
            
            # 基于模型等级和性能生成决策
            base_confidence = model.accuracy * (model.level / 100.0)
            
            # 模拟决策 (实际应该调用具体模型)
            decision_strength = np.random.uniform(-1, 1)  # -1(强卖) 到 1(强买)
            adjusted_strength = decision_strength * base_confidence
            
            decisions[model_type.value] = {
                'action': 'buy' if adjusted_strength > 0.1 else 'sell' if adjusted_strength < -0.1 else 'hold',
                'strength': abs(adjusted_strength),
                'confidence': base_confidence,
                'weight': weight
            }
            confidence_scores[model_type.value] = base_confidence * weight
        
        # 计算融合决策
        weighted_strength = sum(
            decisions[model]['strength'] * decisions[model]['weight'] * 
            (1 if decisions[model]['action'] == 'buy' else -1 if decisions[model]['action'] == 'sell' else 0)
            for model in decisions
        )
        
        total_confidence = sum(confidence_scores.values())
        
        # 最终决策
        final_action = 'hold'
        if weighted_strength > 0.2:
            final_action = 'buy'
        elif weighted_strength < -0.2:
            final_action = 'sell'
        
        return {
            'final_action': final_action,
            'strength': abs(weighted_strength),
            'confidence': total_confidence,
            'individual_decisions': decisions,
            'fusion_weights': self.fusion_weights.copy(),
            'timestamp': time.time()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'models': {},
            'evolution_stats': self.evolution_stats.copy(),
            'fusion_weights': self.fusion_weights.copy(),
            'system_health': 'healthy'
        }
        
        # 模型状态
        for model_type, model in self.models.items():
            level_category = self.get_ai_level_category(model.level)
            status['models'][model_type.value] = {
                'level': model.level,
                'category': level_category.value,
                'experience': model.experience,
                'accuracy': model.accuracy,
                'win_rate': model.win_rate,
                'sharpe_ratio': model.sharpe_ratio,
                'total_trades': model.total_trades,
                'training_hours': model.training_hours,
                'last_updated': model.last_updated
            }
        
        # 系统健康检查
        avg_accuracy = np.mean([model.accuracy for model in self.models.values()])
        if avg_accuracy < 0.4:
            status['system_health'] = 'poor'
        elif avg_accuracy < 0.6:
            status['system_health'] = 'fair'
        elif avg_accuracy < 0.8:
            status['system_health'] = 'good'
        else:
            status['system_health'] = 'excellent'
        
        return status
    
    def save_evolution_state(self, filepath: str):
        """保存进化状态"""
        state = {
            'models': {k.value: v for k, v in self.models.items()},
            'fusion_weights': {k.value: v for k, v in self.fusion_weights.items()},
            'evolution_history': self.evolution_history,
            'evolution_stats': self.evolution_stats,
            'timestamp': time.time()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"进化状态已保存到: {filepath}")
    
    def load_evolution_state(self, filepath: str):
        """加载进化状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复模型状态
            for model_type_str, model_data in state['models'].items():
                model_type = ModelType(model_type_str)
                self.models[model_type] = model_data
            
            # 恢复融合权重
            for model_type_str, weight in state['fusion_weights'].items():
                model_type = ModelType(model_type_str)
                self.fusion_weights[model_type] = weight
            
            self.evolution_history = state['evolution_history']
            self.evolution_stats = state['evolution_stats']
            
            logger.info(f"进化状态已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"加载进化状态失败: {e}")
    
    async def run_evolution_loop(self, interval: float = 60.0):
        """运行进化循环"""
        logger.info("启动AI进化循环")
        
        while True:
            try:
                # 模拟性能数据更新
                for model_type in ModelType:
                    # 模拟性能变化
                    current_accuracy = self.models[model_type].accuracy
                    accuracy_change = np.random.normal(0, 0.01)  # 小幅随机变化
                    new_accuracy = np.clip(current_accuracy + accuracy_change, 0.1, 0.99)
                    
                    performance_data = {
                        'accuracy': new_accuracy,
                        'win_rate': np.random.uniform(0.4, 0.7),
                        'sharpe_ratio': np.random.uniform(-0.5, 2.0),
                        'max_drawdown': np.random.uniform(-0.1, -0.01),
                        'new_trades': np.random.randint(1, 10),
                        'profitable_trades': np.random.randint(0, 5),
                        'training_time': 0.1
                    }
                    
                    self.update_model_performance(model_type, performance_data)
                
                # 定期保存状态
                if int(time.time()) % 300 == 0:  # 每5分钟保存一次
                    self.save_evolution_state("models/evolution_state.pkl")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"进化循环出错: {e}")
                await asyncio.sleep(interval)


# 全局进化系统实例
evolution_system = None


def create_evolution_system(config: EvolutionConfig = None) -> AIEvolutionSystem:
    """创建AI进化系统"""
    global evolution_system
    
    if config is None:
        config = EvolutionConfig()
    
    evolution_system = AIEvolutionSystem(config)
    return evolution_system


async def main():
    """测试主函数"""
    logger.info("启动AI进化系统测试...")
    
    # 创建进化系统
    config = EvolutionConfig()
    system = create_evolution_system(config)
    
    try:
        # 启动进化循环
        evolution_task = asyncio.create_task(system.run_evolution_loop(5.0))
        
        # 运行测试
        for i in range(20):
            # 获取融合决策
            market_data = {'price': 50000 + i * 100}
            decision = system.get_fusion_decision(market_data)
            
            # 获取系统状态
            status = system.get_system_status()
            
            logger.info(f"轮次 {i+1}: 决策={decision['final_action']}, "
                       f"平均等级={status['evolution_stats']['avg_level']:.1f}, "
                       f"系统健康={status['system_health']}")
            
            await asyncio.sleep(1)
        
        # 停止进化循环
        evolution_task.cancel()
        
        # 显示最终状态
        final_status = system.get_system_status()
        logger.info(f"最终状态: {final_status}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    except Exception as e:
        logger.error(f"测试出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
