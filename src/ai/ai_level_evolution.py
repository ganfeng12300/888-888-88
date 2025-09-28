"""
🧠 AI等级进化系统 - 生产级实盘交易AI智能进化引擎
1-100级AI等级系统，基于真实交易表现动态升级降级
支持多模型融合决策、实时学习优化、进化历史追踪
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


class AILevel(Enum):
    """AI等级枚举"""
    NOVICE = "novice"           # 新手级 (1-20)
    BRONZE = "bronze"           # 青铜级 (21-40)
    SILVER = "silver"           # 白银级 (41-60)
    GOLD = "gold"               # 黄金级 (61-80)
    PLATINUM = "platinum"       # 铂金级 (81-95)
    DIAMOND = "diamond"         # 钻石级 (96-99)
    LEGENDARY = "legendary"     # 传奇级 (100)


class PerformanceMetric(Enum):
    """性能指标类型"""
    ACCURACY = "accuracy"               # 预测准确率
    PROFIT_RATE = "profit_rate"         # 盈利率
    SHARPE_RATIO = "sharpe_ratio"       # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"       # 最大回撤
    WIN_RATE = "win_rate"               # 胜率
    PROFIT_FACTOR = "profit_factor"     # 盈利因子
    CALMAR_RATIO = "calmar_ratio"       # 卡玛比率
    SORTINO_RATIO = "sortino_ratio"     # 索提诺比率


@dataclass
class AIPerformanceData:
    """AI性能数据"""
    timestamp: float                    # 时间戳
    accuracy: float                     # 预测准确率
    profit_rate: float                  # 盈利率
    sharpe_ratio: float                 # 夏普比率
    max_drawdown: float                 # 最大回撤
    win_rate: float                     # 胜率
    profit_factor: float                # 盈利因子
    calmar_ratio: float                 # 卡玛比率
    sortino_ratio: float                # 索提诺比率
    trade_count: int                    # 交易次数
    total_profit: float                 # 总盈利
    total_loss: float                   # 总亏损
    avg_trade_duration: float           # 平均交易持续时间
    volatility: float                   # 波动率


@dataclass
class AILevelInfo:
    """AI等级信息"""
    level: int                          # 等级 (1-100)
    tier: AILevel                       # 等级段位
    experience: float                   # 经验值
    next_level_exp: float               # 升级所需经验
    performance_score: float            # 综合性能得分
    upgrade_threshold: float            # 升级阈值
    downgrade_threshold: float          # 降级阈值
    consecutive_wins: int               # 连续胜利次数
    consecutive_losses: int             # 连续失败次数
    last_upgrade_time: float            # 上次升级时间
    last_downgrade_time: float          # 上次降级时间
    total_upgrades: int                 # 总升级次数
    total_downgrades: int               # 总降级次数


@dataclass
class EvolutionEvent:
    """进化事件"""
    timestamp: float                    # 时间戳
    event_type: str                     # 事件类型 (upgrade/downgrade/milestone)
    old_level: int                      # 原等级
    new_level: int                      # 新等级
    old_tier: AILevel                   # 原段位
    new_tier: AILevel                   # 新段位
    trigger_reason: str                 # 触发原因
    performance_data: AIPerformanceData # 性能数据
    description: str                    # 事件描述


class AILevelCalculator:
    """AI等级计算器"""
    
    def __init__(self):
        # 等级配置
        self.max_level = 100
        self.base_exp = 1000.0
        self.exp_multiplier = 1.2
        
        # 性能权重配置
        self.performance_weights = {
            PerformanceMetric.ACCURACY: 0.20,
            PerformanceMetric.PROFIT_RATE: 0.25,
            PerformanceMetric.SHARPE_RATIO: 0.15,
            PerformanceMetric.MAX_DRAWDOWN: 0.10,
            PerformanceMetric.WIN_RATE: 0.10,
            PerformanceMetric.PROFIT_FACTOR: 0.10,
            PerformanceMetric.CALMAR_RATIO: 0.05,
            PerformanceMetric.SORTINO_RATIO: 0.05
        }
        
        # 等级段位配置
        self.tier_ranges = {
            AILevel.NOVICE: (1, 20),
            AILevel.BRONZE: (21, 40),
            AILevel.SILVER: (41, 60),
            AILevel.GOLD: (61, 80),
            AILevel.PLATINUM: (81, 95),
            AILevel.DIAMOND: (96, 99),
            AILevel.LEGENDARY: (100, 100)
        }
        
        logger.info("AI等级计算器初始化完成")
    
    def calculate_required_exp(self, level: int) -> float:
        """计算升级所需经验值"""
        if level >= self.max_level:
            return float('inf')
        
        return self.base_exp * (self.exp_multiplier ** (level - 1))
    
    def calculate_performance_score(self, performance: AIPerformanceData) -> float:
        """计算综合性能得分"""
        try:
            # 标准化各项指标到0-100分
            scores = {}
            
            # 准确率得分 (0-100%)
            scores[PerformanceMetric.ACCURACY] = min(performance.accuracy * 100, 100)
            
            # 盈利率得分 (转换为0-100分)
            profit_score = 50 + (performance.profit_rate * 100)  # 0%盈利率=50分
            scores[PerformanceMetric.PROFIT_RATE] = max(0, min(profit_score, 100))
            
            # 夏普比率得分 (>2为满分)
            sharpe_score = min(performance.sharpe_ratio * 50, 100)
            scores[PerformanceMetric.SHARPE_RATIO] = max(0, sharpe_score)
            
            # 最大回撤得分 (回撤越小得分越高)
            drawdown_score = max(0, 100 - abs(performance.max_drawdown) * 100)
            scores[PerformanceMetric.MAX_DRAWDOWN] = drawdown_score
            
            # 胜率得分
            scores[PerformanceMetric.WIN_RATE] = performance.win_rate * 100
            
            # 盈利因子得分 (>2为满分)
            pf_score = min(performance.profit_factor * 50, 100)
            scores[PerformanceMetric.PROFIT_FACTOR] = max(0, pf_score)
            
            # 卡玛比率得分
            calmar_score = min(performance.calmar_ratio * 50, 100)
            scores[PerformanceMetric.CALMAR_RATIO] = max(0, calmar_score)
            
            # 索提诺比率得分
            sortino_score = min(performance.sortino_ratio * 50, 100)
            scores[PerformanceMetric.SORTINO_RATIO] = max(0, sortino_score)
            
            # 加权计算综合得分
            total_score = 0.0
            for metric, weight in self.performance_weights.items():
                total_score += scores[metric] * weight
            
            return max(0, min(total_score, 100))
            
        except Exception as e:
            logger.error(f"计算性能得分失败: {e}")
            return 0.0
    
    def get_tier_from_level(self, level: int) -> AILevel:
        """根据等级获取段位"""
        for tier, (min_level, max_level) in self.tier_ranges.items():
            if min_level <= level <= max_level:
                return tier
        return AILevel.NOVICE
    
    def calculate_upgrade_threshold(self, level: int) -> float:
        """计算升级阈值"""
        base_threshold = 75.0  # 基础阈值75分
        
        # 等级越高，升级越困难
        level_factor = 1 + (level / 100) * 0.5  # 最高增加50%难度
        
        return base_threshold * level_factor
    
    def calculate_downgrade_threshold(self, level: int) -> float:
        """计算降级阈值"""
        base_threshold = 40.0  # 基础阈值40分
        
        # 等级越高，降级阈值越高（更容易降级）
        level_factor = 1 + (level / 100) * 0.3  # 最高增加30%
        
        return base_threshold * level_factor


class AIEvolutionEngine:
    """AI进化引擎"""
    
    def __init__(self):
        self.calculator = AILevelCalculator()
        self.ai_levels: Dict[str, AILevelInfo] = {}
        self.performance_history: Dict[str, List[AIPerformanceData]] = {}
        self.evolution_history: Dict[str, List[EvolutionEvent]] = {}
        self.evolution_lock = threading.RLock()
        
        # 进化配置
        self.min_performance_samples = 10  # 最少性能样本数
        self.performance_window = 100      # 性能评估窗口
        self.evolution_cooldown = 300      # 进化冷却时间(秒)
        
        logger.info("AI进化引擎初始化完成")
    
    def register_ai_model(self, model_id: str, initial_level: int = 1) -> bool:
        """注册AI模型"""
        try:
            with self.evolution_lock:
                if model_id in self.ai_levels:
                    logger.warning(f"AI模型已存在: {model_id}")
                    return False
                
                # 创建AI等级信息
                level_info = AILevelInfo(
                    level=initial_level,
                    tier=self.calculator.get_tier_from_level(initial_level),
                    experience=0.0,
                    next_level_exp=self.calculator.calculate_required_exp(initial_level),
                    performance_score=0.0,
                    upgrade_threshold=self.calculator.calculate_upgrade_threshold(initial_level),
                    downgrade_threshold=self.calculator.calculate_downgrade_threshold(initial_level),
                    consecutive_wins=0,
                    consecutive_losses=0,
                    last_upgrade_time=0.0,
                    last_downgrade_time=0.0,
                    total_upgrades=0,
                    total_downgrades=0
                )
                
                self.ai_levels[model_id] = level_info
                self.performance_history[model_id] = []
                self.evolution_history[model_id] = []
                
                logger.info(f"AI模型注册成功: {model_id} (等级: {initial_level})")
                return True
                
        except Exception as e:
            logger.error(f"注册AI模型失败: {e}")
            return False
    
    def update_performance(self, model_id: str, performance: AIPerformanceData) -> bool:
        """更新AI性能数据"""
        try:
            with self.evolution_lock:
                if model_id not in self.ai_levels:
                    logger.error(f"AI模型不存在: {model_id}")
                    return False
                
                # 添加性能数据
                self.performance_history[model_id].append(performance)
                
                # 保持历史数据在窗口范围内
                if len(self.performance_history[model_id]) > self.performance_window:
                    self.performance_history[model_id].pop(0)
                
                # 更新性能得分
                score = self.calculator.calculate_performance_score(performance)
                self.ai_levels[model_id].performance_score = score
                
                # 更新连续胜负记录
                if performance.profit_rate > 0:
                    self.ai_levels[model_id].consecutive_wins += 1
                    self.ai_levels[model_id].consecutive_losses = 0
                else:
                    self.ai_levels[model_id].consecutive_losses += 1
                    self.ai_levels[model_id].consecutive_wins = 0
                
                # 检查是否需要进化
                self._check_evolution(model_id)
                
                return True
                
        except Exception as e:
            logger.error(f"更新AI性能失败: {e}")
            return False
    
    def _check_evolution(self, model_id: str):
        """检查AI是否需要进化"""
        try:
            level_info = self.ai_levels[model_id]
            current_time = time.time()
            
            # 检查冷却时间
            last_evolution = max(level_info.last_upgrade_time, level_info.last_downgrade_time)
            if current_time - last_evolution < self.evolution_cooldown:
                return
            
            # 检查性能样本数量
            if len(self.performance_history[model_id]) < self.min_performance_samples:
                return
            
            # 计算最近性能平均分
            recent_performances = self.performance_history[model_id][-self.min_performance_samples:]
            avg_score = sum(self.calculator.calculate_performance_score(p) for p in recent_performances) / len(recent_performances)
            
            # 检查升级条件
            if avg_score >= level_info.upgrade_threshold and level_info.level < self.calculator.max_level:
                self._upgrade_ai(model_id, avg_score, "性能达到升级阈值")
            
            # 检查降级条件
            elif avg_score <= level_info.downgrade_threshold and level_info.level > 1:
                self._downgrade_ai(model_id, avg_score, "性能低于降级阈值")
            
        except Exception as e:
            logger.error(f"检查AI进化失败: {e}")
    
    def _upgrade_ai(self, model_id: str, performance_score: float, reason: str):
        """升级AI"""
        try:
            level_info = self.ai_levels[model_id]
            old_level = level_info.level
            old_tier = level_info.tier
            
            # 升级
            new_level = min(old_level + 1, self.calculator.max_level)
            new_tier = self.calculator.get_tier_from_level(new_level)
            
            # 更新等级信息
            level_info.level = new_level
            level_info.tier = new_tier
            level_info.experience = 0.0
            level_info.next_level_exp = self.calculator.calculate_required_exp(new_level)
            level_info.upgrade_threshold = self.calculator.calculate_upgrade_threshold(new_level)
            level_info.downgrade_threshold = self.calculator.calculate_downgrade_threshold(new_level)
            level_info.last_upgrade_time = time.time()
            level_info.total_upgrades += 1
            
            # 记录进化事件
            event = EvolutionEvent(
                timestamp=time.time(),
                event_type="upgrade",
                old_level=old_level,
                new_level=new_level,
                old_tier=old_tier,
                new_tier=new_tier,
                trigger_reason=reason,
                performance_data=self.performance_history[model_id][-1],
                description=f"AI模型 {model_id} 从 {old_level} 级升级到 {new_level} 级"
            )
            
            self.evolution_history[model_id].append(event)
            
            logger.info(f"🎉 AI升级: {model_id} {old_level}→{new_level} ({old_tier.value}→{new_tier.value})")
            
        except Exception as e:
            logger.error(f"AI升级失败: {e}")
    
    def _downgrade_ai(self, model_id: str, performance_score: float, reason: str):
        """降级AI"""
        try:
            level_info = self.ai_levels[model_id]
            old_level = level_info.level
            old_tier = level_info.tier
            
            # 降级
            new_level = max(old_level - 1, 1)
            new_tier = self.calculator.get_tier_from_level(new_level)
            
            # 更新等级信息
            level_info.level = new_level
            level_info.tier = new_tier
            level_info.experience = 0.0
            level_info.next_level_exp = self.calculator.calculate_required_exp(new_level)
            level_info.upgrade_threshold = self.calculator.calculate_upgrade_threshold(new_level)
            level_info.downgrade_threshold = self.calculator.calculate_downgrade_threshold(new_level)
            level_info.last_downgrade_time = time.time()
            level_info.total_downgrades += 1
            
            # 记录进化事件
            event = EvolutionEvent(
                timestamp=time.time(),
                event_type="downgrade",
                old_level=old_level,
                new_level=new_level,
                old_tier=old_tier,
                new_tier=new_tier,
                trigger_reason=reason,
                performance_data=self.performance_history[model_id][-1],
                description=f"AI模型 {model_id} 从 {old_level} 级降级到 {new_level} 级"
            )
            
            self.evolution_history[model_id].append(event)
            
            logger.warning(f"⬇️ AI降级: {model_id} {old_level}→{new_level} ({old_tier.value}→{new_tier.value})")
            
        except Exception as e:
            logger.error(f"AI降级失败: {e}")
    
    def get_ai_level_info(self, model_id: str) -> Optional[AILevelInfo]:
        """获取AI等级信息"""
        return self.ai_levels.get(model_id)
    
    def get_evolution_history(self, model_id: str, limit: int = 50) -> List[EvolutionEvent]:
        """获取进化历史"""
        if model_id not in self.evolution_history:
            return []
        
        history = self.evolution_history[model_id]
        return history[-limit:] if limit > 0 else history
    
    def get_performance_history(self, model_id: str, limit: int = 100) -> List[AIPerformanceData]:
        """获取性能历史"""
        if model_id not in self.performance_history:
            return []
        
        history = self.performance_history[model_id]
        return history[-limit:] if limit > 0 else history
    
    def get_all_ai_levels(self) -> Dict[str, AILevelInfo]:
        """获取所有AI等级信息"""
        return self.ai_levels.copy()
    
    def get_leaderboard(self) -> List[Tuple[str, AILevelInfo]]:
        """获取AI排行榜"""
        try:
            # 按等级和性能得分排序
            sorted_ais = sorted(
                self.ai_levels.items(),
                key=lambda x: (x[1].level, x[1].performance_score),
                reverse=True
            )
            
            return sorted_ais
            
        except Exception as e:
            logger.error(f"获取排行榜失败: {e}")
            return []
    
    def force_evolution(self, model_id: str, target_level: int, reason: str = "手动调整") -> bool:
        """强制进化到指定等级"""
        try:
            with self.evolution_lock:
                if model_id not in self.ai_levels:
                    logger.error(f"AI模型不存在: {model_id}")
                    return False
                
                if not (1 <= target_level <= self.calculator.max_level):
                    logger.error(f"目标等级无效: {target_level}")
                    return False
                
                level_info = self.ai_levels[model_id]
                old_level = level_info.level
                old_tier = level_info.tier
                
                # 更新等级
                level_info.level = target_level
                level_info.tier = self.calculator.get_tier_from_level(target_level)
                level_info.experience = 0.0
                level_info.next_level_exp = self.calculator.calculate_required_exp(target_level)
                level_info.upgrade_threshold = self.calculator.calculate_upgrade_threshold(target_level)
                level_info.downgrade_threshold = self.calculator.calculate_downgrade_threshold(target_level)
                
                # 记录进化事件
                event_type = "upgrade" if target_level > old_level else "downgrade"
                event = EvolutionEvent(
                    timestamp=time.time(),
                    event_type=event_type,
                    old_level=old_level,
                    new_level=target_level,
                    old_tier=old_tier,
                    new_tier=level_info.tier,
                    trigger_reason=reason,
                    performance_data=self.performance_history[model_id][-1] if self.performance_history[model_id] else None,
                    description=f"AI模型 {model_id} 强制调整从 {old_level} 级到 {target_level} 级"
                )
                
                self.evolution_history[model_id].append(event)
                
                logger.info(f"🔧 强制进化: {model_id} {old_level}→{target_level}")
                return True
                
        except Exception as e:
            logger.error(f"强制进化失败: {e}")
            return False
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """获取进化统计信息"""
        try:
            stats = {
                "total_models": len(self.ai_levels),
                "level_distribution": {},
                "tier_distribution": {},
                "total_upgrades": 0,
                "total_downgrades": 0,
                "avg_level": 0.0,
                "highest_level": 0,
                "lowest_level": 100
            }
            
            if not self.ai_levels:
                return stats
            
            # 统计等级分布
            for level in range(1, 101):
                stats["level_distribution"][level] = 0
            
            for tier in AILevel:
                stats["tier_distribution"][tier.value] = 0
            
            total_level = 0
            for model_id, level_info in self.ai_levels.items():
                # 等级分布
                stats["level_distribution"][level_info.level] += 1
                
                # 段位分布
                stats["tier_distribution"][level_info.tier.value] += 1
                
                # 升降级统计
                stats["total_upgrades"] += level_info.total_upgrades
                stats["total_downgrades"] += level_info.total_downgrades
                
                # 等级统计
                total_level += level_info.level
                stats["highest_level"] = max(stats["highest_level"], level_info.level)
                stats["lowest_level"] = min(stats["lowest_level"], level_info.level)
            
            stats["avg_level"] = total_level / len(self.ai_levels)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取进化统计失败: {e}")
            return {}


# 全局AI进化引擎实例
ai_evolution_engine = AIEvolutionEngine()
