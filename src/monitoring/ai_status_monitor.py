"""
🧠 AI状态监控器 - 生产级实盘交易AI模型状态实时监控系统
监控AI等级进化、模型性能、训练状态、决策质量等AI相关指标
提供AI健康评估、性能优化建议、模型升级提醒
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class AIModelType(Enum):
    """AI模型类型"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # 强化学习
    DEEP_LEARNING = "deep_learning"  # 深度学习
    ENSEMBLE_LEARNING = "ensemble_learning"  # 集成学习
    EXPERT_SYSTEM = "expert_system"  # 专家系统
    META_LEARNING = "meta_learning"  # 元学习
    TRANSFER_LEARNING = "transfer_learning"  # 迁移学习

class AIStatus(Enum):
    """AI状态"""
    TRAINING = "training"  # 训练中
    ACTIVE = "active"  # 激活
    IDLE = "idle"  # 空闲
    ERROR = "error"  # 错误
    UPGRADING = "upgrading"  # 升级中

class AILevel(Enum):
    """AI等级"""
    BRONZE = "bronze"  # 青铜级 (1-20)
    SILVER = "silver"  # 白银级 (21-40)
    GOLD = "gold"  # 黄金级 (41-60)
    PLATINUM = "platinum"  # 铂金级 (61-80)
    DIAMOND = "diamond"  # 钻石级 (81-95)
    EPIC = "epic"  # 史诗级 (96-100)

@dataclass
class AIModelMetrics:
    """AI模型指标"""
    model_id: str  # 模型ID
    model_type: AIModelType  # 模型类型
    ai_level: int  # AI等级 (1-100)
    ai_level_category: AILevel  # AI等级分类
    accuracy: float  # 准确率
    precision: float  # 精确率
    recall: float  # 召回率
    f1_score: float  # F1分数
    training_loss: float  # 训练损失
    validation_loss: float  # 验证损失
    learning_rate: float  # 学习率
    epochs_completed: int  # 完成的训练轮数
    training_time: float  # 训练时间 (秒)
    inference_time: float  # 推理时间 (毫秒)
    memory_usage: float  # 内存使用 (MB)
    gpu_usage: float  # GPU使用率
    status: AIStatus  # 状态
    last_updated: float = field(default_factory=time.time)  # 最后更新时间

@dataclass
class AIEvolutionEvent:
    """AI进化事件"""
    event_id: str  # 事件ID
    model_id: str  # 模型ID
    event_type: str  # 事件类型 (level_up, level_down, upgrade, etc.)
    old_level: int  # 旧等级
    new_level: int  # 新等级
    trigger_reason: str  # 触发原因
    performance_change: float  # 性能变化
    timestamp: float = field(default_factory=time.time)  # 时间戳

@dataclass
class AIDecisionMetrics:
    """AI决策指标"""
    model_id: str  # 模型ID
    total_decisions: int  # 总决策数
    correct_decisions: int  # 正确决策数
    profitable_decisions: int  # 盈利决策数
    decision_accuracy: float  # 决策准确率
    profit_ratio: float  # 盈利比率
    average_confidence: float  # 平均置信度
    decision_speed: float  # 决策速度 (毫秒)
    timestamp: float = field(default_factory=time.time)

class AILevelManager:
    """AI等级管理器"""
    
    def __init__(self):
        # 等级阈值配置
        self.level_thresholds = {
            'accuracy': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # 每10级的准确率要求
            'profit_ratio': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],  # 盈利比率要求
            'stability': [0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9]  # 稳定性要求
        }
        
        logger.info("AI等级管理器初始化完成")
    
    def calculate_ai_level(self, accuracy: float, profit_ratio: float, 
                          stability: float, current_level: int = 1) -> int:
        """计算AI等级"""
        try:
            # 基础分数计算
            accuracy_score = min(accuracy * 100, 100)
            profit_score = min(profit_ratio * 100, 100)
            stability_score = min(stability * 100, 100)
            
            # 综合分数 (加权平均)
            composite_score = (accuracy_score * 0.4 + profit_score * 0.4 + stability_score * 0.2)
            
            # 等级计算
            if composite_score >= 95:
                target_level = min(95 + int((composite_score - 95) * 2), 100)
            elif composite_score >= 90:
                target_level = 85 + int((composite_score - 90))
            elif composite_score >= 80:
                target_level = 70 + int((composite_score - 80) * 1.5)
            elif composite_score >= 70:
                target_level = 50 + int((composite_score - 70) * 2)
            elif composite_score >= 60:
                target_level = 30 + int((composite_score - 60) * 2)
            elif composite_score >= 50:
                target_level = 10 + int((composite_score - 50) * 2)
            else:
                target_level = max(1, int(composite_score / 5))
            
            # 渐进式升级/降级 (防止等级剧烈波动)
            level_change = target_level - current_level
            if abs(level_change) > 5:
                # 限制单次等级变化不超过5级
                if level_change > 0:
                    new_level = current_level + 5
                else:
                    new_level = current_level - 5
            else:
                new_level = target_level
            
            return max(1, min(100, new_level))
        
        except Exception as e:
            logger.error(f"计算AI等级失败: {e}")
            return current_level
    
    def get_level_category(self, level: int) -> AILevel:
        """获取等级分类"""
        if level >= 96:
            return AILevel.EPIC
        elif level >= 81:
            return AILevel.DIAMOND
        elif level >= 61:
            return AILevel.PLATINUM
        elif level >= 41:
            return AILevel.GOLD
        elif level >= 21:
            return AILevel.SILVER
        else:
            return AILevel.BRONZE
    
    def get_upgrade_requirements(self, current_level: int) -> Dict[str, float]:
        """获取升级要求"""
        try:
            tier = min(current_level // 10, 9)
            next_tier = min(tier + 1, 9)
            
            return {
                'accuracy': self.level_thresholds['accuracy'][next_tier],
                'profit_ratio': self.level_thresholds['profit_ratio'][next_tier],
                'stability': self.level_thresholds['stability'][next_tier],
                'target_level': min((tier + 1) * 10 + 10, 100)
            }
        
        except Exception as e:
            logger.error(f"获取升级要求失败: {e}")
            return {}

class AIPerformanceTracker:
    """AI性能跟踪器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.decision_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("AI性能跟踪器初始化完成")
    
    def record_performance(self, model_id: str, accuracy: float, loss: float, 
                          inference_time: float, memory_usage: float):
        """记录性能数据"""
        try:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            performance_record = {
                'accuracy': accuracy,
                'loss': loss,
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'timestamp': time.time()
            }
            
            self.performance_history[model_id].append(performance_record)
            
            # 保持历史记录在合理范围内
            if len(self.performance_history[model_id]) > self.window_size:
                self.performance_history[model_id] = self.performance_history[model_id][-self.window_size//2:]
        
        except Exception as e:
            logger.error(f"记录性能数据失败: {e}")
    
    def record_decision(self, model_id: str, decision_correct: bool, 
                       decision_profitable: bool, confidence: float, decision_time: float):
        """记录决策数据"""
        try:
            if model_id not in self.decision_history:
                self.decision_history[model_id] = []
            
            decision_record = {
                'correct': decision_correct,
                'profitable': decision_profitable,
                'confidence': confidence,
                'decision_time': decision_time,
                'timestamp': time.time()
            }
            
            self.decision_history[model_id].append(decision_record)
            
            # 保持历史记录在合理范围内
            if len(self.decision_history[model_id]) > self.window_size:
                self.decision_history[model_id] = self.decision_history[model_id][-self.window_size//2:]
        
        except Exception as e:
            logger.error(f"记录决策数据失败: {e}")
    
    def calculate_stability(self, model_id: str, window: int = 100) -> float:
        """计算模型稳定性"""
        try:
            if model_id not in self.performance_history:
                return 0.0
            
            recent_performance = self.performance_history[model_id][-window:]
            if len(recent_performance) < 10:
                return 0.0
            
            # 计算准确率的标准差 (稳定性 = 1 - 标准差)
            accuracies = [p['accuracy'] for p in recent_performance]
            accuracy_std = np.std(accuracies)
            
            # 稳定性分数 (标准差越小，稳定性越高)
            stability = max(0, 1 - accuracy_std * 2)
            
            return stability
        
        except Exception as e:
            logger.error(f"计算模型稳定性失败: {e}")
            return 0.0
    
    def get_performance_trend(self, model_id: str, window: int = 50) -> Dict[str, float]:
        """获取性能趋势"""
        try:
            if model_id not in self.performance_history:
                return {}
            
            recent_performance = self.performance_history[model_id][-window:]
            if len(recent_performance) < 10:
                return {}
            
            # 计算趋势 (最近一半 vs 前一半的平均值)
            mid_point = len(recent_performance) // 2
            first_half = recent_performance[:mid_point]
            second_half = recent_performance[mid_point:]
            
            first_accuracy = np.mean([p['accuracy'] for p in first_half])
            second_accuracy = np.mean([p['accuracy'] for p in second_half])
            
            first_loss = np.mean([p['loss'] for p in first_half])
            second_loss = np.mean([p['loss'] for p in second_half])
            
            return {
                'accuracy_trend': second_accuracy - first_accuracy,
                'loss_trend': first_loss - second_loss,  # 损失减少是好的
                'improving': (second_accuracy > first_accuracy) and (second_loss < first_loss)
            }
        
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {}

class AIStatusMonitor:
    """AI状态监控器主类"""
    
    def __init__(self):
        self.level_manager = AILevelManager()
        self.performance_tracker = AIPerformanceTracker()
        
        # AI模型状态
        self.ai_models: Dict[str, AIModelMetrics] = {}
        self.evolution_events: List[AIEvolutionEvent] = []
        
        # 监控配置
        self.monitor_interval = 10  # 监控间隔（秒）
        self.is_monitoring = False
        
        # 回调函数
        self.evolution_callbacks: List[Callable[[AIEvolutionEvent], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("AI状态监控器初始化完成")
    
    def register_ai_model(self, model_id: str, model_type: AIModelType, 
                         initial_level: int = 1) -> bool:
        """注册AI模型"""
        try:
            with self.lock:
                if model_id in self.ai_models:
                    logger.warning(f"AI模型已存在: {model_id}")
                    return False
                
                self.ai_models[model_id] = AIModelMetrics(
                    model_id=model_id,
                    model_type=model_type,
                    ai_level=initial_level,
                    ai_level_category=self.level_manager.get_level_category(initial_level),
                    accuracy=0.5,
                    precision=0.5,
                    recall=0.5,
                    f1_score=0.5,
                    training_loss=1.0,
                    validation_loss=1.0,
                    learning_rate=0.001,
                    epochs_completed=0,
                    training_time=0,
                    inference_time=0,
                    memory_usage=0,
                    gpu_usage=0,
                    status=AIStatus.IDLE
                )
                
                logger.info(f"AI模型注册成功: {model_id} ({model_type.value})")
                return True
        
        except Exception as e:
            logger.error(f"注册AI模型失败: {e}")
            return False
    
    def update_model_metrics(self, model_id: str, **kwargs) -> bool:
        """更新模型指标"""
        try:
            with self.lock:
                if model_id not in self.ai_models:
                    logger.error(f"AI模型不存在: {model_id}")
                    return False
                
                model = self.ai_models[model_id]
                old_level = model.ai_level
                
                # 更新指标
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                
                model.last_updated = time.time()
                
                # 记录性能数据
                if 'accuracy' in kwargs:
                    self.performance_tracker.record_performance(
                        model_id, 
                        kwargs.get('accuracy', model.accuracy),
                        kwargs.get('training_loss', model.training_loss),
                        kwargs.get('inference_time', model.inference_time),
                        kwargs.get('memory_usage', model.memory_usage)
                    )
                
                # 重新计算AI等级
                stability = self.performance_tracker.calculate_stability(model_id)
                profit_ratio = kwargs.get('profit_ratio', 0.5)  # 需要从外部传入
                
                new_level = self.level_manager.calculate_ai_level(
                    model.accuracy, profit_ratio, stability, old_level
                )
                
                # 检查等级变化
                if new_level != old_level:
                    model.ai_level = new_level
                    model.ai_level_category = self.level_manager.get_level_category(new_level)
                    
                    # 创建进化事件
                    evolution_event = AIEvolutionEvent(
                        event_id=f"evolution_{model_id}_{int(time.time())}",
                        model_id=model_id,
                        event_type="level_up" if new_level > old_level else "level_down",
                        old_level=old_level,
                        new_level=new_level,
                        trigger_reason=f"Performance update: accuracy={model.accuracy:.3f}, stability={stability:.3f}",
                        performance_change=new_level - old_level
                    )
                    
                    self._process_evolution_event(evolution_event)
                
                return True
        
        except Exception as e:
            logger.error(f"更新模型指标失败: {e}")
            return False
    
    def record_decision_result(self, model_id: str, correct: bool, profitable: bool, 
                             confidence: float, decision_time: float):
        """记录决策结果"""
        try:
            self.performance_tracker.record_decision(
                model_id, correct, profitable, confidence, decision_time
            )
        
        except Exception as e:
            logger.error(f"记录决策结果失败: {e}")
    
    def _process_evolution_event(self, event: AIEvolutionEvent):
        """处理进化事件"""
        try:
            # 添加到历史记录
            self.evolution_events.append(event)
            
            # 保持历史记录在合理范围内
            if len(self.evolution_events) > 1000:
                self.evolution_events = self.evolution_events[-500:]
            
            # 调用回调函数
            for callback in self.evolution_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"进化事件回调执行失败: {e}")
            
            # 记录日志
            if event.event_type == "level_up":
                logger.info(f"🎉 AI升级: {event.model_id} {event.old_level}→{event.new_level}级")
            else:
                logger.warning(f"⬇️ AI降级: {event.model_id} {event.old_level}→{event.new_level}级")
        
        except Exception as e:
            logger.error(f"处理进化事件失败: {e}")
    
    def start_monitoring(self):
        """启动监控"""
        try:
            self.is_monitoring = True
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            logger.info("AI状态监控启动")
        
        except Exception as e:
            logger.error(f"启动AI状态监控失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("AI状态监控停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                with self.lock:
                    # 检查模型健康状态
                    self._check_model_health()
                
                time.sleep(self.monitor_interval)
            
            except Exception as e:
                logger.error(f"AI监控循环失败: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_model_health(self):
        """检查模型健康状态"""
        try:
            current_time = time.time()
            
            for model_id, model in self.ai_models.items():
                # 检查模型是否长时间未更新
                if current_time - model.last_updated > 300:  # 5分钟
                    if model.status != AIStatus.IDLE:
                        logger.warning(f"AI模型长时间未更新: {model_id}")
                        model.status = AIStatus.IDLE
                
                # 检查性能趋势
                trend = self.performance_tracker.get_performance_trend(model_id)
                if trend and not trend.get('improving', True):
                    logger.warning(f"AI模型性能下降: {model_id}")
        
        except Exception as e:
            logger.error(f"检查模型健康状态失败: {e}")
    
    def add_evolution_callback(self, callback: Callable[[AIEvolutionEvent], None]):
        """添加进化事件回调"""
        self.evolution_callbacks.append(callback)
    
    def update_ai_status(self) -> Dict[str, Any]:
        """更新并获取AI状态"""
        try:
            with self.lock:
                # 检查模型健康状态
                self._check_model_health()
                
                # 获取AI摘要
                return self.get_ai_summary()
        
        except Exception as e:
            logger.error(f"更新AI状态失败: {e}")
            return {}
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """获取整体性能指标"""
        try:
            with self.lock:
                if not self.ai_models:
                    return {
                        'total_models': 0,
                        'average_accuracy': 0.0,
                        'average_confidence': 0.0,
                        'active_models': 0,
                        'performance_score': 0.0
                    }
                
                total_accuracy = 0
                total_confidence = 0
                active_models = 0
                
                for model in self.ai_models.values():
                    if hasattr(model, 'accuracy') and model.accuracy is not None:
                        total_accuracy += model.accuracy
                    if hasattr(model, 'confidence') and model.confidence is not None:
                        total_confidence += model.confidence
                    if hasattr(model, 'status') and model.status == 'active':
                        active_models += 1
                
                total_models = len(self.ai_models)
                avg_accuracy = total_accuracy / total_models if total_models > 0 else 0
                avg_confidence = total_confidence / total_models if total_models > 0 else 0
                performance_score = (avg_accuracy + avg_confidence) / 2
                
                return {
                    'total_models': total_models,
                    'average_accuracy': avg_accuracy,
                    'average_confidence': avg_confidence,
                    'active_models': active_models,
                    'performance_score': performance_score
                }
        
        except Exception as e:
            logger.error(f"获取整体性能失败: {e}")
            return {
                'total_models': 0,
                'average_accuracy': 0.0,
                'average_confidence': 0.0,
                'active_models': 0,
                'performance_score': 0.0
            }
    
    def get_ai_summary(self) -> Dict[str, Any]:
        """获取AI摘要"""
        try:
            with self.lock:
                if not self.ai_models:
                    return {}
                
                # 统计各等级分布
                level_distribution = {}
                status_distribution = {}
                type_distribution = {}
                
                total_accuracy = 0
                total_models = len(self.ai_models)
                
                for model in self.ai_models.values():
                    # 等级分布
                    category = model.ai_level_category.value
                    level_distribution[category] = level_distribution.get(category, 0) + 1
                    
                    # 状态分布
                    status = model.status.value
                    status_distribution[status] = status_distribution.get(status, 0) + 1
                    
                    # 类型分布
                    model_type = model.model_type.value
                    type_distribution[model_type] = type_distribution.get(model_type, 0) + 1
                    
                    total_accuracy += model.accuracy
                
                # 最近进化事件
                recent_evolutions = [e for e in self.evolution_events if time.time() - e.timestamp < 3600]
                
                return {
                    'total_models': total_models,
                    'average_accuracy': total_accuracy / total_models if total_models > 0 else 0,
                    'level_distribution': level_distribution,
                    'status_distribution': status_distribution,
                    'type_distribution': type_distribution,
                    'recent_evolutions': len(recent_evolutions),
                    'total_evolutions': len(self.evolution_events),
                    'monitoring_status': self.is_monitoring
                }
        
        except Exception as e:
            logger.error(f"获取AI摘要失败: {e}")
            return {}
    
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型详细信息"""
        try:
            with self.lock:
                if model_id not in self.ai_models:
                    return None
                
                model = self.ai_models[model_id]
                stability = self.performance_tracker.calculate_stability(model_id)
                trend = self.performance_tracker.get_performance_trend(model_id)
                upgrade_requirements = self.level_manager.get_upgrade_requirements(model.ai_level)
                
                return {
                    'basic_info': {
                        'model_id': model.model_id,
                        'model_type': model.model_type.value,
                        'ai_level': model.ai_level,
                        'ai_level_category': model.ai_level_category.value,
                        'status': model.status.value
                    },
                    'performance_metrics': {
                        'accuracy': model.accuracy,
                        'precision': model.precision,
                        'recall': model.recall,
                        'f1_score': model.f1_score,
                        'stability': stability
                    },
                    'training_metrics': {
                        'training_loss': model.training_loss,
                        'validation_loss': model.validation_loss,
                        'learning_rate': model.learning_rate,
                        'epochs_completed': model.epochs_completed,
                        'training_time': model.training_time
                    },
                    'resource_usage': {
                        'inference_time': model.inference_time,
                        'memory_usage': model.memory_usage,
                        'gpu_usage': model.gpu_usage
                    },
                    'performance_trend': trend,
                    'upgrade_requirements': upgrade_requirements,
                    'last_updated': model.last_updated
                }
        
        except Exception as e:
            logger.error(f"获取模型详细信息失败: {e}")
            return None
    
    def get_recent_evolutions(self, limit: int = 20) -> List[AIEvolutionEvent]:
        """获取最近的进化事件"""
        with self.lock:
            return sorted(self.evolution_events, key=lambda x: x.timestamp, reverse=True)[:limit]

# 全局AI状态监控器实例
ai_status_monitor = AIStatusMonitor()
