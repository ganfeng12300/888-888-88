"""
🎯 实时学习与动态优化系统
生产级实时学习系统，支持在线学习、增量训练、动态参数调整
实现完整的自适应学习流程、性能评估、模型更新等功能
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

@dataclass
class LearningMetrics:
    """学习指标"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptationEvent:
    """适应事件"""
    event_type: str  # 'parameter_update', 'model_retrain', 'weight_adjust'
    model_name: str
    old_value: Any
    new_value: Any
    reason: str
    impact_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class RealTimeLearningEngine:
    """实时学习引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 学习参数
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.decay_rate = 0.95
        self.adaptation_threshold = 0.05
        self.min_samples_for_update = 50
        
        # 数据存储
        self.training_buffer = deque(maxlen=10000)
        self.validation_buffer = deque(maxlen=2000)
        self.performance_history = deque(maxlen=5000)
        self.adaptation_history = deque(maxlen=1000)
        
        # 模型状态
        self.model_states = {}
        self.gradient_cache = {}
        self.momentum_cache = {}
        
        # 性能监控
        self.current_metrics = LearningMetrics()
        self.baseline_metrics = LearningMetrics()
        
    async def start(self):
        """启动实时学习引擎"""
        self.is_running = True
        self.logger.info("🎯 实时学习引擎启动")
        
        # 启动学习循环
        tasks = [
            asyncio.create_task(self._online_learning_loop()),
            asyncio.create_task(self._performance_evaluation_loop()),
            asyncio.create_task(self._adaptation_loop()),
            asyncio.create_task(self._model_optimization_loop())
        ]
        
        await asyncio.gather(*tasks)
        
    async def stop(self):
        """停止实时学习引擎"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("🎯 实时学习引擎停止")
        
    async def _online_learning_loop(self):
        """在线学习循环"""
        while self.is_running:
            try:
                # 检查是否有足够的新数据进行学习
                if len(self.training_buffer) >= self.min_samples_for_update:
                    await self._perform_incremental_learning()
                    
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                self.logger.error(f"在线学习错误: {e}")
                await asyncio.sleep(60)
                
    async def _perform_incremental_learning(self):
        """执行增量学习"""
        try:
            # 获取最新的训练数据
            recent_data = list(self.training_buffer)[-self.min_samples_for_update:]
            
            # 准备训练数据
            X, y = self._prepare_learning_data(recent_data)
            
            if len(X) == 0:
                return
                
            # 对每个模型进行增量学习
            for model_name in self.model_states.keys():
                await self._incremental_update(model_name, X, y)
                
            self.logger.info(f"完成增量学习，样本数: {len(X)}")
            
        except Exception as e:
            self.logger.error(f"增量学习失败: {e}")
            
    def _prepare_learning_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备学习数据"""
        if not data:
            return np.array([]), np.array([])
            
        df = pd.DataFrame(data)
        
        # 特征列
        feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'sma_5', 'sma_20', 'price_change', 'volume_change'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            return np.array([]), np.array([])
            
        X = df[available_columns].fillna(0).values
        
        # 标签（实际收益率）
        if 'actual_return' in df.columns:
            y = (df['actual_return'] > 0).astype(int).values
        else:
            # 如果没有实际收益率，使用价格变化
            y = (df['price'].pct_change() > 0).astype(int).values
            
        # 移除NaN值
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
        
    async def _incremental_update(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """增量更新模型"""
        try:
            if model_name not in self.model_states:
                self._initialize_model_state(model_name, X.shape[1])
                
            # 计算梯度
            gradients = await self._compute_gradients(model_name, X, y)
            
            # 更新模型参数
            await self._update_model_parameters(model_name, gradients)
            
            # 记录更新事件
            self._record_adaptation_event(
                'parameter_update',
                model_name,
                'gradients',
                gradients,
                f'增量学习更新，样本数: {len(X)}'
            )
            
        except Exception as e:
            self.logger.error(f"增量更新模型 {model_name} 失败: {e}")
            
    def _initialize_model_state(self, model_name: str, input_dim: int):
        """初始化模型状态"""
        # 简单的线性模型参数
        self.model_states[model_name] = {
            'weights': np.random.normal(0, 0.1, (input_dim, 1)),
            'bias': np.zeros((1,)),
            'input_dim': input_dim
        }
        
        # 初始化梯度缓存
        self.gradient_cache[model_name] = {
            'weights': np.zeros((input_dim, 1)),
            'bias': np.zeros((1,))
        }
        
        # 初始化动量缓存
        self.momentum_cache[model_name] = {
            'weights': np.zeros((input_dim, 1)),
            'bias': np.zeros((1,))
        }
        
    async def _compute_gradients(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """计算梯度"""
        state = self.model_states[model_name]
        weights = state['weights']
        bias = state['bias']
        
        # 前向传播
        z = np.dot(X, weights) + bias
        predictions = self._sigmoid(z)
        
        # 计算损失梯度
        m = X.shape[0]
        dz = predictions - y.reshape(-1, 1)
        
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz, axis=0)
        
        return {
            'weights': dw,
            'bias': db
        }
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
    async def _update_model_parameters(self, model_name: str, gradients: Dict[str, np.ndarray]):
        """更新模型参数"""
        state = self.model_states[model_name]
        momentum = self.momentum_cache[model_name]
        
        # 动量更新
        momentum['weights'] = self.momentum * momentum['weights'] - self.learning_rate * gradients['weights']
        momentum['bias'] = self.momentum * momentum['bias'] - self.learning_rate * gradients['bias']
        
        # 参数更新
        state['weights'] += momentum['weights']
        state['bias'] += momentum['bias']
        
        # 梯度缓存更新
        self.gradient_cache[model_name] = gradients
        
    async def _performance_evaluation_loop(self):
        """性能评估循环"""
        while self.is_running:
            try:
                await self._evaluate_model_performance()
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # 1分钟评估一次
                
            except Exception as e:
                self.logger.error(f"性能评估错误: {e}")
                await asyncio.sleep(30)
                
    async def _evaluate_model_performance(self):
        """评估模型性能"""
        if len(self.validation_buffer) < 20:
            return
            
        # 获取验证数据
        validation_data = list(self.validation_buffer)[-100:]
        X_val, y_val = self._prepare_learning_data(validation_data)
        
        if len(X_val) == 0:
            return
            
        # 评估每个模型
        for model_name, state in self.model_states.items():
            try:
                # 预测
                predictions = await self._predict_with_state(model_name, X_val)
                
                # 计算指标
                metrics = self._calculate_metrics(y_val, predictions)
                
                # 更新性能历史
                self.performance_history.append({
                    'model_name': model_name,
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 性能失败: {e}")
                
    async def _predict_with_state(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """使用模型状态进行预测"""
        state = self.model_states[model_name]
        z = np.dot(X, state['weights']) + state['bias']
        predictions = self._sigmoid(z)
        return (predictions > 0.5).astype(int).flatten()
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> LearningMetrics:
        """计算性能指标"""
        # 基本分类指标
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return LearningMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            win_rate=accuracy  # 简化处理
        )
        
    async def _update_performance_metrics(self):
        """更新性能指标"""
        if not self.performance_history:
            return
            
        # 计算最近的平均性能
        recent_metrics = list(self.performance_history)[-10:]
        
        if recent_metrics:
            avg_accuracy = np.mean([m['metrics'].accuracy for m in recent_metrics])
            avg_precision = np.mean([m['metrics'].precision for m in recent_metrics])
            avg_recall = np.mean([m['metrics'].recall for m in recent_metrics])
            avg_f1 = np.mean([m['metrics'].f1_score for m in recent_metrics])
            
            self.current_metrics = LearningMetrics(
                accuracy=avg_accuracy,
                precision=avg_precision,
                recall=avg_recall,
                f1_score=avg_f1,
                timestamp=datetime.now()
            )
            
    async def _adaptation_loop(self):
        """自适应调整循环"""
        while self.is_running:
            try:
                await self._check_adaptation_triggers()
                await self._perform_adaptive_adjustments()
                await asyncio.sleep(120)  # 2分钟检查一次
                
            except Exception as e:
                self.logger.error(f"自适应调整错误: {e}")
                await asyncio.sleep(60)
                
    async def _check_adaptation_triggers(self):
        """检查适应触发条件"""
        if len(self.performance_history) < 10:
            return
            
        # 检查性能下降
        recent_performance = [m['metrics'].accuracy for m in list(self.performance_history)[-5:]]
        older_performance = [m['metrics'].accuracy for m in list(self.performance_history)[-10:-5]]
        
        if len(recent_performance) >= 3 and len(older_performance) >= 3:
            recent_avg = np.mean(recent_performance)
            older_avg = np.mean(older_performance)
            
            # 如果性能下降超过阈值，触发适应
            if older_avg - recent_avg > self.adaptation_threshold:
                await self._trigger_adaptation('performance_decline')
                
    async def _trigger_adaptation(self, reason: str):
        """触发适应调整"""
        self.logger.info(f"触发适应调整: {reason}")
        
        # 调整学习率
        if reason == 'performance_decline':
            old_lr = self.learning_rate
            self.learning_rate *= 1.1  # 增加学习率
            self.learning_rate = min(self.learning_rate, 0.01)  # 限制最大值
            
            self._record_adaptation_event(
                'parameter_update',
                'learning_engine',
                old_lr,
                self.learning_rate,
                f'性能下降，调整学习率: {reason}'
            )
            
    async def _perform_adaptive_adjustments(self):
        """执行自适应调整"""
        # 动态调整学习率
        if len(self.performance_history) >= 5:
            recent_metrics = list(self.performance_history)[-5:]
            accuracies = [m['metrics'].accuracy for m in recent_metrics]
            
            # 如果准确率方差太大，降低学习率
            if np.var(accuracies) > 0.01:
                old_lr = self.learning_rate
                self.learning_rate *= 0.95
                self.learning_rate = max(self.learning_rate, 0.0001)
                
                if abs(old_lr - self.learning_rate) > 0.0001:
                    self._record_adaptation_event(
                        'parameter_update',
                        'learning_engine',
                        old_lr,
                        self.learning_rate,
                        '准确率方差过大，降低学习率'
                    )
                    
    async def _model_optimization_loop(self):
        """模型优化循环"""
        while self.is_running:
            try:
                await self._optimize_model_architecture()
                await self._prune_ineffective_features()
                await asyncio.sleep(600)  # 10分钟优化一次
                
            except Exception as e:
                self.logger.error(f"模型优化错误: {e}")
                await asyncio.sleep(300)
                
    async def _optimize_model_architecture(self):
        """优化模型架构"""
        # 这里可以实现模型架构优化
        # 例如：添加/删除层、调整神经元数量等
        pass
        
    async def _prune_ineffective_features(self):
        """剪枝无效特征"""
        # 这里可以实现特征选择和剪枝
        # 基于特征重要性分析
        pass
        
    def _record_adaptation_event(self, event_type: str, model_name: str, 
                                old_value: Any, new_value: Any, reason: str):
        """记录适应事件"""
        event = AdaptationEvent(
            event_type=event_type,
            model_name=model_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            impact_score=0.0  # 可以后续计算影响分数
        )
        
        self.adaptation_history.append(event)
        self.logger.info(f"适应事件: {event_type} - {model_name} - {reason}")
        
    async def add_training_sample(self, sample: Dict):
        """添加训练样本"""
        self.training_buffer.append(sample)
        
    async def add_validation_sample(self, sample: Dict):
        """添加验证样本"""
        self.validation_buffer.append(sample)
        
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            'is_running': self.is_running,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'training_samples': len(self.training_buffer),
            'validation_samples': len(self.validation_buffer),
            'current_metrics': {
                'accuracy': self.current_metrics.accuracy,
                'precision': self.current_metrics.precision,
                'recall': self.current_metrics.recall,
                'f1_score': self.current_metrics.f1_score
            },
            'model_count': len(self.model_states),
            'adaptation_events': len(self.adaptation_history)
        }
        
    def get_recent_adaptations(self, limit: int = 10) -> List[Dict]:
        """获取最近的适应事件"""
        events = list(self.adaptation_history)[-limit:]
        return [
            {
                'event_type': e.event_type,
                'model_name': e.model_name,
                'reason': e.reason,
                'timestamp': e.timestamp.isoformat(),
                'impact_score': e.impact_score
            }
            for e in events
        ]
        
    async def reset_learning_state(self):
        """重置学习状态"""
        self.model_states.clear()
        self.gradient_cache.clear()
        self.momentum_cache.clear()
        self.training_buffer.clear()
        self.validation_buffer.clear()
        self.logger.info("学习状态已重置")
        
    async def save_learning_state(self, filepath: str):
        """保存学习状态"""
        try:
            state = {
                'model_states': self.model_states,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'current_metrics': self.current_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            self.logger.info(f"学习状态已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存学习状态失败: {e}")
            
    async def load_learning_state(self, filepath: str):
        """加载学习状态"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"学习状态文件不存在: {filepath}")
                return
                
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.model_states = state.get('model_states', {})
            self.learning_rate = state.get('learning_rate', 0.001)
            self.momentum = state.get('momentum', 0.9)
            self.current_metrics = state.get('current_metrics', LearningMetrics())
            
            self.logger.info(f"学习状态已从 {filepath} 加载")
            
        except Exception as e:
            self.logger.error(f"加载学习状态失败: {e}")

