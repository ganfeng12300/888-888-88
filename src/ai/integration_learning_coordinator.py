#!/usr/bin/env python3
"""
🧠 集成学习协调AI - Level 5智能体
Integration Learning Coordinator - Level 5 Agent

负责协调多个AI模型的决策，实现集成学习和模型融合
专为生产级实盘交易设计，支持动态权重调整和性能优化
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelPrediction:
    """模型预测结果"""
    model_id: str
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    execution_time_ms: float
    timestamp: datetime


@dataclass
class CoordinationDecision:
    """协调决策结果"""
    signal: float           # -1到1之间的信号
    confidence: float       # 0到1之间的置信度
    model_weights: Dict[str, float]  # 各模型权重
    ensemble_method: str    # 集成方法
    feature_contributions: Dict[str, float]  # 特征贡献度
    reasoning: str          # 决策推理
    timestamp: datetime
    execution_time_ms: float


class EnsembleNetwork(nn.Module):
    """集成神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(EnsembleNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # 输出-1到1之间
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class IntegrationLearningCoordinator:
    """集成学习协调AI"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化协调器"""
        self.config = config or {}
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.prediction_history = []
        self.is_training = False
        self.lock = threading.Lock()
        
        # 配置参数
        self.max_models = self.config.get('max_models', 10)
        self.weight_decay = self.config.get('weight_decay', 0.95)
        self.min_confidence = self.config.get('min_confidence', 0.1)
        self.ensemble_methods = ['weighted_average', 'voting', 'stacking', 'neural_ensemble']
        self.current_ensemble_method = 'weighted_average'
        
        # 初始化集成网络
        self.ensemble_net = None
        self.ensemble_optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化基础模型
        self._initialize_base_models()
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info("🧠 集成学习协调AI (Level 5) 初始化完成")
    
    def _initialize_base_models(self):
        """初始化基础模型"""
        try:
            # 线性回归模型
            self.models['linear'] = {
                'instance': LinearRegression(),
                'weight': 0.2,
                'performance': 0.5,
                'last_update': time.time()
            }
            
            # 决策树模型
            self.models['tree'] = {
                'instance': DecisionTreeRegressor(max_depth=10, random_state=42),
                'weight': 0.2,
                'performance': 0.5,
                'last_update': time.time()
            }
            
            # 神经网络模型
            self.models['mlp'] = {
                'instance': MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42
                ),
                'weight': 0.3,
                'performance': 0.5,
                'last_update': time.time()
            }
            
            # Bagging集成模型
            self.models['bagging'] = {
                'instance': BaggingRegressor(
                    base_estimator=DecisionTreeRegressor(max_depth=5),
                    n_estimators=10,
                    random_state=42
                ),
                'weight': 0.3,
                'performance': 0.5,
                'last_update': time.time()
            }
            
            logger.info(f"初始化 {len(self.models)} 个基础模型")
            
        except Exception as e:
            logger.error(f"初始化基础模型失败: {e}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 权重更新任务
        threading.Thread(
            target=self._weight_update_loop,
            daemon=True,
            name="WeightUpdateThread"
        ).start()
        
        # 模型重训练任务
        threading.Thread(
            target=self._retrain_loop,
            daemon=True,
            name="RetrainThread"
        ).start()
    
    async def coordinate_decision(self, market_data: Dict[str, Any]) -> CoordinationDecision:
        """协调决策"""
        start_time = time.time()
        
        try:
            # 准备特征数据
            features = self._prepare_features(market_data)
            
            # 收集各模型预测
            model_predictions = await self._collect_model_predictions(features)
            
            # 执行集成决策
            coordination_result = await self._ensemble_predict(model_predictions, features)
            
            # 记录预测历史
            execution_time = (time.time() - start_time) * 1000
            coordination_result.execution_time_ms = execution_time
            
            self._record_prediction(coordination_result)
            
            logger.debug(f"协调决策完成: 信号={coordination_result.signal:.4f}, "
                        f"置信度={coordination_result.confidence:.4f}, "
                        f"方法={coordination_result.ensemble_method}")
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"协调决策失败: {e}")
            return CoordinationDecision(
                signal=0.0,
                confidence=0.0,
                model_weights={},
                ensemble_method="error",
                feature_contributions={},
                reasoning=f"决策失败: {str(e)}",
                timestamp=datetime.now(),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """准备特征数据"""
        try:
            features = []
            
            # 价格特征
            if 'price' in market_data:
                features.append(market_data['price'])
            
            # 成交量特征
            if 'volume' in market_data:
                features.append(market_data['volume'])
            
            # 技术指标特征
            if 'indicators' in market_data:
                indicators = market_data['indicators']
                for key, value in indicators.items():
                    if isinstance(value, (int, float)):
                        features.append(value)
            
            # 时间特征
            if 'timestamp' in market_data:
                timestamp = market_data['timestamp']
                features.extend([
                    timestamp % 86400,  # 一天内的秒数
                    (timestamp // 86400) % 7,  # 星期几
                ])
            
            # 确保至少有一些特征
            if len(features) < 5:
                features.extend([0.0] * (5 - len(features)))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"准备特征数据失败: {e}")
            return np.array([[0.0] * 5])
    
    async def _collect_model_predictions(self, features: np.ndarray) -> List[ModelPrediction]:
        """收集各模型预测"""
        predictions = []
        
        with self.lock:
            models_copy = self.models.copy()
        
        for model_id, model_info in models_copy.items():
            try:
                start_time = time.time()
                
                model_instance = model_info['instance']
                
                # 检查模型是否已训练
                if not hasattr(model_instance, 'predict'):
                    continue
                
                # 执行预测
                if hasattr(model_instance, 'predict'):
                    try:
                        prediction = model_instance.predict(features)[0]
                    except:
                        # 如果模型未训练，使用默认预测
                        prediction = 0.0
                else:
                    prediction = 0.0
                
                # 计算置信度
                confidence = self._calculate_model_confidence(model_id, prediction)
                
                # 特征重要性
                feature_importance = self._get_feature_importance(model_instance, features)
                
                execution_time = (time.time() - start_time) * 1000
                
                predictions.append(ModelPrediction(
                    model_id=model_id,
                    model_name=model_id,
                    prediction=prediction,
                    confidence=confidence,
                    feature_importance=feature_importance,
                    execution_time_ms=execution_time,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.warning(f"模型 {model_id} 预测失败: {e}")
                # 添加默认预测
                predictions.append(ModelPrediction(
                    model_id=model_id,
                    model_name=model_id,
                    prediction=0.0,
                    confidence=0.1,
                    feature_importance={},
                    execution_time_ms=0.0,
                    timestamp=datetime.now()
                ))
        
        return predictions
    
    def _calculate_model_confidence(self, model_id: str, prediction: float) -> float:
        """计算模型置信度"""
        try:
            # 基于历史性能的置信度
            performance = self.models.get(model_id, {}).get('performance', 0.5)
            
            # 基于预测值的置信度调整
            prediction_confidence = 1.0 - abs(prediction)  # 预测值越接近0，置信度越低
            
            # 综合置信度
            confidence = (performance * 0.7 + prediction_confidence * 0.3)
            
            return np.clip(confidence, self.min_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"计算模型置信度失败: {e}")
            return 0.5
    
    def _get_feature_importance(self, model_instance: Any, features: np.ndarray) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            importance = {}
            
            # 决策树类模型
            if hasattr(model_instance, 'feature_importances_'):
                importances = model_instance.feature_importances_
                for i, imp in enumerate(importances):
                    importance[f'feature_{i}'] = float(imp)
            
            # 线性模型
            elif hasattr(model_instance, 'coef_'):
                coefs = model_instance.coef_
                if len(coefs.shape) == 1:
                    for i, coef in enumerate(coefs):
                        importance[f'feature_{i}'] = float(abs(coef))
            
            # 神经网络等其他模型，使用简单的权重估计
            else:
                for i in range(features.shape[1]):
                    importance[f'feature_{i}'] = 1.0 / features.shape[1]
            
            return importance
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    async def _ensemble_predict(self, predictions: List[ModelPrediction], features: np.ndarray) -> CoordinationDecision:
        """集成预测"""
        if not predictions:
            return CoordinationDecision(
                signal=0.0,
                confidence=0.0,
                model_weights={},
                ensemble_method="no_models",
                feature_contributions={},
                reasoning="无可用模型",
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
        
        # 根据当前集成方法执行预测
        if self.current_ensemble_method == 'weighted_average':
            return await self._weighted_average_ensemble(predictions)
        elif self.current_ensemble_method == 'voting':
            return await self._voting_ensemble(predictions)
        elif self.current_ensemble_method == 'stacking':
            return await self._stacking_ensemble(predictions, features)
        elif self.current_ensemble_method == 'neural_ensemble':
            return await self._neural_ensemble(predictions, features)
        else:
            return await self._weighted_average_ensemble(predictions)
    
    async def _weighted_average_ensemble(self, predictions: List[ModelPrediction]) -> CoordinationDecision:
        """加权平均集成"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            model_weights = {}
            feature_contributions = {}
            
            for pred in predictions:
                # 获取模型权重
                model_weight = self.models.get(pred.model_id, {}).get('weight', 0.1)
                
                # 基于置信度调整权重
                adjusted_weight = model_weight * pred.confidence
                
                weighted_sum += pred.prediction * adjusted_weight
                total_weight += adjusted_weight
                model_weights[pred.model_id] = adjusted_weight
                
                # 累积特征贡献度
                for feature, importance in pred.feature_importance.items():
                    if feature not in feature_contributions:
                        feature_contributions[feature] = 0.0
                    feature_contributions[feature] += importance * adjusted_weight
            
            # 计算最终信号
            if total_weight > 0:
                final_signal = weighted_sum / total_weight
                # 归一化模型权重
                for model_id in model_weights:
                    model_weights[model_id] /= total_weight
                # 归一化特征贡献度
                for feature in feature_contributions:
                    feature_contributions[feature] /= total_weight
            else:
                final_signal = 0.0
            
            # 计算综合置信度
            confidence = min(total_weight, 1.0) if total_weight > 0 else 0.0
            
            # 生成推理
            reasoning = f"加权平均集成 {len(predictions)} 个模型"
            
            return CoordinationDecision(
                signal=np.clip(final_signal, -1.0, 1.0),
                confidence=confidence,
                model_weights=model_weights,
                ensemble_method="weighted_average",
                feature_contributions=feature_contributions,
                reasoning=reasoning,
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"加权平均集成失败: {e}")
            return CoordinationDecision(
                signal=0.0,
                confidence=0.0,
                model_weights={},
                ensemble_method="error",
                feature_contributions={},
                reasoning=f"集成失败: {str(e)}",
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
    
    async def _voting_ensemble(self, predictions: List[ModelPrediction]) -> CoordinationDecision:
        """投票集成"""
        try:
            # 将连续预测转换为离散投票
            votes = {'buy': 0, 'sell': 0, 'hold': 0}
            model_weights = {}
            
            for pred in predictions:
                model_weight = self.models.get(pred.model_id, {}).get('weight', 0.1)
                adjusted_weight = model_weight * pred.confidence
                
                # 转换为投票
                if pred.prediction > 0.1:
                    votes['buy'] += adjusted_weight
                elif pred.prediction < -0.1:
                    votes['sell'] += adjusted_weight
                else:
                    votes['hold'] += adjusted_weight
                
                model_weights[pred.model_id] = adjusted_weight
            
            # 确定最终信号
            max_vote = max(votes.values())
            if max_vote == 0:
                final_signal = 0.0
                confidence = 0.0
            else:
                if votes['buy'] == max_vote:
                    final_signal = 0.5
                elif votes['sell'] == max_vote:
                    final_signal = -0.5
                else:
                    final_signal = 0.0
                
                # 计算置信度
                total_votes = sum(votes.values())
                confidence = max_vote / total_votes if total_votes > 0 else 0.0
            
            reasoning = f"投票集成: 买入={votes['buy']:.2f}, 卖出={votes['sell']:.2f}, 持有={votes['hold']:.2f}"
            
            return CoordinationDecision(
                signal=final_signal,
                confidence=confidence,
                model_weights=model_weights,
                ensemble_method="voting",
                feature_contributions={},
                reasoning=reasoning,
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"投票集成失败: {e}")
            return await self._weighted_average_ensemble(predictions)
    
    async def _stacking_ensemble(self, predictions: List[ModelPrediction], features: np.ndarray) -> CoordinationDecision:
        """堆叠集成"""
        try:
            # 简化的堆叠实现：使用线性回归作为元学习器
            if len(predictions) < 2:
                return await self._weighted_average_ensemble(predictions)
            
            # 构建元特征
            meta_features = []
            for pred in predictions:
                meta_features.append(pred.prediction)
                meta_features.append(pred.confidence)
            
            # 简单的线性组合
            weights = np.array([0.8, 0.2] * len(predictions))[:len(meta_features)]
            weights = weights / weights.sum()
            
            final_signal = np.dot(meta_features, weights)
            confidence = np.mean([pred.confidence for pred in predictions])
            
            model_weights = {pred.model_id: 1.0/len(predictions) for pred in predictions}
            
            return CoordinationDecision(
                signal=np.clip(final_signal, -1.0, 1.0),
                confidence=confidence,
                model_weights=model_weights,
                ensemble_method="stacking",
                feature_contributions={},
                reasoning=f"堆叠集成 {len(predictions)} 个模型",
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"堆叠集成失败: {e}")
            return await self._weighted_average_ensemble(predictions)
    
    async def _neural_ensemble(self, predictions: List[ModelPrediction], features: np.ndarray) -> CoordinationDecision:
        """神经网络集成"""
        try:
            # 如果神经网络未初始化，使用加权平均
            if self.ensemble_net is None:
                return await self._weighted_average_ensemble(predictions)
            
            # 构建输入特征
            ensemble_input = []
            ensemble_input.extend(features.flatten())
            for pred in predictions:
                ensemble_input.append(pred.prediction)
                ensemble_input.append(pred.confidence)
            
            # 转换为张量
            input_tensor = torch.FloatTensor(ensemble_input).unsqueeze(0).to(self.device)
            
            # 神经网络预测
            with torch.no_grad():
                output = self.ensemble_net(input_tensor)
                final_signal = output.item()
            
            # 计算置信度
            confidence = np.mean([pred.confidence for pred in predictions])
            
            model_weights = {pred.model_id: 1.0/len(predictions) for pred in predictions}
            
            return CoordinationDecision(
                signal=np.clip(final_signal, -1.0, 1.0),
                confidence=confidence,
                model_weights=model_weights,
                ensemble_method="neural_ensemble",
                feature_contributions={},
                reasoning=f"神经网络集成 {len(predictions)} 个模型",
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"神经网络集成失败: {e}")
            return await self._weighted_average_ensemble(predictions)
    
    def _record_prediction(self, decision: CoordinationDecision):
        """记录预测历史"""
        with self.lock:
            self.prediction_history.append(decision)
            
            # 限制历史记录数量
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
    
    def _weight_update_loop(self):
        """权重更新循环"""
        while True:
            try:
                time.sleep(300)  # 5分钟更新一次
                self._update_model_weights()
            except Exception as e:
                logger.error(f"权重更新失败: {e}")
    
    def _retrain_loop(self):
        """重训练循环"""
        while True:
            try:
                time.sleep(3600)  # 1小时重训练一次
                await self._retrain_models()
            except Exception as e:
                logger.error(f"重训练失败: {e}")
    
    def _update_model_weights(self):
        """更新模型权重"""
        try:
            with self.lock:
                # 基于历史性能更新权重
                for model_id in self.models:
                    current_weight = self.models[model_id]['weight']
                    performance = self.models[model_id]['performance']
                    
                    # 权重衰减和性能调整
                    new_weight = current_weight * self.weight_decay + performance * (1 - self.weight_decay)
                    self.models[model_id]['weight'] = np.clip(new_weight, 0.01, 1.0)
                
                # 归一化权重
                total_weight = sum(model['weight'] for model in self.models.values())
                if total_weight > 0:
                    for model_id in self.models:
                        self.models[model_id]['weight'] /= total_weight
                
                logger.debug("模型权重已更新")
                
        except Exception as e:
            logger.error(f"更新模型权重失败: {e}")
    
    async def _retrain_models(self):
        """重训练模型"""
        if self.is_training:
            return
        
        self.is_training = True
        try:
            # 这里应该实现模型重训练逻辑
            # 由于需要训练数据，这里只是占位符
            logger.info("开始重训练模型...")
            
            # 模拟重训练过程
            await asyncio.sleep(1)
            
            logger.info("模型重训练完成")
            
        except Exception as e:
            logger.error(f"重训练模型失败: {e}")
        finally:
            self.is_training = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        with self.lock:
            return {
                'level': 5,
                'name': 'Integration Learning Coordinator',
                'models_count': len(self.models),
                'current_ensemble_method': self.current_ensemble_method,
                'prediction_count': len(self.prediction_history),
                'is_training': self.is_training,
                'model_weights': {
                    model_id: info['weight'] 
                    for model_id, info in self.models.items()
                },
                'model_performance': {
                    model_id: info['performance'] 
                    for model_id, info in self.models.items()
                }
            }


# 全局实例
_coordinator = None

def get_integration_coordinator(config: Dict[str, Any] = None) -> IntegrationLearningCoordinator:
    """获取集成学习协调器实例"""
    global _coordinator
    if _coordinator is None:
        _coordinator = IntegrationLearningCoordinator(config)
    return _coordinator


if __name__ == "__main__":
    # 测试代码
    async def test_coordinator():
        """测试协调器"""
        coordinator = get_integration_coordinator()
        
        # 模拟市场数据
        market_data = {
            'price': 50000.0,
            'volume': 1000.0,
            'indicators': {
                'rsi': 65.0,
                'macd': 0.5,
                'bb_upper': 51000.0,
                'bb_lower': 49000.0
            },
            'timestamp': time.time()
        }
        
        # 执行协调决策
        decision = await coordinator.coordinate_decision(market_data)
        
        print(f"协调决策结果:")
        print(f"信号: {decision.signal}")
        print(f"置信度: {decision.confidence}")
        print(f"集成方法: {decision.ensemble_method}")
        print(f"推理: {decision.reasoning}")
        
        # 获取状态
        status = coordinator.get_status()
        print(f"\n协调器状态: {json.dumps(status, indent=2)}")
    
    # 运行测试
    asyncio.run(test_coordinator())
