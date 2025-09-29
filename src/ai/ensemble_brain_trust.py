#!/usr/bin/env python3
"""
🧠 集成学习智囊团 - 多模型投票决策
使用多种机器学习算法进行集成预测
专为生产级实盘交易设计，支持动态权重调整
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os

@dataclass
class ModelPrediction:
    """单个模型预测结果"""
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    training_score: float
    validation_score: float
    prediction_time: float

@dataclass
class EnsemblePrediction:
    """集成预测结果"""
    final_prediction: float
    weighted_prediction: float
    voting_prediction: float
    confidence_score: float
    model_predictions: List[ModelPrediction]
    model_weights: Dict[str, float]
    consensus_level: float
    uncertainty_score: float
    timestamp: datetime

class BaseMLModel:
    """机器学习模型基类"""
    
    def __init__(self, name: str, model, scaler=None):
        self.name = name
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.is_trained = False
        self.training_score = 0.0
        self.validation_score = 0.0
        self.feature_names = []
        self.feature_importance = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """训练模型"""
        try:
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # 计算训练分数
            train_pred = self.model.predict(X_scaled)
            self.training_score = r2_score(y, train_pred)
            
            # 交叉验证分数
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
            self.validation_score = np.mean(cv_scores)
            
            # 特征重要性
            if feature_names:
                self.feature_names = feature_names
                if hasattr(self.model, 'feature_importances_'):
                    self.feature_importance = dict(zip(
                        feature_names, self.model.feature_importances_
                    ))
                elif hasattr(self.model, 'coef_'):
                    self.feature_importance = dict(zip(
                        feature_names, np.abs(self.model.coef_)
                    ))
            
            logger.debug(f"✅ {self.name} 训练完成 - R²: {self.validation_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.name} 训练失败: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """预测"""
        try:
            if not self.is_trained:
                return 0.0, 0.0
            
            start_time = time.time()
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            prediction = float(self.model.predict(X_scaled)[0])
            prediction_time = time.time() - start_time
            
            # 计算置信度（基于验证分数）
            confidence = max(0.0, min(1.0, self.validation_score))
            
            return prediction, confidence, prediction_time
            
        except Exception as e:
            logger.error(f"❌ {self.name} 预测失败: {e}")
            return 0.0, 0.0, 0.0

class EnsembleBrainTrust:
    """集成学习智囊团"""
    
    def __init__(self, device: str = None, model_dir: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir or "models/ensemble"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化模型集合
        self.models = self._initialize_models()
        
        # 动态权重
        self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        self.weight_decay = 0.95  # 权重衰减因子
        self.min_weight = 0.01   # 最小权重
        
        # 性能追踪
        self.performance_history = {name: [] for name in self.models.keys()}
        self.prediction_history = []
        self.max_history = 1000
        
        # 训练参数
        self.retrain_interval = 100  # 重训练间隔
        self.prediction_count = 0
        self.last_retrain = 0
        
        # 特征工程
        self.feature_names = [
            'price', 'volume', 'rsi', 'macd', 'bb_position', 'atr',
            'ema_12', 'ema_26', 'sma_50', 'volume_sma', 'price_change',
            'volatility', 'sentiment', 'news_impact', 'time_of_day',
            'day_of_week', 'support_level', 'resistance_level', 'trend_strength',
            'momentum', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
            'roc', 'trix', 'dmi_plus', 'dmi_minus', 'adx'
        ]
        
        # 数据缓存
        self.training_data = {'X': [], 'y': []}
        self.max_training_samples = 5000
        
        # 实时状态
        self.last_prediction = None
        self.last_confidence = 0.0
        self.performance_score = 0.5
        
        # 加载预训练模型
        self.load_models()
        
        logger.info(f"🧠 集成学习智囊团初始化完成 - {len(self.models)}个模型")
    
    def _initialize_models(self) -> Dict[str, BaseMLModel]:
        """初始化模型集合"""
        models = {}
        
        # 随机森林
        models['random_forest'] = BaseMLModel(
            'RandomForest',
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            RobustScaler()
        )
        
        # 梯度提升
        models['gradient_boosting'] = BaseMLModel(
            'GradientBoosting',
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=42
            ),
            StandardScaler()
        )
        
        # XGBoost
        models['xgboost'] = BaseMLModel(
            'XGBoost',
            xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            StandardScaler()
        )
        
        # LightGBM
        models['lightgbm'] = BaseMLModel(
            'LightGBM',
            lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            StandardScaler()
        )
        
        # CatBoost
        models['catboost'] = BaseMLModel(
            'CatBoost',
            cb.CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=8,
                subsample=0.8,
                random_state=42,
                verbose=False
            ),
            StandardScaler()
        )
        
        # 支持向量回归
        models['svr'] = BaseMLModel(
            'SVR',
            SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.01
            ),
            StandardScaler()
        )
        
        # 神经网络
        models['mlp'] = BaseMLModel(
            'MLP',
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            StandardScaler()
        )
        
        # 线性模型集合
        models['ridge'] = BaseMLModel(
            'Ridge',
            Ridge(alpha=1.0, random_state=42),
            StandardScaler()
        )
        
        models['lasso'] = BaseMLModel(
            'Lasso',
            Lasso(alpha=0.1, random_state=42),
            StandardScaler()
        )
        
        models['elastic_net'] = BaseMLModel(
            'ElasticNet',
            ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            StandardScaler()
        )
        
        return models
    
    async def get_ensemble_prediction(self, market_data: Dict[str, Any]) -> EnsemblePrediction:
        """获取集成预测"""
        try:
            # 准备特征
            features = self._prepare_features(market_data)
            if features is None:
                return self._create_fallback_prediction(market_data)
            
            # 并行预测
            model_predictions = []
            prediction_tasks = []
            
            for name, model in self.models.items():
                if model.is_trained:
                    task = asyncio.create_task(
                        self._async_model_predict(name, model, features)
                    )
                    prediction_tasks.append(task)
            
            # 等待所有预测完成
            if prediction_tasks:
                results = await asyncio.gather(*prediction_tasks, return_exceptions=True)
                model_predictions = [r for r in results if not isinstance(r, Exception)]
            
            if not model_predictions:
                return self._create_fallback_prediction(market_data)
            
            # 计算集成预测
            weighted_pred = self._calculate_weighted_prediction(model_predictions)
            voting_pred = self._calculate_voting_prediction(model_predictions)
            final_pred = (weighted_pred + voting_pred) / 2.0
            
            # 计算置信度和一致性
            confidence_score = self._calculate_ensemble_confidence(model_predictions)
            consensus_level = self._calculate_consensus_level(model_predictions)
            uncertainty_score = self._calculate_uncertainty_score(model_predictions)
            
            # 创建集成预测结果
            ensemble_pred = EnsemblePrediction(
                final_prediction=final_pred,
                weighted_prediction=weighted_pred,
                voting_prediction=voting_pred,
                confidence_score=confidence_score,
                model_predictions=model_predictions,
                model_weights=self.model_weights.copy(),
                consensus_level=consensus_level,
                uncertainty_score=uncertainty_score,
                timestamp=datetime.now(timezone.utc)
            )
            
            # 更新状态
            self.last_prediction = ensemble_pred
            self.last_confidence = confidence_score
            self.prediction_count += 1
            
            # 记录预测历史
            self.prediction_history.append({
                'timestamp': ensemble_pred.timestamp,
                'prediction': final_pred,
                'confidence': confidence_score,
                'consensus': consensus_level
            })
            
            # 限制历史记录
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]
            
            # 检查是否需要重训练
            if self.prediction_count - self.last_retrain >= self.retrain_interval:
                asyncio.create_task(self._async_retrain())
            
            logger.info(f"🧠 集成预测完成 - 预测: {final_pred:.6f}, 置信度: {confidence_score:.3f}")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"❌ 集成预测失败: {e}")
            return self._create_fallback_prediction(market_data)
    
    async def _async_model_predict(self, name: str, model: BaseMLModel, 
                                 features: np.ndarray) -> ModelPrediction:
        """异步模型预测"""
        try:
            prediction, confidence, pred_time = model.predict(features)
            
            return ModelPrediction(
                model_name=name,
                prediction=prediction,
                confidence=confidence,
                feature_importance=model.feature_importance.copy(),
                training_score=model.training_score,
                validation_score=model.validation_score,
                prediction_time=pred_time
            )
            
        except Exception as e:
            logger.error(f"❌ {name} 异步预测失败: {e}")
            return ModelPrediction(
                model_name=name,
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                training_score=0.0,
                validation_score=0.0,
                prediction_time=0.0
            )
    
    def _prepare_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """准备特征向量"""
        try:
            features = []
            
            # 基础市场数据
            features.extend([
                market_data.get('price', 0.0),
                market_data.get('volume', 0.0),
                market_data.get('rsi', 50.0),
                market_data.get('macd', 0.0),
                market_data.get('bb_position', 0.5),
                market_data.get('atr', 0.0),
                market_data.get('ema_12', 0.0),
                market_data.get('ema_26', 0.0),
                market_data.get('sma_50', 0.0),
                market_data.get('volume_sma', 0.0),
                market_data.get('price_change', 0.0),
                market_data.get('volatility', 0.0),
                market_data.get('sentiment', 0.0),
                market_data.get('news_impact', 0.0),
                market_data.get('time_of_day', 0.5),
                market_data.get('day_of_week', 0.5),
                market_data.get('support_level', 0.0),
                market_data.get('resistance_level', 0.0),
                market_data.get('trend_strength', 0.0)
            ])
            
            # 扩展技术指标
            features.extend([
                market_data.get('momentum', 0.0),
                market_data.get('stoch_k', 50.0),
                market_data.get('stoch_d', 50.0),
                market_data.get('williams_r', -50.0),
                market_data.get('cci', 0.0),
                market_data.get('roc', 0.0),
                market_data.get('trix', 0.0),
                market_data.get('dmi_plus', 25.0),
                market_data.get('dmi_minus', 25.0),
                market_data.get('adx', 25.0)
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ 特征准备失败: {e}")
            return None
    
    def _calculate_weighted_prediction(self, predictions: List[ModelPrediction]) -> float:
        """计算加权预测"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for pred in predictions:
                weight = self.model_weights.get(pred.model_name, 0.0)
                confidence_weight = pred.confidence * weight
                weighted_sum += pred.prediction * confidence_weight
                total_weight += confidence_weight
            
            return weighted_sum / max(total_weight, 1e-8)
            
        except Exception as e:
            logger.error(f"❌ 加权预测计算失败: {e}")
            return 0.0
    
    def _calculate_voting_prediction(self, predictions: List[ModelPrediction]) -> float:
        """计算投票预测"""
        try:
            # 简单平均
            if not predictions:
                return 0.0
            
            return np.mean([pred.prediction for pred in predictions])
            
        except Exception as e:
            logger.error(f"❌ 投票预测计算失败: {e}")
            return 0.0
    
    def _calculate_ensemble_confidence(self, predictions: List[ModelPrediction]) -> float:
        """计算集成置信度"""
        try:
            if not predictions:
                return 0.0
            
            # 基于模型置信度的加权平均
            confidences = [pred.confidence for pred in predictions]
            weights = [self.model_weights.get(pred.model_name, 0.0) for pred in predictions]
            
            weighted_confidence = np.average(confidences, weights=weights)
            
            # 考虑模型一致性
            pred_values = [pred.prediction for pred in predictions]
            consistency = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
            consistency = max(0.0, min(1.0, consistency))
            
            return (weighted_confidence + consistency) / 2.0
            
        except Exception as e:
            logger.error(f"❌ 集成置信度计算失败: {e}")
            return 0.0
    
    def _calculate_consensus_level(self, predictions: List[ModelPrediction]) -> float:
        """计算一致性水平"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            pred_values = [pred.prediction for pred in predictions]
            mean_pred = np.mean(pred_values)
            
            # 计算相对标准差
            if abs(mean_pred) > 1e-8:
                cv = np.std(pred_values) / abs(mean_pred)
                consensus = max(0.0, 1.0 - cv)
            else:
                consensus = 1.0 - np.std(pred_values)
            
            return max(0.0, min(1.0, consensus))
            
        except Exception as e:
            logger.error(f"❌ 一致性计算失败: {e}")
            return 0.0
    
    def _calculate_uncertainty_score(self, predictions: List[ModelPrediction]) -> float:
        """计算不确定性分数"""
        try:
            if not predictions:
                return 1.0
            
            # 基于预测方差和置信度方差
            pred_values = [pred.prediction for pred in predictions]
            confidences = [pred.confidence for pred in predictions]
            
            pred_uncertainty = np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
            conf_uncertainty = np.std(confidences)
            
            uncertainty = (pred_uncertainty + conf_uncertainty) / 2.0
            return max(0.0, min(1.0, uncertainty))
            
        except Exception as e:
            logger.error(f"❌ 不确定性计算失败: {e}")
            return 0.5
    
    def _create_fallback_prediction(self, market_data: Dict[str, Any]) -> EnsemblePrediction:
        """创建后备预测"""
        return EnsemblePrediction(
            final_prediction=0.0,
            weighted_prediction=0.0,
            voting_prediction=0.0,
            confidence_score=0.1,
            model_predictions=[],
            model_weights={},
            consensus_level=0.0,
            uncertainty_score=1.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def train_models(self, training_data: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """训练所有模型"""
        try:
            X = training_data['X']
            y = training_data['y']
            
            if len(X) < 100:  # 最少需要100个样本
                logger.warning("⚠️ 训练数据不足")
                return {}
            
            # 并行训练
            training_tasks = []
            for name, model in self.models.items():
                task = asyncio.create_task(
                    self._async_train_model(name, model, X, y)
                )
                training_tasks.append((name, task))
            
            # 等待训练完成
            results = {}
            for name, task in training_tasks:
                try:
                    success = await task
                    results[name] = success
                    
                    # 更新权重
                    if success:
                        model = self.models[name]
                        performance = model.validation_score
                        self.performance_history[name].append(performance)
                        
                        # 限制历史长度
                        if len(self.performance_history[name]) > 100:
                            self.performance_history[name] = self.performance_history[name][-100:]
                        
                        # 更新权重
                        self._update_model_weight(name, performance)
                    
                except Exception as e:
                    logger.error(f"❌ {name} 训练任务失败: {e}")
                    results[name] = False
            
            # 标准化权重
            self._normalize_weights()
            
            self.last_retrain = self.prediction_count
            logger.info(f"🧠 模型训练完成 - 成功: {sum(results.values())}/{len(results)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            return {}
    
    async def _async_train_model(self, name: str, model: BaseMLModel, 
                               X: np.ndarray, y: np.ndarray) -> bool:
        """异步训练单个模型"""
        try:
            return model.fit(X, y, self.feature_names)
        except Exception as e:
            logger.error(f"❌ {name} 异步训练失败: {e}")
            return False
    
    def _update_model_weight(self, name: str, performance: float):
        """更新模型权重"""
        try:
            # 基于性能调整权重
            if performance > 0.5:  # 好的性能增加权重
                self.model_weights[name] *= (1.0 + performance * 0.1)
            else:  # 差的性能减少权重
                self.model_weights[name] *= self.weight_decay
            
            # 确保最小权重
            self.model_weights[name] = max(self.model_weights[name], self.min_weight)
            
        except Exception as e:
            logger.error(f"❌ 权重更新失败: {e}")
    
    def _normalize_weights(self):
        """标准化权重"""
        try:
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for name in self.model_weights:
                    self.model_weights[name] /= total_weight
        except Exception as e:
            logger.error(f"❌ 权重标准化失败: {e}")
    
    async def _async_retrain(self):
        """异步重训练"""
        try:
            if len(self.training_data['X']) < 100:
                return
            
            X = np.array(self.training_data['X'])
            y = np.array(self.training_data['y'])
            
            await self.train_models({'X': X, 'y': y})
            
        except Exception as e:
            logger.error(f"❌ 异步重训练失败: {e}")
    
    def add_training_sample(self, features: np.ndarray, target: float):
        """添加训练样本"""
        try:
            self.training_data['X'].append(features)
            self.training_data['y'].append(target)
            
            # 限制训练数据大小
            if len(self.training_data['X']) > self.max_training_samples:
                self.training_data['X'] = self.training_data['X'][-self.max_training_samples:]
                self.training_data['y'] = self.training_data['y'][-self.max_training_samples:]
                
        except Exception as e:
            logger.error(f"❌ 添加训练样本失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        try:
            model_status = {}
            for name, model in self.models.items():
                model_status[name] = {
                    'is_trained': model.is_trained,
                    'training_score': model.training_score,
                    'validation_score': model.validation_score,
                    'weight': self.model_weights.get(name, 0.0)
                }
            
            return {
                'model_id': 'ensemble_brain_trust',
                'model_name': '集成学习智囊团',
                'total_models': len(self.models),
                'trained_models': sum(1 for m in self.models.values() if m.is_trained),
                'prediction_count': self.prediction_count,
                'last_confidence': self.last_confidence,
                'performance_score': self.performance_score,
                'model_weights': self.model_weights.copy(),
                'model_status': model_status,
                'training_samples': len(self.training_data['X']),
                'prediction_history_length': len(self.prediction_history)
            }
            
        except Exception as e:
            logger.error(f"❌ 状态获取失败: {e}")
            return {'error': str(e)}
    
    def save_models(self) -> bool:
        """保存所有模型"""
        try:
            for name, model in self.models.items():
                if model.is_trained:
                    model_path = os.path.join(self.model_dir, f"{name}.joblib")
                    scaler_path = os.path.join(self.model_dir, f"{name}_scaler.joblib")
                    
                    joblib.dump(model.model, model_path)
                    joblib.dump(model.scaler, scaler_path)
            
            # 保存权重和配置
            config_path = os.path.join(self.model_dir, "ensemble_config.json")
            config = {
                'model_weights': self.model_weights,
                'performance_history': self.performance_history,
                'feature_names': self.feature_names
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 集成模型已保存到: {self.model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """加载所有模型"""
        try:
            config_path = os.path.join(self.model_dir, "ensemble_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.model_weights = config.get('model_weights', self.model_weights)
                self.performance_history = config.get('performance_history', self.performance_history)
                self.feature_names = config.get('feature_names', self.feature_names)
            
            # 加载各个模型
            loaded_count = 0
            for name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{name}_scaler.joblib")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        model.model = joblib.load(model_path)
                        model.scaler = joblib.load(scaler_path)
                        model.is_trained = True
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ {name} 模型加载失败: {e}")
            
            if loaded_count > 0:
                logger.info(f"📂 已加载 {loaded_count} 个集成模型")
                return True
            else:
                logger.info("📂 未找到预训练的集成模型")
                return False
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False

# 全局实例
ensemble_brain_trust = EnsembleBrainTrust()

def initialize_ensemble_brain_trust(device: str = None, model_dir: str = None) -> EnsembleBrainTrust:
    """初始化集成学习智囊团"""
    global ensemble_brain_trust
    ensemble_brain_trust = EnsembleBrainTrust(device, model_dir)
    return ensemble_brain_trust

if __name__ == "__main__":
    # 测试代码
    async def test_ensemble():
        ensemble = initialize_ensemble_brain_trust()
        
        # 测试预测
        market_data = {
            'price': 50000.0,
            'volume': 1000000.0,
            'rsi': 65.0,
            'macd': 0.1,
            'bb_position': 0.7,
            'atr': 500.0,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.3,
            'news_impact': 0.1
        }
        
        prediction = await ensemble.get_ensemble_prediction(market_data)
        print(f"集成预测: {prediction}")
        
        # 状态报告
        status = ensemble.get_status()
        print(f"状态报告: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    asyncio.run(test_ensemble())
