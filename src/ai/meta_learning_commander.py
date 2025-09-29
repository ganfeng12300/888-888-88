#!/usr/bin/env python3
"""
🧠 元学习AI指挥官 - 决策协调核心
负责协调8大AI模型，进行元学习和决策融合
专为生产级实盘交易设计，支持GPU加速训练
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil

@dataclass
class AIModelStatus:
    """AI模型状态"""
    model_id: str
    model_name: str
    confidence: float
    prediction: float
    last_update: datetime
    performance_score: float
    training_progress: float
    gpu_usage: float
    memory_usage: float
    is_active: bool

@dataclass
class MetaDecision:
    """元决策结果"""
    final_signal: float  # -1到1之间的最终信号
    confidence: float    # 0到1之间的置信度
    contributing_models: List[str]  # 参与决策的模型
    model_weights: Dict[str, float]  # 各模型权重
    risk_assessment: float  # 风险评估
    timestamp: datetime
    reasoning: str  # 决策推理

class MetaLearningNetwork(nn.Module):
    """元学习神经网络"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 注意力机制层
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 决策融合网络
        self.decision_fusion = nn.Sequential(
            nn.Linear(output_dim * 8, hidden_dim),  # 8个AI模型
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出-1到1之间的信号
        )
        
        # 置信度网络
        self.confidence_network = nn.Sequential(
            nn.Linear(output_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0到1之间的置信度
        )
        
    def forward(self, model_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            model_features: [batch_size, num_models, feature_dim]
        Returns:
            decision: 决策信号
            confidence: 置信度
        """
        batch_size, num_models, feature_dim = model_features.shape
        
        # 注意力机制
        attended_features, attention_weights = self.attention(
            model_features, model_features, model_features
        )
        
        # 特征提取
        extracted_features = self.feature_extractor(attended_features)
        
        # 展平特征
        flattened_features = extracted_features.view(batch_size, -1)
        
        # 决策融合
        decision = self.decision_fusion(flattened_features)
        
        # 置信度计算
        confidence = self.confidence_network(flattened_features)
        
        return decision.squeeze(-1), confidence.squeeze(-1)

class MetaLearningCommander:
    """元学习AI指挥官"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_statuses: Dict[str, AIModelStatus] = {}
        self.decision_history: List[MetaDecision] = []
        self.performance_metrics = {
            'total_decisions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # 初始化元学习网络
        self.meta_network = MetaLearningNetwork().to(self.device)
        self.optimizer = optim.AdamW(
            self.meta_network.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
        # 模型权重（动态调整）
        self.model_weights = {
            'reinforcement_trader': 0.20,
            'time_series_prophet': 0.18,
            'ensemble_brain_trust': 0.15,
            'transfer_learning_adapter': 0.12,
            'expert_system_guardian': 0.10,
            'sentiment_scout': 0.10,
            'factor_mining_engine': 0.10,
            'meta_learning_commander': 0.05
        }
        
        # 训练参数
        self.training_enabled = True
        self.batch_size = 32
        self.sequence_length = 100
        self.training_buffer = []
        self.max_buffer_size = 10000
        
        # 性能监控
        self.last_gpu_check = time.time()
        self.gpu_temperature = 0.0
        self.gpu_utilization = 0.0
        self.memory_usage = 0.0
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"🧠 元学习AI指挥官初始化完成 - 设备: {self.device}")
        
    def register_ai_model(self, model_id: str, model_name: str) -> bool:
        """注册AI模型"""
        try:
            self.model_statuses[model_id] = AIModelStatus(
                model_id=model_id,
                model_name=model_name,
                confidence=0.0,
                prediction=0.0,
                last_update=datetime.now(timezone.utc),
                performance_score=0.5,
                training_progress=0.0,
                gpu_usage=0.0,
                memory_usage=0.0,
                is_active=True
            )
            logger.info(f"✅ AI模型已注册: {model_name} ({model_id})")
            return True
        except Exception as e:
            logger.error(f"❌ AI模型注册失败: {e}")
            return False
    
    def update_model_status(self, model_id: str, prediction: float, 
                          confidence: float, performance_score: float = None) -> bool:
        """更新AI模型状态"""
        try:
            if model_id not in self.model_statuses:
                logger.warning(f"⚠️ 未知的AI模型ID: {model_id}")
                return False
            
            status = self.model_statuses[model_id]
            status.prediction = prediction
            status.confidence = confidence
            status.last_update = datetime.now(timezone.utc)
            
            if performance_score is not None:
                status.performance_score = performance_score
            
            # 更新GPU和内存使用情况
            self._update_hardware_metrics(model_id)
            
            return True
        except Exception as e:
            logger.error(f"❌ 更新模型状态失败: {e}")
            return False
    
    def _update_hardware_metrics(self, model_id: str):
        """更新硬件指标"""
        try:
            current_time = time.time()
            if current_time - self.last_gpu_check > 5:  # 每5秒检查一次
                # GPU监控
                if torch.cuda.is_available():
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.gpu_temperature = gpu.temperature
                        self.gpu_utilization = gpu.load * 100
                        self.memory_usage = gpu.memoryUtil * 100
                
                # 更新所有模型的硬件指标
                for status in self.model_statuses.values():
                    status.gpu_usage = self.gpu_utilization
                    status.memory_usage = self.memory_usage
                
                self.last_gpu_check = current_time
                
        except Exception as e:
            logger.warning(f"⚠️ 硬件指标更新失败: {e}")
    
    async def make_meta_decision(self, market_data: Dict[str, Any]) -> MetaDecision:
        """制作元决策"""
        try:
            # 收集所有活跃模型的预测
            active_models = {
                k: v for k, v in self.model_statuses.items() 
                if v.is_active and (datetime.now(timezone.utc) - v.last_update).seconds < 60
            }
            
            if len(active_models) < 3:
                logger.warning("⚠️ 活跃模型数量不足，使用保守决策")
                return self._make_conservative_decision()
            
            # 准备模型特征
            model_features = self._prepare_model_features(active_models, market_data)
            
            # GPU推理
            with torch.no_grad():
                features_tensor = torch.FloatTensor(model_features).unsqueeze(0).to(self.device)
                decision_signal, confidence = self.meta_network(features_tensor)
                
                final_signal = float(decision_signal.cpu().numpy()[0])
                final_confidence = float(confidence.cpu().numpy()[0])
            
            # 计算模型权重
            model_weights = self._calculate_dynamic_weights(active_models)
            
            # 风险评估
            risk_assessment = self._assess_risk(active_models, market_data)
            
            # 创建元决策
            meta_decision = MetaDecision(
                final_signal=final_signal,
                confidence=final_confidence,
                contributing_models=list(active_models.keys()),
                model_weights=model_weights,
                risk_assessment=risk_assessment,
                timestamp=datetime.now(timezone.utc),
                reasoning=self._generate_reasoning(active_models, final_signal, final_confidence)
            )
            
            # 记录决策历史
            self.decision_history.append(meta_decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            # 异步训练
            if self.training_enabled:
                asyncio.create_task(self._async_training_update(model_features, final_signal))
            
            # 更新性能指标
            self._update_performance_metrics()
            
            logger.info(f"🧠 元决策完成 - 信号: {final_signal:.4f}, 置信度: {final_confidence:.4f}")
            return meta_decision
            
        except Exception as e:
            logger.error(f"❌ 元决策制作失败: {e}")
            return self._make_conservative_decision()
    
    def _prepare_model_features(self, active_models: Dict[str, AIModelStatus], 
                              market_data: Dict[str, Any]) -> np.ndarray:
        """准备模型特征"""
        features = []
        
        for model_id, status in active_models.items():
            # 基础特征
            model_feature = [
                status.prediction,
                status.confidence,
                status.performance_score,
                status.training_progress,
                status.gpu_usage / 100.0,
                status.memory_usage / 100.0,
                float(status.is_active),
                (datetime.now(timezone.utc) - status.last_update).seconds / 60.0
            ]
            
            # 市场特征
            if market_data:
                model_feature.extend([
                    market_data.get('price_change', 0.0),
                    market_data.get('volume_ratio', 1.0),
                    market_data.get('volatility', 0.0),
                    market_data.get('rsi', 50.0) / 100.0,
                    market_data.get('macd_signal', 0.0),
                    market_data.get('bb_position', 0.5),
                    market_data.get('sentiment_score', 0.0),
                    market_data.get('news_impact', 0.0)
                ])
            else:
                model_feature.extend([0.0] * 8)
            
            # 填充到64维
            while len(model_feature) < 64:
                model_feature.append(0.0)
            
            features.append(model_feature[:64])
        
        # 确保有8个模型的特征（不足的用零填充）
        while len(features) < 8:
            features.append([0.0] * 64)
        
        return np.array(features[:8])
    
    def _calculate_dynamic_weights(self, active_models: Dict[str, AIModelStatus]) -> Dict[str, float]:
        """计算动态权重"""
        weights = {}
        total_score = sum(status.performance_score * status.confidence for status in active_models.values())
        
        if total_score > 0:
            for model_id, status in active_models.items():
                weights[model_id] = (status.performance_score * status.confidence) / total_score
        else:
            # 均等权重
            weight = 1.0 / len(active_models)
            for model_id in active_models:
                weights[model_id] = weight
        
        return weights
    
    def _assess_risk(self, active_models: Dict[str, AIModelStatus], 
                    market_data: Dict[str, Any]) -> float:
        """评估风险"""
        try:
            risk_factors = []
            
            # 模型一致性风险
            predictions = [status.prediction for status in active_models.values()]
            if predictions:
                prediction_std = np.std(predictions)
                risk_factors.append(prediction_std)
            
            # 置信度风险
            confidences = [status.confidence for status in active_models.values()]
            if confidences:
                avg_confidence = np.mean(confidences)
                risk_factors.append(1.0 - avg_confidence)
            
            # 市场波动风险
            if market_data:
                volatility = market_data.get('volatility', 0.0)
                risk_factors.append(volatility)
            
            # GPU温度风险
            if self.gpu_temperature > 80:
                risk_factors.append(0.5)
            elif self.gpu_temperature > 75:
                risk_factors.append(0.3)
            
            return min(np.mean(risk_factors) if risk_factors else 0.5, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ 风险评估失败: {e}")
            return 0.5
    
    def _generate_reasoning(self, active_models: Dict[str, AIModelStatus], 
                          signal: float, confidence: float) -> str:
        """生成决策推理"""
        try:
            reasoning_parts = []
            
            # 信号强度分析
            if abs(signal) > 0.7:
                reasoning_parts.append(f"强烈{'买入' if signal > 0 else '卖出'}信号({signal:.3f})")
            elif abs(signal) > 0.3:
                reasoning_parts.append(f"中等{'买入' if signal > 0 else '卖出'}信号({signal:.3f})")
            else:
                reasoning_parts.append(f"弱{'买入' if signal > 0 else '卖出'}信号({signal:.3f})")
            
            # 置信度分析
            if confidence > 0.8:
                reasoning_parts.append(f"高置信度({confidence:.3f})")
            elif confidence > 0.6:
                reasoning_parts.append(f"中等置信度({confidence:.3f})")
            else:
                reasoning_parts.append(f"低置信度({confidence:.3f})")
            
            # 模型一致性
            predictions = [status.prediction for status in active_models.values()]
            if predictions:
                agreement = 1.0 - np.std(predictions)
                if agreement > 0.8:
                    reasoning_parts.append("模型高度一致")
                elif agreement > 0.6:
                    reasoning_parts.append("模型基本一致")
                else:
                    reasoning_parts.append("模型存在分歧")
            
            # 活跃模型数量
            reasoning_parts.append(f"{len(active_models)}个模型参与决策")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ 推理生成失败: {e}")
            return f"信号: {signal:.3f}, 置信度: {confidence:.3f}"
    
    def _make_conservative_decision(self) -> MetaDecision:
        """制作保守决策"""
        return MetaDecision(
            final_signal=0.0,
            confidence=0.1,
            contributing_models=[],
            model_weights={},
            risk_assessment=0.9,
            timestamp=datetime.now(timezone.utc),
            reasoning="保守决策 - 模型数量不足或数据异常"
        )
    
    async def _async_training_update(self, model_features: np.ndarray, target_signal: float):
        """异步训练更新"""
        try:
            # 添加到训练缓冲区
            self.training_buffer.append({
                'features': model_features,
                'target': target_signal,
                'timestamp': time.time()
            })
            
            # 限制缓冲区大小
            if len(self.training_buffer) > self.max_buffer_size:
                self.training_buffer = self.training_buffer[-self.max_buffer_size:]
            
            # 批量训练
            if len(self.training_buffer) >= self.batch_size:
                await self._batch_training()
                
        except Exception as e:
            logger.warning(f"⚠️ 异步训练更新失败: {e}")
    
    async def _batch_training(self):
        """批量训练"""
        try:
            if not self.training_enabled or len(self.training_buffer) < self.batch_size:
                return
            
            # 准备训练数据
            batch_data = self.training_buffer[-self.batch_size:]
            features = torch.FloatTensor([item['features'] for item in batch_data]).to(self.device)
            targets = torch.FloatTensor([item['target'] for item in batch_data]).to(self.device)
            
            # 前向传播
            self.meta_network.train()
            predictions, confidences = self.meta_network(features)
            
            # 计算损失
            prediction_loss = nn.MSELoss()(predictions, targets)
            confidence_loss = nn.BCELoss()(confidences, torch.abs(targets))
            total_loss = prediction_loss + 0.1 * confidence_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            self.meta_network.eval()
            
            logger.debug(f"🧠 元学习训练完成 - 损失: {total_loss.item():.6f}")
            
        except Exception as e:
            logger.warning(f"⚠️ 批量训练失败: {e}")
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            if len(self.decision_history) < 2:
                return
            
            recent_decisions = self.decision_history[-100:]  # 最近100个决策
            
            # 基础统计
            self.performance_metrics['total_decisions'] = len(self.decision_history)
            self.performance_metrics['avg_confidence'] = np.mean([d.confidence for d in recent_decisions])
            
            # 计算准确率（需要实际收益数据）
            # 这里使用简化的计算方式
            signals = [d.final_signal for d in recent_decisions]
            if signals:
                self.performance_metrics['avg_return'] = np.mean(signals)
                self.performance_metrics['sharpe_ratio'] = np.mean(signals) / (np.std(signals) + 1e-8)
            
        except Exception as e:
            logger.warning(f"⚠️ 性能指标更新失败: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        try:
            return {
                'commander_status': {
                    'device': self.device,
                    'training_enabled': self.training_enabled,
                    'gpu_temperature': self.gpu_temperature,
                    'gpu_utilization': self.gpu_utilization,
                    'memory_usage': self.memory_usage,
                    'buffer_size': len(self.training_buffer)
                },
                'registered_models': len(self.model_statuses),
                'active_models': sum(1 for s in self.model_statuses.values() if s.is_active),
                'model_statuses': {
                    k: {
                        'name': v.model_name,
                        'confidence': v.confidence,
                        'prediction': v.prediction,
                        'performance_score': v.performance_score,
                        'is_active': v.is_active,
                        'last_update': v.last_update.isoformat()
                    } for k, v in self.model_statuses.items()
                },
                'performance_metrics': self.performance_metrics,
                'recent_decisions': len(self.decision_history),
                'model_weights': self.model_weights
            }
        except Exception as e:
            logger.error(f"❌ 状态报告生成失败: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.meta_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'performance_metrics': self.performance_metrics,
                'model_weights': self.model_weights
            }, filepath)
            logger.info(f"💾 元学习模型已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.meta_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
            self.model_weights = checkpoint.get('model_weights', self.model_weights)
            logger.info(f"📂 元学习模型已加载: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False

# 全局实例
meta_learning_commander = MetaLearningCommander()

def initialize_meta_learning_commander(device: str = None) -> MetaLearningCommander:
    """初始化元学习AI指挥官"""
    global meta_learning_commander
    if device:
        meta_learning_commander = MetaLearningCommander(device)
    return meta_learning_commander

if __name__ == "__main__":
    # 测试代码
    async def test_meta_learning():
        commander = initialize_meta_learning_commander()
        
        # 注册测试模型
        commander.register_ai_model("test_model_1", "测试模型1")
        commander.register_ai_model("test_model_2", "测试模型2")
        
        # 更新模型状态
        commander.update_model_status("test_model_1", 0.5, 0.8, 0.7)
        commander.update_model_status("test_model_2", -0.3, 0.6, 0.6)
        
        # 制作决策
        market_data = {
            'price_change': 0.02,
            'volume_ratio': 1.5,
            'volatility': 0.15,
            'rsi': 65.0,
            'macd_signal': 0.1
        }
        
        decision = await commander.make_meta_decision(market_data)
        print(f"决策结果: {decision}")
        
        # 状态报告
        report = commander.get_status_report()
        print(f"状态报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
    
    asyncio.run(test_meta_learning())

