#!/usr/bin/env python3
"""
🧠 时序深度学习AI - Level 3智能体
Time Series Deep Learning AI - Level 3 Agent

专注于时间序列预测，使用LSTM/Transformer等深度学习模型
实现多时间尺度预测、注意力机制、序列到序列学习
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
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TimeSeriesPrediction:
    """时序预测结果"""
    signal: float           # -1到1之间的信号
    confidence: float       # 0到1之间的置信度
    prediction_horizon: int # 预测时间范围(分钟)
    attention_weights: Dict[str, float]  # 注意力权重
    sequence_importance: List[float]     # 序列重要性
    model_type: str         # 模型类型
    reasoning: str          # 预测推理
    timestamp: datetime
    execution_time_ms: float


class LSTMPredictor(nn.Module):
    """LSTM预测器"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # 输出-1到1之间
        )
        
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        output = self.fc(attn_out[:, -1, :])
        
        return output, attn_weights


class TransformerPredictor(nn.Module):
    """Transformer预测器"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer处理
        transformer_out = self.transformer(x)
        
        # 输出投影
        output = self.output_projection(transformer_out[:, -1, :])
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)


class TimeSeriesDeepLearningAI:
    """时序深度学习AI"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化时序AI"""
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 配置参数
        self.sequence_length = self.config.get('sequence_length', 60)  # 60个时间点
        self.feature_size = self.config.get('feature_size', 10)
        self.prediction_horizons = self.config.get('prediction_horizons', [5, 15, 30, 60])  # 分钟
        
        # 数据缓存
        self.data_buffer = deque(maxlen=1000)
        self.feature_buffer = deque(maxlen=1000)
        
        # 模型
        self.models = {}
        self.optimizers = {}
        self.is_training = False
        self.lock = threading.Lock()
        
        # 预测历史
        self.prediction_history = []
        
        # 初始化模型
        self._initialize_models()
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info("🧠 时序深度学习AI (Level 3) 初始化完成")
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            # LSTM模型
            self.models['lstm'] = LSTMPredictor(
                input_size=self.feature_size,
                hidden_size=128,
                num_layers=2
            ).to(self.device)
            
            self.optimizers['lstm'] = optim.Adam(
                self.models['lstm'].parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
            
            # Transformer模型
            self.models['transformer'] = TransformerPredictor(
                input_size=self.feature_size,
                d_model=128,
                nhead=8,
                num_layers=4
            ).to(self.device)
            
            self.optimizers['transformer'] = optim.Adam(
                self.models['transformer'].parameters(),
                lr=0.0001,
                weight_decay=1e-5
            )
            
            logger.info(f"初始化 {len(self.models)} 个时序模型")
            
        except Exception as e:
            logger.error(f"初始化模型失败: {e}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 模型训练任务
        threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="TrainingThread"
        ).start()
        
        # 数据清理任务
        threading.Thread(
            target=self._data_cleanup_loop,
            daemon=True,
            name="DataCleanupThread"
        ).start()
    
    async def predict(self, market_data: Dict[str, Any]) -> TimeSeriesPrediction:
        """执行时序预测"""
        start_time = time.time()
        
        try:
            # 更新数据缓存
            self._update_data_buffer(market_data)
            
            # 准备序列数据
            sequence_data = self._prepare_sequence_data()
            
            if sequence_data is None:
                return self._get_default_prediction(start_time)
            
            # 执行多模型预测
            predictions = await self._multi_model_predict(sequence_data)
            
            # 融合预测结果
            final_prediction = self._ensemble_predictions(predictions)
            
            # 记录预测历史
            execution_time = (time.time() - start_time) * 1000
            final_prediction.execution_time_ms = execution_time
            
            self._record_prediction(final_prediction)
            
            logger.debug(f"时序预测完成: 信号={final_prediction.signal:.4f}, "
                        f"置信度={final_prediction.confidence:.4f}")
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"时序预测失败: {e}")
            return self._get_default_prediction(start_time, str(e))
    
    def _update_data_buffer(self, market_data: Dict[str, Any]):
        """更新数据缓存"""
        try:
            # 提取特征
            features = self._extract_features(market_data)
            
            # 添加到缓存
            with self.lock:
                self.data_buffer.append(market_data)
                self.feature_buffer.append(features)
            
        except Exception as e:
            logger.error(f"更新数据缓存失败: {e}")
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """提取特征"""
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
                for key in ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12', 'ema_26']:
                    features.append(indicators.get(key, 0.0))
            
            # 时间特征
            if 'timestamp' in market_data:
                timestamp = market_data['timestamp']
                features.extend([
                    timestamp % 86400,  # 一天内的秒数
                    (timestamp // 86400) % 7,  # 星期几
                ])
            
            # 确保特征数量一致
            while len(features) < self.feature_size:
                features.append(0.0)
            
            return np.array(features[:self.feature_size])
            
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return np.zeros(self.feature_size)
    
    def _prepare_sequence_data(self) -> Optional[torch.Tensor]:
        """准备序列数据"""
        try:
            with self.lock:
                if len(self.feature_buffer) < self.sequence_length:
                    return None
                
                # 获取最近的序列数据
                sequence = list(self.feature_buffer)[-self.sequence_length:]
                
            # 转换为张量
            sequence_array = np.array(sequence)
            sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device)
            
            return sequence_tensor
            
        except Exception as e:
            logger.error(f"准备序列数据失败: {e}")
            return None
    
    async def _multi_model_predict(self, sequence_data: torch.Tensor) -> Dict[str, Dict]:
        """多模型预测"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    if model_name == 'lstm':
                        output, attention_weights = model(sequence_data)
                        
                        # 处理注意力权重
                        attn_weights_dict = {}
                        if attention_weights is not None:
                            attn_weights_np = attention_weights.cpu().numpy()[0, 0, :]  # 取第一个头的权重
                            for i, weight in enumerate(attn_weights_np):
                                attn_weights_dict[f'step_{i}'] = float(weight)
                        
                        predictions[model_name] = {
                            'signal': float(output.item()),
                            'confidence': self._calculate_confidence(output, model_name),
                            'attention_weights': attn_weights_dict,
                            'model_type': 'LSTM'
                        }
                        
                    elif model_name == 'transformer':
                        output = model(sequence_data)
                        
                        predictions[model_name] = {
                            'signal': float(output.item()),
                            'confidence': self._calculate_confidence(output, model_name),
                            'attention_weights': {},
                            'model_type': 'Transformer'
                        }
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 预测失败: {e}")
                predictions[model_name] = {
                    'signal': 0.0,
                    'confidence': 0.1,
                    'attention_weights': {},
                    'model_type': model_name
                }
        
        return predictions
    
    def _calculate_confidence(self, output: torch.Tensor, model_name: str) -> float:
        """计算置信度"""
        try:
            # 基于输出值的置信度
            signal_strength = abs(output.item())
            
            # 基于模型历史性能的置信度
            base_confidence = 0.7  # 基础置信度
            
            # 综合置信度
            confidence = base_confidence * (0.5 + signal_strength * 0.5)
            
            return np.clip(confidence, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def _ensemble_predictions(self, predictions: Dict[str, Dict]) -> TimeSeriesPrediction:
        """融合预测结果"""
        try:
            if not predictions:
                return self._get_default_prediction(time.time())
            
            # 计算加权平均
            weighted_signals = []
            weighted_confidences = []
            total_weight = 0.0
            
            # 模型权重
            model_weights = {
                'lstm': 0.6,
                'transformer': 0.4
            }
            
            combined_attention = {}
            model_types = []
            
            for model_name, pred in predictions.items():
                weight = model_weights.get(model_name, 0.5)
                confidence = pred['confidence']
                
                # 基于置信度调整权重
                adjusted_weight = weight * confidence
                
                weighted_signals.append(pred['signal'] * adjusted_weight)
                weighted_confidences.append(confidence * adjusted_weight)
                total_weight += adjusted_weight
                
                # 合并注意力权重
                for key, value in pred['attention_weights'].items():
                    if key not in combined_attention:
                        combined_attention[key] = 0.0
                    combined_attention[key] += value * adjusted_weight
                
                model_types.append(pred['model_type'])
            
            # 计算最终结果
            if total_weight > 0:
                final_signal = sum(weighted_signals) / total_weight
                final_confidence = sum(weighted_confidences) / total_weight
                
                # 归一化注意力权重
                for key in combined_attention:
                    combined_attention[key] /= total_weight
            else:
                final_signal = 0.0
                final_confidence = 0.0
            
            # 生成序列重要性
            sequence_importance = self._calculate_sequence_importance(combined_attention)
            
            # 生成推理
            reasoning = f"时序深度学习融合: {'+'.join(model_types)}"
            
            return TimeSeriesPrediction(
                signal=np.clip(final_signal, -1.0, 1.0),
                confidence=np.clip(final_confidence, 0.0, 1.0),
                prediction_horizon=15,  # 默认15分钟预测
                attention_weights=combined_attention,
                sequence_importance=sequence_importance,
                model_type="Ensemble",
                reasoning=reasoning,
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"融合预测结果失败: {e}")
            return self._get_default_prediction(time.time(), str(e))
    
    def _calculate_sequence_importance(self, attention_weights: Dict[str, float]) -> List[float]:
        """计算序列重要性"""
        try:
            # 从注意力权重计算序列重要性
            importance = []
            
            if attention_weights:
                # 按步骤排序
                sorted_steps = sorted(attention_weights.items(), key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
                importance = [weight for _, weight in sorted_steps]
            
            # 如果没有注意力权重，使用默认重要性
            if not importance:
                importance = [1.0 / self.sequence_length] * self.sequence_length
            
            return importance
            
        except Exception as e:
            logger.error(f"计算序列重要性失败: {e}")
            return [1.0 / self.sequence_length] * self.sequence_length
    
    def _get_default_prediction(self, start_time: float, error_msg: str = "") -> TimeSeriesPrediction:
        """获取默认预测"""
        return TimeSeriesPrediction(
            signal=0.0,
            confidence=0.0,
            prediction_horizon=15,
            attention_weights={},
            sequence_importance=[],
            model_type="Default",
            reasoning=f"默认预测{': ' + error_msg if error_msg else ''}",
            timestamp=datetime.now(),
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _record_prediction(self, prediction: TimeSeriesPrediction):
        """记录预测历史"""
        with self.lock:
            self.prediction_history.append(prediction)
            
            # 限制历史记录数量
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
    
    def _training_loop(self):
        """训练循环"""
        while True:
            try:
                time.sleep(1800)  # 30分钟训练一次
                if not self.is_training and len(self.feature_buffer) > self.sequence_length * 2:
                    asyncio.run(self._train_models())
            except Exception as e:
                logger.error(f"训练循环失败: {e}")
    
    def _data_cleanup_loop(self):
        """数据清理循环"""
        while True:
            try:
                time.sleep(3600)  # 1小时清理一次
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"数据清理失败: {e}")
    
    async def _train_models(self):
        """训练模型"""
        if self.is_training:
            return
        
        self.is_training = True
        try:
            logger.info("开始训练时序模型...")
            
            # 准备训练数据
            train_data = self._prepare_training_data()
            
            if train_data is None:
                logger.warning("训练数据不足")
                return
            
            # 训练每个模型
            for model_name, model in self.models.items():
                try:
                    await self._train_single_model(model_name, model, train_data)
                except Exception as e:
                    logger.error(f"训练模型 {model_name} 失败: {e}")
            
            logger.info("时序模型训练完成")
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
        finally:
            self.is_training = False
    
    def _prepare_training_data(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """准备训练数据"""
        try:
            with self.lock:
                if len(self.feature_buffer) < self.sequence_length + 10:
                    return None
                
                # 构建序列数据
                sequences = []
                targets = []
                
                features_list = list(self.feature_buffer)
                
                for i in range(len(features_list) - self.sequence_length):
                    # 输入序列
                    seq = features_list[i:i + self.sequence_length]
                    sequences.append(seq)
                    
                    # 目标值（简化为价格变化方向）
                    current_price = features_list[i + self.sequence_length - 1][0]  # 假设第一个特征是价格
                    future_price = features_list[i + self.sequence_length][0]
                    
                    # 计算价格变化率
                    price_change = (future_price - current_price) / current_price if current_price != 0 else 0
                    target = np.tanh(price_change * 100)  # 归一化到-1到1之间
                    targets.append(target)
                
                if len(sequences) < 10:
                    return None
                
                # 转换为张量
                X = torch.FloatTensor(sequences).to(self.device)
                y = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
                
                return X, y
                
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return None
    
    async def _train_single_model(self, model_name: str, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor]):
        """训练单个模型"""
        try:
            X, y = train_data
            optimizer = self.optimizers[model_name]
            criterion = nn.MSELoss()
            
            model.train()
            
            # 简单的训练循环
            for epoch in range(10):
                optimizer.zero_grad()
                
                if model_name == 'lstm':
                    output, _ = model(X)
                else:
                    output = model(X)
                
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    logger.debug(f"模型 {model_name} Epoch {epoch}, Loss: {loss.item():.6f}")
            
            logger.info(f"模型 {model_name} 训练完成")
            
        except Exception as e:
            logger.error(f"训练单个模型失败: {e}")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            with self.lock:
                # 清理预测历史
                if len(self.prediction_history) > 500:
                    self.prediction_history = self.prediction_history[-500:]
            
            logger.debug("数据清理完成")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        with self.lock:
            return {
                'level': 3,
                'name': 'Time Series Deep Learning AI',
                'models_count': len(self.models),
                'sequence_length': self.sequence_length,
                'data_buffer_size': len(self.data_buffer),
                'feature_buffer_size': len(self.feature_buffer),
                'prediction_count': len(self.prediction_history),
                'is_training': self.is_training,
                'device': str(self.device),
                'prediction_horizons': self.prediction_horizons
            }


# 全局实例
_time_series_ai = None

def get_time_series_ai(config: Dict[str, Any] = None) -> TimeSeriesDeepLearningAI:
    """获取时序深度学习AI实例"""
    global _time_series_ai
    if _time_series_ai is None:
        _time_series_ai = TimeSeriesDeepLearningAI(config)
    return _time_series_ai


if __name__ == "__main__":
    # 测试代码
    async def test_time_series_ai():
        """测试时序AI"""
        ai = get_time_series_ai()
        
        # 模拟历史数据
        for i in range(100):
            market_data = {
                'price': 50000.0 + np.sin(i * 0.1) * 1000,
                'volume': 1000.0 + np.random.normal(0, 100),
                'indicators': {
                    'rsi': 50 + np.sin(i * 0.05) * 20,
                    'macd': np.sin(i * 0.08) * 0.5,
                    'bb_upper': 51000.0,
                    'bb_lower': 49000.0,
                    'sma_20': 50000.0,
                    'ema_12': 50000.0,
                    'ema_26': 50000.0
                },
                'timestamp': time.time() + i * 60
            }
            ai._update_data_buffer(market_data)
        
        # 执行预测
        prediction = await ai.predict(market_data)
        
        print(f"时序预测结果:")
        print(f"信号: {prediction.signal}")
        print(f"置信度: {prediction.confidence}")
        print(f"模型类型: {prediction.model_type}")
        print(f"推理: {prediction.reasoning}")
        
        # 获取状态
        status = ai.get_status()
        print(f"\n时序AI状态: {json.dumps(status, indent=2)}")
    
    # 运行测试
    asyncio.run(test_time_series_ai())
