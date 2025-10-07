#!/usr/bin/env python3
"""
🧠 888-888-88 AI模型训练器
生产级机器学习模型训练和优化系统
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ModelType(Enum):
    """模型类型"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ATTENTION_LSTM = "attention_lstm"


class FeatureType(Enum):
    """特征类型"""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"


@dataclass
class TrainingConfig:
    """训练配置"""
    model_type: ModelType = ModelType.LSTM
    sequence_length: int = 60
    prediction_horizon: int = 1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    hidden_size: int = 128
    num_layers: int = 2
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class ModelMetrics:
    """模型评估指标"""
    mse: float
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float


class LSTMModel(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer预测模型"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, output_size: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        if seq_len <= self.positional_encoding.size(1):
            x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer编码
        transformer_out = self.transformer(x)
        
        # 取最后一个时间步
        last_output = transformer_out[:, -1, :]
        
        # 输出投影
        output = self.dropout(last_output)
        output = self.output_projection(output)
        
        return output


class ModelTrainer:
    """AI模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # 模型和训练状态
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.training_history = []
        
        # 特征工程
        self.feature_columns = []
        self.feature_importance = {}
        
        logger.info(f"🧠 AI模型训练器初始化完成 (设备: {self.device})")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        logger.info("🔧 开始特征工程...")
        
        # 基础价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # 移动平均特征
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # 波动率特征
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'realized_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # 技术指标
        df = self._add_technical_indicators(df)
        
        # 成交量特征
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / 
                     df['volume'].rolling(20).sum())
        
        # 时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 滞后特征
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # 前瞻特征（目标变量）
        df['target'] = df['close'].shift(-self.config.prediction_horizon)
        df['target_returns'] = df['target'] / df['close'] - 1
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 选择特征列
        self.feature_columns = [col for col in df.columns 
                              if col not in ['target', 'target_returns'] 
                              and not col.startswith('Unnamed')]
        
        logger.info(f"✅ 特征工程完成，共生成 {len(self.feature_columns)} 个特征")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 随机指标
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # 威廉指标
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # 商品通道指数
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        return df
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列序列"""
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(data)):
            X.append(data[i-self.config.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练数据"""
        logger.info("📊 准备训练数据...")
        
        # 特征和目标
        features = df[self.feature_columns].values
        targets = df['target_returns'].values
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # 创建序列
        X, y = self.create_sequences(features_scaled, targets_scaled)
        
        # 时间序列分割
        train_size = int(len(X) * (1 - self.config.validation_split - self.config.test_split))
        val_size = int(len(X) * self.config.validation_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        logger.info(f"✅ 数据准备完成 - 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, input_size: int) -> nn.Module:
        """创建模型"""
        if self.config.model_type == ModelType.LSTM:
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout=self.config.dropout_rate
            )
        elif self.config.model_type == ModelType.TRANSFORMER:
            model = TransformerModel(
                input_size=input_size,
                d_model=self.config.hidden_size,
                nhead=8,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout=self.config.dropout_rate
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
        
        return model.to(self.device)
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """训练模型"""
        logger.info("🚀 开始模型训练...")
        
        # 创建模型
        input_size = train_loader.dataset[0][0].shape[-1]
        self.model = self.create_model(input_size)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 更新学习率
            scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rate'].append(current_lr)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, "
                          f"LR: {current_lr:.6f}")
            
            # 早停
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        self.training_history = history
        logger.info("✅ 模型训练完成")
        
        return history
    
    def evaluate_model(self, test_loader: DataLoader) -> ModelMetrics:
        """评估模型"""
        logger.info("📊 评估模型性能...")
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 反标准化
        predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        
        # 方向准确率
        actual_direction = np.sign(actuals)
        pred_direction = np.sign(predictions)
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # 交易指标
        returns = predictions  # 预测收益率
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 盈利因子
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns)) 
                        if len(negative_returns) > 0 else np.inf)
        
        metrics = ModelMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2=r2,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor
        )
        
        logger.info(f"✅ 模型评估完成 - R²: {r2:.4f}, 方向准确率: {directional_accuracy:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] = None) -> None:
        """保存模型"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        logger.info(f"💾 模型已保存至: {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """加载模型"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # 恢复配置
        config_dict = model_data['config']
        self.config = TrainingConfig(**config_dict)
        
        # 恢复缩放器和特征
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_history = model_data.get('training_history', [])
        
        # 重建模型
        input_size = len(self.feature_columns)
        self.model = self.create_model(input_size)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"📂 模型已从 {filepath} 加载")
        
        return model_data.get('metadata', {})
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        self.model.eval()
        
        # 标准化输入
        data_scaled = self.scaler.transform(data)
        
        # 创建序列
        if len(data_scaled) < self.config.sequence_length:
            raise ValueError(f"数据长度不足，需要至少 {self.config.sequence_length} 个样本")
        
        # 取最后一个序列
        sequence = data_scaled[-self.config.sequence_length:].reshape(1, self.config.sequence_length, -1)
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy()
        
        # 反标准化
        prediction = self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        return prediction
    
    def plot_training_history(self, save_path: str = None) -> None:
        """绘制训练历史"""
        if not self.training_history:
            logger.warning("⚠️ 没有训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 损失曲线
        axes[0].plot(self.training_history['train_loss'], label='训练损失', color='blue')
        axes[0].plot(self.training_history['val_loss'], label='验证损失', color='red')
        axes[0].set_title('训练和验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率曲线
        axes[1].plot(self.training_history['learning_rate'], label='学习率', color='green')
        axes[1].set_title('学习率变化')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 训练历史图表已保存至: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # 测试模型训练器
    config = TrainingConfig(
        model_type=ModelType.LSTM,
        epochs=50,
        batch_size=32
    )
    
    trainer = ModelTrainer(config)
    
    # 生成模拟数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='H')
    np.random.seed(42)
    
    data = {
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000, len(dates))
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 特征工程
    df = trainer.prepare_features(df)
    
    # 准备数据
    train_loader, val_loader, test_loader = trainer.prepare_data(df)
    
    # 训练模型
    history = trainer.train_model(train_loader, val_loader)
    
    # 评估模型
    metrics = trainer.evaluate_model(test_loader)
    print(f"模型评估结果: R² = {metrics.r2:.4f}, 方向准确率 = {metrics.directional_accuracy:.4f}")
    
    # 保存模型
    trainer.save_model('test_model.pth')
    
    logger.info("🎉 模型训练测试完成")

