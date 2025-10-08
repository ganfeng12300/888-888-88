#!/usr/bin/env python3
"""
🧠 分层AI系统 - 高阶AI模型领导低阶AI模型
Hierarchical AI System - High-level AI models leading low-level AI models
"""
import os
import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AIModelConfig:
    """AI模型配置"""
    name: str
    level: int  # 1-6级，6级最高
    model_type: str
    features: List[str]
    target: str
    update_frequency: int  # 秒
    confidence_threshold: float
    max_memory_mb: int
    
@dataclass
class AIDecision:
    """AI决策"""
    model_name: str
    level: int
    action: str  # BUY, SELL, HOLD
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: datetime
    
@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    indicators: Dict[str, float]

class HierarchicalAISystem:
    """分层AI系统"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # AI模型存储
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_configs: Dict[str, AIModelConfig] = {}
        
        # 决策队列
        self.decision_queue = queue.Queue()
        self.market_data_queue = queue.Queue()
        
        # 数据存储
        self.db_path = self.data_dir / "hierarchical_ai.db"
        self.init_database()
        
        # 运行状态
        self.running = False
        self.threads = []
        
        # 初始化AI模型配置
        self.init_ai_models()
        
        logger.info("🧠 分层AI系统初始化完成")
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                level INTEGER,
                action TEXT,
                confidence REAL,
                price_target REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                reasoning TEXT,
                timestamp DATETIME,
                executed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                level INTEGER,
                accuracy REAL,
                profit_loss REAL,
                trades_count INTEGER,
                win_rate REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                volume REAL,
                indicators TEXT,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_ai_models(self):
        """初始化AI模型配置"""
        
        # 6级 - 战略总指挥AI (Strategic Command AI)
        self.model_configs["strategic_commander"] = AIModelConfig(
            name="strategic_commander",
            level=6,
            model_type="ensemble",
            features=["market_trend", "volatility", "volume_profile", "sentiment", "macro_indicators"],
            target="strategic_direction",
            update_frequency=3600,  # 1小时更新
            confidence_threshold=0.85,
            max_memory_mb=500
        )
        
        # 5级 - 战术协调AI (Tactical Coordinator AI)
        self.model_configs["tactical_coordinator"] = AIModelConfig(
            name="tactical_coordinator",
            level=5,
            model_type="gradient_boosting",
            features=["price_action", "support_resistance", "trend_strength", "momentum"],
            target="tactical_signals",
            update_frequency=1800,  # 30分钟更新
            confidence_threshold=0.80,
            max_memory_mb=300
        )
        
        # 4级 - 风险管理AI (Risk Management AI)
        self.model_configs["risk_manager"] = AIModelConfig(
            name="risk_manager",
            level=4,
            model_type="neural_network",
            features=["portfolio_exposure", "volatility", "correlation", "drawdown"],
            target="risk_adjustment",
            update_frequency=900,  # 15分钟更新
            confidence_threshold=0.75,
            max_memory_mb=200
        )
        
        # 3级 - 技术分析AI (Technical Analysis AI)
        self.model_configs["technical_analyst"] = AIModelConfig(
            name="technical_analyst",
            level=3,
            model_type="random_forest",
            features=["rsi", "macd", "bollinger", "stochastic", "williams_r"],
            target="technical_signals",
            update_frequency=300,  # 5分钟更新
            confidence_threshold=0.70,
            max_memory_mb=150
        )
        
        # 2级 - 执行优化AI (Execution Optimizer AI)
        self.model_configs["execution_optimizer"] = AIModelConfig(
            name="execution_optimizer",
            level=2,
            model_type="gradient_boosting",
            features=["order_book", "spread", "liquidity", "slippage"],
            target="execution_timing",
            update_frequency=60,  # 1分钟更新
            confidence_threshold=0.65,
            max_memory_mb=100
        )
        
        # 1级 - 实时监控AI (Real-time Monitor AI)
        self.model_configs["realtime_monitor"] = AIModelConfig(
            name="realtime_monitor",
            level=1,
            model_type="neural_network",
            features=["price", "volume", "bid_ask", "tick_data"],
            target="immediate_signals",
            update_frequency=10,  # 10秒更新
            confidence_threshold=0.60,
            max_memory_mb=50
        )
    
    def create_model(self, config: AIModelConfig) -> Any:
        """创建AI模型"""
        if config.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif config.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif config.model_type == "neural_network":
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        elif config.model_type == "ensemble":
            # 集成模型
            return {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'nn': MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
            }
        else:
            raise ValueError(f"未知模型类型: {config.model_type}")
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """训练AI模型"""
        config = self.model_configs[model_name]
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_name] = scaler
        
        # 创建并训练模型
        model = self.create_model(config)
        
        if config.model_type == "ensemble":
            # 训练集成模型
            for sub_model_name, sub_model in model.items():
                sub_model.fit(X_scaled, y)
        else:
            model.fit(X_scaled, y)
        
        self.models[model_name] = model
        
        # 保存模型
        model_path = self.data_dir / f"{model_name}_model.joblib"
        scaler_path = self.data_dir / f"{model_name}_scaler.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"✅ AI模型 {model_name} (Level {config.level}) 训练完成")
    
    def predict(self, model_name: str, X: np.ndarray) -> Tuple[float, float]:
        """AI模型预测"""
        if model_name not in self.models:
            return 0.0, 0.0
        
        config = self.model_configs[model_name]
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # 数据预处理
        X_scaled = scaler.transform(X.reshape(1, -1))
        
        if config.model_type == "ensemble":
            # 集成预测
            predictions = []
            for sub_model in model.values():
                pred = sub_model.predict(X_scaled)[0]
                predictions.append(pred)
            
            prediction = np.mean(predictions)
            confidence = 1.0 - np.std(predictions) / np.mean(np.abs(predictions)) if np.mean(np.abs(predictions)) > 0 else 0.5
        else:
            prediction = model.predict(X_scaled)[0]
            confidence = min(abs(prediction) / 100.0, 1.0)  # 简化的置信度计算
        
        return prediction, confidence
    
    def make_decision(self, model_name: str, market_data: MarketData) -> Optional[AIDecision]:
        """AI决策制定"""
        config = self.model_configs[model_name]
        
        # 准备特征数据
        features = self.extract_features(market_data, config.features)
        if features is None:
            return None
        
        # AI预测
        prediction, confidence = self.predict(model_name, features)
        
        # 置信度检查
        if confidence < config.confidence_threshold:
            return None
        
        # 决策逻辑
        if prediction > 0.1:
            action = "BUY"
            price_target = market_data.price * (1 + prediction / 100)
            stop_loss = market_data.price * 0.98
            take_profit = market_data.price * 1.05
        elif prediction < -0.1:
            action = "SELL"
            price_target = market_data.price * (1 + prediction / 100)
            stop_loss = market_data.price * 1.02
            take_profit = market_data.price * 0.95
        else:
            action = "HOLD"
            price_target = market_data.price
            stop_loss = market_data.price
            take_profit = market_data.price
        
        # 位置大小计算（基于级别和置信度）
        base_size = 0.1  # 基础仓位10%
        level_multiplier = config.level / 6.0  # 级别权重
        position_size = base_size * level_multiplier * confidence
        
        decision = AIDecision(
            model_name=model_name,
            level=config.level,
            action=action,
            confidence=confidence,
            price_target=price_target,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=f"Level {config.level} AI预测: {prediction:.4f}, 置信度: {confidence:.4f}",
            timestamp=datetime.now()
        )
        
        return decision
    
    def extract_features(self, market_data: MarketData, feature_names: List[str]) -> Optional[np.ndarray]:
        """提取特征"""
        features = []
        
        for feature_name in feature_names:
            if feature_name == "price":
                features.append(market_data.price)
            elif feature_name == "volume":
                features.append(market_data.volume)
            elif feature_name in market_data.indicators:
                features.append(market_data.indicators[feature_name])
            else:
                # 默认值
                features.append(0.0)
        
        return np.array(features) if features else None
    
    def hierarchical_decision_making(self, market_data: MarketData) -> List[AIDecision]:
        """分层决策制定"""
        decisions = []
        
        # 按级别从高到低进行决策
        for level in range(6, 0, -1):
            for model_name, config in self.model_configs.items():
                if config.level == level:
                    decision = self.make_decision(model_name, market_data)
                    if decision:
                        decisions.append(decision)
                        
                        # 高级别决策影响低级别
                        if level >= 4:  # 4级以上的决策会影响下级
                            self.influence_lower_levels(decision)
        
        return decisions
    
    def influence_lower_levels(self, high_level_decision: AIDecision):
        """高级别决策影响低级别"""
        # 高级别的决策会调整低级别模型的参数
        influence_factor = high_level_decision.confidence * (high_level_decision.level / 6.0)
        
        for model_name, config in self.model_configs.items():
            if config.level < high_level_decision.level:
                # 调整置信度阈值
                if high_level_decision.action in ["BUY", "SELL"]:
                    config.confidence_threshold *= (1 - influence_factor * 0.1)
                else:
                    config.confidence_threshold *= (1 + influence_factor * 0.1)
                
                # 确保阈值在合理范围内
                config.confidence_threshold = max(0.5, min(0.9, config.confidence_threshold))
    
    def save_decision(self, decision: AIDecision):
        """保存决策到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_decisions 
            (model_name, level, action, confidence, price_target, stop_loss, take_profit, 
             position_size, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision.model_name, decision.level, decision.action, decision.confidence,
            decision.price_target, decision.stop_loss, decision.take_profit,
            decision.position_size, decision.reasoning, decision.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_consensus_decision(self, decisions: List[AIDecision]) -> Optional[AIDecision]:
        """获取共识决策"""
        if not decisions:
            return None
        
        # 按级别权重计算
        total_weight = 0
        weighted_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for decision in decisions:
            weight = decision.level * decision.confidence
            total_weight += weight
            weighted_actions[decision.action] += weight
        
        # 找出权重最高的行动
        best_action = max(weighted_actions, key=weighted_actions.get)
        
        if total_weight == 0:
            return None
        
        # 计算平均值
        avg_confidence = sum(d.confidence * d.level for d in decisions) / total_weight
        avg_price_target = sum(d.price_target * d.level * d.confidence for d in decisions) / total_weight
        avg_position_size = sum(d.position_size * d.level * d.confidence for d in decisions) / total_weight
        
        # 创建共识决策
        consensus = AIDecision(
            model_name="consensus",
            level=6,  # 最高级别
            action=best_action,
            confidence=avg_confidence,
            price_target=avg_price_target,
            stop_loss=min(d.stop_loss for d in decisions if d.action == best_action),
            take_profit=max(d.take_profit for d in decisions if d.action == best_action),
            position_size=avg_position_size,
            reasoning=f"共识决策基于{len(decisions)}个AI模型",
            timestamp=datetime.now()
        )
        
        return consensus
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除旧的决策记录
        cursor.execute('DELETE FROM ai_decisions WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM model_performance WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"🗑️ 清理了 {deleted_count} 条旧数据记录")
    
    async def start(self):
        """启动分层AI系统"""
        self.running = True
        logger.info("🚀 分层AI系统启动")
        
        # 启动各个线程
        self.threads = [
            threading.Thread(target=self.data_processing_loop, daemon=True),
            threading.Thread(target=self.decision_making_loop, daemon=True),
            threading.Thread(target=self.cleanup_loop, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
    
    def data_processing_loop(self):
        """数据处理循环"""
        while self.running:
            try:
                # 处理市场数据
                if not self.market_data_queue.empty():
                    market_data = self.market_data_queue.get()
                    
                    # 保存市场数据
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO market_data (symbol, price, volume, indicators, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        market_data.symbol, market_data.price, market_data.volume,
                        json.dumps(market_data.indicators), market_data.timestamp
                    ))
                    conn.commit()
                    conn.close()
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"数据处理错误: {e}")
    
    def decision_making_loop(self):
        """决策制定循环"""
        while self.running:
            try:
                # 获取最新市场数据
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, price, volume, indicators, timestamp 
                    FROM market_data 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''')
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    market_data = MarketData(
                        symbol=row[0],
                        price=row[1],
                        volume=row[2],
                        timestamp=datetime.fromisoformat(row[4]),
                        indicators=json.loads(row[3]) if row[3] else {}
                    )
                    
                    # 分层决策
                    decisions = self.hierarchical_decision_making(market_data)
                    
                    # 保存决策
                    for decision in decisions:
                        self.save_decision(decision)
                        self.decision_queue.put(decision)
                    
                    # 获取共识决策
                    consensus = self.get_consensus_decision(decisions)
                    if consensus:
                        self.decision_queue.put(consensus)
                        logger.info(f"🎯 共识决策: {consensus.action} - 置信度: {consensus.confidence:.4f}")
                
                time.sleep(10)  # 10秒检查一次
            except Exception as e:
                logger.error(f"决策制定错误: {e}")
    
    def cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                # 每小时清理一次
                self.cleanup_old_data()
                time.sleep(3600)
            except Exception as e:
                logger.error(f"清理错误: {e}")
    
    def stop(self):
        """停止系统"""
        self.running = False
        logger.info("🛑 分层AI系统停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 统计信息
        cursor.execute('SELECT COUNT(*) FROM ai_decisions WHERE timestamp > datetime("now", "-1 day")')
        daily_decisions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM market_data WHERE timestamp > datetime("now", "-1 hour")')
        hourly_data_points = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "running": self.running,
            "models_loaded": len(self.models),
            "daily_decisions": daily_decisions,
            "hourly_data_points": hourly_data_points,
            "model_configs": {name: asdict(config) for name, config in self.model_configs.items()},
            "timestamp": datetime.now().isoformat()
        }

# 全局实例
hierarchical_ai = HierarchicalAISystem()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 启动系统
        await hierarchical_ai.start()
        
        # 模拟市场数据
        while True:
            market_data = MarketData(
                symbol="BTCUSDT",
                price=50000 + np.random.normal(0, 1000),
                volume=1000000 + np.random.normal(0, 100000),
                timestamp=datetime.now(),
                indicators={
                    "rsi": np.random.uniform(20, 80),
                    "macd": np.random.normal(0, 10),
                    "bollinger": np.random.uniform(-2, 2),
                    "volume_profile": np.random.uniform(0.5, 1.5),
                    "sentiment": np.random.uniform(-1, 1)
                }
            )
            
            hierarchical_ai.market_data_queue.put(market_data)
            await asyncio.sleep(10)
    
    asyncio.run(main())

