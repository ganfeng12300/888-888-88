"""
🦊 猎狐AI量化交易系统（史诗级）- 强化学习交易员
GPU加速的深度强化学习交易智能体，实现PPO算法

核心功能：
1. 实时市场环境建模
2. Actor-Critic双网络架构  
3. GPU加速训练和推理
4. 经验回放和探索策略
5. 动态奖励函数优化
"""

import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path
from collections import deque
import random


@dataclass
class TradingAction:
    """交易动作"""
    action_type: str  # "buy", "sell", "hold"
    position_size: float  # 仓位大小 0-1
    confidence: float  # 置信度 0-1
    reasoning: str  # 决策理由
    timestamp: datetime


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int = 50, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 动作类型输出 (3个动作: hold, buy, sell)
        self.action_head = nn.Linear(hidden_dim // 2, 3)
        
        # 仓位大小输出
        self.position_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state):
        features = self.network(state)
        
        # 动作概率分布
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 仓位大小 (0-1)
        position_size = torch.sigmoid(self.position_head(features))
        
        return action_probs, position_size


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, state_dim: int = 50, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        return self.network(state)


class ReinforcementTrader:
    """强化学习交易员"""
    
    def __init__(self, data_dir: str = "data/rl_trader"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🎮 强化学习交易员使用设备: {self.device}")
        
        # 网络架构
        self.state_dim = 50
        self.actor = ActorNetwork(self.state_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # PPO参数
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 10
        self.batch_size = 64
        
        # 经验缓冲区
        self.memory = deque(maxlen=10000)
        
        # 性能统计
        self.total_steps = 0
        self.total_rewards = 0
        self.episode_rewards = deque(maxlen=100)
        
        # 探索参数
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载模型
        self._load_models()
        
        logger.info("🤖 强化学习交易员初始化完成")
        
    def get_trading_action(self, market_state: Dict[str, float]) -> TradingAction:
        """获取交易动作"""
        with self.lock:
            # 准备状态
            state = self._prepare_state(market_state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 网络推理
            with torch.no_grad():
                self.actor.eval()
                action_probs, position_size = self.actor(state_tensor)
                
                # 选择动作
                if np.random.random() < self.epsilon:
                    # 探索：随机选择
                    action_idx = np.random.randint(0, 3)
                    position_val = np.random.random()
                else:
                    # 利用：根据策略选择
                    action_idx = torch.multinomial(action_probs, 1).item()
                    position_val = position_size.item()
                
                # 计算置信度
                confidence = float(action_probs[0, action_idx].item())
            
            # 转换为交易动作
            action_types = ["hold", "buy", "sell"]
            action = TradingAction(
                action_type=action_types[action_idx],
                position_size=position_val,
                confidence=confidence,
                reasoning=f"RL策略: {action_types[action_idx]} 仓位={position_val:.3f}",
                timestamp=datetime.now()
            )
            
            logger.debug(f"🎯 RL交易动作: {action.action_type} 仓位={action.position_size:.3f} 置信度={action.confidence:.3f}")
            
            return action
    
    def update_experience(self, state: Dict, action: TradingAction, reward: float, next_state: Dict, done: bool):
        """更新经验"""
        with self.lock:
            # 存储经验
            experience = {
                "state": self._prepare_state(state),
                "action": self._action_to_vector(action),
                "reward": reward,
                "next_state": self._prepare_state(next_state),
                "done": done,
                "timestamp": datetime.now()
            }
            
            self.memory.append(experience)
            self.total_steps += 1
            self.total_rewards += reward
            
            # 触发训练
            if len(self.memory) >= self.batch_size and self.total_steps % 100 == 0:
                self._train_model()
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            logger.debug(f"📊 经验更新: 奖励={reward:.4f} 探索率={self.epsilon:.3f}")
    
    def _prepare_state(self, market_state: Dict[str, float]) -> np.array:
        """准备状态向量"""
        state = np.zeros(self.state_dim)
        
        # 基础市场数据 (0-4)
        state[0] = market_state.get("price", 50000) / 100000  # 标准化价格
        state[1] = market_state.get("volume", 0) / 1000000    # 标准化成交量
        state[2] = market_state.get("volatility", 0)          # 波动率
        state[3] = market_state.get("bid_ask_spread", 0)      # 买卖价差
        state[4] = market_state.get("market_depth", 0)        # 市场深度
        
        # 技术指标 (5-34)
        indicators = [
            "rsi", "macd", "bb_upper", "bb_lower", "sma_5", "sma_20",
            "ema_12", "ema_26", "atr", "obv", "stoch_k", "stoch_d",
            "williams_r", "cci", "momentum", "roc", "adx", "aroon_up",
            "aroon_down", "mfi", "bop", "vwap", "pivot_point", "support_1",
            "resistance_1", "fibonacci_38", "fibonacci_62", "ichimoku_tenkan",
            "ichimoku_kijun", "parabolic_sar"
        ]
        
        for i, indicator in enumerate(indicators):
            if i + 5 < self.state_dim:
                value = market_state.get(indicator, 0)
                # 简单标准化
                if abs(value) > 1000:
                    value = value / 1000
                elif abs(value) > 100:
                    value = value / 100
                elif abs(value) > 10:
                    value = value / 10
                state[i + 5] = value
        
        # 时间特征 (35-39)
        now = datetime.now()
        state[35] = now.hour / 24.0
        state[36] = now.minute / 60.0
        state[37] = now.weekday() / 7.0
        state[38] = now.day / 31.0
        state[39] = now.month / 12.0
        
        # 组合状态 (40-49)
        state[40] = market_state.get("portfolio_value", 100000) / 100000
        state[41] = market_state.get("position", 0)
        state[42] = market_state.get("unrealized_pnl", 0) / 10000
        state[43] = market_state.get("cash_balance", 100000) / 100000
        state[44] = market_state.get("total_trades", 0) / 1000
        state[45] = market_state.get("win_rate", 0.5)
        state[46] = market_state.get("sharpe_ratio", 0)
        state[47] = market_state.get("max_drawdown", 0)
        state[48] = market_state.get("profit_factor", 1.0)
        state[49] = market_state.get("avg_trade_return", 0)
        
        return state.astype(np.float32)
    
    def _action_to_vector(self, action: TradingAction) -> np.array:
        """将动作转换为向量"""
        action_types = {"hold": 0, "buy": 1, "sell": 2}
        return np.array([
            action_types.get(action.action_type, 0),
            action.position_size
        ], dtype=np.float32)
    
    def _train_model(self):
        """训练PPO模型"""
        if len(self.memory) < self.batch_size:
            return
            
        try:
            self.actor.train()
            self.critic.train()
            
            # 采样批次数据
            batch = random.sample(list(self.memory), self.batch_size)
            
            states = torch.FloatTensor([exp["state"] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp["action"][0] for exp in batch]).to(self.device)
            position_sizes = torch.FloatTensor([exp["action"][1] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp["reward"] for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp["next_state"] for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp["done"] for exp in batch]).to(self.device)
            
            # 计算优势函数
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze()
                current_values = self.critic(states).squeeze()
                
                # TD目标
                td_targets = rewards + self.gamma * next_values * (~dones)
                advantages = td_targets - current_values
                
                # 标准化优势
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 获取旧策略概率
            with torch.no_grad():
                old_action_probs, _ = self.actor(states)
                old_probs = old_action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # PPO训练循环
            for _ in range(self.ppo_epochs):
                # 当前策略
                action_probs, pred_position_sizes = self.actor(states)
                current_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
                
                # 重要性采样比率
                ratio = current_probs / (old_probs + 1e-8)
                
                # PPO损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 仓位大小损失
                position_loss = F.mse_loss(pred_position_sizes.squeeze(), position_sizes)
                
                # 总Actor损失
                total_actor_loss = actor_loss + position_loss * 0.1
                
                # Critic损失
                values = self.critic(states).squeeze()
                critic_loss = F.mse_loss(values, td_targets)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
            
            # 定期保存模型
            if self.total_steps % 1000 == 0:
                self._save_models()
                
            logger.debug(f"🎓 PPO训练完成: Actor损失={total_actor_loss.item():.4f} Critic损失={critic_loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ PPO训练失败: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.lock:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            
            return {
                "agent_type": "reinforcement_trader",
                "total_steps": self.total_steps,
                "total_rewards": self.total_rewards,
                "avg_episode_reward": avg_reward,
                "epsilon": self.epsilon,
                "memory_size": len(self.memory),
                "device": str(self.device),
                "model_parameters": {
                    "actor_params": sum(p.numel() for p in self.actor.parameters()),
                    "critic_params": sum(p.numel() for p in self.critic.parameters())
                }
            }
    
    def _save_models(self):
        """保存模型"""
        try:
            torch.save({
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
                "total_rewards": self.total_rewards
            }, self.data_dir / "rl_trader_model.pth")
            
            logger.debug("💾 RL模型保存完成")
            
        except Exception as e:
            logger.error(f"❌ 保存RL模型失败: {e}")
    
    def _load_models(self):
        """加载模型"""
        try:
            model_path = self.data_dir / "rl_trader_model.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                self.actor.load_state_dict(checkpoint["actor_state_dict"])
                self.critic.load_state_dict(checkpoint["critic_state_dict"])
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
                self.total_steps = checkpoint.get("total_steps", 0)
                self.epsilon = checkpoint.get("epsilon", 1.0)
                self.total_rewards = checkpoint.get("total_rewards", 0)
                
                logger.info("📥 RL模型加载完成")
                
        except Exception as e:
            logger.warning(f"⚠️ 加载RL模型失败: {e}")


# 全局强化学习交易员实例
rl_trader = ReinforcementTrader()


if __name__ == "__main__":
    # 测试强化学习交易员
    logger.info("🧪 测试强化学习交易员...")
    
    # 模拟市场状态
    market_state = {
        "price": 50000,
        "volume": 1000000,
        "volatility": 0.02,
        "rsi": 65,
        "macd": 100,
        "portfolio_value": 100000,
        "position": 0,
        "unrealized_pnl": 0
    }
    
    # 获取交易动作
    action = rl_trader.get_trading_action(market_state)
    logger.info(f"🎯 交易动作: {action}")
    
    # 模拟经验更新
    rl_trader.update_experience(
        state=market_state,
        action=action,
        reward=0.01,
        next_state=market_state,
        done=False
    )
    
    # 获取性能报告
    report = rl_trader.get_performance_report()
    logger.info(f"📊 性能报告: {report}")
