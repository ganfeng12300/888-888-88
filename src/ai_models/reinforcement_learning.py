"""
🧠 强化学习交易模型
生产级PPO/SAC强化学习算法，支持连续动作空间和多资产交易
实现智能交易决策和风险管理
"""

import asyncio
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import pickle
import json
from pathlib import Path

from loguru import logger
from src.hardware.gpu_manager import GPUTaskType, allocate_gpu_memory, deallocate_gpu_memory
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores


class ActionType(Enum):
    """动作类型"""
    HOLD = 0
    BUY = 1
    SELL = 2


class ModelType(Enum):
    """模型类型"""
    PPO = "ppo"
    SAC = "sac"
    A3C = "a3c"
    DDPG = "ddpg"


@dataclass
class TrainingConfig:
    """训练配置"""
    model_type: ModelType = ModelType.PPO
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_frequency: int = 2048
    epochs_per_update: int = 10
    target_kl: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelMetrics:
    """模型指标"""
    episode_reward: float = 0.0
    episode_length: int = 0
    total_trades: int = 0
    profitable_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    training_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0


class TradingEnvironment(gym.Env):
    """交易环境"""
    
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # 状态空间：价格特征 + 技术指标 + 持仓信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(data.shape[1] + 3,), dtype=np.float32
        )
        
        # 动作空间：连续动作 [-1, 1] (卖出到买入)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # 持仓比例 [-1, 1]
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.returns = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        """执行动作"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # 解析动作
        target_position = np.clip(action[0], -self.max_position, self.max_position)
        
        # 计算当前价格和收益
        current_price = self.data[self.current_step, 0]  # 假设第一列是价格
        next_price = self.data[self.current_step + 1, 0]
        
        # 计算持仓收益
        position_return = (next_price - current_price) / current_price * self.position
        
        # 计算交易成本
        position_change = abs(target_position - self.position)
        transaction_cost = position_change * self.transaction_cost
        
        # 更新投资组合价值
        self.portfolio_value *= (1 + position_return - transaction_cost)
        
        # 记录交易
        if abs(target_position - self.position) > 0.01:
            self.trade_history.append({
                'step': self.current_step,
                'action': target_position,
                'price': current_price,
                'position_change': target_position - self.position
            })
        
        # 更新持仓
        self.position = target_position
        
        # 计算奖励
        reward = self._calculate_reward(position_return, transaction_cost)
        
        # 记录收益
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        self.returns.append(portfolio_return)
        
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1
        truncated = self.portfolio_value <= self.initial_balance * 0.5  # 50%止损
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'total_return': portfolio_return,
            'num_trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        """获取观察状态"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # 市场特征
        market_features = self.data[self.current_step]
        
        # 持仓信息
        portfolio_features = np.array([
            self.position,  # 当前持仓
            (self.portfolio_value - self.initial_balance) / self.initial_balance,  # 总收益率
            len(self.trade_history) / 100.0  # 交易次数（归一化）
        ])
        
        return np.concatenate([market_features, portfolio_features]).astype(np.float32)
    
    def _calculate_reward(self, position_return: float, transaction_cost: float):
        """计算奖励函数"""
        # 基础收益奖励
        reward = position_return * 100  # 放大收益信号
        
        # 交易成本惩罚
        reward -= transaction_cost * 50
        
        # 风险调整
        if len(self.returns) > 20:
            volatility = np.std(self.returns[-20:])
            if volatility > 0:
                reward -= volatility * 10  # 波动率惩罚
        
        # 持仓过度惩罚
        if abs(self.position) > 0.8:
            reward -= abs(self.position) * 5
        
        return reward


class ActorNetwork(nn.Module):
    """策略网络（Actor）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 策略输出
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.mean_head(x))  # 限制在[-1, 1]
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return log_prob


class CriticNetwork(nn.Module):
    """价值网络（Critic）"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)
        
        return value


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # 分配GPU内存
        self.gpu_memory = allocate_gpu_memory(GPUTaskType.REINFORCEMENT_LEARNING, "ppo_agent", 2048)
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.AI_TRAINING_LIGHT, [9, 10, 11, 12])
        
        logger.info("PPO智能体初始化完成")
    
    def get_action(self, state: np.ndarray, training: bool = True):
        """获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
                value = self.critic(state_tensor)
                
                return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
            else:
                mean, _ = self.actor(state_tensor)
                return mean.cpu().numpy()[0], None, None
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储转换"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def update(self):
        """更新网络"""
        if len(self.buffer['states']) < self.config.batch_size:
            return {}
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        rewards = torch.FloatTensor(self.buffer['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.buffer['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        dones = torch.BoolTensor(self.buffer['dones']).to(self.device)
        
        # 计算优势和回报
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.config.epochs_per_update):
            # 随机采样批次
            indices = torch.randperm(len(states))[:self.config.batch_size]
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            batch_old_log_probs = old_log_probs[indices]
            
            # 计算新的策略概率
            new_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            new_values = self.critic(batch_states).squeeze()
            value_loss = F.mse_loss(new_values, batch_returns)
            
            # 熵损失
            _, log_std = self.actor(batch_states)
            entropy_loss = -torch.mean(log_std)
            
            # 总损失
            actor_loss = policy_loss + self.config.entropy_coef * entropy_loss
            
            # 更新网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # KL散度检查
            with torch.no_grad():
                kl_div = torch.mean(batch_old_log_probs - new_log_probs)
                if kl_div > self.config.target_kl:
                    break
        
        # 清空缓冲区
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / self.config.epochs_per_update,
            'value_loss': total_value_loss / self.config.epochs_per_update,
            'kl_divergence': kl_div.item()
        }
    
    def _compute_gae(self, rewards, values, dones):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def clear_buffer(self):
        """清空缓冲区"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"模型已从 {path} 加载")


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, env: TradingEnvironment, agent: PPOAgent, config: TrainingConfig):
        self.env = env
        self.agent = agent
        self.config = config
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_metrics = []
        
        self.is_training = False
        
    async def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """训练智能体"""
        self.is_training = True
        logger.info(f"开始训练，共 {num_episodes} 轮")
        
        for episode in range(num_episodes):
            if not self.is_training:
                break
            
            # 运行一轮
            metrics = await self._run_episode(training=True)
            
            # 记录统计
            self.episode_rewards.append(metrics.episode_reward)
            self.episode_lengths.append(metrics.episode_length)
            self.training_metrics.append(metrics)
            
            # 更新智能体
            if len(self.agent.buffer['states']) >= self.config.update_frequency:
                update_info = self.agent.update()
                metrics.policy_loss = update_info.get('policy_loss', 0)
                metrics.value_loss = update_info.get('value_loss', 0)
            
            # 日志输出
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards))
                avg_length = np.mean(list(self.episode_lengths))
                
                logger.info(
                    f"Episode {episode}: "
                    f"Reward={metrics.episode_reward:.2f}, "
                    f"AvgReward={avg_reward:.2f}, "
                    f"Length={metrics.episode_length}, "
                    f"WinRate={metrics.win_rate:.2f}, "
                    f"Return={metrics.total_return:.4f}"
                )
            
            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                model_path = f"models/ppo_episode_{episode}.pth"
                Path("models").mkdir(exist_ok=True)
                self.agent.save_model(model_path)
        
        logger.info("训练完成")
    
    async def _run_episode(self, training: bool = True) -> ModelMetrics:
        """运行一轮"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        trades = []
        
        while True:
            # 获取动作
            if training:
                action, log_prob, value = self.agent.get_action(state, training=True)
            else:
                action, _, _ = self.agent.get_action(state, training=False)
                log_prob, value = None, None
            
            # 执行动作
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # 存储转换
            if training and log_prob is not None:
                self.agent.store_transition(state, action, reward, value, log_prob, done or truncated)
            
            # 记录交易
            if abs(action[0] - self.env.position) > 0.01:
                trades.append({
                    'action': action[0],
                    'reward': reward,
                    'portfolio_value': info['portfolio_value']
                })
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        # 计算指标
        metrics = ModelMetrics()
        metrics.episode_reward = episode_reward
        metrics.episode_length = episode_length
        metrics.total_trades = len(trades)
        
        if trades:
            profitable_trades = sum(1 for t in trades if t['reward'] > 0)
            metrics.profitable_trades = profitable_trades
            metrics.win_rate = profitable_trades / len(trades)
        
        metrics.total_return = info.get('total_return', 0)
        
        # 计算夏普比率和最大回撤
        if len(self.env.returns) > 1:
            returns_array = np.array(self.env.returns)
            metrics.volatility = np.std(returns_array)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = np.mean(returns_array) / metrics.volatility * np.sqrt(252)
            
            # 计算最大回撤
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            metrics.max_drawdown = np.min(drawdown)
        
        return metrics
    
    def stop_training(self):
        """停止训练"""
        self.is_training = False
        logger.info("训练已停止")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        if not self.training_metrics:
            return {}
        
        recent_metrics = self.training_metrics[-10:]  # 最近10轮
        
        return {
            'total_episodes': len(self.training_metrics),
            'avg_reward': np.mean([m.episode_reward for m in recent_metrics]),
            'avg_length': np.mean([m.episode_length for m in recent_metrics]),
            'avg_win_rate': np.mean([m.win_rate for m in recent_metrics]),
            'avg_return': np.mean([m.total_return for m in recent_metrics]),
            'avg_sharpe_ratio': np.mean([m.sharpe_ratio for m in recent_metrics]),
            'avg_max_drawdown': np.mean([m.max_drawdown for m in recent_metrics]),
            'best_episode_reward': max([m.episode_reward for m in self.training_metrics]),
            'best_return': max([m.total_return for m in self.training_metrics])
        }


# 全局训练器实例
rl_trainer = None


def create_trainer(data: np.ndarray, config: TrainingConfig) -> RLTrainer:
    """创建训练器"""
    global rl_trainer
    
    # 创建环境
    env = TradingEnvironment(data)
    
    # 创建智能体
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, config)
    
    # 创建训练器
    rl_trainer = RLTrainer(env, agent, config)
    
    return rl_trainer


async def main():
    """测试主函数"""
    logger.info("启动强化学习模型测试...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_steps = 1000
    n_features = 10
    
    # 模拟价格数据
    prices = np.cumsum(np.random.randn(n_steps) * 0.01) + 100
    features = np.random.randn(n_steps, n_features - 1)
    data = np.column_stack([prices, features])
    
    # 创建配置
    config = TrainingConfig(
        model_type=ModelType.PPO,
        learning_rate=3e-4,
        batch_size=64,
        update_frequency=512
    )
    
    # 创建训练器
    trainer = create_trainer(data, config)
    
    try:
        # 开始训练
        await trainer.train(num_episodes=100, save_interval=50)
        
        # 获取训练统计
        stats = trainer.get_training_stats()
        logger.info(f"训练统计: {stats}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
        trainer.stop_training()
    finally:
        # 清理GPU内存
        if trainer.agent.gpu_memory:
            deallocate_gpu_memory("ppo_agent")


if __name__ == "__main__":
    asyncio.run(main())
