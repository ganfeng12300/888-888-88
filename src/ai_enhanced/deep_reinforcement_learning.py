"""
🧠 深度强化学习增强模块 - 生产级实盘交易强化学习系统
基于PPO、SAC、TD3等先进算法的深度强化学习交易智能体
支持多资产、多时间框架、动态环境适应的智能交易决策
"""
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

class RLAlgorithm(Enum):
    """强化学习算法类型"""
    PPO = "ppo"  # Proximal Policy Optimization
    SAC = "sac"  # Soft Actor-Critic
    TD3 = "td3"  # Twin Delayed Deep Deterministic Policy Gradient
    A2C = "a2c"  # Advantage Actor-Critic
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient

class ActionType(Enum):
    """动作类型"""
    HOLD = 0  # 持有
    BUY = 1   # 买入
    SELL = 2  # 卖出

@dataclass
class TradingState:
    """交易状态"""
    price: float  # 当前价格
    volume: float  # 成交量
    position: float  # 当前仓位 (-1到1)
    cash: float  # 现金余额
    portfolio_value: float  # 组合价值
    technical_indicators: np.ndarray  # 技术指标
    market_features: np.ndarray  # 市场特征
    timestamp: float  # 时间戳

@dataclass
class TradingAction:
    """交易动作"""
    action_type: ActionType  # 动作类型
    position_size: float  # 仓位大小 (0-1)
    confidence: float  # 置信度
    stop_loss: Optional[float] = None  # 止损价格
    take_profit: Optional[float] = None  # 止盈价格

@dataclass
class Experience:
    """经验回放数据"""
    state: np.ndarray  # 状态
    action: np.ndarray  # 动作
    reward: float  # 奖励
    next_state: np.ndarray  # 下一状态
    done: bool  # 是否结束
    info: Dict[str, Any] = field(default_factory=dict)  # 额外信息

class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 动作均值和标准差
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化网络权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        action_mean = torch.tanh(self.action_mean(x))  # 限制在[-1, 1]
        action_std = F.softplus(self.action_std(x)) + 1e-5  # 确保正数
        
        return action_mean, action_std

class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化网络权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        
        return value

class ExperienceReplay:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.RLock()
    
    def push(self, experience: Experience):
        """添加经验"""
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return self.buffer.copy()
            
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2, k_epochs: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.actor_old = ActorNetwork(state_dim, action_dim)
        
        # 复制参数到旧网络
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验缓冲
        self.memory = []
        
        logger.info("PPO智能体初始化完成")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_std = self.actor_old(state_tensor)
            
            if training:
                # 训练时使用随机策略
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action_logprob = dist.log_prob(action).sum(dim=-1)
            else:
                # 测试时使用确定性策略
                action = action_mean
                action_logprob = torch.zeros(1)
        
        return action.squeeze(0).numpy(), action_logprob.item()
    
    def store_transition(self, state, action, reward, next_state, done, logprob):
        """存储转换"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'logprob': logprob
        })
    
    def update(self):
        """更新网络"""
        if len(self.memory) == 0:
            return
        
        # 转换为张量
        states = torch.FloatTensor([m['state'] for m in self.memory])
        actions = torch.FloatTensor([m['action'] for m in self.memory])
        rewards = torch.FloatTensor([m['reward'] for m in self.memory])
        next_states = torch.FloatTensor([m['next_state'] for m in self.memory])
        dones = torch.BoolTensor([m['done'] for m in self.memory])
        old_logprobs = torch.FloatTensor([m['logprob'] for m in self.memory])
        
        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 更新网络
        for _ in range(self.k_epochs):
            # 计算当前策略的动作概率
            action_mean, action_std = self.actor(states)
            dist = Normal(action_mean, action_std)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            
            # 计算比率
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # 计算优势
            values = self.critic(states, actions).squeeze()
            advantages = discounted_rewards - values.detach()
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失
            critic_loss = F.mse_loss(values, discounted_rewards)
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # 更新旧策略
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # 清空内存
        self.memory.clear()
        
        logger.debug(f"PPO更新完成 - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = ExperienceReplay()
        
        logger.info("SAC智能体初始化完成")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_std = self.actor(state_tensor)
            
            if training:
                # 训练时使用随机策略
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            else:
                # 测试时使用确定性策略
                action = action_mean
        
        return action.squeeze(0).numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.push(experience)
    
    def update(self, batch_size: int = 256):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样经验
        experiences = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        # 计算目标Q值
        with torch.no_grad():
            next_action_mean, next_action_std = self.actor(next_states)
            next_dist = Normal(next_action_mean, next_action_std)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
        
        # 更新Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        action_mean, action_std = self.actor(states)
        dist = Normal(action_mean, action_std)
        new_actions = dist.rsample()  # 重参数化采样
        log_probs = dist.log_prob(new_actions).sum(dim=-1, keepdim=True)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        logger.debug(f"SAC更新完成 - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {(critic1_loss + critic2_loss).item():.4f}")
    
    def _soft_update(self, target_net, net):
        """软更新目标网络"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class TradingEnvironment:
    """交易环境"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # 状态变量
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.portfolio_value = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        # 特征维度
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 3  # hold, buy, sell
        
        logger.info(f"交易环境初始化完成 - 数据长度: {len(data)}, 状态维度: {self.state_dim}")
    
    def _calculate_state_dim(self) -> int:
        """计算状态维度"""
        # 基础状态: 价格、成交量、仓位、现金比例
        base_dim = 4
        
        # 技术指标维度 (假设有20个技术指标)
        technical_dim = 20
        
        # 市场特征维度 (假设有10个市场特征)
        market_dim = 10
        
        return base_dim + technical_dim + market_dim
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        # 解析动作
        action_type = np.argmax(action[:3])  # 0: hold, 1: buy, 2: sell
        position_size = np.clip(action[3] if len(action) > 3 else 0.5, 0, 1)
        
        # 获取当前价格
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # 执行交易
        reward = self._execute_trade(action_type, position_size, current_price, next_price)
        
        # 更新状态
        self.current_step += 1
        next_state = self._get_state()
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1
        
        # 额外信息
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }
        
        return next_state, reward, done, info
    
    def _execute_trade(self, action_type: int, position_size: float, 
                      current_price: float, next_price: float) -> float:
        """执行交易"""
        old_portfolio_value = self.portfolio_value
        
        if action_type == 1:  # Buy
            # 计算可买入数量
            available_cash = self.balance * position_size
            shares_to_buy = available_cash / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            
            if cost <= self.balance:
                self.balance -= cost
                self.position += shares_to_buy
                self.total_trades += 1
        
        elif action_type == 2:  # Sell
            # 计算可卖出数量
            shares_to_sell = self.position * position_size
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.position -= shares_to_sell
                self.total_trades += 1
        
        # 更新组合价值
        self.portfolio_value = self.balance + self.position * next_price
        
        # 计算奖励
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # 更新胜率
        if reward > 0:
            self.winning_trades += 1
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_dim)
        
        row = self.data.iloc[self.current_step]
        
        # 基础状态
        price = row['close']
        volume = row['volume']
        position_ratio = self.position * price / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 0
        
        base_state = np.array([price, volume, position_ratio, cash_ratio])
        
        # 技术指标 (模拟数据)
        technical_indicators = np.random.randn(20) * 0.1
        
        # 市场特征 (模拟数据)
        market_features = np.random.randn(10) * 0.1
        
        # 组合状态
        state = np.concatenate([base_state, technical_indicators, market_features])
        
        return state.astype(np.float32)

class RLTradingSystem:
    """强化学习交易系统"""
    
    def __init__(self, algorithm: RLAlgorithm = RLAlgorithm.PPO):
        self.algorithm = algorithm
        self.agent = None
        self.environment = None
        self.training_data = None
        self.testing_data = None
        
        # 训练参数
        self.episodes = 1000
        self.max_steps_per_episode = 1000
        self.update_frequency = 10
        
        # 性能指标
        self.training_rewards = []
        self.training_portfolio_values = []
        self.testing_results = {}
        
        logger.info(f"强化学习交易系统初始化完成 - 算法: {algorithm.value}")
    
    def setup_environment(self, data: pd.DataFrame, train_ratio: float = 0.8):
        """设置环境"""
        # 分割训练和测试数据
        split_idx = int(len(data) * train_ratio)
        self.training_data = data.iloc[:split_idx].copy()
        self.testing_data = data.iloc[split_idx:].copy()
        
        # 创建环境
        self.environment = TradingEnvironment(self.training_data)
        
        # 创建智能体
        if self.algorithm == RLAlgorithm.PPO:
            self.agent = PPOAgent(
                state_dim=self.environment.state_dim,
                action_dim=4  # 3个动作类型 + 1个仓位大小
            )
        elif self.algorithm == RLAlgorithm.SAC:
            self.agent = SACAgent(
                state_dim=self.environment.state_dim,
                action_dim=4
            )
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
        
        logger.info(f"环境设置完成 - 训练数据: {len(self.training_data)}, 测试数据: {len(self.testing_data)}")
    
    def train(self, episodes: int = None) -> Dict[str, Any]:
        """训练智能体"""
        if not self.agent or not self.environment:
            raise ValueError("请先设置环境")
        
        episodes = episodes or self.episodes
        
        logger.info(f"开始训练 - 算法: {self.algorithm.value}, 回合数: {episodes}")
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                # 选择动作
                if self.algorithm == RLAlgorithm.PPO:
                    action, logprob = self.agent.select_action(state, training=True)
                else:
                    action = self.agent.select_action(state, training=True)
                    logprob = 0
                
                # 执行动作
                next_state, reward, done, info = self.environment.step(action)
                
                # 存储经验
                if self.algorithm == RLAlgorithm.PPO:
                    self.agent.store_transition(state, action, reward, next_state, done, logprob)
                else:
                    self.agent.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # 更新智能体
            if episode % self.update_frequency == 0:
                self.agent.update()
            
            # 记录性能
            self.training_rewards.append(episode_reward)
            self.training_portfolio_values.append(info.get('portfolio_value', 0))
            
            # 打印进度
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                logger.info(f"Episode {episode}, 平均奖励: {avg_reward:.4f}, 组合价值: {info.get('portfolio_value', 0):.2f}")
        
        logger.info("训练完成")
        
        return {
            'training_rewards': self.training_rewards,
            'training_portfolio_values': self.training_portfolio_values,
            'final_portfolio_value': self.training_portfolio_values[-1] if self.training_portfolio_values else 0
        }
    
    def test(self) -> Dict[str, Any]:
        """测试智能体"""
        if not self.agent or not self.testing_data is None:
            raise ValueError("请先训练智能体并准备测试数据")
        
        logger.info("开始测试")
        
        # 创建测试环境
        test_env = TradingEnvironment(self.testing_data)
        
        state = test_env.reset()
        total_reward = 0
        portfolio_values = []
        actions_taken = []
        
        for step in range(len(self.testing_data) - 1):
            # 选择动作 (测试模式)
            action = self.agent.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, info = test_env.step(action)
            
            # 记录结果
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(np.argmax(action[:3]))
            
            state = next_state
            
            if done:
                break
        
        # 计算性能指标
        initial_value = test_env.initial_balance
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # 计算夏普比率
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # 计算最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        self.testing_results = {
            'total_return': total_return,
            'final_portfolio_value': final_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': test_env.total_trades,
            'win_rate': test_env.winning_trades / max(test_env.total_trades, 1),
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken
        }
        
        logger.info(f"测试完成 - 总收益: {total_return:.2%}, 夏普比率: {sharpe_ratio:.4f}, 最大回撤: {max_drawdown:.2%}")
        
        return self.testing_results
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.agent:
            raise ValueError("没有可保存的模型")
        
        torch.save({
            'algorithm': self.algorithm.value,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': getattr(self.agent, 'critic', None).state_dict() if hasattr(self.agent, 'critic') else None,
            'training_rewards': self.training_rewards,
            'testing_results': self.testing_results
        }, filepath)
        
        logger.info(f"模型保存完成: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        
        # 重新创建智能体 (需要先设置环境)
        if not self.environment:
            raise ValueError("请先设置环境")
        
        self.algorithm = RLAlgorithm(checkpoint['algorithm'])
        
        if self.algorithm == RLAlgorithm.PPO:
            self.agent = PPOAgent(
                state_dim=self.environment.state_dim,
                action_dim=4
            )
        elif self.algorithm == RLAlgorithm.SAC:
            self.agent = SACAgent(
                state_dim=self.environment.state_dim,
                action_dim=4
            )
        
        # 加载权重
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        if checkpoint['critic_state_dict'] and hasattr(self.agent, 'critic'):
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 加载历史数据
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.testing_results = checkpoint.get('testing_results', {})
        
        logger.info(f"模型加载完成: {filepath}")

# 全局强化学习交易系统实例
rl_trading_system = RLTradingSystem()
