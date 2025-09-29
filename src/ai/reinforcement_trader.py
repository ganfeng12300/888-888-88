#!/usr/bin/env python3
"""
🎯 强化学习交易员 - GPU加速训练
使用深度强化学习进行实盘交易决策
专为生产级实盘交易设计，支持PPO、A3C、SAC等算法
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import gym
from gym import spaces
import collections
import random
import pickle
import os

@dataclass
class TradingState:
    """交易状态"""
    price: float
    volume: float
    rsi: float
    macd: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    atr: float
    volume_sma: float
    price_change: float
    volatility: float
    sentiment: float
    news_impact: float
    time_of_day: float
    day_of_week: float
    position: float  # 当前仓位 -1到1
    unrealized_pnl: float
    account_balance: float
    drawdown: float
    trades_today: int
    win_rate: float

@dataclass
class TradingAction:
    """交易动作"""
    action_type: str  # 'buy', 'sell', 'hold'
    position_size: float  # 0到1之间的仓位大小
    confidence: float  # 动作置信度
    stop_loss: float  # 止损价格
    take_profit: float  # 止盈价格

@dataclass
class Experience:
    """经验回放数据"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float

class TradingEnvironment:
    """交易环境"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0.0  # 当前仓位
        self.entry_price = 0.0
        self.max_position = 1.0  # 最大仓位
        self.transaction_cost = 0.001  # 交易成本
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.trades_count = 0
        self.winning_trades = 0
        self.current_step = 0
        self.max_steps = 1000
        
        # 状态空间：21维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        
        # 动作空间：3个离散动作 (买入、卖出、持有)
        self.action_space = spaces.Discrete(3)
        
        self.state_history = collections.deque(maxlen=100)
        self.reward_history = collections.deque(maxlen=100)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.trades_count = 0
        self.winning_trades = 0
        self.current_step = 0
        
        # 返回初始状态
        return self._get_current_state()
    
    def step(self, action: int, market_data: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.current_step += 1
        
        # 解析动作
        action_type = ['buy', 'sell', 'hold'][action]
        
        # 计算奖励
        reward = self._calculate_reward(action_type, market_data)
        
        # 执行交易
        self._execute_trade(action_type, market_data)
        
        # 更新状态
        next_state = self._get_current_state(market_data)
        
        # 检查是否结束
        done = self._is_done()
        
        # 信息
        info = {
            'balance': self.current_balance,
            'position': self.position,
            'drawdown': self.max_drawdown,
            'trades': self.trades_count,
            'win_rate': self.winning_trades / max(self.trades_count, 1)
        }
        
        return next_state, reward, done, info
    
    def _get_current_state(self, market_data: Dict[str, float] = None) -> np.ndarray:
        """获取当前状态"""
        if market_data is None:
            market_data = {}
        
        state = np.array([
            market_data.get('price', 0.0) / 10000.0,  # 标准化价格
            market_data.get('volume', 0.0) / 1000000.0,  # 标准化成交量
            market_data.get('rsi', 50.0) / 100.0,  # RSI
            market_data.get('macd', 0.0),  # MACD
            market_data.get('bb_position', 0.5),  # 布林带位置
            market_data.get('atr', 0.0) / 100.0,  # ATR
            market_data.get('price_change', 0.0),  # 价格变化
            market_data.get('volatility', 0.0),  # 波动率
            market_data.get('sentiment', 0.0),  # 情感分数
            market_data.get('news_impact', 0.0),  # 新闻影响
            market_data.get('time_of_day', 0.5),  # 时间
            market_data.get('day_of_week', 0.5),  # 星期
            self.position,  # 当前仓位
            (self.current_balance - self.initial_balance) / self.initial_balance,  # 收益率
            self.max_drawdown,  # 最大回撤
            self.trades_count / 100.0,  # 交易次数
            self.winning_trades / max(self.trades_count, 1),  # 胜率
            min(self.current_step / self.max_steps, 1.0),  # 进度
            market_data.get('support_level', 0.0) / 10000.0,  # 支撑位
            market_data.get('resistance_level', 0.0) / 10000.0,  # 阻力位
            market_data.get('trend_strength', 0.0)  # 趋势强度
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, action_type: str, market_data: Dict[str, float]) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 基础收益奖励
        if self.position != 0:
            price_change = market_data.get('price_change', 0.0)
            position_reward = self.position * price_change * 100  # 放大奖励
            reward += position_reward
        
        # 交易成本惩罚
        if action_type in ['buy', 'sell'] and action_type != 'hold':
            reward -= self.transaction_cost * 10  # 交易成本惩罚
        
        # 风险调整奖励
        if self.max_drawdown > 0.1:  # 回撤超过10%
            reward -= 5.0
        elif self.max_drawdown > 0.05:  # 回撤超过5%
            reward -= 2.0
        
        # 胜率奖励
        if self.trades_count > 10:
            win_rate = self.winning_trades / self.trades_count
            if win_rate > 0.6:
                reward += 2.0
            elif win_rate < 0.4:
                reward -= 1.0
        
        # 持仓时间奖励（避免过度交易）
        if action_type == 'hold' and abs(self.position) > 0.1:
            reward += 0.1
        
        return reward
    
    def _execute_trade(self, action_type: str, market_data: Dict[str, float]):
        """执行交易"""
        current_price = market_data.get('price', 0.0)
        
        if action_type == 'buy' and self.position < self.max_position:
            # 买入
            trade_size = min(0.2, self.max_position - self.position)  # 每次最多买入20%
            self.position += trade_size
            self.entry_price = current_price
            self.trades_count += 1
            
        elif action_type == 'sell' and self.position > -self.max_position:
            # 卖出
            if self.position > 0:
                # 平多仓
                pnl = (current_price - self.entry_price) * self.position
                self.current_balance += pnl * self.initial_balance
                if pnl > 0:
                    self.winning_trades += 1
            
            trade_size = min(0.2, self.position + self.max_position)  # 每次最多卖出20%
            self.position -= trade_size
            self.entry_price = current_price
            self.trades_count += 1
        
        # 更新最大回撤
        self.peak_balance = max(self.peak_balance, self.current_balance)
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        return (self.current_step >= self.max_steps or 
                self.max_drawdown > 0.2 or  # 回撤超过20%
                self.current_balance < self.initial_balance * 0.5)  # 亏损超过50%

class PPONetwork(nn.Module):
    """PPO网络"""
    
    def __init__(self, state_dim: int = 21, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 策略网络（Actor）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络（Critic）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作"""
        policy, value = self.forward(state)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

class ReinforcementTrader:
    """强化学习交易员"""
    
    def __init__(self, device: str = None, model_path: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or "models/reinforcement_trader.pth"
        
        # 初始化环境和网络
        self.env = TradingEnvironment()
        self.network = PPONetwork().to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=3e-4, weight_decay=0.01)
        
        # PPO参数
        self.gamma = 0.99  # 折扣因子
        self.lambda_gae = 0.95  # GAE参数
        self.clip_epsilon = 0.2  # PPO裁剪参数
        self.entropy_coef = 0.01  # 熵系数
        self.value_coef = 0.5  # 价值损失系数
        self.max_grad_norm = 0.5  # 梯度裁剪
        
        # 训练参数
        self.batch_size = 64
        self.update_epochs = 10
        self.buffer_size = 2048
        self.experience_buffer = []
        
        # 性能统计
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'profitable_trades': 0
        }
        
        # 实时交易状态
        self.current_position = 0.0
        self.last_action = 'hold'
        self.last_confidence = 0.0
        self.performance_score = 0.5
        
        # 加载预训练模型
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        
        logger.info(f"🎯 强化学习交易员初始化完成 - 设备: {self.device}")
    
    async def get_trading_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取交易信号"""
        try:
            # 准备状态
            state = self.env._get_current_state(market_data)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取动作
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(state_tensor)
                policy, _ = self.network(state_tensor)
                
                action_idx = int(action.cpu().numpy()[0])
                confidence = float(torch.max(policy).cpu().numpy())
                
            # 转换为交易信号
            action_map = {0: 'buy', 1: 'sell', 2: 'hold'}
            signal_map = {'buy': 1.0, 'sell': -1.0, 'hold': 0.0}
            
            action_type = action_map[action_idx]
            signal_strength = signal_map[action_type]
            
            # 调整信号强度基于置信度
            if action_type != 'hold':
                signal_strength *= confidence
            
            # 更新状态
            self.last_action = action_type
            self.last_confidence = confidence
            
            return {
                'signal': signal_strength,
                'confidence': confidence,
                'action': action_type,
                'position_size': min(confidence * 0.1, 0.05),  # 最大5%仓位
                'reasoning': f"RL决策: {action_type} (置信度: {confidence:.3f})",
                'model_id': 'reinforcement_trader',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 强化学习信号生成失败: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.1,
                'action': 'hold',
                'position_size': 0.0,
                'reasoning': f"错误: {str(e)}",
                'model_id': 'reinforcement_trader',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def train_episode(self, market_data_sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """训练一个回合"""
        try:
            # 重置环境
            state = self.env.reset()
            episode_reward = 0.0
            episode_experiences = []
            
            for step, market_data in enumerate(market_data_sequence):
                # 获取动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.network.get_action(state_tensor)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(int(action.cpu().numpy()[0]), market_data)
                
                # 存储经验
                experience = Experience(
                    state=state,
                    action=int(action.cpu().numpy()[0]),
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=float(log_prob.cpu().numpy()[0]),
                    value=float(value.cpu().numpy()[0])
                )
                episode_experiences.append(experience)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 添加到经验缓冲区
            self.experience_buffer.extend(episode_experiences)
            
            # 限制缓冲区大小
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer = self.experience_buffer[-self.buffer_size:]
            
            # 更新网络
            if len(self.experience_buffer) >= self.batch_size:
                await self._update_network()
            
            # 更新统计
            self.training_stats['episodes'] += 1
            self.training_stats['total_reward'] += episode_reward
            self.training_stats['avg_reward'] = self.training_stats['total_reward'] / self.training_stats['episodes']
            
            # 更新性能分数
            self.performance_score = min(max(self.training_stats['avg_reward'] / 100.0 + 0.5, 0.0), 1.0)
            
            return {
                'episode_reward': episode_reward,
                'episode_length': len(episode_experiences),
                'final_balance': info.get('balance', 0.0),
                'win_rate': info.get('win_rate', 0.0),
                'max_drawdown': info.get('drawdown', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ 强化学习训练失败: {e}")
            return {'episode_reward': 0.0, 'episode_length': 0, 'error': str(e)}
    
    async def _update_network(self):
        """更新网络"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return
            
            # 采样批次
            batch = random.sample(self.experience_buffer, self.batch_size)
            
            # 准备数据
            states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
            old_log_probs = torch.FloatTensor([exp.log_prob for exp in batch]).to(self.device)
            old_values = torch.FloatTensor([exp.value for exp in batch]).to(self.device)
            
            # 计算优势和回报
            with torch.no_grad():
                _, next_values = self.network(next_states)
                next_values = next_values.squeeze(-1)
                
                # GAE计算
                advantages = torch.zeros_like(rewards)
                returns = torch.zeros_like(rewards)
                
                gae = 0
                for t in reversed(range(len(batch))):
                    if t == len(batch) - 1:
                        next_value = next_values[t] if not dones[t] else 0
                    else:
                        next_value = old_values[t + 1]
                    
                    delta = rewards[t] + self.gamma * next_value - old_values[t]
                    gae = delta + self.gamma * self.lambda_gae * gae
                    advantages[t] = gae
                    returns[t] = advantages[t] + old_values[t]
                
                # 标准化优势
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO更新
            for _ in range(self.update_epochs):
                # 前向传播
                policies, values = self.network(states)
                values = values.squeeze(-1)
                
                # 计算策略损失
                dist = Categorical(policies)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(values, returns)
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            logger.debug(f"🎯 RL网络更新完成 - 策略损失: {policy_loss.item():.6f}, 价值损失: {value_loss.item():.6f}")
            
        except Exception as e:
            logger.warning(f"⚠️ RL网络更新失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'model_id': 'reinforcement_trader',
            'model_name': '强化学习交易员',
            'device': self.device,
            'current_position': self.current_position,
            'last_action': self.last_action,
            'last_confidence': self.last_confidence,
            'performance_score': self.performance_score,
            'training_stats': self.training_stats,
            'buffer_size': len(self.experience_buffer),
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'is_training': len(self.experience_buffer) >= self.batch_size
        }
    
    def save_model(self, filepath: str = None) -> bool:
        """保存模型"""
        try:
            filepath = filepath or self.model_path
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'performance_score': self.performance_score,
                'hyperparameters': {
                    'gamma': self.gamma,
                    'lambda_gae': self.lambda_gae,
                    'clip_epsilon': self.clip_epsilon,
                    'entropy_coef': self.entropy_coef,
                    'value_coef': self.value_coef
                }
            }, filepath)
            
            logger.info(f"💾 强化学习模型已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """加载模型"""
        try:
            filepath = filepath or self.model_path
            if not os.path.exists(filepath):
                logger.warning(f"⚠️ 模型文件不存在: {filepath}")
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            self.performance_score = checkpoint.get('performance_score', 0.5)
            
            # 加载超参数
            hyperparams = checkpoint.get('hyperparameters', {})
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.lambda_gae = hyperparams.get('lambda_gae', self.lambda_gae)
            self.clip_epsilon = hyperparams.get('clip_epsilon', self.clip_epsilon)
            self.entropy_coef = hyperparams.get('entropy_coef', self.entropy_coef)
            self.value_coef = hyperparams.get('value_coef', self.value_coef)
            
            logger.info(f"📂 强化学习模型已加载: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False

# 全局实例
reinforcement_trader = ReinforcementTrader()

def initialize_reinforcement_trader(device: str = None, model_path: str = None) -> ReinforcementTrader:
    """初始化强化学习交易员"""
    global reinforcement_trader
    reinforcement_trader = ReinforcementTrader(device, model_path)
    return reinforcement_trader

if __name__ == "__main__":
    # 测试代码
    async def test_reinforcement_trader():
        trader = initialize_reinforcement_trader()
        
        # 测试交易信号
        market_data = {
            'price': 50000.0,
            'volume': 1000000.0,
            'rsi': 65.0,
            'macd': 0.1,
            'bb_position': 0.7,
            'price_change': 0.02,
            'volatility': 0.15,
            'sentiment': 0.3
        }
        
        signal = await trader.get_trading_signal(market_data)
        print(f"交易信号: {signal}")
        
        # 状态报告
        status = trader.get_status()
        print(f"状态报告: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    asyncio.run(test_reinforcement_trader())

