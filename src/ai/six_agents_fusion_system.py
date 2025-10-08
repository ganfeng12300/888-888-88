#!/usr/bin/env python3
"""
🧠 六大智能体融合系统 - 终极AI决策引擎
Six Agents Fusion System - Ultimate AI Decision Engine

6大智能体等级系统：
Level 6: 元学习AI (Meta Learning Commander) - 学习如何学习
Level 5: 集成学习协调AI (Integration Learning Coordinator) - 多模型融合决策  
Level 4: 强化学习执行AI (Reinforcement Learning Executor) - Q-Learning交易决策
Level 3: 时序深度学习AI (Time Series Deep Learning AI) - LSTM/Transformer预测
Level 2: 迁移学习适配AI (Transfer Learning Adapter) - 跨市场知识迁移
Level 1: 专家系统守护AI (Expert System Guardian) - 规则引擎保护

实现智能体间协作、性能驱动进化、实时决策融合
"""

import asyncio
import json
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
import psutil
import GPUtil

# 导入现有AI模块
try:
    from .meta_learning_commander import MetaLearningCommander, MetaDecision
    from .ai_level_evolution import AILevelEvolutionSystem, AILevelInfo
    from .reinforcement_trader import ReinforcementTrader
    from .expert_system_guardian import ExpertSystemGuardian
    from .transfer_learning_adapter import TransferLearningAdapter
    from .ensemble_brain_trust import EnsembleBrainTrust
except ImportError as e:
    logger.warning(f"导入现有AI模块失败: {e}")


class AgentLevel(Enum):
    """智能体等级"""
    EXPERT_GUARDIAN = 1      # 专家系统守护AI
    TRANSFER_ADAPTER = 2     # 迁移学习适配AI  
    TIMESERIES_PROPHET = 3   # 时序深度学习AI
    REINFORCEMENT_EXECUTOR = 4  # 强化学习执行AI
    INTEGRATION_COORDINATOR = 5  # 集成学习协调AI
    META_COMMANDER = 6       # 元学习AI


@dataclass
class AgentStatus:
    """智能体状态"""
    agent_id: str
    agent_name: str
    level: int
    confidence: float
    prediction: float
    performance_score: float
    last_update: datetime
    is_active: bool
    gpu_usage: float
    memory_usage: float
    training_progress: float
    decision_count: int
    success_rate: float


@dataclass
class FusionDecision:
    """融合决策结果"""
    final_signal: float      # -1到1之间的最终信号
    confidence: float        # 0到1之间的置信度
    risk_level: float        # 0到1之间的风险等级
    agent_contributions: Dict[str, float]  # 各智能体贡献度
    decision_path: List[str]  # 决策路径
    reasoning: str           # 决策推理
    timestamp: datetime
    execution_time_ms: float


class SixAgentsFusionSystem:
    """六大智能体融合系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化融合系统"""
        self.config = config or {}
        self.agents = {}
        self.fusion_weights = {}
        self.performance_history = {}
        self.decision_history = []
        self.is_running = False
        self.lock = threading.Lock()
        
        # 系统配置
        self.max_decision_history = 10000
        self.performance_window = 100
        self.weight_update_interval = 60  # 秒
        self.min_confidence_threshold = 0.3
        
        # 初始化智能体权重
        self._initialize_fusion_weights()
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info("🧠 六大智能体融合系统初始化完成")
    
    def _initialize_fusion_weights(self):
        """初始化融合权重"""
        # 基于等级的初始权重分配
        base_weights = {
            AgentLevel.META_COMMANDER: 0.25,        # 元学习AI - 最高权重
            AgentLevel.INTEGRATION_COORDINATOR: 0.20,  # 集成学习协调AI
            AgentLevel.REINFORCEMENT_EXECUTOR: 0.20,   # 强化学习执行AI
            AgentLevel.TIMESERIES_PROPHET: 0.15,       # 时序深度学习AI
            AgentLevel.TRANSFER_ADAPTER: 0.12,         # 迁移学习适配AI
            AgentLevel.EXPERT_GUARDIAN: 0.08           # 专家系统守护AI
        }
        
        for level, weight in base_weights.items():
            self.fusion_weights[level.value] = weight
        
        logger.info(f"初始化融合权重: {self.fusion_weights}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        self.is_running = True
        
        # 权重更新任务
        threading.Thread(
            target=self._weight_update_loop,
            daemon=True,
            name="WeightUpdateThread"
        ).start()
        
        # 性能监控任务
        threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True,
            name="PerformanceMonitorThread"
        ).start()
    
    async def register_agent(self, agent_level: AgentLevel, agent_instance: Any):
        """注册智能体"""
        with self.lock:
            agent_id = f"agent_level_{agent_level.value}"
            
            self.agents[agent_id] = {
                'level': agent_level.value,
                'instance': agent_instance,
                'status': AgentStatus(
                    agent_id=agent_id,
                    agent_name=agent_level.name,
                    level=agent_level.value,
                    confidence=0.5,
                    prediction=0.0,
                    performance_score=0.5,
                    last_update=datetime.now(),
                    is_active=True,
                    gpu_usage=0.0,
                    memory_usage=0.0,
                    training_progress=0.0,
                    decision_count=0,
                    success_rate=0.5
                )
            }
            
            # 初始化性能历史
            self.performance_history[agent_id] = []
            
            logger.info(f"注册智能体: {agent_level.name} (Level {agent_level.value})")
    
    async def make_fusion_decision(self, market_data: Dict[str, Any]) -> FusionDecision:
        """执行融合决策"""
        start_time = time.time()
        
        try:
            # 收集各智能体决策
            agent_decisions = await self._collect_agent_decisions(market_data)
            
            # 执行决策融合
            fusion_result = await self._fuse_decisions(agent_decisions)
            
            # 记录决策历史
            execution_time = (time.time() - start_time) * 1000
            fusion_result.execution_time_ms = execution_time
            
            self._record_decision(fusion_result)
            
            logger.info(f"融合决策完成: 信号={fusion_result.final_signal:.4f}, "
                       f"置信度={fusion_result.confidence:.4f}, "
                       f"执行时间={execution_time:.2f}ms")
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"融合决策失败: {e}")
            # 返回保守决策
            return FusionDecision(
                final_signal=0.0,
                confidence=0.0,
                risk_level=1.0,
                agent_contributions={},
                decision_path=["ERROR"],
                reasoning=f"决策失败: {str(e)}",
                timestamp=datetime.now(),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _collect_agent_decisions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集各智能体决策"""
        decisions = {}
        
        # 并行收集决策
        tasks = []
        for agent_id, agent_info in self.agents.items():
            if agent_info['status'].is_active:
                task = self._get_agent_decision(agent_id, agent_info, market_data)
                tasks.append((agent_id, task))
        
        # 等待所有决策完成
        for agent_id, task in tasks:
            try:
                decision = await asyncio.wait_for(task, timeout=5.0)
                decisions[agent_id] = decision
            except asyncio.TimeoutError:
                logger.warning(f"智能体 {agent_id} 决策超时")
                decisions[agent_id] = self._get_default_decision(agent_id)
            except Exception as e:
                logger.error(f"智能体 {agent_id} 决策失败: {e}")
                decisions[agent_id] = self._get_default_decision(agent_id)
        
        return decisions
    
    async def _get_agent_decision(self, agent_id: str, agent_info: Dict, market_data: Dict) -> Dict:
        """获取单个智能体决策"""
        try:
            agent_instance = agent_info['instance']
            level = agent_info['level']
            
            # 根据智能体等级调用不同的决策方法
            if level == AgentLevel.META_COMMANDER.value:
                # 元学习AI决策
                if hasattr(agent_instance, 'make_meta_decision'):
                    result = await agent_instance.make_meta_decision(market_data)
                    return {
                        'signal': result.final_signal if hasattr(result, 'final_signal') else 0.0,
                        'confidence': result.confidence if hasattr(result, 'confidence') else 0.5,
                        'reasoning': result.reasoning if hasattr(result, 'reasoning') else "元学习决策"
                    }
            
            elif level == AgentLevel.INTEGRATION_COORDINATOR.value:
                # 集成学习协调AI决策
                if hasattr(agent_instance, 'coordinate_decision'):
                    result = await agent_instance.coordinate_decision(market_data)
                    return {
                        'signal': result.get('signal', 0.0),
                        'confidence': result.get('confidence', 0.5),
                        'reasoning': result.get('reasoning', "集成协调决策")
                    }
            
            elif level == AgentLevel.REINFORCEMENT_EXECUTOR.value:
                # 强化学习执行AI决策
                if hasattr(agent_instance, 'get_action'):
                    action = await agent_instance.get_action(market_data)
                    return {
                        'signal': action.get('action_value', 0.0),
                        'confidence': action.get('confidence', 0.5),
                        'reasoning': action.get('reasoning', "强化学习决策")
                    }
            
            # 其他智能体的通用决策接口
            if hasattr(agent_instance, 'predict'):
                prediction = await agent_instance.predict(market_data)
                return {
                    'signal': prediction.get('signal', 0.0),
                    'confidence': prediction.get('confidence', 0.5),
                    'reasoning': prediction.get('reasoning', f"Level {level} 决策")
                }
            
            # 默认决策
            return self._get_default_decision(agent_id)
            
        except Exception as e:
            logger.error(f"获取智能体 {agent_id} 决策失败: {e}")
            return self._get_default_decision(agent_id)
    
    def _get_default_decision(self, agent_id: str) -> Dict:
        """获取默认决策"""
        return {
            'signal': 0.0,
            'confidence': 0.1,
            'reasoning': f"默认决策 - {agent_id}"
        }
    
    async def _fuse_decisions(self, agent_decisions: Dict[str, Any]) -> FusionDecision:
        """融合各智能体决策"""
        if not agent_decisions:
            return FusionDecision(
                final_signal=0.0,
                confidence=0.0,
                risk_level=1.0,
                agent_contributions={},
                decision_path=["NO_AGENTS"],
                reasoning="无可用智能体",
                timestamp=datetime.now(),
                execution_time_ms=0.0
            )
        
        # 计算加权融合信号
        weighted_signals = []
        weighted_confidences = []
        agent_contributions = {}
        decision_path = []
        
        total_weight = 0.0
        
        for agent_id, decision in agent_decisions.items():
            agent_info = self.agents.get(agent_id, {})
            level = agent_info.get('level', 1)
            
            # 获取智能体权重
            base_weight = self.fusion_weights.get(level, 0.1)
            
            # 基于性能调整权重
            performance_weight = self._calculate_performance_weight(agent_id)
            final_weight = base_weight * performance_weight
            
            # 基于置信度调整权重
            confidence = decision.get('confidence', 0.5)
            if confidence < self.min_confidence_threshold:
                final_weight *= 0.5  # 低置信度降权
            
            signal = decision.get('signal', 0.0)
            
            weighted_signals.append(signal * final_weight)
            weighted_confidences.append(confidence * final_weight)
            
            agent_contributions[agent_id] = final_weight
            decision_path.append(f"L{level}({final_weight:.3f})")
            
            total_weight += final_weight
        
        # 归一化权重
        if total_weight > 0:
            final_signal = sum(weighted_signals) / total_weight
            final_confidence = sum(weighted_confidences) / total_weight
            
            # 归一化贡献度
            for agent_id in agent_contributions:
                agent_contributions[agent_id] /= total_weight
        else:
            final_signal = 0.0
            final_confidence = 0.0
        
        # 计算风险等级
        risk_level = self._calculate_risk_level(final_signal, final_confidence, agent_decisions)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(agent_decisions, agent_contributions, final_signal)
        
        return FusionDecision(
            final_signal=np.clip(final_signal, -1.0, 1.0),
            confidence=np.clip(final_confidence, 0.0, 1.0),
            risk_level=np.clip(risk_level, 0.0, 1.0),
            agent_contributions=agent_contributions,
            decision_path=decision_path,
            reasoning=reasoning,
            timestamp=datetime.now(),
            execution_time_ms=0.0
        )
    
    def _calculate_performance_weight(self, agent_id: str) -> float:
        """计算基于性能的权重调整"""
        history = self.performance_history.get(agent_id, [])
        if len(history) < 10:
            return 1.0  # 历史数据不足，使用默认权重
        
        # 计算最近性能
        recent_performance = history[-self.performance_window:]
        avg_performance = np.mean([p['success_rate'] for p in recent_performance])
        
        # 性能权重调整 (0.5 - 2.0)
        performance_weight = 0.5 + (avg_performance * 1.5)
        return np.clip(performance_weight, 0.1, 2.0)
    
    def _calculate_risk_level(self, signal: float, confidence: float, decisions: Dict) -> float:
        """计算风险等级"""
        # 基于信号强度的风险
        signal_risk = abs(signal)
        
        # 基于置信度的风险 (低置信度 = 高风险)
        confidence_risk = 1.0 - confidence
        
        # 基于决策一致性的风险
        signals = [d.get('signal', 0.0) for d in decisions.values()]
        if len(signals) > 1:
            signal_std = np.std(signals)
            consistency_risk = min(signal_std, 1.0)
        else:
            consistency_risk = 0.5
        
        # 综合风险评估
        total_risk = (signal_risk * 0.3 + confidence_risk * 0.4 + consistency_risk * 0.3)
        return np.clip(total_risk, 0.0, 1.0)
    
    def _generate_reasoning(self, decisions: Dict, contributions: Dict, final_signal: float) -> str:
        """生成决策推理"""
        reasoning_parts = []
        
        # 主导智能体
        if contributions:
            dominant_agent = max(contributions.items(), key=lambda x: x[1])
            agent_info = self.agents.get(dominant_agent[0], {})
            level = agent_info.get('level', 0)
            reasoning_parts.append(f"主导: Level {level} ({dominant_agent[1]:.2%})")
        
        # 信号方向
        if final_signal > 0.1:
            reasoning_parts.append("看涨信号")
        elif final_signal < -0.1:
            reasoning_parts.append("看跌信号")
        else:
            reasoning_parts.append("中性信号")
        
        # 参与智能体数量
        active_agents = len([d for d in decisions.values() if d.get('confidence', 0) > 0.3])
        reasoning_parts.append(f"{active_agents}个智能体参与")
        
        return " | ".join(reasoning_parts)
    
    def _record_decision(self, decision: FusionDecision):
        """记录决策历史"""
        with self.lock:
            self.decision_history.append(decision)
            
            # 限制历史记录数量
            if len(self.decision_history) > self.max_decision_history:
                self.decision_history = self.decision_history[-self.max_decision_history:]
    
    def _weight_update_loop(self):
        """权重更新循环"""
        while self.is_running:
            try:
                time.sleep(self.weight_update_interval)
                self._update_fusion_weights()
            except Exception as e:
                logger.error(f"权重更新失败: {e}")
    
    def _performance_monitor_loop(self):
        """性能监控循环"""
        while self.is_running:
            try:
                time.sleep(30)  # 30秒监控一次
                self._update_agent_performance()
            except Exception as e:
                logger.error(f"性能监控失败: {e}")
    
    def _update_fusion_weights(self):
        """更新融合权重"""
        try:
            with self.lock:
                for agent_id, agent_info in self.agents.items():
                    level = agent_info['level']
                    performance_weight = self._calculate_performance_weight(agent_id)
                    
                    # 动态调整权重
                    base_weight = {
                        6: 0.25, 5: 0.20, 4: 0.20, 3: 0.15, 2: 0.12, 1: 0.08
                    }.get(level, 0.1)
                    
                    new_weight = base_weight * performance_weight
                    self.fusion_weights[level] = new_weight
                
                # 归一化权重
                total_weight = sum(self.fusion_weights.values())
                if total_weight > 0:
                    for level in self.fusion_weights:
                        self.fusion_weights[level] /= total_weight
                
                logger.debug(f"更新融合权重: {self.fusion_weights}")
                
        except Exception as e:
            logger.error(f"更新融合权重失败: {e}")
    
    def _update_agent_performance(self):
        """更新智能体性能"""
        try:
            for agent_id, agent_info in self.agents.items():
                # 更新系统资源使用情况
                gpu_usage = 0.0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                except:
                    pass
                
                memory_usage = psutil.virtual_memory().percent
                
                # 更新智能体状态
                status = agent_info['status']
                status.gpu_usage = gpu_usage
                status.memory_usage = memory_usage
                status.last_update = datetime.now()
                
                # 记录性能历史
                performance_record = {
                    'timestamp': time.time(),
                    'success_rate': status.success_rate,
                    'confidence': status.confidence,
                    'performance_score': status.performance_score
                }
                
                if agent_id not in self.performance_history:
                    self.performance_history[agent_id] = []
                
                self.performance_history[agent_id].append(performance_record)
                
                # 限制历史记录数量
                if len(self.performance_history[agent_id]) > self.performance_window * 2:
                    self.performance_history[agent_id] = self.performance_history[agent_id][-self.performance_window:]
                
        except Exception as e:
            logger.error(f"更新智能体性能失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            return {
                'total_agents': len(self.agents),
                'active_agents': len([a for a in self.agents.values() if a['status'].is_active]),
                'fusion_weights': self.fusion_weights.copy(),
                'decision_count': len(self.decision_history),
                'last_decision_time': self.decision_history[-1].timestamp if self.decision_history else None,
                'agents_status': {
                    agent_id: {
                        'level': info['status'].level,
                        'confidence': info['status'].confidence,
                        'performance_score': info['status'].performance_score,
                        'is_active': info['status'].is_active,
                        'decision_count': info['status'].decision_count,
                        'success_rate': info['status'].success_rate
                    }
                    for agent_id, info in self.agents.items()
                }
            }
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭六大智能体融合系统...")
        self.is_running = False
        
        # 等待后台线程结束
        time.sleep(2)
        
        logger.info("六大智能体融合系统已关闭")


# 全局实例
_fusion_system = None

def get_fusion_system(config: Dict[str, Any] = None) -> SixAgentsFusionSystem:
    """获取融合系统实例"""
    global _fusion_system
    if _fusion_system is None:
        _fusion_system = SixAgentsFusionSystem(config)
    return _fusion_system


if __name__ == "__main__":
    # 测试代码
    async def test_fusion_system():
        """测试融合系统"""
        system = get_fusion_system()
        
        # 模拟市场数据
        market_data = {
            'price': 50000.0,
            'volume': 1000.0,
            'timestamp': time.time()
        }
        
        # 执行融合决策
        decision = await system.make_fusion_decision(market_data)
        
        print(f"融合决策结果:")
        print(f"信号: {decision.final_signal}")
        print(f"置信度: {decision.confidence}")
        print(f"风险等级: {decision.risk_level}")
        print(f"推理: {decision.reasoning}")
        
        # 获取系统状态
        status = system.get_system_status()
        print(f"\n系统状态: {json.dumps(status, indent=2, default=str)}")
    
    # 运行测试
    asyncio.run(test_fusion_system())
