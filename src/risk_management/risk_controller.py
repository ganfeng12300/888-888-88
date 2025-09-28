"""
🛡️ AI等级驱动风险控制系统
生产级风险管理，基于AI等级动态调整风控参数
实现智能止损、动态杠杆和实时风险监控
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from src.hardware.cpu_manager import CPUTaskType, assign_cpu_cores
from src.ai_models.ai_evolution_system import AIEvolutionSystem, AILevel, ModelType


class RiskLevel(Enum):
    """风险等级"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    STOP_LOSS = "stop_loss"
    MARGIN_CALL = "margin_call"


@dataclass
class RiskMetrics:
    """风险指标"""
    timestamp: float
    portfolio_value: float
    total_exposure: float
    leverage_ratio: float
    var_1d: float  # 1日风险价值
    var_5d: float  # 5日风险价值
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float


@dataclass
class PositionRisk:
    """持仓风险"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    stop_loss_price: float
    take_profit_price: float
    risk_score: float
    max_loss_amount: float
    position_ratio: float  # 占总资金比例


@dataclass
class RiskLimits:
    """风险限制"""
    max_position_size: float = 0.1  # 单个持仓最大比例
    max_total_exposure: float = 1.0  # 总敞口比例
    max_leverage: float = 3.0  # 最大杠杆
    max_daily_loss: float = 0.05  # 最大日损失比例
    max_drawdown: float = 0.15  # 最大回撤比例
    min_liquidity_ratio: float = 0.2  # 最小流动性比例
    max_correlation: float = 0.7  # 最大相关性
    var_limit: float = 0.02  # VaR限制


@dataclass
class RiskAlert:
    """风险告警"""
    alert_type: AlertType
    severity: RiskLevel
    message: str
    current_value: float
    threshold: float
    symbol: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class AILevelRiskController:
    """AI等级驱动风险控制器"""
    
    def __init__(self, evolution_system: AIEvolutionSystem, initial_capital: float = 100000.0):
        self.evolution_system = evolution_system
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 基础风险限制
        self.base_risk_limits = RiskLimits()
        self.current_risk_limits = RiskLimits()
        
        # 持仓和风险数据
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_metrics_history: deque = deque(maxlen=1000)
        self.alerts: List[RiskAlert] = []
        self.max_alerts = 100
        
        # 风险计算参数
        self.confidence_level = 0.95  # VaR置信度
        self.lookback_days = 252  # 历史数据回看天数
        self.rebalance_threshold = 0.05  # 再平衡阈值
        
        # 监控状态
        self.monitoring_active = False
        self.emergency_stop = False
        
        # 分配CPU核心
        assign_cpu_cores(CPUTaskType.RISK_MANAGEMENT, [17, 18, 19, 20])
        
        logger.info("AI等级驱动风险控制器初始化完成")
    
    def update_risk_limits_by_ai_level(self):
        """根据AI等级更新风险限制"""
        try:
            # 获取AI系统状态
            ai_status = self.evolution_system.get_system_status()
            avg_level = ai_status['evolution_stats']['avg_level']
            system_health = ai_status['system_health']
            
            # 根据AI等级调整风险参数
            level_multiplier = self._calculate_level_multiplier(avg_level)
            health_multiplier = self._calculate_health_multiplier(system_health)
            
            # 综合调整系数
            adjustment_factor = level_multiplier * health_multiplier
            
            # 更新风险限制
            self.current_risk_limits = RiskLimits(
                max_position_size=min(0.2, self.base_risk_limits.max_position_size * adjustment_factor),
                max_total_exposure=min(2.0, self.base_risk_limits.max_total_exposure * adjustment_factor),
                max_leverage=min(10.0, self.base_risk_limits.max_leverage * adjustment_factor),
                max_daily_loss=max(0.02, self.base_risk_limits.max_daily_loss / adjustment_factor),
                max_drawdown=max(0.08, self.base_risk_limits.max_drawdown / adjustment_factor),
                min_liquidity_ratio=max(0.1, self.base_risk_limits.min_liquidity_ratio / adjustment_factor),
                max_correlation=min(0.9, self.base_risk_limits.max_correlation * adjustment_factor),
                var_limit=max(0.01, self.base_risk_limits.var_limit / adjustment_factor)
            )
            
            logger.info(f"风险限制已更新 - AI等级: {avg_level:.1f}, "
                       f"调整系数: {adjustment_factor:.2f}, "
                       f"最大持仓: {self.current_risk_limits.max_position_size:.2%}")
            
        except Exception as e:
            logger.error(f"更新风险限制失败: {e}")
    
    def _calculate_level_multiplier(self, avg_level: float) -> float:
        """计算等级调整系数"""
        # AI等级越高，允许更高的风险
        if avg_level >= 80:  # 钻石/史诗级
            return 2.0
        elif avg_level >= 60:  # 铂金级
            return 1.5
        elif avg_level >= 40:  # 黄金级
            return 1.2
        elif avg_level >= 20:  # 白银级
            return 1.0
        else:  # 青铜级
            return 0.7
    
    def _calculate_health_multiplier(self, system_health: str) -> float:
        """计算健康状态调整系数"""
        health_multipliers = {
            'excellent': 1.2,
            'good': 1.0,
            'fair': 0.8,
            'poor': 0.5,
            'critical': 0.3
        }
        return health_multipliers.get(system_health, 0.8)
    
    def calculate_position_risk(self, symbol: str, position_size: float, 
                               current_price: float, entry_price: float) -> PositionRisk:
        """计算持仓风险"""
        try:
            market_value = abs(position_size * current_price)
            unrealized_pnl = position_size * (current_price - entry_price)
            position_ratio = market_value / self.current_capital
            
            # 计算止损价格
            if position_size > 0:  # 多头
                stop_loss_price = entry_price * (1 - self.current_risk_limits.max_daily_loss)
                take_profit_price = entry_price * (1 + self.current_risk_limits.max_daily_loss * 2)
            else:  # 空头
                stop_loss_price = entry_price * (1 + self.current_risk_limits.max_daily_loss)
                take_profit_price = entry_price * (1 - self.current_risk_limits.max_daily_loss * 2)
            
            # 计算最大损失
            max_loss_amount = market_value * self.current_risk_limits.max_daily_loss
            
            # 计算风险评分 (0-100)
            risk_score = self._calculate_risk_score(position_ratio, unrealized_pnl, market_value)
            
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_score=risk_score,
                max_loss_amount=max_loss_amount,
                position_ratio=position_ratio
            )
            
        except Exception as e:
            logger.error(f"计算持仓风险失败 {symbol}: {e}")
            return None
    
    def _calculate_risk_score(self, position_ratio: float, unrealized_pnl: float, 
                             market_value: float) -> float:
        """计算风险评分"""
        # 基础风险分数
        base_score = min(position_ratio / self.current_risk_limits.max_position_size * 50, 50)
        
        # 未实现盈亏风险
        pnl_ratio = unrealized_pnl / market_value if market_value > 0 else 0
        pnl_score = max(0, -pnl_ratio * 100) if pnl_ratio < 0 else 0
        
        # 综合风险评分
        total_score = min(100, base_score + pnl_score)
        
        return total_score
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """计算投资组合风险"""
        try:
            current_time = time.time()
            
            # 计算总敞口
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            leverage_ratio = total_exposure / self.current_capital if self.current_capital > 0 else 0
            
            # 计算当前投资组合价值
            portfolio_value = self.current_capital + sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # 计算VaR (简化版本)
            if len(self.risk_metrics_history) > 10:
                returns = [m.portfolio_value / self.initial_capital - 1 
                          for m in list(self.risk_metrics_history)[-252:]]
                returns_array = np.array(returns)
                
                var_1d = np.percentile(returns_array, (1 - self.confidence_level) * 100) * portfolio_value
                var_5d = var_1d * np.sqrt(5)
                
                volatility = np.std(returns_array) * np.sqrt(252)
                sharpe_ratio = np.mean(returns_array) / volatility if volatility > 0 else 0
            else:
                var_1d = var_5d = volatility = sharpe_ratio = 0.0
            
            # 计算最大回撤
            if len(self.risk_metrics_history) > 1:
                values = [m.portfolio_value for m in self.risk_metrics_history]
                peak = np.maximum.accumulate(values)
                drawdown = (np.array(values) - peak) / peak
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0.0
            
            # 计算相关性风险 (简化)
            correlation_risk = self._calculate_correlation_risk()
            
            # 计算流动性风险
            liquidity_risk = self._calculate_liquidity_risk()
            
            # 计算集中度风险
            concentration_risk = self._calculate_concentration_risk()
            
            risk_metrics = RiskMetrics(
                timestamp=current_time,
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                leverage_ratio=leverage_ratio,
                var_1d=var_1d,
                var_5d=var_5d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=1.0,  # 简化为1
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk
            )
            
            # 添加到历史记录
            self.risk_metrics_history.append(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"计算投资组合风险失败: {e}")
            return None
    
    def _calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        if len(self.positions) < 2:
            return 0.0
        
        # 简化的相关性风险计算
        # 实际应该基于历史价格数据计算相关系数
        symbols = list(self.positions.keys())
        
        # 模拟相关性 (实际应该从历史数据计算)
        correlations = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                # 简化假设：同类资产相关性较高
                if symbols[i][:3] == symbols[j][:3]:  # 如BTC/ETH
                    correlations.append(0.8)
                else:
                    correlations.append(0.3)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_liquidity_risk(self) -> float:
        """计算流动性风险"""
        if not self.positions:
            return 0.0
        
        # 简化的流动性风险评估
        # 实际应该基于交易量、买卖价差等数据
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        # 假设大额持仓流动性风险更高
        large_positions = sum(pos.market_value for pos in self.positions.values() 
                             if pos.position_ratio > 0.05)
        
        return large_positions / total_value if total_value > 0 else 0.0
    
    def _calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        if not self.positions:
            return 0.0
        
        # 计算赫芬达尔指数
        total_value = sum(pos.market_value for pos in self.positions.values())
        if total_value == 0:
            return 0.0
        
        weights = [pos.market_value / total_value for pos in self.positions.values()]
        hhi = sum(w ** 2 for w in weights)
        
        return hhi
    
    def check_risk_violations(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """检查风险违规"""
        alerts = []
        
        try:
            # 检查杠杆比率
            if risk_metrics.leverage_ratio > self.current_risk_limits.max_leverage:
                alerts.append(RiskAlert(
                    alert_type=AlertType.LEVERAGE,
                    severity=RiskLevel.HIGH,
                    message=f"杠杆比率过高: {risk_metrics.leverage_ratio:.2f}",
                    current_value=risk_metrics.leverage_ratio,
                    threshold=self.current_risk_limits.max_leverage
                ))
            
            # 检查最大回撤
            if abs(risk_metrics.max_drawdown) > self.current_risk_limits.max_drawdown:
                alerts.append(RiskAlert(
                    alert_type=AlertType.DRAWDOWN,
                    severity=RiskLevel.CRITICAL,
                    message=f"最大回撤超限: {risk_metrics.max_drawdown:.2%}",
                    current_value=abs(risk_metrics.max_drawdown),
                    threshold=self.current_risk_limits.max_drawdown
                ))
            
            # 检查VaR
            var_ratio = abs(risk_metrics.var_1d) / risk_metrics.portfolio_value if risk_metrics.portfolio_value > 0 else 0
            if var_ratio > self.current_risk_limits.var_limit:
                alerts.append(RiskAlert(
                    alert_type=AlertType.POSITION_SIZE,
                    severity=RiskLevel.HIGH,
                    message=f"VaR超限: {var_ratio:.2%}",
                    current_value=var_ratio,
                    threshold=self.current_risk_limits.var_limit
                ))
            
            # 检查波动率
            if risk_metrics.volatility > 0.5:  # 50%年化波动率
                alerts.append(RiskAlert(
                    alert_type=AlertType.VOLATILITY,
                    severity=RiskLevel.MEDIUM,
                    message=f"波动率过高: {risk_metrics.volatility:.2%}",
                    current_value=risk_metrics.volatility,
                    threshold=0.5
                ))
            
            # 检查相关性风险
            if risk_metrics.correlation_risk > self.current_risk_limits.max_correlation:
                alerts.append(RiskAlert(
                    alert_type=AlertType.CORRELATION,
                    severity=RiskLevel.MEDIUM,
                    message=f"相关性风险过高: {risk_metrics.correlation_risk:.2f}",
                    current_value=risk_metrics.correlation_risk,
                    threshold=self.current_risk_limits.max_correlation
                ))
            
            # 检查单个持仓风险
            for symbol, position in self.positions.items():
                if position.position_ratio > self.current_risk_limits.max_position_size:
                    alerts.append(RiskAlert(
                        alert_type=AlertType.POSITION_SIZE,
                        severity=RiskLevel.HIGH,
                        message=f"持仓比例过高: {symbol} {position.position_ratio:.2%}",
                        current_value=position.position_ratio,
                        threshold=self.current_risk_limits.max_position_size,
                        symbol=symbol
                    ))
            
            # 添加到告警列表
            for alert in alerts:
                self.alerts.append(alert)
                logger.warning(f"风险告警: {alert.message}")
            
            # 限制告警数量
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            return alerts
            
        except Exception as e:
            logger.error(f"检查风险违规失败: {e}")
            return []
    
    def should_stop_trading(self, risk_metrics: RiskMetrics) -> bool:
        """判断是否应该停止交易"""
        # 紧急停止条件
        emergency_conditions = [
            abs(risk_metrics.max_drawdown) > self.current_risk_limits.max_drawdown * 1.5,
            risk_metrics.leverage_ratio > self.current_risk_limits.max_leverage * 2,
            risk_metrics.portfolio_value < self.initial_capital * 0.5  # 损失50%
        ]
        
        if any(emergency_conditions):
            self.emergency_stop = True
            logger.critical("触发紧急停止条件，停止所有交易!")
            return True
        
        return False
    
    def get_position_size_recommendation(self, symbol: str, signal_strength: float, 
                                       current_price: float, ai_confidence: float) -> float:
        """获取建议持仓大小"""
        try:
            # 基础持仓大小 (基于AI置信度和信号强度)
            base_size = signal_strength * ai_confidence * self.current_risk_limits.max_position_size
            
            # 根据当前风险状况调整
            current_risk = self.calculate_portfolio_risk()
            if current_risk:
                # 如果当前风险较高，减少持仓
                risk_adjustment = 1.0 - (current_risk.leverage_ratio / self.current_risk_limits.max_leverage) * 0.5
                risk_adjustment = max(0.1, risk_adjustment)
                
                base_size *= risk_adjustment
            
            # 考虑资金管理
            available_capital = self.current_capital * (1 - sum(pos.position_ratio for pos in self.positions.values()))
            max_position_value = available_capital * self.current_risk_limits.max_position_size
            max_position_size = max_position_value / current_price
            
            # 返回较小值
            recommended_size = min(base_size, max_position_size)
            
            logger.info(f"持仓建议 {symbol}: 信号强度={signal_strength:.2f}, "
                       f"AI置信度={ai_confidence:.2f}, 建议大小={recommended_size:.4f}")
            
            return recommended_size
            
        except Exception as e:
            logger.error(f"计算建议持仓大小失败: {e}")
            return 0.0
    
    def update_position(self, symbol: str, position_size: float, 
                       current_price: float, entry_price: float):
        """更新持仓信息"""
        try:
            if position_size == 0:
                # 清空持仓
                if symbol in self.positions:
                    del self.positions[symbol]
                    logger.info(f"清空持仓: {symbol}")
            else:
                # 更新或创建持仓
                position_risk = self.calculate_position_risk(symbol, position_size, current_price, entry_price)
                if position_risk:
                    self.positions[symbol] = position_risk
                    logger.info(f"更新持仓: {symbol}, 大小={position_size:.4f}, "
                               f"风险评分={position_risk.risk_score:.1f}")
            
        except Exception as e:
            logger.error(f"更新持仓失败 {symbol}: {e}")
    
    async def start_risk_monitoring(self, interval: float = 10.0):
        """启动风险监控"""
        self.monitoring_active = True
        logger.info("启动风险监控系统")
        
        while self.monitoring_active:
            try:
                # 更新AI等级驱动的风险限制
                self.update_risk_limits_by_ai_level()
                
                # 计算投资组合风险
                risk_metrics = self.calculate_portfolio_risk()
                
                if risk_metrics:
                    # 检查风险违规
                    alerts = self.check_risk_violations(risk_metrics)
                    
                    # 检查是否需要停止交易
                    if self.should_stop_trading(risk_metrics):
                        logger.critical("风险过高，建议停止交易!")
                    
                    # 定期日志
                    if int(time.time()) % 60 == 0:  # 每分钟记录一次
                        logger.info(f"风险监控: 投资组合价值={risk_metrics.portfolio_value:.2f}, "
                                   f"杠杆={risk_metrics.leverage_ratio:.2f}, "
                                   f"最大回撤={risk_metrics.max_drawdown:.2%}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"风险监控出错: {e}")
                await asyncio.sleep(interval)
    
    def stop_risk_monitoring(self):
        """停止风险监控"""
        self.monitoring_active = False
        logger.info("风险监控已停止")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if not self.risk_metrics_history:
            return {}
        
        latest_metrics = self.risk_metrics_history[-1]
        
        return {
            'portfolio_value': latest_metrics.portfolio_value,
            'total_exposure': latest_metrics.total_exposure,
            'leverage_ratio': latest_metrics.leverage_ratio,
            'max_drawdown': latest_metrics.max_drawdown,
            'sharpe_ratio': latest_metrics.sharpe_ratio,
            'volatility': latest_metrics.volatility,
            'var_1d': latest_metrics.var_1d,
            'risk_limits': {
                'max_position_size': self.current_risk_limits.max_position_size,
                'max_leverage': self.current_risk_limits.max_leverage,
                'max_drawdown': self.current_risk_limits.max_drawdown,
                'var_limit': self.current_risk_limits.var_limit
            },
            'positions': {
                symbol: {
                    'size': pos.position_size,
                    'value': pos.market_value,
                    'pnl': pos.unrealized_pnl,
                    'risk_score': pos.risk_score,
                    'ratio': pos.position_ratio
                }
                for symbol, pos in self.positions.items()
            },
            'active_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'emergency_stop': self.emergency_stop
        }


# 全局风险控制器实例
risk_controller = None


def create_risk_controller(evolution_system: AIEvolutionSystem, 
                          initial_capital: float = 100000.0) -> AILevelRiskController:
    """创建风险控制器"""
    global risk_controller
    risk_controller = AILevelRiskController(evolution_system, initial_capital)
    return risk_controller


async def main():
    """测试主函数"""
    logger.info("启动风险控制系统测试...")
    
    # 创建AI进化系统 (模拟)
    from src.ai_models.ai_evolution_system import create_evolution_system, EvolutionConfig
    
    evolution_system = create_evolution_system(EvolutionConfig())
    
    # 创建风险控制器
    controller = create_risk_controller(evolution_system, 100000.0)
    
    try:
        # 启动风险监控
        monitor_task = asyncio.create_task(controller.start_risk_monitoring(5.0))
        
        # 模拟交易
        for i in range(20):
            # 模拟持仓更新
            symbol = f"BTC/USDT"
            position_size = np.random.uniform(-0.1, 0.1)
            current_price = 50000 + i * 100
            entry_price = 50000
            
            controller.update_position(symbol, position_size, current_price, entry_price)
            
            # 获取风险摘要
            risk_summary = controller.get_risk_summary()
            
            if risk_summary:
                logger.info(f"轮次 {i+1}: 投资组合价值={risk_summary.get('portfolio_value', 0):.2f}, "
                           f"杠杆={risk_summary.get('leverage_ratio', 0):.2f}")
            
            await asyncio.sleep(2)
        
        # 停止监控
        controller.stop_risk_monitoring()
        monitor_task.cancel()
        
        # 显示最终风险摘要
        final_summary = controller.get_risk_summary()
        logger.info(f"最终风险摘要: {final_summary}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
        controller.stop_risk_monitoring()
    except Exception as e:
        logger.error(f"测试出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
