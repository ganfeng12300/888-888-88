#!/usr/bin/env python3
"""
🛡️ 高级风险控制器 - 严格风控系统
Advanced Risk Controller - Strict Risk Management System

确保日回撤<3%的多层级风险保护：
- 实时风险监控
- 动态仓位管理
- 多层级止损机制
- 紧急熔断保护
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from loguru import logger


class RiskLevel(Enum):
    """风险等级"""
    LOW = 1      # 低风险
    MEDIUM = 2   # 中等风险
    HIGH = 3     # 高风险
    CRITICAL = 4 # 危险风险


class AlertType(Enum):
    """警报类型"""
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    SYSTEM = "system"


@dataclass
class RiskMetrics:
    """风险指标"""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    total_exposure: float
    position_count: int
    avg_correlation: float
    volatility: float
    sharpe_ratio: float
    var_95: float  # 95% VaR
    timestamp: datetime


@dataclass
class RiskAlert:
    """风险警报"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    current_value: float
    threshold: float
    symbol: Optional[str]
    timestamp: datetime


@dataclass
class PositionRisk:
    """持仓风险"""
    symbol: str
    size: float
    exposure: float
    unrealized_pnl: float
    risk_score: float
    stop_loss: float
    take_profit: float
    max_loss: float
    correlation_risk: float
    timestamp: datetime


class AdvancedRiskController:
    """高级风险控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化风险控制器"""
        self.config = config or {}
        self.is_running = False
        self.lock = threading.Lock()
        
        # 风控参数
        self.max_daily_drawdown = self.config.get('max_daily_drawdown', 0.03)  # 3%
        self.max_total_drawdown = self.config.get('max_total_drawdown', 0.15)  # 15%
        self.max_position_size = self.config.get('max_position_size', 0.25)    # 25%
        self.max_total_exposure = self.config.get('max_total_exposure', 0.80)  # 80%
        self.max_correlation = self.config.get('max_correlation', 0.70)        # 70%
        self.volatility_threshold = self.config.get('volatility_threshold', 0.05)  # 5%
        
        # 止损参数
        self.hard_stop_loss = self.config.get('hard_stop_loss', 0.03)  # 3%硬止损
        self.soft_stop_loss = self.config.get('soft_stop_loss', 0.02)  # 2%软止损
        self.trailing_stop = self.config.get('trailing_stop', 0.015)   # 1.5%跟踪止损
        
        # 状态数据
        self.risk_metrics_history = []
        self.position_risks = {}
        self.active_alerts = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.account_balance = 0.0
        self.emergency_stop = False
        
        # 监控配置
        self.monitoring_interval = self.config.get('monitoring_interval', 1)  # 1秒
        self.alert_cooldown = self.config.get('alert_cooldown', 60)  # 60秒冷却
        self.last_alert_time = {}
        
        # 启动监控
        self._start_monitoring()
        
        logger.info("🛡️ 高级风险控制器初始化完成")
    
    def _start_monitoring(self):
        """启动风险监控"""
        self.is_running = True
        
        # 风险监控线程
        threading.Thread(
            target=self._risk_monitor_loop,
            daemon=True,
            name="RiskMonitorThread"
        ).start()
        
        # 警报处理线程
        threading.Thread(
            target=self._alert_handler_loop,
            daemon=True,
            name="AlertHandlerThread"
        ).start()
    
    def update_account_info(self, balance: float, daily_pnl: float):
        """更新账户信息"""
        with self.lock:
            self.account_balance = balance
            self.daily_pnl = daily_pnl
            
            # 计算回撤
            if daily_pnl < 0:
                current_drawdown = abs(daily_pnl) / balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def update_position_risk(self, symbol: str, size: float, entry_price: float, 
                           current_price: float, unrealized_pnl: float):
        """更新持仓风险"""
        try:
            # 计算风险指标
            exposure = abs(size * current_price)
            exposure_pct = exposure / self.account_balance if self.account_balance > 0 else 0
            
            # 计算止损价格
            if size > 0:  # 多头
                stop_loss = entry_price * (1 - self.hard_stop_loss)
                take_profit = entry_price * (1 + self.hard_stop_loss * 2)
            else:  # 空头
                stop_loss = entry_price * (1 + self.hard_stop_loss)
                take_profit = entry_price * (1 - self.hard_stop_loss * 2)
            
            # 计算最大损失
            max_loss = abs(size) * abs(entry_price - stop_loss)
            
            # 计算风险评分
            risk_score = self._calculate_position_risk_score(
                exposure_pct, unrealized_pnl, max_loss
            )
            
            position_risk = PositionRisk(
                symbol=symbol,
                size=size,
                exposure=exposure,
                unrealized_pnl=unrealized_pnl,
                risk_score=risk_score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_loss=max_loss,
                correlation_risk=0.0,  # 需要额外计算
                timestamp=datetime.now()
            )
            
            with self.lock:
                self.position_risks[symbol] = position_risk
            
            logger.debug(f"更新持仓风险: {symbol} 风险评分={risk_score:.2f}")
            
        except Exception as e:
            logger.error(f"更新持仓风险失败: {e}")
    
    def _calculate_position_risk_score(self, exposure_pct: float, 
                                     unrealized_pnl: float, max_loss: float) -> float:
        """计算持仓风险评分"""
        try:
            # 基础风险评分
            base_score = exposure_pct * 100  # 仓位比例风险
            
            # 未实现盈亏风险
            if self.account_balance > 0:
                pnl_risk = abs(unrealized_pnl) / self.account_balance * 100
            else:
                pnl_risk = 0
            
            # 最大损失风险
            if self.account_balance > 0:
                loss_risk = max_loss / self.account_balance * 100
            else:
                loss_risk = 0
            
            # 综合风险评分
            total_score = base_score * 0.4 + pnl_risk * 0.3 + loss_risk * 0.3
            
            return min(total_score, 100.0)  # 限制在100以内
            
        except Exception as e:
            logger.error(f"计算风险评分失败: {e}")
            return 50.0  # 默认中等风险
    
    def check_position_risk(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """检查持仓风险"""
        try:
            # 计算新仓位的风险
            exposure = abs(size * price)
            exposure_pct = exposure / self.account_balance if self.account_balance > 0 else 0
            
            # 检查单个仓位大小
            if exposure_pct > self.max_position_size:
                return False, f"单个仓位过大: {exposure_pct:.2%} > {self.max_position_size:.2%}"
            
            # 检查总敞口
            total_exposure = sum(pos.exposure for pos in self.position_risks.values())
            new_total_exposure = (total_exposure + exposure) / self.account_balance
            
            if new_total_exposure > self.max_total_exposure:
                return False, f"总敞口过大: {new_total_exposure:.2%} > {self.max_total_exposure:.2%}"
            
            # 检查日回撤限制
            if self.daily_pnl < -self.max_daily_drawdown * self.account_balance:
                return False, f"达到日回撤限制: {abs(self.daily_pnl/self.account_balance):.2%}"
            
            # 检查紧急停止状态
            if self.emergency_stop:
                return False, "系统处于紧急停止状态"
            
            return True, "风险检查通过"
            
        except Exception as e:
            logger.error(f"检查持仓风险失败: {e}")
            return False, f"风险检查异常: {str(e)}"
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """检查平仓条件"""
        try:
            position_risk = self.position_risks.get(symbol)
            if not position_risk:
                return False, "无持仓信息"
            
            # 检查硬止损
            if position_risk.size > 0:  # 多头
                if current_price <= position_risk.stop_loss:
                    return True, f"触发硬止损: {current_price} <= {position_risk.stop_loss}"
            else:  # 空头
                if current_price >= position_risk.stop_loss:
                    return True, f"触发硬止损: {current_price} >= {position_risk.stop_loss}"
            
            # 检查止盈
            if position_risk.size > 0:  # 多头
                if current_price >= position_risk.take_profit:
                    return True, f"触发止盈: {current_price} >= {position_risk.take_profit}"
            else:  # 空头
                if current_price <= position_risk.take_profit:
                    return True, f"触发止盈: {current_price} <= {position_risk.take_profit}"
            
            # 检查风险评分
            if position_risk.risk_score > 80:
                return True, f"风险评分过高: {position_risk.risk_score:.1f}"
            
            return False, "无平仓条件"
            
        except Exception as e:
            logger.error(f"检查平仓条件失败: {e}")
            return False, f"检查异常: {str(e)}"
    
    def calculate_risk_metrics(self, positions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """计算风险指标"""
        try:
            with self.lock:
                # 如果传入了positions参数，使用传入的数据计算
                if positions:
                    total_pnl = sum(float(pos.get('pnl', 0)) for pos in positions)
                    total_exposure = sum(abs(float(pos.get('size', 0)) * float(pos.get('entry_price', 0))) for pos in positions)
                    position_count = len([pos for pos in positions if float(pos.get('size', 0)) != 0])
                else:
                    # 使用内部状态计算
                    total_pnl = self.daily_pnl
                    total_exposure = sum(pos.exposure for pos in self.position_risks.values())
                    position_count = len([pos for pos in self.position_risks.values() if pos.size != 0])
                
                # 当前回撤
                current_drawdown = 0.0
                if self.account_balance > 0 and total_pnl < 0:
                    current_drawdown = abs(total_pnl) / self.account_balance
                
                # 敞口比例
                exposure_pct = total_exposure / self.account_balance if self.account_balance > 0 else 0
                
                # 平均相关性（简化计算）
                avg_correlation = 0.5  # 需要实际计算
                
                # 波动率（基于历史数据）
                volatility = self._calculate_volatility()
                
                # 夏普比率（简化计算）
                sharpe_ratio = self._calculate_sharpe_ratio()
                
                # VaR 95%（简化计算）
                var_95 = self._calculate_var()
                
                # 风险评分
                risk_score = self._calculate_overall_risk_score(
                    current_drawdown, exposure_pct, position_count
                )
                
                return {
                    'total_exposure': exposure_pct,
                    'total_pnl': total_pnl,
                    'max_drawdown': max(self.max_drawdown, current_drawdown),
                    'current_drawdown': current_drawdown,
                    'daily_pnl': total_pnl,
                    'position_count': position_count,
                    'avg_correlation': avg_correlation,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'var_95': var_95,
                    'risk_score': risk_score,
                    'account_balance': self.account_balance,
                    'emergency_stop': self.emergency_stop,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return {
                'total_exposure': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'daily_pnl': 0.0,
                'position_count': 0,
                'avg_correlation': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'var_95': 0.0,
                'risk_score': 50.0,
                'account_balance': self.account_balance,
                'emergency_stop': self.emergency_stop,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_overall_risk_score(self, drawdown: float, exposure: float, position_count: int) -> float:
        """计算综合风险评分"""
        try:
            # 回撤风险 (0-40分)
            drawdown_score = min(drawdown * 100 * 2, 40)
            
            # 敞口风险 (0-30分)
            exposure_score = min(exposure * 100, 30)
            
            # 仓位数量风险 (0-20分)
            position_score = min(position_count * 2, 20)
            
            # 紧急状态风险 (0-10分)
            emergency_score = 10 if self.emergency_stop else 0
            
            total_score = drawdown_score + exposure_score + position_score + emergency_score
            
            return min(total_score, 100.0)
            
        except Exception as e:
            logger.error(f"计算综合风险评分失败: {e}")
            return 50.0
    
    def check_risk_limits(self, position_or_order: Dict[str, Any]) -> Dict[str, Any]:
        """检查风险限制"""
        try:
            symbol = position_or_order.get('symbol', 'unknown')
            size = float(position_or_order.get('size', 0))
            price = float(position_or_order.get('entry_price', position_or_order.get('price', 0)))
            
            # 计算仓位价值
            position_value = abs(size) * price
            
            # 检查单个仓位限制
            if self.account_balance > 0:
                position_ratio = position_value / self.account_balance
                if position_ratio > self.max_position_size:
                    return {
                        'allowed': False,
                        'reason': f'单个仓位过大: {position_ratio:.2%} > {self.max_position_size:.2%}',
                        'risk_level': 'high',
                        'position_ratio': position_ratio
                    }
            
            # 检查日回撤限制
            if self.daily_pnl < -self.max_daily_drawdown * self.account_balance:
                return {
                    'allowed': False,
                    'reason': f'达到日回撤限制: {abs(self.daily_pnl/self.account_balance):.2%}',
                    'risk_level': 'high',
                    'daily_pnl': self.daily_pnl
                }
            
            # 检查紧急停止状态
            if self.emergency_stop:
                return {
                    'allowed': False,
                    'reason': '系统处于紧急停止状态',
                    'risk_level': 'high',
                    'emergency_stop': True
                }
            
            # 检查总敞口
            current_total_exposure = sum(pos.exposure for pos in self.position_risks.values())
            new_total_exposure = (current_total_exposure + position_value) / self.account_balance
            
            if new_total_exposure > self.max_total_exposure:
                return {
                    'allowed': False,
                    'reason': f'总敞口过大: {new_total_exposure:.2%} > {self.max_total_exposure:.2%}',
                    'risk_level': 'high',
                    'total_exposure': new_total_exposure
                }
            
            # 计算风险评分
            risk_score = self._calculate_position_risk_score(
                position_ratio if self.account_balance > 0 else 0,
                position_or_order.get('pnl', 0),
                position_value * 0.02  # 假设2%的最大损失
            )
            
            # 风险等级评估
            if risk_score < 30:
                risk_level = 'low'
            elif risk_score < 70:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'allowed': True,
                'reason': '风险检查通过',
                'risk_level': risk_level,
                'risk_score': risk_score,
                'position_ratio': position_ratio if self.account_balance > 0 else 0,
                'total_exposure': new_total_exposure,
                'daily_pnl': self.daily_pnl,
                'account_balance': self.account_balance
            }
            
        except Exception as e:
            logger.error(f"风险限制检查失败: {e}")
            return {
                'allowed': False,
                'reason': f'风险检查异常: {str(e)}',
                'risk_level': 'high',
                'error': str(e)
            }
    
    def suggest_stop_loss(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """建议止损价格"""
        try:
            symbol = position.get('symbol', 'unknown')
            size = float(position.get('size', 0))
            entry_price = float(position.get('entry_price', 0))
            current_price = float(position.get('current_price', entry_price))
            
            if size == 0 or entry_price == 0:
                return {
                    'status': 'failed',
                    'reason': '无效的仓位信息',
                    'symbol': symbol
                }
            
            # 基于配置的止损比例
            if size > 0:  # 多头仓位
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                trailing_stop_price = current_price * (1 - self.trailing_stop)
                recommended_stop = max(stop_loss_price, trailing_stop_price)
            else:  # 空头仓位
                stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                trailing_stop_price = current_price * (1 + self.trailing_stop)
                recommended_stop = min(stop_loss_price, trailing_stop_price)
            
            # 计算潜在损失
            if size > 0:
                potential_loss = (entry_price - recommended_stop) * abs(size)
            else:
                potential_loss = (recommended_stop - entry_price) * abs(size)
            
            # 损失比例
            loss_percentage = potential_loss / (self.account_balance if self.account_balance > 0 else 1000)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'recommended_stop_loss': recommended_stop,
                'hard_stop_loss': stop_loss_price,
                'trailing_stop_loss': trailing_stop_price,
                'potential_loss': potential_loss,
                'loss_percentage': loss_percentage,
                'position_side': 'long' if size > 0 else 'short',
                'urgency': 'high' if loss_percentage > 0.01 else 'medium' if loss_percentage > 0.005 else 'low'
            }
            
        except Exception as e:
            logger.error(f"生成止损建议失败: {e}")
            return {
                'status': 'failed',
                'reason': f'止损建议生成异常: {str(e)}',
                'symbol': position.get('symbol', 'unknown'),
                'error': str(e)
            }
    
    def generate_risk_report(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成风险报告"""
        try:
            # 计算风险指标
            risk_metrics = self.calculate_risk_metrics(positions)
            
            # 分析每个仓位
            position_analyses = []
            total_risk_score = 0
            high_risk_positions = 0
            
            for position in positions:
                risk_check = self.check_risk_limits(position)
                stop_loss_suggestion = self.suggest_stop_loss(position)
                
                analysis = {
                    'symbol': position.get('symbol'),
                    'size': position.get('size'),
                    'pnl': position.get('pnl'),
                    'risk_check': risk_check,
                    'stop_loss_suggestion': stop_loss_suggestion
                }
                
                position_analyses.append(analysis)
                
                # 累计风险评分
                if 'risk_score' in risk_check:
                    total_risk_score += risk_check['risk_score']
                    if risk_check['risk_level'] == 'high':
                        high_risk_positions += 1
            
            # 综合风险评估
            avg_risk_score = total_risk_score / len(positions) if positions else 0
            
            # 生成建议
            recommendations = []
            if high_risk_positions > 0:
                recommendations.append(f"有{high_risk_positions}个高风险仓位，建议立即关注")
            
            if risk_metrics['current_drawdown'] > 0.02:
                recommendations.append("当前回撤较大，建议减少仓位")
            
            if risk_metrics['total_exposure'] > 0.8:
                recommendations.append("总敞口过高，建议降低杠杆")
            
            if not recommendations:
                recommendations.append("风险状况良好，继续监控")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_positions': len(positions),
                    'high_risk_positions': high_risk_positions,
                    'average_risk_score': avg_risk_score,
                    'overall_risk_level': 'high' if avg_risk_score > 70 else 'medium' if avg_risk_score > 30 else 'low'
                },
                'risk_metrics': risk_metrics,
                'position_analyses': position_analyses,
                'recommendations': recommendations,
                'emergency_actions': self._generate_emergency_actions(risk_metrics, high_risk_positions)
            }
            
        except Exception as e:
            logger.error(f"生成风险报告失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'summary': {'total_positions': len(positions) if positions else 0},
                'recommendations': ['风险报告生成失败，建议手动检查系统状态']
            }
    
    def _generate_emergency_actions(self, risk_metrics: Dict[str, Any], high_risk_count: int) -> List[str]:
        """生成紧急行动建议"""
        actions = []
        
        if risk_metrics.get('current_drawdown', 0) > 0.05:
            actions.append("立即停止新开仓")
            actions.append("考虑平仓部分高风险仓位")
        
        if high_risk_count > 3:
            actions.append("启动紧急风控模式")
            actions.append("逐步平仓高风险仓位")
        
        if risk_metrics.get('total_exposure', 0) > 0.9:
            actions.append("立即降低总敞口至安全水平")
        
        if risk_metrics.get('emergency_stop', False):
            actions.append("系统已启动紧急停止，等待手动干预")
        
        return actions if actions else ["继续正常监控"]
    
    def adjust_risk_limits(self, market_volatility: float) -> Dict[str, Any]:
        """动态调整风险限制"""
        try:
            # 基于市场波动率调整风险参数
            volatility_multiplier = 1.0
            
            if market_volatility > 0.05:  # 高波动
                volatility_multiplier = 0.7  # 降低风险限制
            elif market_volatility > 0.03:  # 中等波动
                volatility_multiplier = 0.85
            elif market_volatility < 0.01:  # 低波动
                volatility_multiplier = 1.2  # 可以适当放宽
            
            # 调整后的限制
            adjusted_limits = {
                'max_position_size': self.max_position_size * volatility_multiplier,
                'max_total_exposure': self.max_total_exposure * volatility_multiplier,
                'stop_loss_pct': self.stop_loss_pct / volatility_multiplier,  # 波动大时止损更严格
                'max_daily_drawdown': self.max_daily_drawdown * volatility_multiplier
            }
            
            # 记录调整
            logger.info(f"基于市场波动率{market_volatility:.3f}调整风险限制，乘数: {volatility_multiplier:.2f}")
            
            return {
                'status': 'success',
                'market_volatility': market_volatility,
                'volatility_multiplier': volatility_multiplier,
                'original_limits': {
                    'max_position_size': self.max_position_size,
                    'max_total_exposure': self.max_total_exposure,
                    'stop_loss_pct': self.stop_loss_pct,
                    'max_daily_drawdown': self.max_daily_drawdown
                },
                'adjusted_limits': adjusted_limits,
                'adjustment_reason': self._get_volatility_reason(market_volatility)
            }
            
        except Exception as e:
            logger.error(f"调整风险限制失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'market_volatility': market_volatility
            }
    
    def _get_volatility_reason(self, volatility: float) -> str:
        """获取波动率调整原因"""
        if volatility > 0.05:
            return "市场高波动，降低风险敞口"
        elif volatility > 0.03:
            return "市场中等波动，适度调整风险参数"
        elif volatility < 0.01:
            return "市场低波动，可适当放宽风险限制"
        else:
            return "市场波动正常，维持标准风险参数"
    
    def _calculate_volatility(self) -> float:
        """计算波动率"""
        try:
            if len(self.risk_metrics_history) < 10:
                return 0.0
            
            # 使用最近的PnL数据计算波动率
            recent_pnls = [m.daily_pnl for m in self.risk_metrics_history[-20:]]
            if len(recent_pnls) > 1:
                return float(np.std(recent_pnls))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.risk_metrics_history) < 10:
                return 0.0
            
            # 简化的夏普比率计算
            recent_pnls = [m.daily_pnl for m in self.risk_metrics_history[-30:]]
            if len(recent_pnls) > 1:
                mean_return = np.mean(recent_pnls)
                std_return = np.std(recent_pnls)
                
                if std_return > 0:
                    return float(mean_return / std_return)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    def _calculate_var(self) -> float:
        """计算VaR"""
        try:
            if len(self.risk_metrics_history) < 20:
                return 0.0
            
            # 使用历史模拟法计算VaR
            recent_pnls = [m.daily_pnl for m in self.risk_metrics_history[-100:]]
            if len(recent_pnls) > 20:
                return float(np.percentile(recent_pnls, 5))  # 95% VaR
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0
    
    def _check_risk_alerts(self, metrics: RiskMetrics):
        """检查风险警报"""
        alerts = []
        
        # 检查回撤警报
        if metrics.current_drawdown > self.max_daily_drawdown:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                level=RiskLevel.CRITICAL,
                message=f"日回撤超限: {metrics.current_drawdown:.2%}",
                current_value=metrics.current_drawdown,
                threshold=self.max_daily_drawdown,
                symbol=None,
                timestamp=datetime.now()
            ))
        
        # 检查总敞口警报
        if metrics.total_exposure > self.max_total_exposure:
            alerts.append(RiskAlert(
                alert_type=AlertType.POSITION_SIZE,
                level=RiskLevel.HIGH,
                message=f"总敞口过大: {metrics.total_exposure:.2%}",
                current_value=metrics.total_exposure,
                threshold=self.max_total_exposure,
                symbol=None,
                timestamp=datetime.now()
            ))
        
        # 检查波动率警报
        if metrics.volatility > self.volatility_threshold:
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY,
                level=RiskLevel.MEDIUM,
                message=f"波动率过高: {metrics.volatility:.2%}",
                current_value=metrics.volatility,
                threshold=self.volatility_threshold,
                symbol=None,
                timestamp=datetime.now()
            ))
        
        # 处理警报
        for alert in alerts:
            self._handle_risk_alert(alert)
    
    def _handle_risk_alert(self, alert: RiskAlert):
        """处理风险警报"""
        try:
            # 检查冷却时间
            alert_key = f"{alert.alert_type.value}_{alert.symbol or 'global'}"
            last_time = self.last_alert_time.get(alert_key, 0)
            
            if time.time() - last_time < self.alert_cooldown:
                return  # 在冷却期内，跳过
            
            # 记录警报
            with self.lock:
                self.active_alerts.append(alert)
                
                # 限制警报数量
                if len(self.active_alerts) > 100:
                    self.active_alerts = self.active_alerts[-100:]
            
            # 更新冷却时间
            self.last_alert_time[alert_key] = time.time()
            
            # 根据警报级别采取行动
            if alert.level == RiskLevel.CRITICAL:
                self._handle_critical_alert(alert)
            elif alert.level == RiskLevel.HIGH:
                self._handle_high_alert(alert)
            
            logger.warning(f"🚨 风险警报: {alert.message}")
            
        except Exception as e:
            logger.error(f"处理风险警报失败: {e}")
    
    def _handle_critical_alert(self, alert: RiskAlert):
        """处理危险级警报"""
        if alert.alert_type == AlertType.DRAWDOWN:
            # 触发紧急停止
            self.emergency_stop = True
            logger.critical("🚨 触发紧急停止：日回撤超限")
        
        elif alert.alert_type == AlertType.SYSTEM:
            # 系统级警报
            self.emergency_stop = True
            logger.critical(f"🚨 系统级警报：{alert.message}")
    
    def _handle_high_alert(self, alert: RiskAlert):
        """处理高级警报"""
        if alert.alert_type == AlertType.POSITION_SIZE:
            logger.warning("⚠️ 建议减少仓位大小")
        
        elif alert.alert_type == AlertType.VOLATILITY:
            logger.warning("⚠️ 市场波动率过高，建议谨慎交易")
    
    def _risk_monitor_loop(self):
        """风险监控循环"""
        while self.is_running:
            try:
                # 计算风险指标
                metrics = self.calculate_risk_metrics()
                
                # 记录历史
                with self.lock:
                    self.risk_metrics_history.append(metrics)
                    
                    # 限制历史记录数量
                    if len(self.risk_metrics_history) > 1000:
                        self.risk_metrics_history = self.risk_metrics_history[-1000:]
                
                # 检查风险警报
                self._check_risk_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"风险监控失败: {e}")
                time.sleep(self.monitoring_interval)
    
    def _alert_handler_loop(self):
        """警报处理循环"""
        while self.is_running:
            try:
                time.sleep(10)  # 每10秒检查一次
                
                # 清理过期警报
                current_time = datetime.now()
                with self.lock:
                    self.active_alerts = [
                        alert for alert in self.active_alerts
                        if (current_time - alert.timestamp).seconds < 3600  # 保留1小时内的警报
                    ]
                
            except Exception as e:
                logger.error(f"警报处理失败: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        try:
            with self.lock:
                latest_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "emergency_stop": self.emergency_stop,
                    "current_metrics": {
                        "daily_drawdown": latest_metrics.current_drawdown if latest_metrics else 0,
                        "max_drawdown": latest_metrics.max_drawdown if latest_metrics else 0,
                        "daily_pnl": latest_metrics.daily_pnl if latest_metrics else 0,
                        "total_exposure": latest_metrics.total_exposure if latest_metrics else 0,
                        "position_count": latest_metrics.position_count if latest_metrics else 0,
                        "volatility": latest_metrics.volatility if latest_metrics else 0,
                        "sharpe_ratio": latest_metrics.sharpe_ratio if latest_metrics else 0,
                        "var_95": latest_metrics.var_95 if latest_metrics else 0
                    },
                    "risk_limits": {
                        "max_daily_drawdown": self.max_daily_drawdown,
                        "max_total_drawdown": self.max_total_drawdown,
                        "max_position_size": self.max_position_size,
                        "max_total_exposure": self.max_total_exposure,
                        "volatility_threshold": self.volatility_threshold
                    },
                    "position_risks": {
                        symbol: {
                            "size": pos.size,
                            "exposure": pos.exposure,
                            "risk_score": pos.risk_score,
                            "stop_loss": pos.stop_loss,
                            "take_profit": pos.take_profit,
                            "max_loss": pos.max_loss
                        }
                        for symbol, pos in self.position_risks.items()
                    },
                    "active_alerts": [
                        {
                            "type": alert.alert_type.value,
                            "level": alert.level.name,
                            "message": alert.message,
                            "current_value": alert.current_value,
                            "threshold": alert.threshold,
                            "symbol": alert.symbol,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in self.active_alerts[-10:]  # 最近10个警报
                    ]
                }
                
        except Exception as e:
            logger.error(f"获取风险报告失败: {e}")
            return {"error": str(e)}
    
    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        with self.lock:
            self.emergency_stop = False
            logger.info("紧急停止状态已重置")
    
    def shutdown(self):
        """关闭风险控制器"""
        logger.info("正在关闭高级风险控制器...")
        self.is_running = False
        
        # 等待线程结束
        time.sleep(2)
        
        logger.info("高级风险控制器已关闭")


# 全局实例
_risk_controller = None

def get_risk_controller(config: Dict[str, Any] = None) -> AdvancedRiskController:
    """获取风险控制器实例"""
    global _risk_controller
    if _risk_controller is None:
        _risk_controller = AdvancedRiskController(config)
    return _risk_controller


if __name__ == "__main__":
    # 测试代码
    def test_risk_controller():
        """测试风险控制器"""
        config = {
            "max_daily_drawdown": 0.03,
            "max_position_size": 0.25,
            "max_total_exposure": 0.80
        }
        
        controller = get_risk_controller(config)
        
        # 模拟账户更新
        controller.update_account_info(10000.0, -200.0)  # 2%亏损
        
        # 模拟持仓更新
        controller.update_position_risk("BTCUSDT", 0.1, 50000, 49000, -100)
        
        # 检查持仓风险
        can_trade, reason = controller.check_position_risk("ETHUSDT", 1.0, 3000)
        print(f"可以交易: {can_trade}, 原因: {reason}")
        
        # 获取风险报告
        report = controller.get_risk_report()
        print("风险报告:")
        import json
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 运行一段时间
        time.sleep(5)
        
        controller.shutdown()
    
    test_risk_controller()
