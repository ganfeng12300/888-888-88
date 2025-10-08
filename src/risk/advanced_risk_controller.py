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
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标"""
        try:
            with self.lock:
                # 当前回撤
                current_drawdown = 0.0
                if self.account_balance > 0 and self.daily_pnl < 0:
                    current_drawdown = abs(self.daily_pnl) / self.account_balance
                
                # 总敞口
                total_exposure = sum(pos.exposure for pos in self.position_risks.values())
                exposure_pct = total_exposure / self.account_balance if self.account_balance > 0 else 0
                
                # 持仓数量
                position_count = len([pos for pos in self.position_risks.values() if pos.size != 0])
                
                # 平均相关性（简化计算）
                avg_correlation = 0.5  # 需要实际计算
                
                # 波动率（基于历史数据）
                volatility = self._calculate_volatility()
                
                # 夏普比率（简化计算）
                sharpe_ratio = self._calculate_sharpe_ratio()
                
                # VaR 95%（简化计算）
                var_95 = self._calculate_var()
                
                return RiskMetrics(
                    current_drawdown=current_drawdown,
                    max_drawdown=self.max_drawdown,
                    daily_pnl=self.daily_pnl,
                    total_exposure=exposure_pct,
                    position_count=position_count,
                    avg_correlation=avg_correlation,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return RiskMetrics(
                current_drawdown=0.0,
                max_drawdown=0.0,
                daily_pnl=0.0,
                total_exposure=0.0,
                position_count=0,
                avg_correlation=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                var_95=0.0,
                timestamp=datetime.now()
            )
    
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
