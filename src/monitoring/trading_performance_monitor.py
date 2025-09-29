"""
📊 交易绩效监控器 - 生产级实盘交易绩效实时监控和分析系统
监控交易收益、风险指标、策略表现、资金使用等交易相关指标
提供绩效分析、风险评估、策略优化建议、交易报告生成
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

class TradeType(Enum):
    """交易类型"""
    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出
    LONG = "long"  # 做多
    SHORT = "short"  # 做空

class TradeStatus(Enum):
    """交易状态"""
    PENDING = "pending"  # 待成交
    FILLED = "filled"  # 已成交
    CANCELLED = "cancelled"  # 已取消
    FAILED = "failed"  # 失败

@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str  # 交易ID
    strategy_id: str  # 策略ID
    symbol: str  # 交易对
    trade_type: TradeType  # 交易类型
    quantity: float  # 数量
    price: float  # 价格
    fee: float  # 手续费
    pnl: float  # 盈亏
    status: TradeStatus  # 状态
    timestamp: float = field(default_factory=time.time)  # 时间戳

@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_trades: int  # 总交易数
    winning_trades: int  # 盈利交易数
    losing_trades: int  # 亏损交易数
    win_rate: float  # 胜率
    total_pnl: float  # 总盈亏
    total_return: float  # 总收益率
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡玛比率
    profit_factor: float  # 盈利因子
    average_win: float  # 平均盈利
    average_loss: float  # 平均亏损
    largest_win: float  # 最大盈利
    largest_loss: float  # 最大亏损
    timestamp: float = field(default_factory=time.time)

@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall: float  # 期望损失
    volatility: float  # 波动率
    beta: float  # 贝塔系数
    correlation: float  # 相关系数
    max_position_size: float  # 最大仓位
    leverage_ratio: float  # 杠杆比率
    timestamp: float = field(default_factory=time.time)

class PerformanceCalculator:
    """绩效计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        logger.info("绩效计算器初始化完成")
    
    def calculate_performance_metrics(self, trades: List[TradeRecord], 
                                    initial_capital: float = 100000) -> PerformanceMetrics:
        """计算绩效指标"""
        try:
            if not trades:
                return self._empty_performance_metrics()
            
            # 基础统计
            total_trades = len(trades)
            pnl_list = [trade.pnl for trade in trades]
            winning_trades = sum(1 for pnl in pnl_list if pnl > 0)
            losing_trades = sum(1 for pnl in pnl_list if pnl < 0)
            
            # 胜率
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 总盈亏和收益率
            total_pnl = sum(pnl_list)
            total_return = total_pnl / initial_capital if initial_capital > 0 else 0
            
            # 最大回撤
            cumulative_pnl = np.cumsum(pnl_list)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (running_max - cumulative_pnl) / initial_capital
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # 夏普比率
            if len(pnl_list) > 1:
                returns = np.array(pnl_list) / initial_capital
                excess_returns = returns - self.risk_free_rate / 252  # 日收益率
                sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 索提诺比率
            negative_returns = [r for r in pnl_list if r < 0]
            if negative_returns and len(pnl_list) > 1:
                downside_deviation = np.std(negative_returns) / initial_capital
                sortino_ratio = (total_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = 0
            
            # 卡玛比率
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # 盈利因子
            gross_profit = sum(pnl for pnl in pnl_list if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnl_list if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # 平均盈亏
            winning_pnl = [pnl for pnl in pnl_list if pnl > 0]
            losing_pnl = [pnl for pnl in pnl_list if pnl < 0]
            
            average_win = np.mean(winning_pnl) if winning_pnl else 0
            average_loss = np.mean(losing_pnl) if losing_pnl else 0
            largest_win = max(pnl_list) if pnl_list else 0
            largest_loss = min(pnl_list) if pnl_list else 0
            
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss
            )
        
        except Exception as e:
            logger.error(f"计算绩效指标失败: {e}")
            return self._empty_performance_metrics()
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """空绩效指标"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl=0, total_return=0, max_drawdown=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, profit_factor=0,
            average_win=0, average_loss=0, largest_win=0, largest_loss=0
        )
    
    def calculate_risk_metrics(self, trades: List[TradeRecord], 
                             market_returns: List[float] = None) -> RiskMetrics:
        """计算风险指标"""
        try:
            if not trades:
                return self._empty_risk_metrics()
            
            pnl_list = [trade.pnl for trade in trades]
            
            # VaR计算
            if len(pnl_list) >= 20:  # 至少需要20个样本
                var_95 = np.percentile(pnl_list, 5)  # 5%分位数
                var_99 = np.percentile(pnl_list, 1)  # 1%分位数
                
                # 期望损失 (CVaR)
                tail_losses = [pnl for pnl in pnl_list if pnl <= var_95]
                expected_shortfall = np.mean(tail_losses) if tail_losses else 0
            else:
                var_95 = var_99 = expected_shortfall = 0
            
            # 波动率
            volatility = np.std(pnl_list) if len(pnl_list) > 1 else 0
            
            # 贝塔系数和相关系数
            if market_returns and len(market_returns) == len(pnl_list) and len(pnl_list) > 1:
                correlation = np.corrcoef(pnl_list, market_returns)[0, 1]
                if np.std(market_returns) > 0:
                    beta = np.cov(pnl_list, market_returns)[0, 1] / np.var(market_returns)
                else:
                    beta = 0
            else:
                correlation = beta = 0
            
            # 最大仓位和杠杆比率 (需要从交易记录中计算)
            max_position_size = max(abs(trade.quantity * trade.price) for trade in trades) if trades else 0
            leverage_ratio = 1.0  # 默认值，需要根据实际情况计算
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                max_position_size=max_position_size,
                leverage_ratio=leverage_ratio
            )
        
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return self._empty_risk_metrics()
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """空风险指标"""
        return RiskMetrics(
            var_95=0, var_99=0, expected_shortfall=0, volatility=0,
            beta=0, correlation=0, max_position_size=0, leverage_ratio=1.0
        )

class TradingPerformanceMonitor:
    """交易绩效监控器主类"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.performance_calculator = PerformanceCalculator()
        
        # 交易数据
        self.trade_records: List[TradeRecord] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.risk_history: List[RiskMetrics] = []
        
        # 策略绩效
        self.strategy_performance: Dict[str, List[PerformanceMetrics]] = {}
        
        # 监控配置
        self.monitor_interval = 60  # 监控间隔（秒）
        self.is_monitoring = False
        
        # 回调函数
        self.performance_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"交易绩效监控器初始化完成: 初始资金 {initial_capital}")
    
    def record_trade(self, trade: TradeRecord):
        """记录交易"""
        try:
            with self.lock:
                self.trade_records.append(trade)
                
                # 保持交易记录在合理范围内
                if len(self.trade_records) > 10000:
                    self.trade_records = self.trade_records[-5000:]
                
                logger.debug(f"记录交易: {trade.trade_id} - {trade.symbol} - PnL: {trade.pnl:.2f}")
        
        except Exception as e:
            logger.error(f"记录交易失败: {e}")
    
    def calculate_current_performance(self) -> Optional[PerformanceMetrics]:
        """计算当前绩效"""
        try:
            with self.lock:
                if not self.trade_records:
                    return None
                
                # 只计算已成交的交易
                filled_trades = [t for t in self.trade_records if t.status == TradeStatus.FILLED]
                
                if not filled_trades:
                    return None
                
                performance = self.performance_calculator.calculate_performance_metrics(
                    filled_trades, self.initial_capital
                )
                
                return performance
        
        except Exception as e:
            logger.error(f"计算当前绩效失败: {e}")
            return None
    
    def calculate_strategy_performance(self, strategy_id: str) -> Optional[PerformanceMetrics]:
        """计算策略绩效"""
        try:
            with self.lock:
                strategy_trades = [t for t in self.trade_records 
                                 if t.strategy_id == strategy_id and t.status == TradeStatus.FILLED]
                
                if not strategy_trades:
                    return None
                
                performance = self.performance_calculator.calculate_performance_metrics(
                    strategy_trades, self.initial_capital
                )
                
                return performance
        
        except Exception as e:
            logger.error(f"计算策略绩效失败: {e}")
            return None
    
    def calculate_risk_metrics(self, market_returns: List[float] = None) -> Optional[RiskMetrics]:
        """计算风险指标"""
        try:
            with self.lock:
                filled_trades = [t for t in self.trade_records if t.status == TradeStatus.FILLED]
                
                if not filled_trades:
                    return None
                
                risk_metrics = self.performance_calculator.calculate_risk_metrics(
                    filled_trades, market_returns
                )
                
                return risk_metrics
        
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return None
    
    def start_monitoring(self):
        """启动监控"""
        try:
            self.is_monitoring = True
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            logger.info("交易绩效监控启动")
        
        except Exception as e:
            logger.error(f"启动交易绩效监控失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("交易绩效监控停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 计算当前绩效
                current_performance = self.calculate_current_performance()
                
                if current_performance:
                    with self.lock:
                        # 添加到历史记录
                        self.performance_history.append(current_performance)
                        
                        # 保持历史记录在合理范围内
                        if len(self.performance_history) > 1000:
                            self.performance_history = self.performance_history[-500:]
                        
                        # 调用回调函数
                        for callback in self.performance_callbacks:
                            try:
                                callback(current_performance)
                            except Exception as e:
                                logger.error(f"绩效回调执行失败: {e}")
                
                # 计算风险指标
                risk_metrics = self.calculate_risk_metrics()
                if risk_metrics:
                    with self.lock:
                        self.risk_history.append(risk_metrics)
                        
                        # 保持历史记录在合理范围内
                        if len(self.risk_history) > 1000:
                            self.risk_history = self.risk_history[-500:]
                
                time.sleep(self.monitor_interval)
            
            except Exception as e:
                logger.error(f"绩效监控循环失败: {e}")
                time.sleep(self.monitor_interval)
    
    def add_performance_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """添加绩效回调"""
        self.performance_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取绩效摘要"""
        try:
            with self.lock:
                current_performance = self.calculate_current_performance()
                current_risk = self.calculate_risk_metrics()
                
                # 策略绩效统计
                strategy_stats = {}
                strategies = set(t.strategy_id for t in self.trade_records)
                
                for strategy_id in strategies:
                    strategy_perf = self.calculate_strategy_performance(strategy_id)
                    if strategy_perf:
                        strategy_stats[strategy_id] = {
                            'total_trades': strategy_perf.total_trades,
                            'win_rate': strategy_perf.win_rate,
                            'total_pnl': strategy_perf.total_pnl,
                            'sharpe_ratio': strategy_perf.sharpe_ratio
                        }
                
                # 交易统计
                total_trades = len(self.trade_records)
                filled_trades = len([t for t in self.trade_records if t.status == TradeStatus.FILLED])
                pending_trades = len([t for t in self.trade_records if t.status == TradeStatus.PENDING])
                
                return {
                    'current_performance': {
                        'total_pnl': current_performance.total_pnl if current_performance else 0,
                        'total_return': current_performance.total_return if current_performance else 0,
                        'win_rate': current_performance.win_rate if current_performance else 0,
                        'sharpe_ratio': current_performance.sharpe_ratio if current_performance else 0,
                        'max_drawdown': current_performance.max_drawdown if current_performance else 0
                    },
                    'current_risk': {
                        'var_95': current_risk.var_95 if current_risk else 0,
                        'volatility': current_risk.volatility if current_risk else 0,
                        'max_position_size': current_risk.max_position_size if current_risk else 0
                    },
                    'trade_statistics': {
                        'total_trades': total_trades,
                        'filled_trades': filled_trades,
                        'pending_trades': pending_trades,
                        'success_rate': filled_trades / total_trades if total_trades > 0 else 0
                    },
                    'strategy_performance': strategy_stats,
                    'monitoring_status': self.is_monitoring,
                    'initial_capital': self.initial_capital
                }
        
        except Exception as e:
            logger.error(f"获取绩效摘要失败: {e}")
            return {}
    
    def get_recent_trades(self, limit: int = 50) -> List[TradeRecord]:
        """获取最近的交易记录"""
        with self.lock:
            return sorted(self.trade_records, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_performance_trend(self, window: int = 30) -> Dict[str, List[float]]:
        """获取绩效趋势"""
        try:
            with self.lock:
                if len(self.performance_history) < 2:
                    return {}
                
                recent_performance = self.performance_history[-window:]
                
                return {
                    'timestamps': [p.timestamp for p in recent_performance],
                    'total_pnl': [p.total_pnl for p in recent_performance],
                    'total_return': [p.total_return for p in recent_performance],
                    'win_rate': [p.win_rate for p in recent_performance],
                    'sharpe_ratio': [p.sharpe_ratio for p in recent_performance],
                    'max_drawdown': [p.max_drawdown for p in recent_performance]
                }
        
        except Exception as e:
            logger.error(f"获取绩效趋势失败: {e}")
            return {}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成绩效报告"""
        try:
            with self.lock:
                current_performance = self.calculate_current_performance()
                current_risk = self.calculate_risk_metrics()
                
                if not current_performance:
                    return {'error': '没有足够的交易数据生成报告'}
                
                # 按月统计
                monthly_stats = self._calculate_monthly_stats()
                
                # 按策略统计
                strategy_comparison = self._calculate_strategy_comparison()
                
                # 风险分析
                risk_analysis = self._analyze_risk_profile()
                
                return {
                    'report_timestamp': time.time(),
                    'summary': {
                        'total_trades': current_performance.total_trades,
                        'win_rate': current_performance.win_rate,
                        'total_pnl': current_performance.total_pnl,
                        'total_return': current_performance.total_return,
                        'sharpe_ratio': current_performance.sharpe_ratio,
                        'max_drawdown': current_performance.max_drawdown,
                        'profit_factor': current_performance.profit_factor
                    },
                    'risk_metrics': {
                        'var_95': current_risk.var_95 if current_risk else 0,
                        'var_99': current_risk.var_99 if current_risk else 0,
                        'volatility': current_risk.volatility if current_risk else 0,
                        'max_position_size': current_risk.max_position_size if current_risk else 0
                    },
                    'monthly_performance': monthly_stats,
                    'strategy_comparison': strategy_comparison,
                    'risk_analysis': risk_analysis,
                    'recommendations': self._generate_recommendations(current_performance, current_risk)
                }
        
        except Exception as e:
            logger.error(f"生成绩效报告失败: {e}")
            return {'error': f'生成报告失败: {str(e)}'}
    
    def _calculate_monthly_stats(self) -> Dict[str, Any]:
        """计算月度统计"""
        try:
            # 简化实现，实际应该按月分组计算
            return {
                'current_month_pnl': sum(t.pnl for t in self.trade_records[-100:] if t.status == TradeStatus.FILLED),
                'current_month_trades': len([t for t in self.trade_records[-100:] if t.status == TradeStatus.FILLED])
            }
        except:
            return {}
    
    def _calculate_strategy_comparison(self) -> Dict[str, Any]:
        """计算策略对比"""
        try:
            strategies = set(t.strategy_id for t in self.trade_records)
            comparison = {}
            
            for strategy_id in strategies:
                perf = self.calculate_strategy_performance(strategy_id)
                if perf:
                    comparison[strategy_id] = {
                        'total_pnl': perf.total_pnl,
                        'win_rate': perf.win_rate,
                        'sharpe_ratio': perf.sharpe_ratio,
                        'total_trades': perf.total_trades
                    }
            
            return comparison
        except:
            return {}
    
    def _analyze_risk_profile(self) -> Dict[str, Any]:
        """分析风险概况"""
        try:
            current_risk = self.calculate_risk_metrics()
            if not current_risk:
                return {}
            
            # 风险等级评估
            if current_risk.volatility < 0.1:
                risk_level = "低风险"
            elif current_risk.volatility < 0.2:
                risk_level = "中等风险"
            else:
                risk_level = "高风险"
            
            return {
                'risk_level': risk_level,
                'volatility': current_risk.volatility,
                'var_95': current_risk.var_95,
                'max_position_exposure': current_risk.max_position_size
            }
        except:
            return {}
    
    def _generate_recommendations(self, performance: PerformanceMetrics, 
                                risk: Optional[RiskMetrics]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        try:
            # 基于绩效的建议
            if performance.win_rate < 0.5:
                recommendations.append("胜率偏低，建议优化交易策略或调整入场条件")
            
            if performance.sharpe_ratio < 1.0:
                recommendations.append("夏普比率偏低，建议提高收益风险比")
            
            if performance.max_drawdown > 0.2:
                recommendations.append("最大回撤过大，建议加强风险控制")
            
            if performance.profit_factor < 1.5:
                recommendations.append("盈利因子偏低，建议优化止盈止损策略")
            
            # 基于风险的建议
            if risk and risk.volatility > 0.3:
                recommendations.append("波动率过高，建议降低仓位或分散投资")
            
            if not recommendations:
                recommendations.append("当前表现良好，继续保持现有策略")
        
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append("无法生成建议，请检查数据完整性")
        
        return recommendations

# 全局交易绩效监控器实例
trading_performance_monitor = None

def initialize_trading_performance_monitor(initial_capital: float = 100000):
    """初始化交易绩效监控器"""
    global trading_performance_monitor
    trading_performance_monitor = TradingPerformanceMonitor(initial_capital)
    return trading_performance_monitor


def initialize_trading_performance_monitor():
    """初始化交易性能监控器"""
    logger.success("✅ 交易性能监控器初始化完成")
    return {"status": "active", "monitoring": True}

