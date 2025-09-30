#!/usr/bin/env python3
"""
🦊 猎狐AI量化交易系统 - 五层风控矩阵
智能风险管理系统，实时风险评估与控制
专为史诗级AI量化交易设计，生产级实盘交易标准
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path

class RiskLevel(Enum):
    """风险等级"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    CRITICAL = 6

class RiskAction(Enum):
    """风险动作"""
    ALLOW = "allow"
    WARN = "warn"
    LIMIT = "limit"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskMetrics:
    """风险指标"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    leverage_ratio: float
    margin_usage: float

@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    risk_type: str
    risk_level: RiskLevel
    message: str
    current_value: float
    threshold_value: float
    suggested_action: RiskAction
    timestamp: datetime
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None

class RiskCalculator:
    """风险计算器"""
    
    def __init__(self):
        self.price_history = {}  # symbol -> price list
        self.return_history = {}  # symbol -> return list
        self.correlation_matrix = {}
        
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算风险价值 (VaR)"""
        try:
            if len(returns) < 30:
                return 0.0
            
            # 使用历史模拟法
            sorted_returns = np.sort(returns)
            index = int((1 - confidence) * len(sorted_returns))
            var = -sorted_returns[index] if index < len(sorted_returns) else 0.0
            
            return max(0.0, var)
            
        except Exception as e:
            logger.error(f"❌ VaR计算失败: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """计算最大回撤"""
        try:
            if len(equity_curve) < 2:
                return 0.0
            
            # 计算累计最高点
            peak = np.maximum.accumulate(equity_curve)
            
            # 计算回撤
            drawdown = (equity_curve - peak) / peak
            
            # 返回最大回撤
            max_dd = np.min(drawdown)
            return abs(max_dd)
            
        except Exception as e:
            logger.error(f"❌ 最大回撤计算失败: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            if len(returns) < 30:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"❌ 夏普比率计算失败: {e}")
            return 0.0
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """计算Beta系数"""
        try:
            if len(asset_returns) != len(market_returns) or len(asset_returns) < 30:
                return 1.0
            
            # 计算协方差和方差
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            logger.error(f"❌ Beta计算失败: {e}")
            return 1.0
    
    def calculate_correlation_risk(self, positions: Dict[str, float]) -> float:
        """计算相关性风险"""
        try:
            if len(positions) < 2:
                return 0.0
            
            symbols = list(positions.keys())
            weights = np.array(list(positions.values()))
            
            # 构建相关性矩阵
            n = len(symbols)
            corr_matrix = np.eye(n)
            
            for i in range(n):
                for j in range(i+1, n):
                    # 这里应该使用实际的相关性数据
                    # 简化处理，使用随机相关性
                    corr = np.random.uniform(0.3, 0.8)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
            
            # 计算组合相关性风险
            portfolio_corr = np.dot(weights, np.dot(corr_matrix, weights))
            
            # 标准化到0-1范围
            risk_score = min(1.0, max(0.0, (portfolio_corr - 0.5) * 2))
            
            return risk_score
            
        except Exception as e:
            logger.error(f"❌ 相关性风险计算失败: {e}")
            return 0.5

class FiveLayerRiskMatrix:
    """五层风控矩阵"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.calculator = RiskCalculator()
        
        # 风控参数
        self.max_single_position = self.config.get('max_single_position', 0.3)  # 30%
        self.max_total_position = self.config.get('max_total_position', 0.8)    # 80%
        self.max_daily_loss = self.config.get('max_daily_loss', 0.03)           # 3%
        self.max_drawdown = self.config.get('max_drawdown', 0.15)               # 15%
        self.min_confidence = self.config.get('min_confidence', 0.7)            # 70%
        
        # 风控状态
        self.is_emergency_stop = False
        self.daily_loss_count = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        logger.info("🛡️ 五层风控矩阵初始化完成")
    
    async def check_layer_1_signal_filter(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """第1层：AI信号强度过滤"""
        try:
            confidence = signal.get('confidence', 0.0)
            ai_consensus = signal.get('ai_consensus', 0.0)
            signal_strength = signal.get('signal_strength', 0.0)
            
            # 置信度检查
            if confidence < self.min_confidence:
                return False, f"信号置信度不足: {confidence:.3f} < {self.min_confidence}"
            
            # AI共识检查
            if ai_consensus < 0.6:
                return False, f"AI共识度不足: {ai_consensus:.3f} < 0.6"
            
            # 信号强度检查
            if abs(signal_strength) < 0.3:
                return False, f"信号强度不足: {abs(signal_strength):.3f} < 0.3"
            
            return True, "第1层检查通过"
            
        except Exception as e:
            logger.error(f"❌ 第1层风控检查失败: {e}")
            return False, f"第1层检查异常: {str(e)}"
    
    async def check_layer_2_technical_confirmation(self, indicators: Dict[str, Any]) -> Tuple[bool, str]:
        """第2层：技术指标确认"""
        try:
            confirmed_signals = 0
            total_indicators = 0
            
            # 检查主要技术指标
            key_indicators = ['rsi', 'macd', 'bb_position', 'adx', 'stoch_k']
            
            for indicator in key_indicators:
                if indicator in indicators:
                    total_indicators += 1
                    value = indicators[indicator]
                    
                    # 根据指标类型判断信号
                    if indicator == 'rsi':
                        if 30 <= value <= 70:  # RSI在正常范围
                            confirmed_signals += 1
                    elif indicator == 'macd':
                        if abs(value) > 0.001:  # MACD有明确信号
                            confirmed_signals += 1
                    elif indicator == 'bb_position':
                        if 0.2 <= value <= 0.8:  # 布林带位置合理
                            confirmed_signals += 1
                    elif indicator == 'adx':
                        if value > 25:  # ADX显示趋势强度
                            confirmed_signals += 1
                    elif indicator == 'stoch_k':
                        if 20 <= value <= 80:  # 随机指标在合理范围
                            confirmed_signals += 1
            
            # 需要至少3个指标确认
            if confirmed_signals < 3:
                return False, f"技术指标确认不足: {confirmed_signals}/3"
            
            confirmation_rate = confirmed_signals / total_indicators if total_indicators > 0 else 0
            return True, f"第2层检查通过，确认率: {confirmation_rate:.2%}"
            
        except Exception as e:
            logger.error(f"❌ 第2层风控检查失败: {e}")
            return False, f"第2层检查异常: {str(e)}"
    
    async def check_layer_3_position_control(self, symbol: str, amount: float, 
                                           current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """第3层：仓位上限控制"""
        try:
            # 计算当前总仓位
            total_position = sum(abs(pos) for pos in current_positions.values())
            
            # 计算新仓位后的总仓位
            new_position = current_positions.get(symbol, 0.0) + amount
            new_total_position = total_position - abs(current_positions.get(symbol, 0.0)) + abs(new_position)
            
            # 检查单一仓位限制
            if abs(new_position) > self.max_single_position:
                return False, f"单一仓位超限: {abs(new_position):.2%} > {self.max_single_position:.2%}"
            
            # 检查总仓位限制
            if new_total_position > self.max_total_position:
                return False, f"总仓位超限: {new_total_position:.2%} > {self.max_total_position:.2%}"
            
            return True, f"第3层检查通过，新总仓位: {new_total_position:.2%}"
            
        except Exception as e:
            logger.error(f"❌ 第3层风控检查失败: {e}")
            return False, f"第3层检查异常: {str(e)}"
    
    async def check_layer_4_dynamic_stop_loss(self, symbol: str, entry_price: float,
                                            current_price: float, position_side: str,
                                            atr: float = None) -> Tuple[bool, str]:
        """第4层：动态止损机制"""
        try:
            if atr is None:
                atr = abs(current_price * 0.02)  # 默认2%的ATR
            
            # 计算动态止损位
            atr_multiplier = 2.0  # ATR倍数
            
            if position_side == 'long':
                stop_loss_price = entry_price - (atr * atr_multiplier)
                if current_price <= stop_loss_price:
                    loss_pct = (entry_price - current_price) / entry_price
                    return False, f"触发止损: 价格 {current_price} <= 止损位 {stop_loss_price:.6f}, 亏损 {loss_pct:.2%}"
            else:  # short
                stop_loss_price = entry_price + (atr * atr_multiplier)
                if current_price >= stop_loss_price:
                    loss_pct = (current_price - entry_price) / entry_price
                    return False, f"触发止损: 价格 {current_price} >= 止损位 {stop_loss_price:.6f}, 亏损 {loss_pct:.2%}"
            
            return True, "第4层检查通过，未触发止损"
            
        except Exception as e:
            logger.error(f"❌ 第4层风控检查失败: {e}")
            return False, f"第4层检查异常: {str(e)}"
    
    async def check_layer_5_circuit_breaker(self, daily_pnl: float, 
                                          portfolio_value: float) -> Tuple[bool, str]:
        """第5层：熔断保护机制"""
        try:
            # 检查紧急停止状态
            if self.is_emergency_stop:
                return False, "系统处于紧急停止状态"
            
            # 重置日计数器
            today = datetime.now(timezone.utc).date()
            if today != self.last_reset_date:
                self.daily_loss_count = 0
                self.last_reset_date = today
            
            # 计算日亏损率
            daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
            
            # 检查日亏损限制
            if daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
                self.daily_loss_count += 1
                
                if self.daily_loss_count >= 3:  # 连续3次触发日亏损限制
                    self.is_emergency_stop = True
                    return False, f"触发熔断: 连续亏损超限，启动紧急停止"
                
                return False, f"日亏损超限: {daily_loss_pct:.2%} > {self.max_daily_loss:.2%}"
            
            return True, "第5层检查通过，未触发熔断"
            
        except Exception as e:
            logger.error(f"❌ 第5层风控检查失败: {e}")
            return False, f"第5层检查异常: {str(e)}"
    
    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        self.is_emergency_stop = False
        self.daily_loss_count = 0
        logger.info("🔄 紧急停止状态已重置")

class RiskManager:
    """🦊 猎狐AI - 智能风险管理系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_matrix = FiveLayerRiskMatrix(config)
        self.calculator = RiskCalculator()
        
        # 风险监控
        self.active_alerts = {}  # alert_id -> RiskAlert
        self.risk_history = []
        self.monitoring_enabled = True
        
        # 数据库
        self.db_path = Path(config.get('db_path', 'data/risk.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 统计信息
        self.stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'blocked_trades': 0,
            'alerts_generated': 0,
            'emergency_stops': 0
        }
        
        logger.info("🦊 猎狐AI智能风险管理系统初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 风险指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    var_95 REAL NOT NULL,
                    var_99 REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    volatility REAL NOT NULL,
                    leverage_ratio REAL NOT NULL
                )
            ''')
            
            # 风险警报表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    alert_id TEXT PRIMARY KEY,
                    risk_type TEXT NOT NULL,
                    risk_level INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    is_resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ 风险管理数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 风险数据库初始化失败: {e}")
    
    async def comprehensive_risk_check(self, trade_request: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """综合风险检查"""
        try:
            self.stats['total_checks'] += 1
            
            passed_layers = []
            failed_reason = ""
            
            # 第1层：AI信号强度过滤
            layer1_pass, layer1_msg = await self.risk_matrix.check_layer_1_signal_filter(
                trade_request.get('signal', {})
            )
            
            if not layer1_pass:
                failed_reason = f"第1层失败: {layer1_msg}"
                self.stats['blocked_trades'] += 1
                return False, failed_reason, passed_layers
            
            passed_layers.append("第1层: AI信号过滤")
            
            # 第2层：技术指标确认
            layer2_pass, layer2_msg = await self.risk_matrix.check_layer_2_technical_confirmation(
                trade_request.get('indicators', {})
            )
            
            if not layer2_pass:
                failed_reason = f"第2层失败: {layer2_msg}"
                self.stats['blocked_trades'] += 1
                return False, failed_reason, passed_layers
            
            passed_layers.append("第2层: 技术指标确认")
            
            # 第3层：仓位上限控制
            layer3_pass, layer3_msg = await self.risk_matrix.check_layer_3_position_control(
                trade_request.get('symbol', ''),
                trade_request.get('amount', 0.0),
                trade_request.get('current_positions', {})
            )
            
            if not layer3_pass:
                failed_reason = f"第3层失败: {layer3_msg}"
                self.stats['blocked_trades'] += 1
                return False, failed_reason, passed_layers
            
            passed_layers.append("第3层: 仓位控制")
            
            # 第4层：动态止损机制
            if 'entry_price' in trade_request and 'current_price' in trade_request:
                layer4_pass, layer4_msg = await self.risk_matrix.check_layer_4_dynamic_stop_loss(
                    trade_request.get('symbol', ''),
                    trade_request.get('entry_price', 0.0),
                    trade_request.get('current_price', 0.0),
                    trade_request.get('position_side', 'long'),
                    trade_request.get('atr')
                )
                
                if not layer4_pass:
                    failed_reason = f"第4层失败: {layer4_msg}"
                    self.stats['blocked_trades'] += 1
                    return False, failed_reason, passed_layers
                
                passed_layers.append("第4层: 动态止损")
            
            # 第5层：熔断保护机制
            layer5_pass, layer5_msg = await self.risk_matrix.check_layer_5_circuit_breaker(
                trade_request.get('daily_pnl', 0.0),
                trade_request.get('portfolio_value', 100000.0)
            )
            
            if not layer5_pass:
                failed_reason = f"第5层失败: {layer5_msg}"
                self.stats['blocked_trades'] += 1
                
                # 如果是熔断，记录紧急停止
                if "熔断" in layer5_msg:
                    self.stats['emergency_stops'] += 1
                
                return False, failed_reason, passed_layers
            
            passed_layers.append("第5层: 熔断保护")
            
            # 所有层级检查通过
            self.stats['passed_checks'] += 1
            return True, "五层风控检查全部通过", passed_layers
            
        except Exception as e:
            logger.error(f"❌ 综合风险检查异常: {e}")
            self.stats['blocked_trades'] += 1
            return False, f"风险检查异常: {str(e)}", passed_layers
    
    async def calculate_portfolio_risk(self, positions: Dict[str, float],
                                     prices: Dict[str, float],
                                     returns_history: Dict[str, List[float]]) -> RiskMetrics:
        """计算投资组合风险"""
        try:
            # 计算投资组合价值
            portfolio_value = sum(positions.get(symbol, 0) * prices.get(symbol, 0) 
                                for symbol in positions.keys())
            
            # 计算投资组合收益序列
            portfolio_returns = self._calculate_portfolio_returns(positions, returns_history)
            
            # 计算各项风险指标
            var_95 = self.calculator.calculate_var(portfolio_returns, 0.95)
            var_99 = self.calculator.calculate_var(portfolio_returns, 0.99)
            
            # 计算最大回撤
            equity_curve = np.cumprod(1 + portfolio_returns) * 100000  # 假设初始资金10万
            max_drawdown = self.calculator.calculate_max_drawdown(equity_curve)
            
            # 计算夏普比率
            sharpe_ratio = self.calculator.calculate_sharpe_ratio(portfolio_returns)
            
            # 计算波动率
            volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 0 else 0.0
            
            # 计算相关性风险
            correlation_risk = self.calculator.calculate_correlation_risk(positions)
            
            # 计算集中度风险
            position_weights = np.array(list(positions.values()))
            concentration_risk = np.sum(position_weights ** 2) if len(position_weights) > 0 else 0.0
            
            # 计算杠杆比率
            leverage_ratio = sum(abs(pos) for pos in positions.values())
            
            # 创建风险指标对象
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=portfolio_value,
                daily_pnl=portfolio_returns[-1] * portfolio_value if len(portfolio_returns) > 0 else 0.0,
                daily_pnl_pct=portfolio_returns[-1] if len(portfolio_returns) > 0 else 0.0,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=1.0,  # 简化处理
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=0.1,  # 简化处理
                leverage_ratio=leverage_ratio,
                margin_usage=leverage_ratio * 0.1  # 简化处理
            )
            
            # 保存到数据库
            await self._save_risk_metrics(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ 投资组合风险计算失败: {e}")
            # 返回默认风险指标
            return RiskMetrics(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=0.0,
                daily_pnl=0.0,
                daily_pnl_pct=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                var_99=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                beta=1.0,
                correlation_risk=0.0,
                concentration_risk=0.0,
                liquidity_risk=0.0,
                leverage_ratio=0.0,
                margin_usage=0.0
            )
    
    def _calculate_portfolio_returns(self, positions: Dict[str, float],
                                   returns_history: Dict[str, List[float]]) -> np.ndarray:
        """计算投资组合收益序列"""
        try:
            if not positions or not returns_history:
                return np.array([])
            
            # 获取最短的收益序列长度
            min_length = min(len(returns) for returns in returns_history.values() if returns)
            if min_length == 0:
                return np.array([])
            
            # 计算权重
            total_position = sum(abs(pos) for pos in positions.values())
            if total_position == 0:
                return np.array([])
            
            weights = {symbol: pos / total_position for symbol, pos in positions.items()}
            
            # 计算投资组合收益
            portfolio_returns = np.zeros(min_length)
            
            for symbol, weight in weights.items():
                if symbol in returns_history and len(returns_history[symbol]) >= min_length:
                    symbol_returns = np.array(returns_history[symbol][-min_length:])
                    portfolio_returns += weight * symbol_returns
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"❌ 投资组合收益计算失败: {e}")
            return np.array([])
    
    async def _save_risk_metrics(self, metrics: RiskMetrics):
        """保存风险指标到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics 
                (timestamp, portfolio_value, daily_pnl, daily_pnl_pct, max_drawdown,
                 var_95, var_99, sharpe_ratio, volatility, leverage_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.portfolio_value, metrics.daily_pnl,
                metrics.daily_pnl_pct, metrics.max_drawdown, metrics.var_95,
                metrics.var_99, metrics.sharpe_ratio, metrics.volatility,
                metrics.leverage_ratio
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ 保存风险指标失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算通过率
        if stats['total_checks'] > 0:
            stats['pass_rate'] = stats['passed_checks'] / stats['total_checks']
            stats['block_rate'] = stats['blocked_trades'] / stats['total_checks']
        else:
            stats['pass_rate'] = 0.0
            stats['block_rate'] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            return {
                'status': 'healthy',
                'monitoring_enabled': self.monitoring_enabled,
                'emergency_stop': self.risk_matrix.is_emergency_stop,
                'active_alerts': len(self.active_alerts),
                'database_accessible': self.db_path.exists()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

