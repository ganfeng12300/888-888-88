"""
🔍 异常检测系统 - 生产级实盘交易异常行为检测和预警系统
基于统计学习和机器学习的多维度异常检测，实时监控交易行为和市场异常
提供价格异常、交易量异常、行为异常、系统异常等全方位异常检测功能
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
from collections import deque
from loguru import logger

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available, some anomaly detection features will be limited")

class AnomalyType(Enum):
    """异常类型"""
    PRICE = "price"  # 价格异常
    VOLUME = "volume"  # 交易量异常
    SPREAD = "spread"  # 价差异常
    BEHAVIOR = "behavior"  # 行为异常
    SYSTEM = "system"  # 系统异常
    MARKET = "market"  # 市场异常

class AnomalySeverity(Enum):
    """异常严重程度"""
    LOW = "low"  # 低
    MEDIUM = "medium"  # 中等
    HIGH = "high"  # 高
    CRITICAL = "critical"  # 关键

@dataclass
class AnomalyEvent:
    """异常事件"""
    event_id: str  # 事件ID
    anomaly_type: AnomalyType  # 异常类型
    severity: AnomalySeverity  # 严重程度
    symbol: str  # 交易对
    description: str  # 描述
    current_value: float  # 当前值
    expected_range: Tuple[float, float]  # 期望范围
    confidence: float  # 置信度
    timestamp: float = field(default_factory=time.time)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class MarketData:
    """市场数据"""
    symbol: str  # 交易对
    price: float  # 价格
    volume: float  # 成交量
    bid: float  # 买价
    ask: float  # 卖价
    timestamp: float  # 时间戳

class StatisticalDetector:
    """统计异常检测器"""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_windows: Dict[str, Dict[str, deque]] = {}
        
        logger.info("统计异常检测器初始化完成")
    
    def detect_price_anomaly(self, symbol: str, price: float) -> Optional[AnomalyEvent]:
        """检测价格异常"""
        try:
            # 初始化数据窗口
            if symbol not in self.data_windows:
                self.data_windows[symbol] = {'prices': deque(maxlen=self.window_size)}
            
            price_window = self.data_windows[symbol]['prices']
            
            # 如果数据不足，不进行检测
            if len(price_window) < 30:
                price_window.append(price)
                return None
            
            # 计算统计指标
            prices = list(price_window)
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price == 0:
                price_window.append(price)
                return None
            
            # 计算Z分数
            z_score = abs(price - mean_price) / std_price
            
            # 添加新价格到窗口
            price_window.append(price)
            
            # 检查是否异常
            if z_score > self.z_threshold:
                severity = self._determine_severity(z_score, self.z_threshold)
                
                return AnomalyEvent(
                    event_id=f"price_anomaly_{symbol}_{int(time.time())}",
                    anomaly_type=AnomalyType.PRICE,
                    severity=severity,
                    symbol=symbol,
                    description=f"价格异常: Z分数 {z_score:.2f}",
                    current_value=price,
                    expected_range=(mean_price - 2*std_price, mean_price + 2*std_price),
                    confidence=min(z_score / self.z_threshold, 1.0),
                    metadata={'z_score': z_score, 'mean': mean_price, 'std': std_price}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"价格异常检测失败: {e}")
            return None
    
    def detect_volume_anomaly(self, symbol: str, volume: float) -> Optional[AnomalyEvent]:
        """检测交易量异常"""
        try:
            # 初始化数据窗口
            if symbol not in self.data_windows:
                self.data_windows[symbol] = {}
            if 'volumes' not in self.data_windows[symbol]:
                self.data_windows[symbol]['volumes'] = deque(maxlen=self.window_size)
            
            volume_window = self.data_windows[symbol]['volumes']
            
            # 如果数据不足，不进行检测
            if len(volume_window) < 30:
                volume_window.append(volume)
                return None
            
            # 计算统计指标
            volumes = list(volume_window)
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            if std_volume == 0 or mean_volume == 0:
                volume_window.append(volume)
                return None
            
            # 使用对数变换处理交易量的偏态分布
            log_volumes = np.log1p(volumes)  # log(1+x) 避免log(0)
            log_volume = np.log1p(volume)
            
            mean_log_volume = np.mean(log_volumes)
            std_log_volume = np.std(log_volumes)
            
            if std_log_volume == 0:
                volume_window.append(volume)
                return None
            
            # 计算Z分数
            z_score = abs(log_volume - mean_log_volume) / std_log_volume
            
            # 添加新交易量到窗口
            volume_window.append(volume)
            
            # 检查是否异常
            if z_score > self.z_threshold:
                severity = self._determine_severity(z_score, self.z_threshold)
                
                return AnomalyEvent(
                    event_id=f"volume_anomaly_{symbol}_{int(time.time())}",
                    anomaly_type=AnomalyType.VOLUME,
                    severity=severity,
                    symbol=symbol,
                    description=f"交易量异常: Z分数 {z_score:.2f}",
                    current_value=volume,
                    expected_range=(np.expm1(mean_log_volume - 2*std_log_volume), 
                                  np.expm1(mean_log_volume + 2*std_log_volume)),
                    confidence=min(z_score / self.z_threshold, 1.0),
                    metadata={'z_score': z_score, 'mean_volume': mean_volume, 'std_volume': std_volume}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"交易量异常检测失败: {e}")
            return None
    
    def detect_spread_anomaly(self, symbol: str, bid: float, ask: float) -> Optional[AnomalyEvent]:
        """检测价差异常"""
        try:
            if bid <= 0 or ask <= 0 or ask <= bid:
                return None
            
            spread = (ask - bid) / bid
            
            # 初始化数据窗口
            if symbol not in self.data_windows:
                self.data_windows[symbol] = {}
            if 'spreads' not in self.data_windows[symbol]:
                self.data_windows[symbol]['spreads'] = deque(maxlen=self.window_size)
            
            spread_window = self.data_windows[symbol]['spreads']
            
            # 如果数据不足，不进行检测
            if len(spread_window) < 30:
                spread_window.append(spread)
                return None
            
            # 计算统计指标
            spreads = list(spread_window)
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            
            if std_spread == 0:
                spread_window.append(spread)
                return None
            
            # 计算Z分数
            z_score = abs(spread - mean_spread) / std_spread
            
            # 添加新价差到窗口
            spread_window.append(spread)
            
            # 检查是否异常
            if z_score > self.z_threshold:
                severity = self._determine_severity(z_score, self.z_threshold)
                
                return AnomalyEvent(
                    event_id=f"spread_anomaly_{symbol}_{int(time.time())}",
                    anomaly_type=AnomalyType.SPREAD,
                    severity=severity,
                    symbol=symbol,
                    description=f"价差异常: {spread:.4%} (Z分数 {z_score:.2f})",
                    current_value=spread,
                    expected_range=(mean_spread - 2*std_spread, mean_spread + 2*std_spread),
                    confidence=min(z_score / self.z_threshold, 1.0),
                    metadata={'z_score': z_score, 'mean_spread': mean_spread, 'std_spread': std_spread}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"价差异常检测失败: {e}")
            return None
    
    def _determine_severity(self, z_score: float, threshold: float) -> AnomalySeverity:
        """确定异常严重程度"""
        if z_score > threshold * 3:
            return AnomalySeverity.CRITICAL
        elif z_score > threshold * 2:
            return AnomalySeverity.HIGH
        elif z_score > threshold * 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class MLAnomalyDetector:
    """机器学习异常检测器"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_history: Dict[str, List[List[float]]] = {}
        self.min_samples = 100
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn不可用，机器学习异常检测功能受限")
        
        logger.info("机器学习异常检测器初始化完成")
    
    def update_and_detect(self, symbol: str, features: List[float]) -> Optional[AnomalyEvent]:
        """更新模型并检测异常"""
        try:
            if not SKLEARN_AVAILABLE:
                return None
            
            # 初始化特征历史
            if symbol not in self.feature_history:
                self.feature_history[symbol] = []
            
            # 添加新特征
            self.feature_history[symbol].append(features)
            
            # 保持历史数据在合理范围内
            if len(self.feature_history[symbol]) > 1000:
                self.feature_history[symbol] = self.feature_history[symbol][-500:]
            
            # 如果数据不足，不进行检测
            if len(self.feature_history[symbol]) < self.min_samples:
                return None
            
            # 准备训练数据
            X = np.array(self.feature_history[symbol])
            
            # 数据标准化
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()
                X_scaled = self.scalers[symbol].fit_transform(X)
            else:
                X_scaled = self.scalers[symbol].transform(X)
            
            # 训练或更新模型
            if symbol not in self.models or len(self.feature_history[symbol]) % 50 == 0:
                self.models[symbol] = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.models[symbol].fit(X_scaled)
            
            # 检测当前样本
            current_features_scaled = self.scalers[symbol].transform([features])
            anomaly_score = self.models[symbol].decision_function(current_features_scaled)[0]
            is_anomaly = self.models[symbol].predict(current_features_scaled)[0] == -1
            
            if is_anomaly:
                # 计算置信度
                confidence = abs(anomaly_score)
                
                # 确定严重程度
                if confidence > 0.5:
                    severity = AnomalySeverity.HIGH
                elif confidence > 0.3:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                return AnomalyEvent(
                    event_id=f"ml_anomaly_{symbol}_{int(time.time())}",
                    anomaly_type=AnomalyType.BEHAVIOR,
                    severity=severity,
                    symbol=symbol,
                    description=f"机器学习检测到行为异常 (异常分数: {anomaly_score:.3f})",
                    current_value=anomaly_score,
                    expected_range=(-0.1, 0.1),  # 正常范围的近似值
                    confidence=confidence,
                    metadata={'anomaly_score': anomaly_score, 'features': features}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"机器学习异常检测失败: {e}")
            return None

class SystemAnomalyDetector:
    """系统异常检测器"""
    
    def __init__(self):
        self.api_response_times: Dict[str, deque] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_error_reset = time.time()
        
        # 阈值设置
        self.max_response_time = 5.0  # 最大响应时间（秒）
        self.max_error_rate = 0.1  # 最大错误率
        self.error_reset_interval = 3600  # 错误计数重置间隔（秒）
        
        logger.info("系统异常检测器初始化完成")
    
    def record_api_response(self, exchange: str, response_time: float, success: bool) -> Optional[AnomalyEvent]:
        """记录API响应并检测异常"""
        try:
            # 初始化响应时间历史
            if exchange not in self.api_response_times:
                self.api_response_times[exchange] = deque(maxlen=100)
                self.error_counts[exchange] = 0
            
            # 记录响应时间
            self.api_response_times[exchange].append(response_time)
            
            # 记录错误
            if not success:
                self.error_counts[exchange] += 1
            
            # 检查是否需要重置错误计数
            if time.time() - self.last_error_reset > self.error_reset_interval:
                self.error_counts = {k: 0 for k in self.error_counts}
                self.last_error_reset = time.time()
            
            anomalies = []
            
            # 检查响应时间异常
            if response_time > self.max_response_time:
                anomalies.append(AnomalyEvent(
                    event_id=f"response_time_{exchange}_{int(time.time())}",
                    anomaly_type=AnomalyType.SYSTEM,
                    severity=AnomalySeverity.MEDIUM,
                    symbol=exchange,
                    description=f"API响应时间过长: {response_time:.2f}s",
                    current_value=response_time,
                    expected_range=(0, self.max_response_time),
                    confidence=min(response_time / self.max_response_time, 1.0),
                    metadata={'response_time': response_time}
                ))
            
            # 检查错误率异常
            total_requests = len(self.api_response_times[exchange])
            if total_requests > 10:
                error_rate = self.error_counts[exchange] / total_requests
                if error_rate > self.max_error_rate:
                    anomalies.append(AnomalyEvent(
                        event_id=f"error_rate_{exchange}_{int(time.time())}",
                        anomaly_type=AnomalyType.SYSTEM,
                        severity=AnomalySeverity.HIGH,
                        symbol=exchange,
                        description=f"API错误率过高: {error_rate:.2%}",
                        current_value=error_rate,
                        expected_range=(0, self.max_error_rate),
                        confidence=min(error_rate / self.max_error_rate, 1.0),
                        metadata={'error_rate': error_rate, 'error_count': self.error_counts[exchange]}
                    ))
            
            return anomalies[0] if anomalies else None
        
        except Exception as e:
            logger.error(f"系统异常检测失败: {e}")
            return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            health_status = {}
            
            for exchange in self.api_response_times:
                response_times = list(self.api_response_times[exchange])
                total_requests = len(response_times)
                
                if total_requests > 0:
                    avg_response_time = np.mean(response_times)
                    error_rate = self.error_counts[exchange] / total_requests
                    
                    health_status[exchange] = {
                        'avg_response_time': avg_response_time,
                        'error_rate': error_rate,
                        'total_requests': total_requests,
                        'error_count': self.error_counts[exchange],
                        'health_score': self._calculate_health_score(avg_response_time, error_rate)
                    }
            
            return health_status
        
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {}
    
    def _calculate_health_score(self, avg_response_time: float, error_rate: float) -> float:
        """计算健康分数"""
        # 响应时间分数 (0-50分)
        response_score = max(0, 50 - (avg_response_time / self.max_response_time) * 50)
        
        # 错误率分数 (0-50分)
        error_score = max(0, 50 - (error_rate / self.max_error_rate) * 50)
        
        return response_score + error_score

class AnomalyDetectionSystem:
    """异常检测系统主类"""
    
    def __init__(self):
        self.statistical_detector = StatisticalDetector()
        self.ml_detector = MLAnomalyDetector()
        self.system_detector = SystemAnomalyDetector()
        
        # 异常事件历史
        self.anomaly_events: List[AnomalyEvent] = []
        
        # 回调函数
        self.anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("异常检测系统初始化完成")
    
    def detect_market_anomalies(self, market_data: MarketData) -> List[AnomalyEvent]:
        """检测市场异常"""
        anomalies = []
        
        try:
            with self.lock:
                # 价格异常检测
                price_anomaly = self.statistical_detector.detect_price_anomaly(
                    market_data.symbol, market_data.price
                )
                if price_anomaly:
                    anomalies.append(price_anomaly)
                
                # 交易量异常检测
                volume_anomaly = self.statistical_detector.detect_volume_anomaly(
                    market_data.symbol, market_data.volume
                )
                if volume_anomaly:
                    anomalies.append(volume_anomaly)
                
                # 价差异常检测
                spread_anomaly = self.statistical_detector.detect_spread_anomaly(
                    market_data.symbol, market_data.bid, market_data.ask
                )
                if spread_anomaly:
                    anomalies.append(spread_anomaly)
                
                # 机器学习异常检测
                features = [
                    market_data.price,
                    market_data.volume,
                    (market_data.ask - market_data.bid) / market_data.bid,  # 价差
                    market_data.timestamp % 86400  # 时间特征
                ]
                
                ml_anomaly = self.ml_detector.update_and_detect(market_data.symbol, features)
                if ml_anomaly:
                    anomalies.append(ml_anomaly)
                
                # 处理检测到的异常
                for anomaly in anomalies:
                    self._process_anomaly(anomaly)
                
                return anomalies
        
        except Exception as e:
            logger.error(f"市场异常检测失败: {e}")
            return []
    
    def record_system_event(self, exchange: str, response_time: float, success: bool) -> Optional[AnomalyEvent]:
        """记录系统事件"""
        try:
            with self.lock:
                anomaly = self.system_detector.record_api_response(exchange, response_time, success)
                if anomaly:
                    self._process_anomaly(anomaly)
                return anomaly
        
        except Exception as e:
            logger.error(f"记录系统事件失败: {e}")
            return None
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]):
        """添加异常回调函数"""
        self.anomaly_callbacks.append(callback)
    
    def _process_anomaly(self, anomaly: AnomalyEvent):
        """处理异常事件"""
        try:
            # 添加到历史记录
            self.anomaly_events.append(anomaly)
            
            # 保持历史记录在合理范围内
            if len(self.anomaly_events) > 10000:
                self.anomaly_events = self.anomaly_events[-5000:]
            
            # 调用回调函数
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"异常回调函数执行失败: {e}")
            
            # 记录日志
            if anomaly.severity == AnomalySeverity.CRITICAL:
                logger.critical(f"关键异常: {anomaly.description}")
            elif anomaly.severity == AnomalySeverity.HIGH:
                logger.error(f"高级异常: {anomaly.description}")
            elif anomaly.severity == AnomalySeverity.MEDIUM:
                logger.warning(f"中级异常: {anomaly.description}")
            else:
                logger.info(f"低级异常: {anomaly.description}")
        
        except Exception as e:
            logger.error(f"处理异常事件失败: {e}")
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        try:
            with self.lock:
                # 最近异常统计
                recent_anomalies = [a for a in self.anomaly_events if time.time() - a.timestamp < 3600]
                
                # 按类型统计
                type_counts = {}
                severity_counts = {}
                
                for anomaly in recent_anomalies:
                    anomaly_type = anomaly.anomaly_type.value
                    severity = anomaly.severity.value
                    
                    type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                return {
                    'total_anomalies': len(self.anomaly_events),
                    'recent_anomalies': len(recent_anomalies),
                    'anomaly_type_counts': type_counts,
                    'severity_counts': severity_counts,
                    'system_health': self.system_detector.get_system_health()
                }
        
        except Exception as e:
            logger.error(f"获取异常摘要失败: {e}")
            return {}
    
    def get_recent_anomalies(self, limit: int = 50) -> List[AnomalyEvent]:
        """获取最近的异常事件"""
        with self.lock:
            return sorted(self.anomaly_events, key=lambda x: x.timestamp, reverse=True)[:limit]

# 全局异常检测系统实例
anomaly_detection_system = AnomalyDetectionSystem()



def initialize_anomaly_detection():
    """初始化异常检测系统"""
    from src.security.anomaly_detection import AnomalyDetectionSystem
    system = AnomalyDetectionSystem()
    logger.success("✅ 异常检测系统初始化完成")
    return system

