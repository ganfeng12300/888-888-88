#!/usr/bin/env python3
"""
🌍 时区智能调度系统 - 24/7全球交易优化
Timezone Intelligent Scheduler - 24/7 Global Trading Optimization

优化中美时差交易效率：
- 智能时段分析
- 24/7不间断运行
- 市场活跃度监控
- 动态策略调整
"""

import time
import threading
import pytz
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from loguru import logger


class MarketSession(Enum):
    """市场时段"""
    ASIAN = "asian"          # 亚洲时段
    EUROPEAN = "european"    # 欧洲时段
    AMERICAN = "american"    # 美洲时段
    OVERLAP = "overlap"      # 重叠时段
    QUIET = "quiet"          # 安静时段


class TradingMode(Enum):
    """交易模式"""
    AGGRESSIVE = "aggressive"    # 激进模式
    MODERATE = "moderate"       # 温和模式
    CONSERVATIVE = "conservative" # 保守模式
    SLEEP = "sleep"            # 休眠模式


@dataclass
class MarketActivity:
    """市场活跃度"""
    session: MarketSession
    volatility: float
    volume: float
    spread: float
    liquidity_score: float
    activity_score: float
    timestamp: datetime


@dataclass
class TradingSchedule:
    """交易计划"""
    start_time: datetime
    end_time: datetime
    session: MarketSession
    mode: TradingMode
    max_positions: int
    position_size_multiplier: float
    risk_multiplier: float
    frequency_multiplier: float
    description: str


@dataclass
class TimezoneConfig:
    """时区配置"""
    local_timezone: str
    target_markets: List[str]
    trading_hours: Dict[str, Tuple[int, int]]  # 24小时制
    break_hours: Dict[str, List[Tuple[int, int]]]  # 休息时间
    priority_sessions: List[MarketSession]


class TimezoneScheduler:
    """时区智能调度系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化调度系统"""
        self.config = config or {}
        self.is_running = False
        self.lock = threading.Lock()
        
        # 时区配置
        self.local_tz = pytz.timezone(self.config.get('local_timezone', 'Asia/Shanghai'))
        self.utc_tz = pytz.UTC
        
        # 主要市场时区
        self.market_timezones = {
            'tokyo': pytz.timezone('Asia/Tokyo'),
            'shanghai': pytz.timezone('Asia/Shanghai'),
            'london': pytz.timezone('Europe/London'),
            'new_york': pytz.timezone('America/New_York'),
            'sydney': pytz.timezone('Australia/Sydney')
        }
        
        # 交易时段配置
        self.session_configs = {
            MarketSession.ASIAN: {
                'start_hour': 0,   # UTC时间
                'end_hour': 9,
                'peak_hours': [1, 2, 3, 4, 5, 6, 7, 8],
                'description': '亚洲时段 - 东京/上海/悉尼'
            },
            MarketSession.EUROPEAN: {
                'start_hour': 7,
                'end_hour': 16,
                'peak_hours': [8, 9, 10, 11, 12, 13, 14, 15],
                'description': '欧洲时段 - 伦敦/法兰克福'
            },
            MarketSession.AMERICAN: {
                'start_hour': 13,
                'end_hour': 22,
                'peak_hours': [14, 15, 16, 17, 18, 19, 20, 21],
                'description': '美洲时段 - 纽约/芝加哥'
            }
        }
        
        # 重叠时段（高活跃度）
        self.overlap_sessions = [
            {'start': 7, 'end': 9, 'name': '亚欧重叠'},    # 亚洲-欧洲
            {'start': 13, 'end': 16, 'name': '欧美重叠'}   # 欧洲-美洲
        ]
        
        # 状态数据
        self.current_session = MarketSession.QUIET
        self.current_mode = TradingMode.CONSERVATIVE
        self.market_activity_history = []
        self.trading_schedules = []
        self.session_performance = {}
        
        # 调度配置
        self.check_interval = self.config.get('check_interval', 60)  # 60秒检查一次
        self.activity_window = self.config.get('activity_window', 300)  # 5分钟活跃度窗口
        
        # 初始化调度
        self._initialize_schedules()
        
        # 启动调度器
        self._start_scheduler()
        
        logger.info("🌍 时区智能调度系统初始化完成")
    
    def _initialize_schedules(self):
        """初始化交易计划"""
        try:
            # 生成24小时交易计划
            base_date = datetime.now(self.utc_tz).replace(hour=0, minute=0, second=0, microsecond=0)
            
            for hour in range(24):
                current_time = base_date + timedelta(hours=hour)
                session = self._determine_market_session(hour)
                mode = self._determine_trading_mode(session, hour)
                
                schedule = TradingSchedule(
                    start_time=current_time,
                    end_time=current_time + timedelta(hours=1),
                    session=session,
                    mode=mode,
                    max_positions=self._get_max_positions(mode),
                    position_size_multiplier=self._get_position_multiplier(mode),
                    risk_multiplier=self._get_risk_multiplier(mode),
                    frequency_multiplier=self._get_frequency_multiplier(mode),
                    description=f"{session.value.title()} - {mode.value.title()}"
                )
                
                self.trading_schedules.append(schedule)
            
            logger.info(f"生成了 {len(self.trading_schedules)} 个小时的交易计划")
            
        except Exception as e:
            logger.error(f"初始化交易计划失败: {e}")
    
    def _determine_market_session(self, utc_hour: int) -> MarketSession:
        """确定市场时段"""
        # 检查重叠时段
        for overlap in self.overlap_sessions:
            if overlap['start'] <= utc_hour < overlap['end']:
                return MarketSession.OVERLAP
        
        # 检查主要时段
        for session, config in self.session_configs.items():
            if config['start_hour'] <= utc_hour < config['end_hour']:
                return session
        
        return MarketSession.QUIET
    
    def _determine_trading_mode(self, session: MarketSession, utc_hour: int) -> TradingMode:
        """确定交易模式"""
        if session == MarketSession.OVERLAP:
            return TradingMode.AGGRESSIVE  # 重叠时段最激进
        
        elif session in [MarketSession.ASIAN, MarketSession.EUROPEAN, MarketSession.AMERICAN]:
            # 检查是否在峰值时间
            config = self.session_configs[session]
            if utc_hour in config['peak_hours']:
                return TradingMode.MODERATE
            else:
                return TradingMode.CONSERVATIVE
        
        else:  # QUIET时段
            return TradingMode.SLEEP
    
    def _get_max_positions(self, mode: TradingMode) -> int:
        """获取最大持仓数"""
        multipliers = {
            TradingMode.AGGRESSIVE: 8,
            TradingMode.MODERATE: 5,
            TradingMode.CONSERVATIVE: 3,
            TradingMode.SLEEP: 1
        }
        return multipliers.get(mode, 3)
    
    def _get_position_multiplier(self, mode: TradingMode) -> float:
        """获取仓位倍数"""
        multipliers = {
            TradingMode.AGGRESSIVE: 1.5,
            TradingMode.MODERATE: 1.0,
            TradingMode.CONSERVATIVE: 0.7,
            TradingMode.SLEEP: 0.3
        }
        return multipliers.get(mode, 1.0)
    
    def _get_risk_multiplier(self, mode: TradingMode) -> float:
        """获取风险倍数"""
        multipliers = {
            TradingMode.AGGRESSIVE: 1.2,
            TradingMode.MODERATE: 1.0,
            TradingMode.CONSERVATIVE: 0.8,
            TradingMode.SLEEP: 0.5
        }
        return multipliers.get(mode, 1.0)
    
    def _get_frequency_multiplier(self, mode: TradingMode) -> float:
        """获取频率倍数"""
        multipliers = {
            TradingMode.AGGRESSIVE: 2.0,
            TradingMode.MODERATE: 1.0,
            TradingMode.CONSERVATIVE: 0.6,
            TradingMode.SLEEP: 0.2
        }
        return multipliers.get(mode, 1.0)
    
    def _start_scheduler(self):
        """启动调度器"""
        self.is_running = True
        
        # 主调度线程
        threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="SchedulerThread"
        ).start()
        
        # 活跃度监控线程
        threading.Thread(
            target=self._activity_monitor_loop,
            daemon=True,
            name="ActivityMonitorThread"
        ).start()
    
    def update_market_activity(self, volatility: float, volume: float, 
                             spread: float, liquidity_score: float):
        """更新市场活跃度"""
        try:
            # 计算活跃度评分
            activity_score = self._calculate_activity_score(
                volatility, volume, spread, liquidity_score
            )
            
            activity = MarketActivity(
                session=self.current_session,
                volatility=volatility,
                volume=volume,
                spread=spread,
                liquidity_score=liquidity_score,
                activity_score=activity_score,
                timestamp=datetime.now(self.utc_tz)
            )
            
            with self.lock:
                self.market_activity_history.append(activity)
                
                # 限制历史记录数量
                if len(self.market_activity_history) > 1000:
                    self.market_activity_history = self.market_activity_history[-1000:]
            
            logger.debug(f"更新市场活跃度: {activity_score:.2f}")
            
        except Exception as e:
            logger.error(f"更新市场活跃度失败: {e}")
    
    def _calculate_activity_score(self, volatility: float, volume: float, 
                                spread: float, liquidity_score: float) -> float:
        """计算活跃度评分"""
        try:
            # 标准化各项指标 (0-100)
            vol_score = min(volatility * 1000, 100)  # 波动率评分
            vol_score = min(volume / 1000000, 100)   # 成交量评分
            spread_score = max(100 - spread * 10000, 0)  # 点差评分（越小越好）
            liq_score = min(liquidity_score, 100)    # 流动性评分
            
            # 加权平均
            activity_score = (
                vol_score * 0.3 +
                vol_score * 0.3 +
                spread_score * 0.2 +
                liq_score * 0.2
            )
            
            return min(activity_score, 100.0)
            
        except Exception as e:
            logger.error(f"计算活跃度评分失败: {e}")
            return 50.0  # 默认中等活跃度
    
    def get_current_schedule(self) -> Optional[TradingSchedule]:
        """获取当前交易计划"""
        try:
            current_time = datetime.now(self.utc_tz)
            current_hour = current_time.hour
            
            # 查找当前小时的计划
            for schedule in self.trading_schedules:
                if schedule.start_time.hour == current_hour:
                    return schedule
            
            return None
            
        except Exception as e:
            logger.error(f"获取当前交易计划失败: {e}")
            return None
    
    def get_next_schedule(self) -> Optional[TradingSchedule]:
        """获取下一个交易计划"""
        try:
            current_time = datetime.now(self.utc_tz)
            next_hour = (current_time.hour + 1) % 24
            
            # 查找下一小时的计划
            for schedule in self.trading_schedules:
                if schedule.start_time.hour == next_hour:
                    return schedule
            
            return None
            
        except Exception as e:
            logger.error(f"获取下一个交易计划失败: {e}")
            return None
    
    def should_trade_now(self) -> Tuple[bool, str]:
        """判断当前是否应该交易"""
        try:
            current_schedule = self.get_current_schedule()
            if not current_schedule:
                return False, "无当前交易计划"
            
            # 检查交易模式
            if current_schedule.mode == TradingMode.SLEEP:
                return False, f"当前为休眠模式 ({current_schedule.session.value})"
            
            # 检查市场活跃度
            if len(self.market_activity_history) > 0:
                recent_activity = self.market_activity_history[-1]
                if recent_activity.activity_score < 20:  # 活跃度过低
                    return False, f"市场活跃度过低: {recent_activity.activity_score:.1f}"
            
            return True, f"可以交易 - {current_schedule.description}"
            
        except Exception as e:
            logger.error(f"判断交易条件失败: {e}")
            return False, f"判断异常: {str(e)}"
    
    def get_trading_parameters(self) -> Dict[str, Any]:
        """获取当前交易参数"""
        try:
            current_schedule = self.get_current_schedule()
            if not current_schedule:
                return self._get_default_parameters()
            
            return {
                "session": current_schedule.session.value,
                "mode": current_schedule.mode.value,
                "max_positions": current_schedule.max_positions,
                "position_size_multiplier": current_schedule.position_size_multiplier,
                "risk_multiplier": current_schedule.risk_multiplier,
                "frequency_multiplier": current_schedule.frequency_multiplier,
                "description": current_schedule.description
            }
            
        except Exception as e:
            logger.error(f"获取交易参数失败: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认交易参数"""
        return {
            "session": "quiet",
            "mode": "conservative",
            "max_positions": 3,
            "position_size_multiplier": 0.7,
            "risk_multiplier": 0.8,
            "frequency_multiplier": 0.6,
            "description": "默认保守模式"
        }
    
    def get_market_times(self) -> Dict[str, str]:
        """获取各市场当前时间"""
        try:
            current_utc = datetime.now(self.utc_tz)
            market_times = {}
            
            for market, tz in self.market_timezones.items():
                local_time = current_utc.astimezone(tz)
                market_times[market] = local_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            return market_times
            
        except Exception as e:
            logger.error(f"获取市场时间失败: {e}")
            return {}
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                # 更新当前时段和模式
                current_time = datetime.now(self.utc_tz)
                new_session = self._determine_market_session(current_time.hour)
                
                if new_session != self.current_session:
                    logger.info(f"市场时段切换: {self.current_session.value} -> {new_session.value}")
                    self.current_session = new_session
                    
                    # 更新交易模式
                    new_mode = self._determine_trading_mode(new_session, current_time.hour)
                    if new_mode != self.current_mode:
                        logger.info(f"交易模式切换: {self.current_mode.value} -> {new_mode.value}")
                        self.current_mode = new_mode
                
                # 记录时段性能
                self._update_session_performance()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"调度器循环失败: {e}")
                time.sleep(self.check_interval)
    
    def _activity_monitor_loop(self):
        """活跃度监控循环"""
        while self.is_running:
            try:
                time.sleep(self.activity_window)
                
                # 分析最近的活跃度趋势
                self._analyze_activity_trends()
                
                # 动态调整交易参数
                self._adjust_trading_parameters()
                
            except Exception as e:
                logger.error(f"活跃度监控失败: {e}")
    
    def _update_session_performance(self):
        """更新时段性能"""
        try:
            if self.current_session not in self.session_performance:
                self.session_performance[self.current_session] = {
                    'total_time': 0,
                    'active_time': 0,
                    'avg_activity': 0.0,
                    'trade_count': 0
                }
            
            # 更新统计数据
            perf = self.session_performance[self.current_session]
            perf['total_time'] += self.check_interval
            
            if len(self.market_activity_history) > 0:
                recent_activity = self.market_activity_history[-1]
                if recent_activity.activity_score > 30:
                    perf['active_time'] += self.check_interval
                
                # 更新平均活跃度
                perf['avg_activity'] = (
                    perf['avg_activity'] * 0.9 + recent_activity.activity_score * 0.1
                )
            
        except Exception as e:
            logger.error(f"更新时段性能失败: {e}")
    
    def _analyze_activity_trends(self):
        """分析活跃度趋势"""
        try:
            if len(self.market_activity_history) < 10:
                return
            
            # 分析最近的活跃度变化
            recent_activities = self.market_activity_history[-10:]
            activity_scores = [a.activity_score for a in recent_activities]
            
            # 计算趋势
            if len(activity_scores) > 1:
                trend = activity_scores[-1] - activity_scores[0]
                avg_activity = sum(activity_scores) / len(activity_scores)
                
                logger.debug(f"活跃度趋势: {trend:.2f}, 平均: {avg_activity:.2f}")
            
        except Exception as e:
            logger.error(f"分析活跃度趋势失败: {e}")
    
    def _adjust_trading_parameters(self):
        """动态调整交易参数"""
        try:
            if len(self.market_activity_history) < 5:
                return
            
            # 基于最近活跃度调整参数
            recent_activity = self.market_activity_history[-1]
            
            # 如果活跃度异常高，可能需要降低风险
            if recent_activity.activity_score > 80:
                logger.info("检测到异常高活跃度，建议降低风险")
            
            # 如果活跃度异常低，可能需要暂停交易
            elif recent_activity.activity_score < 10:
                logger.info("检测到异常低活跃度，建议暂停交易")
            
        except Exception as e:
            logger.error(f"调整交易参数失败: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        try:
            current_schedule = self.get_current_schedule()
            next_schedule = self.get_next_schedule()
            
            return {
                "timestamp": datetime.now(self.utc_tz).isoformat(),
                "is_running": self.is_running,
                "current_session": self.current_session.value,
                "current_mode": self.current_mode.value,
                "current_schedule": {
                    "session": current_schedule.session.value,
                    "mode": current_schedule.mode.value,
                    "max_positions": current_schedule.max_positions,
                    "position_size_multiplier": current_schedule.position_size_multiplier,
                    "risk_multiplier": current_schedule.risk_multiplier,
                    "frequency_multiplier": current_schedule.frequency_multiplier,
                    "description": current_schedule.description
                } if current_schedule else None,
                "next_schedule": {
                    "session": next_schedule.session.value,
                    "mode": next_schedule.mode.value,
                    "start_time": next_schedule.start_time.isoformat(),
                    "description": next_schedule.description
                } if next_schedule else None,
                "market_times": self.get_market_times(),
                "recent_activity": {
                    "score": self.market_activity_history[-1].activity_score,
                    "volatility": self.market_activity_history[-1].volatility,
                    "volume": self.market_activity_history[-1].volume,
                    "timestamp": self.market_activity_history[-1].timestamp.isoformat()
                } if self.market_activity_history else None,
                "session_performance": {
                    session.value: {
                        "avg_activity": perf["avg_activity"],
                        "active_time_pct": (perf["active_time"] / max(perf["total_time"], 1)) * 100,
                        "trade_count": perf["trade_count"]
                    }
                    for session, perf in self.session_performance.items()
                }
            }
            
        except Exception as e:
            logger.error(f"获取调度器状态失败: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """关闭调度器"""
        logger.info("正在关闭时区智能调度系统...")
        self.is_running = False
        
        # 等待线程结束
        time.sleep(2)
        
        logger.info("时区智能调度系统已关闭")


# 全局实例
_scheduler = None

def get_timezone_scheduler(config: Dict[str, Any] = None) -> TimezoneScheduler:
    """获取时区调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TimezoneScheduler(config)
    return _scheduler


if __name__ == "__main__":
    # 测试代码
    def test_timezone_scheduler():
        """测试时区调度器"""
        config = {
            "local_timezone": "Asia/Shanghai",
            "check_interval": 5
        }
        
        scheduler = get_timezone_scheduler(config)
        
        # 模拟市场活跃度更新
        scheduler.update_market_activity(0.02, 1000000, 0.0001, 80.0)
        
        # 检查是否应该交易
        should_trade, reason = scheduler.should_trade_now()
        print(f"应该交易: {should_trade}, 原因: {reason}")
        
        # 获取交易参数
        params = scheduler.get_trading_parameters()
        print(f"交易参数: {params}")
        
        # 获取调度器状态
        status = scheduler.get_scheduler_status()
        print("调度器状态:")
        import json
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 运行一段时间
        time.sleep(10)
        
        scheduler.shutdown()
    
    test_timezone_scheduler()
