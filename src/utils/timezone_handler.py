"""
🌍 时区处理模块
处理中国与美国市场的时差问题，确保交易时间准确
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import pytz
import pendulum
from loguru import logger

from src.core.config import settings


class TimezoneHandler:
    """时区处理器"""
    
    def __init__(self):
        self.local_timezone = pytz.timezone(settings.timezone)  # Asia/Shanghai
        self.utc_timezone = pytz.UTC
        
        # 主要市场时区
        self.market_timezones = {
            "china": pytz.timezone("Asia/Shanghai"),
            "us_eastern": pytz.timezone("US/Eastern"),
            "us_central": pytz.timezone("US/Central"),
            "us_mountain": pytz.timezone("US/Mountain"),
            "us_pacific": pytz.timezone("US/Pacific"),
            "london": pytz.timezone("Europe/London"),
            "tokyo": pytz.timezone("Asia/Tokyo"),
            "sydney": pytz.timezone("Australia/Sydney"),
        }
        
        # 主要交易所时区
        self.exchange_timezones = {
            "binance": pytz.UTC,  # Binance使用UTC
            "okx": pytz.timezone("Asia/Shanghai"),  # OKX使用北京时间
            "bybit": pytz.UTC,  # Bybit使用UTC
            "huobi": pytz.timezone("Asia/Shanghai"),  # 火币使用北京时间
            "kucoin": pytz.UTC,  # KuCoin使用UTC
        }
        
        logger.info(f"时区处理器初始化完成，本地时区: {settings.timezone}")
    
    async def setup_timezone(self):
        """设置时区配置"""
        try:
            logger.info("配置时区处理...")
            
            # 获取当前时间信息
            now_local = datetime.now(self.local_timezone)
            now_utc = datetime.now(self.utc_timezone)
            
            logger.info(f"本地时间: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            logger.info(f"UTC时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # 计算与主要市场的时差
            time_differences = await self._calculate_time_differences()
            
            for market, diff in time_differences.items():
                logger.info(f"与{market}时差: {diff}")
            
            logger.success("时区配置完成")
            
        except Exception as e:
            logger.error(f"时区配置失败: {e}")
    
    async def _calculate_time_differences(self) -> Dict[str, str]:
        """计算与主要市场的时差"""
        differences = {}
        now_local = datetime.now(self.local_timezone)
        
        for market_name, market_tz in self.market_timezones.items():
            market_time = now_local.astimezone(market_tz)
            diff_hours = (market_time.utcoffset() - now_local.utcoffset()).total_seconds() / 3600
            
            if diff_hours > 0:
                differences[market_name] = f"+{diff_hours:.0f}小时"
            elif diff_hours < 0:
                differences[market_name] = f"{diff_hours:.0f}小时"
            else:
                differences[market_name] = "无时差"
        
        return differences
    
    def get_current_time(self, timezone_name: Optional[str] = None) -> datetime:
        """获取指定时区的当前时间"""
        if timezone_name is None:
            return datetime.now(self.local_timezone)
        
        if timezone_name in self.market_timezones:
            return datetime.now(self.market_timezones[timezone_name])
        elif timezone_name in self.exchange_timezones:
            return datetime.now(self.exchange_timezones[timezone_name])
        else:
            try:
                tz = pytz.timezone(timezone_name)
                return datetime.now(tz)
            except:
                logger.warning(f"未知时区: {timezone_name}，返回本地时间")
                return datetime.now(self.local_timezone)
    
    def convert_time(self, dt: datetime, from_tz: str, to_tz: str) -> datetime:
        """转换时间时区"""
        try:
            # 获取源时区
            if from_tz in self.market_timezones:
                from_timezone = self.market_timezones[from_tz]
            elif from_tz in self.exchange_timezones:
                from_timezone = self.exchange_timezones[from_tz]
            else:
                from_timezone = pytz.timezone(from_tz)
            
            # 获取目标时区
            if to_tz in self.market_timezones:
                to_timezone = self.market_timezones[to_tz]
            elif to_tz in self.exchange_timezones:
                to_timezone = self.exchange_timezones[to_tz]
            else:
                to_timezone = pytz.timezone(to_tz)
            
            # 如果输入时间没有时区信息，添加源时区
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            
            # 转换到目标时区
            return dt.astimezone(to_timezone)
            
        except Exception as e:
            logger.error(f"时间转换失败: {e}")
            return dt
    
    def get_market_hours(self, market: str) -> Dict[str, Any]:
        """获取市场交易时间"""
        market_hours = {
            "us_stock": {
                "timezone": "US/Eastern",
                "regular_hours": {
                    "start": "09:30",
                    "end": "16:00",
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                },
                "pre_market": {
                    "start": "04:00",
                    "end": "09:30"
                },
                "after_market": {
                    "start": "16:00",
                    "end": "20:00"
                }
            },
            "crypto": {
                "timezone": "UTC",
                "regular_hours": {
                    "start": "00:00",
                    "end": "23:59",
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                }
            },
            "forex": {
                "timezone": "UTC",
                "regular_hours": {
                    "start": "22:00",  # 周日22:00 UTC开始
                    "end": "22:00",   # 周五22:00 UTC结束
                    "days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                }
            }
        }
        
        return market_hours.get(market, {})
    
    def is_market_open(self, market: str) -> bool:
        """检查市场是否开放"""
        try:
            market_info = self.get_market_hours(market)
            if not market_info:
                return True  # 默认开放（如加密货币）
            
            market_tz = pytz.timezone(market_info["timezone"])
            current_time = datetime.now(market_tz)
            
            # 检查是否在交易日
            current_day = current_time.strftime("%A")
            trading_days = market_info["regular_hours"]["days"]
            
            if current_day not in trading_days:
                return False
            
            # 检查是否在交易时间
            start_time = datetime.strptime(market_info["regular_hours"]["start"], "%H:%M").time()
            end_time = datetime.strptime(market_info["regular_hours"]["end"], "%H:%M").time()
            current_time_only = current_time.time()
            
            return start_time <= current_time_only <= end_time
            
        except Exception as e:
            logger.error(f"检查市场开放状态失败: {e}")
            return True  # 默认开放
    
    def get_next_market_open(self, market: str) -> Optional[datetime]:
        """获取下次市场开放时间"""
        try:
            market_info = self.get_market_hours(market)
            if not market_info:
                return None
            
            market_tz = pytz.timezone(market_info["timezone"])
            current_time = datetime.now(market_tz)
            
            # 如果当前市场开放，返回None
            if self.is_market_open(market):
                return None
            
            # 计算下次开放时间
            trading_days = market_info["regular_hours"]["days"]
            start_time = datetime.strptime(market_info["regular_hours"]["start"], "%H:%M").time()
            
            # 从今天开始查找
            for i in range(7):  # 最多查找一周
                check_date = current_time + timedelta(days=i)
                check_day = check_date.strftime("%A")
                
                if check_day in trading_days:
                    next_open = market_tz.localize(
                        datetime.combine(check_date.date(), start_time)
                    )
                    
                    if next_open > current_time:
                        return next_open
            
            return None
            
        except Exception as e:
            logger.error(f"获取下次市场开放时间失败: {e}")
            return None
    
    def get_exchange_time(self, exchange: str) -> datetime:
        """获取交易所时间"""
        if exchange in self.exchange_timezones:
            return datetime.now(self.exchange_timezones[exchange])
        else:
            logger.warning(f"未知交易所: {exchange}，返回UTC时间")
            return datetime.now(self.utc_timezone)
    
    def format_time_for_display(self, dt: datetime, include_timezone: bool = True) -> str:
        """格式化时间用于显示"""
        if include_timezone:
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_trading_session_info(self) -> Dict[str, Any]:
        """获取当前交易时段信息"""
        now_utc = datetime.now(self.utc_timezone)
        now_local = datetime.now(self.local_timezone)
        
        session_info = {
            "local_time": self.format_time_for_display(now_local),
            "utc_time": self.format_time_for_display(now_utc),
            "active_sessions": [],
            "upcoming_sessions": []
        }
        
        # 检查各个市场的状态
        markets = ["us_stock", "crypto", "forex"]
        for market in markets:
            is_open = self.is_market_open(market)
            next_open = self.get_next_market_open(market)
            
            if is_open:
                session_info["active_sessions"].append({
                    "market": market,
                    "status": "open"
                })
            elif next_open:
                session_info["upcoming_sessions"].append({
                    "market": market,
                    "next_open": self.format_time_for_display(next_open.astimezone(self.local_timezone))
                })
        
        return session_info
    
    async def schedule_market_events(self):
        """调度市场事件"""
        logger.info("开始调度市场事件...")
        
        while True:
            try:
                # 检查市场开放/关闭事件
                session_info = self.get_trading_session_info()
                
                # 记录活跃交易时段
                if session_info["active_sessions"]:
                    active_markets = [s["market"] for s in session_info["active_sessions"]]
                    logger.debug(f"当前活跃市场: {', '.join(active_markets)}")
                
                # 等待下次检查
                await asyncio.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                logger.error(f"市场事件调度出错: {e}")
                await asyncio.sleep(60)  # 出错时等待1分钟
    
    def get_optimal_trading_hours(self) -> Dict[str, str]:
        """获取最佳交易时间段"""
        # 基于中国时区的最佳交易时间
        optimal_hours = {
            "crypto_high_volume": "21:00-02:00",  # 美国交易时间
            "crypto_asia_active": "09:00-12:00",  # 亚洲活跃时间
            "forex_london_open": "15:00-17:00",   # 伦敦开盘
            "forex_ny_open": "21:00-23:00",       # 纽约开盘
            "us_stock_premarket": "17:00-21:30",  # 美股盘前
            "us_stock_regular": "21:30-04:00",    # 美股正常交易
        }
        
        return optimal_hours
