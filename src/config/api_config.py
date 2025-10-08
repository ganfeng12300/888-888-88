#!/usr/bin/env python3
"""
🔧 888-888-88 API配置管理器
Production-Grade API Configuration Manager
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from loguru import logger
import ccxt
import ccxt.pro as ccxtpro

from src.core.error_handling_system import handle_errors, critical_section

@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    api_key: str
    secret: str
    passphrase: Optional[str] = None
    sandbox: bool = False
    testnet: bool = False
    rate_limit: int = 1000
    timeout: int = 30000
    enable_rate_limit: bool = True
    
@dataclass
class TradingConfig:
    """交易配置"""
    max_position_size: float = 0.1
    max_daily_trades: int = 50
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_trade_amount: float = 10.0
    max_trade_amount: float = 1000.0
    allowed_symbols: List[str] = None
    
@dataclass
class AIConfig:
    """AI配置"""
    model_update_interval: int = 3600
    prediction_threshold: float = 0.7
    ensemble_weight_decay: float = 0.95
    max_models_loaded: int = 10
    batch_prediction_size: int = 100
    
@dataclass
class MonitoringConfig:
    """监控配置"""
    health_check_interval: int = 60
    performance_log_interval: int = 300
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class APIConfigManager:
    """API配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 配置文件路径
        self.exchange_config_file = self.config_dir / "exchanges.json"
        self.trading_config_file = self.config_dir / "trading.json"
        self.ai_config_file = self.config_dir / "ai.json"
        self.monitoring_config_file = self.config_dir / "monitoring.json"
        
        # 配置对象
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.trading_config = TradingConfig()
        self.ai_config = AIConfig()
        self.monitoring_config = MonitoringConfig()
        
        logger.info("🔧 API配置管理器初始化完成")
    
    @critical_section
    async def initialize(self):
        """初始化配置"""
        try:
            # 加载现有配置
            await self.load_all_configs()
            
            # 创建默认配置（如果不存在）
            await self.create_default_configs()
            
            # 验证配置
            await self.validate_configs()
            
            logger.info("✅ API配置初始化完成")
            
        except Exception as e:
            logger.error(f"❌ API配置初始化失败: {e}")
            raise
    
    async def load_all_configs(self):
        """加载所有配置"""
        try:
            # 加载交易所配置
            if self.exchange_config_file.exists():
                with open(self.exchange_config_file, 'r', encoding='utf-8') as f:
                    exchange_data = json.load(f)
                    for name, config in exchange_data.items():
                        self.exchanges[name] = ExchangeConfig(**config)
            
            # 加载交易配置
            if self.trading_config_file.exists():
                with open(self.trading_config_file, 'r', encoding='utf-8') as f:
                    trading_data = json.load(f)
                    self.trading_config = TradingConfig(**trading_data)
            
            # 加载AI配置
            if self.ai_config_file.exists():
                with open(self.ai_config_file, 'r', encoding='utf-8') as f:
                    ai_data = json.load(f)
                    self.ai_config = AIConfig(**ai_data)
            
            # 加载监控配置
            if self.monitoring_config_file.exists():
                with open(self.monitoring_config_file, 'r', encoding='utf-8') as f:
                    monitoring_data = json.load(f)
                    self.monitoring_config = MonitoringConfig(**monitoring_data)
            
            logger.info("📂 配置文件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise
    
    async def create_default_configs(self):
        """创建默认配置"""
        try:
            # 默认交易所配置
            if not self.exchanges:
                default_exchanges = {
                    "bitget": ExchangeConfig(
                        name="bitget",
                        api_key=os.getenv("BITGET_API_KEY", ""),
                        secret=os.getenv("BITGET_SECRET_KEY", ""),
                        passphrase=os.getenv("BITGET_PASSPHRASE", ""),
                        sandbox=False,  # 生产环境实盘交易
                        rate_limit=600
                    ),
                    "binance": ExchangeConfig(
                        name="binance",
                        api_key=os.getenv("BINANCE_API_KEY", ""),
                        secret=os.getenv("BINANCE_SECRET_KEY", ""),
                        sandbox=False,  # 生产环境实盘交易
                        rate_limit=1200
                    ),
                    "okx": ExchangeConfig(
                        name="okx",
                        api_key=os.getenv("OKX_API_KEY", ""),
                        secret=os.getenv("OKX_SECRET_KEY", ""),
                        passphrase=os.getenv("OKX_PASSPHRASE", ""),
                        sandbox=False,  # 生产环境实盘交易
                        rate_limit=600
                    )
                }
                self.exchanges = default_exchanges
                await self.save_exchange_configs()
            
            # 默认交易配置
            if not self.trading_config_file.exists():
                self.trading_config = TradingConfig(
                    allowed_symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
                )
                await self.save_trading_config()
            
            # 默认AI配置
            if not self.ai_config_file.exists():
                await self.save_ai_config()
            
            # 默认监控配置
            if not self.monitoring_config_file.exists():
                self.monitoring_config = MonitoringConfig(
                    alert_email=os.getenv("ALERT_EMAIL"),
                    slack_webhook=os.getenv("SLACK_WEBHOOK"),
                    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
                    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID")
                )
                await self.save_monitoring_config()
            
            logger.info("📝 默认配置创建完成")
            
        except Exception as e:
            logger.error(f"❌ 默认配置创建失败: {e}")
            raise
    
    async def validate_configs(self):
        """验证配置"""
        try:
            validation_results = {
                "exchanges": {},
                "trading": True,
                "ai": True,
                "monitoring": True
            }
            
            # 验证交易所配置
            for name, config in self.exchanges.items():
                try:
                    # 创建交易所实例测试连接
                    exchange_class = getattr(ccxt, config.name)
                    exchange = exchange_class({
                        'apiKey': config.api_key,
                        'secret': config.secret,
                        'password': config.passphrase,
                        'sandbox': config.sandbox,
                        'enableRateLimit': config.enable_rate_limit,
                        'timeout': config.timeout
                    })
                    
                    # 测试连接
                    if config.api_key and config.secret:
                        await asyncio.wait_for(
                            asyncio.to_thread(exchange.fetch_balance),
                            timeout=10
                        )
                        validation_results["exchanges"][name] = "connected"
                        logger.info(f"✅ {name} 连接成功")
                    else:
                        validation_results["exchanges"][name] = "no_credentials"
                        logger.warning(f"⚠️ {name} 缺少API凭证")
                    
                except asyncio.TimeoutError:
                    validation_results["exchanges"][name] = "timeout"
                    logger.warning(f"⚠️ {name} 连接超时")
                except Exception as e:
                    validation_results["exchanges"][name] = f"error: {str(e)}"
                    logger.warning(f"⚠️ {name} 连接失败: {e}")
            
            # 验证交易配置
            if self.trading_config.max_position_size <= 0:
                validation_results["trading"] = False
                logger.error("❌ 交易配置：最大仓位大小必须大于0")
            
            # 验证AI配置
            if self.ai_config.prediction_threshold < 0 or self.ai_config.prediction_threshold > 1:
                validation_results["ai"] = False
                logger.error("❌ AI配置：预测阈值必须在0-1之间")
            
            logger.info(f"🔍 配置验证完成: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")
            raise
    
    async def save_exchange_configs(self):
        """保存交易所配置"""
        try:
            exchange_data = {
                name: asdict(config) for name, config in self.exchanges.items()
            }
            with open(self.exchange_config_file, 'w', encoding='utf-8') as f:
                json.dump(exchange_data, f, ensure_ascii=False, indent=2)
            logger.info("💾 交易所配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存交易所配置失败: {e}")
            raise
    
    async def save_trading_config(self):
        """保存交易配置"""
        try:
            with open(self.trading_config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.trading_config), f, ensure_ascii=False, indent=2)
            logger.info("💾 交易配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存交易配置失败: {e}")
            raise
    
    async def save_ai_config(self):
        """保存AI配置"""
        try:
            with open(self.ai_config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.ai_config), f, ensure_ascii=False, indent=2)
            logger.info("💾 AI配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存AI配置失败: {e}")
            raise
    
    async def save_monitoring_config(self):
        """保存监控配置"""
        try:
            with open(self.monitoring_config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.monitoring_config), f, ensure_ascii=False, indent=2)
            logger.info("💾 监控配置已保存")
        except Exception as e:
            logger.error(f"❌ 保存监控配置失败: {e}")
            raise
    
    async def add_exchange(self, name: str, config: ExchangeConfig):
        """添加交易所"""
        try:
            self.exchanges[name] = config
            await self.save_exchange_configs()
            logger.info(f"➕ 添加交易所: {name}")
        except Exception as e:
            logger.error(f"❌ 添加交易所失败 {name}: {e}")
            raise
    
    async def remove_exchange(self, name: str):
        """移除交易所"""
        try:
            if name in self.exchanges:
                del self.exchanges[name]
                await self.save_exchange_configs()
                logger.info(f"➖ 移除交易所: {name}")
            else:
                logger.warning(f"⚠️ 交易所不存在: {name}")
        except Exception as e:
            logger.error(f"❌ 移除交易所失败 {name}: {e}")
            raise
    
    async def update_trading_config(self, **kwargs):
        """更新交易配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.trading_config, key):
                    setattr(self.trading_config, key, value)
            await self.save_trading_config()
            logger.info("🔄 交易配置已更新")
        except Exception as e:
            logger.error(f"❌ 更新交易配置失败: {e}")
            raise
    
    async def update_ai_config(self, **kwargs):
        """更新AI配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.ai_config, key):
                    setattr(self.ai_config, key, value)
            await self.save_ai_config()
            logger.info("🔄 AI配置已更新")
        except Exception as e:
            logger.error(f"❌ 更新AI配置失败: {e}")
            raise
    
    async def get_active_exchanges(self) -> List[str]:
        """获取活跃的交易所"""
        try:
            active_exchanges = []
            for name, config in self.exchanges.items():
                if config.api_key and config.secret:
                    active_exchanges.append(name)
            return active_exchanges
        except Exception as e:
            logger.error(f"❌ 获取活跃交易所失败: {e}")
            return []
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        try:
            return {
                "exchanges": {
                    name: {
                        "name": config.name,
                        "sandbox": config.sandbox,
                        "has_credentials": bool(config.api_key and config.secret),
                        "rate_limit": config.rate_limit
                    }
                    for name, config in self.exchanges.items()
                },
                "trading": {
                    "max_position_size": self.trading_config.max_position_size,
                    "max_daily_trades": self.trading_config.max_daily_trades,
                    "risk_per_trade": self.trading_config.risk_per_trade,
                    "allowed_symbols_count": len(self.trading_config.allowed_symbols or [])
                },
                "ai": {
                    "prediction_threshold": self.ai_config.prediction_threshold,
                    "max_models_loaded": self.ai_config.max_models_loaded,
                    "model_update_interval": self.ai_config.model_update_interval
                },
                "monitoring": {
                    "health_check_interval": self.monitoring_config.health_check_interval,
                    "has_email_alerts": bool(self.monitoring_config.alert_email),
                    "has_slack_alerts": bool(self.monitoring_config.slack_webhook),
                    "has_telegram_alerts": bool(self.monitoring_config.telegram_bot_token)
                }
            }
        except Exception as e:
            logger.error(f"❌ 获取配置摘要失败: {e}")
            return {}

# 全局配置管理器实例
api_config_manager = APIConfigManager()

# 导出主要组件
__all__ = [
    'APIConfigManager',
    'ExchangeConfig',
    'TradingConfig',
    'AIConfig',
    'MonitoringConfig',
    'api_config_manager'
]
