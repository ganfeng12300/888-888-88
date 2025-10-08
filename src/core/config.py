"""
🔧 系统配置管理
统一管理所有系统配置参数
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(PydanticBaseSettings):
    """数据库配置"""
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    clickhouse_url: str = Field(default="clickhouse://localhost:9000/quant_data", env="CLICKHOUSE_URL")
    postgres_url: str = Field(default="postgresql://localhost:5432/quant_system", env="POSTGRES_URL")
    
    # 连接池配置
    redis_max_connections: int = 100
    clickhouse_max_connections: int = 50
    postgres_max_connections: int = 20


class ExchangeSettings(PydanticBaseSettings):
    """交易所配置"""
    # 支持的交易所列表
    supported_exchanges: List[str] = [
        "binance", "okx", "bybit", "huobi", "kucoin", 
        "gate", "mexc", "bitget", "coinex"
    ]
    
    # 默认启用的交易所
    enabled_exchanges: List[str] = ["binance", "okx", "bybit"]
    
    # API配置 (从环境变量读取)
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    
    okx_api_key: Optional[str] = Field(default=None, env="OKX_API_KEY")
    okx_secret_key: Optional[str] = Field(default=None, env="OKX_SECRET_KEY")
    okx_passphrase: Optional[str] = Field(default=None, env="OKX_PASSPHRASE")
    
    bybit_api_key: Optional[str] = Field(default=None, env="BYBIT_API_KEY")
    bybit_secret_key: Optional[str] = Field(default=None, env="BYBIT_SECRET_KEY")
    
    # 交易配置
    default_leverage: int = 10
    max_leverage: int = 50
    min_order_size: float = 10.0  # USDT
    max_order_size: float = 10000.0  # USDT


class AISettings(PydanticBaseSettings):
    """AI模型配置"""
    # GPU配置
    cuda_device: str = Field(default="cuda:0", env="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = 0.8
    
    # 模型配置
    model_update_interval: int = 3600  # 秒
    model_retrain_interval: int = 86400  # 秒
    
    # 各AI模型权重
    meta_learning_weight: float = 0.15
    ensemble_learning_weight: float = 0.20
    reinforcement_learning_weight: float = 0.15
    time_series_weight: float = 0.20
    transfer_learning_weight: float = 0.10
    expert_system_weight: float = 0.10
    gan_weight: float = 0.05
    graph_neural_weight: float = 0.05
    
    @validator('*_weight')
    def validate_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('权重必须在0-1之间')
        return v


class RiskSettings(PydanticBaseSettings):
    """风险控制配置"""
    # 收益目标
    weekly_profit_target: float = 0.20  # 20%
    
    # 风险限制
    max_daily_drawdown: float = 0.03  # 3%
    max_position_size: float = 0.10  # 单个仓位最大10%
    max_total_exposure: float = 0.80  # 总敞口最大80%
    
    # 止损配置
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.05  # 5%
    
    # 风险监控
    risk_check_interval: int = 5  # 秒
    emergency_stop_threshold: float = 0.05  # 5%紧急止损


class TradingSettings(PydanticBaseSettings):
    """交易配置"""
    # 交易模式
    trading_mode: str = "live"  # live, paper, backtest
    
    # 交易时间
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    
    # 交易品种
    trading_symbols: List[str] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
        "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT"
    ]
    
    # 订单配置
    order_timeout: int = 30  # 秒
    max_retry_attempts: int = 3
    slippage_tolerance: float = 0.001  # 0.1%


class WebSettings(PydanticBaseSettings):
    """Web界面配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # WebSocket配置
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    
    # 静态文件
    static_files_path: str = "web/build"
    
    # CORS配置
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


class HardwareSettings(PydanticBaseSettings):
    """硬件配置"""
    # CPU配置
    cpu_cores: int = 20
    max_cpu_usage: float = 0.80  # 80%
    
    # GPU配置
    gpu_name: str = "RTX 3060"
    gpu_memory_gb: int = 12
    max_gpu_usage: float = 0.90  # 90%
    
    # 内存配置
    memory_gb: int = 128
    max_memory_usage: float = 0.85  # 85%
    
    # 存储配置
    storage_path: str = "/app/data"
    max_storage_usage: float = 0.90  # 90%


class LoggingSettings(PydanticBaseSettings):
    """日志配置"""
    log_level: str = "INFO"
    log_file: str = "logs/quant_system.log"
    log_rotation: str = "100 MB"
    log_retention: str = "30 days"
    
    # 日志格式
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"


class Settings(PydanticBaseSettings):
    """主配置类"""
    
    # 基本信息
    app_name: str = "AI量化交易系统"
    app_version: str = "1.0.0"
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # 时区配置
    timezone: str = "Asia/Shanghai"
    
    # 各模块配置
    database: DatabaseSettings = DatabaseSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    ai: AISettings = AISettings()
    risk: RiskSettings = RiskSettings()
    trading: TradingSettings = TradingSettings()
    web: WebSettings = WebSettings()
    hardware: HardwareSettings = HardwareSettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """获取指定交易所的配置"""
        exchange_configs = {
            "binance": {
                "api_key": self.exchange.binance_api_key,
                "secret": self.exchange.binance_secret_key,
                "sandbox": False,  # 强制生产环境
                "enableRateLimit": True,
            },
            "okx": {
                "api_key": self.exchange.okx_api_key,
                "secret": self.exchange.okx_secret_key,
                "password": self.exchange.okx_passphrase,
                "sandbox": False,  # 强制生产环境
                "enableRateLimit": True,
            },
            "bybit": {
                "api_key": self.exchange.bybit_api_key,
                "secret": self.exchange.bybit_secret_key,
                "testnet": False,  # 强制生产环境
                "enableRateLimit": True,
            }
        }
        
        return exchange_configs.get(exchange_name, {})
    
    def validate_ai_weights(self) -> bool:
        """验证AI模型权重总和"""
        total_weight = (
            self.ai.meta_learning_weight +
            self.ai.ensemble_learning_weight +
            self.ai.reinforcement_learning_weight +
            self.ai.time_series_weight +
            self.ai.transfer_learning_weight +
            self.ai.expert_system_weight +
            self.ai.gan_weight +
            self.ai.graph_neural_weight
        )
        
        return abs(total_weight - 1.0) < 0.001  # 允许小的浮点误差
    
    def get_model_weights(self) -> Dict[str, float]:
        """获取AI模型权重字典"""
        return {
            "meta_learning": self.ai.meta_learning_weight,
            "ensemble_learning": self.ai.ensemble_learning_weight,
            "reinforcement_learning": self.ai.reinforcement_learning_weight,
            "time_series": self.ai.time_series_weight,
            "transfer_learning": self.ai.transfer_learning_weight,
            "expert_system": self.ai.expert_system_weight,
            "gan": self.ai.gan_weight,
            "graph_neural": self.ai.graph_neural_weight,
        }


# 全局配置实例
settings = Settings()

# 验证配置
if not settings.validate_ai_weights():
    raise ValueError("AI模型权重总和必须等于1.0")
