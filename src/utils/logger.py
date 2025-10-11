"""
日志配置模块
提供统一的日志记录功能
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
import colorlog


class Logger:
    """统一日志管理器"""
    
    _instances = {}
    
    def __new__(cls, name: str = "arbitrage"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str = "arbitrage"):
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self._initialized = True
    
    def _setup_logger(self):
        """设置日志配置"""
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if self.logger.handlers:
            return
        
        # 创建日志目录
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 控制台处理器 - 彩色输出
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器 - 轮转日志
        log_file = os.path.join(log_dir, f"{self.name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 错误日志处理器
        error_log_file = os.path.join(log_dir, f"{self.name}_error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（包含堆栈信息）"""
        self.logger.exception(message, **kwargs)


# 创建默认日志实例
default_logger = Logger("arbitrage")

# 便捷函数
def get_logger(name: str = "arbitrage") -> Logger:
    """获取日志实例"""
    return Logger(name)

def log_trade(exchange: str, symbol: str, side: str, amount: float, price: float, order_id: str = None):
    """记录交易日志"""
    trade_logger = get_logger("trade")
    message = f"[{exchange}] {side.upper()} {amount} {symbol} @ {price}"
    if order_id:
        message += f" (Order: {order_id})"
    trade_logger.info(message)

def log_pnl(strategy: str, pnl: float, total_pnl: float):
    """记录盈亏日志"""
    pnl_logger = get_logger("pnl")
    pnl_logger.info(f"[{strategy}] PnL: {pnl:.4f} USDT, Total: {total_pnl:.4f} USDT")

def log_error(module: str, error: Exception, context: Optional[str] = None):
    """记录错误日志"""
    error_logger = get_logger("error")
    message = f"[{module}] {type(error).__name__}: {str(error)}"
    if context:
        message += f" | Context: {context}"
    error_logger.exception(message)

def log_performance(operation: str, duration: float, success: bool = True):
    """记录性能日志"""
    perf_logger = get_logger("performance")
    status = "SUCCESS" if success else "FAILED"
    perf_logger.info(f"[{operation}] {status} - Duration: {duration:.3f}s")


if __name__ == "__main__":
    # 测试日志功能
    logger = get_logger("test")
    
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    # 测试交易日志
    log_trade("binance", "BTC/USDT", "buy", 0.001, 45000.0, "12345")
    
    # 测试盈亏日志
    log_pnl("funding_rate", 5.23, 125.67)
    
    # 测试性能日志
    log_performance("order_execution", 0.125, True)
