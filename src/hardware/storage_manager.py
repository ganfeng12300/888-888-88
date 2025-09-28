"""
💾 1TB NVMe智能存储管理器
生产级存储资源管理、自动清理和数据生命周期管理系统
实现智能分区、压缩存储和性能优化
"""

import asyncio
import os
import shutil
import time
import threading
import schedule
import lz4.frame
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import psutil
import subprocess

from loguru import logger


class DataType(Enum):
    """数据类型"""
    TICK_DATA = "tick_data"
    KLINE_DATA = "kline_data"
    ORDER_BOOK = "order_book"
    AI_MODEL = "ai_model"
    MODEL_CHECKPOINT = "model_checkpoint"
    SYSTEM_LOG = "system_log"
    TRADE_LOG = "trade_log"
    ERROR_LOG = "error_log"
    TEMP_FILE = "temp_file"
    CACHE_DATA = "cache_data"


@dataclass
class StoragePartition:
    """存储分区配置"""
    name: str
    path: str
    size_gb: float
    usage_limit: float  # 使用率限制 (0-1)
    description: str
    auto_cleanup: bool = True
    compression_enabled: bool = True


@dataclass
class DataLifecycleRule:
    """数据生命周期规则"""
    data_type: DataType
    hot_days: int  # 热数据保留天数
    warm_days: int  # 温数据保留天数
    cold_days: int  # 冷数据保留天数
    archive_days: int  # 归档后删除天数
    compression_after_days: int  # 多少天后压缩
    sampling_ratio: float = 1.0  # 采样比例


@dataclass
class StorageMetrics:
    """存储性能指标"""
    timestamp: float
    total_size_gb: float
    used_size_gb: float
    free_size_gb: float
    usage_percent: float
    read_speed_mb_s: float
    write_speed_mb_s: float
    iops_read: int
    iops_write: int
    partition_usage: Dict[str, float]


class IntelligentStorageManager:
    """智能存储管理器"""
    
    def __init__(self, storage_path: str = "/app/data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 存储分区配置 (基于1TB NVMe)
        self.partitions = {
            "system": StoragePartition(
                name="system",
                path=str(self.storage_path / "system"),
                size_gb=200.0,
                usage_limit=0.90,
                description="系统和程序文件",
                auto_cleanup=False,
                compression_enabled=False
            ),
            "realtime": StoragePartition(
                name="realtime",
                path=str(self.storage_path / "realtime"),
                size_gb=300.0,
                usage_limit=0.85,
                description="实时数据缓存(7天滚动)",
                auto_cleanup=True,
                compression_enabled=True
            ),
            "models": StoragePartition(
                name="models",
                path=str(self.storage_path / "models"),
                size_gb=200.0,
                usage_limit=0.80,
                description="AI模型和检查点",
                auto_cleanup=True,
                compression_enabled=True
            ),
            "historical": StoragePartition(
                name="historical",
                path=str(self.storage_path / "historical"),
                size_gb=150.0,
                usage_limit=0.85,
                description="历史数据(压缩存储30天)",
                auto_cleanup=True,
                compression_enabled=True
            ),
            "logs": StoragePartition(
                name="logs",
                path=str(self.storage_path / "logs"),
                size_gb=100.0,
                usage_limit=0.80,
                description="日志和监控数据(15天)",
                auto_cleanup=True,
                compression_enabled=True
            ),
            "temp": StoragePartition(
                name="temp",
                path=str(self.storage_path / "temp"),
                size_gb=50.0,
                usage_limit=0.70,
                description="临时文件和缓冲区",
                auto_cleanup=True,
                compression_enabled=False
            ),
        }
        
        # 数据生命周期规则
        self.lifecycle_rules = {
            DataType.TICK_DATA: DataLifecycleRule(
                data_type=DataType.TICK_DATA,
                hot_days=1,
                warm_days=7,
                cold_days=30,
                archive_days=90,
                compression_after_days=3,
                sampling_ratio=0.01  # 保留1%的tick数据
            ),
            DataType.KLINE_DATA: DataLifecycleRule(
                data_type=DataType.KLINE_DATA,
                hot_days=7,
                warm_days=30,
                cold_days=90,
                archive_days=365,
                compression_after_days=7,
                sampling_ratio=1.0  # 保留所有K线数据
            ),
            DataType.AI_MODEL: DataLifecycleRule(
                data_type=DataType.AI_MODEL,
                hot_days=30,
                warm_days=90,
                cold_days=180,
                archive_days=365,
                compression_after_days=30,
                sampling_ratio=1.0
            ),
            DataType.MODEL_CHECKPOINT: DataLifecycleRule(
                data_type=DataType.MODEL_CHECKPOINT,
                hot_days=1,
                warm_days=7,
                cold_days=30,
                archive_days=90,
                compression_after_days=1,
                sampling_ratio=0.1  # 只保留10%的检查点
            ),
            DataType.SYSTEM_LOG: DataLifecycleRule(
                data_type=DataType.SYSTEM_LOG,
                hot_days=7,
                warm_days=15,
                cold_days=30,
                archive_days=90,
                compression_after_days=7,
                sampling_ratio=1.0
            ),
            DataType.TRADE_LOG: DataLifecycleRule(
                data_type=DataType.TRADE_LOG,
                hot_days=30,
                warm_days=90,
                cold_days=180,
                archive_days=365,
                compression_after_days=30,
                sampling_ratio=1.0
            ),
        }
        
        # 初始化分区
        self._initialize_partitions()
        
        # 性能监控
        self.metrics_history: List[StorageMetrics] = []
        self.max_history_size = 3600  # 1小时历史数据
        self.monitoring = False
        
        # 清理调度器
        self.cleanup_scheduler_running = False
        self.cleanup_lock = threading.Lock()
        
        # 数据库连接 (用于元数据管理)
        self.db_path = self.storage_path / "metadata.db"
        self._initialize_database()
        
        logger.info("智能存储管理器初始化完成")
    
    def _initialize_partitions(self):
        """初始化存储分区"""
        for partition in self.partitions.values():
            Path(partition.path).mkdir(parents=True, exist_ok=True)
            logger.info(f"初始化分区: {partition.name} -> {partition.path}")
    
    def _initialize_database(self):
        """初始化元数据数据库"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 创建文件元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    data_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_time REAL NOT NULL,
                    last_access_time REAL NOT NULL,
                    compressed BOOLEAN DEFAULT FALSE,
                    partition_name TEXT NOT NULL,
                    importance_score REAL DEFAULT 1.0
                )
            ''')
            
            # 创建清理历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cleanup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cleanup_time REAL NOT NULL,
                    files_deleted INTEGER NOT NULL,
                    space_freed_mb REAL NOT NULL,
                    cleanup_type TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("元数据数据库初始化完成")
            
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
    
    def store_file(self, file_path: str, data_type: DataType, 
                   partition_name: str = None, importance_score: float = 1.0) -> bool:
        """存储文件并记录元数据"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"源文件不存在: {file_path}")
                return False
            
            # 确定目标分区
            if partition_name is None:
                partition_name = self._determine_partition(data_type)
            
            if partition_name not in self.partitions:
                logger.error(f"未知分区: {partition_name}")
                return False
            
            partition = self.partitions[partition_name]
            
            # 检查分区空间
            if not self._check_partition_space(partition_name, source_path.stat().st_size):
                logger.warning(f"分区 {partition_name} 空间不足")
                # 尝试清理空间
                self._cleanup_partition(partition_name)
                
                # 再次检查
                if not self._check_partition_space(partition_name, source_path.stat().st_size):
                    logger.error(f"分区 {partition_name} 空间不足，无法存储文件")
                    return False
            
            # 生成目标路径
            target_dir = Path(partition.path) / data_type.value
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            
            # 记录元数据
            self._record_file_metadata(
                str(target_path), data_type, target_path.stat().st_size,
                partition_name, importance_score
            )
            
            logger.info(f"文件已存储: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"存储文件失败: {e}")
            return False
    
    def _determine_partition(self, data_type: DataType) -> str:
        """根据数据类型确定分区"""
        partition_mapping = {
            DataType.TICK_DATA: "realtime",
            DataType.KLINE_DATA: "realtime",
            DataType.ORDER_BOOK: "realtime",
            DataType.AI_MODEL: "models",
            DataType.MODEL_CHECKPOINT: "models",
            DataType.SYSTEM_LOG: "logs",
            DataType.TRADE_LOG: "logs",
            DataType.ERROR_LOG: "logs",
            DataType.TEMP_FILE: "temp",
            DataType.CACHE_DATA: "temp",
        }
        return partition_mapping.get(data_type, "temp")
    
    def _check_partition_space(self, partition_name: str, required_bytes: int) -> bool:
        """检查分区是否有足够空间"""
        partition = self.partitions[partition_name]
        partition_path = Path(partition.path)
        
        if not partition_path.exists():
            return True
        
        # 获取分区使用情况
        usage = shutil.disk_usage(partition_path)
        current_usage = (usage.total - usage.free) / (partition.size_gb * 1024**3)
        
        # 计算添加文件后的使用率
        new_usage = (usage.total - usage.free + required_bytes) / (partition.size_gb * 1024**3)
        
        return new_usage <= partition.usage_limit
    
    def _record_file_metadata(self, file_path: str, data_type: DataType, 
                             size_bytes: int, partition_name: str, importance_score: float):
        """记录文件元数据"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            current_time = time.time()
            
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (file_path, data_type, size_bytes, created_time, last_access_time, 
                 partition_name, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, data_type.value, size_bytes, current_time, 
                  current_time, partition_name, importance_score))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"记录文件元数据失败: {e}")
    
    async def start_performance_monitoring(self, interval: float = 10.0):
        """启动存储性能监控"""
        self.monitoring = True
        logger.info("开始存储性能监控...")
        
        while self.monitoring:
            try:
                metrics = await self._collect_storage_metrics()
                self._store_metrics(metrics)
                
                # 检查存储问题
                await self._check_storage_issues(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"存储性能监控出错: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_storage_metrics(self) -> StorageMetrics:
        """收集存储性能指标"""
        timestamp = time.time()
        
        # 获取总体存储使用情况
        usage = shutil.disk_usage(self.storage_path)
        total_size_gb = usage.total / (1024**3)
        used_size_gb = (usage.total - usage.free) / (1024**3)
        free_size_gb = usage.free / (1024**3)
        usage_percent = (used_size_gb / total_size_gb) * 100
        
        # 获取I/O性能指标
        disk_io = psutil.disk_io_counters()
        if hasattr(self, '_last_disk_io'):
            time_delta = timestamp - self._last_io_timestamp
            read_speed_mb_s = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2) / time_delta
            write_speed_mb_s = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2) / time_delta
            iops_read = int((disk_io.read_count - self._last_disk_io.read_count) / time_delta)
            iops_write = int((disk_io.write_count - self._last_disk_io.write_count) / time_delta)
        else:
            read_speed_mb_s = 0
            write_speed_mb_s = 0
            iops_read = 0
            iops_write = 0
        
        self._last_disk_io = disk_io
        self._last_io_timestamp = timestamp
        
        # 获取各分区使用情况
        partition_usage = {}
        for name, partition in self.partitions.items():
            try:
                partition_path = Path(partition.path)
                if partition_path.exists():
                    partition_usage_info = shutil.disk_usage(partition_path)
                    partition_used = (partition_usage_info.total - partition_usage_info.free) / (1024**3)
                    partition_usage[name] = (partition_used / partition.size_gb) * 100
                else:
                    partition_usage[name] = 0.0
            except:
                partition_usage[name] = 0.0
        
        return StorageMetrics(
            timestamp=timestamp,
            total_size_gb=total_size_gb,
            used_size_gb=used_size_gb,
            free_size_gb=free_size_gb,
            usage_percent=usage_percent,
            read_speed_mb_s=read_speed_mb_s,
            write_speed_mb_s=write_speed_mb_s,
            iops_read=iops_read,
            iops_write=iops_write,
            partition_usage=partition_usage
        )
    
    def _store_metrics(self, metrics: StorageMetrics):
        """存储性能指标"""
        self.metrics_history.append(metrics)
        
        # 限制历史数据大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    async def _check_storage_issues(self, metrics: StorageMetrics):
        """检查存储问题"""
        issues = []
        
        # 检查总体使用率
        if metrics.usage_percent > 85:
            issues.append(f"存储使用率过高: {metrics.usage_percent:.1f}%")
        
        # 检查各分区使用率
        for partition_name, usage in metrics.partition_usage.items():
            partition = self.partitions[partition_name]
            if usage > partition.usage_limit * 100:
                issues.append(f"分区 {partition_name} 使用率过高: {usage:.1f}%")
        
        # 检查I/O性能
        if metrics.read_speed_mb_s < 100 and metrics.read_speed_mb_s > 0:
            issues.append(f"读取速度较慢: {metrics.read_speed_mb_s:.1f}MB/s")
        
        if metrics.write_speed_mb_s < 100 and metrics.write_speed_mb_s > 0:
            issues.append(f"写入速度较慢: {metrics.write_speed_mb_s:.1f}MB/s")
        
        if issues:
            logger.warning(f"存储问题: {'; '.join(issues)}")
            # 触发清理
            await self._trigger_emergency_cleanup()
    
    async def _trigger_emergency_cleanup(self):
        """触发紧急清理"""
        logger.info("触发紧急存储清理...")
        
        # 清理临时文件
        self._cleanup_partition("temp")
        
        # 清理过期数据
        await self._cleanup_expired_data()
        
        # 压缩数据
        await self._compress_old_data()
    
    def start_cleanup_scheduler(self):
        """启动清理调度器"""
        if self.cleanup_scheduler_running:
            return
        
        self.cleanup_scheduler_running = True
        
        # 每小时清理
        schedule.every().hour.do(self._hourly_cleanup)
        
        # 每天清理
        schedule.every().day.at("02:00").do(self._daily_cleanup)
        
        # 每周清理
        schedule.every().sunday.at("03:00").do(self._weekly_cleanup)
        
        # 启动调度器线程
        def run_scheduler():
            while self.cleanup_scheduler_running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("清理调度器已启动")
    
    def _hourly_cleanup(self):
        """每小时清理任务"""
        with self.cleanup_lock:
            logger.info("执行每小时清理任务...")
            
            # 清理临时文件
            self._cleanup_partition("temp")
            
            # 检查并清理过期缓存
            self._cleanup_expired_cache()
    
    def _daily_cleanup(self):
        """每日清理任务"""
        with self.cleanup_lock:
            logger.info("执行每日清理任务...")
            
            # 压缩昨天的数据
            asyncio.run(self._compress_yesterday_data())
            
            # 清理过期模型检查点
            self._cleanup_old_model_checkpoints()
            
            # 清理过期日志
            self._cleanup_old_logs()
    
    def _weekly_cleanup(self):
        """每周清理任务"""
        with self.cleanup_lock:
            logger.info("执行每周清理任务...")
            
            # 深度数据压缩
            asyncio.run(self._deep_data_compression())
            
            # 清理低价值历史数据
            self._cleanup_low_value_data()
            
            # 数据库维护
            self._maintain_database()
    
    def _cleanup_partition(self, partition_name: str) -> int:
        """清理指定分区"""
        try:
            partition = self.partitions[partition_name]
            partition_path = Path(partition.path)
            
            if not partition_path.exists():
                return 0
            
            files_deleted = 0
            space_freed = 0
            
            # 获取分区中的所有文件
            for file_path in partition_path.rglob("*"):
                if file_path.is_file():
                    # 检查文件是否应该被清理
                    if self._should_cleanup_file(file_path):
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_deleted += 1
                        space_freed += file_size
            
            if files_deleted > 0:
                logger.info(f"分区 {partition_name} 清理完成: 删除 {files_deleted} 个文件，释放 {space_freed / (1024**2):.1f}MB")
            
            return files_deleted
            
        except Exception as e:
            logger.error(f"清理分区 {partition_name} 失败: {e}")
            return 0
    
    def _should_cleanup_file(self, file_path: Path) -> bool:
        """判断文件是否应该被清理"""
        try:
            # 获取文件信息
            stat = file_path.stat()
            file_age_days = (time.time() - stat.st_mtime) / 86400
            
            # 根据文件类型和年龄判断
            if file_path.suffix in ['.tmp', '.temp', '.cache']:
                return file_age_days > 1  # 临时文件1天后删除
            
            if 'checkpoint' in file_path.name.lower():
                return file_age_days > 7  # 检查点7天后删除
            
            if file_path.suffix in ['.log']:
                return file_age_days > 15  # 日志文件15天后删除
            
            return False
            
        except Exception:
            return False
    
    async def _compress_old_data(self):
        """压缩旧数据"""
        try:
            logger.info("开始压缩旧数据...")
            
            compressed_count = 0
            space_saved = 0
            
            # 遍历所有分区
            for partition_name, partition in self.partitions.items():
                if not partition.compression_enabled:
                    continue
                
                partition_path = Path(partition.path)
                if not partition_path.exists():
                    continue
                
                # 查找需要压缩的文件
                for file_path in partition_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.endswith('.lz4'):
                        # 检查文件年龄
                        file_age_days = (time.time() - file_path.stat().st_mtime) / 86400
                        
                        if file_age_days > 3:  # 3天后压缩
                            original_size = file_path.stat().st_size
                            if await self._compress_file(file_path):
                                compressed_size = file_path.with_suffix(file_path.suffix + '.lz4').stat().st_size
                                space_saved += original_size - compressed_size
                                compressed_count += 1
            
            if compressed_count > 0:
                logger.info(f"数据压缩完成: 压缩 {compressed_count} 个文件，节省 {space_saved / (1024**2):.1f}MB")
            
        except Exception as e:
            logger.error(f"压缩旧数据失败: {e}")
    
    async def _compress_file(self, file_path: Path) -> bool:
        """压缩单个文件"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.lz4')
            
            with open(file_path, 'rb') as f_in:
                data = f_in.read()
            
            compressed_data = lz4.frame.compress(data)
            
            with open(compressed_path, 'wb') as f_out:
                f_out.write(compressed_data)
            
            # 删除原文件
            file_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"压缩文件 {file_path} 失败: {e}")
            return False
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """获取存储摘要"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "total_size_gb": latest_metrics.total_size_gb,
            "used_size_gb": latest_metrics.used_size_gb,
            "free_size_gb": latest_metrics.free_size_gb,
            "usage_percent": latest_metrics.usage_percent,
            "partition_usage": latest_metrics.partition_usage,
            "partitions": {
                name: {
                    "size_gb": partition.size_gb,
                    "usage_limit": partition.usage_limit * 100,
                    "description": partition.description,
                    "auto_cleanup": partition.auto_cleanup,
                    "compression_enabled": partition.compression_enabled
                }
                for name, partition in self.partitions.items()
            }
        }
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.cleanup_scheduler_running = False
        logger.info("存储监控已停止")


# 全局存储管理器实例
storage_manager = IntelligentStorageManager()


async def main():
    """测试主函数"""
    logger.info("启动存储管理器测试...")
    
    # 启动清理调度器
    storage_manager.start_cleanup_scheduler()
    
    # 启动性能监控
    monitor_task = asyncio.create_task(storage_manager.start_performance_monitoring())
    
    try:
        # 运行30秒测试
        await asyncio.sleep(30)
        
        # 获取存储摘要
        summary = storage_manager.get_storage_summary()
        logger.info(f"存储摘要: {json.dumps(summary, indent=2)}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号...")
    finally:
        storage_manager.stop_monitoring()
        monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
