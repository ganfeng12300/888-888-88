#!/usr/bin/env python3
"""
🗑️ 硬盘自动清理系统 - Automatic Disk Cleanup System
1T硬盘空间管理，自动删除历史数据
"""
import os
import shutil
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import threading
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiskCleanupManager:
    """硬盘清理管理器"""
    
    def __init__(self, max_disk_usage_gb: float = 800):  # 1T硬盘，保留200GB空间
        self.max_disk_usage_gb = max_disk_usage_gb
        self.cleanup_rules = {
            # 数据类型: (保留天数, 文件扩展名)
            'logs': (7, ['.log', '.txt']),
            'market_data': (30, ['.csv', '.json']),
            'ai_models': (90, ['.joblib', '.pkl', '.h5']),
            'backups': (14, ['.bak', '.backup']),
            'temp_files': (1, ['.tmp', '.temp']),
            'cache': (3, ['.cache']),
            'screenshots': (7, ['.png', '.jpg', '.jpeg']),
            'reports': (30, ['.json', '.html', '.pdf'])
        }
        
        # 关键目录
        self.data_dirs = [
            'data',
            'logs',
            'models',
            'cache',
            'temp',
            'backups',
            'reports',
            'web/static/images'
        ]
        
        # 数据库文件清理规则
        self.db_cleanup_rules = {
            'market_data': 30,      # 市场数据保留30天
            'ai_decisions': 60,     # AI决策保留60天
            'trading_history': 180, # 交易历史保留180天
            'system_logs': 14       # 系统日志保留14天
        }
        
        self.running = False
        self.cleanup_thread = None
        
        logger.info("🗑️ 硬盘清理管理器初始化完成")
    
    def get_disk_usage(self, path: str = ".") -> Tuple[float, float, float]:
        """获取磁盘使用情况 (总空间GB, 已用空间GB, 可用空间GB)"""
        try:
            total, used, free = shutil.disk_usage(path)
            return (
                total / (1024**3),  # 转换为GB
                used / (1024**3),
                free / (1024**3)
            )
        except Exception as e:
            logger.error(f"获取磁盘使用情况失败: {e}")
            return (0, 0, 0)
    
    def get_directory_size(self, path: str) -> float:
        """获取目录大小 (GB)"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**3)
        except Exception as e:
            logger.error(f"获取目录大小失败 {path}: {e}")
            return 0
    
    def cleanup_old_files(self, directory: str, days_to_keep: int, extensions: List[str] = None) -> int:
        """清理旧文件"""
        if not os.path.exists(directory):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        deleted_size = 0
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # 检查文件扩展名
                    if extensions:
                        if not any(file.lower().endswith(ext) for ext in extensions):
                            continue
                    
                    try:
                        # 检查文件修改时间
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                        
                        if file_mtime < cutoff_date:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            deleted_count += 1
                            deleted_size += file_size
                            
                    except Exception as e:
                        logger.warning(f"删除文件失败 {filepath}: {e}")
            
            if deleted_count > 0:
                logger.info(f"🗑️ 清理目录 {directory}: 删除 {deleted_count} 个文件, 释放 {deleted_size/(1024**2):.2f} MB")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理目录失败 {directory}: {e}")
            return 0
    
    def cleanup_database_records(self, db_path: str) -> int:
        """清理数据库记录"""
        if not os.path.exists(db_path):
            return 0
        
        total_deleted = 0
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for table_name, days_to_keep in self.db_cleanup_rules.items():
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                try:
                    # 检查表是否存在
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                    if not cursor.fetchone():
                        continue
                    
                    # 删除旧记录
                    cursor.execute(f"DELETE FROM {table_name} WHERE timestamp < ?", (cutoff_date,))
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    
                    if deleted_count > 0:
                        logger.info(f"🗑️ 清理数据库表 {table_name}: 删除 {deleted_count} 条记录")
                        
                except Exception as e:
                    logger.warning(f"清理数据库表失败 {table_name}: {e}")
            
            conn.commit()
            
            # 优化数据库
            cursor.execute("VACUUM")
            conn.close()
            
            if total_deleted > 0:
                logger.info(f"🗑️ 数据库清理完成: 总共删除 {total_deleted} 条记录")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"数据库清理失败 {db_path}: {e}")
            return 0
    
    def cleanup_empty_directories(self, root_dir: str = ".") -> int:
        """清理空目录"""
        deleted_count = 0
        
        try:
            for root, dirs, files in os.walk(root_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # 目录为空
                            os.rmdir(dir_path)
                            deleted_count += 1
                            logger.info(f"🗑️ 删除空目录: {dir_path}")
                    except Exception as e:
                        logger.warning(f"删除空目录失败 {dir_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理空目录失败: {e}")
            return 0
    
    def emergency_cleanup(self) -> bool:
        """紧急清理 - 当磁盘空间不足时"""
        logger.warning("🚨 磁盘空间不足，执行紧急清理...")
        
        emergency_actions = [
            # (描述, 清理函数)
            ("清理临时文件", lambda: self.cleanup_old_files("temp", 0, ['.tmp', '.temp'])),
            ("清理缓存文件", lambda: self.cleanup_old_files("cache", 0)),
            ("清理7天前的日志", lambda: self.cleanup_old_files("logs", 7, ['.log'])),
            ("清理30天前的市场数据", lambda: self.cleanup_old_files("data", 30, ['.csv', '.json'])),
            ("清理旧的AI模型", lambda: self.cleanup_old_files("models", 30, ['.joblib', '.pkl'])),
            ("清理数据库记录", lambda: self.cleanup_database_records("data/trading.db")),
            ("清理空目录", lambda: self.cleanup_empty_directories())
        ]
        
        for description, cleanup_func in emergency_actions:
            try:
                logger.info(f"执行紧急清理: {description}")
                cleanup_func()
                
                # 检查是否有足够空间
                _, used_gb, free_gb = self.get_disk_usage()
                if free_gb > 50:  # 如果有超过50GB空间，停止紧急清理
                    logger.info("✅ 紧急清理完成，磁盘空间充足")
                    return True
                    
            except Exception as e:
                logger.error(f"紧急清理失败 {description}: {e}")
        
        return False
    
    def perform_routine_cleanup(self) -> Dict[str, int]:
        """执行常规清理"""
        logger.info("🧹 开始常规清理...")
        
        cleanup_results = {}
        
        # 1. 清理各类文件
        for data_type, (days_to_keep, extensions) in self.cleanup_rules.items():
            for data_dir in self.data_dirs:
                if os.path.exists(data_dir):
                    deleted_count = self.cleanup_old_files(data_dir, days_to_keep, extensions)
                    cleanup_results[f"{data_type}_{data_dir}"] = deleted_count
        
        # 2. 清理数据库
        db_files = [
            "data/trading.db",
            "data/hierarchical_ai.db",
            "data/market_data.db"
        ]
        
        for db_file in db_files:
            if os.path.exists(db_file):
                deleted_count = self.cleanup_database_records(db_file)
                cleanup_results[f"database_{os.path.basename(db_file)}"] = deleted_count
        
        # 3. 清理空目录
        empty_dirs = self.cleanup_empty_directories()
        cleanup_results["empty_directories"] = empty_dirs
        
        # 4. 生成清理报告
        total_operations = sum(cleanup_results.values())
        logger.info(f"✅ 常规清理完成: 总共执行 {total_operations} 次清理操作")
        
        return cleanup_results
    
    def check_disk_space_and_cleanup(self) -> bool:
        """检查磁盘空间并执行清理"""
        total_gb, used_gb, free_gb = self.get_disk_usage()
        
        logger.info(f"💾 磁盘使用情况: 总空间 {total_gb:.1f}GB, 已用 {used_gb:.1f}GB, 可用 {free_gb:.1f}GB")
        
        # 检查是否需要清理
        if used_gb > self.max_disk_usage_gb:
            logger.warning(f"⚠️ 磁盘使用超过限制 ({used_gb:.1f}GB > {self.max_disk_usage_gb}GB)")
            
            # 先执行常规清理
            self.perform_routine_cleanup()
            
            # 重新检查磁盘空间
            _, used_gb_after, free_gb_after = self.get_disk_usage()
            
            if used_gb_after > self.max_disk_usage_gb:
                # 如果常规清理不够，执行紧急清理
                return self.emergency_cleanup()
            else:
                logger.info("✅ 常规清理后磁盘空间充足")
                return True
        
        elif free_gb < 100:  # 可用空间少于100GB时预警
            logger.warning(f"⚠️ 磁盘可用空间不足 ({free_gb:.1f}GB < 100GB)")
            self.perform_routine_cleanup()
            return True
        
        else:
            logger.info("✅ 磁盘空间充足，无需清理")
            return True
    
    def get_cleanup_report(self) -> Dict[str, any]:
        """生成清理报告"""
        total_gb, used_gb, free_gb = self.get_disk_usage()
        
        # 获取各目录大小
        dir_sizes = {}
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                dir_sizes[data_dir] = self.get_directory_size(data_dir)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "disk_usage": {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "usage_percentage": round((used_gb / total_gb) * 100, 2)
            },
            "directory_sizes": {k: round(v, 2) for k, v in dir_sizes.items()},
            "cleanup_rules": self.cleanup_rules,
            "max_disk_usage_gb": self.max_disk_usage_gb,
            "status": "healthy" if used_gb < self.max_disk_usage_gb else "needs_cleanup"
        }
    
    def start_monitoring(self, check_interval_hours: int = 6):
        """启动磁盘监控"""
        self.running = True
        
        def monitoring_loop():
            while self.running:
                try:
                    self.check_disk_space_and_cleanup()
                    time.sleep(check_interval_hours * 3600)  # 转换为秒
                except Exception as e:
                    logger.error(f"磁盘监控错误: {e}")
                    time.sleep(300)  # 错误时等待5分钟
        
        self.cleanup_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"🚀 磁盘监控已启动，每 {check_interval_hours} 小时检查一次")
    
    def stop_monitoring(self):
        """停止磁盘监控"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("🛑 磁盘监控已停止")

# 全局实例
disk_cleanup_manager = DiskCleanupManager()

if __name__ == "__main__":
    # 测试清理系统
    print("🗑️ 硬盘清理系统测试")
    
    # 生成清理报告
    report = disk_cleanup_manager.get_cleanup_report()
    print(f"磁盘使用情况: {report['disk_usage']}")
    
    # 执行一次清理
    disk_cleanup_manager.check_disk_space_and_cleanup()
    
    # 启动监控
    disk_cleanup_manager.start_monitoring(check_interval_hours=1)  # 测试时1小时检查一次
    
    try:
        while True:
            time.sleep(60)  # 每分钟显示一次状态
            report = disk_cleanup_manager.get_cleanup_report()
            print(f"状态: {report['status']}, 已用空间: {report['disk_usage']['used_gb']:.1f}GB")
    except KeyboardInterrupt:
        disk_cleanup_manager.stop_monitoring()
        print("清理系统已停止")
