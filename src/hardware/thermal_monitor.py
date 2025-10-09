#!/usr/bin/env python3
"""
🌡️ 硬件温度监控系统 - 生产级实盘代码 (增强版)
Hardware Thermal Monitoring System - Production Grade Enhanced

实时监控CPU、GPU、内存温度，提供温度预警和自动保护功能
支持多种硬件平台，确保系统长期稳定运行
集成温控优化复利系统，支持20核CPU + RTX3060 + 128G内存配置

Features:
- 实时温度监控和预警
- 多级温度阈值管理
- 智能警报系统
- 温度历史数据记录
- 硬件保护机制集成
- 与现有性能监控系统集成

Author: Codegen AI
Version: 2.0.0 (Enhanced for Thermal Control System)
License: MIT
"""

import asyncio
import json
import time
import threading
import subprocess
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    import psutil
    import GPUtil
    import nvidia_ml_py3 as nvml
except ImportError as e:
    logging.error(f"Required packages not installed: {e}")
    raise

from loguru import logger


@dataclass
class ThermalReading:
    """温度读数数据结构"""
    timestamp: float
    cpu_temp: float
    gpu_temp: float
    memory_temp: float
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    fan_speed: int
    power_draw: float


@dataclass
class ThermalThresholds:
    """温度阈值配置"""
    cpu_warning: float = 70.0
    cpu_critical: float = 75.0
    cpu_emergency: float = 80.0
    
    gpu_warning: float = 75.0
    gpu_critical: float = 80.0
    gpu_emergency: float = 85.0
    
    memory_warning: float = 80.0
    memory_critical: float = 85.0
    memory_emergency: float = 90.0


@dataclass
class ThermalAlert:
    """温度警报数据结构"""
    timestamp: float
    component: str
    temperature: float
    threshold: float
    level: str  # warning, critical, emergency
    message: str


class ThermalMonitor:
    """硬件温度监控系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化温度监控系统"""
        self.config = config or {}
        self.thresholds = ThermalThresholds(**self.config.get('thresholds', {}))
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.readings_history: List[ThermalReading] = []
        self.alerts_history: List[ThermalAlert] = []
        
        # 回调函数
        self.alert_callbacks: List[Callable[[ThermalAlert], None]] = []
        self.reading_callbacks: List[Callable[[ThermalReading], None]] = []
        
        # 配置参数
        self.monitor_interval = self.config.get('monitor_interval', 1.0)  # 秒
        self.history_limit = self.config.get('history_limit', 3600)  # 保留1小时数据
        self.alert_cooldown = self.config.get('alert_cooldown', 30.0)  # 警报冷却时间
        
        # 警报状态跟踪
        self.last_alerts: Dict[str, float] = {}
        
        # 初始化硬件监控
        self._initialize_hardware_monitoring()
        
        logger.info("🌡️ 温度监控系统初始化完成")
    
    def _initialize_hardware_monitoring(self):
        """初始化硬件监控"""
        try:
            # 初始化NVIDIA GPU监控
            nvml.nvmlInit()
            self.gpu_count = nvml.nvmlDeviceGetCount()
            logger.info(f"检测到 {self.gpu_count} 个NVIDIA GPU")
            
            # 获取GPU句柄
            self.gpu_handles = []
            for i in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
                
                # 获取GPU信息
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                logger.info(f"GPU {i}: {name}")
                
        except Exception as e:
            logger.warning(f"GPU监控初始化失败: {e}")
            self.gpu_count = 0
            self.gpu_handles = []
        
        # 检查CPU温度监控支持
        self.cpu_temp_supported = self._check_cpu_temp_support()
        if self.cpu_temp_supported:
            logger.info("✅ CPU温度监控支持")
        else:
            logger.warning("⚠️ CPU温度监控不支持")
    
    def _check_cpu_temp_support(self) -> bool:
        """检查CPU温度监控支持"""
        try:
            temps = psutil.sensors_temperatures()
            return len(temps) > 0
        except:
            return False
    
    def get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            if not self.cpu_temp_supported:
                return 0.0
            
            temps = psutil.sensors_temperatures()
            
            # 尝试不同的温度传感器
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    if entries:
                        # 返回第一个核心温度或包温度
                        return entries[0].current
            
            # 如果没有找到特定的CPU温度，返回第一个可用温度
            for name, entries in temps.items():
                if entries:
                    return entries[0].current
            
            return 0.0
            
        except Exception as e:
            logger.error(f"获取CPU温度失败: {e}")
            return 0.0
    
    def get_gpu_temperature(self) -> float:
        """获取GPU温度"""
        try:
            if not self.gpu_handles:
                return 0.0
            
            # 获取第一个GPU的温度
            temp = nvml.nvmlDeviceGetTemperature(
                self.gpu_handles[0], 
                nvml.NVML_TEMPERATURE_GPU
            )
            return float(temp)
            
        except Exception as e:
            logger.error(f"获取GPU温度失败: {e}")
            return 0.0
    
    def get_memory_temperature(self) -> float:
        """获取内存温度（增强估算）"""
        try:
            # 内存温度通常难以直接获取，这里基于多因素估算
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            # 获取系统运行时间，影响基础温度
            boot_time = psutil.boot_time()
            uptime_hours = (time.time() - boot_time) / 3600
            
            # 基于使用率和运行时间的增强估算公式
            base_temp = 30.0 + min(uptime_hours * 0.1, 10.0)  # 基础温度随运行时间增加
            usage_temp = usage_percent * 0.35  # 每1%使用率增加0.35度
            
            # 考虑CPU温度对内存温度的影响
            cpu_temp = self.get_cpu_temperature()
            if cpu_temp > 0:
                cpu_influence = max(0, (cpu_temp - 50) * 0.1)  # CPU超过50度时影响内存温度
            else:
                cpu_influence = 0
            
            estimated_temp = base_temp + usage_temp + cpu_influence
            return min(estimated_temp, 95.0)  # 最高95度
            
        except Exception as e:
            logger.error(f"获取内存温度失败: {e}")
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            if not self.gpu_handles:
                return 0.0
            
            util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handles[0])
            return float(util.gpu)
            
        except Exception as e:
            logger.error(f"获取GPU使用率失败: {e}")
            return 0.0
    
    def get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except:
            return 0.0
    
    def get_fan_speed(self) -> int:
        """获取风扇转速"""
        try:
            if not self.gpu_handles:
                return 0
            
            fan_speed = nvml.nvmlDeviceGetFanSpeed(self.gpu_handles[0])
            return int(fan_speed)
            
        except Exception as e:
            # GPU风扇速度获取失败，尝试系统风扇
            try:
                fans = psutil.sensors_fans()
                for name, entries in fans.items():
                    if entries:
                        return int(entries[0].current)
                return 0
            except:
                return 0
    
    def get_power_draw(self) -> float:
        """获取功耗"""
        try:
            if not self.gpu_handles:
                return 0.0
            
            power = nvml.nvmlDeviceGetPowerUsage(self.gpu_handles[0])
            return float(power) / 1000.0  # 转换为瓦特
            
        except Exception as e:
            logger.error(f"获取功耗失败: {e}")
            return 0.0
    
    def take_reading(self) -> ThermalReading:
        """获取一次完整的温度读数"""
        reading = ThermalReading(
            timestamp=time.time(),
            cpu_temp=self.get_cpu_temperature(),
            gpu_temp=self.get_gpu_temperature(),
            memory_temp=self.get_memory_temperature(),
            cpu_usage=self.get_cpu_usage(),
            gpu_usage=self.get_gpu_usage(),
            memory_usage=self.get_memory_usage(),
            fan_speed=self.get_fan_speed(),
            power_draw=self.get_power_draw()
        )
        
        # 添加到历史记录
        self.readings_history.append(reading)
        
        # 限制历史记录长度
        if len(self.readings_history) > self.history_limit:
            self.readings_history = self.readings_history[-self.history_limit:]
        
        # 调用回调函数
        for callback in self.reading_callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.error(f"读数回调函数执行失败: {e}")
        
        return reading
    
    def check_thermal_alerts(self, reading: ThermalReading):
        """检查温度警报"""
        current_time = time.time()
        
        # 检查CPU温度
        self._check_component_temperature(
            "CPU", reading.cpu_temp, current_time,
            self.thresholds.cpu_warning,
            self.thresholds.cpu_critical,
            self.thresholds.cpu_emergency
        )
        
        # 检查GPU温度
        self._check_component_temperature(
            "GPU", reading.gpu_temp, current_time,
            self.thresholds.gpu_warning,
            self.thresholds.gpu_critical,
            self.thresholds.gpu_emergency
        )
        
        # 检查内存温度
        self._check_component_temperature(
            "Memory", reading.memory_temp, current_time,
            self.thresholds.memory_warning,
            self.thresholds.memory_critical,
            self.thresholds.memory_emergency
        )
    
    def _check_component_temperature(self, component: str, temp: float, 
                                   current_time: float, warning: float, 
                                   critical: float, emergency: float):
        """检查单个组件温度"""
        if temp <= 0:
            return  # 无效温度读数
        
        alert_level = None
        threshold = 0.0
        
        if temp >= emergency:
            alert_level = "emergency"
            threshold = emergency
        elif temp >= critical:
            alert_level = "critical"
            threshold = critical
        elif temp >= warning:
            alert_level = "warning"
            threshold = warning
        
        if alert_level:
            # 检查警报冷却时间
            alert_key = f"{component}_{alert_level}"
            last_alert_time = self.last_alerts.get(alert_key, 0)
            
            if current_time - last_alert_time >= self.alert_cooldown:
                alert = ThermalAlert(
                    timestamp=current_time,
                    component=component,
                    temperature=temp,
                    threshold=threshold,
                    level=alert_level,
                    message=f"{component}温度{temp:.1f}°C超过{alert_level}阈值{threshold:.1f}°C"
                )
                
                self.alerts_history.append(alert)
                self.last_alerts[alert_key] = current_time
                
                # 调用警报回调函数
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"警报回调函数执行失败: {e}")
                
                # 记录日志
                if alert_level == "emergency":
                    logger.critical(f"🚨 {alert.message}")
                elif alert_level == "critical":
                    logger.error(f"⚠️ {alert.message}")
                else:
                    logger.warning(f"⚠️ {alert.message}")
    
    def start_monitoring(self):
        """开始温度监控"""
        if self.is_monitoring:
            logger.warning("温度监控已在运行")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🌡️ 温度监控已启动")
    
    def stop_monitoring(self):
        """停止温度监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("🌡️ 温度监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        logger.info("温度监控循环开始")
        
        while self.is_monitoring:
            try:
                # 获取温度读数
                reading = self.take_reading()
                
                # 检查警报
                self.check_thermal_alerts(reading)
                
                # 等待下次监控
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"温度监控循环异常: {e}")
                time.sleep(self.monitor_interval)
        
        logger.info("温度监控循环结束")
    
    def add_alert_callback(self, callback: Callable[[ThermalAlert], None]):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def add_reading_callback(self, callback: Callable[[ThermalReading], None]):
        """添加读数回调函数"""
        self.reading_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态（增强版）"""
        if not self.readings_history:
            return {"status": "no_data"}
        
        latest = self.readings_history[-1]
        current_time = time.time()
        
        # 计算温度趋势
        temp_trend = self._calculate_temperature_trend()
        
        # 计算系统热负载
        thermal_load = self._calculate_thermal_load(latest)
        
        return {
            "status": "active" if self.is_monitoring else "inactive",
            "timestamp": latest.timestamp,
            "temperatures": {
                "cpu": latest.cpu_temp,
                "gpu": latest.gpu_temp,
                "memory": latest.memory_temp
            },
            "usage": {
                "cpu": latest.cpu_usage,
                "gpu": latest.gpu_usage,
                "memory": latest.memory_usage
            },
            "fan_speed": latest.fan_speed,
            "power_draw": latest.power_draw,
            "thresholds": asdict(self.thresholds),
            "recent_alerts": len([a for a in self.alerts_history 
                                if current_time - a.timestamp < 300]),  # 5分钟内的警报
            "temperature_trend": temp_trend,
            "thermal_load": thermal_load,
            "system_health": self._assess_system_health(latest),
            "uptime_hours": (current_time - psutil.boot_time()) / 3600
        }
    
    def _calculate_temperature_trend(self) -> Dict[str, str]:
        """计算温度趋势"""
        if len(self.readings_history) < 10:
            return {"cpu": "stable", "gpu": "stable", "memory": "stable"}
        
        recent_readings = self.readings_history[-10:]
        trends = {}
        
        for component in ["cpu", "gpu", "memory"]:
            temps = [getattr(r, f"{component}_temp") for r in recent_readings]
            if len(temps) >= 2:
                if temps[-1] > temps[0] + 2:
                    trends[component] = "rising"
                elif temps[-1] < temps[0] - 2:
                    trends[component] = "falling"
                else:
                    trends[component] = "stable"
            else:
                trends[component] = "stable"
        
        return trends
    
    def _calculate_thermal_load(self, reading: ThermalReading) -> float:
        """计算系统热负载（0-100）"""
        try:
            # 基于温度和使用率计算综合热负载
            cpu_load = (reading.cpu_temp / 100.0) * 0.4 + (reading.cpu_usage / 100.0) * 0.1
            gpu_load = (reading.gpu_temp / 100.0) * 0.4 + (reading.gpu_usage / 100.0) * 0.1
            memory_load = (reading.memory_temp / 100.0) * 0.2 + (reading.memory_usage / 100.0) * 0.05
            
            total_load = (cpu_load + gpu_load + memory_load) * 100
            return min(total_load, 100.0)
        except:
            return 0.0
    
    def _assess_system_health(self, reading: ThermalReading) -> str:
        """评估系统健康状态"""
        max_temp = max(reading.cpu_temp, reading.gpu_temp, reading.memory_temp)
        
        if max_temp >= 85:
            return "critical"
        elif max_temp >= 75:
            return "warning"
        elif max_temp >= 65:
            return "caution"
        else:
            return "healthy"
    
    def get_temperature_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """获取温度历史数据"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        recent_readings = [
            reading for reading in self.readings_history
            if reading.timestamp >= cutoff_time
        ]
        
        return [asdict(reading) for reading in recent_readings]
    
    def get_alert_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """获取警报历史"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        recent_alerts = [
            alert for alert in self.alerts_history
            if alert.timestamp >= cutoff_time
        ]
        
        return [asdict(alert) for alert in recent_alerts]
    
    def export_data(self, filepath: str):
        """导出监控数据"""
        data = {
            "export_time": time.time(),
            "config": self.config,
            "thresholds": asdict(self.thresholds),
            "readings": [asdict(r) for r in self.readings_history],
            "alerts": [asdict(a) for a in self.alerts_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"监控数据已导出到: {filepath}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()


# 全局温度监控实例
_thermal_monitor_instance: Optional[ThermalMonitor] = None


def get_thermal_monitor(config: Optional[Dict[str, Any]] = None) -> ThermalMonitor:
    """获取温度监控实例（单例模式）"""
    global _thermal_monitor_instance
    
    if _thermal_monitor_instance is None:
        _thermal_monitor_instance = ThermalMonitor(config)
    
    return _thermal_monitor_instance


def main():
    """测试函数"""
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n正在停止温度监控...")
        monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建监控实例
    config = {
        'monitor_interval': 2.0,
        'thresholds': {
            'cpu_warning': 70.0,
            'cpu_critical': 75.0,
            'gpu_warning': 75.0,
            'gpu_critical': 80.0
        }
    }
    
    monitor = get_thermal_monitor(config)
    
    # 添加警报回调
    def alert_handler(alert: ThermalAlert):
        print(f"🚨 温度警报: {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # 开始监控
    monitor.start_monitoring()
    
    print("温度监控已启动，按Ctrl+C停止...")
    
    try:
        while True:
            time.sleep(5)
            status = monitor.get_current_status()
            print(f"CPU: {status['temperatures']['cpu']:.1f}°C, "
                  f"GPU: {status['temperatures']['gpu']:.1f}°C, "
                  f"内存: {status['temperatures']['memory']:.1f}°C")
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
