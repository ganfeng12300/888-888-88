#!/usr/bin/env python3
"""
📈 888-888-88 AI性能监控器
Production-Grade AI Performance Monitoring System
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
from loguru import logger

# 导入错误处理系统
from src.core.error_handling_system import ai_operation, handle_errors

@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_id: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    prediction_count: int
    error_count: int
    confidence_avg: float
    confidence_std: float

@dataclass
class PredictionRecord:
    """预测记录"""
    model_id: str
    timestamp: datetime
    features: Dict[str, Any]
    prediction: Any
    confidence: float
    actual_result: Optional[Any] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None

class AIPerformanceMonitor:
    """AI性能监控器"""
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.prediction_records: deque = deque(maxlen=max_records)
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 性能统计
        self.total_predictions = 0
        self.total_errors = 0
        self.model_prediction_counts = defaultdict(int)
        self.model_error_counts = defaultdict(int)
        
        # 创建监控日志目录
        self.log_dir = Path("logs/ai_performance")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📈 AI性能监控器初始化完成")
    
    @ai_operation
    async def record_prediction(self, model_id: str, features: Dict[str, Any], 
                               prediction: Any, confidence: float, 
                               processing_time_ms: float, actual_result: Optional[Any] = None,
                               error: Optional[str] = None):
        """记录预测结果"""
        try:
            record = PredictionRecord(
                model_id=model_id,
                timestamp=datetime.now(),
                features=features,
                prediction=prediction,
                confidence=confidence,
                actual_result=actual_result,
                processing_time_ms=processing_time_ms,
                error=error
            )
            
            self.prediction_records.append(record)
            
            # 更新统计
            self.total_predictions += 1
            self.model_prediction_counts[model_id] += 1
            
            if error:
                self.total_errors += 1
                self.model_error_counts[model_id] += 1
            
            # 异步记录到文件
            asyncio.create_task(self._log_prediction_record(record))
            
        except Exception as e:
            logger.error(f"❌ 记录预测结果失败: {e}")
    
    async def _log_prediction_record(self, record: PredictionRecord):
        """记录预测到文件"""
        try:
            log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            record_data = {
                "model_id": record.model_id,
                "timestamp": record.timestamp.isoformat(),
                "prediction": str(record.prediction),
                "confidence": record.confidence,
                "processing_time_ms": record.processing_time_ms,
                "has_actual": record.actual_result is not None,
                "has_error": record.error is not None
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"❌ 记录预测日志失败: {e}")
    
    @ai_operation
    async def calculate_model_accuracy(self, model_id: str, hours: int = 24) -> float:
        """计算模型准确率"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 获取有实际结果的预测记录
            relevant_records = [
                record for record in self.prediction_records
                if (record.model_id == model_id and 
                    record.timestamp >= cutoff_time and
                    record.actual_result is not None and
                    record.error is None)
            ]
            
            if not relevant_records:
                return 0.0
            
            # 计算准确率（这里需要根据具体业务逻辑调整）
            correct_predictions = 0
            
            for record in relevant_records:
                # 对于数值预测，使用阈值判断
                if isinstance(record.prediction, (int, float)) and isinstance(record.actual_result, (int, float)):
                    # 如果预测值与实际值的相对误差小于10%，认为预测正确
                    relative_error = abs(record.prediction - record.actual_result) / max(abs(record.actual_result), 1e-6)
                    if relative_error < 0.1:
                        correct_predictions += 1
                # 对于分类预测，直接比较
                elif record.prediction == record.actual_result:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(relevant_records)
            
            # 更新模型统计
            self.model_stats[model_id]['accuracy'] = accuracy
            self.model_stats[model_id]['accuracy_updated'] = datetime.now()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"❌ 计算模型准确率失败 {model_id}: {e}")
            return 0.0
    
    @ai_operation
    async def get_average_accuracy(self, hours: int = 24) -> float:
        """获取平均准确率"""
        try:
            model_ids = set(record.model_id for record in self.prediction_records)
            
            if not model_ids:
                return 0.0
            
            accuracies = []
            for model_id in model_ids:
                accuracy = await self.calculate_model_accuracy(model_id, hours)
                if accuracy > 0:
                    accuracies.append(accuracy)
            
            return sum(accuracies) / len(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logger.error(f"❌ 获取平均准确率失败: {e}")
            return 0.0
    
    @ai_operation
    async def get_average_inference_time(self, model_id: Optional[str] = None, hours: int = 24) -> float:
        """获取平均推理时间"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            relevant_records = [
                record for record in self.prediction_records
                if (record.timestamp >= cutoff_time and
                    record.error is None and
                    (model_id is None or record.model_id == model_id))
            ]
            
            if not relevant_records:
                return 0.0
            
            processing_times = [record.processing_time_ms for record in relevant_records]
            return sum(processing_times) / len(processing_times)
            
        except Exception as e:
            logger.error(f"❌ 获取平均推理时间失败: {e}")
            return 0.0
    
    @ai_operation
    async def get_predictions_count(self, model_id: Optional[str] = None, hours: int = 24) -> int:
        """获取预测次数"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            relevant_records = [
                record for record in self.prediction_records
                if (record.timestamp >= cutoff_time and
                    (model_id is None or record.model_id == model_id))
            ]
            
            return len(relevant_records)
            
        except Exception as e:
            logger.error(f"❌ 获取预测次数失败: {e}")
            return 0
    
    @ai_operation
    async def get_error_count(self, model_id: Optional[str] = None, hours: int = 24) -> int:
        """获取错误次数"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            error_records = [
                record for record in self.prediction_records
                if (record.timestamp >= cutoff_time and
                    record.error is not None and
                    (model_id is None or record.model_id == model_id))
            ]
            
            return len(error_records)
            
        except Exception as e:
            logger.error(f"❌ 获取错误次数失败: {e}")
            return 0
    
    @ai_operation
    async def get_model_performance_summary(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """获取模型性能摘要"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 获取相关记录
            model_records = [
                record for record in self.prediction_records
                if record.model_id == model_id and record.timestamp >= cutoff_time
            ]
            
            if not model_records:
                return {
                    "model_id": model_id,
                    "period_hours": hours,
                    "total_predictions": 0,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "accuracy": 0.0,
                    "avg_inference_time_ms": 0.0,
                    "avg_confidence": 0.0,
                    "confidence_std": 0.0
                }
            
            # 计算统计指标
            total_predictions = len(model_records)
            error_records = [r for r in model_records if r.error is not None]
            error_count = len(error_records)
            error_rate = error_count / total_predictions
            
            # 计算准确率
            accuracy = await self.calculate_model_accuracy(model_id, hours)
            
            # 计算推理时间
            successful_records = [r for r in model_records if r.error is None]
            if successful_records:
                inference_times = [r.processing_time_ms for r in successful_records]
                avg_inference_time = sum(inference_times) / len(inference_times)
                
                confidences = [r.confidence for r in successful_records]
                avg_confidence = sum(confidences) / len(confidences)
                confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
            else:
                avg_inference_time = 0.0
                avg_confidence = 0.0
                confidence_std = 0.0
            
            return {
                "model_id": model_id,
                "period_hours": hours,
                "total_predictions": total_predictions,
                "error_count": error_count,
                "error_rate": error_rate,
                "accuracy": accuracy,
                "avg_inference_time_ms": avg_inference_time,
                "avg_confidence": avg_confidence,
                "confidence_std": confidence_std,
                "last_prediction": model_records[-1].timestamp.isoformat() if model_records else None
            }
            
        except Exception as e:
            logger.error(f"❌ 获取模型性能摘要失败 {model_id}: {e}")
            return {}
    
    @ai_operation
    async def get_all_models_performance(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的性能摘要"""
        try:
            model_ids = set(record.model_id for record in self.prediction_records)
            
            performance_data = {}
            
            for model_id in model_ids:
                performance_data[model_id] = await self.get_model_performance_summary(model_id, hours)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"❌ 获取所有模型性能失败: {e}")
            return {}
    
    @ai_operation
    async def get_performance_trends(self, model_id: str, days: int = 7) -> Dict[str, List[Any]]:
        """获取性能趋势数据"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 按小时分组统计
            hourly_stats = defaultdict(lambda: {
                'predictions': 0,
                'errors': 0,
                'inference_times': [],
                'confidences': []
            })
            
            for record in self.prediction_records:
                if (record.model_id == model_id and 
                    start_time <= record.timestamp <= end_time):
                    
                    hour_key = record.timestamp.replace(minute=0, second=0, microsecond=0)
                    
                    hourly_stats[hour_key]['predictions'] += 1
                    
                    if record.error:
                        hourly_stats[hour_key]['errors'] += 1
                    else:
                        hourly_stats[hour_key]['inference_times'].append(record.processing_time_ms)
                        hourly_stats[hour_key]['confidences'].append(record.confidence)
            
            # 构建趋势数据
            timestamps = []
            prediction_counts = []
            error_rates = []
            avg_inference_times = []
            avg_confidences = []
            
            # 按时间排序
            sorted_hours = sorted(hourly_stats.keys())
            
            for hour in sorted_hours:
                stats = hourly_stats[hour]
                
                timestamps.append(hour.isoformat())
                prediction_counts.append(stats['predictions'])
                
                error_rate = stats['errors'] / max(1, stats['predictions'])
                error_rates.append(error_rate)
                
                avg_inference_time = (
                    sum(stats['inference_times']) / len(stats['inference_times'])
                    if stats['inference_times'] else 0.0
                )
                avg_inference_times.append(avg_inference_time)
                
                avg_confidence = (
                    sum(stats['confidences']) / len(stats['confidences'])
                    if stats['confidences'] else 0.0
                )
                avg_confidences.append(avg_confidence)
            
            return {
                "model_id": model_id,
                "period_days": days,
                "timestamps": timestamps,
                "prediction_counts": prediction_counts,
                "error_rates": error_rates,
                "avg_inference_times_ms": avg_inference_times,
                "avg_confidences": avg_confidences
            }
            
        except Exception as e:
            logger.error(f"❌ 获取性能趋势失败 {model_id}: {e}")
            return {}
    
    @ai_operation
    async def detect_performance_anomalies(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """检测性能异常"""
        try:
            anomalies = []
            
            # 获取模型性能摘要
            performance = await self.get_model_performance_summary(model_id, hours)
            
            if not performance or performance['total_predictions'] == 0:
                return anomalies
            
            # 检查错误率异常
            if performance['error_rate'] > 0.1:  # 错误率超过10%
                anomalies.append({
                    "type": "high_error_rate",
                    "severity": "high" if performance['error_rate'] > 0.2 else "medium",
                    "message": f"模型 {model_id} 错误率过高: {performance['error_rate']:.2%}",
                    "value": performance['error_rate'],
                    "threshold": 0.1
                })
            
            # 检查推理时间异常
            if performance['avg_inference_time_ms'] > 1000:  # 推理时间超过1秒
                anomalies.append({
                    "type": "slow_inference",
                    "severity": "high" if performance['avg_inference_time_ms'] > 5000 else "medium",
                    "message": f"模型 {model_id} 推理时间过长: {performance['avg_inference_time_ms']:.1f}ms",
                    "value": performance['avg_inference_time_ms'],
                    "threshold": 1000
                })
            
            # 检查准确率异常
            if performance['accuracy'] < 0.6:  # 准确率低于60%
                anomalies.append({
                    "type": "low_accuracy",
                    "severity": "high" if performance['accuracy'] < 0.4 else "medium",
                    "message": f"模型 {model_id} 准确率过低: {performance['accuracy']:.2%}",
                    "value": performance['accuracy'],
                    "threshold": 0.6
                })
            
            # 检查置信度异常
            if performance['avg_confidence'] < 0.5:  # 平均置信度低于50%
                anomalies.append({
                    "type": "low_confidence",
                    "severity": "medium",
                    "message": f"模型 {model_id} 平均置信度过低: {performance['avg_confidence']:.2%}",
                    "value": performance['avg_confidence'],
                    "threshold": 0.5
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ 检测性能异常失败 {model_id}: {e}")
            return []
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控器统计信息"""
        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "overall_error_rate": self.total_errors / max(1, self.total_predictions),
            "monitored_models": len(set(record.model_id for record in self.prediction_records)),
            "records_count": len(self.prediction_records),
            "max_records": self.max_records
        }
    
    async def cleanup_old_records(self, days: int = 30):
        """清理旧记录"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # 清理内存中的记录
            original_count = len(self.prediction_records)
            self.prediction_records = deque(
                (record for record in self.prediction_records if record.timestamp >= cutoff_time),
                maxlen=self.max_records
            )
            
            cleaned_count = original_count - len(self.prediction_records)
            
            if cleaned_count > 0:
                logger.info(f"🧹 清理了 {cleaned_count} 条旧的预测记录")
            
        except Exception as e:
            logger.error(f"❌ 清理旧记录失败: {e}")

# 全局AI性能监控器实例
ai_performance_monitor = AIPerformanceMonitor()

# 导出主要组件
__all__ = [
    'AIPerformanceMonitor',
    'ModelPerformanceMetrics',
    'PredictionRecord',
    'ai_performance_monitor'
]
