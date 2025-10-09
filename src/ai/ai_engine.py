"""
🧠 AI决策引擎
多AI模型融合的交易决策系统
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger

from src.core.config import settings


class AIEngine:
    """AI决策引擎"""
    
    def __init__(self, config=None):
        self.settings = config or settings
        self.models_loaded = False
        self.running = False
        logger.info("AI决策引擎初始化完成")
    
    async def initialize_models(self):
        """初始化AI模型"""
        logger.info("加载AI模型...")
        
        try:
            # 初始化模型存储
            self.models = {}
            self.model_configs = {}
            
            # 加载XGBoost模型
            await self._load_xgboost_model()
            
            # 加载LSTM模型
            await self._load_lstm_model()
            
            # 加载随机森林模型
            await self._load_random_forest_model()
            
            # 初始化集成学习器
            await self._initialize_ensemble()
            
            self.models_loaded = True
            logger.success(f"AI模型加载完成，共加载 {len(self.models)} 个模型")
            
        except Exception as e:
            logger.error(f"AI模型加载失败: {e}")
            self.models_loaded = False
            raise
    
    async def _load_xgboost_model(self):
        """加载XGBoost模型"""
        try:
            from xgboost import XGBClassifier
            
            model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['xgboost'] = model
            self.model_configs['xgboost'] = {
                'type': 'gradient_boosting',
                'features': ['price', 'volume', 'rsi', 'macd', 'bb_position', 'volatility'],
                'target': 'signal',
                'weight': 0.35
            }
            logger.info("✅ XGBoost模型初始化完成")
            
        except Exception as e:
            logger.error(f"XGBoost模型加载失败: {e}")
    
    async def _load_lstm_model(self):
        """加载LSTM模型"""
        try:
            # LSTM模型配置
            lstm_config = {
                'sequence_length': 60,
                'features': 8,
                'lstm_units': [128, 64, 32],
                'dropout': 0.2,
                'dense_units': [16, 8],
                'output_units': 3,  # BUY, SELL, HOLD
                'activation': 'softmax'
            }
            
            self.models['lstm'] = lstm_config
            self.model_configs['lstm'] = {
                'type': 'deep_learning',
                'architecture': 'LSTM',
                'weight': 0.35
            }
            logger.info("✅ LSTM模型配置完成")
            
        except Exception as e:
            logger.error(f"LSTM模型配置失败: {e}")
    
    async def _load_random_forest_model(self):
        """加载随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['random_forest'] = model
            self.model_configs['random_forest'] = {
                'type': 'ensemble_tree',
                'weight': 0.20
            }
            logger.info("✅ 随机森林模型初始化完成")
            
        except Exception as e:
            logger.error(f"随机森林模型加载失败: {e}")
    
    async def _initialize_ensemble(self):
        """初始化集成学习器"""
        try:
            self.ensemble_config = {
                'voting_strategy': 'weighted',
                'confidence_threshold': 0.6,
                'models': list(self.models.keys()),
                'weights': {
                    'xgboost': 0.35,
                    'lstm': 0.35,
                    'random_forest': 0.20,
                    'technical': 0.10
                }
            }
            logger.info("✅ 集成学习器初始化完成")
            
        except Exception as e:
            logger.error(f"集成学习器初始化失败: {e}")
    
    async def start_decision_loop(self):
        """启动AI决策循环"""
        self.running = True
        logger.info("AI决策引擎已启动")
        
        while self.running:
            try:
                # 获取最新市场数据
                market_data = await self._get_market_data()
                
                if market_data and self.models_loaded:
                    # 生成AI交易信号
                    signals = await self._generate_trading_signals(market_data)
                    
                    # 发送信号到交易执行引擎
                    if signals and signals.get('confidence', 0) > 0.6:
                        await self._send_trading_signal(signals)
                    
                    # 更新模型性能统计
                    await self._update_model_performance(signals)
                
                # 控制循环频率（每秒执行一次）
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"AI决策循环错误: {e}")
                await asyncio.sleep(5)  # 错误时等待更长时间
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """获取市场数据"""
        try:
            # 这里应该从数据收集模块获取实时数据
            # 暂时返回模拟数据结构
            return {
                'symbol': 'BTC/USDT',
                'price': 45000.0,
                'volume': 1000000,
                'timestamp': time.time(),
                'rsi': 55.0,
                'macd': 0.02,
                'bb_upper': 46000,
                'bb_lower': 44000,
                'volatility': 0.025
            }
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None
    
    async def _generate_trading_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        try:
            signals = {}
            confidences = {}
            
            # XGBoost预测
            if 'xgboost' in self.models:
                xgb_signal = await self._predict_xgboost(market_data)
                signals['xgboost'] = xgb_signal['signal']
                confidences['xgboost'] = xgb_signal['confidence']
            
            # LSTM预测
            if 'lstm' in self.models:
                lstm_signal = await self._predict_lstm(market_data)
                signals['lstm'] = lstm_signal['signal']
                confidences['lstm'] = lstm_signal['confidence']
            
            # 随机森林预测
            if 'random_forest' in self.models:
                rf_signal = await self._predict_random_forest(market_data)
                signals['random_forest'] = rf_signal['signal']
                confidences['random_forest'] = rf_signal['confidence']
            
            # 技术指标信号
            tech_signal = await self._generate_technical_signal(market_data)
            signals['technical'] = tech_signal['signal']
            confidences['technical'] = tech_signal['confidence']
            
            # 集成决策
            final_signal = await self._ensemble_decision(signals, confidences)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _predict_xgboost(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """XGBoost预测"""
        try:
            # 提取特征
            features = [
                market_data.get('price', 0),
                market_data.get('volume', 0),
                market_data.get('rsi', 50),
                market_data.get('macd', 0),
                (market_data.get('price', 0) - market_data.get('bb_lower', 0)) / 
                (market_data.get('bb_upper', 1) - market_data.get('bb_lower', 1)),
                market_data.get('volatility', 0.02)
            ]
            
            # 简化的信号生成逻辑
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            
            if rsi < 30 and macd > 0:
                return {'signal': 'BUY', 'confidence': 0.8}
            elif rsi > 70 and macd < 0:
                return {'signal': 'SELL', 'confidence': 0.8}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"XGBoost预测失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _predict_lstm(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """LSTM预测"""
        try:
            # LSTM需要序列数据，这里简化处理
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0.02)
            
            # 基于价格动量和波动率的简化预测
            if volatility < 0.01:  # 低波动
                return {'signal': 'HOLD', 'confidence': 0.6}
            elif volume > 500000:  # 高成交量
                if market_data.get('macd', 0) > 0:
                    return {'signal': 'BUY', 'confidence': 0.75}
                else:
                    return {'signal': 'SELL', 'confidence': 0.75}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"LSTM预测失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _predict_random_forest(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """随机森林预测"""
        try:
            rsi = market_data.get('rsi', 50)
            bb_position = (market_data.get('price', 0) - market_data.get('bb_lower', 0)) / \
                         (market_data.get('bb_upper', 1) - market_data.get('bb_lower', 1))
            
            # 基于多个指标的决策树逻辑
            if rsi < 25 and bb_position < 0.2:
                return {'signal': 'BUY', 'confidence': 0.85}
            elif rsi > 75 and bb_position > 0.8:
                return {'signal': 'SELL', 'confidence': 0.85}
            elif 40 <= rsi <= 60:
                return {'signal': 'HOLD', 'confidence': 0.7}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"随机森林预测失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _generate_technical_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成技术指标信号"""
        try:
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            price = market_data.get('price', 0)
            bb_upper = market_data.get('bb_upper', price * 1.02)
            bb_lower = market_data.get('bb_lower', price * 0.98)
            
            buy_score = 0
            sell_score = 0
            
            # RSI信号
            if rsi < 30:
                buy_score += 2
            elif rsi > 70:
                sell_score += 2
            
            # MACD信号
            if macd > 0:
                buy_score += 1
            else:
                sell_score += 1
            
            # 布林带信号
            if price < bb_lower:
                buy_score += 1
            elif price > bb_upper:
                sell_score += 1
            
            # 决策
            if buy_score >= 3:
                return {'signal': 'BUY', 'confidence': min(0.9, 0.5 + buy_score * 0.1)}
            elif sell_score >= 3:
                return {'signal': 'SELL', 'confidence': min(0.9, 0.5 + sell_score * 0.1)}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"技术指标信号生成失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _ensemble_decision(self, signals: Dict, confidences: Dict) -> Dict[str, Any]:
        """集成决策"""
        try:
            weights = self.ensemble_config.get('weights', {})
            
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_weight = 0
            
            for model, signal in signals.items():
                weight = weights.get(model, 0.25)
                confidence = confidences.get(model, 0.5)
                
                signal_scores[signal] += weight * confidence
                total_weight += weight
            
            # 归一化
            if total_weight > 0:
                for signal in signal_scores:
                    signal_scores[signal] /= total_weight
            
            # 选择最佳信号
            best_signal = max(signal_scores, key=signal_scores.get)
            best_confidence = signal_scores[best_signal]
            
            return {
                'signal': best_signal,
                'confidence': best_confidence,
                'individual_signals': signals,
                'individual_confidences': confidences,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"集成决策失败: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def _send_trading_signal(self, signal: Dict[str, Any]):
        """发送交易信号"""
        try:
            logger.info(f"🎯 发送交易信号: {signal['signal']} (置信度: {signal['confidence']:.2f})")
            # 这里应该发送到交易执行引擎
            # 暂时只记录日志
        except Exception as e:
            logger.error(f"发送交易信号失败: {e}")
    
    async def _update_model_performance(self, signal: Dict[str, Any]):
        """更新模型性能统计"""
        try:
            # 记录信号统计
            if not hasattr(self, 'signal_stats'):
                self.signal_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            signal_type = signal.get('signal', 'HOLD')
            self.signal_stats[signal_type] += 1
            
        except Exception as e:
            logger.error(f"更新模型性能失败: {e}")
    
    async def shutdown(self):
        """关闭AI引擎"""
        self.running = False
        logger.info("AI决策引擎已关闭")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        active_models = len(self.models) if hasattr(self, 'models') else 0
        
        status_info = {
            "status": "running" if self.running else "stopped",
            "models_loaded": self.models_loaded,
            "active_models": active_models,
            "model_details": {}
        }
        
        # 添加模型详细信息
        if hasattr(self, 'models'):
            for model_name, model in self.models.items():
                status_info["model_details"][model_name] = {
                    "loaded": model is not None,
                    "type": self.model_configs.get(model_name, {}).get('type', 'unknown')
                }
        
        # 添加信号统计
        if hasattr(self, 'signal_stats'):
            status_info["signal_statistics"] = self.signal_stats
        
        return status_info
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查GPU可用性
            gpu_available = await self._check_gpu_availability()
            
            # 检查模型状态
            models_healthy = self._check_models_health()
            
            # 检查内存使用
            memory_status = await self._check_memory_status()
            
            # 综合健康状态
            overall_healthy = (
                self.models_loaded and 
                models_healthy and 
                memory_status.get('available_memory_gb', 0) > 1.0
            )
            
            return {
                "healthy": overall_healthy,
                "models": "loaded" if self.models_loaded else "loading",
                "models_healthy": models_healthy,
                "gpu_available": gpu_available,
                "memory_status": memory_status,
                "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
                "last_signal_time": getattr(self, 'last_signal_time', None)
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "models": "error",
                "gpu_available": False
            }
    
    async def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
        except Exception:
            return False
    
    def _check_models_health(self) -> bool:
        """检查模型健康状态"""
        try:
            if not hasattr(self, 'models') or not self.models:
                return False
            
            # 检查每个模型是否正常
            for model_name, model in self.models.items():
                if model is None:
                    logger.warning(f"模型 {model_name} 未加载")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")
            return False
    
    async def _check_memory_status(self) -> Dict[str, Any]:
        """检查内存状态"""
        try:
            import psutil
            
            # 系统内存
            memory = psutil.virtual_memory()
            
            # GPU内存（如果可用）
            gpu_memory = {}
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory[f'gpu_{i}'] = {
                            'total_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                            'allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                            'cached_gb': torch.cuda.memory_reserved(i) / 1e9
                        }
            except Exception as e:
                logger.warning(f"GPU内存检查失败: {e}")
                gpu_memory = {}
            
            return {
                'total_memory_gb': memory.total / 1e9,
                'available_memory_gb': memory.available / 1e9,
                'used_memory_percent': memory.percent,
                'gpu_memory': gpu_memory
            }
            
        except Exception as e:
            logger.error(f"内存状态检查失败: {e}")
            return {
                'total_memory_gb': 0,
                'available_memory_gb': 0,
                'used_memory_percent': 0,
                'error': str(e)
            }
