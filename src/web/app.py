#!/usr/bin/env python3
"""
🌐 888-888-88 Web应用
生产级Web管理界面和API服务
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from loguru import logger
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.api_config_manager import APIConfigManager
from src.risk_management.risk_manager import get_risk_manager
from src.monitoring.system_monitor import SystemMonitor
from src.trading.real_trading_manager import get_real_trading_manager
from src.ai.enhanced_ai_status_monitor import get_enhanced_ai_status_monitor


class WebApp:
    """Web应用管理器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="888-888-88 量化交易系统",
            description="生产级量化交易管理平台",
            version="1.0.0"
        )
        
        # 组件实例
        self.api_config = APIConfigManager()
        self.risk_manager = get_risk_manager()
        self.system_monitor = SystemMonitor()
        self.trading_manager = get_real_trading_manager()
        self.ai_monitor = get_ai_status_monitor()
        
        # WebSocket连接管理
        self.websocket_connections: List[WebSocket] = []
        
        # 配置中间件
        self._setup_middleware()
        
        # 配置路由
        self._setup_routes()
        
        logger.info("🌐 Web应用初始化完成")
    
    def _setup_middleware(self) -> None:
        """配置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """配置路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """主仪表板"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>888-888-88 量化交易系统</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
                    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .metric { text-align: center; padding: 15px; background: #ecf0f1; border-radius: 5px; margin-bottom: 10px; }
                    .metric-small { text-align: center; padding: 10px; background: #ecf0f1; border-radius: 5px; margin-bottom: 8px; }
                    .metric-value { font-size: 2em; font-weight: bold; color: #27ae60; }
                    .metric-value-small { font-size: 1.5em; font-weight: bold; color: #27ae60; }
                    .metric-label { color: #7f8c8d; margin-top: 5px; font-size: 0.9em; }
                    .status-good { color: #27ae60; }
                    .status-warning { color: #f39c12; }
                    .status-error { color: #e74c3c; }
                    .status-info { color: #3498db; }
                    button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
                    button:hover { background: #2980b9; }
                    .log-container { height: 300px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; }
                    .positions-table { width: 100%; border-collapse: collapse; }
                    .positions-table th, .positions-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    .positions-table th { background-color: #f2f2f2; }
                    .ai-model { background: #e8f5e8; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
                    .ai-model-name { font-weight: bold; color: #27ae60; }
                    .ai-model-stats { font-size: 0.9em; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 888-888-88 量化交易系统</h1>
                        <p>生产级实盘交易管理平台 | 实时AI决策 | 多交易所支持</p>
                        <div style="margin-top: 10px;">
                            <span id="connection-status" class="status-good">🟢 已连接</span>
                            <span style="margin-left: 20px;">最后更新: <span id="last-update">--</span></span>
                        </div>
                    </div>
                    
                    <!-- 第一行：核心指标 -->
                    <div class="grid">
                        <div class="card">
                            <h3>💰 账户资产</h3>
                            <div class="metric">
                                <div class="metric-value" id="total-balance">$0.00</div>
                                <div class="metric-label">总资产 (USDT)</div>
                            </div>
                            <div class="grid-2">
                                <div class="metric-small">
                                    <div class="metric-value-small" id="available-balance">$0.00</div>
                                    <div class="metric-label">可用余额</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-value-small" id="used-balance">$0.00</div>
                                    <div class="metric-label">已用余额</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>📈 持仓概览</h3>
                            <div class="metric">
                                <div class="metric-value" id="positions-count">0</div>
                                <div class="metric-label">当前持仓数</div>
                            </div>
                            <div class="grid-2">
                                <div class="metric-small">
                                    <div class="metric-value-small status-good" id="total-pnl">$0.00</div>
                                    <div class="metric-label">总盈亏</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-value-small" id="leverage-ratio">1.0x</div>
                                    <div class="metric-label">平均杠杆</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🤖 AI模型状态</h3>
                            <div class="metric">
                                <div class="metric-value status-info" id="ai-status">运行中</div>
                                <div class="metric-label">AI引擎状态</div>
                            </div>
                            <div class="grid-2">
                                <div class="metric-small">
                                    <div class="metric-value-small" id="active-models">0</div>
                                    <div class="metric-label">活跃模型</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-value-small" id="prediction-accuracy">0%</div>
                                    <div class="metric-label">预测准确率</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🛡️ 风险管理</h3>
                            <div class="metric">
                                <div class="metric-value status-good" id="risk-level">低风险</div>
                                <div class="metric-label">风险等级</div>
                            </div>
                            <div class="grid-2">
                                <div class="metric-small">
                                    <div class="metric-value-small" id="max-drawdown">0%</div>
                                    <div class="metric-label">最大回撤</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-value-small" id="var-95">$0.00</div>
                                    <div class="metric-label">VaR(95%)</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 第二行：详细信息 -->
                    <div class="grid-2">
                        <div class="card">
                            <h3>📊 当前持仓</h3>
                            <div id="positions-container">
                                <table class="positions-table">
                                    <thead>
                                        <tr>
                                            <th>交易对</th>
                                            <th>方向</th>
                                            <th>数量</th>
                                            <th>开仓价</th>
                                            <th>当前价</th>
                                            <th>盈亏</th>
                                            <th>杠杆</th>
                                        </tr>
                                    </thead>
                                    <tbody id="positions-table-body">
                                        <tr>
                                            <td colspan="7" style="text-align: center; color: #999;">暂无持仓</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🤖 AI模型详情</h3>
                            <div id="ai-models-container">
                                <div class="ai-model">
                                    <div class="ai-model-name">LSTM模型</div>
                                    <div class="ai-model-stats">准确率: <span id="lstm-accuracy">--</span> | 信心度: <span id="lstm-confidence">--</span></div>
                                </div>
                                <div class="ai-model">
                                    <div class="ai-model-name">Transformer模型</div>
                                    <div class="ai-model-stats">准确率: <span id="transformer-accuracy">--</span> | 信心度: <span id="transformer-confidence">--</span></div>
                                </div>
                                <div class="ai-model">
                                    <div class="ai-model-name">CNN模型</div>
                                    <div class="ai-model-stats">准确率: <span id="cnn-accuracy">--</span> | 信心度: <span id="cnn-confidence">--</span></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 第三行：交易历史和控制面板 -->
                    <div class="grid-2">
                        <div class="card">
                            <h3>📈 最近交易</h3>
                            <div id="trades-container">
                                <table class="positions-table">
                                    <thead>
                                        <tr>
                                            <th>时间</th>
                                            <th>交易对</th>
                                            <th>方向</th>
                                            <th>数量</th>
                                            <th>价格</th>
                                            <th>手续费</th>
                                        </tr>
                                    </thead>
                                    <tbody id="trades-table-body">
                                        <tr>
                                            <td colspan="6" style="text-align: center; color: #999;">暂无交易记录</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🔧 系统控制</h3>
                            <div style="margin-bottom: 20px;">
                                <button onclick="startSystem()">🚀 启动系统</button>
                                <button onclick="stopSystem()">⏹️ 停止系统</button>
                                <button onclick="refreshData()">🔄 刷新数据</button>
                                <button onclick="emergencyStop()">🚨 紧急停止</button>
                            </div>
                            <div class="metric">
                                <div class="metric-value-small status-info" id="system-uptime">00:00:00</div>
                                <div class="metric-label">系统运行时间</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>📝 系统日志</h3>
                        <div class="log-container" id="system-logs">
                            <div>[系统启动] 888-888-88量化交易系统正在初始化...</div>
                        </div>
                    </div>
                </div>
                
                <script>
                    // WebSocket连接
                    const ws = new WebSocket('ws://localhost:8888/ws');
                    let startTime = Date.now();
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    function updateDashboard(data) {
                        if (data.type === 'system_status') {
                            document.getElementById('connection-status').innerHTML = '🟢 已连接';
                        } else if (data.type === 'log') {
                            addLogEntry(data.message);
                        } else if (data.type === 'data_update') {
                            updateAllData(data);
                        }
                        
                        // 更新最后更新时间
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    }
                    
                    function updateAllData(data) {
                        // 更新账户信息
                        if (data.account) {
                            document.getElementById('total-balance').textContent = `$${data.account.total_balance.toFixed(2)}`;
                            document.getElementById('available-balance').textContent = `$${data.account.available_balance.toFixed(2)}`;
                            document.getElementById('used-balance').textContent = `$${data.account.used_balance.toFixed(2)}`;
                        }
                        
                        // 更新持仓信息
                        if (data.positions) {
                            updatePositionsTable(data.positions);
                            document.getElementById('positions-count').textContent = Object.keys(data.positions).length;
                            
                            // 计算总盈亏和平均杠杆
                            let totalPnl = 0;
                            let totalLeverage = 0;
                            let count = 0;
                            
                            for (const pos of Object.values(data.positions)) {
                                totalPnl += pos.unrealized_pnl;
                                totalLeverage += pos.leverage;
                                count++;
                            }
                            
                            document.getElementById('total-pnl').textContent = `$${totalPnl.toFixed(2)}`;
                            document.getElementById('total-pnl').className = totalPnl >= 0 ? 'metric-value-small status-good' : 'metric-value-small status-error';
                            document.getElementById('leverage-ratio').textContent = count > 0 ? `${(totalLeverage/count).toFixed(1)}x` : '1.0x';
                        }
                        
                        // 更新AI状态
                        if (data.ai_status) {
                            document.getElementById('ai-status').textContent = data.ai_status.status;
                            document.getElementById('active-models').textContent = data.ai_status.active_models.length;
                            
                            if (data.ai_status.signal_stats) {
                                const accuracy = (data.ai_status.signal_stats.signal_accuracy * 100).toFixed(1);
                                document.getElementById('prediction-accuracy').textContent = `${accuracy}%`;
                            }
                            
                            // 更新模型详情
                            updateAIModels(data.ai_status.model_performance);
                        }
                        
                        // 更新交易历史
                        if (data.trades) {
                            updateTradesTable(data.trades);
                        }
                        
                        // 更新风险指标
                        if (data.risk) {
                            document.getElementById('max-drawdown').textContent = `${(data.risk.max_drawdown * 100).toFixed(2)}%`;
                            document.getElementById('var-95').textContent = `$${data.risk.var_95.toFixed(2)}`;
                        }
                    }
                    
                    function updatePositionsTable(positions) {
                        const tbody = document.getElementById('positions-table-body');
                        tbody.innerHTML = '';
                        
                        if (Object.keys(positions).length === 0) {
                            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #999;">暂无持仓</td></tr>';
                            return;
                        }
                        
                        for (const [key, pos] of Object.entries(positions)) {
                            const row = document.createElement('tr');
                            const pnlClass = pos.unrealized_pnl >= 0 ? 'status-good' : 'status-error';
                            
                            row.innerHTML = `
                                <td>${pos.symbol}</td>
                                <td><span class="${pos.side === 'long' ? 'status-good' : 'status-error'}">${pos.side.toUpperCase()}</span></td>
                                <td>${pos.size.toFixed(4)}</td>
                                <td>$${pos.entry_price.toFixed(4)}</td>
                                <td>$${pos.current_price.toFixed(4)}</td>
                                <td><span class="${pnlClass}">$${pos.unrealized_pnl.toFixed(2)}</span></td>
                                <td>${pos.leverage.toFixed(1)}x</td>
                            `;
                            tbody.appendChild(row);
                        }
                    }
                    
                    function updateTradesTable(trades) {
                        const tbody = document.getElementById('trades-table-body');
                        tbody.innerHTML = '';
                        
                        if (trades.length === 0) {
                            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">暂无交易记录</td></tr>';
                            return;
                        }
                        
                        // 显示最近10笔交易
                        const recentTrades = trades.slice(-10).reverse();
                        
                        for (const trade of recentTrades) {
                            const row = document.createElement('tr');
                            const sideClass = trade.side === 'buy' ? 'status-good' : 'status-error';
                            const time = new Date(trade.timestamp).toLocaleTimeString();
                            
                            row.innerHTML = `
                                <td>${time}</td>
                                <td>${trade.symbol}</td>
                                <td><span class="${sideClass}">${trade.side.toUpperCase()}</span></td>
                                <td>${trade.amount.toFixed(4)}</td>
                                <td>$${trade.price.toFixed(4)}</td>
                                <td>$${trade.fee.toFixed(4)}</td>
                            `;
                            tbody.appendChild(row);
                        }
                    }
                    
                    function updateAIModels(modelPerformance) {
                        if (modelPerformance.LSTM) {
                            document.getElementById('lstm-accuracy').textContent = `${(modelPerformance.LSTM.accuracy * 100).toFixed(1)}%`;
                            document.getElementById('lstm-confidence').textContent = `${(modelPerformance.LSTM.avg_confidence * 100).toFixed(1)}%`;
                        }
                        
                        if (modelPerformance.Transformer) {
                            document.getElementById('transformer-accuracy').textContent = `${(modelPerformance.Transformer.accuracy * 100).toFixed(1)}%`;
                            document.getElementById('transformer-confidence').textContent = `${(modelPerformance.Transformer.avg_confidence * 100).toFixed(1)}%`;
                        }
                        
                        if (modelPerformance.CNN) {
                            document.getElementById('cnn-accuracy').textContent = `${(modelPerformance.CNN.accuracy * 100).toFixed(1)}%`;
                            document.getElementById('cnn-confidence').textContent = `${(modelPerformance.CNN.avg_confidence * 100).toFixed(1)}%`;
                        }
                    }
                    
                    function addLogEntry(message) {
                        const logContainer = document.getElementById('system-logs');
                        const logEntry = document.createElement('div');
                        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                        logContainer.appendChild(logEntry);
                        logContainer.scrollTop = logContainer.scrollHeight;
                        
                        // 限制日志条数
                        while (logContainer.children.length > 100) {
                            logContainer.removeChild(logContainer.firstChild);
                        }
                    }
                    
                    function updateUptime() {
                        const uptime = Date.now() - startTime;
                        const hours = Math.floor(uptime / 3600000);
                        const minutes = Math.floor((uptime % 3600000) / 60000);
                        const seconds = Math.floor((uptime % 60000) / 1000);
                        
                        document.getElementById('system-uptime').textContent = 
                            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    }
                    
                    function startSystem() {
                        fetch('/api/system/start', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                addLogEntry('系统启动命令已发送');
                                console.log('系统启动:', data);
                            });
                    }
                    
                    function stopSystem() {
                        fetch('/api/system/stop', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                addLogEntry('系统停止命令已发送');
                                console.log('系统停止:', data);
                            });
                    }
                    
                    function emergencyStop() {
                        if (confirm('确定要执行紧急停止吗？这将立即停止所有交易活动。')) {
                            fetch('/api/emergency/stop', {method: 'POST'})
                                .then(response => response.json())
                                .then(data => {
                                    addLogEntry('🚨 紧急停止已执行');
                                    console.log('紧急停止:', data);
                                });
                        }
                    }
                    
                    function refreshData() {
                        Promise.all([
                            fetch('/api/status').then(r => r.json()),
                            fetch('/api/trading/data').then(r => r.json()),
                            fetch('/api/ai/status').then(r => r.json())
                        ]).then(([status, trading, ai]) => {
                            updateAllData({
                                account: trading.account_info,
                                positions: trading.positions,
                                trades: trading.trades,
                                ai_status: ai,
                                risk: status.risk_management
                            });
                            addLogEntry('数据刷新完成');
                        }).catch(err => {
                            addLogEntry(`数据刷新失败: ${err.message}`);
                        });
                    }
                    
                    // 定期刷新数据
                    setInterval(refreshData, 30000);
                    setInterval(updateUptime, 1000);
                    
                    // 页面加载完成后立即刷新数据
                    window.onload = function() {
                        addLogEntry('Web界面已加载');
                        setTimeout(refreshData, 1000);
                    };
                </script>
            </body>
            </html>
            """
        
        @self.app.get("/api/status")
        async def get_system_status():
            """获取系统状态"""
            try:
                # 获取各组件状态
                risk_report = self.risk_manager.get_risk_report()
                monitor_report = self.system_monitor.get_monitoring_report()
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": "healthy",
                    "risk_management": {
                        "current_balance": risk_report.get("current_balance", 0),
                        "total_positions": risk_report.get("position_count", 0),
                        "emergency_stop": risk_report.get("emergency_stop", False)
                    },
                    "monitoring": {
                        "cpu_usage": monitor_report["current_metrics"].get("cpu_usage", 0),
                        "memory_usage": monitor_report["current_metrics"].get("memory_usage", 0),
                        "active_alerts": monitor_report["alert_summary"]["active_alerts"]
                    },
                    "api_config": {
                        "configured_exchanges": len(self.api_config.list_configured_exchanges())
                    }
                }
            except Exception as e:
                logger.error(f"❌ 获取系统状态失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/system/start")
        async def start_system():
            """启动系统"""
            try:
                # 启动各个组件
                self.system_monitor.start_monitoring()
                
                await self._broadcast_message({
                    "type": "log",
                    "message": "系统启动成功"
                })
                
                return {"status": "success", "message": "系统已启动"}
            except Exception as e:
                logger.error(f"❌ 启动系统失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/system/stop")
        async def stop_system():
            """停止系统"""
            try:
                # 停止各个组件
                self.system_monitor.stop_monitoring()
                
                await self._broadcast_message({
                    "type": "log",
                    "message": "系统已停止"
                })
                
                return {"status": "success", "message": "系统已停止"}
            except Exception as e:
                logger.error(f"❌ 停止系统失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/risk/report")
        async def get_risk_report():
            """获取风险报告"""
            try:
                return self.risk_manager.get_risk_report()
            except Exception as e:
                logger.error(f"❌ 获取风险报告失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/monitoring/report")
        async def get_monitoring_report():
            """获取监控报告"""
            try:
                return self.system_monitor.get_monitoring_report()
            except Exception as e:
                logger.error(f"❌ 获取监控报告失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/exchanges")
        async def get_exchanges():
            """获取交易所配置"""
            try:
                exchanges = self.api_config.list_configured_exchanges()
                return {"exchanges": exchanges}
            except Exception as e:
                logger.error(f"❌ 获取交易所配置失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/trading/data")
        async def get_trading_data():
            """获取真实交易数据"""
            try:
                # 初始化交易管理器
                await self.trading_manager.initialize_exchanges()
                
                # 获取所有交易数据
                data = await self.trading_manager.update_all_data()
                
                return {
                    "account_info": data.get('account_info').__dict__ if data.get('account_info') else None,
                    "positions": {k: v.__dict__ for k, v in data.get('positions', {}).items()},
                    "trades": [t.__dict__ for t in data.get('trades', [])],
                    "trading_summary": self.trading_manager.get_trading_summary(),
                    "update_time": data.get('update_time')
                }
            except Exception as e:
                logger.error(f"❌ 获取交易数据失败: {e}")
                return {
                    "account_info": None,
                    "positions": {},
                    "trades": [],
                    "trading_summary": {},
                    "error": str(e)
                }
        
        @self.app.get("/api/ai/status")
        async def get_ai_status():
            """获取AI状态"""
            try:
                # 启动AI监控
                if not self.ai_monitor.monitoring:
                    self.ai_monitor.start_monitoring()
                
                return self.ai_monitor.get_ai_status_report()
            except Exception as e:
                logger.error(f"❌ 获取AI状态失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/emergency/stop")
        async def emergency_stop():
            """紧急停止"""
            try:
                # 停止所有交易活动
                self.risk_manager.emergency_stop = True
                
                await self._broadcast_message({
                    "type": "log",
                    "message": "🚨 紧急停止已激活，所有交易活动已暂停"
                })
                
                return {"status": "success", "message": "紧急停止已执行"}
            except Exception as e:
                logger.error(f"❌ 紧急停止失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket端点"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # 发送初始状态
                await websocket.send_text(json.dumps({
                    "type": "system_status",
                    "status": "已连接"
                }))
                
                # 保持连接
                while True:
                    await websocket.receive_text()
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                logger.error(f"❌ WebSocket异常: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    async def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """广播消息到所有WebSocket连接"""
        if not self.websocket_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_text)
            except:
                disconnected.append(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    async def start_background_tasks(self) -> None:
        """启动后台任务"""
        # 定期广播系统状态
        asyncio.create_task(self._status_broadcast_loop())
    
    async def _status_broadcast_loop(self) -> None:
        """状态广播循环"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒广播一次
                
                status_data = {
                    "type": "system_status",
                    "status": "运行中",
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_message(status_data)
                
            except Exception as e:
                logger.error(f"❌ 状态广播异常: {e}")
    
    def run(self) -> None:
        """运行Web应用"""
        logger.info(f"🌐 启动Web服务器: http://{self.host}:{self.port}")
        
        # 启动后台任务
        asyncio.create_task(self.start_background_tasks())
        
        # 启动服务器
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# 全局Web应用实例
web_app = None

def get_web_app(host: str = "0.0.0.0", port: int = 8888) -> WebApp:
    """获取Web应用实例"""
    global web_app
    if web_app is None:
        web_app = WebApp(host, port)
    return web_app


if __name__ == "__main__":
    # 启动Web应用
    app = WebApp()
    app.run()
