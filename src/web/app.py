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
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                    .metric { text-align: center; padding: 15px; background: #ecf0f1; border-radius: 5px; }
                    .metric-value { font-size: 2em; font-weight: bold; color: #27ae60; }
                    .metric-label { color: #7f8c8d; margin-top: 5px; }
                    .status-good { color: #27ae60; }
                    .status-warning { color: #f39c12; }
                    .status-error { color: #e74c3c; }
                    button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #2980b9; }
                    .log-container { height: 300px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 888-888-88 量化交易系统</h1>
                        <p>生产级实盘交易管理平台</p>
                    </div>
                    
                    <div class="grid">
                        <div class="card">
                            <h3>📊 系统状态</h3>
                            <div id="system-status">
                                <div class="metric">
                                    <div class="metric-value status-good" id="system-health">运行中</div>
                                    <div class="metric-label">系统健康状态</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>💰 交易概览</h3>
                            <div id="trading-overview">
                                <div class="metric">
                                    <div class="metric-value" id="total-balance">$100,000</div>
                                    <div class="metric-label">总资产</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🛡️ 风险管理</h3>
                            <div id="risk-metrics">
                                <div class="metric">
                                    <div class="metric-value status-good" id="risk-level">低风险</div>
                                    <div class="metric-label">当前风险等级</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>🔧 快速操作</h3>
                            <button onclick="startSystem()">启动系统</button>
                            <button onclick="stopSystem()">停止系统</button>
                            <button onclick="refreshData()">刷新数据</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>📝 系统日志</h3>
                        <div class="log-container" id="system-logs">
                            <div>系统启动中...</div>
                        </div>
                    </div>
                </div>
                
                <script>
                    // WebSocket连接
                    const ws = new WebSocket('ws://localhost:8888/ws');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    function updateDashboard(data) {
                        if (data.type === 'system_status') {
                            document.getElementById('system-health').textContent = data.status;
                        } else if (data.type === 'log') {
                            const logContainer = document.getElementById('system-logs');
                            const logEntry = document.createElement('div');
                            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${data.message}`;
                            logContainer.appendChild(logEntry);
                            logContainer.scrollTop = logContainer.scrollHeight;
                        }
                    }
                    
                    function startSystem() {
                        fetch('/api/system/start', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => console.log('系统启动:', data));
                    }
                    
                    function stopSystem() {
                        fetch('/api/system/stop', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => console.log('系统停止:', data));
                    }
                    
                    function refreshData() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                console.log('状态更新:', data);
                                // 更新界面数据
                            });
                    }
                    
                    // 定期刷新数据
                    setInterval(refreshData, 30000);
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

