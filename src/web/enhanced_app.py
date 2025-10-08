#!/usr/bin/env python3
"""
🌐 888-888-88 增强Web应用
Production-Grade Web Management Interface
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# 导入系统组件
from src.config.api_config import api_config_manager
from src.core.error_handling_system import error_handler
from src.monitoring.system_monitor import system_monitor
from src.ai.ai_model_manager import ai_model_manager
from src.ai.ai_performance_monitor import ai_performance_monitor
from src.ai.enhanced_ai_fusion_engine import enhanced_ai_fusion_engine

# 创建FastAPI应用
app = FastAPI(
    title="888-888-88 量化交易系统",
    description="生产级AI量化交易系统Web管理界面",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件和模板
static_dir = Path("src/web/static")
templates_dir = Path("src/web/templates")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    templates = None

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# 数据模型
class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str
    uptime: float
    components: Dict[str, Any]
    timestamp: str

class ConfigUpdate(BaseModel):
    """配置更新模型"""
    config_type: str
    config_data: Dict[str, Any]

class TradeRequest(BaseModel):
    """交易请求模型"""
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        logger.info("🚀 启动增强Web应用...")
        
        # 初始化配置管理器
        await api_config_manager.initialize()
        
        # 启动系统监控
        await system_monitor.start_monitoring()
        
        # 初始化AI组件
        await ai_model_manager.initialize()
        await enhanced_ai_fusion_engine.initialize()
        
        logger.info("✅ 增强Web应用启动完成")
        
    except Exception as e:
        logger.error(f"❌ Web应用启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    try:
        logger.info("🛑 关闭增强Web应用...")
        
        # 关闭AI组件
        await enhanced_ai_fusion_engine.shutdown()
        await ai_model_manager.shutdown()
        
        logger.info("✅ 增强Web应用已关闭")
        
    except Exception as e:
        logger.error(f"❌ Web应用关闭失败: {e}")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """主仪表板"""
    try:
        if not templates:
            return HTMLResponse(content=get_default_dashboard_html())
        
        # 获取系统状态
        system_status = await get_comprehensive_system_status()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "system_status": system_status,
            "page_title": "888-888-88 量化交易系统"
        })
        
    except Exception as e:
        logger.error(f"❌ 仪表板加载失败: {e}")
        return HTMLResponse(content=get_error_page_html(str(e)))

def get_default_dashboard_html() -> str:
    """获取默认仪表板HTML"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>888-888-88 量化交易系统</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
            .status-item { text-align: center; }
            .status-value { font-size: 2em; font-weight: bold; color: #27ae60; }
            .status-label { color: #7f8c8d; margin-top: 5px; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #2980b9; }
            .loading { text-align: center; padding: 40px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 888-888-88 量化交易系统</h1>
                <p>生产级AI量化交易系统 - Web管理界面</p>
            </div>
            
            <div class="card">
                <h2>📊 系统状态</h2>
                <div id="system-status" class="loading">正在加载系统状态...</div>
            </div>
            
            <div class="card">
                <h2>🔧 快速操作</h2>
                <button class="btn" onclick="refreshStatus()">刷新状态</button>
                <button class="btn" onclick="viewLogs()">查看日志</button>
                <button class="btn" onclick="viewConfig()">系统配置</button>
            </div>
        </div>
        
        <script>
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/system/status');
                    const data = await response.json();
                    displaySystemStatus(data);
                } catch (error) {
                    document.getElementById('system-status').innerHTML = '❌ 加载失败: ' + error.message;
                }
            }
            
            function displaySystemStatus(data) {
                const statusHtml = `
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="status-value">${data.status}</div>
                            <div class="status-label">系统状态</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">${Object.keys(data.components || {}).length}</div>
                            <div class="status-label">活跃组件</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value">${new Date(data.timestamp).toLocaleString()}</div>
                            <div class="status-label">最后更新</div>
                        </div>
                    </div>
                `;
                document.getElementById('system-status').innerHTML = statusHtml;
            }
            
            function refreshStatus() {
                loadSystemStatus();
            }
            
            function viewLogs() {
                window.open('/api/logs', '_blank');
            }
            
            function viewConfig() {
                window.open('/api/config', '_blank');
            }
            
            // 页面加载时获取状态
            loadSystemStatus();
            
            // 每30秒自动刷新
            setInterval(loadSystemStatus, 30000);
        </script>
    </body>
    </html>
    """

def get_error_page_html(error_message: str) -> str:
    """获取错误页面HTML"""
    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>系统错误 - 888-888-88</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; text-align: center; }}
            .error-card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .error-icon {{ font-size: 4em; color: #e74c3c; margin-bottom: 20px; }}
            .error-title {{ font-size: 2em; color: #2c3e50; margin-bottom: 20px; }}
            .error-message {{ color: #7f8c8d; margin-bottom: 30px; }}
            .btn {{ background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="error-card">
                <div class="error-icon">❌</div>
                <div class="error-title">系统错误</div>
                <div class="error-message">{error_message}</div>
                <a href="/" class="btn">返回首页</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点"""
    await manager.connect(websocket)
    try:
        while True:
            # 发送实时状态更新
            status = await get_comprehensive_system_status()
            await websocket.send_json(status)
            await asyncio.sleep(5)  # 每5秒发送一次更新
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态API"""
    try:
        status_data = await get_comprehensive_system_status()
        return JSONResponse(content=status_data)
        
    except Exception as e:
        logger.error(f"❌ 获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_comprehensive_system_status() -> Dict[str, Any]:
    """获取综合系统状态"""
    try:
        status_data = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "uptime": 0,
            "components": {},
            "performance": {},
            "alerts": []
        }
        
        # 系统监控状态
        try:
            monitor_status = system_monitor.get_current_status()
            status_data["components"]["system_monitor"] = {
                "status": "active",
                "cpu_usage": monitor_status.get("cpu_percent", 0),
                "memory_usage": monitor_status.get("memory_percent", 0),
                "disk_usage": monitor_status.get("disk_percent", 0),
                "network_io": monitor_status.get("network_io", {}),
                "process_count": monitor_status.get("process_count", 0)
            }
        except Exception as e:
            status_data["components"]["system_monitor"] = {"status": "error", "error": str(e)}
        
        # AI模型管理器状态
        try:
            ai_manager_stats = ai_model_manager.get_manager_stats()
            status_data["components"]["ai_model_manager"] = {
                "status": "active",
                "loaded_models": ai_manager_stats.get("loaded_models", 0),
                "total_models": ai_manager_stats.get("total_models", 0),
                "memory_usage_mb": ai_manager_stats.get("memory_usage_mb", 0),
                "total_predictions": ai_manager_stats.get("total_predictions", 0)
            }
        except Exception as e:
            status_data["components"]["ai_model_manager"] = {"status": "error", "error": str(e)}
        
        # AI性能监控器状态
        try:
            perf_stats = await ai_performance_monitor.get_overall_performance()
            status_data["components"]["ai_performance_monitor"] = {
                "status": "active",
                "monitored_models": len(perf_stats.get("models", {})),
                "avg_accuracy": perf_stats.get("average_accuracy", 0),
                "total_predictions": perf_stats.get("total_predictions", 0),
                "avg_processing_time": perf_stats.get("avg_processing_time_ms", 0)
            }
        except Exception as e:
            status_data["components"]["ai_performance_monitor"] = {"status": "error", "error": str(e)}
        
        # AI融合引擎状态
        try:
            fusion_stats = enhanced_ai_fusion_engine.get_engine_stats()
            status_data["components"]["ai_fusion_engine"] = {
                "status": "active" if fusion_stats.get("is_running", False) else "inactive",
                "total_signals": fusion_stats.get("total_signals_processed", 0),
                "total_decisions": fusion_stats.get("total_decisions", 0),
                "success_rate": fusion_stats.get("success_rate", 0),
                "monitored_symbols": fusion_stats.get("monitored_symbols", 0)
            }
        except Exception as e:
            status_data["components"]["ai_fusion_engine"] = {"status": "error", "error": str(e)}
        
        # 配置管理器状态
        try:
            config_summary = api_config_manager.get_config_summary()
            status_data["components"]["config_manager"] = {
                "status": "active",
                "exchanges_configured": len(config_summary.get("exchanges", {})),
                "trading_config": config_summary.get("trading", {}),
                "ai_config": config_summary.get("ai", {}),
                "monitoring_config": config_summary.get("monitoring", {})
            }
        except Exception as e:
            status_data["components"]["config_manager"] = {"status": "error", "error": str(e)}
        
        # 错误处理系统状态
        try:
            error_stats = error_handler.get_error_statistics()
            status_data["components"]["error_handler"] = {
                "status": "active",
                "total_errors": error_stats.get("total_errors", 0),
                "critical_errors": error_stats.get("critical_errors", 0),
                "recovery_attempts": error_stats.get("recovery_attempts", 0),
                "success_rate": error_stats.get("recovery_success_rate", 0)
            }
        except Exception as e:
            status_data["components"]["error_handler"] = {"status": "error", "error": str(e)}
        
        # 计算整体健康度
        active_components = sum(1 for comp in status_data["components"].values() if comp.get("status") == "active")
        total_components = len(status_data["components"])
        health_score = (active_components / total_components * 100) if total_components > 0 else 0
        
        status_data["health_score"] = health_score
        status_data["overall_status"] = "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical"
        
        return status_data
        
    except Exception as e:
        logger.error(f"❌ 获取综合系统状态失败: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {}
        }

@app.get("/api/config")
async def get_config():
    """获取系统配置API"""
    try:
        config_summary = api_config_manager.get_config_summary()
        return JSONResponse(content=config_summary)
        
    except Exception as e:
        logger.error(f"❌ 获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    """更新系统配置API"""
    try:
        config_type = config_update.config_type
        config_data = config_update.config_data
        
        if config_type == "trading":
            await api_config_manager.update_trading_config(**config_data)
        elif config_type == "ai":
            await api_config_manager.update_ai_config(**config_data)
        else:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        return JSONResponse(content={"status": "success", "message": "配置更新成功"})
        
    except Exception as e:
        logger.error(f"❌ 更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs():
    """获取系统日志API"""
    try:
        log_files = []
        logs_dir = Path("logs")
        
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        recent_lines = lines[-100:] if len(lines) > 100 else lines
                    
                    log_files.append({
                        "filename": log_file.name,
                        "size": log_file.stat().st_size,
                        "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                        "recent_content": "".join(recent_lines)
                    })
                except Exception as e:
                    logger.error(f"读取日志文件失败 {log_file}: {e}")
        
        return JSONResponse(content={"logs": log_files})
        
    except Exception as e:
        logger.error(f"❌ 获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/models")
async def get_ai_models():
    """获取AI模型信息API"""
    try:
        models_info = ai_model_manager.get_all_models_info()
        return JSONResponse(content={"models": models_info})
        
    except Exception as e:
        logger.error(f"❌ 获取AI模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/performance")
async def get_ai_performance():
    """获取AI性能数据API"""
    try:
        performance_data = await ai_performance_monitor.get_overall_performance()
        return JSONResponse(content=performance_data)
        
    except Exception as e:
        logger.error(f"❌ 获取AI性能数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.web.enhanced_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
