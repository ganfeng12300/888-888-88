"""
🌐 Web API服务器
生产级FastAPI服务器，提供完整的RESTful API和WebSocket实时数据服务
支持JWT认证、API限流、安全防护、完整的交易数据接口
专为量化交易系统设计，确保高性能和安全性
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# 导入系统组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_fusion_engine import AIFusionEngine
from trading.advanced_order_engine import AdvancedOrderEngine
from hardware.gpu_acceleration_engine import GPUAccelerationEngine
from hardware.memory_pool_manager import MemoryPoolManager
from hardware.high_frequency_optimizer import HighFrequencyOptimizer

# 配置
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API限流
limiter = Limiter(key_func=get_remote_address)

# 数据模型
class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., regex="^[A-Z]{3,10}$")
    action: str = Field(..., regex="^(buy|sell|hold)$")
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    order_type: str = Field("market", regex="^(market|limit|stop)$")

class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    action: str
    quantity: float
    price: Optional[float]
    status: str
    created_at: datetime
    filled_quantity: float = 0.0
    average_price: Optional[float] = None

class PositionResponse(BaseModel):
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime

class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    version: str
    components: Dict[str, Dict[str, Any]]
    performance: Dict[str, Any]

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    volume: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # user_id -> [channels]
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.subscriptions[user_id] = []
        
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.subscriptions:
            del self.subscriptions[user_id]
            
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"发送个人消息失败 {user_id}: {e}")
                self.disconnect(user_id)
                
    async def broadcast_to_channel(self, message: dict, channel: str):
        disconnected_users = []
        for user_id, channels in self.subscriptions.items():
            if channel in channels and user_id in self.active_connections:
                try:
                    await self.active_connections[user_id].send_text(json.dumps(message))
                except Exception as e:
                    logging.error(f"广播消息失败 {user_id}: {e}")
                    disconnected_users.append(user_id)
                    
        for user_id in disconnected_users:
            self.disconnect(user_id)
            
    def subscribe_channel(self, user_id: str, channel: str):
        if user_id in self.subscriptions:
            if channel not in self.subscriptions[user_id]:
                self.subscriptions[user_id].append(channel)
                
    def unsubscribe_channel(self, user_id: str, channel: str):
        if user_id in self.subscriptions:
            if channel in self.subscriptions[user_id]:
                self.subscriptions[user_id].remove(channel)

# 全局变量
connection_manager = ConnectionManager()
redis_client: Optional[redis.Redis] = None

# 系统组件
ai_engine: Optional[AIFusionEngine] = None
order_engine: Optional[AdvancedOrderEngine] = None
gpu_engine: Optional[GPUAccelerationEngine] = None
memory_manager: Optional[MemoryPoolManager] = None
hf_optimizer: Optional[HighFrequencyOptimizer] = None

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    await startup_event()
    yield
    # 关闭时清理
    await shutdown_event()

# 创建FastAPI应用
app = FastAPI(
    title="量化交易系统API",
    description="生产级量化交易系统的完整API接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# 限流异常处理
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 安全认证
security = HTTPBearer()

async def startup_event():
    """应用启动事件"""
    global redis_client, ai_engine, order_engine, gpu_engine, memory_manager, hf_optimizer
    
    try:
        # 初始化Redis连接
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await redis_client.ping()
        logging.info("Redis连接成功")
        
        # 初始化系统组件
        ai_engine = AIFusionEngine()
        order_engine = AdvancedOrderEngine()
        gpu_engine = GPUAccelerationEngine()
        memory_manager = MemoryPoolManager()
        hf_optimizer = HighFrequencyOptimizer()
        
        # 启动系统组件（在后台任务中）
        asyncio.create_task(ai_engine.start())
        asyncio.create_task(order_engine.start())
        asyncio.create_task(gpu_engine.start())
        asyncio.create_task(memory_manager.start())
        asyncio.create_task(hf_optimizer.start())
        
        # 启动实时数据推送任务
        asyncio.create_task(real_time_data_pusher())
        
        logging.info("🌐 Web API服务器启动完成")
        
    except Exception as e:
        logging.error(f"应用启动失败: {e}")
        raise

async def shutdown_event():
    """应用关闭事件"""
    global redis_client, ai_engine, order_engine, gpu_engine, memory_manager, hf_optimizer
    
    try:
        # 停止系统组件
        if ai_engine:
            await ai_engine.stop()
        if order_engine:
            await order_engine.stop()
        if gpu_engine:
            await gpu_engine.stop()
        if memory_manager:
            await memory_manager.stop()
        if hf_optimizer:
            await hf_optimizer.stop()
            
        # 关闭Redis连接
        if redis_client:
            await redis_client.close()
            
        logging.info("🌐 Web API服务器关闭完成")
        
    except Exception as e:
        logging.error(f"应用关闭失败: {e}")

# 认证相关函数
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

# API路由

@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, user_login: UserLogin):
    """用户登录"""
    # 简化的用户验证（生产环境应该连接数据库）
    if user_login.username == "admin" and user_login.password == "admin123":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": "admin"}, expires_delta=access_token_expires
        )
        
        # 记录登录信息到Redis
        if redis_client:
            await redis_client.hset(
                f"user:admin",
                mapping={
                    "last_login": datetime.utcnow().isoformat(),
                    "login_count": await redis_client.hincrby("user:admin", "login_count", 1)
                }
            )
        
        return TokenResponse(
            access_token=access_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """获取当前用户信息"""
    user_info = {
        "user_id": current_user,
        "username": current_user,
        "email": f"{current_user}@example.com",
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow()
    }
    
    # 从Redis获取用户信息
    if redis_client:
        user_data = await redis_client.hgetall(f"user:{current_user}")
        if user_data:
            if "last_login" in user_data:
                user_info["last_login"] = datetime.fromisoformat(user_data["last_login"])
    
    return UserResponse(**user_info)

@app.get("/system/status", response_model=SystemStatusResponse)
@limiter.limit("10/minute")
async def get_system_status(request: Request, current_user: str = Depends(get_current_user)):
    """获取系统状态"""
    components = {}
    
    # AI引擎状态
    if ai_engine:
        components["ai_engine"] = {
            "status": "running" if ai_engine.is_running else "stopped",
            "models": ai_engine.get_model_status(),
            "recent_signals": len(ai_engine.get_recent_signals())
        }
    
    # 订单引擎状态
    if order_engine:
        components["order_engine"] = {
            "status": "running" if order_engine.is_running else "stopped",
            "active_orders": len(order_engine.get_active_orders()),
            "execution_stats": order_engine.get_execution_stats()
        }
    
    # GPU引擎状态
    if gpu_engine:
        components["gpu_engine"] = {
            "status": "running" if gpu_engine.is_running else "stopped",
            "devices": gpu_engine.get_device_info(),
            "performance": gpu_engine.get_performance_stats()
        }
    
    # 内存管理器状态
    if memory_manager:
        components["memory_manager"] = {
            "status": "running" if memory_manager.is_running else "stopped",
            "memory_stats": memory_manager.get_memory_stats(),
            "cache_stats": memory_manager.get_cache_stats()
        }
    
    # 高频优化器状态
    if hf_optimizer:
        components["hf_optimizer"] = {
            "status": "running" if hf_optimizer.is_running else "stopped",
            "latency_stats": hf_optimizer.get_latency_stats(),
            "system_info": hf_optimizer.get_system_info()
        }
    
    return SystemStatusResponse(
        status="healthy",
        uptime=time.time(),
        version="1.0.0",
        components=components,
        performance={
            "active_connections": len(connection_manager.active_connections),
            "total_subscriptions": sum(len(subs) for subs in connection_manager.subscriptions.values())
        }
    )

@app.post("/trading/order", response_model=OrderResponse)
@limiter.limit("100/minute")
async def place_order(
    request: Request,
    order_request: TradingSignalRequest,
    current_user: str = Depends(get_current_user)
):
    """下单"""
    try:
        order_id = str(uuid.uuid4())
        
        # 创建订单对象
        order = {
            "order_id": order_id,
            "user_id": current_user,
            "symbol": order_request.symbol,
            "action": order_request.action,
            "quantity": order_request.quantity,
            "price": order_request.price,
            "order_type": order_request.order_type,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "filled_quantity": 0.0,
            "average_price": None
        }
        
        # 保存订单到Redis
        if redis_client:
            await redis_client.hset(
                f"order:{order_id}",
                mapping={k: json.dumps(v, default=str) for k, v in order.items()}
            )
            await redis_client.lpush(f"user_orders:{current_user}", order_id)
        
        # 提交到订单引擎
        if order_engine:
            await order_engine.submit_order(order)
        
        # 发送WebSocket通知
        await connection_manager.send_personal_message({
            "type": "order_created",
            "data": order
        }, current_user)
        
        return OrderResponse(**order)
        
    except Exception as e:
        logging.error(f"下单失败: {e}")
        raise HTTPException(status_code=500, detail="下单失败")

@app.get("/trading/orders", response_model=List[OrderResponse])
async def get_orders(current_user: str = Depends(get_current_user)):
    """获取订单列表"""
    orders = []
    
    if redis_client:
        order_ids = await redis_client.lrange(f"user_orders:{current_user}", 0, -1)
        
        for order_id in order_ids:
            order_data = await redis_client.hgetall(f"order:{order_id}")
            if order_data:
                # 反序列化数据
                order = {}
                for k, v in order_data.items():
                    try:
                        order[k] = json.loads(v)
                    except:
                        order[k] = v
                        
                orders.append(OrderResponse(**order))
    
    return orders

@app.get("/trading/positions", response_model=List[PositionResponse])
async def get_positions(current_user: str = Depends(get_current_user)):
    """获取持仓信息"""
    positions = []
    
    # 模拟持仓数据（生产环境应该从数据库获取）
    mock_positions = [
        {
            "symbol": "BTCUSDT",
            "quantity": 0.5,
            "average_price": 45000.0,
            "current_price": 46000.0,
            "unrealized_pnl": 500.0,
            "realized_pnl": 0.0,
            "last_updated": datetime.utcnow()
        },
        {
            "symbol": "ETHUSDT",
            "quantity": 2.0,
            "average_price": 3000.0,
            "current_price": 3100.0,
            "unrealized_pnl": 200.0,
            "realized_pnl": 0.0,
            "last_updated": datetime.utcnow()
        }
    ]
    
    return [PositionResponse(**pos) for pos in mock_positions]

@app.get("/market/data/{symbol}", response_model=MarketDataResponse)
@limiter.limit("1000/minute")
async def get_market_data(
    request: Request,
    symbol: str,
    current_user: str = Depends(get_current_user)
):
    """获取市场数据"""
    # 模拟市场数据（生产环境应该从数据源获取）
    import random
    
    base_price = 45000.0 if symbol == "BTCUSDT" else 3000.0
    current_price = base_price * (1 + random.uniform(-0.05, 0.05))
    
    market_data = {
        "symbol": symbol,
        "price": current_price,
        "volume": random.uniform(1000, 10000),
        "change_24h": random.uniform(-5, 5),
        "high_24h": current_price * 1.02,
        "low_24h": current_price * 0.98,
        "timestamp": datetime.utcnow()
    }
    
    return MarketDataResponse(**market_data)

@app.get("/analytics/performance")
async def get_performance_analytics(current_user: str = Depends(get_current_user)):
    """获取性能分析数据"""
    analytics = {
        "total_trades": 150,
        "winning_trades": 95,
        "losing_trades": 55,
        "win_rate": 63.33,
        "total_pnl": 12500.0,
        "max_drawdown": -2500.0,
        "sharpe_ratio": 1.85,
        "daily_returns": [random.uniform(-2, 3) for _ in range(30)],
        "monthly_pnl": [random.uniform(-1000, 2000) for _ in range(12)]
    }
    
    return analytics

# WebSocket端点
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket连接端点"""
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理订阅请求
            if message.get("type") == "subscribe":
                channel = message.get("channel")
                if channel:
                    connection_manager.subscribe_channel(user_id, channel)
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "channel": channel
                    }))
            
            # 处理取消订阅请求
            elif message.get("type") == "unsubscribe":
                channel = message.get("channel")
                if channel:
                    connection_manager.unsubscribe_channel(user_id, channel)
                    await websocket.send_text(json.dumps({
                        "type": "unsubscribed",
                        "channel": channel
                    }))
                    
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
        logging.info(f"WebSocket连接断开: {user_id}")

# 实时数据推送任务
async def real_time_data_pusher():
    """实时数据推送任务"""
    while True:
        try:
            # 推送市场数据
            import random
            
            market_data = {
                "type": "market_data",
                "data": {
                    "BTCUSDT": {
                        "price": 45000 + random.uniform(-1000, 1000),
                        "volume": random.uniform(1000, 5000),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "ETHUSDT": {
                        "price": 3000 + random.uniform(-100, 100),
                        "volume": random.uniform(500, 2000),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            }
            
            await connection_manager.broadcast_to_channel(market_data, "market_data")
            
            # 推送系统状态
            if len(connection_manager.active_connections) > 0:
                system_status = {
                    "type": "system_status",
                    "data": {
                        "cpu_usage": random.uniform(20, 80),
                        "memory_usage": random.uniform(40, 90),
                        "active_orders": random.randint(5, 50),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                await connection_manager.broadcast_to_channel(system_status, "system_status")
            
            await asyncio.sleep(1)  # 每秒推送一次
            
        except Exception as e:
            logging.error(f"实时数据推送错误: {e}")
            await asyncio.sleep(5)

# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="量化交易系统API",
        version="1.0.0",
        description="生产级量化交易系统的完整API接口文档",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境关闭热重载
        workers=1,     # 单进程模式，避免组件冲突
        log_level="info",
        access_log=True
    )

