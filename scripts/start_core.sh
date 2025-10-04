#!/bin/bash

# 量化交易核心服务启动脚本
# 生产级启动脚本，包含完整的初始化、健康检查、故障恢复等功能

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# 错误处理
error_exit() {
    log_error "$1"
    exit 1
}

# 信号处理
cleanup() {
    log_info "接收到终止信号，正在优雅关闭..."
    
    # 停止所有后台进程
    if [[ -n "${CORE_PID:-}" ]]; then
        log_info "停止核心服务进程 (PID: $CORE_PID)"
        kill -TERM "$CORE_PID" 2>/dev/null || true
        wait "$CORE_PID" 2>/dev/null || true
    fi
    
    if [[ -n "${WS_PID:-}" ]]; then
        log_info "停止WebSocket服务进程 (PID: $WS_PID)"
        kill -TERM "$WS_PID" 2>/dev/null || true
        wait "$WS_PID" 2>/dev/null || true
    fi
    
    if [[ -n "${MONITOR_PID:-}" ]]; then
        log_info "停止监控服务进程 (PID: $MONITOR_PID)"
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    
    log_info "所有服务已停止"
    exit 0
}

# 设置信号处理
trap cleanup SIGTERM SIGINT SIGQUIT

# 环境变量验证
validate_environment() {
    log_info "验证环境变量..."
    
    # 必需的环境变量
    local required_vars=(
        "POSTGRES_HOST"
        "POSTGRES_PORT"
        "POSTGRES_DB"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "REDIS_HOST"
        "REDIS_PORT"
        "INFLUX_HOST"
        "INFLUX_PORT"
        "INFLUX_ORG"
        "INFLUX_BUCKET"
        "INFLUX_TOKEN"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error_exit "环境变量 $var 未设置"
        fi
    done
    
    # 设置默认值
    export APP_ENV="${APP_ENV:-production}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"
    export PYTHONPATH="/app/src:${PYTHONPATH:-}"
    
    log_info "环境变量验证完成"
}

# 等待依赖服务
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-60}
    
    log_info "等待 $service_name 服务启动 ($host:$port)..."
    
    local count=0
    while ! nc -z "$host" "$port" >/dev/null 2>&1; do
        if [[ $count -ge $timeout ]]; then
            error_exit "$service_name 服务启动超时"
        fi
        sleep 1
        ((count++))
    done
    
    log_info "$service_name 服务已就绪"
}

# 数据库初始化
init_database() {
    log_info "初始化数据库..."
    
    # 等待PostgreSQL
    wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT" "PostgreSQL" 120
    
    # 运行数据库迁移
    python -c "
import sys
sys.path.append('/app/src')
from database.migrations import run_migrations
run_migrations()
" || error_exit "数据库初始化失败"
    
    log_info "数据库初始化完成"
}

# Redis初始化
init_redis() {
    log_info "初始化Redis..."
    
    # 等待Redis
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis" 60
    
    # 测试Redis连接
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null || error_exit "Redis连接失败"
    
    log_info "Redis初始化完成"
}

# InfluxDB初始化
init_influxdb() {
    log_info "初始化InfluxDB..."
    
    # 等待InfluxDB
    wait_for_service "$INFLUX_HOST" "$INFLUX_PORT" "InfluxDB" 120
    
    # 创建bucket（如果不存在）
    python -c "
import sys
sys.path.append('/app/src')
from database.influx_client import init_influxdb
init_influxdb()
" || error_exit "InfluxDB初始化失败"
    
    log_info "InfluxDB初始化完成"
}

# 系统优化
optimize_system() {
    log_info "应用系统优化..."
    
    # 设置CPU亲和性（如果支持）
    if command -v taskset >/dev/null 2>&1; then
        log_debug "设置CPU亲和性"
        # 将进程绑定到特定CPU核心
        export TASKSET_CPUS="${TASKSET_CPUS:-0-3}"
    fi
    
    # 设置内存优化
    export MALLOC_ARENA_MAX=4
    export MALLOC_MMAP_THRESHOLD_=131072
    export MALLOC_TRIM_THRESHOLD_=131072
    export MALLOC_TOP_PAD_=131072
    export MALLOC_MMAP_MAX_=65536
    
    # 设置Python优化
    export PYTHONHASHSEED=0
    export PYTHONOPTIMIZE=1
    
    # 设置OpenMP线程数
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    
    # 设置CUDA优化
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_CACHE_DISABLE=0
        export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
    fi
    
    log_info "系统优化完成"
}

# GPU检查
check_gpu() {
    log_info "检查GPU可用性..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        log_info "检测到 $gpu_count 个GPU设备"
        
        # 显示GPU信息
        nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r name memory_total memory_free temp util; do
            log_info "GPU: $name, 内存: ${memory_free}MB/${memory_total}MB, 温度: ${temp}°C, 利用率: ${util}%"
        done
    else
        log_warn "未检测到GPU设备或nvidia-smi不可用"
    fi
}

# 启动核心服务
start_core_service() {
    log_info "启动量化交易核心服务..."
    
    cd /app
    
    # 启动FastAPI服务
    if command -v taskset >/dev/null 2>&1 && [[ -n "${TASKSET_CPUS:-}" ]]; then
        taskset -c "$TASKSET_CPUS" python -m uvicorn src.web.api_server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 \
            --loop uvloop \
            --http httptools \
            --log-level info \
            --access-log \
            --use-colors &
    else
        python -m uvicorn src.web.api_server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 1 \
            --loop uvloop \
            --http httptools \
            --log-level info \
            --access-log \
            --use-colors &
    fi
    
    CORE_PID=$!
    log_info "核心服务已启动 (PID: $CORE_PID)"
}

# 启动WebSocket服务
start_websocket_service() {
    log_info "启动WebSocket服务..."
    
    cd /app
    
    # 启动WebSocket服务
    python -c "
import sys
import asyncio
sys.path.append('/app/src')
from web.websocket_server import start_websocket_server
asyncio.run(start_websocket_server())
" &
    
    WS_PID=$!
    log_info "WebSocket服务已启动 (PID: $WS_PID)"
}

# 启动监控服务
start_monitor_service() {
    log_info "启动监控服务..."
    
    cd /app
    
    # 启动监控服务
    python -c "
import sys
import asyncio
sys.path.append('/app/src')
from monitoring.monitor_server import start_monitor_server
asyncio.run(start_monitor_server())
" &
    
    MONITOR_PID=$!
    log_info "监控服务已启动 (PID: $MONITOR_PID)"
}

# 健康检查
health_check() {
    local max_attempts=30
    local attempt=1
    
    log_info "执行健康检查..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
            log_info "健康检查通过"
            return 0
        fi
        
        log_debug "健康检查失败，重试 $attempt/$max_attempts"
        sleep 2
        ((attempt++))
    done
    
    error_exit "健康检查失败，服务启动异常"
}

# 监控进程状态
monitor_processes() {
    log_info "开始监控进程状态..."
    
    while true; do
        # 检查核心服务
        if ! kill -0 "$CORE_PID" 2>/dev/null; then
            log_error "核心服务进程异常退出"
            cleanup
        fi
        
        # 检查WebSocket服务
        if ! kill -0 "$WS_PID" 2>/dev/null; then
            log_error "WebSocket服务进程异常退出"
            cleanup
        fi
        
        # 检查监控服务
        if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
            log_error "监控服务进程异常退出"
            cleanup
        fi
        
        # 检查服务健康状态
        if ! curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
            log_warn "服务健康检查失败"
        fi
        
        sleep 30
    done
}

# 主函数
main() {
    log_info "🚀 启动量化交易核心服务..."
    log_info "环境: ${APP_ENV}"
    log_info "日志级别: ${LOG_LEVEL}"
    
    # 验证环境
    validate_environment
    
    # 系统优化
    optimize_system
    
    # GPU检查
    check_gpu
    
    # 初始化依赖服务
    init_redis
    init_database
    init_influxdb
    
    # 启动服务
    start_core_service
    start_websocket_service
    start_monitor_service
    
    # 健康检查
    health_check
    
    log_info "✅ 所有服务启动完成"
    log_info "核心服务: http://localhost:8000"
    log_info "WebSocket: ws://localhost:8001"
    log_info "监控服务: http://localhost:8002"
    
    # 监控进程
    monitor_processes
}

# 执行主函数
main "$@"

