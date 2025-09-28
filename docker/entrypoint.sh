#!/bin/bash
# 🚀 Docker容器入口点脚本
# AI量化交易系统 - 生产级容器启动脚本

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
    if [[ "${TRADING_LOG_LEVEL:-INFO}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# 错误处理
handle_error() {
    local exit_code=$?
    log_error "脚本执行失败，退出码: $exit_code"
    exit $exit_code
}

trap handle_error ERR

# 显示启动信息
show_banner() {
    cat << 'EOF'
    
    ╔══════════════════════════════════════════════════════════════╗
    ║                   🚀 AI量化交易系统                          ║
    ║                Production Trading System                     ║
    ║                                                              ║
    ║  🎯 高频交易策略引擎                                          ║
    ║  📊 市场微观结构分析                                          ║
    ║  🔧 系统集成管理                                              ║
    ║  🏥 健康监控系统                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
EOF
}

# 环境变量验证
validate_environment() {
    log_info "验证环境变量..."
    
    # 必需的环境变量
    local required_vars=(
        "TRADING_ENV"
        "TRADING_CONFIG_DIR"
        "TRADING_DATA_DIR"
        "TRADING_LOGS_DIR"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "必需的环境变量未设置: $var"
            exit 1
        fi
        log_debug "$var = ${!var}"
    done
    
    # 设置默认值
    export TRADING_LOG_LEVEL="${TRADING_LOG_LEVEL:-INFO}"
    export TRADING_DEBUG="${TRADING_DEBUG:-false}"
    export TRADING_CPU_LIMIT="${TRADING_CPU_LIMIT:-16}"
    export TRADING_MEMORY_LIMIT="${TRADING_MEMORY_LIMIT:-107374182400}"
    
    log_info "环境变量验证完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    local dirs=(
        "$TRADING_CONFIG_DIR"
        "$TRADING_DATA_DIR"
        "$TRADING_LOGS_DIR"
        "/app/tmp"
        "/app/data/market_data"
        "/app/data/backtest"
        "/app/data/live_trading"
        "/app/logs/application"
        "/app/logs/trading"
        "/app/logs/system"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "创建目录: $dir"
        fi
    done
    
    # 设置权限
    chmod 755 "$TRADING_CONFIG_DIR" "$TRADING_DATA_DIR" "$TRADING_LOGS_DIR"
    
    log_info "目录创建完成"
}

# 配置文件初始化
initialize_config() {
    log_info "初始化配置文件..."
    
    local config_file="$TRADING_CONFIG_DIR/app.json"
    
    # 如果配置文件不存在，创建默认配置
    if [[ ! -f "$config_file" ]]; then
        log_info "创建默认配置文件: $config_file"
        
        cat > "$config_file" << EOF
{
    "environment": "${TRADING_ENV}",
    "log_level": "${TRADING_LOG_LEVEL}",
    "debug": ${TRADING_DEBUG},
    "system": {
        "cpu_limit": ${TRADING_CPU_LIMIT},
        "memory_limit": ${TRADING_MEMORY_LIMIT},
        "health_check_interval": 30,
        "startup_timeout": 300
    },
    "redis": {
        "url": "redis://redis:6379/0",
        "max_connections": 100,
        "retry_on_timeout": true
    },
    "database": {
        "url": "postgresql://trading:trading@postgres:5432/trading_db",
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30
    },
    "trading": {
        "enabled": true,
        "max_position": 1000.0,
        "risk_limit": 10000.0,
        "strategy_timeout": 60
    },
    "monitoring": {
        "prometheus_port": 8001,
        "health_port": 8002,
        "metrics_interval": 10
    }
}
EOF
    else
        log_info "使用现有配置文件: $config_file"
    fi
    
    # 验证配置文件格式
    if ! python -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
        log_error "配置文件格式错误: $config_file"
        exit 1
    fi
    
    log_info "配置文件初始化完成"
}

# 系统资源检查
check_system_resources() {
    log_info "检查系统资源..."
    
    # 检查CPU核心数
    local cpu_cores=$(nproc)
    log_info "可用CPU核心数: $cpu_cores"
    
    if [[ $cpu_cores -lt 4 ]]; then
        log_warn "CPU核心数较少 ($cpu_cores)，建议至少4核"
    fi
    
    # 检查内存
    local memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local memory_gb=$((memory_kb / 1024 / 1024))
    log_info "可用内存: ${memory_gb}GB"
    
    if [[ $memory_gb -lt 8 ]]; then
        log_warn "内存较少 (${memory_gb}GB)，建议至少8GB"
    fi
    
    # 检查磁盘空间
    local disk_usage=$(df -h /app | tail -1 | awk '{print $5}' | sed 's/%//')
    log_info "磁盘使用率: ${disk_usage}%"
    
    if [[ $disk_usage -gt 80 ]]; then
        log_warn "磁盘使用率较高 (${disk_usage}%)，建议清理空间"
    fi
    
    log_info "系统资源检查完成"
}

# 依赖服务检查
check_dependencies() {
    log_info "检查依赖服务..."
    
    # 检查Redis连接
    if [[ -n "${REDIS_URL:-}" ]]; then
        log_info "检查Redis连接..."
        local redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
        local redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\).*/\1/p')
        
        if command -v redis-cli >/dev/null 2>&1; then
            if timeout 5 redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
                log_info "Redis连接正常"
            else
                log_warn "Redis连接失败，将使用内存消息总线"
            fi
        else
            log_debug "redis-cli未安装，跳过Redis连接检查"
        fi
    fi
    
    # 检查PostgreSQL连接
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log_info "检查PostgreSQL连接..."
        if command -v pg_isready >/dev/null 2>&1; then
            local db_host=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\).*/\1/p')
            local db_port=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
            
            if timeout 5 pg_isready -h "$db_host" -p "$db_port" >/dev/null 2>&1; then
                log_info "PostgreSQL连接正常"
            else
                log_warn "PostgreSQL连接失败"
            fi
        else
            log_debug "pg_isready未安装，跳过PostgreSQL连接检查"
        fi
    fi
    
    log_info "依赖服务检查完成"
}

# Python环境检查
check_python_environment() {
    log_info "检查Python环境..."
    
    # 检查Python版本
    local python_version=$(python --version 2>&1)
    log_info "Python版本: $python_version"
    
    # 检查关键依赖包
    local packages=(
        "asyncio"
        "numpy"
        "pandas"
        "redis"
        "psutil"
        "loguru"
    )
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            log_debug "包 $package 可用"
        else
            log_error "关键包 $package 不可用"
            exit 1
        fi
    done
    
    # 检查应用模块
    if python -c "import src" 2>/dev/null; then
        log_info "应用模块导入成功"
    else
        log_error "应用模块导入失败"
        exit 1
    fi
    
    log_info "Python环境检查完成"
}

# 设置信号处理
setup_signal_handlers() {
    log_info "设置信号处理器..."
    
    # 优雅关闭处理
    graceful_shutdown() {
        log_info "接收到关闭信号，开始优雅关闭..."
        
        # 如果有子进程，发送TERM信号
        if [[ -n "${APP_PID:-}" ]]; then
            log_info "向应用进程发送TERM信号: $APP_PID"
            kill -TERM "$APP_PID" 2>/dev/null || true
            
            # 等待进程结束
            local count=0
            while kill -0 "$APP_PID" 2>/dev/null && [[ $count -lt 30 ]]; do
                sleep 1
                ((count++))
            done
            
            # 如果进程仍在运行，强制终止
            if kill -0 "$APP_PID" 2>/dev/null; then
                log_warn "强制终止应用进程: $APP_PID"
                kill -KILL "$APP_PID" 2>/dev/null || true
            fi
        fi
        
        log_info "优雅关闭完成"
        exit 0
    }
    
    # 注册信号处理器
    trap graceful_shutdown SIGTERM SIGINT
    
    log_info "信号处理器设置完成"
}

# 启动应用
start_application() {
    log_info "启动应用程序..."
    
    # 设置Python路径
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # 启动应用
    if [[ $# -eq 0 ]]; then
        # 默认启动命令
        log_info "使用默认启动命令"
        exec python -m src.main
    else
        # 自定义命令
        log_info "使用自定义命令: $*"
        exec "$@"
    fi
}

# 主函数
main() {
    show_banner
    
    log_info "开始容器初始化..."
    
    # 执行初始化步骤
    validate_environment
    create_directories
    initialize_config
    check_system_resources
    check_dependencies
    check_python_environment
    setup_signal_handlers
    
    log_info "容器初始化完成，启动应用..."
    
    # 启动应用
    start_application "$@"
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
