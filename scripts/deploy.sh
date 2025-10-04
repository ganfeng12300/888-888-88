#!/bin/bash

# 量化交易系统一键部署脚本
# 生产级部署脚本，支持完整的环境检查、依赖安装、服务启动、健康检查等功能

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# 默认配置
DEFAULT_POSTGRES_PASSWORD="$(openssl rand -base64 32)"
DEFAULT_INFLUX_TOKEN="quant_trading_token_$(date +%s)"
DEFAULT_SECRET_KEY="$(openssl rand -hex 32)"

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 错误处理
error_exit() {
    log_error "$1"
    exit 1
}

# 显示横幅
show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    量化交易系统部署工具                        ║
║                  Quantitative Trading System                ║
║                        Deploy Script                        ║
╠══════════════════════════════════════════════════════════════╣
║  🚀 一键部署生产级量化交易系统                                 ║
║  📊 包含完整的监控、告警、日志系统                             ║
║  🔧 支持GPU加速、高频优化、实时数据处理                        ║
║  🌐 提供Web界面、API服务、WebSocket实时通信                   ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -h, --help              显示帮助信息
    -e, --env ENV           设置环境 (dev|test|prod) [默认: prod]
    -f, --force             强制重新部署
    -s, --skip-checks       跳过系统检查
    -d, --debug             启用调试模式
    -c, --config FILE       指定配置文件
    --no-gpu                禁用GPU支持
    --no-monitoring         禁用监控服务
    --quick                 快速部署（跳过非必要步骤）

示例:
    $0                      # 默认生产环境部署
    $0 -e dev -d            # 开发环境部署，启用调试
    $0 --force --no-gpu     # 强制重新部署，禁用GPU
    $0 --quick              # 快速部署

EOF
}

# 解析命令行参数
parse_args() {
    ENVIRONMENT="prod"
    FORCE_DEPLOY=false
    SKIP_CHECKS=false
    DEBUG=false
    CONFIG_FILE=""
    ENABLE_GPU=true
    ENABLE_MONITORING=true
    QUICK_DEPLOY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -s|--skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            -d|--debug)
                DEBUG=true
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --no-gpu)
                ENABLE_GPU=false
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --quick)
                QUICK_DEPLOY=true
                shift
                ;;
            *)
                error_exit "未知参数: $1"
                ;;
        esac
    done
    
    # 验证环境参数
    if [[ ! "$ENVIRONMENT" =~ ^(dev|test|prod)$ ]]; then
        error_exit "无效的环境参数: $ENVIRONMENT (支持: dev, test, prod)"
    fi
    
    export DEBUG
}

# 系统检查
check_system() {
    if [[ "$SKIP_CHECKS" == "true" ]]; then
        log_warn "跳过系统检查"
        return 0
    fi
    
    log_step "执行系统检查..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "操作系统: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "操作系统: macOS"
    else
        log_warn "未测试的操作系统: $OSTYPE"
    fi
    
    # 检查内存
    local total_mem
    if command -v free >/dev/null 2>&1; then
        total_mem=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $total_mem -lt 8 ]]; then
            log_warn "系统内存不足8GB，可能影响性能"
        else
            log_info "系统内存: ${total_mem}GB"
        fi
    fi
    
    # 检查磁盘空间
    local available_space
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $available_space -lt 20 ]]; then
        log_warn "磁盘可用空间不足20GB，可能影响运行"
    else
        log_info "磁盘可用空间: ${available_space}GB"
    fi
    
    # 检查CPU核心数
    local cpu_cores
    cpu_cores=$(nproc)
    if [[ $cpu_cores -lt 4 ]]; then
        log_warn "CPU核心数不足4个，可能影响性能"
    else
        log_info "CPU核心数: $cpu_cores"
    fi
    
    log_success "系统检查完成"
}

# 检查依赖
check_dependencies() {
    log_step "检查依赖软件..."
    
    local missing_deps=()
    
    # 检查Docker
    if ! command -v docker >/dev/null 2>&1; then
        missing_deps+=("docker")
    else
        local docker_version
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "Docker版本: $docker_version"
        
        # 检查Docker是否运行
        if ! docker info >/dev/null 2>&1; then
            error_exit "Docker未运行，请启动Docker服务"
        fi
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        missing_deps+=("docker-compose")
    else
        if command -v docker-compose >/dev/null 2>&1; then
            local compose_version
            compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
            log_info "Docker Compose版本: $compose_version"
        else
            local compose_version
            compose_version=$(docker compose version --short)
            log_info "Docker Compose版本: $compose_version"
        fi
    fi
    
    # 检查Git
    if ! command -v git >/dev/null 2>&1; then
        missing_deps+=("git")
    else
        local git_version
        git_version=$(git --version | cut -d' ' -f3)
        log_info "Git版本: $git_version"
    fi
    
    # 检查curl
    if ! command -v curl >/dev/null 2>&1; then
        missing_deps+=("curl")
    fi
    
    # 检查openssl
    if ! command -v openssl >/dev/null 2>&1; then
        missing_deps+=("openssl")
    fi
    
    # 检查GPU支持
    if [[ "$ENABLE_GPU" == "true" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_count
            gpu_count=$(nvidia-smi --list-gpus | wc -l)
            log_info "检测到 $gpu_count 个GPU设备"
            
            # 检查NVIDIA Docker支持
            if ! docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
                log_warn "NVIDIA Docker支持未正确配置"
            else
                log_info "NVIDIA Docker支持正常"
            fi
        else
            log_warn "未检测到NVIDIA GPU或驱动"
            ENABLE_GPU=false
        fi
    fi
    
    # 报告缺失的依赖
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "缺失以下依赖软件:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        error_exit "请安装缺失的依赖软件后重试"
    fi
    
    log_success "依赖检查完成"
}

# 创建环境配置文件
create_env_file() {
    log_step "创建环境配置文件..."
    
    if [[ -f "$ENV_FILE" && "$FORCE_DEPLOY" != "true" ]]; then
        log_info "环境配置文件已存在，跳过创建"
        return 0
    fi
    
    cat > "$ENV_FILE" << EOF
# 量化交易系统环境配置
# 生成时间: $(date)
# 环境: $ENVIRONMENT

# 应用配置
APP_ENV=$ENVIRONMENT
LOG_LEVEL=${LOG_LEVEL:-INFO}
SECRET_KEY=$DEFAULT_SECRET_KEY
DEBUG=${DEBUG:-false}

# 数据库配置
POSTGRES_DB=quant_trading
POSTGRES_USER=quant_user
POSTGRES_PASSWORD=$DEFAULT_POSTGRES_PASSWORD

# Redis配置
REDIS_PASSWORD=

# InfluxDB配置
INFLUX_USERNAME=admin
INFLUX_PASSWORD=admin123456
INFLUX_ORG=quant_trading
INFLUX_BUCKET=market_data
INFLUX_TOKEN=$DEFAULT_INFLUX_TOKEN

# Grafana配置
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin123456

# 交易所API配置（请填入真实的API密钥）
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
OKEX_API_KEY=
OKEX_SECRET_KEY=
OKEX_PASSPHRASE=

# GPU配置
ENABLE_GPU=$ENABLE_GPU
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 性能配置
OMP_NUM_THREADS=4
TASKSET_CPUS=0-3

# 监控配置
ENABLE_MONITORING=$ENABLE_MONITORING

EOF
    
    # 设置文件权限
    chmod 600 "$ENV_FILE"
    
    log_success "环境配置文件创建完成: $ENV_FILE"
}

# 创建必要的目录
create_directories() {
    log_step "创建必要的目录..."
    
    local dirs=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/logs/nginx"
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/ssl"
        "$PROJECT_ROOT/config/nginx/conf.d"
        "$PROJECT_ROOT/config/grafana/provisioning/datasources"
        "$PROJECT_ROOT/config/grafana/provisioning/dashboards"
        "$PROJECT_ROOT/config/grafana/dashboards"
        "$PROJECT_ROOT/config/prometheus/rules"
        "$PROJECT_ROOT/config/alertmanager"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "创建目录: $dir"
        fi
    done
    
    log_success "目录创建完成"
}

# 生成配置文件
generate_configs() {
    if [[ "$QUICK_DEPLOY" == "true" ]]; then
        log_info "快速部署模式，跳过配置文件生成"
        return 0
    fi
    
    log_step "生成配置文件..."
    
    # 生成Nginx配置
    generate_nginx_config
    
    # 生成Prometheus配置
    generate_prometheus_config
    
    # 生成AlertManager配置
    generate_alertmanager_config
    
    # 生成Grafana配置
    generate_grafana_config
    
    log_success "配置文件生成完成"
}

# 生成Nginx配置
generate_nginx_config() {
    local nginx_conf="$PROJECT_ROOT/config/nginx/nginx.conf"
    
    cat > "$nginx_conf" << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    upstream api_backend {
        server quant_core:8000;
        keepalive 32;
    }
    
    upstream frontend_backend {
        server quant_frontend:80;
        keepalive 32;
    }
    
    server {
        listen 80;
        server_name _;
        
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        location /ws/ {
            proxy_pass http://quant_core:8001/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location / {
            proxy_pass http://frontend_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF
    
    log_debug "生成Nginx配置: $nginx_conf"
}

# 生成Prometheus配置
generate_prometheus_config() {
    local prometheus_conf="$PROJECT_ROOT/config/prometheus/prometheus.yml"
    
    cat > "$prometheus_conf" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'quant-core'
    static_configs:
      - targets: ['quant_core:8002']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node_exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    
    log_debug "生成Prometheus配置: $prometheus_conf"
}

# 生成AlertManager配置
generate_alertmanager_config() {
    local alertmanager_conf="$PROJECT_ROOT/config/alertmanager/alertmanager.yml"
    
    cat > "$alertmanager_conf" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@quant-trading.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://quant_core:8000/api/alerts/webhook'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
    
    log_debug "生成AlertManager配置: $alertmanager_conf"
}

# 生成Grafana配置
generate_grafana_config() {
    local datasource_conf="$PROJECT_ROOT/config/grafana/provisioning/datasources/datasources.yml"
    
    cat > "$datasource_conf" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    database: quant_trading
    user: admin
    password: admin123456
EOF
    
    log_debug "生成Grafana数据源配置: $datasource_conf"
}

# 拉取Docker镜像
pull_images() {
    log_step "拉取Docker镜像..."
    
    # 使用docker-compose拉取镜像
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f "$COMPOSE_FILE" pull
    else
        docker compose -f "$COMPOSE_FILE" pull
    fi
    
    log_success "Docker镜像拉取完成"
}

# 构建自定义镜像
build_images() {
    log_step "构建自定义镜像..."
    
    # 构建核心服务镜像
    docker build -f "$PROJECT_ROOT/Dockerfile.core" -t quant-core:latest "$PROJECT_ROOT"
    
    # 构建前端镜像
    if [[ -f "$PROJECT_ROOT/frontend/Dockerfile" ]]; then
        docker build -f "$PROJECT_ROOT/frontend/Dockerfile" -t quant-frontend:latest "$PROJECT_ROOT/frontend"
    fi
    
    # 构建数据采集镜像
    if [[ -f "$PROJECT_ROOT/Dockerfile.collector" ]]; then
        docker build -f "$PROJECT_ROOT/Dockerfile.collector" -t quant-collector:latest "$PROJECT_ROOT"
    fi
    
    log_success "自定义镜像构建完成"
}

# 启动服务
start_services() {
    log_step "启动服务..."
    
    # 停止现有服务（如果存在）
    if [[ "$FORCE_DEPLOY" == "true" ]]; then
        log_info "强制重新部署，停止现有服务..."
        stop_services
    fi
    
    # 启动服务
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    fi
    
    log_success "服务启动完成"
}

# 停止服务
stop_services() {
    log_info "停止现有服务..."
    
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    else
        docker compose -f "$COMPOSE_FILE" down --remove-orphans
    fi
}

# 健康检查
health_check() {
    log_step "执行健康检查..."
    
    local services=(
        "http://localhost:8000/health:核心服务"
        "http://localhost:3001:前端服务"
        "http://localhost:3000:Grafana"
        "http://localhost:9090:Prometheus"
    )
    
    local max_attempts=30
    local attempt=1
    
    for service_info in "${services[@]}"; do
        local url="${service_info%%:*}"
        local name="${service_info##*:}"
        
        log_info "检查 $name ($url)..."
        
        attempt=1
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f -s "$url" >/dev/null 2>&1; then
                log_success "$name 健康检查通过"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                log_warn "$name 健康检查失败"
            else
                log_debug "$name 健康检查失败，重试 $attempt/$max_attempts"
                sleep 5
            fi
            
            ((attempt++))
        done
    done
    
    log_success "健康检查完成"
}

# 显示部署信息
show_deployment_info() {
    log_step "部署信息"
    
    echo -e "${GREEN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                      部署完成！                               ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    echo "🌐 服务访问地址:"
    echo "  • 主页面:        http://localhost"
    echo "  • API文档:       http://localhost/api/docs"
    echo "  • 核心服务:      http://localhost:8000"
    echo "  • 前端界面:      http://localhost:3001"
    echo "  • Grafana监控:   http://localhost:3000 (admin/admin123456)"
    echo "  • Prometheus:    http://localhost:9090"
    echo "  • AlertManager:  http://localhost:9093"
    echo ""
    echo "📊 监控信息:"
    echo "  • 系统监控:      http://localhost:9100/metrics"
    echo "  • 容器监控:      http://localhost:8080"
    echo "  • Redis监控:     redis://localhost:6379"
    echo "  • 数据库监控:    postgresql://localhost:5432"
    echo ""
    echo "🔧 管理命令:"
    echo "  • 查看日志:      docker-compose logs -f [service]"
    echo "  • 重启服务:      docker-compose restart [service]"
    echo "  • 停止服务:      docker-compose down"
    echo "  • 查看状态:      docker-compose ps"
    echo ""
    echo "📁 重要文件:"
    echo "  • 环境配置:      $ENV_FILE"
    echo "  • 日志目录:      $PROJECT_ROOT/logs"
    echo "  • 数据目录:      $PROJECT_ROOT/data"
    echo ""
    
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        echo -e "${YELLOW}⚠️  生产环境提醒:${NC}"
        echo "  • 请修改默认密码"
        echo "  • 配置SSL证书"
        echo "  • 设置防火墙规则"
        echo "  • 配置备份策略"
        echo "  • 填入真实的交易所API密钥"
        echo ""
    fi
}

# 主函数
main() {
    show_banner
    parse_args "$@"
    
    log_info "开始部署量化交易系统..."
    log_info "环境: $ENVIRONMENT"
    log_info "项目目录: $PROJECT_ROOT"
    
    # 执行部署步骤
    check_system
    check_dependencies
    create_directories
    create_env_file
    generate_configs
    
    if [[ "$QUICK_DEPLOY" != "true" ]]; then
        pull_images
        build_images
    fi
    
    start_services
    health_check
    show_deployment_info
    
    log_success "🎉 量化交易系统部署完成！"
}

# 执行主函数
main "$@"

