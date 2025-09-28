#!/bin/bash
# 🏥 Docker健康检查脚本
# AI量化交易系统 - 容器健康状态检查

set -euo pipefail

# 配置
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_PORT=${TRADING_HEALTH_PORT:-8002}
HEALTH_CHECK_URL="http://localhost:${HEALTH_CHECK_PORT}/health"

# 日志函数
log_info() {
    echo "[HEALTH] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[HEALTH] $(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1" >&2
}

# 检查HTTP健康端点
check_http_health() {
    if command -v curl >/dev/null 2>&1; then
        # 使用curl检查
        if curl -f -s --max-time "$HEALTH_CHECK_TIMEOUT" "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        # 使用wget检查
        if wget -q --timeout="$HEALTH_CHECK_TIMEOUT" --tries=1 -O /dev/null "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        log_error "curl和wget都不可用，无法检查HTTP健康端点"
        return 1
    fi
}

# 检查进程状态
check_process_health() {
    # 检查Python主进程是否运行
    if pgrep -f "python.*src.main" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 检查端口监听
check_port_health() {
    local port=$1
    
    if command -v netstat >/dev/null 2>&1; then
        if netstat -ln | grep ":${port} " >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -ln | grep ":${port} " >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        # 尝试连接端口
        if timeout 3 bash -c "</dev/tcp/localhost/${port}" >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
}

# 检查系统资源
check_system_resources() {
    # 检查内存使用率
    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [[ $memory_usage -gt 95 ]]; then
        log_error "内存使用率过高: ${memory_usage}%"
        return 1
    fi
    
    # 检查磁盘使用率
    local disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 95 ]]; then
        log_error "磁盘使用率过高: ${disk_usage}%"
        return 1
    fi
    
    return 0
}

# 检查关键文件
check_critical_files() {
    local files=(
        "/app/config/app.json"
        "/app/src/main.py"
    )
    
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "关键文件不存在: $file"
            return 1
        fi
    done
    
    return 0
}

# 主健康检查函数
main_health_check() {
    local checks_passed=0
    local total_checks=0
    
    # 检查1: HTTP健康端点
    ((total_checks++))
    if check_http_health; then
        log_info "✓ HTTP健康端点检查通过"
        ((checks_passed++))
    else
        log_error "✗ HTTP健康端点检查失败"
    fi
    
    # 检查2: 进程状态
    ((total_checks++))
    if check_process_health; then
        log_info "✓ 进程状态检查通过"
        ((checks_passed++))
    else
        log_error "✗ 进程状态检查失败"
    fi
    
    # 检查3: 端口监听
    ((total_checks++))
    if check_port_health "$HEALTH_CHECK_PORT"; then
        log_info "✓ 端口监听检查通过"
        ((checks_passed++))
    else
        log_error "✗ 端口监听检查失败"
    fi
    
    # 检查4: 系统资源
    ((total_checks++))
    if check_system_resources; then
        log_info "✓ 系统资源检查通过"
        ((checks_passed++))
    else
        log_error "✗ 系统资源检查失败"
    fi
    
    # 检查5: 关键文件
    ((total_checks++))
    if check_critical_files; then
        log_info "✓ 关键文件检查通过"
        ((checks_passed++))
    else
        log_error "✗ 关键文件检查失败"
    fi
    
    # 评估健康状态
    local health_percentage=$((checks_passed * 100 / total_checks))
    
    if [[ $health_percentage -ge 80 ]]; then
        log_info "健康检查通过 (${checks_passed}/${total_checks}, ${health_percentage}%)"
        exit 0
    else
        log_error "健康检查失败 (${checks_passed}/${total_checks}, ${health_percentage}%)"
        exit 1
    fi
}

# 快速健康检查（用于频繁检查）
quick_health_check() {
    # 只检查最关键的指标
    if check_process_health && check_port_health "$HEALTH_CHECK_PORT"; then
        log_info "快速健康检查通过"
        exit 0
    else
        log_error "快速健康检查失败"
        exit 1
    fi
}

# 详细健康检查（用于诊断）
detailed_health_check() {
    log_info "开始详细健康检查..."
    
    # 显示系统信息
    log_info "系统信息:"
    log_info "  - 运行时间: $(uptime)"
    log_info "  - 内存使用: $(free -h | grep Mem)"
    log_info "  - 磁盘使用: $(df -h /app | tail -1)"
    log_info "  - CPU负载: $(cat /proc/loadavg)"
    
    # 显示进程信息
    log_info "进程信息:"
    if pgrep -f "python.*src.main" >/dev/null 2>&1; then
        local pid=$(pgrep -f "python.*src.main")
        log_info "  - 主进程PID: $pid"
        log_info "  - 进程状态: $(ps -p $pid -o pid,ppid,state,pcpu,pmem,etime,cmd --no-headers)"
    else
        log_error "  - 主进程未运行"
    fi
    
    # 显示网络信息
    log_info "网络信息:"
    if command -v netstat >/dev/null 2>&1; then
        log_info "  - 监听端口: $(netstat -ln | grep LISTEN | head -5)"
    elif command -v ss >/dev/null 2>&1; then
        log_info "  - 监听端口: $(ss -ln | grep LISTEN | head -5)"
    fi
    
    # 执行主健康检查
    main_health_check
}

# 解析命令行参数
case "${1:-main}" in
    "main"|"")
        main_health_check
        ;;
    "quick")
        quick_health_check
        ;;
    "detailed")
        detailed_health_check
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [main|quick|detailed|help]"
        echo "  main     - 标准健康检查 (默认)"
        echo "  quick    - 快速健康检查"
        echo "  detailed - 详细健康检查"
        echo "  help     - 显示帮助信息"
        exit 0
        ;;
    *)
        log_error "未知参数: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac
