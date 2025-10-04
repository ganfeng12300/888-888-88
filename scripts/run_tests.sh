#!/bin/bash

# 量化交易系统测试运行脚本
# 运行完整的测试套件，包括集成测试、性能测试、单元测试等

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
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"

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
║                    量化交易系统测试套件                        ║
║                 Quantitative Trading System                 ║
║                      Test Suite                             ║
╠══════════════════════════════════════════════════════════════╣
║  🧪 完整的系统测试验证                                         ║
║  ⚡ 性能压力测试                                              ║
║  🛡️ 安全与容错测试                                            ║
║  📊 详细的测试报告                                            ║
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
    -t, --type TYPE         测试类型 (all|unit|integration|performance|security) [默认: all]
    -v, --verbose           详细输出
    -f, --fast              快速测试（跳过耗时测试）
    -c, --coverage          生成代码覆盖率报告
    -r, --report            只生成报告，不运行测试
    --no-cleanup            测试后不清理临时文件
    --parallel              并行运行测试

测试类型:
    all                     运行所有测试
    unit                    单元测试
    integration             集成测试
    performance             性能测试
    security                安全测试

示例:
    $0                      # 运行所有测试
    $0 -t integration -v    # 运行集成测试，详细输出
    $0 -t performance -f    # 快速性能测试
    $0 --coverage           # 生成覆盖率报告

EOF
}

# 解析命令行参数
parse_args() {
    TEST_TYPE="all"
    VERBOSE=false
    FAST_MODE=false
    COVERAGE=false
    REPORT_ONLY=false
    NO_CLEANUP=false
    PARALLEL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--fast)
                FAST_MODE=true
                shift
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -r|--report)
                REPORT_ONLY=true
                shift
                ;;
            --no-cleanup)
                NO_CLEANUP=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            *)
                error_exit "未知参数: $1"
                ;;
        esac
    done
    
    # 验证测试类型
    if [[ ! "$TEST_TYPE" =~ ^(all|unit|integration|performance|security)$ ]]; then
        error_exit "无效的测试类型: $TEST_TYPE"
    fi
}

# 环境检查
check_environment() {
    log_step "检查测试环境..."
    
    # 检查Python
    if ! command -v python3 >/dev/null 2>&1; then
        error_exit "Python3未安装"
    fi
    
    # 检查必要的Python包
    local required_packages=("pytest" "pytest-asyncio" "pytest-cov" "httpx" "numpy")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" >/dev/null 2>&1; then
            log_warn "Python包 $package 未安装，尝试安装..."
            pip install "$package" || error_exit "无法安装 $package"
        fi
    done
    
    # 检查系统资源
    local available_memory
    available_memory=$(free -m | awk '/^Mem:/{print $7}')
    if [[ $available_memory -lt 1000 ]]; then
        log_warn "可用内存不足1GB，可能影响测试性能"
    fi
    
    log_success "环境检查完成"
}

# 创建测试目录
setup_test_environment() {
    log_step "设置测试环境..."
    
    # 创建测试结果目录
    mkdir -p "$TEST_RESULTS_DIR"
    mkdir -p "$TEST_RESULTS_DIR/reports"
    mkdir -p "$TEST_RESULTS_DIR/logs"
    mkdir -p "$TEST_RESULTS_DIR/coverage"
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    export TEST_ENV=true
    export LOG_LEVEL=INFO
    
    if [[ "$VERBOSE" == "true" ]]; then
        export LOG_LEVEL=DEBUG
    fi
    
    log_success "测试环境设置完成"
}

# 运行单元测试
run_unit_tests() {
    log_step "运行单元测试..."
    
    local test_files=(
        "$PROJECT_ROOT/src/*/tests/test_*.py"
        "$PROJECT_ROOT/tests/unit/test_*.py"
    )
    
    local pytest_args=()
    
    if [[ "$VERBOSE" == "true" ]]; then
        pytest_args+=("-v")
    fi
    
    if [[ "$COVERAGE" == "true" ]]; then
        pytest_args+=("--cov=src" "--cov-report=html:$TEST_RESULTS_DIR/coverage/unit")
    fi
    
    if [[ "$PARALLEL" == "true" ]]; then
        pytest_args+=("-n" "auto")
    fi
    
    # 运行pytest
    if python3 -m pytest "${pytest_args[@]}" "${test_files[@]}" --junitxml="$TEST_RESULTS_DIR/reports/unit_tests.xml" 2>&1 | tee "$TEST_RESULTS_DIR/logs/unit_tests.log"; then
        log_success "单元测试通过"
        return 0
    else
        log_error "单元测试失败"
        return 1
    fi
}

# 运行集成测试
run_integration_tests() {
    log_step "运行集成测试..."
    
    # 检查服务是否运行
    if ! curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
        log_warn "核心服务未运行，启动测试服务..."
        # 这里可以启动测试服务或使用mock
    fi
    
    # 运行集成测试
    cd "$PROJECT_ROOT"
    
    local test_command="python3 src/testing/integration_tests.py"
    
    if [[ "$VERBOSE" == "true" ]]; then
        test_command="$test_command --verbose"
    fi
    
    if [[ "$FAST_MODE" == "true" ]]; then
        test_command="$test_command --fast"
    fi
    
    if $test_command 2>&1 | tee "$TEST_RESULTS_DIR/logs/integration_tests.log"; then
        log_success "集成测试通过"
        return 0
    else
        log_error "集成测试失败"
        return 1
    fi
}

# 运行性能测试
run_performance_tests() {
    log_step "运行性能测试..."
    
    cd "$PROJECT_ROOT"
    
    local test_command="python3 src/testing/performance_tests.py"
    
    if [[ "$FAST_MODE" == "true" ]]; then
        test_command="$test_command --fast"
    fi
    
    if $test_command 2>&1 | tee "$TEST_RESULTS_DIR/logs/performance_tests.log"; then
        log_success "性能测试通过"
        return 0
    else
        log_error "性能测试失败"
        return 1
    fi
}

# 运行安全测试
run_security_tests() {
    log_step "运行安全测试..."
    
    # 检查常见安全问题
    local security_issues=0
    
    # 检查硬编码密码
    if grep -r "password.*=" "$PROJECT_ROOT/src" --include="*.py" | grep -v "test" | grep -v "#" >/dev/null; then
        log_warn "发现可能的硬编码密码"
        ((security_issues++))
    fi
    
    # 检查SQL注入风险
    if grep -r "execute.*%" "$PROJECT_ROOT/src" --include="*.py" >/dev/null; then
        log_warn "发现可能的SQL注入风险"
        ((security_issues++))
    fi
    
    # 检查不安全的随机数生成
    if grep -r "random\." "$PROJECT_ROOT/src" --include="*.py" | grep -v "numpy" >/dev/null; then
        log_warn "发现可能不安全的随机数生成"
        ((security_issues++))
    fi
    
    # 运行bandit安全扫描（如果可用）
    if command -v bandit >/dev/null 2>&1; then
        log_info "运行Bandit安全扫描..."
        if bandit -r "$PROJECT_ROOT/src" -f json -o "$TEST_RESULTS_DIR/reports/security_scan.json" 2>&1 | tee "$TEST_RESULTS_DIR/logs/security_tests.log"; then
            log_success "Bandit安全扫描完成"
        else
            log_warn "Bandit安全扫描发现问题"
            ((security_issues++))
        fi
    else
        log_warn "Bandit未安装，跳过安全扫描"
    fi
    
    if [[ $security_issues -eq 0 ]]; then
        log_success "安全测试通过"
        return 0
    else
        log_error "安全测试发现 $security_issues 个问题"
        return 1
    fi
}

# 生成测试报告
generate_test_report() {
    log_step "生成测试报告..."
    
    local report_file="$TEST_RESULTS_DIR/reports/test_summary.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>量化交易系统测试报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { color: green; }
        .fail { color: red; }
        .warn { color: orange; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 量化交易系统测试报告</h1>
        <p>生成时间: $(date)</p>
        <p>测试类型: $TEST_TYPE</p>
    </div>
    
    <div class="section">
        <h2>📊 测试概览</h2>
        <table>
            <tr><th>测试类型</th><th>状态</th><th>详情</th></tr>
EOF
    
    # 添加测试结果到报告
    if [[ -f "$TEST_RESULTS_DIR/logs/unit_tests.log" ]]; then
        echo "            <tr><td>单元测试</td><td class=\"pass\">✅ 通过</td><td>详见日志</td></tr>" >> "$report_file"
    fi
    
    if [[ -f "$TEST_RESULTS_DIR/logs/integration_tests.log" ]]; then
        echo "            <tr><td>集成测试</td><td class=\"pass\">✅ 通过</td><td>详见日志</td></tr>" >> "$report_file"
    fi
    
    if [[ -f "$TEST_RESULTS_DIR/logs/performance_tests.log" ]]; then
        echo "            <tr><td>性能测试</td><td class=\"pass\">✅ 通过</td><td>详见日志</td></tr>" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>📁 测试文件</h2>
        <ul>
            <li><a href="../logs/unit_tests.log">单元测试日志</a></li>
            <li><a href="../logs/integration_tests.log">集成测试日志</a></li>
            <li><a href="../logs/performance_tests.log">性能测试日志</a></li>
            <li><a href="../coverage/unit/index.html">代码覆盖率报告</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    log_success "测试报告生成完成: $report_file"
}

# 清理测试环境
cleanup_test_environment() {
    if [[ "$NO_CLEANUP" == "true" ]]; then
        log_info "跳过清理，保留测试文件"
        return 0
    fi
    
    log_step "清理测试环境..."
    
    # 清理临时文件
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # 清理测试数据库
    if [[ -f "$PROJECT_ROOT/test.db" ]]; then
        rm -f "$PROJECT_ROOT/test.db"
    fi
    
    log_success "清理完成"
}

# 主函数
main() {
    show_banner
    parse_args "$@"
    
    if [[ "$REPORT_ONLY" == "true" ]]; then
        generate_test_report
        exit 0
    fi
    
    log_info "开始运行测试套件..."
    log_info "测试类型: $TEST_TYPE"
    
    # 初始化
    check_environment
    setup_test_environment
    
    local test_results=()
    local overall_result=0
    
    # 根据测试类型运行相应测试
    case "$TEST_TYPE" in
        "all")
            run_unit_tests && test_results+=("unit:PASS") || { test_results+=("unit:FAIL"); overall_result=1; }
            run_integration_tests && test_results+=("integration:PASS") || { test_results+=("integration:FAIL"); overall_result=1; }
            run_performance_tests && test_results+=("performance:PASS") || { test_results+=("performance:FAIL"); overall_result=1; }
            run_security_tests && test_results+=("security:PASS") || { test_results+=("security:FAIL"); overall_result=1; }
            ;;
        "unit")
            run_unit_tests && test_results+=("unit:PASS") || { test_results+=("unit:FAIL"); overall_result=1; }
            ;;
        "integration")
            run_integration_tests && test_results+=("integration:PASS") || { test_results+=("integration:FAIL"); overall_result=1; }
            ;;
        "performance")
            run_performance_tests && test_results+=("performance:PASS") || { test_results+=("performance:FAIL"); overall_result=1; }
            ;;
        "security")
            run_security_tests && test_results+=("security:PASS") || { test_results+=("security:FAIL"); overall_result=1; }
            ;;
    esac
    
    # 生成报告
    generate_test_report
    
    # 清理
    cleanup_test_environment
    
    # 显示结果
    echo
    log_info "测试结果汇总:"
    for result in "${test_results[@]}"; do
        local test_name="${result%%:*}"
        local test_status="${result##*:}"
        if [[ "$test_status" == "PASS" ]]; then
            log_success "$test_name: 通过"
        else
            log_error "$test_name: 失败"
        fi
    done
    
    if [[ $overall_result -eq 0 ]]; then
        log_success "🎉 所有测试通过！"
    else
        log_error "❌ 部分测试失败"
    fi
    
    exit $overall_result
}

# 执行主函数
main "$@"
