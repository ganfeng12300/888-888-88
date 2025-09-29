#!/bin/bash
# AI量化交易系统一键启动脚本
# 自动检查环境、安装依赖、启动系统

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 显示启动横幅
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          🚀 AI量化交易系统 v2.0 Pro                          ║"
    echo "║                                                                              ║"
    echo "║  🎯 专为交易所带单设计的生产级AI量化交易系统                                  ║"
    echo "║  💰 目标收益: 周收益20%+ | 最大日回撤≤3%                                     ║"
    echo "║  🧠 多AI融合: 强化学习+深度学习+集成学习+专家系统+元学习+迁移学习             ║"
    echo "║  🔧 硬件优化: 20核CPU + RTX3060 12GB + 128GB内存 + 1TB NVMe                ║"
    echo "║  🌐 实时监控: 黑金科技风格Web界面 + 全方位系统监控                           ║"
    echo "║                                                                              ║"
    echo "║  📊 代码规模: 12,600+行生产级代码 | 100%实盘交易标准                        ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_step "检查系统要求..."
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_info "Python版本: $PYTHON_VERSION"
        
        # 检查Python版本是否>=3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_info "✅ Python版本满足要求 (>=3.8)"
        else
            log_error "❌ Python版本过低，需要Python 3.8或更高版本"
            exit 1
        fi
    else
        log_error "❌ 未找到Python3，请先安装Python"
        exit 1
    fi
    
    # 检查pip
    if command -v pip3 &> /dev/null; then
        log_info "✅ pip3已安装"
    else
        log_error "❌ 未找到pip3，请先安装pip"
        exit 1
    fi
    
    # 检查Git
    if command -v git &> /dev/null; then
        log_info "✅ Git已安装"
    else
        log_warn "⚠️ 未找到Git，某些功能可能受限"
    fi
    
    # 检查系统资源
    log_info "检查系统资源..."
    
    # 检查内存
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        log_info "系统内存: ${TOTAL_MEM}GB"
        if [ "$TOTAL_MEM" -lt 8 ]; then
            log_warn "⚠️ 系统内存较少，建议至少8GB内存"
        fi
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -h . | awk 'NR==2{print $4}')
    log_info "可用磁盘空间: $DISK_SPACE"
    
    # 检查NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        log_info "✅ 检测到GPU: $GPU_INFO"
    else
        log_warn "⚠️ 未检测到NVIDIA GPU或驱动，AI训练性能可能受限"
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    log_step "创建Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
        log_info "✅ 虚拟环境创建完成"
    else
        log_info "✅ 虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    log_info "激活虚拟环境..."
    source venv/bin/activate
    
    # 升级pip
    log_info "升级pip..."
    pip install --upgrade pip
}

# 安装依赖
install_dependencies() {
    log_step "安装Python依赖包..."
    
    if [ -f "requirements.txt" ]; then
        log_info "安装requirements.txt中的依赖..."
        pip install -r requirements.txt
        log_info "✅ 依赖安装完成"
    else
        log_error "❌ 未找到requirements.txt文件"
        exit 1
    fi
    
    # 安装TA-Lib (如果需要)
    log_info "检查TA-Lib安装..."
    if python3 -c "import talib" 2>/dev/null; then
        log_info "✅ TA-Lib已安装"
    else
        log_warn "⚠️ TA-Lib未安装，尝试安装..."
        
        # 根据系统类型安装TA-Lib
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y libta-lib-dev
            elif command -v yum &> /dev/null; then
                sudo yum install -y ta-lib-devel
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install ta-lib
            fi
        fi
        
        pip install TA-Lib
    fi
}

# 创建必要目录
create_directories() {
    log_step "创建必要目录..."
    
    directories=(
        "logs"
        "data"
        "models"
        "backups"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "✅ 创建目录: $dir"
        fi
    done
}

# 检查配置文件
check_configuration() {
    log_step "检查配置文件..."
    
    # 检查主程序文件
    if [ -f "main.py" ]; then
        log_info "✅ 主程序文件存在"
    else
        log_error "❌ 未找到main.py文件"
        exit 1
    fi
    
    # 检查Web界面文件
    if [ -f "web/app.py" ]; then
        log_info "✅ Web界面文件存在"
    else
        log_error "❌ 未找到web/app.py文件"
        exit 1
    fi
    
    # 检查源代码目录
    if [ -d "src" ]; then
        log_info "✅ 源代码目录存在"
    else
        log_error "❌ 未找到src目录"
        exit 1
    fi
}

# 设置环境变量
setup_environment() {
    log_step "设置环境变量..."
    
    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 设置CUDA环境变量（如果有GPU）
    if command -v nvidia-smi &> /dev/null; then
        export CUDA_VISIBLE_DEVICES=0
        log_info "✅ 设置CUDA环境变量"
    fi
    
    # 设置时区为中国时区
    export TZ='Asia/Shanghai'
    log_info "✅ 设置时区为中国时区"
}

# 启动系统
start_system() {
    log_step "启动AI量化交易系统..."
    
    # 检查端口是否被占用
    PORT=8080
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "⚠️ 端口$PORT已被占用，尝试使用其他端口"
        PORT=8081
    fi
    
    log_info "🚀 启动系统主程序..."
    log_info "📊 Web界面将在 http://localhost:$PORT 启动"
    log_info "💡 请保持终端窗口打开，系统正在运行..."
    log_info "🛑 按 Ctrl+C 可以安全停止系统"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 AI量化交易系统启动中...${NC}"
    echo -e "${GREEN}🌐 Web监控界面: http://localhost:$PORT${NC}"
    echo -e "${GREEN}📊 实时监控: 硬件性能 | AI训练 | 交易绩效 | 系统健康${NC}"
    echo -e "${GREEN}💰 交易目标: 周收益20%+ | 最大日回撤≤3%${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # 启动主程序
    python3 main.py
}

# 清理函数
cleanup() {
    log_info "🔄 正在清理资源..."
    
    # 杀死可能的后台进程
    pkill -f "python3 main.py" 2>/dev/null || true
    pkill -f "web/app.py" 2>/dev/null || true
    
    log_info "✅ 清理完成"
}

# 错误处理
handle_error() {
    log_error "❌ 启动过程中发生错误，正在清理..."
    cleanup
    exit 1
}

# 信号处理
trap cleanup EXIT
trap handle_error ERR

# 主函数
main() {
    # 显示启动横幅
    show_banner
    
    # 检查系统要求
    check_system_requirements
    
    # 创建虚拟环境
    create_virtual_environment
    
    # 安装依赖
    install_dependencies
    
    # 创建必要目录
    create_directories
    
    # 检查配置文件
    check_configuration
    
    # 设置环境变量
    setup_environment
    
    # 启动系统
    start_system
}

# 检查是否以root权限运行
if [ "$EUID" -eq 0 ]; then
    log_warn "⚠️ 不建议以root权限运行此脚本"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 运行主函数
main "$@"

