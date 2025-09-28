# 🐳 生产级多阶段Docker构建
# AI量化交易系统 - 优化构建+安全配置+资源限制

# ==================== 构建阶段 ====================
FROM python:3.11-slim as builder

# 设置构建参数
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# 添加标签
LABEL maintainer="AI Trading System <trading@example.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="ai-trading-system" \
      org.label-schema.description="Production-grade AI quantitative trading system" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/ganfeng12300/888-888-88" \
      org.label-schema.schema-version="1.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装Poetry
RUN pip install poetry==1.6.1

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml poetry.lock ./

# 安装Python依赖
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# ==================== 运行时阶段 ====================
FROM python:3.11-slim as runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRADING_ENV=production \
    TRADING_LOG_LEVEL=INFO \
    TRADING_CONFIG_DIR=/app/config \
    TRADING_DATA_DIR=/app/data \
    TRADING_LOGS_DIR=/app/logs

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    dumb-init \
    gosu \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非root用户
RUN groupadd -r trading && useradd -r -g trading -d /app -s /bin/bash trading

# 创建必要目录
RUN mkdir -p /app/config /app/data /app/logs /app/tmp \
    && chown -R trading:trading /app

# 从构建阶段复制虚拟环境
COPY --from=builder --chown=trading:trading /app/.venv /app/.venv

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=trading:trading . .

# 设置权限
RUN chmod +x docker/entrypoint.sh \
    && chmod +x docker/healthcheck.sh

# 创建配置文件模板
RUN mkdir -p config/production && \
    echo '{}' > config/production/app.json && \
    chown -R trading:trading config/

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["./docker/healthcheck.sh"]

# 暴露端口
EXPOSE 8000 8001 8002

# 设置资源限制
# CPU: 最多使用16核 (保留4核给系统)
# Memory: 最多使用100GB (保留28GB给系统)
ENV TRADING_CPU_LIMIT=16 \
    TRADING_MEMORY_LIMIT=107374182400

# 数据卷
VOLUME ["/app/data", "/app/logs", "/app/config"]

# 入口点
ENTRYPOINT ["dumb-init", "--", "./docker/entrypoint.sh"]

# 默认命令
CMD ["python", "-m", "src.main"]

# ==================== 开发阶段 ====================
FROM runtime as development

# 安装开发依赖
USER root
COPY --from=builder /app/.venv /app/.venv
RUN /app/.venv/bin/pip install pytest pytest-cov pytest-asyncio black isort mypy

# 设置开发环境变量
ENV TRADING_ENV=development \
    TRADING_LOG_LEVEL=DEBUG

# 切换回非root用户
USER trading

# ==================== 测试阶段 ====================
FROM development as testing

# 复制测试文件
COPY --chown=trading:trading tests/ tests/

# 运行测试
RUN python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ==================== 生产阶段 ====================
FROM runtime as production

# 确保使用非root用户
USER trading

# 设置生产环境变量
ENV TRADING_ENV=production \
    TRADING_LOG_LEVEL=INFO \
    TRADING_DEBUG=false

# 最终检查
RUN python -c "import sys; print(f'Python version: {sys.version}')" && \
    python -c "import src; print('Application imports successful')"
