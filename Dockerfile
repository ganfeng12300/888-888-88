# 🚀 AI量化交易系统 - 生产级Docker镜像
# 支持20核CPU + RTX3060 + 128GB内存 + 1TB NVMe硬件优化

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TZ=Asia/Shanghai

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    iotop \
    nvidia-utils-525 \
    cpufrequtils \
    lm-sensors \
    smartmontools \
    iftop \
    nethogs \
    sysstat \
    redis-tools \
    postgresql-client \
    sqlite3 \
    lz4 \
    zstd \
    unzip \
    tar \
    gzip \
    rsync \
    cron \
    supervisor \
    nginx \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    libta-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建Python 3.11软链接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 升级pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装基础Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装PyTorch with CUDA支持
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装量化交易相关库
RUN pip install \
    ccxt==4.1.* \
    websocket-client==1.6.* \
    python-binance==1.0.* \
    ta-lib==0.4.* \
    pandas-ta==0.3.* \
    vectorbt==0.25.* \
    backtrader==1.9.* \
    zipline-reloaded==3.0.* \
    alpaca-trade-api==3.0.* \
    yfinance==0.2.*

# 安装AI/ML库
RUN pip install \
    scikit-learn==1.3.* \
    xgboost==2.0.* \
    lightgbm==4.1.* \
    catboost==1.2.* \
    stable-baselines3==2.2.* \
    gymnasium==0.29.* \
    transformers==4.35.* \
    datasets==2.14.* \
    accelerate==0.24.*

# 安装数据处理库
RUN pip install \
    numpy==1.24.* \
    pandas==2.1.* \
    polars==0.19.* \
    pyarrow==14.0.* \
    dask==2023.10.* \
    numba==0.58.* \
    cython==3.0.*

# 安装监控和系统库
RUN pip install \
    psutil==5.9.* \
    pynvml==11.5.* \
    GPUtil==1.4.* \
    prometheus-client==0.18.* \
    grafana-api==1.0.* \
    loguru==0.7.* \
    rich==13.6.* \
    typer==0.9.*

# 安装数据库连接库
RUN pip install \
    redis==5.0.* \
    clickhouse-driver==0.2.* \
    psycopg2-binary==2.9.* \
    sqlalchemy==2.0.* \
    alembic==1.12.*

# 安装消息队列库
RUN pip install \
    kafka-python==2.0.* \
    celery==5.3.* \
    kombu==5.3.*

# 安装Web框架
RUN pip install \
    fastapi==0.104.* \
    uvicorn==0.24.* \
    websockets==12.0.* \
    aiohttp==3.9.* \
    httpx==0.25.*

# 安装压缩和序列化库
RUN pip install \
    lz4==4.3.* \
    zstandard==0.22.* \
    msgpack==1.0.* \
    orjson==3.9.* \
    pickle5==0.0.*

# 安装调度和任务库
RUN pip install \
    schedule==1.2.* \
    apscheduler==3.10.* \
    asyncio-mqtt==0.16.*

# 创建应用目录结构
RUN mkdir -p /app/src \
    /app/config \
    /app/data \
    /app/logs \
    /app/models \
    /app/temp \
    /app/scripts \
    /app/tests \
    /app/web

# 创建数据目录结构
RUN mkdir -p /app/data/system \
    /app/data/realtime \
    /app/data/models \
    /app/data/historical \
    /app/data/logs \
    /app/data/temp

# 复制应用代码
COPY . /app/

# 设置Python路径
ENV PYTHONPATH=/app:$PYTHONPATH

# 设置权限
RUN chmod -R 755 /app
RUN chmod +x /app/start.py

# 创建非root用户
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 暴露端口
EXPOSE 8000 8001 8002 9090 3000

# 启动命令
CMD ["python3", "start.py"]
