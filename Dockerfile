# 使用NVIDIA CUDA基础镜像支持GPU加速
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

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
    libffi-dev \
    liblzma-dev \
    # TA-Lib依赖
    libta-dev \
    # 时区设置
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建符号链接
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# 升级pip
RUN python -m pip install --upgrade pip

# 安装Poetry
RUN pip install poetry==1.7.1

# 配置Poetry
RUN poetry config virtualenvs.create false

# 复制依赖文件
COPY pyproject.toml poetry.lock* ./

# 安装Python依赖
RUN poetry install --no-dev --no-interaction --no-ansi

# 安装TA-Lib Python包
RUN pip install TA-Lib

# 创建必要的目录
RUN mkdir -p /app/src /app/config /app/logs /app/models /app/data /app/web

# 复制应用代码
COPY src/ /app/src/
COPY config/ /app/config/
COPY web/ /app/web/
COPY start.py /app/

# 设置权限
RUN chmod +x /app/start.py

# 创建非root用户
RUN useradd -m -u 1000 quantuser && chown -R quantuser:quantuser /app
USER quantuser

# 暴露端口
EXPOSE 8000 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "start.py"]
