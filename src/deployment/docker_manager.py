#!/usr/bin/env python3
"""
🐳 Docker管理器 - 生产级容器化部署系统
Docker Manager - Production-Grade Containerized Deployment System

生产级特性：
- 自动化容器部署
- 多环境配置管理
- 服务健康检查
- 滚动更新部署
- 容器监控告警
"""

import docker
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os
import yaml

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

@dataclass
class ContainerConfig:
    """容器配置"""
    name: str
    image: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: Dict[str, str]
    restart_policy: str = "unless-stopped"
    memory_limit: str = "1g"
    cpu_limit: float = 1.0

@dataclass
class ServiceStatus:
    """服务状态"""
    name: str
    status: str
    container_id: str
    image: str
    created_at: datetime
    ports: List[str]
    health: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

class DockerManager:
    """Docker管理器主类"""
    
    def __init__(self):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "DockerManager")
        
        try:
            self.client = docker.from_env()
            self.logger.info("Docker客户端连接成功")
        except Exception as e:
            self.logger.error(f"Docker客户端连接失败: {e}")
            self.client = None
        
        self.services = {}
        self.monitoring_thread = None
        self._monitoring = False
        
        # 预定义服务配置
        self._setup_service_configs()
        
        self.logger.info("Docker管理器初始化完成")
    
    def _setup_service_configs(self):
        """设置服务配置"""
        self.service_configs = {
            'trading-engine': ContainerConfig(
                name='trading-engine',
                image='trading-system:latest',
                ports={'5000': 5000},
                environment={
                    'PYTHONPATH': '/app',
                    'ENV': 'production',
                    'LOG_LEVEL': 'INFO'
                },
                volumes={
                    './logs': '/app/logs',
                    './data': '/app/data',
                    './config': '/app/config'
                },
                memory_limit='2g',
                cpu_limit=2.0
            ),
            'web-dashboard': ContainerConfig(
                name='web-dashboard',
                image='trading-dashboard:latest',
                ports={'8080': 5000},
                environment={
                    'FLASK_ENV': 'production',
                    'API_URL': 'http://trading-engine:5000'
                },
                volumes={
                    './static': '/app/static'
                },
                memory_limit='512m',
                cpu_limit=0.5
            ),
            'redis': ContainerConfig(
                name='redis',
                image='redis:7-alpine',
                ports={'6379': 6379},
                environment={},
                volumes={
                    './redis-data': '/data'
                },
                memory_limit='256m',
                cpu_limit=0.5
            ),
            'postgres': ContainerConfig(
                name='postgres',
                image='postgres:15-alpine',
                ports={'5432': 5432},
                environment={
                    'POSTGRES_DB': 'trading_system',
                    'POSTGRES_USER': 'trading_user',
                    'POSTGRES_PASSWORD': 'secure_password_2024'
                },
                volumes={
                    './postgres-data': '/var/lib/postgresql/data'
                },
                memory_limit='1g',
                cpu_limit=1.0
            )
        }
    
    def create_dockerfile(self, service_name: str) -> str:
        """创建Dockerfile"""
        dockerfiles = {
            'trading-engine': '''
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/

# 创建必要目录
RUN mkdir -p logs data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# 启动命令
CMD ["python", "-m", "src.main"]
            ''',
            'trading-dashboard': '''
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-web.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-web.txt

# 复制应用代码
COPY src/ui/ ./src/ui/
COPY src/monitoring/ ./src/monitoring/

# 创建静态文件目录
RUN mkdir -p static templates

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=src.ui.web_dashboard
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:5000/ || exit 1

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "src.ui.web_dashboard:app"]
            '''
        }
        
        return dockerfiles.get(service_name, '')
    
    def create_docker_compose(self) -> str:
        """创建docker-compose.yml"""
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'trading-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'postgres-data': {},
                'redis-data': {},
                'trading-logs': {},
                'trading-data': {}
            }
        }
        
        # 添加服务配置
        for service_name, config in self.service_configs.items():
            service_config = {
                'image': config.image,
                'container_name': config.name,
                'restart': config.restart_policy,
                'networks': ['trading-network'],
                'environment': config.environment,
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': config.memory_limit,
                            'cpus': str(config.cpu_limit)
                        }
                    }
                }
            }
            
            # 添加端口映射
            if config.ports:
                service_config['ports'] = [
                    f"{host_port}:{container_port}" 
                    for container_port, host_port in config.ports.items()
                ]
            
            # 添加卷映射
            if config.volumes:
                service_config['volumes'] = [
                    f"{host_path}:{container_path}" 
                    for host_path, container_path in config.volumes.items()
                ]
            
            # 添加依赖关系
            if service_name == 'trading-engine':
                service_config['depends_on'] = ['postgres', 'redis']
            elif service_name == 'web-dashboard':
                service_config['depends_on'] = ['trading-engine']
            
            compose_config['services'][service_name] = service_config
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def build_image(self, service_name: str, dockerfile_content: str = None) -> bool:
        """构建Docker镜像"""
        try:
            if not self.client:
                self.logger.error("Docker客户端未连接")
                return False
            
            config = self.service_configs.get(service_name)
            if not config:
                self.logger.error(f"服务配置不存在: {service_name}")
                return False
            
            # 创建构建上下文
            build_context = f"./build/{service_name}"
            os.makedirs(build_context, exist_ok=True)
            
            # 写入Dockerfile
            if not dockerfile_content:
                dockerfile_content = self.create_dockerfile(service_name)
            
            with open(f"{build_context}/Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # 构建镜像
            self.logger.info(f"开始构建镜像: {config.image}")
            
            image, build_logs = self.client.images.build(
                path=build_context,
                tag=config.image,
                rm=True,
                forcerm=True
            )
            
            # 输出构建日志
            for log in build_logs:
                if 'stream' in log:
                    self.logger.info(f"构建日志: {log['stream'].strip()}")
            
            self.logger.info(f"镜像构建成功: {config.image}")
            return True
            
        except Exception as e:
            self.logger.error(f"构建镜像失败 {service_name}: {e}")
            return False
    
    def deploy_service(self, service_name: str) -> bool:
        """部署服务"""
        try:
            if not self.client:
                self.logger.error("Docker客户端未连接")
                return False
            
            config = self.service_configs.get(service_name)
            if not config:
                self.logger.error(f"服务配置不存在: {service_name}")
                return False
            
            # 停止现有容器
            self.stop_service(service_name)
            
            # 创建卷映射
            volumes = {}
            for host_path, container_path in config.volumes.items():
                # 确保主机目录存在
                os.makedirs(host_path, exist_ok=True)
                volumes[os.path.abspath(host_path)] = {
                    'bind': container_path,
                    'mode': 'rw'
                }
            
            # 创建端口映射
            ports = {}
            for container_port, host_port in config.ports.items():
                ports[f"{container_port}/tcp"] = host_port
            
            # 启动容器
            self.logger.info(f"启动服务: {service_name}")
            
            container = self.client.containers.run(
                image=config.image,
                name=config.name,
                ports=ports,
                volumes=volumes,
                environment=config.environment,
                restart_policy={"Name": config.restart_policy},
                mem_limit=config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(config.cpu_limit * 100000),
                detach=True,
                remove=False
            )
            
            self.services[service_name] = container
            self.logger.info(f"服务启动成功: {service_name} ({container.id[:12]})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"部署服务失败 {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """停止服务"""
        try:
            if not self.client:
                return False
            
            # 查找现有容器
            try:
                container = self.client.containers.get(self.service_configs[service_name].name)
                container.stop(timeout=30)
                container.remove()
                self.logger.info(f"服务已停止: {service_name}")
            except docker.errors.NotFound:
                self.logger.info(f"容器不存在: {service_name}")
            
            # 从服务列表中移除
            if service_name in self.services:
                del self.services[service_name]
            
            return True
            
        except Exception as e:
            self.logger.error(f"停止服务失败 {service_name}: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """重启服务"""
        try:
            self.logger.info(f"重启服务: {service_name}")
            
            if self.stop_service(service_name):
                time.sleep(2)  # 等待容器完全停止
                return self.deploy_service(service_name)
            
            return False
            
        except Exception as e:
            self.logger.error(f"重启服务失败 {service_name}: {e}")
            return False
    
    def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """获取服务状态"""
        try:
            if not self.client:
                return None
            
            config = self.service_configs.get(service_name)
            if not config:
                return None
            
            try:
                container = self.client.containers.get(config.name)
                
                # 获取容器统计信息
                stats = container.stats(stream=False)
                
                # 计算CPU使用率
                cpu_usage = 0.0
                if 'cpu_stats' in stats and 'precpu_stats' in stats:
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                    if system_delta > 0:
                        cpu_usage = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                
                # 计算内存使用率
                memory_usage = 0.0
                if 'memory_stats' in stats:
                    memory_stats = stats['memory_stats']
                    if 'usage' in memory_stats and 'limit' in memory_stats:
                        memory_usage = (memory_stats['usage'] / memory_stats['limit']) * 100
                
                # 获取端口信息
                ports = []
                if container.attrs['NetworkSettings']['Ports']:
                    for container_port, host_bindings in container.attrs['NetworkSettings']['Ports'].items():
                        if host_bindings:
                            for binding in host_bindings:
                                ports.append(f"{binding['HostPort']}:{container_port}")
                
                return ServiceStatus(
                    name=service_name,
                    status=container.status,
                    container_id=container.id[:12],
                    image=container.image.tags[0] if container.image.tags else container.image.id[:12],
                    created_at=datetime.fromisoformat(container.attrs['Created'].replace('Z', '+00:00')),
                    ports=ports,
                    health=container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown'),
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage
                )
                
            except docker.errors.NotFound:
                return ServiceStatus(
                    name=service_name,
                    status='not_found',
                    container_id='',
                    image='',
                    created_at=datetime.now(),
                    ports=[],
                    health='unknown'
                )
                
        except Exception as e:
            self.logger.error(f"获取服务状态失败 {service_name}: {e}")
            return None
    
    def get_all_services_status(self) -> Dict[str, ServiceStatus]:
        """获取所有服务状态"""
        status_dict = {}
        
        for service_name in self.service_configs.keys():
            status = self.get_service_status(service_name)
            if status:
                status_dict[service_name] = status
        
        return status_dict
    
    def start_monitoring(self):
        """启动服务监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("服务监控已启动")
    
    def stop_monitoring(self):
        """停止服务监控"""
        self._monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("服务监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self._monitoring:
            try:
                # 检查所有服务状态
                services_status = self.get_all_services_status()
                
                for service_name, status in services_status.items():
                    # 检查服务健康状态
                    if status.status == 'exited':
                        self.logger.warning(f"服务异常退出: {service_name}")
                        # 可以在这里添加自动重启逻辑
                        
                    elif status.health == 'unhealthy':
                        self.logger.warning(f"服务健康检查失败: {service_name}")
                        
                    # 检查资源使用率
                    if status.cpu_usage > 80:
                        self.logger.warning(f"服务CPU使用率过高: {service_name} ({status.cpu_usage:.1f}%)")
                        
                    if status.memory_usage > 80:
                        self.logger.warning(f"服务内存使用率过高: {service_name} ({status.memory_usage:.1f}%)")
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(60)
    
    def deploy_all_services(self) -> bool:
        """部署所有服务"""
        try:
            self.logger.info("开始部署所有服务")
            
            # 按依赖顺序部署
            deployment_order = ['postgres', 'redis', 'trading-engine', 'web-dashboard']
            
            for service_name in deployment_order:
                if service_name in self.service_configs:
                    self.logger.info(f"部署服务: {service_name}")
                    
                    if not self.deploy_service(service_name):
                        self.logger.error(f"服务部署失败: {service_name}")
                        return False
                    
                    # 等待服务启动
                    time.sleep(5)
            
            self.logger.info("所有服务部署完成")
            return True
            
        except Exception as e:
            self.logger.error(f"部署所有服务失败: {e}")
            return False
    
    def create_deployment_files(self):
        """创建部署文件"""
        try:
            # 创建目录
            os.makedirs('deployment', exist_ok=True)
            
            # 创建docker-compose.yml
            compose_content = self.create_docker_compose()
            with open('deployment/docker-compose.yml', 'w') as f:
                f.write(compose_content)
            
            # 创建Dockerfile
            for service_name in ['trading-engine', 'trading-dashboard']:
                dockerfile_content = self.create_dockerfile(service_name)
                if dockerfile_content:
                    os.makedirs(f'deployment/{service_name}', exist_ok=True)
                    with open(f'deployment/{service_name}/Dockerfile', 'w') as f:
                        f.write(dockerfile_content)
            
            # 创建部署脚本
            deploy_script = '''#!/bin/bash
set -e

echo "开始部署交易系统..."

# 构建镜像
echo "构建Docker镜像..."
docker-compose build

# 启动服务
echo "启动服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 30

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

echo "部署完成！"
echo "Web界面: http://localhost:8080"
echo "API接口: http://localhost:5000"
            '''
            
            with open('deployment/deploy.sh', 'w') as f:
                f.write(deploy_script)
            
            # 设置执行权限
            os.chmod('deployment/deploy.sh', 0o755)
            
            self.logger.info("部署文件创建完成")
            
        except Exception as e:
            self.logger.error(f"创建部署文件失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建Docker管理器
    docker_manager = DockerManager()
    
    try:
        # 创建部署文件
        docker_manager.create_deployment_files()
        
        # 获取服务状态
        services_status = docker_manager.get_all_services_status()
        
        print("服务状态:")
        for service_name, status in services_status.items():
            print(f"  {service_name}: {status.status}")
        
        # 启动监控
        docker_manager.start_monitoring()
        
        # 保持运行
        time.sleep(60)
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        docker_manager.stop_monitoring()
