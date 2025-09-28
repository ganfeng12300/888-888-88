"""
⚙️ 配置管理系统
生产级配置管理系统，实现动态配置、热更新、环境隔离等完整功能
支持多环境配置、配置验证、配置监控和配置回滚
"""

import asyncio
import json
import yaml
import os
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import shutil
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from loguru import logger
from src.core.config import settings


class ConfigFormat(Enum):
    """配置格式"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigSource(Enum):
    """配置源"""
    FILE = "file"               # 文件配置
    ENVIRONMENT = "environment" # 环境变量
    DATABASE = "database"       # 数据库配置
    REMOTE = "remote"           # 远程配置
    MEMORY = "memory"           # 内存配置


@dataclass
class ConfigItem:
    """配置项"""
    key: str                                # 配置键
    value: Any                              # 配置值
    source: ConfigSource                    # 配置源
    format: ConfigFormat                    # 配置格式
    environment: str = "default"            # 环境
    description: str = ""                   # 描述
    required: bool = False                  # 是否必需
    sensitive: bool = False                 # 是否敏感
    validator: Optional[Callable[[Any], bool]] = None  # 验证函数
    default_value: Any = None               # 默认值
    created_at: float = field(default_factory=time.time)  # 创建时间
    updated_at: float = field(default_factory=time.time)  # 更新时间
    version: int = 1                        # 版本号


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    key: str                                # 配置键
    old_value: Any                          # 旧值
    new_value: Any                          # 新值
    source: ConfigSource                    # 配置源
    environment: str                        # 环境
    timestamp: float = field(default_factory=time.time)  # 时间戳
    change_type: str = "update"             # 变更类型: create/update/delete


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type) -> bool:
        """验证类型"""
        try:
            if expected_type == bool and isinstance(value, str):
                return value.lower() in ['true', 'false', '1', '0', 'yes', 'no']
            return isinstance(value, expected_type)
        except Exception:
            return False
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """验证数值范围"""
        try:
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_choices(value: Any, choices: List[Any]) -> bool:
        """验证选择项"""
        return value in choices
    
    @staticmethod
    def validate_regex(value: str, pattern: str) -> bool:
        """验证正则表达式"""
        try:
            import re
            return bool(re.match(pattern, value))
        except Exception:
            return False
    
    @staticmethod
    def validate_url(value: str) -> bool:
        """验证URL"""
        try:
            from urllib.parse import urlparse
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_email(value: str) -> bool:
        """验证邮箱"""
        try:
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, value))
        except Exception:
            return False


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.file_hashes: Dict[str, str] = {}
    
    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self._is_config_file(file_path):
            # 检查文件是否真的改变了
            current_hash = self._get_file_hash(file_path)
            if current_hash != self.file_hashes.get(file_path):
                self.file_hashes[file_path] = current_hash
                asyncio.create_task(self.config_manager._reload_config_file(file_path))
    
    def _is_config_file(self, file_path: str) -> bool:
        """检查是否为配置文件"""
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.env']
        return any(file_path.endswith(ext) for ext in config_extensions)
    
    def _get_file_hash(self, file_path: str) -> str:
        """获取文件哈希"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config", environment: str = "default"):
        self.config_dir = Path(config_dir)
        self.environment = environment
        
        # 配置存储
        self.configs: Dict[str, ConfigItem] = {}
        self.config_history: Dict[str, List[ConfigChangeEvent]] = defaultdict(list)
        
        # 配置监听器
        self.change_listeners: Dict[str, List[Callable[[ConfigChangeEvent], None]]] = defaultdict(list)
        
        # 文件监控
        self.file_watcher = None
        self.observer = None
        
        # 环境配置
        self.environments = ["default", "development", "testing", "staging", "production"]
        
        # 配置缓存
        self.config_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5分钟缓存
        self.cache_timestamps: Dict[str, float] = {}
        
        # 配置验证规则
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        
        # 运行状态
        self.running = False
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"配置管理器初始化完成 - 目录: {self.config_dir}, 环境: {self.environment}")
    
    async def start(self):
        """启动配置管理器"""
        try:
            self.running = True
            
            # 加载所有配置文件
            await self._load_all_configs()
            
            # 启动文件监控
            if WATCHDOG_AVAILABLE:
                self._start_file_watcher()
            
            logger.info("配置管理器启动成功")
            
        except Exception as e:
            logger.error(f"配置管理器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止配置管理器"""
        try:
            self.running = False
            
            # 停止文件监控
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            logger.info("配置管理器已停止")
            
        except Exception as e:
            logger.error(f"停止配置管理器失败: {e}")
    
    def _start_file_watcher(self):
        """启动文件监控"""
        try:
            self.file_watcher = ConfigFileWatcher(self)
            self.observer = Observer()
            self.observer.schedule(self.file_watcher, str(self.config_dir), recursive=True)
            self.observer.start()
            
            logger.info("配置文件监控已启动")
            
        except Exception as e:
            logger.error(f"启动文件监控失败: {e}")
    
    async def _load_all_configs(self):
        """加载所有配置文件"""
        try:
            # 加载默认配置
            await self._load_environment_configs("default")
            
            # 加载当前环境配置
            if self.environment != "default":
                await self._load_environment_configs(self.environment)
            
            # 加载环境变量配置
            self._load_environment_variables()
            
            logger.info(f"配置加载完成 - 共 {len(self.configs)} 项配置")
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    async def _load_environment_configs(self, env: str):
        """加载环境配置"""
        env_dir = self.config_dir / env
        if not env_dir.exists():
            return
        
        for config_file in env_dir.glob("*"):
            if config_file.is_file():
                await self._load_config_file(config_file, env)
    
    async def _load_config_file(self, file_path: Path, environment: str = None):
        """加载配置文件"""
        try:
            if environment is None:
                environment = self.environment
            
            # 确定配置格式
            config_format = self._detect_config_format(file_path)
            
            # 读取配置内容
            config_data = self._read_config_file(file_path, config_format)
            
            # 处理配置数据
            self._process_config_data(config_data, ConfigSource.FILE, config_format, environment, str(file_path))
            
            logger.info(f"配置文件加载成功: {file_path}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {file_path} - {e}")
    
    def _detect_config_format(self, file_path: Path) -> ConfigFormat:
        """检测配置格式"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.toml':
            return ConfigFormat.TOML
        elif suffix == '.ini':
            return ConfigFormat.INI
        elif suffix == '.env':
            return ConfigFormat.ENV
        else:
            return ConfigFormat.JSON  # 默认JSON
    
    def _read_config_file(self, file_path: Path, config_format: ConfigFormat) -> Dict[str, Any]:
        """读取配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if config_format == ConfigFormat.JSON:
                return json.loads(content)
            elif config_format == ConfigFormat.YAML:
                return yaml.safe_load(content) or {}
            elif config_format == ConfigFormat.TOML:
                try:
                    import toml
                    return toml.loads(content)
                except ImportError:
                    logger.warning("TOML库未安装，跳过TOML配置文件")
                    return {}
            elif config_format == ConfigFormat.INI:
                import configparser
                config = configparser.ConfigParser()
                config.read_string(content)
                return {section: dict(config[section]) for section in config.sections()}
            elif config_format == ConfigFormat.ENV:
                return self._parse_env_file(content)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"读取配置文件失败: {file_path} - {e}")
            return {}
    
    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """解析环境变量文件"""
        config = {}
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"\'')
        return config
    
    def _process_config_data(self, data: Dict[str, Any], source: ConfigSource, 
                           config_format: ConfigFormat, environment: str, source_path: str = ""):
        """处理配置数据"""
        def process_nested(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        process_nested(value, full_key)
                    else:
                        self._set_config_item(full_key, value, source, config_format, environment)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    full_key = f"{prefix}[{i}]"
                    if isinstance(item, (dict, list)):
                        process_nested(item, full_key)
                    else:
                        self._set_config_item(full_key, item, source, config_format, environment)
        
        process_nested(data)
    
    def _set_config_item(self, key: str, value: Any, source: ConfigSource, 
                        config_format: ConfigFormat, environment: str):
        """设置配置项"""
        try:
            # 检查是否已存在
            old_value = None
            if key in self.configs:
                old_value = self.configs[key].value
            
            # 创建配置项
            config_item = ConfigItem(
                key=key,
                value=value,
                source=source,
                format=config_format,
                environment=environment,
                updated_at=time.time()
            )
            
            # 验证配置
            if not self._validate_config_item(config_item):
                logger.warning(f"配置验证失败: {key} = {value}")
                return
            
            # 存储配置
            self.configs[key] = config_item
            
            # 清除缓存
            if key in self.config_cache:
                del self.config_cache[key]
                del self.cache_timestamps[key]
            
            # 记录变更事件
            change_event = ConfigChangeEvent(
                key=key,
                old_value=old_value,
                new_value=value,
                source=source,
                environment=environment,
                change_type="create" if old_value is None else "update"
            )
            
            self.config_history[key].append(change_event)
            
            # 通知监听器
            self._notify_change_listeners(key, change_event)
            
        except Exception as e:
            logger.error(f"设置配置项失败: {key} - {e}")
    
    def _load_environment_variables(self):
        """加载环境变量配置"""
        try:
            for key, value in os.environ.items():
                # 只加载特定前缀的环境变量
                if key.startswith('APP_') or key.startswith('TRADING_'):
                    config_key = key.lower()
                    self._set_config_item(
                        config_key, value, ConfigSource.ENVIRONMENT, 
                        ConfigFormat.ENV, self.environment
                    )
            
            logger.info("环境变量配置加载完成")
            
        except Exception as e:
            logger.error(f"加载环境变量失败: {e}")
    
    def get(self, key: str, default: Any = None, use_cache: bool = True) -> Any:
        """获取配置值"""
        try:
            # 检查缓存
            if use_cache and key in self.config_cache:
                cache_time = self.cache_timestamps.get(key, 0)
                if time.time() - cache_time < self.cache_ttl:
                    return self.config_cache[key]
            
            # 获取配置项
            if key in self.configs:
                value = self.configs[key].value
                
                # 类型转换
                value = self._convert_value_type(value)
                
                # 更新缓存
                if use_cache:
                    self.config_cache[key] = value
                    self.cache_timestamps[key] = time.time()
                
                return value
            
            return default
            
        except Exception as e:
            logger.error(f"获取配置失败: {key} - {e}")
            return default
    
    def set(self, key: str, value: Any, environment: str = None, 
            source: ConfigSource = ConfigSource.MEMORY) -> bool:
        """设置配置值"""
        try:
            if environment is None:
                environment = self.environment
            
            self._set_config_item(key, value, source, ConfigFormat.JSON, environment)
            return True
            
        except Exception as e:
            logger.error(f"设置配置失败: {key} - {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除配置"""
        try:
            if key in self.configs:
                old_value = self.configs[key].value
                del self.configs[key]
                
                # 清除缓存
                if key in self.config_cache:
                    del self.config_cache[key]
                    del self.cache_timestamps[key]
                
                # 记录变更事件
                change_event = ConfigChangeEvent(
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    source=ConfigSource.MEMORY,
                    environment=self.environment,
                    change_type="delete"
                )
                
                self.config_history[key].append(change_event)
                self._notify_change_listeners(key, change_event)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除配置失败: {key} - {e}")
            return False
    
    def get_all(self, environment: str = None, prefix: str = "") -> Dict[str, Any]:
        """获取所有配置"""
        try:
            if environment is None:
                environment = self.environment
            
            result = {}
            for key, config_item in self.configs.items():
                if config_item.environment == environment and key.startswith(prefix):
                    result[key] = self._convert_value_type(config_item.value)
            
            return result
            
        except Exception as e:
            logger.error(f"获取所有配置失败: {e}")
            return {}
    
    def _convert_value_type(self, value: Any) -> Any:
        """转换值类型"""
        if isinstance(value, str):
            # 尝试转换布尔值
            if value.lower() in ['true', 'yes', '1']:
                return True
            elif value.lower() in ['false', 'no', '0']:
                return False
            
            # 尝试转换数字
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                pass
        
        return value
    
    def add_validation_rule(self, key: str, rule_type: str, **kwargs):
        """添加验证规则"""
        if key not in self.validation_rules:
            self.validation_rules[key] = {}
        
        self.validation_rules[key][rule_type] = kwargs
    
    def _validate_config_item(self, config_item: ConfigItem) -> bool:
        """验证配置项"""
        try:
            key = config_item.key
            value = config_item.value
            
            # 检查验证规则
            if key in self.validation_rules:
                rules = self.validation_rules[key]
                
                for rule_type, rule_params in rules.items():
                    if rule_type == 'type':
                        expected_type = rule_params.get('type')
                        if not ConfigValidator.validate_type(value, expected_type):
                            return False
                    
                    elif rule_type == 'range':
                        min_val = rule_params.get('min')
                        max_val = rule_params.get('max')
                        if not ConfigValidator.validate_range(value, min_val, max_val):
                            return False
                    
                    elif rule_type == 'choices':
                        choices = rule_params.get('choices', [])
                        if not ConfigValidator.validate_choices(value, choices):
                            return False
                    
                    elif rule_type == 'regex':
                        pattern = rule_params.get('pattern')
                        if not ConfigValidator.validate_regex(str(value), pattern):
                            return False
                    
                    elif rule_type == 'url':
                        if not ConfigValidator.validate_url(str(value)):
                            return False
                    
                    elif rule_type == 'email':
                        if not ConfigValidator.validate_email(str(value)):
                            return False
            
            # 调用自定义验证函数
            if config_item.validator:
                return config_item.validator(value)
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {config_item.key} - {e}")
            return False
    
    def add_change_listener(self, key: str, listener: Callable[[ConfigChangeEvent], None]):
        """添加配置变更监听器"""
        self.change_listeners[key].append(listener)
    
    def remove_change_listener(self, key: str, listener: Callable[[ConfigChangeEvent], None]):
        """移除配置变更监听器"""
        if key in self.change_listeners:
            try:
                self.change_listeners[key].remove(listener)
            except ValueError:
                pass
    
    def _notify_change_listeners(self, key: str, change_event: ConfigChangeEvent):
        """通知配置变更监听器"""
        try:
            # 通知特定键的监听器
            for listener in self.change_listeners.get(key, []):
                try:
                    listener(change_event)
                except Exception as e:
                    logger.error(f"配置变更监听器执行失败: {key} - {e}")
            
            # 通知通配符监听器
            for listener in self.change_listeners.get("*", []):
                try:
                    listener(change_event)
                except Exception as e:
                    logger.error(f"通配符配置变更监听器执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"通知配置变更监听器失败: {e}")
    
    async def _reload_config_file(self, file_path: str):
        """重新加载配置文件"""
        try:
            logger.info(f"重新加载配置文件: {file_path}")
            await self._load_config_file(Path(file_path))
        except Exception as e:
            logger.error(f"重新加载配置文件失败: {file_path} - {e}")
    
    def backup_config(self, backup_path: str = None) -> str:
        """备份配置"""
        try:
            if backup_path is None:
                timestamp = int(time.time())
                backup_path = f"config_backup_{timestamp}.json"
            
            backup_data = {
                'timestamp': time.time(),
                'environment': self.environment,
                'configs': {}
            }
            
            for key, config_item in self.configs.items():
                backup_data['configs'][key] = {
                    'value': config_item.value,
                    'source': config_item.source.value,
                    'format': config_item.format.value,
                    'environment': config_item.environment,
                    'created_at': config_item.created_at,
                    'updated_at': config_item.updated_at
                }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置备份完成: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"配置备份失败: {e}")
            return ""
    
    def restore_config(self, backup_path: str) -> bool:
        """恢复配置"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            for key, config_data in backup_data.get('configs', {}).items():
                config_item = ConfigItem(
                    key=key,
                    value=config_data['value'],
                    source=ConfigSource(config_data['source']),
                    format=ConfigFormat(config_data['format']),
                    environment=config_data['environment'],
                    created_at=config_data['created_at'],
                    updated_at=config_data['updated_at']
                )
                
                self.configs[key] = config_item
            
            # 清除缓存
            self.config_cache.clear()
            self.cache_timestamps.clear()
            
            logger.info(f"配置恢复完成: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置恢复失败: {e}")
            return False
    
    def get_config_history(self, key: str) -> List[ConfigChangeEvent]:
        """获取配置变更历史"""
        return self.config_history.get(key, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_configs': len(self.configs),
            'environments': self.environments,
            'current_environment': self.environment,
            'cache_size': len(self.config_cache),
            'cache_hit_rate': len(self.config_cache) / max(len(self.configs), 1),
            'validation_rules': len(self.validation_rules),
            'change_listeners': sum(len(listeners) for listeners in self.change_listeners.values()),
            'config_sources': list(set(config.source.value for config in self.configs.values())),
            'config_formats': list(set(config.format.value for config in self.configs.values()))
        }


# 全局配置管理器实例
config_manager = ConfigManager()
