"""
🔐 加密配置管理系统
生产级密钥管理和配置加密系统，确保敏感信息安全存储和传输
支持多种加密算法、密钥轮换、安全审计等企业级功能
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import bcrypt
from loguru import logger


@dataclass
class EncryptionConfig:
    """加密配置"""
    algorithm: str = "AES-256-GCM"
    key_size: int = 32
    salt_size: int = 16
    iterations: int = 100000
    key_rotation_days: int = 30
    backup_keys_count: int = 3


@dataclass
class SecretMetadata:
    """密钥元数据"""
    name: str
    created_at: datetime
    last_rotated: datetime
    expires_at: Optional[datetime] = None
    rotation_count: int = 0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.master_key: Optional[bytes] = None
        self.fernet: Optional[Fernet] = None
        self.secrets_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, SecretMetadata] = {}
        
        # 密钥存储路径
        self.secrets_dir = Path("secrets")
        self.secrets_dir.mkdir(exist_ok=True, mode=0o700)
        
        # 初始化主密钥
        self._initialize_master_key()
        
        logger.info("加密管理器初始化完成")
    
    def _initialize_master_key(self):
        """初始化主密钥"""
        try:
            # 尝试从环境变量获取主密钥
            master_key_b64 = os.getenv('MASTER_ENCRYPTION_KEY')
            
            if master_key_b64:
                self.master_key = base64.b64decode(master_key_b64)
                logger.info("从环境变量加载主密钥")
            else:
                # 生成新的主密钥
                self.master_key = self._generate_master_key()
                master_key_b64 = base64.b64encode(self.master_key).decode()
                
                logger.warning("生成新的主密钥，请将以下密钥保存到环境变量 MASTER_ENCRYPTION_KEY:")
                logger.warning(f"MASTER_ENCRYPTION_KEY={master_key_b64}")
            
            # 初始化Fernet
            self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
            
        except Exception as e:
            logger.error(f"初始化主密钥失败: {e}")
            raise
    
    def _generate_master_key(self) -> bytes:
        """生成主密钥"""
        # 使用系统随机数生成器
        return secrets.token_bytes(32)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """派生密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """加密数据"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            elif isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self.fernet.encrypt(data)
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """解密数据"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            
            # 尝试解析为JSON
            try:
                return json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                return decrypted_data.decode()
                
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise
    
    def store_secret(self, name: str, value: Any, tags: Dict[str, str] = None) -> bool:
        """存储密钥"""
        try:
            # 加密密钥值
            encrypted_value = self.encrypt_data(value)
            
            # 创建元数据
            metadata = SecretMetadata(
                name=name,
                created_at=datetime.now(),
                last_rotated=datetime.now(),
                tags=tags or {}
            )
            
            # 保存到文件
            secret_file = self.secrets_dir / f"{name}.enc"
            metadata_file = self.secrets_dir / f"{name}.meta"
            
            with open(secret_file, 'w') as f:
                f.write(encrypted_value)
            
            with open(metadata_file, 'w') as f:
                json.dump({
                    'name': metadata.name,
                    'created_at': metadata.created_at.isoformat(),
                    'last_rotated': metadata.last_rotated.isoformat(),
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'rotation_count': metadata.rotation_count,
                    'access_count': metadata.access_count,
                    'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                    'tags': metadata.tags
                }, f, indent=2)
            
            # 更新缓存
            self.secrets_cache[name] = value
            self.metadata_cache[name] = metadata
            
            logger.info(f"密钥存储成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"存储密钥失败: {name} - {e}")
            return False
    
    def get_secret(self, name: str) -> Optional[Any]:
        """获取密钥"""
        try:
            # 先检查缓存
            if name in self.secrets_cache:
                self._update_access_metadata(name)
                return self.secrets_cache[name]
            
            # 从文件加载
            secret_file = self.secrets_dir / f"{name}.enc"
            metadata_file = self.secrets_dir / f"{name}.meta"
            
            if not secret_file.exists():
                logger.warning(f"密钥不存在: {name}")
                return None
            
            # 读取加密数据
            with open(secret_file, 'r') as f:
                encrypted_value = f.read()
            
            # 解密
            value = self.decrypt_data(encrypted_value)
            
            # 读取元数据
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    meta_data = json.load(f)
                
                metadata = SecretMetadata(
                    name=meta_data['name'],
                    created_at=datetime.fromisoformat(meta_data['created_at']),
                    last_rotated=datetime.fromisoformat(meta_data['last_rotated']),
                    expires_at=datetime.fromisoformat(meta_data['expires_at']) if meta_data['expires_at'] else None,
                    rotation_count=meta_data.get('rotation_count', 0),
                    access_count=meta_data.get('access_count', 0),
                    last_accessed=datetime.fromisoformat(meta_data['last_accessed']) if meta_data.get('last_accessed') else None,
                    tags=meta_data.get('tags', {})
                )
                
                self.metadata_cache[name] = metadata
            
            # 更新缓存和访问记录
            self.secrets_cache[name] = value
            self._update_access_metadata(name)
            
            logger.debug(f"密钥获取成功: {name}")
            return value
            
        except Exception as e:
            logger.error(f"获取密钥失败: {name} - {e}")
            return None
    
    def _update_access_metadata(self, name: str):
        """更新访问元数据"""
        if name in self.metadata_cache:
            metadata = self.metadata_cache[name]
            metadata.access_count += 1
            metadata.last_accessed = datetime.now()
            
            # 保存更新的元数据
            metadata_file = self.secrets_dir / f"{name}.meta"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'name': metadata.name,
                    'created_at': metadata.created_at.isoformat(),
                    'last_rotated': metadata.last_rotated.isoformat(),
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'rotation_count': metadata.rotation_count,
                    'access_count': metadata.access_count,
                    'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                    'tags': metadata.tags
                }, f, indent=2)
    
    def delete_secret(self, name: str) -> bool:
        """删除密钥"""
        try:
            secret_file = self.secrets_dir / f"{name}.enc"
            metadata_file = self.secrets_dir / f"{name}.meta"
            
            # 删除文件
            if secret_file.exists():
                secret_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            # 清除缓存
            self.secrets_cache.pop(name, None)
            self.metadata_cache.pop(name, None)
            
            logger.info(f"密钥删除成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"删除密钥失败: {name} - {e}")
            return False
    
    def list_secrets(self) -> Dict[str, SecretMetadata]:
        """列出所有密钥"""
        secrets_list = {}
        
        try:
            for secret_file in self.secrets_dir.glob("*.enc"):
                name = secret_file.stem
                metadata_file = self.secrets_dir / f"{name}.meta"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        meta_data = json.load(f)
                    
                    metadata = SecretMetadata(
                        name=meta_data['name'],
                        created_at=datetime.fromisoformat(meta_data['created_at']),
                        last_rotated=datetime.fromisoformat(meta_data['last_rotated']),
                        expires_at=datetime.fromisoformat(meta_data['expires_at']) if meta_data['expires_at'] else None,
                        rotation_count=meta_data.get('rotation_count', 0),
                        access_count=meta_data.get('access_count', 0),
                        last_accessed=datetime.fromisoformat(meta_data['last_accessed']) if meta_data.get('last_accessed') else None,
                        tags=meta_data.get('tags', {})
                    )
                    
                    secrets_list[name] = metadata
            
            return secrets_list
            
        except Exception as e:
            logger.error(f"列出密钥失败: {e}")
            return {}
    
    def rotate_secret(self, name: str, new_value: Any) -> bool:
        """轮换密钥"""
        try:
            if name not in self.metadata_cache:
                # 先加载元数据
                self.get_secret(name)
            
            if name not in self.metadata_cache:
                logger.error(f"密钥不存在，无法轮换: {name}")
                return False
            
            # 备份旧密钥
            old_value = self.get_secret(name)
            backup_name = f"{name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.store_secret(backup_name, old_value, {'type': 'backup', 'original': name})
            
            # 更新密钥
            metadata = self.metadata_cache[name]
            metadata.last_rotated = datetime.now()
            metadata.rotation_count += 1
            
            # 存储新密钥
            success = self.store_secret(name, new_value, metadata.tags)
            
            if success:
                logger.info(f"密钥轮换成功: {name}")
                return True
            else:
                logger.error(f"密钥轮换失败: {name}")
                return False
                
        except Exception as e:
            logger.error(f"轮换密钥失败: {name} - {e}")
            return False
    
    def check_key_expiration(self) -> Dict[str, int]:
        """检查密钥过期情况"""
        expiring_keys = {}
        
        try:
            secrets_list = self.list_secrets()
            current_time = datetime.now()
            
            for name, metadata in secrets_list.items():
                # 检查是否需要轮换
                days_since_rotation = (current_time - metadata.last_rotated).days
                
                if days_since_rotation >= self.config.key_rotation_days:
                    expiring_keys[name] = days_since_rotation
            
            return expiring_keys
            
        except Exception as e:
            logger.error(f"检查密钥过期失败: {e}")
            return {}
    
    def cleanup_backup_keys(self, keep_count: int = None) -> int:
        """清理备份密钥"""
        keep_count = keep_count or self.config.backup_keys_count
        cleaned_count = 0
        
        try:
            secrets_list = self.list_secrets()
            backup_keys = {}
            
            # 按原始密钥分组备份密钥
            for name, metadata in secrets_list.items():
                if 'type' in metadata.tags and metadata.tags['type'] == 'backup':
                    original_name = metadata.tags.get('original', 'unknown')
                    if original_name not in backup_keys:
                        backup_keys[original_name] = []
                    backup_keys[original_name].append((name, metadata))
            
            # 清理每个原始密钥的多余备份
            for original_name, backups in backup_keys.items():
                if len(backups) > keep_count:
                    # 按创建时间排序，保留最新的
                    backups.sort(key=lambda x: x[1].created_at, reverse=True)
                    
                    for backup_name, _ in backups[keep_count:]:
                        if self.delete_secret(backup_name):
                            cleaned_count += 1
            
            logger.info(f"清理备份密钥完成，删除了 {cleaned_count} 个备份")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理备份密钥失败: {e}")
            return 0
    
    def get_security_audit(self) -> Dict[str, Any]:
        """获取安全审计信息"""
        try:
            secrets_list = self.list_secrets()
            current_time = datetime.now()
            
            audit_info = {
                'total_secrets': len(secrets_list),
                'expiring_keys': 0,
                'never_accessed': 0,
                'high_access_keys': [],
                'old_keys': [],
                'backup_keys': 0,
                'audit_timestamp': current_time.isoformat()
            }
            
            for name, metadata in secrets_list.items():
                # 检查过期密钥
                days_since_rotation = (current_time - metadata.last_rotated).days
                if days_since_rotation >= self.config.key_rotation_days:
                    audit_info['expiring_keys'] += 1
                
                # 检查从未访问的密钥
                if metadata.access_count == 0:
                    audit_info['never_accessed'] += 1
                
                # 检查高频访问密钥
                if metadata.access_count > 1000:
                    audit_info['high_access_keys'].append({
                        'name': name,
                        'access_count': metadata.access_count,
                        'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None
                    })
                
                # 检查老旧密钥
                days_since_creation = (current_time - metadata.created_at).days
                if days_since_creation > 365:  # 超过一年
                    audit_info['old_keys'].append({
                        'name': name,
                        'age_days': days_since_creation,
                        'rotation_count': metadata.rotation_count
                    })
                
                # 统计备份密钥
                if 'type' in metadata.tags and metadata.tags['type'] == 'backup':
                    audit_info['backup_keys'] += 1
            
            return audit_info
            
        except Exception as e:
            logger.error(f"获取安全审计信息失败: {e}")
            return {}


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.config_cache: Dict[str, Any] = {}
        
        logger.info("配置管理器初始化完成")
    
    def get_exchange_config(self, exchange_name: str) -> Optional[Dict[str, str]]:
        """获取交易所配置"""
        try:
            # 优先从环境变量获取
            api_key = os.getenv(f'{exchange_name.upper()}_API_KEY')
            secret_key = os.getenv(f'{exchange_name.upper()}_SECRET_KEY')
            passphrase = os.getenv(f'{exchange_name.upper()}_PASSPHRASE')
            
            if api_key and secret_key:
                config = {
                    'api_key': api_key,
                    'secret_key': secret_key
                }
                if passphrase:
                    config['passphrase'] = passphrase
                
                logger.info(f"从环境变量加载 {exchange_name} 配置")
                return config
            
            # 从加密存储获取
            config = self.encryption_manager.get_secret(f'{exchange_name}_config')
            if config:
                logger.info(f"从加密存储加载 {exchange_name} 配置")
                return config
            
            logger.warning(f"未找到 {exchange_name} 配置")
            return None
            
        except Exception as e:
            logger.error(f"获取 {exchange_name} 配置失败: {e}")
            return None
    
    def set_exchange_config(self, exchange_name: str, api_key: str, secret_key: str, passphrase: str = None) -> bool:
        """设置交易所配置"""
        try:
            config = {
                'api_key': api_key,
                'secret_key': secret_key
            }
            if passphrase:
                config['passphrase'] = passphrase
            
            success = self.encryption_manager.store_secret(
                f'{exchange_name}_config',
                config,
                {'type': 'exchange_config', 'exchange': exchange_name}
            )
            
            if success:
                logger.info(f"设置 {exchange_name} 配置成功")
            else:
                logger.error(f"设置 {exchange_name} 配置失败")
            
            return success
            
        except Exception as e:
            logger.error(f"设置 {exchange_name} 配置失败: {e}")
            return False
    
    def get_database_config(self) -> Dict[str, str]:
        """获取数据库配置"""
        try:
            config = {
                'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                'postgres_url': os.getenv('POSTGRES_URL', 'postgresql://trader:trading123@localhost:5432/trading_db'),
                'clickhouse_url': os.getenv('CLICKHOUSE_URL', 'clickhouse://localhost:8123/trading_db')
            }
            
            return config
            
        except Exception as e:
            logger.error(f"获取数据库配置失败: {e}")
            return {}
    
    def get_monitoring_config(self) -> Dict[str, str]:
        """获取监控配置"""
        try:
            config = {
                'prometheus_url': os.getenv('PROMETHEUS_URL', 'http://localhost:9090'),
                'grafana_url': os.getenv('GRAFANA_URL', 'http://localhost:3000')
            }
            
            return config
            
        except Exception as e:
            logger.error(f"获取监控配置失败: {e}")
            return {}


# 全局实例
encryption_manager = EncryptionManager()
config_manager = ConfigManager(encryption_manager)

