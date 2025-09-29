"""
🔐 API安全管理器 - 生产级实盘交易API密钥安全管理系统
提供API密钥加密存储、权限控制、访问审计、密钥轮换等全方位安全功能
支持多交易所API密钥统一管理，确保交易账户安全
"""
import asyncio
import hashlib
import hmac
import json
import os
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Cryptography library not available, API security features will be limited")

from loguru import logger

class APIPermission(Enum):
    """API权限类型"""
    READ_ONLY = "read_only"  # 只读权限
    TRADE = "trade"  # 交易权限
    WITHDRAW = "withdraw"  # 提现权限
    FULL_ACCESS = "full_access"  # 完全访问权限

class SecurityLevel(Enum):
    """安全级别"""
    LOW = "low"  # 低安全级别
    MEDIUM = "medium"  # 中等安全级别
    HIGH = "high"  # 高安全级别
    CRITICAL = "critical"  # 关键安全级别

@dataclass
class APICredentials:
    """API凭证"""
    exchange: str  # 交易所名称
    api_key: str  # API密钥
    api_secret: str  # API密钥
    passphrase: Optional[str] = None  # 密码短语（OKX等需要）
    permissions: List[APIPermission] = field(default_factory=list)  # 权限列表
    security_level: SecurityLevel = SecurityLevel.MEDIUM  # 安全级别
    created_at: float = field(default_factory=time.time)  # 创建时间
    last_used: float = 0.0  # 最后使用时间
    usage_count: int = 0  # 使用次数
    is_active: bool = True  # 是否激活
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class APIAccessLog:
    """API访问日志"""
    exchange: str  # 交易所
    api_key_hash: str  # API密钥哈希
    endpoint: str  # 访问端点
    method: str  # HTTP方法
    ip_address: str  # IP地址
    user_agent: str  # 用户代理
    timestamp: float  # 时间戳
    success: bool  # 是否成功
    error_message: Optional[str] = None  # 错误信息
    response_time: float = 0.0  # 响应时间

class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, master_password: str):
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library is required for encryption")
        
        self.master_password = master_password.encode()
        self._fernet = self._create_fernet()
        
        logger.info("加密管理器初始化完成")
    
    def _create_fernet(self):
        """创建Fernet加密器"""
        # 使用PBKDF2从主密码派生密钥
        salt = b'stable_salt_for_api_keys'  # 在生产环境中应该使用随机盐
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        try:
            encrypted_data = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise
    
    def hash_api_key(self, api_key: str) -> str:
        """生成API密钥哈希"""
        return hashlib.sha256(api_key.encode()).hexdigest()

class APISecurityManager:
    """API安全管理器"""
    
    def __init__(self, master_password: str, storage_path: str = "api_credentials.json"):
        self.storage_path = storage_path
        self.credentials: Dict[str, APICredentials] = {}
        self.access_logs: List[APIAccessLog] = []
        self.encryption_manager = None
        
        if CRYPTO_AVAILABLE:
            self.encryption_manager = EncryptionManager(master_password)
        
        # 安全配置
        self.max_failed_attempts = 5  # 最大失败尝试次数
        self.lockout_duration = 300  # 锁定持续时间（秒）
        self.key_rotation_interval = 86400 * 30  # 密钥轮换间隔（30天）
        
        # 失败尝试跟踪
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_times: Dict[str, float] = {}
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载现有凭证
        self._load_credentials()
        
        logger.info("API安全管理器初始化完成")
    
    def add_credentials(self, exchange: str, api_key: str, api_secret: str,
                       passphrase: Optional[str] = None,
                       permissions: List[APIPermission] = None,
                       security_level: SecurityLevel = SecurityLevel.MEDIUM) -> bool:
        """添加API凭证"""
        try:
            with self.lock:
                if permissions is None:
                    permissions = [APIPermission.READ_ONLY, APIPermission.TRADE]
                
                # 验证API密钥格式
                if not self._validate_api_key_format(exchange, api_key, api_secret):
                    logger.error(f"API密钥格式验证失败: {exchange}")
                    return False
                
                # 创建凭证对象
                credentials = APICredentials(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    permissions=permissions,
                    security_level=security_level
                )
                
                # 生成唯一标识符
                credential_id = f"{exchange}_{self.encryption_manager.hash_api_key(api_key)[:8]}"
                
                # 存储凭证
                self.credentials[credential_id] = credentials
                
                # 保存到文件
                self._save_credentials()
                
                logger.info(f"API凭证添加成功: {exchange}")
                return True
        
        except Exception as e:
            logger.error(f"添加API凭证失败: {e}")
            return False
    
    def get_credentials(self, exchange: str) -> Optional[APICredentials]:
        """获取API凭证"""
        try:
            with self.lock:
                # 检查是否被锁定
                if self._is_locked(exchange):
                    logger.warning(f"交易所API访问被锁定: {exchange}")
                    return None
                
                # 查找匹配的凭证
                for credential_id, credentials in self.credentials.items():
                    if credentials.exchange == exchange and credentials.is_active:
                        # 更新使用统计
                        credentials.last_used = time.time()
                        credentials.usage_count += 1
                        
                        # 记录访问日志
                        self._log_api_access(
                            exchange=exchange,
                            api_key_hash=self.encryption_manager.hash_api_key(credentials.api_key),
                            endpoint="get_credentials",
                            method="GET",
                            success=True
                        )
                        
                        return credentials
                
                logger.warning(f"未找到活跃的API凭证: {exchange}")
                return None
        
        except Exception as e:
            logger.error(f"获取API凭证失败: {e}")
            return None
    
    def update_credentials(self, exchange: str, **kwargs) -> bool:
        """更新API凭证"""
        try:
            with self.lock:
                for credential_id, credentials in self.credentials.items():
                    if credentials.exchange == exchange:
                        # 更新指定字段
                        for key, value in kwargs.items():
                            if hasattr(credentials, key):
                                setattr(credentials, key, value)
                        
                        # 保存更改
                        self._save_credentials()
                        
                        logger.info(f"API凭证更新成功: {exchange}")
                        return True
                
                logger.warning(f"未找到要更新的API凭证: {exchange}")
                return False
        
        except Exception as e:
            logger.error(f"更新API凭证失败: {e}")
            return False
    
    def remove_credentials(self, exchange: str) -> bool:
        """移除API凭证"""
        try:
            with self.lock:
                to_remove = []
                for credential_id, credentials in self.credentials.items():
                    if credentials.exchange == exchange:
                        to_remove.append(credential_id)
                
                for credential_id in to_remove:
                    del self.credentials[credential_id]
                
                if to_remove:
                    self._save_credentials()
                    logger.info(f"API凭证移除成功: {exchange}")
                    return True
                else:
                    logger.warning(f"未找到要移除的API凭证: {exchange}")
                    return False
        
        except Exception as e:
            logger.error(f"移除API凭证失败: {e}")
            return False
    
    def validate_api_access(self, exchange: str, required_permission: APIPermission) -> bool:
        """验证API访问权限"""
        try:
            credentials = self.get_credentials(exchange)
            if not credentials:
                return False
            
            # 检查权限
            if required_permission not in credentials.permissions:
                logger.warning(f"API权限不足: {exchange} - 需要 {required_permission.value}")
                return False
            
            # 检查安全级别
            if credentials.security_level == SecurityLevel.CRITICAL:
                # 关键级别需要额外验证
                if not self._additional_security_check(credentials):
                    logger.warning(f"关键安全级别验证失败: {exchange}")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"API访问验证失败: {e}")
            return False
    
    def record_api_failure(self, exchange: str, error_message: str):
        """记录API失败"""
        try:
            with self.lock:
                # 增加失败计数
                self.failed_attempts[exchange] = self.failed_attempts.get(exchange, 0) + 1
                
                # 检查是否需要锁定
                if self.failed_attempts[exchange] >= self.max_failed_attempts:
                    self.lockout_times[exchange] = time.time()
                    logger.warning(f"交易所API访问被锁定: {exchange} - 失败次数过多")
                
                # 记录访问日志
                credentials = self.get_credentials(exchange)
                if credentials:
                    self._log_api_access(
                        exchange=exchange,
                        api_key_hash=self.encryption_manager.hash_api_key(credentials.api_key),
                        endpoint="api_call",
                        method="POST",
                        success=False,
                        error_message=error_message
                    )
        
        except Exception as e:
            logger.error(f"记录API失败失败: {e}")
    
    def reset_failed_attempts(self, exchange: str):
        """重置失败尝试计数"""
        with self.lock:
            self.failed_attempts.pop(exchange, None)
            self.lockout_times.pop(exchange, None)
            logger.info(f"重置失败尝试计数: {exchange}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        try:
            with self.lock:
                status = {
                    'total_credentials': len(self.credentials),
                    'active_credentials': sum(1 for c in self.credentials.values() if c.is_active),
                    'locked_exchanges': list(self.lockout_times.keys()),
                    'failed_attempts': dict(self.failed_attempts),
                    'recent_access_logs': len([log for log in self.access_logs if time.time() - log.timestamp < 3600]),
                    'security_levels': {}
                }
                
                # 统计安全级别分布
                for credentials in self.credentials.values():
                    level = credentials.security_level.value
                    status['security_levels'][level] = status['security_levels'].get(level, 0) + 1
                
                return status
        
        except Exception as e:
            logger.error(f"获取安全状态失败: {e}")
            return {}
    
    def rotate_api_keys(self) -> Dict[str, bool]:
        """轮换API密钥"""
        try:
            results = {}
            
            with self.lock:
                for credential_id, credentials in self.credentials.items():
                    # 检查是否需要轮换
                    if time.time() - credentials.created_at > self.key_rotation_interval:
                        # 在实际应用中，这里应该调用交易所API来生成新的密钥
                        # 目前只是标记需要轮换
                        credentials.metadata['needs_rotation'] = True
                        results[credentials.exchange] = True
                        logger.info(f"标记API密钥需要轮换: {credentials.exchange}")
                    else:
                        results[credentials.exchange] = False
                
                self._save_credentials()
            
            return results
        
        except Exception as e:
            logger.error(f"API密钥轮换失败: {e}")
            return {}
    
    def get_access_logs(self, exchange: str = None, limit: int = 100) -> List[APIAccessLog]:
        """获取访问日志"""
        try:
            logs = self.access_logs
            
            if exchange:
                logs = [log for log in logs if log.exchange == exchange]
            
            # 按时间倒序排列
            logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return logs[:limit]
        
        except Exception as e:
            logger.error(f"获取访问日志失败: {e}")
            return []
    
    def _validate_api_key_format(self, exchange: str, api_key: str, api_secret: str) -> bool:
        """验证API密钥格式"""
        try:
            # 基本长度检查
            if len(api_key) < 10 or len(api_secret) < 10:
                return False
            
            # 交易所特定验证
            if exchange.lower() == 'binance':
                # 币安API密钥通常是64字符
                return len(api_key) >= 60 and len(api_secret) >= 60
            elif exchange.lower() == 'huobi':
                # 火币API密钥格式验证
                return len(api_key) >= 20 and len(api_secret) >= 40
            elif exchange.lower() == 'okx':
                # OKX API密钥格式验证
                return len(api_key) >= 30 and len(api_secret) >= 40
            
            # 默认验证
            return True
        
        except Exception as e:
            logger.error(f"API密钥格式验证失败: {e}")
            return False
    
    def _is_locked(self, exchange: str) -> bool:
        """检查是否被锁定"""
        if exchange not in self.lockout_times:
            return False
        
        # 检查锁定是否已过期
        if time.time() - self.lockout_times[exchange] > self.lockout_duration:
            # 锁定已过期，清除锁定状态
            self.lockout_times.pop(exchange, None)
            self.failed_attempts.pop(exchange, None)
            return False
        
        return True
    
    def _additional_security_check(self, credentials: APICredentials) -> bool:
        """额外安全检查"""
        try:
            # 检查使用频率
            if credentials.usage_count > 1000 and time.time() - credentials.last_used < 60:
                logger.warning("API使用频率过高")
                return False
            
            # 检查权限组合
            if APIPermission.WITHDRAW in credentials.permissions and APIPermission.FULL_ACCESS in credentials.permissions:
                logger.warning("高风险权限组合")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"额外安全检查失败: {e}")
            return False
    
    def _log_api_access(self, exchange: str, api_key_hash: str, endpoint: str,
                       method: str, success: bool, error_message: str = None,
                       ip_address: str = "127.0.0.1", user_agent: str = "TradingBot"):
        """记录API访问日志"""
        try:
            log = APIAccessLog(
                exchange=exchange,
                api_key_hash=api_key_hash,
                endpoint=endpoint,
                method=method,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=time.time(),
                success=success,
                error_message=error_message
            )
            
            self.access_logs.append(log)
            
            # 保持日志数量在合理范围内
            if len(self.access_logs) > 10000:
                self.access_logs = self.access_logs[-5000:]
        
        except Exception as e:
            logger.error(f"记录API访问日志失败: {e}")
    
    def _save_credentials(self):
        """保存凭证到文件"""
        try:
            if not self.encryption_manager:
                logger.warning("加密管理器不可用，跳过保存")
                return
            
            # 准备要保存的数据
            data_to_save = {}
            
            for credential_id, credentials in self.credentials.items():
                # 加密敏感数据
                encrypted_data = {
                    'exchange': credentials.exchange,
                    'api_key': self.encryption_manager.encrypt(credentials.api_key),
                    'api_secret': self.encryption_manager.encrypt(credentials.api_secret),
                    'passphrase': self.encryption_manager.encrypt(credentials.passphrase) if credentials.passphrase else None,
                    'permissions': [p.value for p in credentials.permissions],
                    'security_level': credentials.security_level.value,
                    'created_at': credentials.created_at,
                    'last_used': credentials.last_used,
                    'usage_count': credentials.usage_count,
                    'is_active': credentials.is_active,
                    'metadata': credentials.metadata
                }
                
                data_to_save[credential_id] = encrypted_data
            
            # 写入文件
            with open(self.storage_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            logger.debug("API凭证保存成功")
        
        except Exception as e:
            logger.error(f"保存API凭证失败: {e}")
    
    def _load_credentials(self):
        """从文件加载凭证"""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("凭证文件不存在，使用空凭证集")
                return
            
            if not self.encryption_manager:
                logger.warning("加密管理器不可用，无法加载凭证")
                return
            
            with open(self.storage_path, 'r') as f:
                encrypted_data = json.load(f)
            
            for credential_id, data in encrypted_data.items():
                try:
                    # 解密敏感数据
                    credentials = APICredentials(
                        exchange=data['exchange'],
                        api_key=self.encryption_manager.decrypt(data['api_key']),
                        api_secret=self.encryption_manager.decrypt(data['api_secret']),
                        passphrase=self.encryption_manager.decrypt(data['passphrase']) if data['passphrase'] else None,
                        permissions=[APIPermission(p) for p in data['permissions']],
                        security_level=SecurityLevel(data['security_level']),
                        created_at=data['created_at'],
                        last_used=data['last_used'],
                        usage_count=data['usage_count'],
                        is_active=data['is_active'],
                        metadata=data['metadata']
                    )
                    
                    self.credentials[credential_id] = credentials
                
                except Exception as e:
                    logger.error(f"加载凭证失败: {credential_id} - {e}")
            
            logger.info(f"API凭证加载完成: {len(self.credentials)}个")
        
        except Exception as e:
            logger.error(f"加载API凭证失败: {e}")

# 全局API安全管理器实例（需要在使用前初始化）
api_security_manager = None

def initialize_api_security(master_password: str, storage_path: str = "api_credentials.json"):
    """初始化API安全管理器"""
    global api_security_manager
    api_security_manager = APISecurityManager(master_password, storage_path)
    return api_security_manager


def initialize_api_security_manager():
    """初始化API安全管理器"""
    manager = initialize_api_security("default_password")
    logger.success("✅ API安全管理器初始化完成")
    return manager
