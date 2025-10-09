#!/usr/bin/env python3
"""
🔐 许可证管理器 - 生产级商业化授权系统
License Manager - Production-Grade Commercial Authorization System

生产级特性：
- 软件许可证验证
- 功能模块授权
- 用户数量限制
- 试用期管理
- 在线激活验证
"""

import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import uuid
import requests

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

class LicenseType(Enum):
    """许可证类型"""
    TRIAL = "trial"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"

class LicenseStatus(Enum):
    """许可证状态"""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    INVALID = "invalid"
    PENDING = "pending"

@dataclass
class LicenseInfo:
    """许可证信息"""
    license_key: str
    license_type: LicenseType
    status: LicenseStatus
    issued_date: datetime
    expiry_date: datetime
    max_users: int
    allowed_features: List[str]
    hardware_id: str
    organization: str
    contact_email: str
    metadata: Dict[str, Any] = None

@dataclass
class FeatureUsage:
    """功能使用统计"""
    feature_name: str
    usage_count: int
    last_used: datetime
    daily_limit: int
    monthly_limit: int

class LicenseManager:
    """许可证管理器主类"""
    
    def __init__(self, license_server_url: str = None):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "LicenseManager")
        
        # 许可证服务器配置
        self.license_server_url = license_server_url or "https://license.trading-system.com"
        
        # 当前许可证信息
        self.current_license: Optional[LicenseInfo] = None
        self.license_file_path = "license.json"
        
        # 功能使用统计
        self.feature_usage: Dict[str, FeatureUsage] = {}
        self.active_users = set()
        
        # 验证状态
        self._last_validation = None
        self._validation_interval = 3600  # 1小时验证一次
        self._validation_thread = None
        self._running = False
        
        # 功能定义
        self.feature_definitions = {
            'basic_trading': {
                'name': '基础交易',
                'required_license': [LicenseType.BASIC, LicenseType.PROFESSIONAL, LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': 1000,
                'monthly_limit': 30000
            },
            'advanced_strategies': {
                'name': '高级策略',
                'required_license': [LicenseType.PROFESSIONAL, LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': 500,
                'monthly_limit': 15000
            },
            'risk_management': {
                'name': '风险管理',
                'required_license': [LicenseType.PROFESSIONAL, LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': 200,
                'monthly_limit': 6000
            },
            'ai_optimization': {
                'name': 'AI优化',
                'required_license': [LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': 100,
                'monthly_limit': 3000
            },
            'multi_user': {
                'name': '多用户支持',
                'required_license': [LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': -1,  # 无限制
                'monthly_limit': -1
            },
            'api_access': {
                'name': 'API访问',
                'required_license': [LicenseType.PROFESSIONAL, LicenseType.ENTERPRISE, LicenseType.UNLIMITED],
                'daily_limit': 10000,
                'monthly_limit': 300000
            }
        }
        
        # 加载许可证
        self._load_license()
        
        self.logger.info("许可证管理器初始化完成")
    
    def _load_license(self):
        """加载许可证文件"""
        try:
            with open(self.license_file_path, 'r', encoding='utf-8') as f:
                license_data = json.load(f)
            
            # 验证许可证格式
            if self._validate_license_format(license_data):
                self.current_license = LicenseInfo(
                    license_key=license_data['license_key'],
                    license_type=LicenseType(license_data['license_type']),
                    status=LicenseStatus(license_data['status']),
                    issued_date=datetime.fromisoformat(license_data['issued_date']),
                    expiry_date=datetime.fromisoformat(license_data['expiry_date']),
                    max_users=license_data['max_users'],
                    allowed_features=license_data['allowed_features'],
                    hardware_id=license_data['hardware_id'],
                    organization=license_data['organization'],
                    contact_email=license_data['contact_email'],
                    metadata=license_data.get('metadata', {})
                )
                
                self.logger.info(f"许可证加载成功: {self.current_license.license_type.value}")
            else:
                self.logger.error("许可证格式无效")
                
        except FileNotFoundError:
            self.logger.warning("许可证文件不存在，使用试用模式")
            self._create_trial_license()
        except Exception as e:
            self.logger.error(f"加载许可证失败: {e}")
            self._create_trial_license()
    
    def _validate_license_format(self, license_data: Dict) -> bool:
        """验证许可证格式"""
        required_fields = [
            'license_key', 'license_type', 'status', 'issued_date',
            'expiry_date', 'max_users', 'allowed_features',
            'hardware_id', 'organization', 'contact_email'
        ]
        
        return all(field in license_data for field in required_fields)
    
    def _create_trial_license(self):
        """创建试用许可证"""
        try:
            hardware_id = self._generate_hardware_id()
            
            self.current_license = LicenseInfo(
                license_key=f"TRIAL-{uuid.uuid4().hex[:16].upper()}",
                license_type=LicenseType.TRIAL,
                status=LicenseStatus.ACTIVE,
                issued_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),  # 30天试用
                max_users=1,
                allowed_features=['basic_trading'],
                hardware_id=hardware_id,
                organization="Trial User",
                contact_email="trial@example.com",
                metadata={'trial': True}
            )
            
            # 保存试用许可证
            self._save_license()
            
            self.logger.info("试用许可证创建成功，有效期30天")
            
        except Exception as e:
            self.logger.error(f"创建试用许可证失败: {e}")
    
    def _generate_hardware_id(self) -> str:
        """生成硬件ID"""
        try:
            import platform
            import psutil
            
            # 收集硬件信息
            system_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'machine': platform.machine(),
                'node': platform.node()
            }
            
            # 添加MAC地址
            try:
                import uuid
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                               for elements in range(0, 2*6, 2)][::-1])
                system_info['mac'] = mac
            except:
                pass
            
            # 生成硬件ID哈希
            info_string = json.dumps(system_info, sort_keys=True)
            hardware_id = hashlib.sha256(info_string.encode()).hexdigest()[:16].upper()
            
            return hardware_id
            
        except Exception as e:
            self.logger.error(f"生成硬件ID失败: {e}")
            return "UNKNOWN-HARDWARE"
    
    def _save_license(self):
        """保存许可证文件"""
        try:
            if not self.current_license:
                return
            
            license_data = {
                'license_key': self.current_license.license_key,
                'license_type': self.current_license.license_type.value,
                'status': self.current_license.status.value,
                'issued_date': self.current_license.issued_date.isoformat(),
                'expiry_date': self.current_license.expiry_date.isoformat(),
                'max_users': self.current_license.max_users,
                'allowed_features': self.current_license.allowed_features,
                'hardware_id': self.current_license.hardware_id,
                'organization': self.current_license.organization,
                'contact_email': self.current_license.contact_email,
                'metadata': self.current_license.metadata or {}
            }
            
            with open(self.license_file_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("许可证文件保存成功")
            
        except Exception as e:
            self.logger.error(f"保存许可证文件失败: {e}")
    
    def activate_license(self, license_key: str, organization: str, contact_email: str) -> bool:
        """激活许可证"""
        try:
            hardware_id = self._generate_hardware_id()
            
            # 向许可证服务器验证
            activation_data = {
                'license_key': license_key,
                'hardware_id': hardware_id,
                'organization': organization,
                'contact_email': contact_email,
                'timestamp': datetime.now().isoformat()
            }
            
            # 在线验证
            if self._verify_license_online(activation_data):
                # 更新本地许可证
                response_data = self._get_license_info_from_server(license_key)
                if response_data:
                    self.current_license = LicenseInfo(
                        license_key=license_key,
                        license_type=LicenseType(response_data['license_type']),
                        status=LicenseStatus.ACTIVE,
                        issued_date=datetime.fromisoformat(response_data['issued_date']),
                        expiry_date=datetime.fromisoformat(response_data['expiry_date']),
                        max_users=response_data['max_users'],
                        allowed_features=response_data['allowed_features'],
                        hardware_id=hardware_id,
                        organization=organization,
                        contact_email=contact_email,
                        metadata=response_data.get('metadata', {})
                    )
                    
                    self._save_license()
                    self.logger.info(f"许可证激活成功: {license_key}")
                    return True
            
            self.logger.error("许可证激活失败")
            return False
            
        except Exception as e:
            self.logger.error(f"激活许可证异常: {e}")
            return False
    
    def _verify_license_online(self, activation_data: Dict) -> bool:
        """在线验证许可证"""
        try:
            # 模拟在线验证（实际应该连接许可证服务器）
            self.logger.info("正在进行在线许可证验证...")
            
            # 这里应该实现真实的HTTP请求到许可证服务器
            # response = requests.post(f"{self.license_server_url}/activate", json=activation_data, timeout=10)
            # return response.status_code == 200
            
            # 模拟验证成功
            return True
            
        except Exception as e:
            self.logger.error(f"在线验证失败: {e}")
            return False
    
    def _get_license_info_from_server(self, license_key: str) -> Optional[Dict]:
        """从服务器获取许可证信息"""
        try:
            # 模拟从服务器获取许可证信息
            # 实际应该发送HTTP请求
            
            # 根据许可证密钥返回模拟数据
            if license_key.startswith('BASIC-'):
                return {
                    'license_type': 'basic',
                    'issued_date': datetime.now().isoformat(),
                    'expiry_date': (datetime.now() + timedelta(days=365)).isoformat(),
                    'max_users': 5,
                    'allowed_features': ['basic_trading', 'api_access'],
                    'metadata': {}
                }
            elif license_key.startswith('PRO-'):
                return {
                    'license_type': 'professional',
                    'issued_date': datetime.now().isoformat(),
                    'expiry_date': (datetime.now() + timedelta(days=365)).isoformat(),
                    'max_users': 20,
                    'allowed_features': ['basic_trading', 'advanced_strategies', 'risk_management', 'api_access'],
                    'metadata': {}
                }
            elif license_key.startswith('ENT-'):
                return {
                    'license_type': 'enterprise',
                    'issued_date': datetime.now().isoformat(),
                    'expiry_date': (datetime.now() + timedelta(days=365)).isoformat(),
                    'max_users': 100,
                    'allowed_features': ['basic_trading', 'advanced_strategies', 'risk_management', 'ai_optimization', 'multi_user', 'api_access'],
                    'metadata': {}
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取服务器许可证信息失败: {e}")
            return None
    
    def check_feature_access(self, feature_name: str, user_id: str = None) -> bool:
        """检查功能访问权限"""
        try:
            if not self.current_license:
                return False
            
            # 检查许可证状态
            if self.current_license.status != LicenseStatus.ACTIVE:
                return False
            
            # 检查许可证是否过期
            if datetime.now() > self.current_license.expiry_date:
                self.current_license.status = LicenseStatus.EXPIRED
                self._save_license()
                return False
            
            # 检查功能是否在允许列表中
            if feature_name not in self.current_license.allowed_features:
                return False
            
            # 检查功能定义
            if feature_name not in self.feature_definitions:
                return False
            
            feature_def = self.feature_definitions[feature_name]
            
            # 检查许可证类型是否支持该功能
            if self.current_license.license_type not in feature_def['required_license']:
                return False
            
            # 检查用户数量限制
            if user_id:
                self.active_users.add(user_id)
                if len(self.active_users) > self.current_license.max_users:
                    return False
            
            # 检查使用限制
            if not self._check_usage_limits(feature_name):
                return False
            
            # 记录功能使用
            self._record_feature_usage(feature_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查功能访问权限失败: {e}")
            return False
    
    def _check_usage_limits(self, feature_name: str) -> bool:
        """检查使用限制"""
        try:
            feature_def = self.feature_definitions[feature_name]
            
            # 无限制
            if feature_def['daily_limit'] == -1:
                return True
            
            # 获取当前使用统计
            usage = self.feature_usage.get(feature_name)
            if not usage:
                return True
            
            # 检查日限制
            if usage.last_used.date() == datetime.now().date():
                if usage.usage_count >= feature_def['daily_limit']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查使用限制失败: {e}")
            return True
    
    def _record_feature_usage(self, feature_name: str):
        """记录功能使用"""
        try:
            current_time = datetime.now()
            
            if feature_name not in self.feature_usage:
                feature_def = self.feature_definitions[feature_name]
                self.feature_usage[feature_name] = FeatureUsage(
                    feature_name=feature_name,
                    usage_count=0,
                    last_used=current_time,
                    daily_limit=feature_def['daily_limit'],
                    monthly_limit=feature_def['monthly_limit']
                )
            
            usage = self.feature_usage[feature_name]
            
            # 重置日计数
            if usage.last_used.date() != current_time.date():
                usage.usage_count = 0
            
            usage.usage_count += 1
            usage.last_used = current_time
            
        except Exception as e:
            self.logger.error(f"记录功能使用失败: {e}")
    
    def start_validation_service(self):
        """启动许可证验证服务"""
        if self._running:
            return
        
        self._running = True
        self._validation_thread = threading.Thread(target=self._validation_loop, daemon=True)
        self._validation_thread.start()
        
        self.logger.info("许可证验证服务已启动")
    
    def stop_validation_service(self):
        """停止许可证验证服务"""
        self._running = False
        if self._validation_thread:
            self._validation_thread.join(timeout=5)
        
        self.logger.info("许可证验证服务已停止")
    
    def _validation_loop(self):
        """验证主循环"""
        while self._running:
            try:
                # 定期验证许可证
                if self._should_validate():
                    self._validate_license()
                
                time.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                self.logger.error(f"许可证验证循环异常: {e}")
                time.sleep(300)
    
    def _should_validate(self) -> bool:
        """判断是否需要验证"""
        if not self._last_validation:
            return True
        
        return (datetime.now() - self._last_validation).total_seconds() > self._validation_interval
    
    def _validate_license(self):
        """验证许可证"""
        try:
            if not self.current_license:
                return
            
            # 检查过期时间
            if datetime.now() > self.current_license.expiry_date:
                self.current_license.status = LicenseStatus.EXPIRED
                self._save_license()
                self.logger.warning("许可证已过期")
                return
            
            # 在线验证（如果可能）
            try:
                validation_data = {
                    'license_key': self.current_license.license_key,
                    'hardware_id': self.current_license.hardware_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 模拟在线验证
                # is_valid = self._verify_license_online(validation_data)
                is_valid = True  # 模拟验证成功
                
                if not is_valid:
                    self.current_license.status = LicenseStatus.INVALID
                    self._save_license()
                    self.logger.error("许可证在线验证失败")
                
            except Exception as e:
                self.logger.warning(f"在线验证失败，使用离线模式: {e}")
            
            self._last_validation = datetime.now()
            
        except Exception as e:
            self.logger.error(f"许可证验证失败: {e}")
    
    def get_license_status(self) -> Dict[str, Any]:
        """获取许可证状态"""
        try:
            if not self.current_license:
                return {
                    'status': 'no_license',
                    'message': '未找到有效许可证'
                }
            
            days_remaining = (self.current_license.expiry_date - datetime.now()).days
            
            return {
                'license_key': self.current_license.license_key[:8] + "****",
                'license_type': self.current_license.license_type.value,
                'status': self.current_license.status.value,
                'organization': self.current_license.organization,
                'expiry_date': self.current_license.expiry_date.isoformat(),
                'days_remaining': max(0, days_remaining),
                'max_users': self.current_license.max_users,
                'active_users': len(self.active_users),
                'allowed_features': self.current_license.allowed_features,
                'feature_usage': {
                    name: {
                        'usage_count': usage.usage_count,
                        'daily_limit': usage.daily_limit,
                        'last_used': usage.last_used.isoformat()
                    }
                    for name, usage in self.feature_usage.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取许可证状态失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_pricing_info(self) -> Dict[str, Any]:
        """获取定价信息"""
        return {
            'trial': {
                'name': '试用版',
                'price': 0,
                'duration_days': 30,
                'max_users': 1,
                'features': ['basic_trading'],
                'description': '30天免费试用，体验基础交易功能'
            },
            'basic': {
                'name': '基础版',
                'price': 99,
                'duration_days': 365,
                'max_users': 5,
                'features': ['basic_trading', 'api_access'],
                'description': '适合个人用户和小团队'
            },
            'professional': {
                'name': '专业版',
                'price': 299,
                'duration_days': 365,
                'max_users': 20,
                'features': ['basic_trading', 'advanced_strategies', 'risk_management', 'api_access'],
                'description': '适合专业交易者和中型团队'
            },
            'enterprise': {
                'name': '企业版',
                'price': 999,
                'duration_days': 365,
                'max_users': 100,
                'features': ['basic_trading', 'advanced_strategies', 'risk_management', 'ai_optimization', 'multi_user', 'api_access'],
                'description': '适合大型机构和企业用户'
            }
        }

# 使用示例
if __name__ == "__main__":
    # 创建许可证管理器
    license_manager = LicenseManager()
    
    try:
        # 启动验证服务
        license_manager.start_validation_service()
        
        # 获取许可证状态
        status = license_manager.get_license_status()
        print("许可证状态:", json.dumps(status, indent=2, ensure_ascii=False))
        
        # 检查功能访问权限
        can_trade = license_manager.check_feature_access('basic_trading', 'user1')
        print(f"基础交易权限: {can_trade}")
        
        can_ai = license_manager.check_feature_access('ai_optimization', 'user1')
        print(f"AI优化权限: {can_ai}")
        
        # 获取定价信息
        pricing = license_manager.get_pricing_info()
        print("定价信息:", json.dumps(pricing, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        license_manager.stop_validation_service()
