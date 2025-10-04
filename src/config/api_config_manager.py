#!/usr/bin/env python3
"""
🔐 API配置管理器
生产级交易所API密钥管理系统
支持安全存储、加密、验证和一键配置
"""

import os
import json
import getpass
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from loguru import logger


@dataclass
class ExchangeCredentials:
    """交易所凭证"""
    exchange_name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox: bool = False
    enabled: bool = True
    created_at: str = ""
    last_used: str = ""


@dataclass
class APIConfiguration:
    """API配置"""
    credentials: Dict[str, ExchangeCredentials]
    default_exchange: str = "binance"
    risk_limits: Dict[str, Any] = None
    trading_pairs: List[str] = None
    
    def __post_init__(self):
        if self.risk_limits is None:
            self.risk_limits = {
                "max_position_size": 0.1,
                "max_daily_loss": 0.02,
                "stop_loss": 0.05,
                "take_profit": 0.15
            }
        if self.trading_pairs is None:
            self.trading_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]


class APIConfigManager:
    """API配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        
        self.config_file = self.config_dir / "api_config.enc"
        self.key_file = self.config_dir / "master.key"
        
        self._fernet = None
        self._config: Optional[APIConfiguration] = None
        
        # 支持的交易所
        self.supported_exchanges = {
            "binance": {
                "name": "Binance",
                "api_url": "https://api.binance.com",
                "testnet_url": "https://testnet.binance.vision",
                "requires_passphrase": False
            },
            "okx": {
                "name": "OKX",
                "api_url": "https://www.okx.com",
                "testnet_url": "https://www.okx.com",
                "requires_passphrase": True
            },
            "bybit": {
                "name": "Bybit",
                "api_url": "https://api.bybit.com",
                "testnet_url": "https://api-testnet.bybit.com",
                "requires_passphrase": False
            },
            "huobi": {
                "name": "Huobi",
                "api_url": "https://api.huobi.pro",
                "testnet_url": "https://api.testnet.huobi.pro",
                "requires_passphrase": False
            },
            "kucoin": {
                "name": "KuCoin",
                "api_url": "https://api.kucoin.com",
                "testnet_url": "https://openapi-sandbox.kucoin.com",
                "requires_passphrase": True
            }
        }
    
    def _generate_key(self, password: str) -> bytes:
        """生成加密密钥"""
        password_bytes = password.encode()
        salt = b'stable_salt_for_api_config'  # 生产环境应使用随机盐
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _get_fernet(self, password: str = None) -> Fernet:
        """获取加密器"""
        if self._fernet is None:
            if password is None:
                if self.key_file.exists():
                    # 使用保存的密钥
                    with open(self.key_file, 'rb') as f:
                        key = f.read()
                else:
                    # 首次使用，创建新密钥
                    password = getpass.getpass("请设置主密码（用于加密API密钥）: ")
                    key = self._generate_key(password)
                    with open(self.key_file, 'wb') as f:
                        f.write(key)
                    os.chmod(self.key_file, 0o600)
            else:
                key = self._generate_key(password)
            
            self._fernet = Fernet(key)
        
        return self._fernet
    
    def initialize_config(self, password: str = None) -> bool:
        """初始化配置"""
        try:
            self._get_fernet(password)
            
            if not self.config_file.exists():
                # 创建默认配置
                default_config = APIConfiguration(
                    credentials={},
                    default_exchange="binance"
                )
                self._save_config(default_config)
                logger.info("✅ 创建默认API配置")
            
            self._load_config()
            return True
            
        except Exception as e:
            logger.error(f"❌ 初始化配置失败: {e}")
            return False
    
    def _save_config(self, config: APIConfiguration):
        """保存配置"""
        try:
            config_data = asdict(config)
            config_json = json.dumps(config_data, indent=2, ensure_ascii=False)
            
            encrypted_data = self._get_fernet().encrypt(config_json.encode())
            
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_data)
            
            os.chmod(self.config_file, 0o600)
            self._config = config
            
        except Exception as e:
            logger.error(f"❌ 保存配置失败: {e}")
            raise
    
    def _load_config(self) -> APIConfiguration:
        """加载配置"""
        try:
            if not self.config_file.exists():
                raise FileNotFoundError("配置文件不存在")
            
            with open(self.config_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._get_fernet().decrypt(encrypted_data)
            config_data = json.loads(decrypted_data.decode())
            
            # 重建凭证对象
            credentials = {}
            for name, cred_data in config_data.get('credentials', {}).items():
                credentials[name] = ExchangeCredentials(**cred_data)
            
            config_data['credentials'] = credentials
            self._config = APIConfiguration(**config_data)
            
            return self._config
            
        except Exception as e:
            logger.error(f"❌ 加载配置失败: {e}")
            raise
    
    def add_exchange_credentials(self, exchange_name: str, api_key: str, 
                               api_secret: str, passphrase: str = None,
                               sandbox: bool = False) -> bool:
        """添加交易所凭证"""
        try:
            if exchange_name not in self.supported_exchanges:
                logger.error(f"❌ 不支持的交易所: {exchange_name}")
                return False
            
            # 验证API密钥格式
            if not self._validate_api_credentials(exchange_name, api_key, api_secret):
                logger.error(f"❌ API密钥格式无效: {exchange_name}")
                return False
            
            # 检查是否需要passphrase
            if self.supported_exchanges[exchange_name]["requires_passphrase"] and not passphrase:
                logger.error(f"❌ {exchange_name} 需要passphrase")
                return False
            
            if self._config is None:
                self._load_config()
            
            credentials = ExchangeCredentials(
                exchange_name=exchange_name,
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                sandbox=sandbox,
                enabled=True,
                created_at=str(int(time.time())),
                last_used=""
            )
            
            self._config.credentials[exchange_name] = credentials
            self._save_config(self._config)
            
            logger.info(f"✅ 添加 {exchange_name} API凭证成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 添加API凭证失败: {e}")
            return False
    
    def _validate_api_credentials(self, exchange: str, api_key: str, api_secret: str) -> bool:
        """验证API凭证格式"""
        if not api_key or not api_secret:
            return False
        
        # 基本长度检查
        if len(api_key) < 10 or len(api_secret) < 10:
            return False
        
        # 交易所特定验证
        if exchange == "binance":
            return len(api_key) >= 60 and len(api_secret) >= 60
        elif exchange == "okx":
            return len(api_key) >= 20 and len(api_secret) >= 40
        elif exchange == "bybit":
            return len(api_key) >= 30 and len(api_secret) >= 40
        
        return True
    
    def get_exchange_credentials(self, exchange_name: str) -> Optional[ExchangeCredentials]:
        """获取交易所凭证"""
        try:
            if self._config is None:
                self._load_config()
            
            return self._config.credentials.get(exchange_name)
            
        except Exception as e:
            logger.error(f"❌ 获取API凭证失败: {e}")
            return None
    
    def list_configured_exchanges(self) -> List[str]:
        """列出已配置的交易所"""
        try:
            if self._config is None:
                self._load_config()
            
            return list(self._config.credentials.keys())
            
        except Exception as e:
            logger.error(f"❌ 列出交易所失败: {e}")
            return []
    
    def remove_exchange_credentials(self, exchange_name: str) -> bool:
        """移除交易所凭证"""
        try:
            if self._config is None:
                self._load_config()
            
            if exchange_name in self._config.credentials:
                del self._config.credentials[exchange_name]
                self._save_config(self._config)
                logger.info(f"✅ 移除 {exchange_name} API凭证成功")
                return True
            else:
                logger.warning(f"⚠️ 未找到 {exchange_name} 的API凭证")
                return False
                
        except Exception as e:
            logger.error(f"❌ 移除API凭证失败: {e}")
            return False
    
    def test_exchange_connection(self, exchange_name: str) -> bool:
        """测试交易所连接"""
        try:
            credentials = self.get_exchange_credentials(exchange_name)
            if not credentials:
                logger.error(f"❌ 未找到 {exchange_name} 的API凭证")
                return False
            
            # 这里应该实现实际的连接测试
            # 暂时返回True，实际实现中应该调用交易所API
            logger.info(f"🔍 测试 {exchange_name} 连接...")
            
            # TODO: 实现真实的API连接测试
            import asyncio
            import aiohttp
            
            async def test_connection():
                exchange_info = self.supported_exchanges[exchange_name]
                test_url = exchange_info["testnet_url"] if credentials.sandbox else exchange_info["api_url"]
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{test_url}/api/v3/ping", timeout=5) as response:
                            return response.status == 200
                    except:
                        return False
            
            result = asyncio.run(test_connection())
            
            if result:
                logger.info(f"✅ {exchange_name} 连接测试成功")
            else:
                logger.error(f"❌ {exchange_name} 连接测试失败")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 测试连接失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        try:
            if self._config is None:
                self._load_config()
            
            summary = {
                "configured_exchanges": len(self._config.credentials),
                "default_exchange": self._config.default_exchange,
                "exchanges": {}
            }
            
            for name, cred in self._config.credentials.items():
                summary["exchanges"][name] = {
                    "enabled": cred.enabled,
                    "sandbox": cred.sandbox,
                    "has_passphrase": bool(cred.passphrase),
                    "created_at": cred.created_at,
                    "last_used": cred.last_used
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 获取配置摘要失败: {e}")
            return {}
    
    def interactive_setup(self) -> bool:
        """交互式设置"""
        try:
            print("\n🚀 888-888-88 量化交易系统 - API配置向导")
            print("=" * 50)
            
            # 初始化配置
            if not self.initialize_config():
                return False
            
            while True:
                print("\n📋 支持的交易所:")
                for i, (key, info) in enumerate(self.supported_exchanges.items(), 1):
                    status = "✅" if key in self.list_configured_exchanges() else "⚪"
                    print(f"  {i}. {status} {info['name']} ({key})")
                
                print("\n🔧 操作选项:")
                print("  a. 添加/更新交易所API")
                print("  t. 测试连接")
                print("  l. 查看配置")
                print("  d. 删除配置")
                print("  q. 完成配置")
                
                choice = input("\n请选择操作 (a/t/l/d/q): ").lower().strip()
                
                if choice == 'q':
                    break
                elif choice == 'a':
                    self._interactive_add_exchange()
                elif choice == 't':
                    self._interactive_test_connection()
                elif choice == 'l':
                    self._show_config_summary()
                elif choice == 'd':
                    self._interactive_remove_exchange()
                else:
                    print("❌ 无效选择，请重试")
            
            # 验证至少配置了一个交易所
            configured = self.list_configured_exchanges()
            if not configured:
                print("⚠️ 警告: 未配置任何交易所API")
                return False
            
            print(f"\n✅ 配置完成！已配置 {len(configured)} 个交易所")
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消配置")
            return False
        except Exception as e:
            logger.error(f"❌ 交互式设置失败: {e}")
            return False
    
    def _interactive_add_exchange(self):
        """交互式添加交易所"""
        try:
            print("\n📝 添加交易所API配置")
            print("-" * 30)
            
            # 选择交易所
            exchanges = list(self.supported_exchanges.keys())
            for i, exchange in enumerate(exchanges, 1):
                print(f"  {i}. {self.supported_exchanges[exchange]['name']}")
            
            while True:
                try:
                    choice = int(input("请选择交易所 (1-{}): ".format(len(exchanges))))
                    if 1 <= choice <= len(exchanges):
                        exchange_name = exchanges[choice - 1]
                        break
                    else:
                        print("❌ 无效选择")
                except ValueError:
                    print("❌ 请输入数字")
            
            exchange_info = self.supported_exchanges[exchange_name]
            print(f"\n配置 {exchange_info['name']} API:")
            
            # 输入API密钥
            api_key = input("API Key: ").strip()
            if not api_key:
                print("❌ API Key不能为空")
                return
            
            api_secret = getpass.getpass("API Secret: ").strip()
            if not api_secret:
                print("❌ API Secret不能为空")
                return
            
            passphrase = None
            if exchange_info["requires_passphrase"]:
                passphrase = getpass.getpass("Passphrase: ").strip()
                if not passphrase:
                    print("❌ Passphrase不能为空")
                    return
            
            # 选择环境
            sandbox = input("使用测试环境? (y/N): ").lower().strip() == 'y'
            
            # 添加凭证
            if self.add_exchange_credentials(exchange_name, api_key, api_secret, passphrase, sandbox):
                print(f"✅ {exchange_info['name']} API配置成功")
                
                # 测试连接
                test = input("是否测试连接? (Y/n): ").lower().strip()
                if test != 'n':
                    self.test_exchange_connection(exchange_name)
            else:
                print(f"❌ {exchange_info['name']} API配置失败")
                
        except Exception as e:
            logger.error(f"❌ 添加交易所失败: {e}")
    
    def _interactive_test_connection(self):
        """交互式测试连接"""
        configured = self.list_configured_exchanges()
        if not configured:
            print("❌ 未配置任何交易所")
            return
        
        print("\n🔍 测试交易所连接")
        print("-" * 20)
        
        for exchange in configured:
            print(f"测试 {exchange}...")
            self.test_exchange_connection(exchange)
    
    def _show_config_summary(self):
        """显示配置摘要"""
        summary = self.get_config_summary()
        
        print("\n📊 当前配置摘要")
        print("-" * 20)
        print(f"已配置交易所: {summary['configured_exchanges']}")
        print(f"默认交易所: {summary['default_exchange']}")
        
        if summary['exchanges']:
            print("\n交易所详情:")
            for name, info in summary['exchanges'].items():
                status = "✅ 启用" if info['enabled'] else "❌ 禁用"
                env = "🧪 测试" if info['sandbox'] else "🔴 实盘"
                print(f"  {name}: {status} | {env}")
    
    def _interactive_remove_exchange(self):
        """交互式移除交易所"""
        configured = self.list_configured_exchanges()
        if not configured:
            print("❌ 未配置任何交易所")
            return
        
        print("\n🗑️ 移除交易所配置")
        print("-" * 20)
        
        for i, exchange in enumerate(configured, 1):
            print(f"  {i}. {exchange}")
        
        try:
            choice = int(input(f"请选择要移除的交易所 (1-{len(configured)}): "))
            if 1 <= choice <= len(configured):
                exchange_name = configured[choice - 1]
                confirm = input(f"确认移除 {exchange_name}? (y/N): ").lower().strip()
                if confirm == 'y':
                    self.remove_exchange_credentials(exchange_name)
                else:
                    print("❌ 取消移除")
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入数字")


def main():
    """主函数 - 用于独立运行配置向导"""
    import time
    
    config_manager = APIConfigManager()
    
    if config_manager.interactive_setup():
        print("\n🎉 API配置完成！")
        print("现在可以启动交易系统了:")
        print("  python start_production_system.py")
    else:
        print("\n❌ API配置失败")
        exit(1)


if __name__ == "__main__":
    main()
