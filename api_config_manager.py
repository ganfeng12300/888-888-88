#!/usr/bin/env python3
"""
🔧 API配置管理器
API Configuration Manager

功能：
- 一键启动后的API输入界面
- 本地API配置保存和加载
- API修改功能
- 多交易所支持
- 安全的配置文件加密存储
"""

import os
import json
import getpass
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from cryptography.fernet import Fernet
import base64

class APIConfigManager:
    """API配置管理器"""
    
    def __init__(self, config_file: str = "exchange_configs.json"):
        self.config_file = config_file
        self.encrypted_config_file = f"{config_file}.enc"
        self.supported_exchanges = {
            "bitget": {
                "name": "Bitget",
                "fields": ["api_key", "secret_key", "passphrase"],
                "description": "Bitget合约交易所"
            },
            "binance": {
                "name": "Binance",
                "fields": ["api_key", "secret_key"],
                "description": "币安交易所"
            },
            "okx": {
                "name": "OKX",
                "fields": ["api_key", "secret_key", "passphrase"],
                "description": "OKX交易所"
            },
            "huobi": {
                "name": "Huobi",
                "fields": ["api_key", "secret_key"],
                "description": "火币交易所"
            },
            "bybit": {
                "name": "Bybit",
                "fields": ["api_key", "secret_key"],
                "description": "Bybit交易所"
            },
            "gate": {
                "name": "Gate.io",
                "fields": ["api_key", "secret_key"],
                "description": "Gate.io交易所"
            },
            "kucoin": {
                "name": "KuCoin",
                "fields": ["api_key", "secret_key", "passphrase"],
                "description": "KuCoin交易所"
            }
        }
        self.configs = {}
        self.encryption_key = None
        
    def _generate_key(self, password: str) -> bytes:
        """从密码生成加密密钥"""
        return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
    
    def _encrypt_data(self, data: str, password: str) -> bytes:
        """加密数据"""
        key = self._generate_key(password)
        f = Fernet(key)
        return f.encrypt(data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes, password: str) -> str:
        """解密数据"""
        key = self._generate_key(password)
        f = Fernet(key)
        return f.decrypt(encrypted_data).decode()
    
    def load_configs(self, password: Optional[str] = None) -> bool:
        """加载配置文件"""
        try:
            # 优先尝试加载加密配置
            if os.path.exists(self.encrypted_config_file):
                if not password:
                    password = getpass.getpass("🔐 请输入配置文件密码: ")
                
                with open(self.encrypted_config_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self._decrypt_data(encrypted_data, password)
                self.configs = json.loads(decrypted_data)
                self.encryption_key = password
                print("✅ 加密配置文件加载成功")
                return True
                
            # 尝试加载普通配置文件
            elif os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.configs = json.load(f)
                print("✅ 配置文件加载成功")
                return True
            else:
                print("📝 未找到配置文件，将创建新配置")
                self.configs = {}
                return False
                
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            self.configs = {}
            return False
    
    def save_configs(self, password: Optional[str] = None, encrypt: bool = True) -> bool:
        """保存配置文件"""
        try:
            config_data = {
                "last_updated": datetime.now().isoformat(),
                "exchanges": self.configs
            }
            
            if encrypt:
                if not password:
                    password = self.encryption_key or getpass.getpass("🔐 请设置配置文件密码: ")
                
                encrypted_data = self._encrypt_data(json.dumps(config_data, indent=2), password)
                
                with open(self.encrypted_config_file, 'wb') as f:
                    f.write(encrypted_data)
                
                # 删除明文配置文件
                if os.path.exists(self.config_file):
                    os.remove(self.config_file)
                
                self.encryption_key = password
                print("✅ 加密配置文件保存成功")
            else:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print("✅ 配置文件保存成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 配置文件保存失败: {e}")
            return False
    
    def display_supported_exchanges(self):
        """显示支持的交易所"""
        print("\n" + "="*60)
        print("🏦 支持的交易所列表")
        print("="*60)
        
        for i, (key, info) in enumerate(self.supported_exchanges.items(), 1):
            status = "✅ 已配置" if key in self.configs else "⚪ 未配置"
            print(f"{i:2d}. {info['name']:12} - {info['description']} {status}")
        
        print("="*60)
    
    def get_exchange_count(self) -> int:
        """获取已配置的交易所数量"""
        return len(self.configs)
    
    def input_exchange_config(self, exchange_key: str) -> bool:
        """输入交易所配置"""
        if exchange_key not in self.supported_exchanges:
            print(f"❌ 不支持的交易所: {exchange_key}")
            return False
        
        exchange_info = self.supported_exchanges[exchange_key]
        print(f"\n🔧 配置 {exchange_info['name']} API")
        print("-" * 40)
        
        config = {}
        
        for field in exchange_info['fields']:
            if 'secret' in field.lower() or 'passphrase' in field.lower():
                value = getpass.getpass(f"请输入 {field}: ")
            else:
                value = input(f"请输入 {field}: ").strip()
            
            if not value:
                print(f"❌ {field} 不能为空")
                return False
            
            config[field] = value
        
        # 测试连接
        print("🔍 测试API连接...")
        if self._test_api_connection(exchange_key, config):
            self.configs[exchange_key] = {
                **config,
                "created_at": datetime.now().isoformat(),
                "last_tested": datetime.now().isoformat(),
                "status": "active"
            }
            print(f"✅ {exchange_info['name']} API配置成功")
            return True
        else:
            print(f"❌ {exchange_info['name']} API连接测试失败")
            retry = input("是否仍要保存配置? (y/N): ").lower()
            if retry == 'y':
                self.configs[exchange_key] = {
                    **config,
                    "created_at": datetime.now().isoformat(),
                    "last_tested": datetime.now().isoformat(),
                    "status": "error"
                }
                return True
            return False
    
    def _test_api_connection(self, exchange_key: str, config: Dict) -> bool:
        """测试API连接"""
        try:
            if exchange_key == "bitget":
                from src.exchanges.bitget_api import BitgetAPI, BitgetConfig
                
                bitget_config = BitgetConfig(
                    api_key=config['api_key'],
                    secret_key=config['secret_key'],
                    passphrase=config['passphrase']
                )
                api = BitgetAPI(bitget_config)
                
                # 测试获取账户信息
                account_info = api.get_futures_account()
                return account_info is not None
                
            # 其他交易所的测试逻辑可以在这里添加
            else:
                print(f"⚠️ {exchange_key} 连接测试暂未实现")
                return True
                
        except Exception as e:
            print(f"❌ API连接测试失败: {e}")
            return False
    
    def modify_exchange_config(self, exchange_key: str) -> bool:
        """修改交易所配置"""
        if exchange_key not in self.configs:
            print(f"❌ 未找到 {exchange_key} 的配置")
            return False
        
        exchange_info = self.supported_exchanges[exchange_key]
        current_config = self.configs[exchange_key]
        
        print(f"\n✏️ 修改 {exchange_info['name']} API配置")
        print("-" * 40)
        print("提示: 直接回车保持原值不变")
        
        new_config = {}
        
        for field in exchange_info['fields']:
            current_value = current_config.get(field, "")
            masked_value = f"{current_value[:8]}***{current_value[-4:]}" if len(current_value) > 12 else "***"
            
            if 'secret' in field.lower() or 'passphrase' in field.lower():
                new_value = getpass.getpass(f"{field} (当前: {masked_value}): ")
            else:
                new_value = input(f"{field} (当前: {masked_value}): ").strip()
            
            new_config[field] = new_value if new_value else current_config[field]
        
        # 测试新配置
        print("🔍 测试新API配置...")
        if self._test_api_connection(exchange_key, new_config):
            self.configs[exchange_key].update(new_config)
            self.configs[exchange_key]["last_updated"] = datetime.now().isoformat()
            self.configs[exchange_key]["last_tested"] = datetime.now().isoformat()
            self.configs[exchange_key]["status"] = "active"
            print(f"✅ {exchange_info['name']} API配置更新成功")
            return True
        else:
            print(f"❌ {exchange_info['name']} API连接测试失败")
            return False
    
    def delete_exchange_config(self, exchange_key: str) -> bool:
        """删除交易所配置"""
        if exchange_key not in self.configs:
            print(f"❌ 未找到 {exchange_key} 的配置")
            return False
        
        exchange_info = self.supported_exchanges[exchange_key]
        confirm = input(f"⚠️ 确定要删除 {exchange_info['name']} 的配置吗? (y/N): ").lower()
        
        if confirm == 'y':
            del self.configs[exchange_key]
            print(f"✅ {exchange_info['name']} 配置已删除")
            return True
        else:
            print("❌ 取消删除操作")
            return False
    
    def display_current_configs(self):
        """显示当前配置"""
        if not self.configs:
            print("📝 暂无交易所配置")
            return
        
        print("\n" + "="*60)
        print("📊 当前交易所配置")
        print("="*60)
        
        for exchange_key, config in self.configs.items():
            exchange_info = self.supported_exchanges.get(exchange_key, {"name": exchange_key})
            status_icon = "✅" if config.get("status") == "active" else "❌"
            
            print(f"{status_icon} {exchange_info['name']}")
            print(f"   创建时间: {config.get('created_at', 'N/A')}")
            print(f"   最后测试: {config.get('last_tested', 'N/A')}")
            print(f"   状态: {config.get('status', 'unknown')}")
            print()
    
    def get_exchange_config(self, exchange_key: str) -> Optional[Dict]:
        """获取指定交易所配置"""
        return self.configs.get(exchange_key)
    
    def get_all_configs(self) -> Dict:
        """获取所有配置"""
        return self.configs.copy()
    
    def interactive_setup(self):
        """交互式设置"""
        print("\n" + "="*60)
        print("🚀 AI量化交易系统 - API配置管理")
        print("="*60)
        
        # 加载现有配置
        self.load_configs()
        
        while True:
            print(f"\n当前已配置 {self.get_exchange_count()} 个交易所")
            self.display_supported_exchanges()
            
            print("\n📋 操作菜单:")
            print("1. 添加新交易所配置")
            print("2. 修改现有配置")
            print("3. 删除配置")
            print("4. 查看当前配置")
            print("5. 测试所有连接")
            print("6. 保存并退出")
            print("0. 退出不保存")
            
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "1":
                self._add_exchange_config()
            elif choice == "2":
                self._modify_existing_config()
            elif choice == "3":
                self._delete_existing_config()
            elif choice == "4":
                self.display_current_configs()
            elif choice == "5":
                self._test_all_connections()
            elif choice == "6":
                if self.save_configs():
                    print("✅ 配置已保存，退出程序")
                    break
                else:
                    print("❌ 配置保存失败")
            elif choice == "0":
                print("❌ 退出程序，未保存更改")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _add_exchange_config(self):
        """添加交易所配置"""
        print("\n请选择要配置的交易所:")
        exchanges = list(self.supported_exchanges.keys())
        
        for i, key in enumerate(exchanges, 1):
            info = self.supported_exchanges[key]
            status = "✅ 已配置" if key in self.configs else "⚪ 未配置"
            print(f"{i}. {info['name']} {status}")
        
        try:
            choice = int(input(f"\n请选择 (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                exchange_key = exchanges[choice]
                if exchange_key in self.configs:
                    overwrite = input(f"⚠️ {self.supported_exchanges[exchange_key]['name']} 已配置，是否覆盖? (y/N): ").lower()
                    if overwrite != 'y':
                        return
                
                self.input_exchange_config(exchange_key)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入有效数字")
    
    def _modify_existing_config(self):
        """修改现有配置"""
        if not self.configs:
            print("❌ 暂无配置可修改")
            return
        
        print("\n请选择要修改的交易所:")
        exchanges = list(self.configs.keys())
        
        for i, key in enumerate(exchanges, 1):
            info = self.supported_exchanges[key]
            print(f"{i}. {info['name']}")
        
        try:
            choice = int(input(f"\n请选择 (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                exchange_key = exchanges[choice]
                self.modify_exchange_config(exchange_key)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入有效数字")
    
    def _delete_existing_config(self):
        """删除现有配置"""
        if not self.configs:
            print("❌ 暂无配置可删除")
            return
        
        print("\n请选择要删除的交易所:")
        exchanges = list(self.configs.keys())
        
        for i, key in enumerate(exchanges, 1):
            info = self.supported_exchanges[key]
            print(f"{i}. {info['name']}")
        
        try:
            choice = int(input(f"\n请选择 (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                exchange_key = exchanges[choice]
                self.delete_exchange_config(exchange_key)
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入有效数字")
    
    def _test_all_connections(self):
        """测试所有连接"""
        if not self.configs:
            print("❌ 暂无配置可测试")
            return
        
        print("\n🔍 测试所有API连接...")
        print("-" * 40)
        
        for exchange_key, config in self.configs.items():
            exchange_info = self.supported_exchanges[exchange_key]
            print(f"测试 {exchange_info['name']}...", end=" ")
            
            if self._test_api_connection(exchange_key, config):
                print("✅ 连接成功")
                self.configs[exchange_key]["status"] = "active"
                self.configs[exchange_key]["last_tested"] = datetime.now().isoformat()
            else:
                print("❌ 连接失败")
                self.configs[exchange_key]["status"] = "error"
                self.configs[exchange_key]["last_tested"] = datetime.now().isoformat()

def main():
    """主函数"""
    try:
        # 检查依赖
        import cryptography
    except ImportError:
        print("❌ 缺少加密依赖，正在安装...")
        os.system("pip install cryptography")
        import cryptography
    
    manager = APIConfigManager()
    manager.interactive_setup()

if __name__ == "__main__":
    main()
