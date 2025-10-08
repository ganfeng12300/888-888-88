#!/usr/bin/env python3
"""
🏥 系统健康检查工具
System Health Check Tool

检查终极合约交易系统的所有核心组件是否正常工作
"""

import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from loguru import logger

class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.results = {}
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.warnings = 0
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
        
    def run_check(self, check_name: str, check_func):
        """运行单个检查"""
        self.total_checks += 1
        logger.info(f"🔍 检查: {check_name}")
        
        try:
            start_time = time.time()
            result = check_func()
            duration = time.time() - start_time
            
            if result.get('status') == 'success':
                self.passed_checks += 1
                logger.success(f"✅ {check_name} - 通过 ({duration:.2f}s)")
            elif result.get('status') == 'warning':
                self.warnings += 1
                logger.warning(f"⚠️ {check_name} - 警告: {result.get('message', '')}")
            else:
                self.failed_checks += 1
                logger.error(f"❌ {check_name} - 失败: {result.get('message', '')}")
                
            self.results[check_name] = {
                'status': result.get('status', 'failed'),
                'message': result.get('message', ''),
                'duration': duration,
                'details': result.get('details', {})
            }
            
        except Exception as e:
            self.failed_checks += 1
            error_msg = f"异常: {str(e)}"
            logger.error(f"❌ {check_name} - {error_msg}")
            self.results[check_name] = {
                'status': 'failed',
                'message': error_msg,
                'duration': 0,
                'details': {'traceback': traceback.format_exc()}
            }
    
    def check_config_file(self):
        """检查配置文件"""
        try:
            config_path = Path('config.json')
            if not config_path.exists():
                return {'status': 'failed', 'message': 'config.json文件不存在'}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            required_sections = [
                'system', 'gpu_optimizer', 'bybit_trader', 
                'risk_controller', 'timezone_scheduler', 'fusion_system'
            ]
            
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                return {
                    'status': 'failed', 
                    'message': f'缺少配置节: {", ".join(missing_sections)}'
                }
            
            return {
                'status': 'success',
                'message': f'配置文件完整，包含{len(config)}个配置节',
                'details': {'sections': list(config.keys())}
            }
            
        except json.JSONDecodeError as e:
            return {'status': 'failed', 'message': f'JSON格式错误: {str(e)}'}
        except Exception as e:
            return {'status': 'failed', 'message': f'配置文件检查失败: {str(e)}'}
    
    def check_core_modules(self):
        """检查核心模块导入"""
        modules = {
            'GPU性能优化器': 'src.hardware.gpu_performance_optimizer',
            'Bybit交易器': 'src.exchange.bybit_contract_trader',
            '风险控制器': 'src.risk.advanced_risk_controller',
            '时区调度器': 'src.scheduler.timezone_scheduler',
            'AI融合系统': 'src.ai.six_agents_fusion_system',
            '系统启动器': 'start_ultimate_system'
        }
        
        failed_modules = []
        success_modules = []
        
        for name, module_path in modules.items():
            try:
                __import__(module_path)
                success_modules.append(name)
            except Exception as e:
                failed_modules.append(f"{name}: {str(e)}")
        
        if failed_modules:
            return {
                'status': 'failed',
                'message': f'模块导入失败: {", ".join(failed_modules)}',
                'details': {'failed': failed_modules, 'success': success_modules}
            }
        
        return {
            'status': 'success',
            'message': f'所有{len(modules)}个核心模块导入成功',
            'details': {'modules': success_modules}
        }
    
    def check_dependencies(self):
        """检查关键依赖包"""
        critical_packages = [
            'numpy', 'pandas', 'scipy', 'requests', 'psutil', 
            'pytz', 'loguru', 'websocket', 'ccxt', 'aiohttp'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return {
                'status': 'failed',
                'message': f'缺少关键依赖: {", ".join(missing_packages)}',
                'details': {'missing': missing_packages, 'installed': installed_packages}
            }
        
        return {
            'status': 'success',
            'message': f'所有{len(critical_packages)}个关键依赖已安装',
            'details': {'packages': installed_packages}
        }
    
    def check_system_launcher(self):
        """检查系统启动器"""
        try:
            from start_ultimate_system import UltimateSystemLauncher
            
            # 测试初始化
            launcher = UltimateSystemLauncher()
            
            # 检查配置加载
            if not hasattr(launcher, 'config') or not launcher.config:
                return {'status': 'failed', 'message': '启动器配置加载失败'}
            
            # 检查关键组件
            required_components = ['gpu_optimizer', 'bybit_trader', 'risk_controller', 'timezone_scheduler', 'fusion_system']
            missing_components = []
            
            for component in required_components:
                if component not in launcher.config:
                    missing_components.append(component)
            
            if missing_components:
                return {
                    'status': 'warning',
                    'message': f'启动器缺少组件配置: {", ".join(missing_components)}'
                }
            
            return {
                'status': 'success',
                'message': '系统启动器初始化成功',
                'details': {
                    'config_sections': len(launcher.config),
                    'components': required_components
                }
            }
            
        except Exception as e:
            return {'status': 'failed', 'message': f'启动器检查失败: {str(e)}'}
    
    def check_gpu_support(self):
        """检查GPU支持"""
        try:
            import psutil
            
            # 检查GPU相关包
            gpu_packages = []
            gpu_warnings = []
            
            try:
                import torch
                gpu_packages.append('PyTorch')
                if torch.cuda.is_available():
                    gpu_packages.append(f'CUDA (设备数: {torch.cuda.device_count()})')
            except ImportError:
                gpu_warnings.append('PyTorch未安装，GPU加速功能受限')
            
            try:
                import cupy
                gpu_packages.append('CuPy')
            except ImportError:
                gpu_warnings.append('CuPy未安装，部分GPU计算功能受限')
            
            # 检查系统GPU
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = [f"{gpu.name} ({gpu.memoryTotal}MB)" for gpu in gpus]
                    gpu_packages.extend(gpu_info)
            except:
                gpu_warnings.append('无法检测GPU硬件信息')
            
            if gpu_warnings:
                return {
                    'status': 'warning',
                    'message': f'GPU支持有限: {"; ".join(gpu_warnings)}',
                    'details': {'available': gpu_packages, 'warnings': gpu_warnings}
                }
            
            return {
                'status': 'success',
                'message': f'GPU支持良好',
                'details': {'support': gpu_packages}
            }
            
        except Exception as e:
            return {'status': 'warning', 'message': f'GPU检查异常: {str(e)}'}
    
    def check_file_structure(self):
        """检查文件结构"""
        required_files = [
            'config.json',
            'requirements.txt',
            'start_ultimate_system.py',
            'README_ULTIMATE_SYSTEM.md'
        ]
        
        required_dirs = [
            'src/hardware',
            'src/exchange', 
            'src/risk',
            'src/scheduler',
            'src/ai'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        issues = []
        if missing_files:
            issues.append(f"缺少文件: {', '.join(missing_files)}")
        if missing_dirs:
            issues.append(f"缺少目录: {', '.join(missing_dirs)}")
        
        if issues:
            return {
                'status': 'failed',
                'message': '; '.join(issues),
                'details': {'missing_files': missing_files, 'missing_dirs': missing_dirs}
            }
        
        return {
            'status': 'success',
            'message': '文件结构完整',
            'details': {'files': required_files, 'dirs': required_dirs}
        }
    
    def check_log_directory(self):
        """检查日志目录"""
        log_dir = Path('logs')
        
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                return {
                    'status': 'success',
                    'message': '日志目录已创建',
                    'details': {'path': str(log_dir.absolute())}
                }
            except Exception as e:
                return {'status': 'failed', 'message': f'无法创建日志目录: {str(e)}'}
        
        # 检查写入权限
        try:
            test_file = log_dir / 'test_write.tmp'
            test_file.write_text('test')
            test_file.unlink()
            
            return {
                'status': 'success',
                'message': '日志目录可写',
                'details': {'path': str(log_dir.absolute())}
            }
        except Exception as e:
            return {'status': 'failed', 'message': f'日志目录无写入权限: {str(e)}'}
    
    def generate_report(self):
        """生成检查报告"""
        logger.info("\n" + "="*60)
        logger.info("🏥 系统健康检查报告")
        logger.info("="*60)
        
        # 总体统计
        success_rate = (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        
        logger.info(f"📊 检查统计:")
        logger.info(f"   总检查项: {self.total_checks}")
        logger.info(f"   ✅ 通过: {self.passed_checks}")
        logger.info(f"   ⚠️ 警告: {self.warnings}")
        logger.info(f"   ❌ 失败: {self.failed_checks}")
        logger.info(f"   📈 成功率: {success_rate:.1f}%")
        
        # 系统状态评估
        if self.failed_checks == 0:
            if self.warnings == 0:
                status = "🟢 优秀"
                message = "系统完全健康，所有功能正常"
            else:
                status = "🟡 良好"
                message = "系统基本健康，有少量警告"
        elif self.failed_checks <= 2:
            status = "🟠 一般"
            message = "系统有一些问题，需要修复"
        else:
            status = "🔴 差"
            message = "系统有严重问题，需要立即修复"
        
        logger.info(f"\n🎯 系统状态: {status}")
        logger.info(f"💬 评估: {message}")
        
        # 详细结果
        if self.failed_checks > 0 or self.warnings > 0:
            logger.info(f"\n📋 问题详情:")
            for check_name, result in self.results.items():
                if result['status'] in ['failed', 'warning']:
                    status_icon = "❌" if result['status'] == 'failed' else "⚠️"
                    logger.info(f"   {status_icon} {check_name}: {result['message']}")
        
        # 保存报告到文件
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': self.total_checks,
                'passed': self.passed_checks,
                'warnings': self.warnings,
                'failed': self.failed_checks,
                'success_rate': success_rate,
                'status': status,
                'message': message
            },
            'results': self.results
        }
        
        try:
            with open('system_health_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"\n📄 详细报告已保存到: system_health_report.json")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
        
        return success_rate >= 80  # 80%以上认为系统健康
    
    def run_all_checks(self):
        """运行所有健康检查"""
        logger.info("🚀 开始系统健康检查...")
        logger.info("="*60)
        
        # 定义所有检查项
        checks = [
            ("文件结构检查", self.check_file_structure),
            ("配置文件检查", self.check_config_file),
            ("依赖包检查", self.check_dependencies),
            ("核心模块检查", self.check_core_modules),
            ("系统启动器检查", self.check_system_launcher),
            ("GPU支持检查", self.check_gpu_support),
            ("日志目录检查", self.check_log_directory),
        ]
        
        # 运行所有检查
        for check_name, check_func in checks:
            self.run_check(check_name, check_func)
        
        # 生成报告
        return self.generate_report()


def main():
    """主函数"""
    print("🏥 终极合约交易系统 - 健康检查工具")
    print("="*60)
    
    checker = SystemHealthChecker()
    is_healthy = checker.run_all_checks()
    
    # 返回适当的退出码
    sys.exit(0 if is_healthy else 1)


if __name__ == "__main__":
    main()

