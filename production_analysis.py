#!/usr/bin/env python3
"""
888-888-88 量化交易系统生产级代码分析工具
Production-Grade Code Analysis Tool
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import re

class ProductionAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "888-888-88 量化交易系统",
            "analysis_version": "1.0.0",
            "overall_score": 0.0,
            "categories": {},
            "issues": [],
            "recommendations": [],
            "statistics": {}
        }
        
    def analyze_code_quality(self) -> Dict[str, Any]:
        """分析代码质量"""
        print("🔍 分析代码质量...")
        
        python_files = list(self.project_root.rglob("*.py"))
        total_files = len(python_files)
        total_lines = 0
        documented_functions = 0
        total_functions = 0
        documented_classes = 0
        total_classes = 0
        error_handling_coverage = 0
        total_methods = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    # 解析AST
                    try:
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                total_functions += 1
                                if ast.get_docstring(node):
                                    documented_functions += 1
                                    
                                # 检查错误处理
                                for child in ast.walk(node):
                                    if isinstance(child, ast.Try):
                                        error_handling_coverage += 1
                                        break
                                        
                            elif isinstance(node, ast.ClassDef):
                                total_classes += 1
                                if ast.get_docstring(node):
                                    documented_classes += 1
                                    
                                # 统计类方法
                                for item in node.body:
                                    if isinstance(item, ast.FunctionDef):
                                        total_methods += 1
                                        
                    except SyntaxError as e:
                        self.results["issues"].append({
                            "type": "syntax_error",
                            "file": str(py_file),
                            "message": f"语法错误: {e}",
                            "severity": "critical"
                        })
                        
            except Exception as e:
                self.results["issues"].append({
                    "type": "file_error",
                    "file": str(py_file),
                    "message": f"文件读取错误: {e}",
                    "severity": "medium"
                })
        
        # 计算覆盖率
        doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        class_doc_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 0
        error_coverage = (error_handling_coverage / total_functions * 100) if total_functions > 0 else 0
        
        quality_score = (doc_coverage + class_doc_coverage + error_coverage) / 3
        
        return {
            "score": quality_score,
            "total_files": total_files,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "documented_functions": documented_functions,
            "function_doc_coverage": doc_coverage,
            "total_classes": total_classes,
            "documented_classes": documented_classes,
            "class_doc_coverage": class_doc_coverage,
            "error_handling_coverage": error_coverage,
            "total_methods": total_methods
        }
    
    def analyze_ai_components(self) -> Dict[str, Any]:
        """分析AI组件"""
        print("🤖 分析AI组件...")
        
        ai_patterns = [
            r'class.*Model.*:',
            r'class.*Engine.*:',
            r'class.*Predictor.*:',
            r'class.*Trader.*:',
            r'def.*predict.*\(',
            r'def.*train.*\(',
            r'def.*fit.*\(',
            r'import.*tensorflow.*',
            r'import.*torch.*',
            r'import.*sklearn.*',
            r'from.*ai.*import',
        ]
        
        ai_files = []
        ai_models = []
        ai_engines = []
        ai_predictors = []
        ai_traders = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 检查是否包含AI相关代码
                    is_ai_file = False
                    for pattern in ai_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            is_ai_file = True
                            break
                    
                    if is_ai_file:
                        ai_files.append(str(py_file))
                        
                        # 分类AI组件
                        if re.search(r'class.*Model.*:', content):
                            ai_models.extend(re.findall(r'class\s+(\w*Model\w*)\s*\(', content))
                        if re.search(r'class.*Engine.*:', content):
                            ai_engines.extend(re.findall(r'class\s+(\w*Engine\w*)\s*\(', content))
                        if re.search(r'class.*Predictor.*:', content):
                            ai_predictors.extend(re.findall(r'class\s+(\w*Predictor\w*)\s*\(', content))
                        if re.search(r'class.*Trader.*:', content):
                            ai_traders.extend(re.findall(r'class\s+(\w*Trader\w*)\s*\(', content))
                            
            except Exception as e:
                continue
        
        total_ai_components = len(ai_models) + len(ai_engines) + len(ai_predictors) + len(ai_traders)
        ai_score = min(100, total_ai_components * 2)  # 每个组件2分，最高100分
        
        return {
            "score": ai_score,
            "ai_files": len(ai_files),
            "total_components": total_ai_components,
            "ai_models": len(ai_models),
            "ai_engines": len(ai_engines),
            "ai_predictors": len(ai_predictors),
            "ai_traders": len(ai_traders),
            "model_list": ai_models[:10],  # 显示前10个
            "engine_list": ai_engines,
            "predictor_list": ai_predictors,
            "trader_list": ai_traders
        }
    
    def analyze_trading_features(self) -> Dict[str, Any]:
        """分析交易功能"""
        print("💹 分析交易功能...")
        
        trading_patterns = [
            r'def.*place_order.*\(',
            r'def.*cancel_order.*\(',
            r'def.*get_positions.*\(',
            r'def.*get_balance.*\(',
            r'def.*execute_trade.*\(',
            r'class.*Exchange.*:',
            r'class.*Trading.*:',
            r'sandbox.*=.*False',
            r'testnet.*=.*False',
        ]
        
        trading_features = []
        real_trading_indicators = 0
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern in trading_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            trading_features.extend(matches)
                            
                    # 检查实盘交易指标
                    if re.search(r'sandbox.*=.*False', content, re.IGNORECASE):
                        real_trading_indicators += 1
                    if re.search(r'testnet.*=.*False', content, re.IGNORECASE):
                        real_trading_indicators += 1
                        
            except Exception as e:
                continue
        
        trading_score = min(100, len(set(trading_features)) * 10 + real_trading_indicators * 5)
        
        return {
            "score": trading_score,
            "trading_features": len(set(trading_features)),
            "real_trading_indicators": real_trading_indicators,
            "feature_list": list(set(trading_features))[:10]
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能指标"""
        print("⚡ 分析性能指标...")
        
        try:
            # 模拟性能数据（实际应该从监控系统获取）
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU信息（如果可用）
            gpu_info = "N/A"
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = f"{gpu.name} ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)"
            except:
                pass
            
            performance_score = 100 - cpu_percent  # 简单的性能评分
            
            return {
                "score": performance_score,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "gpu_info": gpu_info,
                "available_memory_gb": memory.available / (1024**3),
                "total_memory_gb": memory.total / (1024**3)
            }
            
        except Exception as e:
            return {
                "score": 50,
                "error": str(e),
                "cpu_usage": "N/A",
                "memory_usage": "N/A",
                "disk_usage": "N/A"
            }
    
    def analyze_security(self) -> Dict[str, Any]:
        """分析安全性"""
        print("🔒 分析安全性...")
        
        security_issues = []
        security_score = 100
        
        # 检查硬编码密钥
        sensitive_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern in sensitive_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            security_issues.append({
                                "type": "hardcoded_secret",
                                "file": str(py_file),
                                "pattern": pattern,
                                "severity": "high"
                            })
                            security_score -= 10
                            
            except Exception:
                continue
        
        # 检查环境变量使用
        env_usage = 0
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'os.getenv' in content or 'os.environ' in content:
                        env_usage += 1
            except Exception:
                continue
        
        if env_usage > 0:
            security_score += 10  # 使用环境变量是好的做法
        
        return {
            "score": max(0, security_score),
            "security_issues": len(security_issues),
            "env_usage_files": env_usage,
            "issues": security_issues
        }
    
    def analyze_testing(self) -> Dict[str, Any]:
        """分析测试覆盖率"""
        print("🧪 分析测试覆盖率...")
        
        test_files = list(self.project_root.rglob("test_*.py")) + \
                    list(self.project_root.rglob("*_test.py")) + \
                    list(self.project_root.rglob("tests/*.py"))
        
        total_test_files = len(test_files)
        total_py_files = len(list(self.project_root.rglob("*.py")))
        
        test_coverage = (total_test_files / total_py_files * 100) if total_py_files > 0 else 0
        
        # 分析测试类型
        unit_tests = 0
        integration_tests = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'unittest' in content or 'pytest' in content:
                        unit_tests += 1
                    if 'integration' in str(test_file).lower():
                        integration_tests += 1
            except Exception:
                continue
        
        testing_score = min(100, test_coverage * 2)
        
        return {
            "score": testing_score,
            "test_files": total_test_files,
            "test_coverage": test_coverage,
            "unit_tests": unit_tests,
            "integration_tests": integration_tests
        }
    
    def check_production_readiness(self) -> Dict[str, Any]:
        """检查生产就绪性"""
        print("🚀 检查生产就绪性...")
        
        readiness_checks = {
            "config_management": False,
            "logging_system": False,
            "error_handling": False,
            "monitoring": False,
            "database_connection": False,
            "api_documentation": False,
            "deployment_config": False,
            "environment_separation": False
        }
        
        # 检查配置管理
        config_files = list(self.project_root.rglob("config*.py")) + \
                      list(self.project_root.rglob("settings*.py")) + \
                      list(self.project_root.rglob("*.yaml")) + \
                      list(self.project_root.rglob("*.json"))
        
        if config_files:
            readiness_checks["config_management"] = True
        
        # 检查日志系统
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'logging' in content or 'logger' in content:
                        readiness_checks["logging_system"] = True
                        break
            except Exception:
                continue
        
        # 检查其他指标...
        readiness_score = sum(readiness_checks.values()) / len(readiness_checks) * 100
        
        return {
            "score": readiness_score,
            "checks": readiness_checks,
            "config_files": len(config_files)
        }
    
    def generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        code_quality = self.results["categories"].get("code_quality", {})
        if code_quality.get("function_doc_coverage", 0) < 80:
            recommendations.append({
                "category": "代码质量",
                "priority": "高",
                "description": "函数文档覆盖率不足80%，建议增加函数文档字符串",
                "action": "为所有公共函数添加详细的文档字符串"
            })
        
        if code_quality.get("error_handling_coverage", 0) < 70:
            recommendations.append({
                "category": "错误处理",
                "priority": "高",
                "description": "错误处理覆盖率不足70%，建议增加异常处理",
                "action": "为关键函数添加try-except错误处理"
            })
        
        testing = self.results["categories"].get("testing", {})
        if testing.get("test_coverage", 0) < 50:
            recommendations.append({
                "category": "测试覆盖",
                "priority": "中",
                "description": "测试覆盖率不足50%，建议增加单元测试",
                "action": "为核心功能模块编写单元测试"
            })
        
        security = self.results["categories"].get("security", {})
        if security.get("security_issues", 0) > 0:
            recommendations.append({
                "category": "安全性",
                "priority": "高",
                "description": f"发现{security['security_issues']}个安全问题",
                "action": "修复硬编码密钥，使用环境变量管理敏感信息"
            })
        
        self.results["recommendations"] = recommendations
    
    def run_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        print("🎯 开始888-888-88量化交易系统生产级分析...")
        print("=" * 60)
        
        # 执行各项分析
        self.results["categories"]["code_quality"] = self.analyze_code_quality()
        self.results["categories"]["ai_components"] = self.analyze_ai_components()
        self.results["categories"]["trading_features"] = self.analyze_trading_features()
        self.results["categories"]["performance"] = self.analyze_performance()
        self.results["categories"]["security"] = self.analyze_security()
        self.results["categories"]["testing"] = self.analyze_testing()
        self.results["categories"]["production_readiness"] = self.check_production_readiness()
        
        # 计算总分
        scores = [cat["score"] for cat in self.results["categories"].values()]
        self.results["overall_score"] = sum(scores) / len(scores)
        
        # 生成建议
        self.generate_recommendations()
        
        # 统计信息
        self.results["statistics"] = {
            "total_python_files": self.results["categories"]["code_quality"]["total_files"],
            "total_lines_of_code": self.results["categories"]["code_quality"]["total_lines"],
            "total_ai_components": self.results["categories"]["ai_components"]["total_components"],
            "analysis_duration": "完成",
            "grade": self.get_grade(self.results["overall_score"])
        }
        
        return self.results
    
    def get_grade(self, score: float) -> str:
        """获取等级评定"""
        if score >= 90:
            return "A+ (生产就绪)"
        elif score >= 80:
            return "A (接近生产级)"
        elif score >= 70:
            return "B+ (预生产级)"
        elif score >= 60:
            return "B (开发级)"
        elif score >= 50:
            return "C (原型级)"
        else:
            return "D (需要重构)"
    
    def save_report(self, filename: str = "production_analysis_report.json"):
        """保存分析报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"📊 分析报告已保存: {filename}")
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "=" * 60)
        print("📊 888-888-88 量化交易系统生产级分析报告")
        print("=" * 60)
        
        print(f"🎯 总体评分: {self.results['overall_score']:.1f}/100")
        print(f"🏆 系统等级: {self.results['statistics']['grade']}")
        print(f"📁 代码文件: {self.results['statistics']['total_python_files']} 个")
        print(f"📝 代码行数: {self.results['statistics']['total_lines_of_code']:,} 行")
        print(f"🤖 AI组件: {self.results['statistics']['total_ai_components']} 个")
        
        print("\n📈 各项评分:")
        for category, data in self.results["categories"].items():
            score = data["score"]
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"  {status} {category.replace('_', ' ').title()}: {score:.1f}/100")
        
        print(f"\n🔍 发现问题: {len(self.results['issues'])} 个")
        print(f"💡 改进建议: {len(self.results['recommendations'])} 条")
        
        if self.results["recommendations"]:
            print("\n🚀 主要改进建议:")
            for i, rec in enumerate(self.results["recommendations"][:3], 1):
                print(f"  {i}. [{rec['priority']}] {rec['description']}")

if __name__ == "__main__":
    analyzer = ProductionAnalyzer()
    results = analyzer.run_analysis()
    analyzer.print_summary()
    analyzer.save_report()
