#!/usr/bin/env python3
"""
🔒 888-888-88 系统安全检查工具
Security Check Tool for 888-888-88 Trading System
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class SecurityChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def check_hardcoded_secrets(self) -> List[Dict[str, Any]]:
        """检查硬编码密钥"""
        print("🔍 检查硬编码密钥...")
        
        secret_patterns = [
            (r'api_key\s*=\s*["\'][^"\']{10,}["\']', "API Key"),
            (r'secret\s*=\s*["\'][^"\']{10,}["\']', "Secret Key"),
            (r'password\s*=\s*["\'][^"\']{3,}["\']', "Password"),
            (r'token\s*=\s*["\'][^"\']{10,}["\']', "Token"),
            (r'key\s*=\s*["\'][^"\']{10,}["\']', "Key"),
            (r'passphrase\s*=\s*["\'][^"\']{3,}["\']', "Passphrase"),
        ]
        
        issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern, secret_type in secret_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # 排除明显的测试和示例代码
                                if any(keyword in line.lower() for keyword in ['test', 'example', 'demo', 'your_', 'changeme']):
                                    continue
                                    
                                issues.append({
                                    "type": "hardcoded_secret",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "secret_type": secret_type,
                                    "content": line.strip(),
                                    "severity": "high"
                                })
                                
            except Exception as e:
                continue
                
        return issues
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """检查环境变量使用情况"""
        print("🌍 检查环境变量使用...")
        
        env_usage = []
        missing_env_template = not (self.project_root / ".env.template").exists()
        missing_env_example = not (self.project_root / ".env.example").exists()
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 查找环境变量使用
                    env_patterns = [
                        r'os\.getenv\(["\']([^"\']+)["\']',
                        r'os\.environ\[["\']([^"\']+)["\']\]',
                        r'getenv\(["\']([^"\']+)["\']'
                    ]
                    
                    for pattern in env_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            env_usage.extend([(str(py_file.relative_to(self.project_root)), var) for var in matches])
                            
            except Exception:
                continue
        
        return {
            "env_variables_used": len(set([var for _, var in env_usage])),
            "files_using_env": len(set([file for file, _ in env_usage])),
            "env_variables": list(set([var for _, var in env_usage])),
            "missing_env_template": missing_env_template,
            "missing_env_example": missing_env_example
        }
    
    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """检查文件权限"""
        print("📁 检查文件权限...")
        
        issues = []
        sensitive_files = [
            ".env", ".env.local", ".env.production",
            "config.json", "secrets.json", "credentials.json"
        ]
        
        for file_pattern in sensitive_files:
            for file_path in self.project_root.rglob(file_pattern):
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # 检查是否过于宽松的权限
                    if mode in ['777', '666', '755', '644']:
                        issues.append({
                            "type": "file_permission",
                            "file": str(file_path.relative_to(self.project_root)),
                            "permission": mode,
                            "severity": "medium" if mode in ['755', '644'] else "high"
                        })
                        
                except Exception:
                    continue
                    
        return issues
    
    def check_sql_injection_risks(self) -> List[Dict[str, Any]]:
        """检查SQL注入风险"""
        print("💉 检查SQL注入风险...")
        
        issues = []
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*%s[^"\']*["\']',
            r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
            r'SELECT\s+.*\+.*FROM',
            r'INSERT\s+.*\+.*VALUES',
            r'UPDATE\s+.*\+.*SET',
            r'DELETE\s+.*\+.*WHERE'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern in sql_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issues.append({
                                    "type": "sql_injection_risk",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "content": line.strip(),
                                    "severity": "high"
                                })
                                
            except Exception:
                continue
                
        return issues
    
    def check_logging_security(self) -> List[Dict[str, Any]]:
        """检查日志安全性"""
        print("📝 检查日志安全性...")
        
        issues = []
        sensitive_log_patterns = [
            r'log.*password',
            r'log.*api_key',
            r'log.*secret',
            r'log.*token',
            r'print.*password',
            r'print.*api_key',
            r'print.*secret',
            r'print.*token'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern in sensitive_log_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issues.append({
                                    "type": "sensitive_logging",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "content": line.strip(),
                                    "severity": "medium"
                                })
                                
            except Exception:
                continue
                
        return issues
    
    def generate_recommendations(self, results: Dict[str, Any]):
        """生成安全建议"""
        recommendations = []
        
        if results["hardcoded_secrets"]:
            recommendations.append({
                "category": "密钥管理",
                "priority": "高",
                "description": f"发现{len(results['hardcoded_secrets'])}个硬编码密钥",
                "action": "将所有硬编码密钥移至环境变量，使用.env文件管理"
            })
        
        if results["environment_variables"]["missing_env_template"]:
            recommendations.append({
                "category": "配置管理",
                "priority": "中",
                "description": "缺少.env.template文件",
                "action": "创建.env.template文件作为环境变量配置模板"
            })
        
        if results["file_permissions"]:
            recommendations.append({
                "category": "文件权限",
                "priority": "中",
                "description": f"发现{len(results['file_permissions'])}个文件权限问题",
                "action": "修改敏感文件权限为600或更严格"
            })
        
        if results["sql_injection_risks"]:
            recommendations.append({
                "category": "SQL安全",
                "priority": "高",
                "description": f"发现{len(results['sql_injection_risks'])}个潜在SQL注入风险",
                "action": "使用参数化查询替代字符串拼接"
            })
        
        if results["logging_security"]:
            recommendations.append({
                "category": "日志安全",
                "priority": "中",
                "description": f"发现{len(results['logging_security'])}个敏感信息日志记录",
                "action": "避免在日志中记录敏感信息，使用脱敏处理"
            })
        
        return recommendations
    
    def run_security_check(self) -> Dict[str, Any]:
        """运行完整安全检查"""
        print("🔒 开始888-888-88系统安全检查...")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "888-888-88 量化交易系统",
            "hardcoded_secrets": self.check_hardcoded_secrets(),
            "environment_variables": self.check_environment_variables(),
            "file_permissions": self.check_file_permissions(),
            "sql_injection_risks": self.check_sql_injection_risks(),
            "logging_security": self.check_logging_security()
        }
        
        # 生成建议
        results["recommendations"] = self.generate_recommendations(results)
        
        # 计算安全评分
        total_issues = (
            len(results["hardcoded_secrets"]) * 3 +  # 高权重
            len(results["file_permissions"]) * 2 +
            len(results["sql_injection_risks"]) * 3 +  # 高权重
            len(results["logging_security"]) * 1
        )
        
        # 基础分100，每个问题扣分
        security_score = max(0, 100 - total_issues * 5)
        results["security_score"] = security_score
        results["security_grade"] = self.get_security_grade(security_score)
        
        return results
    
    def get_security_grade(self, score: float) -> str:
        """获取安全等级"""
        if score >= 90:
            return "A (优秀)"
        elif score >= 80:
            return "B (良好)"
        elif score >= 70:
            return "C (一般)"
        elif score >= 60:
            return "D (较差)"
        else:
            return "F (危险)"
    
    def print_security_report(self, results: Dict[str, Any]):
        """打印安全报告"""
        print("\n" + "=" * 60)
        print("🔒 888-888-88 系统安全检查报告")
        print("=" * 60)
        
        print(f"🎯 安全评分: {results['security_score']:.1f}/100")
        print(f"🏆 安全等级: {results['security_grade']}")
        
        print(f"\n📊 检查结果:")
        print(f"  🔑 硬编码密钥: {len(results['hardcoded_secrets'])} 个")
        print(f"  🌍 环境变量使用: {results['environment_variables']['env_variables_used']} 个")
        print(f"  📁 文件权限问题: {len(results['file_permissions'])} 个")
        print(f"  💉 SQL注入风险: {len(results['sql_injection_risks'])} 个")
        print(f"  📝 日志安全问题: {len(results['logging_security'])} 个")
        
        if results["hardcoded_secrets"]:
            print(f"\n🚨 硬编码密钥问题:")
            for issue in results["hardcoded_secrets"][:5]:  # 显示前5个
                print(f"  ❌ {issue['file']}:{issue['line']} - {issue['secret_type']}")
        
        if results["recommendations"]:
            print(f"\n💡 安全建议:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. [{rec['priority']}] {rec['description']}")
                print(f"     👉 {rec['action']}")
        
        print(f"\n✅ 环境变量使用情况:")
        env_vars = results['environment_variables']['env_variables'][:10]
        for var in env_vars:
            print(f"  🌍 {var}")
        
        if len(results['environment_variables']['env_variables']) > 10:
            print(f"  ... 还有 {len(results['environment_variables']['env_variables']) - 10} 个")
    
    def save_security_report(self, results: Dict[str, Any], filename: str = "security_report.json"):
        """保存安全报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📊 安全报告已保存: {filename}")

if __name__ == "__main__":
    checker = SecurityChecker()
    results = checker.run_security_check()
    checker.print_security_report(results)
    checker.save_security_report(results)

