#!/usr/bin/env python3
"""
🔍 发现所有AI模型
扫描系统中的所有AI模型并生成完整列表
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from loguru import logger
import json

class AIModelDiscovery:
    """AI模型发现器"""
    
    def __init__(self):
        self.ai_models = {}
        self.ai_engines = {}
        self.ai_predictors = {}
        self.ai_traders = {}
        self.discovered_classes = set()
        
    def discover_all_models(self) -> Dict[str, Any]:
        """发现所有AI模型"""
        logger.info("🔍 开始扫描所有AI模型...")
        
        # 扫描目录
        directories_to_scan = [
            "src/ai",
            "src/ai_models", 
            "src/ai_enhanced",
            "src/system",
            "src/monitoring",
            "performance/gpu"
        ]
        
        for directory in directories_to_scan:
            if os.path.exists(directory):
                self._scan_directory(directory)
        
        # 生成报告
        report = self._generate_discovery_report()
        
        logger.info(f"✅ 发现完成！共找到 {len(self.discovered_classes)} 个AI相关类")
        return report
    
    def _scan_directory(self, directory: str):
        """扫描目录中的Python文件"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    self._analyze_python_file(file_path)
    
    def _analyze_python_file(self, file_path: str):
        """分析Python文件中的AI模型类"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return
            
            # 查找类定义
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # 分类AI相关的类
                    if self._is_ai_model_class(class_name, content):
                        self._categorize_ai_class(class_name, file_path, content)
                        self.discovered_classes.add(class_name)
                        
        except Exception as e:
            logger.debug(f"分析文件失败 {file_path}: {e}")
    
    def _is_ai_model_class(self, class_name: str, content: str) -> bool:
        """判断是否为AI模型相关的类"""
        ai_keywords = [
            'Model', 'Engine', 'Predictor', 'Trader', 'AI', 'ML', 
            'Neural', 'LSTM', 'CNN', 'Transformer', 'XGBoost',
            'RandomForest', 'SVM', 'Ensemble', 'Reinforcement',
            'Learning', 'Brain', 'Intelligence', 'Decision',
            'Fusion', 'Evolution', 'Meta', 'Prophet', 'Scout'
        ]
        
        # 检查类名
        for keyword in ai_keywords:
            if keyword.lower() in class_name.lower():
                return True
        
        # 检查内容中的AI相关导入和方法
        ai_content_keywords = [
            'torch', 'tensorflow', 'sklearn', 'xgboost', 'lightgbm',
            'predict', 'train', 'fit', 'neural', 'model', 'ai_'
        ]
        
        content_lower = content.lower()
        for keyword in ai_content_keywords:
            if keyword in content_lower:
                return True
                
        return False
    
    def _categorize_ai_class(self, class_name: str, file_path: str, content: str):
        """对AI类进行分类"""
        relative_path = file_path.replace(os.getcwd() + '/', '')
        
        # 分析类的功能
        functionality = self._analyze_class_functionality(class_name, content)
        
        class_info = {
            'name': class_name,
            'file_path': relative_path,
            'functionality': functionality,
            'category': self._determine_category(class_name, content),
            'description': self._extract_class_description(content, class_name)
        }
        
        # 根据名称和功能分类
        if any(keyword in class_name.lower() for keyword in ['model', 'lstm', 'cnn', 'transformer', 'xgboost', 'forest', 'svm']):
            self.ai_models[class_name] = class_info
        elif any(keyword in class_name.lower() for keyword in ['engine', 'fusion', 'brain', 'ensemble']):
            self.ai_engines[class_name] = class_info
        elif any(keyword in class_name.lower() for keyword in ['predictor', 'prophet', 'forecast']):
            self.ai_predictors[class_name] = class_info
        elif any(keyword in class_name.lower() for keyword in ['trader', 'trading', 'reinforcement']):
            self.ai_traders[class_name] = class_info
        else:
            # 默认归类为模型
            self.ai_models[class_name] = class_info
    
    def _analyze_class_functionality(self, class_name: str, content: str) -> List[str]:
        """分析类的功能"""
        functionalities = []
        
        # 检查方法名
        method_patterns = {
            'predict': '预测',
            'train': '训练',
            'fit': '拟合',
            'forecast': '预测',
            'decide': '决策',
            'optimize': '优化',
            'evolve': '进化',
            'learn': '学习',
            'analyze': '分析',
            'monitor': '监控'
        }
        
        for pattern, func in method_patterns.items():
            if f'def {pattern}' in content or f'async def {pattern}' in content:
                functionalities.append(func)
        
        return functionalities
    
    def _determine_category(self, class_name: str, content: str) -> str:
        """确定类别"""
        if any(keyword in class_name.lower() for keyword in ['lstm', 'cnn', 'transformer', 'neural']):
            return '深度学习模型'
        elif any(keyword in class_name.lower() for keyword in ['xgboost', 'forest', 'svm', 'ensemble']):
            return '机器学习模型'
        elif any(keyword in class_name.lower() for keyword in ['engine', 'fusion']):
            return 'AI引擎'
        elif any(keyword in class_name.lower() for keyword in ['trader', 'trading']):
            return '交易AI'
        elif any(keyword in class_name.lower() for keyword in ['monitor', 'status']):
            return 'AI监控'
        else:
            return 'AI组件'
    
    def _extract_class_description(self, content: str, class_name: str) -> str:
        """提取类的描述"""
        lines = content.split('\n')
        in_class = False
        description = ""
        
        for line in lines:
            if f'class {class_name}' in line:
                in_class = True
                continue
            
            if in_class:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    # 提取文档字符串
                    description = line.replace('"""', '').replace("'''", '').strip()
                    break
                elif line.startswith('#'):
                    # 提取注释
                    description = line.replace('#', '').strip()
                    break
                elif line and not line.startswith('def') and not line.startswith('class'):
                    break
        
        return description or f"{class_name} AI组件"
    
    def _generate_discovery_report(self) -> Dict[str, Any]:
        """生成发现报告"""
        total_models = len(self.ai_models)
        total_engines = len(self.ai_engines)
        total_predictors = len(self.ai_predictors)
        total_traders = len(self.ai_traders)
        total_classes = len(self.discovered_classes)
        
        report = {
            "discovery_time": "2025-10-07T15:20:00",
            "summary": {
                "total_ai_classes": total_classes,
                "ai_models": total_models,
                "ai_engines": total_engines,
                "ai_predictors": total_predictors,
                "ai_traders": total_traders
            },
            "categories": {
                "AI模型": self.ai_models,
                "AI引擎": self.ai_engines,
                "AI预测器": self.ai_predictors,
                "AI交易器": self.ai_traders
            },
            "all_discovered_classes": sorted(list(self.discovered_classes))
        }
        
        return report


def main():
    """主函数"""
    print("🔍 888-888-88 AI模型发现系统")
    print("=" * 50)
    
    # 设置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # 创建发现器
    discovery = AIModelDiscovery()
    
    # 发现所有模型
    report = discovery.discover_all_models()
    
    # 保存报告
    with open("ai_models_discovery_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 显示结果
    print("\n" + "=" * 50)
    print("🎉 AI模型发现完成！")
    print("=" * 50)
    print(f"📊 总计发现: {report['summary']['total_ai_classes']} 个AI相关类")
    print(f"🤖 AI模型: {report['summary']['ai_models']} 个")
    print(f"⚙️ AI引擎: {report['summary']['ai_engines']} 个")
    print(f"🔮 AI预测器: {report['summary']['ai_predictors']} 个")
    print(f"💰 AI交易器: {report['summary']['ai_traders']} 个")
    print()
    
    print("📋 发现的主要AI模型:")
    for category, models in report['categories'].items():
        if models:
            print(f"\n🔸 {category}:")
            for name, info in list(models.items())[:5]:  # 显示前5个
                print(f"  • {name}: {info['description']}")
            if len(models) > 5:
                print(f"  ... 还有 {len(models) - 5} 个")
    
    print(f"\n📄 详细报告已保存到: ai_models_discovery_report.json")
    print("=" * 50)
    
    return report


if __name__ == "__main__":
    report = main()

