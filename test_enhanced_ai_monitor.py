#!/usr/bin/env python3
"""
🧪 测试增强版AI监控器
"""

import asyncio
import time
import json
from src.ai.enhanced_ai_status_monitor import get_enhanced_ai_status_monitor

async def test_enhanced_monitor():
    """测试增强版AI监控器"""
    print("🧪 测试增强版AI监控器")
    print("=" * 50)
    
    # 获取监控器实例
    monitor = get_enhanced_ai_status_monitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 等待初始化
    await asyncio.sleep(5)
    
    # 获取状态报告
    report = monitor.get_enhanced_ai_status_report()
    
    print("📊 增强版AI状态报告:")
    print("=" * 50)
    print(f"🤖 总模型数: {report['system_status']['total_models']}")
    print(f"⚡ 活跃模型数: {report['system_status']['active_models']}")
    print(f"🔄 运行时间: {report['system_status']['uptime_seconds']:.1f}秒")
    print()
    
    print("📋 按类别统计:")
    for category, stats in report['models_by_category'].items():
        print(f"  🔸 {category}: {stats['active']}/{stats['total']} 活跃")
    print()
    
    print("🏆 顶级模型性能:")
    for i, model in enumerate(report['top_performing_models'][:5], 1):
        print(f"  {i}. {model['name']} ({model['category']})")
        print(f"     准确率: {model['accuracy']:.1%} | 成功率: {model['success_rate']:.1%}")
        print(f"     信心度: {model['confidence']:.1%} | 预测数: {model['predictions']}")
    print()
    
    print("📈 信号统计:")
    signals = report['signal_statistics']
    if signals:
        print(f"  总信号数: {signals['total_signals']}")
        print(f"  买入信号: {signals['buy_signals']}")
        print(f"  卖出信号: {signals['sell_signals']}")
        print(f"  持有信号: {signals['hold_signals']}")
        print(f"  平均信心度: {signals['avg_confidence']:.1%}")
        print(f"  信号准确率: {signals['signal_accuracy']:.1%}")
    print()
    
    print("💻 资源使用:")
    resources = report['resource_usage']
    print(f"  平均CPU使用: {resources['avg_cpu']:.1%}")
    print(f"  平均内存使用: {resources['avg_memory']:.1%}")
    print(f"  平均GPU使用: {resources['avg_gpu']:.1%}")
    
    print("=" * 50)
    print("✅ 测试完成！")
    
    # 停止监控
    monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_enhanced_monitor())

