#!/usr/bin/env python3
"""
🧪 测试现货和合约账户功能
Test Spot and Futures Accounts Functionality
"""

import asyncio
import sys
import json
from pathlib import Path
from loguru import logger

# 添加项目路径
sys.path.append('.')

from src.web.real_trading_dashboard import real_dashboard

async def test_account_separation():
    """测试现货和合约账户分离功能"""
    
    print("🧪 测试现货和合约账户分离功能...")
    print("="*80)
    
    try:
        # 1. 测试账户余额获取
        print("\n📊 1. 测试账户余额获取")
        balance_data = await real_dashboard.get_real_account_balance()
        
        print(f"   ✅ 总账户余额: ${balance_data['account_balance']:,.2f}")
        print(f"   ✅ 总可用余额: ${balance_data['available_balance']:,.2f}")
        
        # 2. 测试现货账户
        if 'spot_account' in balance_data:
            print("\n💰 2. 现货账户详情")
            spot = balance_data['spot_account']
            print(f"   ├── 账户类型: {spot['account_type']}")
            print(f"   ├── 现货资产: ${spot['total_value']:,.2f}")
            print(f"   ├── 可用资金: ${spot['available_balance']:,.2f}")
            print(f"   └── 币种数量: {len(spot['currencies'])}种")
            
            # 显示现货币种详情
            if spot['currencies']:
                print("   现货币种详情:")
                for currency, data in spot['currencies'].items():
                    if data['total'] > 0:
                        print(f"     • {currency}: {data['total']:.8f} (可用: {data['free']:.8f})")
        else:
            print("\n❌ 2. 现货账户数据缺失")
        
        # 3. 测试合约账户
        if 'futures_account' in balance_data:
            print("\n📈 3. 合约账户详情")
            futures = balance_data['futures_account']
            print(f"   ├── 账户类型: {futures['account_type']}")
            print(f"   ├── 合约资产: ${futures['total_value']:,.2f}")
            print(f"   ├── 可用保证金: ${futures['available_balance']:,.2f}")
            print(f"   ├── 已用保证金: ${futures['used_margin']:,.2f}")
            print(f"   ├── 保证金率: {futures['margin_ratio']:,.2f}%")
            print(f"   └── 币种数量: {len(futures['currencies'])}种")
            
            # 显示合约币种详情
            if futures['currencies']:
                print("   合约币种详情:")
                for currency, data in futures['currencies'].items():
                    if data['total'] > 0:
                        print(f"     • {currency}: {data['total']:.2f} (可用: {data['free']:.2f}, 冻结: {data['used']:.2f})")
        else:
            print("\n❌ 3. 合约账户数据缺失")
        
        # 4. 测试账户汇总
        if 'account_summary' in balance_data:
            print("\n📋 4. 账户汇总")
            summary = balance_data['account_summary']
            print(f"   ├── 现货价值: ${summary['spot_value']:,.2f}")
            print(f"   ├── 合约价值: ${summary['futures_value']:,.2f}")
            print(f"   └── 总价值: ${summary['total_value']:,.2f}")
            
            # 计算资产分布
            total = summary['total_value']
            if total > 0:
                spot_pct = (summary['spot_value'] / total) * 100
                futures_pct = (summary['futures_value'] / total) * 100
                print(f"   资产分布:")
                print(f"     • 现货占比: {spot_pct:.1f}%")
                print(f"     • 合约占比: {futures_pct:.1f}%")
        else:
            print("\n❌ 4. 账户汇总数据缺失")
        
        # 5. 测试完整仪表板数据
        print("\n🌐 5. 测试完整仪表板数据")
        dashboard_data = await real_dashboard.get_complete_dashboard_data()
        
        if 'risk_management' in dashboard_data:
            risk_data = dashboard_data['risk_management']
            print(f"   ✅ 风险管理数据包含 {len(risk_data)} 个字段")
            
            # 验证数据结构
            required_fields = ['account_balance', 'spot_account', 'futures_account', 'account_summary']
            missing_fields = [field for field in required_fields if field not in risk_data]
            
            if not missing_fields:
                print("   ✅ 所有必需字段都存在")
            else:
                print(f"   ❌ 缺失字段: {missing_fields}")
        else:
            print("   ❌ 风险管理数据缺失")
        
        # 6. 生成测试报告
        print("\n" + "="*80)
        print("📋 测试报告")
        print("="*80)
        
        # 功能检查
        checks = [
            ("账户余额获取", 'account_balance' in balance_data),
            ("现货账户数据", 'spot_account' in balance_data),
            ("合约账户数据", 'futures_account' in balance_data),
            ("账户汇总数据", 'account_summary' in balance_data),
            ("完整仪表板数据", 'risk_management' in dashboard_data)
        ]
        
        passed_checks = sum(1 for _, passed in checks if passed)
        total_checks = len(checks)
        
        print(f"✅ 通过检查: {passed_checks}/{total_checks}")
        print(f"📊 成功率: {(passed_checks/total_checks)*100:.1f}%")
        
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
        
        # 评分
        if passed_checks == total_checks:
            grade = "A+ (完美)"
            status = "🟢 完全就绪"
        elif passed_checks >= total_checks * 0.8:
            grade = "A (优秀)"
            status = "🟡 基本就绪"
        else:
            grade = "B (需要改进)"
            status = "🔴 需要修复"
        
        print(f"\n🏆 测试等级: {grade}")
        print(f"🚦 系统状态: {status}")
        
        # 7. 保存测试结果
        test_results = {
            'timestamp': dashboard_data.get('timestamp', ''),
            'account_data': balance_data,
            'test_results': {
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'success_rate': (passed_checks/total_checks)*100,
                'grade': grade,
                'status': status
            }
        }
        
        # 保存到文件
        results_file = Path('test_results_spot_futures.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 测试结果已保存到: {results_file}")
        
        return passed_checks == total_checks
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        print(f"\n❌ 测试失败: {e}")
        return False

async def main():
    """主函数"""
    print("🚀 启动现货和合约账户功能测试")
    print("⚠️  注意: 这将连接到真实的Bitget API")
    print()
    
    success = await test_account_separation()
    
    if success:
        print("\n🎉 所有测试通过！现货和合约账户功能正常工作。")
        print("🌐 可以启动Web界面查看详细信息:")
        print("   python start_real_trading_system.py")
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接。")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
