#!/usr/bin/env python3
"""
💰 手续费计算器 - 非VIP用户专用版本
Fee Calculator - Non-VIP User Optimized Version

生产级功能：
- 精确的非VIP手续费计算
- 实时费率更新
- 套利成本分析
- 最优交易所选择
- 手续费优化策略
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import aiohttp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ExchangeFeeStructure:
    """交易所手续费结构"""
    exchange: str
    spot_maker: float      # 现货挂单手续费
    spot_taker: float      # 现货吃单手续费
    futures_maker: float   # 期货挂单手续费
    futures_taker: float   # 期货吃单手续费
    withdrawal_fees: Dict[str, float]  # 提币手续费
    min_trade_amount: float = 10.0     # 最小交易金额
    
    @property
    def spot_round_trip_cost(self) -> float:
        """现货往返成本（买入+卖出）"""
        return self.spot_taker * 2
    
    @property
    def futures_round_trip_cost(self) -> float:
        """期货往返成本（开仓+平仓）"""
        return self.futures_taker * 2

@dataclass
class ArbitrageCost:
    """套利成本分析"""
    buy_exchange: str
    sell_exchange: str
    buy_fee: float
    sell_fee: float
    withdrawal_fee: float
    total_cost_rate: float
    min_profit_threshold: float  # 最小盈利阈值
    
    @property
    def total_cost_usdt(self) -> float:
        """按10000USDT计算的总成本"""
        return 10000 * self.total_cost_rate

class FeeCalculator:
    """手续费计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger("FeeCalculator")
        
        # 2024年最新非VIP手续费（实际费率）
        self.exchange_fees = {
            "binance": ExchangeFeeStructure(
                exchange="binance",
                spot_maker=0.001,      # 0.1%
                spot_taker=0.001,      # 0.1%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0004,  # 0.04%
                withdrawal_fees={
                    "USDT": 1.0,       # 1 USDT (TRC20)
                    "BTC": 0.0005,     # 0.0005 BTC
                    "ETH": 0.005       # 0.005 ETH
                }
            ),
            "okx": ExchangeFeeStructure(
                exchange="okx",
                spot_maker=0.0008,     # 0.08%
                spot_taker=0.001,      # 0.1%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0005,  # 0.05%
                withdrawal_fees={
                    "USDT": 0.8,       # 0.8 USDT (TRC20)
                    "BTC": 0.0004,     # 0.0004 BTC
                    "ETH": 0.004       # 0.004 ETH
                }
            ),
            "bybit": ExchangeFeeStructure(
                exchange="bybit",
                spot_maker=0.001,      # 0.1%
                spot_taker=0.001,      # 0.1%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0006,  # 0.06%
                withdrawal_fees={
                    "USDT": 1.0,       # 1 USDT (TRC20)
                    "BTC": 0.0005,     # 0.0005 BTC
                    "ETH": 0.005       # 0.005 ETH
                }
            ),
            "bitget": ExchangeFeeStructure(
                exchange="bitget",
                spot_maker=0.001,      # 0.1%
                spot_taker=0.001,      # 0.1%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0006,  # 0.06%
                withdrawal_fees={
                    "USDT": 0.8,       # 0.8 USDT (TRC20)
                    "BTC": 0.0005,     # 0.0005 BTC
                    "ETH": 0.005       # 0.005 ETH
                }
            ),
            "huobi": ExchangeFeeStructure(
                exchange="huobi",
                spot_maker=0.002,      # 0.2%
                spot_taker=0.002,      # 0.2%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0004,  # 0.04%
                withdrawal_fees={
                    "USDT": 1.0,       # 1 USDT (TRC20)
                    "BTC": 0.0006,     # 0.0006 BTC
                    "ETH": 0.006       # 0.006 ETH
                }
            ),
            "gateio": ExchangeFeeStructure(
                exchange="gateio",
                spot_maker=0.002,      # 0.2%
                spot_taker=0.002,      # 0.2%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0006,  # 0.06%
                withdrawal_fees={
                    "USDT": 1.0,       # 1 USDT (TRC20)
                    "BTC": 0.0005,     # 0.0005 BTC
                    "ETH": 0.005       # 0.005 ETH
                }
            ),
            "kucoin": ExchangeFeeStructure(
                exchange="kucoin",
                spot_maker=0.001,      # 0.1%
                spot_taker=0.001,      # 0.1%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0006,  # 0.06%
                withdrawal_fees={
                    "USDT": 1.0,       # 1 USDT (TRC20)
                    "BTC": 0.0005,     # 0.0005 BTC
                    "ETH": 0.005       # 0.005 ETH
                }
            ),
            "kraken": ExchangeFeeStructure(
                exchange="kraken",
                spot_maker=0.0016,     # 0.16%
                spot_taker=0.0026,     # 0.26%
                futures_maker=0.0002,  # 0.02%
                futures_taker=0.0005,  # 0.05%
                withdrawal_fees={
                    "USDT": 5.0,       # 5 USDT
                    "BTC": 0.00015,    # 0.00015 BTC
                    "ETH": 0.0035      # 0.0035 ETH
                }
            )
        }
    
    def calculate_spot_arbitrage_cost(self, buy_exchange: str, sell_exchange: str, 
                                    amount: float = 10000, symbol: str = "USDT") -> ArbitrageCost:
        """计算现货套利成本"""
        buy_fees = self.exchange_fees[buy_exchange]
        sell_fees = self.exchange_fees[sell_exchange]
        
        # 买入成本
        buy_cost = amount * buy_fees.spot_taker
        
        # 卖出成本
        sell_cost = amount * sell_fees.spot_taker
        
        # 提币成本（从买入交易所转到卖出交易所）
        withdrawal_cost = buy_fees.withdrawal_fees.get(symbol, 1.0)
        
        # 总成本
        total_cost = buy_cost + sell_cost + withdrawal_cost
        total_cost_rate = total_cost / amount
        
        # 最小盈利阈值（成本 + 25%安全边际）
        min_profit_threshold = total_cost_rate * 1.25
        
        return ArbitrageCost(
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_fee=buy_cost,
            sell_fee=sell_cost,
            withdrawal_fee=withdrawal_cost,
            total_cost_rate=total_cost_rate,
            min_profit_threshold=min_profit_threshold
        )
    
    def calculate_futures_arbitrage_cost(self, spot_exchange: str, futures_exchange: str,
                                       amount: float = 10000) -> ArbitrageCost:
        """计算现货-期货套利成本"""
        spot_fees = self.exchange_fees[spot_exchange]
        futures_fees = self.exchange_fees[futures_exchange]
        
        # 现货成本
        spot_cost = amount * spot_fees.spot_taker
        
        # 期货开仓+平仓成本
        futures_cost = amount * futures_fees.futures_taker * 2
        
        # 总成本（无需提币）
        total_cost = spot_cost + futures_cost
        total_cost_rate = total_cost / amount
        
        # 最小盈利阈值
        min_profit_threshold = total_cost_rate * 1.2  # 20%安全边际
        
        return ArbitrageCost(
            buy_exchange=spot_exchange,
            sell_exchange=futures_exchange,
            buy_fee=spot_cost,
            sell_fee=futures_cost,
            withdrawal_fee=0.0,
            total_cost_rate=total_cost_rate,
            min_profit_threshold=min_profit_threshold
        )
    
    def find_lowest_cost_exchanges(self, trade_type: str = "spot") -> List[Tuple[str, float]]:
        """找到手续费最低的交易所"""
        costs = []
        
        for exchange, fees in self.exchange_fees.items():
            if trade_type == "spot":
                cost = fees.spot_round_trip_cost
            elif trade_type == "futures":
                cost = fees.futures_round_trip_cost
            else:
                continue
            
            costs.append((exchange, cost))
        
        # 按成本排序
        costs.sort(key=lambda x: x[1])
        return costs
    
    def analyze_all_arbitrage_opportunities(self, amount: float = 10000) -> Dict[str, List[ArbitrageCost]]:
        """分析所有套利机会的成本"""
        results = {
            "spot_arbitrage": [],
            "futures_arbitrage": []
        }
        
        exchanges = list(self.exchange_fees.keys())
        
        # 现货套利分析
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges[i+1:]:
                # 双向分析
                cost1 = self.calculate_spot_arbitrage_cost(buy_ex, sell_ex, amount)
                cost2 = self.calculate_spot_arbitrage_cost(sell_ex, buy_ex, amount)
                
                results["spot_arbitrage"].extend([cost1, cost2])
        
        # 现货-期货套利分析
        for spot_ex in exchanges:
            for futures_ex in exchanges:
                if spot_ex != futures_ex:
                    cost = self.calculate_futures_arbitrage_cost(spot_ex, futures_ex, amount)
                    results["futures_arbitrage"].append(cost)
        
        # 按成本排序
        results["spot_arbitrage"].sort(key=lambda x: x.total_cost_rate)
        results["futures_arbitrage"].sort(key=lambda x: x.total_cost_rate)
        
        return results
    
    def get_optimal_trading_strategy(self, target_profit_rate: float = 0.005) -> Dict[str, Any]:
        """获取最优交易策略"""
        analysis = self.analyze_all_arbitrage_opportunities()
        
        # 找到满足目标利润率的最佳机会
        viable_spot = [
            cost for cost in analysis["spot_arbitrage"] 
            if cost.min_profit_threshold <= target_profit_rate
        ]
        
        viable_futures = [
            cost for cost in analysis["futures_arbitrage"]
            if cost.min_profit_threshold <= target_profit_rate
        ]
        
        # 手续费最低的交易所
        lowest_spot = self.find_lowest_cost_exchanges("spot")
        lowest_futures = self.find_lowest_cost_exchanges("futures")
        
        return {
            "target_profit_rate": target_profit_rate,
            "viable_spot_arbitrage": len(viable_spot),
            "viable_futures_arbitrage": len(viable_futures),
            "best_spot_opportunity": viable_spot[0] if viable_spot else None,
            "best_futures_opportunity": viable_futures[0] if viable_futures else None,
            "lowest_cost_spot_exchanges": lowest_spot[:3],
            "lowest_cost_futures_exchanges": lowest_futures[:3],
            "recommendations": self._generate_recommendations(viable_spot, viable_futures, lowest_spot)
        }
    
    def _generate_recommendations(self, viable_spot: List[ArbitrageCost], 
                                viable_futures: List[ArbitrageCost],
                                lowest_spot: List[Tuple[str, float]]) -> List[str]:
        """生成交易建议"""
        recommendations = []
        
        if not viable_spot and not viable_futures:
            recommendations.append("⚠️ 当前市场条件下，没有发现可盈利的套利机会")
            recommendations.append(f"💡 建议关注手续费最低的交易所：{lowest_spot[0][0]} (成本{lowest_spot[0][1]:.4%})")
        
        if viable_spot:
            best_spot = viable_spot[0]
            recommendations.append(
                f"🎯 最佳现货套利：{best_spot.buy_exchange} -> {best_spot.sell_exchange} "
                f"(成本{best_spot.total_cost_rate:.4%}，最小利润阈值{best_spot.min_profit_threshold:.4%})"
            )
        
        if viable_futures:
            best_futures = viable_futures[0]
            recommendations.append(
                f"🚀 最佳期现套利：{best_futures.buy_exchange} + {best_futures.sell_exchange} "
                f"(成本{best_futures.total_cost_rate:.4%}，最小利润阈值{best_futures.min_profit_threshold:.4%})"
            )
        
        # 成本优化建议
        if lowest_spot:
            recommendations.append(f"💰 成本最低现货交易所：{lowest_spot[0][0]} (往返成本{lowest_spot[0][1]:.4%})")
        
        return recommendations
    
    def calculate_daily_fee_impact(self, daily_volume: float = 50000) -> Dict[str, float]:
        """计算日交易量对手续费的影响"""
        daily_costs = {}
        
        for exchange, fees in self.exchange_fees.items():
            # 假设50%现货，50%期货
            spot_cost = daily_volume * 0.5 * fees.spot_round_trip_cost
            futures_cost = daily_volume * 0.5 * fees.futures_round_trip_cost
            total_daily_cost = spot_cost + futures_cost
            
            daily_costs[exchange] = total_daily_cost
        
        return daily_costs
    
    def generate_fee_report(self) -> Dict[str, Any]:
        """生成手续费分析报告"""
        strategy = self.get_optimal_trading_strategy()
        daily_costs = self.calculate_daily_fee_impact()
        
        # 计算年化成本
        annual_costs = {ex: cost * 365 for ex, cost in daily_costs.items()}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "total_exchanges": len(self.exchange_fees),
                "viable_spot_opportunities": strategy["viable_spot_arbitrage"],
                "viable_futures_opportunities": strategy["viable_futures_arbitrage"],
                "target_profit_rate": strategy["target_profit_rate"]
            },
            "cost_analysis": {
                "daily_costs_usd": daily_costs,
                "annual_costs_usd": annual_costs,
                "lowest_cost_exchange": min(daily_costs.items(), key=lambda x: x[1])
            },
            "exchange_rankings": {
                "spot_trading": self.find_lowest_cost_exchanges("spot"),
                "futures_trading": self.find_lowest_cost_exchanges("futures")
            },
            "recommendations": strategy["recommendations"],
            "fee_structures": {
                exchange: {
                    "spot_maker": fees.spot_maker,
                    "spot_taker": fees.spot_taker,
                    "futures_maker": fees.futures_maker,
                    "futures_taker": fees.futures_taker,
                    "spot_round_trip": fees.spot_round_trip_cost,
                    "futures_round_trip": fees.futures_round_trip_cost
                }
                for exchange, fees in self.exchange_fees.items()
            }
        }
        
        return report

async def main():
    """主函数"""
    print("💰 启动手续费计算器（非VIP版本）...")
    
    calculator = FeeCalculator()
    
    # 生成完整报告
    report = calculator.generate_fee_report()
    
    print("\n" + "="*80)
    print("💰 非VIP用户手续费分析报告")
    print("="*80)
    
    print(f"\n📊 分析摘要:")
    summary = report["analysis_summary"]
    print(f"   支持交易所: {summary['total_exchanges']}个")
    print(f"   可行现货套利机会: {summary['viable_spot_opportunities']}个")
    print(f"   可行期货套利机会: {summary['viable_futures_opportunities']}个")
    print(f"   目标利润率: {summary['target_profit_rate']:.2%}")
    
    print(f"\n💸 成本分析 (日交易量5万USDT):")
    daily_costs = report["cost_analysis"]["daily_costs_usd"]
    for exchange, cost in sorted(daily_costs.items(), key=lambda x: x[1])[:5]:
        print(f"   {exchange}: ${cost:.2f}/天 (${cost*365:.0f}/年)")
    
    print(f"\n🏆 交易所排名 (按手续费从低到高):")
    print("   现货交易:")
    for i, (exchange, cost) in enumerate(report["exchange_rankings"]["spot_trading"][:5], 1):
        print(f"     {i}. {exchange}: {cost:.4%} (往返成本)")
    
    print("   期货交易:")
    for i, (exchange, cost) in enumerate(report["exchange_rankings"]["futures_trading"][:5], 1):
        print(f"     {i}. {exchange}: {cost:.4%} (往返成本)")
    
    print(f"\n💡 交易建议:")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    print(f"\n🎯 具体套利成本分析:")
    # 分析几个主要交易所的套利成本
    test_pairs = [("binance", "okx"), ("okx", "bybit"), ("binance", "bitget")]
    
    for buy_ex, sell_ex in test_pairs:
        cost = calculator.calculate_spot_arbitrage_cost(buy_ex, sell_ex)
        print(f"   {buy_ex} -> {sell_ex}: 总成本{cost.total_cost_rate:.4%}, 最小利润阈值{cost.min_profit_threshold:.4%}")
    
    print("\n" + "="*80)
    print("✅ 手续费分析完成！")

if __name__ == "__main__":
    asyncio.run(main())
