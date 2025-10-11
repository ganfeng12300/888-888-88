"""
套利引擎
负责发现和执行跨交易所套利机会
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import time
from decimal import Decimal

class ArbitrageEngine:
    """套利引擎"""
    
    def __init__(self, exchanges: Dict, ai_engine):
        self.exchanges = exchanges
        self.ai_engine = ai_engine
        self.logger = logging.getLogger("ArbitrageEngine")
        self.running = False
        self.opportunities = []
        self.executed_trades = []
        self.total_profit = 0.0
        
        # 配置参数
        self.min_profit_threshold = 0.15  # 最小利润阈值 0.15%
        self.max_trade_amount = 1000.0    # 最大交易金额 USDT
        self.min_trade_amount = 10.0      # 最小交易金额 USDT
        self.max_concurrent_trades = 5    # 最大并发交易数
        
    async def initialize(self):
        """初始化套利引擎"""
        self.logger.info("🔄 初始化套利引擎...")
        
        # 验证交易所连接
        for name, exchange in self.exchanges.items():
            try:
                balance = await exchange.get_balance()
                self.logger.info(f"✅ {name} 连接正常，余额: {len(balance)} 种资产")
            except Exception as e:
                self.logger.error(f"❌ {name} 连接失败: {e}")
                
        self.logger.info("✅ 套利引擎初始化完成")
        
    async def run(self):
        """运行套利引擎"""
        self.running = True
        self.logger.info("🚀 套利引擎开始运行...")
        
        while self.running:
            try:
                # 扫描套利机会
                opportunities = await self._scan_arbitrage_opportunities()
                
                # AI分析和筛选
                if opportunities:
                    filtered_opportunities = await self._ai_filter_opportunities(opportunities)
                    
                    # 执行套利交易
                    if filtered_opportunities:
                        await self._execute_arbitrage_opportunities(filtered_opportunities)
                
                # 等待下一个扫描周期
                await asyncio.sleep(2)  # 2秒扫描一次
                
            except Exception as e:
                self.logger.error(f"套利引擎运行错误: {e}")
                await asyncio.sleep(5)
                
    async def stop(self):
        """停止套利引擎"""
        self.running = False
        self.logger.info("🛑 套利引擎已停止")
        
    async def _scan_arbitrage_opportunities(self) -> List[Dict]:
        """扫描套利机会"""
        opportunities = []
        
        try:
            # 获取所有交易所的市场数据
            market_data = {}
            for name, exchange in self.exchanges.items():
                try:
                    data = await exchange.get_market_data()
                    market_data[name] = data
                except Exception as e:
                    self.logger.warning(f"获取{name}市场数据失败: {e}")
                    continue
            
            # 比较价格，寻找套利机会
            symbols = set()
            for exchange_data in market_data.values():
                symbols.update(exchange_data.keys())
            
            for symbol in symbols:
                # 收集该交易对在各交易所的价格
                prices = {}
                for exchange_name, data in market_data.items():
                    if symbol in data:
                        prices[exchange_name] = {
                            'bid': data[symbol]['bid'],
                            'ask': data[symbol]['ask'],
                            'price': data[symbol]['price']
                        }
                
                # 寻找价差机会
                if len(prices) >= 2:
                    opportunity = await self._find_price_difference(symbol, prices)
                    if opportunity:
                        opportunities.append(opportunity)
                        
        except Exception as e:
            self.logger.error(f"扫描套利机会失败: {e}")
            
        return opportunities
        
    async def _find_price_difference(self, symbol: str, prices: Dict) -> Optional[Dict]:
        """寻找价格差异"""
        try:
            exchange_names = list(prices.keys())
            best_opportunity = None
            max_profit = 0
            
            # 比较所有交易所组合
            for i in range(len(exchange_names)):
                for j in range(i + 1, len(exchange_names)):
                    exchange1 = exchange_names[i]
                    exchange2 = exchange_names[j]
                    
                    price1 = prices[exchange1]
                    price2 = prices[exchange2]
                    
                    # 计算买低卖高的利润
                    # 在exchange1买入，在exchange2卖出
                    if price1['ask'] > 0 and price2['bid'] > 0:
                        profit1 = (price2['bid'] - price1['ask']) / price1['ask'] * 100
                        
                        if profit1 > self.min_profit_threshold and profit1 > max_profit:
                            max_profit = profit1
                            best_opportunity = {
                                'symbol': symbol,
                                'buy_exchange': exchange1,
                                'sell_exchange': exchange2,
                                'buy_price': price1['ask'],
                                'sell_price': price2['bid'],
                                'profit_pct': profit1,
                                'direction': 'buy_sell',
                                'timestamp': int(time.time() * 1000)
                            }
                    
                    # 在exchange2买入，在exchange1卖出
                    if price2['ask'] > 0 and price1['bid'] > 0:
                        profit2 = (price1['bid'] - price2['ask']) / price2['ask'] * 100
                        
                        if profit2 > self.min_profit_threshold and profit2 > max_profit:
                            max_profit = profit2
                            best_opportunity = {
                                'symbol': symbol,
                                'buy_exchange': exchange2,
                                'sell_exchange': exchange1,
                                'buy_price': price2['ask'],
                                'sell_price': price1['bid'],
                                'profit_pct': profit2,
                                'direction': 'buy_sell',
                                'timestamp': int(time.time() * 1000)
                            }
            
            return best_opportunity
            
        except Exception as e:
            self.logger.error(f"计算价格差异失败: {e}")
            return None
            
    async def _ai_filter_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """AI筛选套利机会"""
        try:
            if not self.ai_engine:
                return opportunities[:3]  # 没有AI时返回前3个
                
            # 使用AI分析套利机会
            analysis = await self.ai_engine.analyze_arbitrage_opportunities(opportunities)
            
            # 根据AI分析结果筛选
            filtered = []
            for opp in opportunities:
                ai_score = analysis.get(opp['symbol'], {}).get('score', 0)
                if ai_score > 0.7:  # AI评分大于0.7
                    opp['ai_score'] = ai_score
                    filtered.append(opp)
            
            # 按AI评分排序
            filtered.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            
            return filtered[:3]  # 返回前3个最佳机会
            
        except Exception as e:
            self.logger.error(f"AI筛选失败: {e}")
            return opportunities[:3]
            
    async def _execute_arbitrage_opportunities(self, opportunities: List[Dict]):
        """执行套利机会"""
        tasks = []
        
        for opportunity in opportunities:
            if len(tasks) >= self.max_concurrent_trades:
                break
                
            task = asyncio.create_task(
                self._execute_single_arbitrage(opportunity)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"套利执行异常: {result}")
                elif result and result.get('success'):
                    self.total_profit += result.get('profit', 0)
                    self.executed_trades.append(result)
                    self.logger.info(f"✅ 套利成功: {result['symbol']} 利润: {result['profit']:.4f} USDT")
                    
    async def _execute_single_arbitrage(self, opportunity: Dict) -> Dict:
        """执行单个套利交易"""
        try:
            symbol = opportunity['symbol']
            buy_exchange_name = opportunity['buy_exchange']
            sell_exchange_name = opportunity['sell_exchange']
            buy_price = opportunity['buy_price']
            sell_price = opportunity['sell_price']
            
            buy_exchange = self.exchanges[buy_exchange_name]
            sell_exchange = self.exchanges[sell_exchange_name]
            
            # 计算交易金额
            trade_amount = await self._calculate_trade_amount(opportunity)
            
            if trade_amount < self.min_trade_amount:
                return {'success': False, 'error': '交易金额太小'}
            
            # 同时执行买入和卖出
            buy_task = asyncio.create_task(
                buy_exchange.place_order(symbol, 'BUY', 'MARKET', trade_amount / buy_price)
            )
            sell_task = asyncio.create_task(
                sell_exchange.place_order(symbol, 'SELL', 'MARKET', trade_amount / sell_price)
            )
            
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
            
            # 检查执行结果
            if isinstance(buy_result, Exception) or isinstance(sell_result, Exception):
                return {
                    'success': False,
                    'error': f'执行失败: buy={buy_result}, sell={sell_result}'
                }
            
            # 计算实际利润
            actual_profit = (sell_price - buy_price) * (trade_amount / buy_price)
            
            return {
                'success': True,
                'symbol': symbol,
                'buy_exchange': buy_exchange_name,
                'sell_exchange': sell_exchange_name,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'trade_amount': trade_amount,
                'profit': actual_profit,
                'profit_pct': opportunity['profit_pct'],
                'timestamp': int(time.time() * 1000),
                'buy_order': buy_result,
                'sell_order': sell_result
            }
            
        except Exception as e:
            self.logger.error(f"执行套利交易失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _calculate_trade_amount(self, opportunity: Dict) -> float:
        """计算交易金额"""
        try:
            symbol = opportunity['symbol']
            buy_exchange_name = opportunity['buy_exchange']
            sell_exchange_name = opportunity['sell_exchange']
            
            # 获取两个交易所的余额
            buy_exchange = self.exchanges[buy_exchange_name]
            sell_exchange = self.exchanges[sell_exchange_name]
            
            buy_balance = await buy_exchange.get_balance()
            sell_balance = await sell_exchange.get_balance()
            
            # 计算可用资金
            base_asset = symbol.replace('USDT', '')
            quote_asset = 'USDT'
            
            # 买入交易所需要USDT
            buy_usdt = buy_balance.get(quote_asset, {}).get('free', 0)
            
            # 卖出交易所需要基础资产
            sell_base = sell_balance.get(base_asset, {}).get('free', 0)
            sell_base_value = sell_base * opportunity['sell_price']
            
            # 取较小值，并限制在最大交易金额内
            available_amount = min(buy_usdt, sell_base_value, self.max_trade_amount)
            
            return max(available_amount * 0.8, self.min_trade_amount)  # 使用80%的可用资金
            
        except Exception as e:
            self.logger.error(f"计算交易金额失败: {e}")
            return self.min_trade_amount
            
    async def execute_signal(self, signal: Dict):
        """执行AI信号"""
        try:
            if signal.get('type') == 'arbitrage':
                await self._execute_single_arbitrage(signal)
        except Exception as e:
            self.logger.error(f"执行AI信号失败: {e}")
            
    async def get_profit(self) -> float:
        """获取总利润"""
        return self.total_profit
        
    async def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_profit': self.total_profit,
            'total_trades': len(self.executed_trades),
            'success_rate': len([t for t in self.executed_trades if t.get('success')]) / max(len(self.executed_trades), 1),
            'current_opportunities': len(self.opportunities),
            'running': self.running
        }

