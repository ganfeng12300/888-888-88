"""
监控面板
提供实时交易监控和数据可视化
"""

import asyncio
import logging
from typing import Dict, List, Optional
import time
import json
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web

class Dashboard:
    """监控面板"""
    
    def __init__(self, exchanges: Dict, strategies: Dict, risk_manager):
        self.exchanges = exchanges
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.logger = logging.getLogger("Dashboard")
        
        # 面板配置
        self.port = 8080
        self.host = '0.0.0.0'
        self.app = None
        self.runner = None
        self.site = None
        
        # 数据存储
        self.real_time_data = {
            'prices': {},
            'trades': [],
            'profits': {},
            'risks': {},
            'system_status': {}
        }
        
        # 统计数据
        self.statistics = {
            'total_profit': 0.0,
            'total_trades': 0,
            'success_rate': 0.0,
            'daily_profit': 0.0,
            'active_strategies': 0,
            'system_uptime': 0
        }
        
        self.start_time = time.time()
        
    async def initialize(self):
        """初始化监控面板"""
        self.logger.info("📈 初始化监控面板...")
        
        # 创建Web应用
        self.app = web.Application()
        
        # 设置路由
        self._setup_routes()
        
        # 启动Web服务器
        await self._start_web_server()
        
        self.logger.info(f"✅ 监控面板启动成功: http://{self.host}:{self.port}")
        
    async def run(self):
        """运行监控面板"""
        self.logger.info("🚀 监控面板开始运行...")
        
        while True:
            try:
                # 更新实时数据
                await self._update_real_time_data()
                
                # 更新统计数据
                await self._update_statistics()
                
                # 检查系统状态
                await self._check_system_status()
                
                # 等待下一个更新周期
                await asyncio.sleep(2)  # 2秒更新一次
                
            except Exception as e:
                self.logger.error(f"监控面板运行错误: {e}")
                await asyncio.sleep(5)
                
    def _setup_routes(self):
        """设置路由"""
        # 静态文件
        self.app.router.add_get('/', self._handle_index)
        
        # API路由
        self.app.router.add_get('/api/status', self._handle_status)
        self.app.router.add_get('/api/prices', self._handle_prices)
        self.app.router.add_get('/api/trades', self._handle_trades)
        self.app.router.add_get('/api/profits', self._handle_profits)
        self.app.router.add_get('/api/risks', self._handle_risks)
        self.app.router.add_get('/api/statistics', self._handle_statistics)
        self.app.router.add_get('/api/exchanges', self._handle_exchanges)
        self.app.router.add_get('/api/strategies', self._handle_strategies)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self._handle_websocket)
        
    async def _start_web_server(self):
        """启动Web服务器"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
        except Exception as e:
            self.logger.error(f"启动Web服务器失败: {e}")
            raise
            
    async def _handle_index(self, request):
        """处理首页请求"""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
        
    async def _handle_status(self, request):
        """处理状态API请求"""
        status = {
            'system_status': 'running',
            'uptime': time.time() - self.start_time,
            'exchanges_status': {},
            'strategies_status': {},
            'risk_status': 'normal',
            'timestamp': int(time.time() * 1000)
        }
        
        # 获取交易所状态
        for name, exchange in self.exchanges.items():
            try:
                # 这里应该检查交易所连接状态
                status['exchanges_status'][name] = 'connected'
            except:
                status['exchanges_status'][name] = 'disconnected'
                
        # 获取策略状态
        for name, strategy in self.strategies.items():
            status['strategies_status'][name] = 'running' if hasattr(strategy, 'running') and strategy.running else 'stopped'
            
        # 获取风险状态
        if self.risk_manager:
            risk_metrics = await self.risk_manager.get_risk_metrics()
            if risk_metrics.get('emergency_stop'):
                status['risk_status'] = 'emergency_stop'
            elif risk_metrics.get('current_drawdown', 0) > 0.02:
                status['risk_status'] = 'high_risk'
            else:
                status['risk_status'] = 'normal'
                
        return web.json_response(status)
        
    async def _handle_prices(self, request):
        """处理价格API请求"""
        return web.json_response(self.real_time_data['prices'])
        
    async def _handle_trades(self, request):
        """处理交易API请求"""
        # 返回最近100笔交易
        recent_trades = self.real_time_data['trades'][-100:]
        return web.json_response(recent_trades)
        
    async def _handle_profits(self, request):
        """处理利润API请求"""
        return web.json_response(self.real_time_data['profits'])
        
    async def _handle_risks(self, request):
        """处理风险API请求"""
        if self.risk_manager:
            risk_report = await self.risk_manager.get_risk_report()
            return web.json_response(risk_report)
        else:
            return web.json_response({'error': 'Risk manager not available'})
            
    async def _handle_statistics(self, request):
        """处理统计API请求"""
        return web.json_response(self.statistics)
        
    async def _handle_exchanges(self, request):
        """处理交易所API请求"""
        exchanges_info = {}
        
        for name, exchange in self.exchanges.items():
            try:
                # 获取交易所信息
                balance = await exchange.get_balance()
                market_data = await exchange.get_market_data()
                
                exchanges_info[name] = {
                    'name': name,
                    'status': 'connected',
                    'balance_count': len(balance),
                    'market_data_count': len(market_data),
                    'last_update': int(time.time() * 1000)
                }
            except Exception as e:
                exchanges_info[name] = {
                    'name': name,
                    'status': 'error',
                    'error': str(e),
                    'last_update': int(time.time() * 1000)
                }
                
        return web.json_response(exchanges_info)
        
    async def _handle_strategies(self, request):
        """处理策略API请求"""
        strategies_info = {}
        
        for name, strategy in self.strategies.items():
            try:
                # 获取策略统计
                stats = await strategy.get_statistics()
                
                strategies_info[name] = {
                    'name': name,
                    'status': 'running' if stats.get('running') else 'stopped',
                    'total_profit': stats.get('total_profit', 0),
                    'total_trades': stats.get('total_trades', 0),
                    'success_rate': stats.get('success_rate', 0),
                    'last_update': int(time.time() * 1000)
                }
            except Exception as e:
                strategies_info[name] = {
                    'name': name,
                    'status': 'error',
                    'error': str(e),
                    'last_update': int(time.time() * 1000)
                }
                
        return web.json_response(strategies_info)
        
    async def _handle_websocket(self, request):
        """处理WebSocket连接"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            # 发送实时数据
            while not ws.closed:
                # 发送当前状态
                data = {
                    'type': 'update',
                    'prices': self.real_time_data['prices'],
                    'statistics': self.statistics,
                    'timestamp': int(time.time() * 1000)
                }
                
                await ws.send_str(json.dumps(data))
                await asyncio.sleep(1)  # 1秒发送一次
                
        except Exception as e:
            self.logger.error(f"WebSocket错误: {e}")
        finally:
            return ws
            
    async def _update_real_time_data(self):
        """更新实时数据"""
        try:
            # 更新价格数据
            for name, exchange in self.exchanges.items():
                try:
                    market_data = await exchange.get_market_data()
                    self.real_time_data['prices'][name] = market_data
                except Exception as e:
                    self.logger.warning(f"获取{name}价格数据失败: {e}")
                    
            # 更新利润数据
            for name, strategy in self.strategies.items():
                try:
                    profit = await strategy.get_profit()
                    self.real_time_data['profits'][name] = profit
                except Exception as e:
                    self.logger.warning(f"获取{name}利润数据失败: {e}")
                    
            # 更新风险数据
            if self.risk_manager:
                try:
                    risk_metrics = await self.risk_manager.get_risk_metrics()
                    self.real_time_data['risks'] = risk_metrics
                except Exception as e:
                    self.logger.warning(f"获取风险数据失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"更新实时数据失败: {e}")
            
    async def _update_statistics(self):
        """更新统计数据"""
        try:
            # 计算总利润
            total_profit = 0.0
            total_trades = 0
            successful_trades = 0
            
            for name, strategy in self.strategies.items():
                try:
                    stats = await strategy.get_statistics()
                    total_profit += stats.get('total_profit', 0)
                    strategy_trades = stats.get('total_trades', 0)
                    total_trades += strategy_trades
                    successful_trades += int(strategy_trades * stats.get('success_rate', 0))
                except:
                    continue
                    
            # 更新统计
            self.statistics.update({
                'total_profit': total_profit,
                'total_trades': total_trades,
                'success_rate': successful_trades / max(total_trades, 1),
                'active_strategies': len([s for s in self.strategies.values() if hasattr(s, 'running') and s.running]),
                'system_uptime': time.time() - self.start_time
            })
            
        except Exception as e:
            self.logger.error(f"更新统计数据失败: {e}")
            
    async def _check_system_status(self):
        """检查系统状态"""
        try:
            system_status = {
                'timestamp': int(time.time() * 1000),
                'exchanges_online': 0,
                'strategies_running': 0,
                'total_exchanges': len(self.exchanges),
                'total_strategies': len(self.strategies),
                'memory_usage': 0,  # 可以添加内存使用监控
                'cpu_usage': 0      # 可以添加CPU使用监控
            }
            
            # 检查交易所状态
            for exchange in self.exchanges.values():
                try:
                    # 简单的连接检查
                    await exchange.get_balance()
                    system_status['exchanges_online'] += 1
                except:
                    pass
                    
            # 检查策略状态
            for strategy in self.strategies.values():
                if hasattr(strategy, 'running') and strategy.running:
                    system_status['strategies_running'] += 1
                    
            self.real_time_data['system_status'] = system_status
            
        except Exception as e:
            self.logger.error(f"检查系统状态失败: {e}")
            
    def _generate_dashboard_html(self) -> str:
        """生成面板HTML"""
        return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI加密货币量化套利交易系统 - 监控面板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.8; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 { margin-bottom: 15px; color: #4CAF50; }
        .metric { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .metric-value { font-weight: bold; color: #FFD700; }
        .status-online { color: #4CAF50; }
        .status-offline { color: #f44336; }
        .profit-positive { color: #4CAF50; }
        .profit-negative { color: #f44336; }
        .refresh-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
        }
        .refresh-btn:hover { background: #45a049; }
        .loading { text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI量化套利交易系统</h1>
            <p>实时监控面板 - 收益拉满 + 低回撤 = 财富自由</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 系统状态</h3>
                <div id="system-status" class="loading">加载中...</div>
                <button class="refresh-btn" onclick="refreshData()">刷新数据</button>
            </div>
            
            <div class="card">
                <h3>💰 交易统计</h3>
                <div id="trading-stats" class="loading">加载中...</div>
            </div>
            
            <div class="card">
                <h3>🏦 交易所状态</h3>
                <div id="exchanges-status" class="loading">加载中...</div>
            </div>
            
            <div class="card">
                <h3>⚡ 策略状态</h3>
                <div id="strategies-status" class="loading">加载中...</div>
            </div>
            
            <div class="card">
                <h3>🛡️ 风险监控</h3>
                <div id="risk-status" class="loading">加载中...</div>
            </div>
            
            <div class="card">
                <h3>📈 实时价格</h3>
                <div id="prices" class="loading">加载中...</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'update') {
                    updatePrices(data.prices);
                    updateStatistics(data.statistics);
                }
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 5000); // 重连
            };
        }
        
        async function refreshData() {
            try {
                await Promise.all([
                    loadSystemStatus(),
                    loadTradingStats(),
                    loadExchangesStatus(),
                    loadStrategiesStatus(),
                    loadRiskStatus(),
                    loadPrices()
                ]);
            } catch (error) {
                console.error('刷新数据失败:', error);
            }
        }
        
        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('system-status').innerHTML = `
                    <div class="metric">
                        <span>系统状态:</span>
                        <span class="metric-value status-${data.system_status === 'running' ? 'online' : 'offline'}">
                            ${data.system_status === 'running' ? '🟢 运行中' : '🔴 停止'}
                        </span>
                    </div>
                    <div class="metric">
                        <span>运行时间:</span>
                        <span class="metric-value">${Math.floor(data.uptime / 3600)}小时</span>
                    </div>
                    <div class="metric">
                        <span>风险状态:</span>
                        <span class="metric-value">${getRiskStatusText(data.risk_status)}</span>
                    </div>
                `;
            } catch (error) {
                document.getElementById('system-status').innerHTML = '❌ 加载失败';
            }
        }
        
        async function loadTradingStats() {
            try {
                const response = await fetch('/api/statistics');
                const data = await response.json();
                
                document.getElementById('trading-stats').innerHTML = `
                    <div class="metric">
                        <span>总利润:</span>
                        <span class="metric-value ${data.total_profit >= 0 ? 'profit-positive' : 'profit-negative'}">
                            ${data.total_profit.toFixed(4)} USDT
                        </span>
                    </div>
                    <div class="metric">
                        <span>总交易数:</span>
                        <span class="metric-value">${data.total_trades}</span>
                    </div>
                    <div class="metric">
                        <span>成功率:</span>
                        <span class="metric-value">${(data.success_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span>活跃策略:</span>
                        <span class="metric-value">${data.active_strategies}</span>
                    </div>
                `;
            } catch (error) {
                document.getElementById('trading-stats').innerHTML = '❌ 加载失败';
            }
        }
        
        async function loadExchangesStatus() {
            try {
                const response = await fetch('/api/exchanges');
                const data = await response.json();
                
                let html = '';
                for (const [name, info] of Object.entries(data)) {
                    html += `
                        <div class="metric">
                            <span>${name}:</span>
                            <span class="metric-value status-${info.status === 'connected' ? 'online' : 'offline'}">
                                ${info.status === 'connected' ? '🟢 已连接' : '🔴 断开'}
                            </span>
                        </div>
                    `;
                }
                
                document.getElementById('exchanges-status').innerHTML = html;
            } catch (error) {
                document.getElementById('exchanges-status').innerHTML = '❌ 加载失败';
            }
        }
        
        async function loadStrategiesStatus() {
            try {
                const response = await fetch('/api/strategies');
                const data = await response.json();
                
                let html = '';
                for (const [name, info] of Object.entries(data)) {
                    html += `
                        <div class="metric">
                            <span>${name}:</span>
                            <span class="metric-value status-${info.status === 'running' ? 'online' : 'offline'}">
                                ${info.status === 'running' ? '🟢 运行中' : '🔴 停止'}
                            </span>
                        </div>
                        <div class="metric">
                            <span>利润:</span>
                            <span class="metric-value ${info.total_profit >= 0 ? 'profit-positive' : 'profit-negative'}">
                                ${info.total_profit.toFixed(4)} USDT
                            </span>
                        </div>
                    `;
                }
                
                document.getElementById('strategies-status').innerHTML = html;
            } catch (error) {
                document.getElementById('strategies-status').innerHTML = '❌ 加载失败';
            }
        }
        
        async function loadRiskStatus() {
            try {
                const response = await fetch('/api/risks');
                const data = await response.json();
                
                document.getElementById('risk-status').innerHTML = `
                    <div class="metric">
                        <span>风险等级:</span>
                        <span class="metric-value">${getRiskLevelText(data.risk_level)}</span>
                    </div>
                    <div class="metric">
                        <span>当日回撤:</span>
                        <span class="metric-value">${(data.metrics.current_drawdown * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <span>仓位比例:</span>
                        <span class="metric-value">${(data.metrics.exposure_ratio * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <span>活跃交易:</span>
                        <span class="metric-value">${data.metrics.active_trades_count}</span>
                    </div>
                `;
            } catch (error) {
                document.getElementById('risk-status').innerHTML = '❌ 加载失败';
            }
        }
        
        async function loadPrices() {
            try {
                const response = await fetch('/api/prices');
                const data = await response.json();
                updatePrices(data);
            } catch (error) {
                document.getElementById('prices').innerHTML = '❌ 加载失败';
            }
        }
        
        function updatePrices(data) {
            let html = '';
            for (const [exchange, symbols] of Object.entries(data)) {
                html += `<h4>${exchange}</h4>`;
                for (const [symbol, info] of Object.entries(symbols)) {
                    const changeClass = info.change >= 0 ? 'profit-positive' : 'profit-negative';
                    html += `
                        <div class="metric">
                            <span>${symbol}:</span>
                            <span class="metric-value">
                                $${info.price.toFixed(4)} 
                                <span class="${changeClass}">(${info.change.toFixed(2)}%)</span>
                            </span>
                        </div>
                    `;
                }
            }
            document.getElementById('prices').innerHTML = html;
        }
        
        function updateStatistics(data) {
            // 更新统计数据的显示
            // 这个函数会被WebSocket调用
        }
        
        function getRiskStatusText(status) {
            const statusMap = {
                'normal': '🟢 正常',
                'high_risk': '🟡 高风险',
                'emergency_stop': '🔴 紧急停止'
            };
            return statusMap[status] || '❓ 未知';
        }
        
        function getRiskLevelText(level) {
            const levelMap = {
                'low': '🟢 低',
                'medium': '🟡 中',
                'high': '🔴 高'
            };
            return levelMap[level] || '❓ 未知';
        }
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
            connectWebSocket();
            
            // 定期刷新数据
            setInterval(refreshData, 30000); // 30秒刷新一次
        });
    </script>
</body>
</html>
        '''
        
    async def stop(self):
        """停止监控面板"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        self.logger.info("🛑 监控面板已停止")

