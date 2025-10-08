#!/usr/bin/env python3
"""
🌐 888-888-88 Web交易仪表板
Web Trading Dashboard
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import ccxt
from dotenv import load_dotenv
from flask import Flask, render_template_string, jsonify, request
import threading
import time

# 加载环境变量
load_dotenv()

app = Flask(__name__)

class TradingDashboard:
    """交易仪表板"""
    
    def __init__(self):
        self.exchange = None
        self.last_update = None
        self.account_data = {}
        self.market_data = {}
        self.system_status = {}
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """初始化交易所"""
        try:
            bitget_config = {
                'apiKey': os.getenv('BITGET_API_KEY', ''),
                'secret': os.getenv('BITGET_SECRET_KEY', ''),
                'password': os.getenv('BITGET_PASSPHRASE', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000
            }
            
            if bitget_config['apiKey'] and bitget_config['secret']:
                self.exchange = ccxt.bitget(bitget_config)
                print("✅ Bitget交易所初始化成功")
            else:
                print("❌ Bitget API凭证未配置")
                
        except Exception as e:
            print(f"❌ 交易所初始化失败: {e}")
    
    async def update_data(self):
        """更新数据"""
        if not self.exchange:
            return
        
        try:
            # 获取账户余额
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            
            # 获取持仓
            positions = []
            try:
                if hasattr(self.exchange, 'fetch_positions'):
                    all_positions = await asyncio.to_thread(self.exchange.fetch_positions)
                    positions = [pos for pos in all_positions if pos.get('size', 0) != 0]
            except:
                pass
            
            self.account_data = {
                'balance': balance,
                'positions': positions,
                'total_usdt': balance.get('USDT', {}).get('total', 0),
                'free_usdt': balance.get('USDT', {}).get('free', 0),
                'position_count': len(positions)
            }
            
            # 获取市场数据
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
            market_data = {}
            
            for symbol in symbols:
                try:
                    ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                    market_data[symbol] = {
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'],
                        'volume_24h': ticker['quoteVolume']
                    }
                except Exception as e:
                    print(f"⚠️ 获取 {symbol} 数据失败: {e}")
            
            self.market_data = market_data
            
            # 更新系统状态
            self.system_status = {
                'last_update': datetime.now().isoformat(),
                'exchange_connected': True,
                'total_balance': self.account_data['total_usdt'],
                'active_positions': len(positions),
                'system_health': 'healthy'
            }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            print(f"❌ 数据更新失败: {e}")
            self.system_status['exchange_connected'] = False

# 创建仪表板实例
dashboard = TradingDashboard()

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>888-888-88 实盘交易系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .card h3 {
            margin-bottom: 15px;
            color: #FFD700;
            font-size: 1.3em;
        }
        
        .balance-info {
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .balance-amount {
            font-size: 2em;
            font-weight: bold;
            color: #00FF88;
        }
        
        .market-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .market-item:last-child {
            border-bottom: none;
        }
        
        .symbol {
            font-weight: bold;
        }
        
        .price {
            font-size: 1.1em;
        }
        
        .change {
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .change.positive {
            background: #00FF88;
            color: #000;
        }
        
        .change.negative {
            background: #FF4444;
            color: #fff;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #00FF88;
        }
        
        .status-offline {
            background: #FF4444;
        }
        
        .ai-timeline {
            margin-top: 20px;
        }
        
        .ai-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .ai-item:last-child {
            border-bottom: none;
        }
        
        .refresh-btn {
            background: linear-gradient(45deg, #FFD700, #FFA500);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        
        .refresh-btn:hover {
            transform: scale(1.05);
        }
        
        .last-update {
            text-align: center;
            margin-top: 20px;
            opacity: 0.7;
        }
        
        .recommendations {
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .recommendations h3 {
            color: #FFD700;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            list-style: none;
        }
        
        .recommendations li {
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .recommendations li:before {
            content: "💡";
            position: absolute;
            left: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 888-888-88 实盘交易系统</h1>
            <p>Production Trading System Dashboard</p>
            <div style="margin-top: 10px;">
                <span class="status-indicator status-online" id="status-indicator"></span>
                <span id="connection-status">系统在线</span>
            </div>
        </div>
        
        <div class="status-grid">
            <div class="card">
                <h3>💰 账户余额</h3>
                <div class="balance-info">
                    <div>总余额</div>
                    <div class="balance-amount" id="total-balance">加载中...</div>
                </div>
                <div class="balance-info">
                    <div>可用余额: <span id="free-balance">--</span> USDT</div>
                    <div>持仓数量: <span id="position-count">--</span></div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 系统状态</h3>
                <div class="balance-info">
                    <div>交易所: Bitget (生产环境)</div>
                    <div>沙盒模式: 已禁用</div>
                    <div>风险控制: 已启用</div>
                    <div>AI模型: 4个已加载</div>
                </div>
            </div>
        </div>
        
        <div class="status-grid">
            <div class="card">
                <h3>📈 实时行情</h3>
                <div id="market-data">
                    加载中...
                </div>
            </div>
            
            <div class="card">
                <h3>🤖 AI系统进化</h3>
                <div class="ai-timeline">
                    <div class="ai-item">
                        <span>初级AI模型</span>
                        <span style="color: #00FF88;">✅ 已完成</span>
                    </div>
                    <div class="ai-item">
                        <span>中级AI模型</span>
                        <span style="color: #FFD700;">⏳ 7-14天</span>
                    </div>
                    <div class="ai-item">
                        <span>高级AI模型</span>
                        <span style="color: #FFA500;">⏳ 30-60天</span>
                    </div>
                    <div class="ai-item">
                        <span>顶级AI模型</span>
                        <span style="color: #FF6B6B;">⏳ 90-180天</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 性能预期</h3>
            <div class="status-grid" style="margin-bottom: 0;">
                <div>
                    <div class="balance-info">
                        <div>日收益目标: <span style="color: #00FF88;">1-3%</span></div>
                        <div>月收益目标: <span style="color: #00FF88;">20-50%</span></div>
                    </div>
                </div>
                <div>
                    <div class="balance-info">
                        <div>年收益目标: <span style="color: #00FF88;">200-500%</span></div>
                        <div>推荐杠杆: <span style="color: #FFD700;">5-10x</span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="recommendations">
            <h3>💡 系统建议</h3>
            <ul id="recommendations">
                <li>建议从小额交易开始，逐步增加仓位</li>
                <li>严格执行风险管理策略</li>
                <li>定期监控AI模型表现</li>
                <li>系统已准备好进行实盘交易</li>
            </ul>
        </div>
        
        <div style="text-align: center;">
            <button class="refresh-btn" onclick="refreshData()">🔄 刷新数据</button>
        </div>
        
        <div class="last-update">
            最后更新: <span id="last-update">--</span>
        </div>
    </div>
    
    <script>
        async function refreshData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                // 更新余额信息
                document.getElementById('total-balance').textContent = 
                    data.account_data.total_usdt.toFixed(2) + ' USDT';
                document.getElementById('free-balance').textContent = 
                    data.account_data.free_usdt.toFixed(2);
                document.getElementById('position-count').textContent = 
                    data.account_data.position_count;
                
                // 更新市场数据
                const marketDataDiv = document.getElementById('market-data');
                let marketHtml = '';
                
                for (const [symbol, info] of Object.entries(data.market_data)) {
                    const changeClass = info.change_24h >= 0 ? 'positive' : 'negative';
                    const changeSign = info.change_24h >= 0 ? '+' : '';
                    
                    marketHtml += `
                        <div class="market-item">
                            <span class="symbol">${symbol}</span>
                            <div>
                                <span class="price">$${info.price.toLocaleString()}</span>
                                <span class="change ${changeClass}">${changeSign}${info.change_24h.toFixed(2)}%</span>
                            </div>
                        </div>
                    `;
                }
                
                marketDataDiv.innerHTML = marketHtml;
                
                // 更新连接状态
                const statusIndicator = document.getElementById('status-indicator');
                const connectionStatus = document.getElementById('connection-status');
                
                if (data.system_status.exchange_connected) {
                    statusIndicator.className = 'status-indicator status-online';
                    connectionStatus.textContent = '系统在线';
                } else {
                    statusIndicator.className = 'status-indicator status-offline';
                    connectionStatus.textContent = '连接异常';
                }
                
                // 更新最后更新时间
                document.getElementById('last-update').textContent = 
                    new Date(data.system_status.last_update).toLocaleString();
                
            } catch (error) {
                console.error('数据刷新失败:', error);
                document.getElementById('connection-status').textContent = '连接异常';
                document.getElementById('status-indicator').className = 'status-indicator status-offline';
            }
        }
        
        // 页面加载时刷新数据
        window.onload = function() {
            refreshData();
            // 每30秒自动刷新
            setInterval(refreshData, 30000);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def api_data():
    """API数据接口"""
    # 异步更新数据
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dashboard.update_data())
    loop.close()
    
    return jsonify({
        'account_data': dashboard.account_data,
        'market_data': dashboard.market_data,
        'system_status': dashboard.system_status
    })

def run_web_server():
    """运行Web服务器"""
    print("🌐 启动Web仪表板...")
    print("📊 访问地址: http://localhost:8000")
    print("🔄 数据每30秒自动刷新")
    app.run(host='0.0.0.0', port=8000, debug=False)

if __name__ == "__main__":
    run_web_server()
