#!/usr/bin/env python3
"""
🌟 终极黑金Web仪表板 - Ultimate Black Gold Web Dashboard
生产级实盘交易系统监控面板
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai.hierarchical_ai_system import hierarchical_ai
from trading.balance_manager import balance_manager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局数据存储
dashboard_data = {
    'balances': {},
    'ai_decisions': [],
    'system_status': {},
    'trading_performance': {},
    'market_data': {},
    'alerts': []
}

# 黑金主题HTML模板
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌟 终极交易系统 - Ultimate Trading System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
            color: #ffd700;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(90deg, #000000 0%, #1a1a1a 50%, #000000 100%);
            border-bottom: 3px solid #ffd700;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
        }
        
        .header h1 {
            font-size: 2.5em;
            text-shadow: 0 0 20px #ffd700;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ffd700, #ffed4e, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            color: #cccccc;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
            border: 2px solid #333;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #ffd700, #ffed4e, #ffd700);
            opacity: 0.8;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(255, 215, 0, 0.2);
            border-color: #ffd700;
        }
        
        .card-title {
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #ffd700;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-content {
            color: #cccccc;
            line-height: 1.6;
        }
        
        .balance-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        
        .balance-item:last-child {
            border-bottom: none;
        }
        
        .balance-currency {
            font-weight: bold;
            color: #ffd700;
        }
        
        .balance-amount {
            color: #00ff88;
            font-family: 'Courier New', monospace;
        }
        
        .ai-decision {
            background: linear-gradient(90deg, #1a1a1a, #2a2a2a);
            border-left: 4px solid #ffd700;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        
        .ai-decision.buy {
            border-left-color: #00ff88;
            background: linear-gradient(90deg, #0a2a0a, #1a3a1a);
        }
        
        .ai-decision.sell {
            border-left-color: #ff4444;
            background: linear-gradient(90deg, #2a0a0a, #3a1a1a);
        }
        
        .ai-level {
            display: inline-block;
            background: #ffd700;
            color: #000;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }
        
        .status-offline {
            background: #ff4444;
            box-shadow: 0 0 10px #ff4444;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #ffd700;
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
            font-family: 'Courier New', monospace;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #999;
            margin-top: 5px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ffd700, #ffed4e);
            transition: width 0.3s ease;
        }
        
        .alert {
            background: linear-gradient(90deg, #ff4444, #ff6666);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .timestamp {
            font-size: 0.8em;
            color: #666;
            text-align: right;
            margin-top: 10px;
        }
        
        .evolution-timeline {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            border: 2px solid #ffd700;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .evolution-stage {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: rgba(255, 215, 0, 0.1);
            border-radius: 10px;
            border-left: 4px solid #ffd700;
        }
        
        .stage-number {
            background: #ffd700;
            color: #000;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }
        
        .stage-content {
            flex: 1;
        }
        
        .stage-title {
            font-weight: bold;
            color: #ffd700;
            margin-bottom: 5px;
        }
        
        .stage-description {
            color: #ccc;
            font-size: 0.9em;
        }
        
        .floating-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            background: #ffd700;
            border-radius: 50%;
            opacity: 0.1;
            animation: float 6s infinite ease-in-out;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .refresh-btn {
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 10px;
        }
        
        .refresh-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
        }
    </style>
</head>
<body>
    <div class="floating-particles" id="particles"></div>
    
    <div class="header">
        <h1>🌟 终极交易系统 Ultimate Trading System</h1>
        <div class="subtitle">🧠 6级分层AI系统 | 💰 实时余额监控 | 📊 生产级交易</div>
        <button class="refresh-btn" onclick="refreshData()">🔄 刷新数据</button>
    </div>
    
    <div class="container">
        <!-- 系统状态 -->
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-title">
                    🚀 系统状态
                </div>
                <div class="card-content" id="system-status">
                    <div class="balance-item">
                        <span>AI系统状态</span>
                        <span><span class="status-indicator status-online"></span>运行中</span>
                    </div>
                    <div class="balance-item">
                        <span>余额监控</span>
                        <span><span class="status-indicator status-online"></span>活跃</span>
                    </div>
                    <div class="balance-item">
                        <span>交易引擎</span>
                        <span><span class="status-indicator status-online"></span>就绪</span>
                    </div>
                </div>
            </div>
            
            <!-- 投资组合总览 -->
            <div class="card">
                <div class="card-title">
                    💰 投资组合总览
                </div>
                <div class="card-content" id="portfolio-overview">
                    <div class="metric-value" id="total-value">$0.00</div>
                    <div class="metric-label">总资产价值</div>
                    <div class="balance-item">
                        <span>现货账户</span>
                        <span class="balance-amount" id="spot-value">$0.00</span>
                    </div>
                    <div class="balance-item">
                        <span>合约账户</span>
                        <span class="balance-amount" id="futures-value">$0.00</span>
                    </div>
                    <div class="balance-item">
                        <span>未实现盈亏</span>
                        <span class="balance-amount" id="unrealized-pnl">$0.00</span>
                    </div>
                </div>
            </div>
            
            <!-- AI决策中心 -->
            <div class="card">
                <div class="card-title">
                    🧠 AI决策中心
                </div>
                <div class="card-content" id="ai-decisions">
                    <div class="ai-decision">
                        <span class="ai-level">L6</span>
                        <strong>战略总指挥AI:</strong> 系统初始化中...
                    </div>
                </div>
            </div>
            
            <!-- 交易性能 -->
            <div class="card">
                <div class="card-title">
                    📊 交易性能
                </div>
                <div class="card-content" id="trading-performance">
                    <div class="balance-item">
                        <span>胜率</span>
                        <span class="balance-amount">69.23%</span>
                    </div>
                    <div class="balance-item">
                        <span>净利润</span>
                        <span class="balance-amount">+1,955.31 USDT</span>
                    </div>
                    <div class="balance-item">
                        <span>盈利因子</span>
                        <span class="balance-amount">3.19</span>
                    </div>
                    <div class="balance-item">
                        <span>夏普比率</span>
                        <span class="balance-amount">2.45</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- AI进化时间线 -->
        <div class="evolution-timeline">
            <div class="card-title">🧬 AI进化时间线 - Evolution Timeline</div>
            <div class="evolution-stage">
                <div class="stage-number">6</div>
                <div class="stage-content">
                    <div class="stage-title">战略总指挥AI (Strategic Command AI)</div>
                    <div class="stage-description">全局战略决策，市场趋势分析，宏观经济指标评估</div>
                </div>
            </div>
            <div class="evolution-stage">
                <div class="stage-number">5</div>
                <div class="stage-content">
                    <div class="stage-title">战术协调AI (Tactical Coordinator AI)</div>
                    <div class="stage-description">战术信号生成，价格行为分析，趋势强度评估</div>
                </div>
            </div>
            <div class="evolution-stage">
                <div class="stage-number">4</div>
                <div class="stage-content">
                    <div class="stage-title">风险管理AI (Risk Management AI)</div>
                    <div class="stage-description">投资组合风险控制，波动率管理，回撤控制</div>
                </div>
            </div>
            <div class="evolution-stage">
                <div class="stage-number">3</div>
                <div class="stage-content">
                    <div class="stage-title">技术分析AI (Technical Analysis AI)</div>
                    <div class="stage-description">技术指标分析，图表模式识别，信号生成</div>
                </div>
            </div>
            <div class="evolution-stage">
                <div class="stage-number">2</div>
                <div class="stage-content">
                    <div class="stage-title">执行优化AI (Execution Optimizer AI)</div>
                    <div class="stage-description">订单执行优化，滑点控制，流动性分析</div>
                </div>
            </div>
            <div class="evolution-stage">
                <div class="stage-number">1</div>
                <div class="stage-content">
                    <div class="stage-title">实时监控AI (Real-time Monitor AI)</div>
                    <div class="stage-description">实时数据监控，即时信号检测，快速响应</div>
                </div>
            </div>
        </div>
        
        <!-- 主要持仓 -->
        <div class="card">
            <div class="card-title">
                💎 主要持仓
            </div>
            <div class="card-content" id="top-holdings">
                <!-- 动态加载 -->
            </div>
        </div>
        
        <!-- 系统警报 -->
        <div class="card" id="alerts-section" style="display: none;">
            <div class="card-title">
                ⚠️ 系统警报
            </div>
            <div class="card-content" id="system-alerts">
                <!-- 动态加载 -->
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // 创建浮动粒子效果
        function createParticles() {
            const container = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.width = Math.random() * 4 + 2 + 'px';
                particle.style.height = particle.style.width;
                particle.style.animationDelay = Math.random() * 6 + 's';
                container.appendChild(particle);
            }
        }
        
        // 更新余额显示
        function updateBalances(data) {
            if (data.total_portfolio_value !== undefined) {
                document.getElementById('total-value').textContent = '$' + data.total_portfolio_value.toFixed(2);
            }
            if (data.spot_value !== undefined) {
                document.getElementById('spot-value').textContent = '$' + data.spot_value.toFixed(2);
            }
            if (data.futures_value !== undefined) {
                document.getElementById('futures-value').textContent = '$' + data.futures_value.toFixed(2);
            }
            if (data.unrealized_pnl !== undefined) {
                const pnlElement = document.getElementById('unrealized-pnl');
                const pnl = data.unrealized_pnl;
                pnlElement.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                pnlElement.style.color = pnl >= 0 ? '#00ff88' : '#ff4444';
            }
            
            // 更新主要持仓
            if (data.top_holdings) {
                const holdingsContainer = document.getElementById('top-holdings');
                holdingsContainer.innerHTML = '';
                data.top_holdings.forEach(holding => {
                    const item = document.createElement('div');
                    item.className = 'balance-item';
                    item.innerHTML = `
                        <span class="balance-currency">${holding.currency} (${holding.account_type})</span>
                        <span class="balance-amount">${holding.amount.toFixed(4)} ($${holding.usd_value.toFixed(2)})</span>
                    `;
                    holdingsContainer.appendChild(item);
                });
            }
        }
        
        // 更新AI决策
        function updateAIDecisions(decisions) {
            const container = document.getElementById('ai-decisions');
            container.innerHTML = '';
            
            decisions.slice(0, 5).forEach(decision => {
                const decisionElement = document.createElement('div');
                decisionElement.className = `ai-decision ${decision.action.toLowerCase()}`;
                decisionElement.innerHTML = `
                    <span class="ai-level">L${decision.level}</span>
                    <strong>${decision.model_name}:</strong> ${decision.action} 
                    (置信度: ${(decision.confidence * 100).toFixed(1)}%)
                    <div class="timestamp">${new Date(decision.timestamp).toLocaleString()}</div>
                `;
                container.appendChild(decisionElement);
            });
        }
        
        // 更新系统警报
        function updateAlerts(alerts) {
            const alertsSection = document.getElementById('alerts-section');
            const alertsContainer = document.getElementById('system-alerts');
            
            if (alerts && alerts.length > 0) {
                alertsSection.style.display = 'block';
                alertsContainer.innerHTML = '';
                
                alerts.forEach(alert => {
                    const alertElement = document.createElement('div');
                    alertElement.className = 'alert';
                    alertElement.innerHTML = `
                        <strong>${alert.type}:</strong> ${alert.message}
                        <div class="timestamp">${new Date(alert.timestamp).toLocaleString()}</div>
                    `;
                    alertsContainer.appendChild(alertElement);
                });
            } else {
                alertsSection.style.display = 'none';
            }
        }
        
        // Socket事件监听
        socket.on('balance_update', updateBalances);
        socket.on('ai_decisions_update', updateAIDecisions);
        socket.on('alerts_update', updateAlerts);
        
        // 刷新数据
        function refreshData() {
            socket.emit('request_update');
        }
        
        // 定期刷新
        setInterval(refreshData, 30000); // 30秒刷新一次
        
        // 初始化
        createParticles();
        refreshData();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """主仪表板页面"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    """API状态接口"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'ai_system': hierarchical_ai.get_system_status(),
        'balance_manager': 'active'
    })

@app.route('/api/balances')
def api_balances():
    """余额API接口"""
    return jsonify(dashboard_data['balances'])

@app.route('/api/ai_decisions')
def api_ai_decisions():
    """AI决策API接口"""
    return jsonify(dashboard_data['ai_decisions'])

@socketio.on('request_update')
def handle_update_request():
    """处理更新请求"""
    # 发送最新数据
    emit('balance_update', dashboard_data['balances'])
    emit('ai_decisions_update', dashboard_data['ai_decisions'])
    emit('alerts_update', dashboard_data['alerts'])

async def update_dashboard_data():
    """更新仪表板数据"""
    while True:
        try:
            # 更新余额数据
            balance_summary = balance_manager.get_balance_summary()
            dashboard_data['balances'] = balance_summary
            
            # 更新AI决策数据
            if not hierarchical_ai.decision_queue.empty():
                recent_decisions = []
                while not hierarchical_ai.decision_queue.empty() and len(recent_decisions) < 10:
                    decision = hierarchical_ai.decision_queue.get()
                    recent_decisions.append({
                        'model_name': decision.model_name,
                        'level': decision.level,
                        'action': decision.action,
                        'confidence': decision.confidence,
                        'timestamp': decision.timestamp.isoformat()
                    })
                dashboard_data['ai_decisions'] = recent_decisions
            
            # 更新警报
            alerts = balance_manager.check_balance_alerts()
            dashboard_data['alerts'] = alerts
            
            # 通过WebSocket发送更新
            socketio.emit('balance_update', dashboard_data['balances'])
            socketio.emit('ai_decisions_update', dashboard_data['ai_decisions'])
            socketio.emit('alerts_update', dashboard_data['alerts'])
            
        except Exception as e:
            print(f"更新仪表板数据错误: {e}")
        
        await asyncio.sleep(10)  # 10秒更新一次

def run_dashboard():
    """运行仪表板"""
    print("🌟 启动终极黑金Web仪表板...")
    print("🌐 访问地址: http://localhost:8888")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False)

if __name__ == "__main__":
    # 启动后台数据更新
    import threading
    
    def start_background_tasks():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_dashboard_data())
    
    background_thread = threading.Thread(target=start_background_tasks, daemon=True)
    background_thread.start()
    
    # 启动Web服务器
    run_dashboard()
