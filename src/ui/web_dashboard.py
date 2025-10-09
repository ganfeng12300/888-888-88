#!/usr/bin/env python3
"""
🖥️ Web仪表板 - 生产级交易系统界面
Web Dashboard - Production-Grade Trading System Interface

生产级特性：
- 实时数据展示
- 交互式图表
- 订单管理界面
- 策略监控面板
- 风险控制界面
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory
from ..trading.advanced_trading_engine import AdvancedTradingEngine
from ..trading.strategy_manager import StrategyManager
from ..risk.enhanced_risk_manager import EnhancedRiskManager

class WebDashboard:
    """Web仪表板主类"""
    
    def __init__(self, trading_engine: AdvancedTradingEngine = None,
                 strategy_manager: StrategyManager = None,
                 risk_manager: EnhancedRiskManager = None):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "WebDashboard")
        
        # 初始化Flask应用
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.secret_key = 'trading_system_secret_key_2024'
        
        # 初始化SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 业务组件
        self.trading_engine = trading_engine
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        
        # 数据缓存
        self.market_data_cache = {}
        self.performance_cache = {}
        
        # 设置路由
        self._setup_routes()
        self._setup_socketio_events()
        
        # 启动数据推送线程
        self._start_data_push_thread()
        
        self.logger.info("Web仪表板初始化完成")
    
    def _setup_routes(self):
        """设置Web路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/system/status')
        def system_status():
            """系统状态API"""
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'trading_engine': {
                        'status': 'running' if self.trading_engine else 'stopped',
                        'stats': self.trading_engine.get_trading_stats() if self.trading_engine else {}
                    },
                    'strategy_manager': {
                        'status': 'running' if self.strategy_manager else 'stopped',
                        'strategies': self.strategy_manager.get_all_strategies_status() if self.strategy_manager else {}
                    },
                    'risk_manager': {
                        'status': 'running' if self.risk_manager else 'stopped',
                        'summary': self.risk_manager.get_risk_summary() if self.risk_manager else {}
                    }
                }
                return jsonify(status)
            except Exception as e:
                self.logger.error(f"获取系统状态失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """获取持仓信息"""
            try:
                if self.trading_engine:
                    positions = self.trading_engine.get_positions()
                    return jsonify(positions)
                return jsonify({})
            except Exception as e:
                self.logger.error(f"获取持仓失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/orders')
        def get_orders():
            """获取订单信息"""
            try:
                if self.trading_engine:
                    active_orders = self.trading_engine.order_manager.get_active_orders()
                    orders_data = [
                        {
                            'order_id': order.order_id,
                            'symbol': order.symbol,
                            'side': order.side.value,
                            'order_type': order.order_type.value,
                            'quantity': order.quantity,
                            'price': order.price,
                            'status': order.status.value,
                            'filled_quantity': order.filled_quantity,
                            'created_at': order.created_at.isoformat() if order.created_at else None
                        }
                        for order in active_orders
                    ]
                    return jsonify(orders_data)
                return jsonify([])
            except Exception as e:
                self.logger.error(f"获取订单失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/place_order', methods=['POST'])
        def place_order():
            """下单API"""
            try:
                if not self.trading_engine:
                    return jsonify({'error': '交易引擎未启动'}), 400
                
                data = request.get_json()
                
                order_id = self.trading_engine.place_order(
                    symbol=data['symbol'],
                    side=data['side'],
                    order_type=data['order_type'],
                    quantity=float(data['quantity']),
                    price=float(data.get('price', 0)) if data.get('price') else None,
                    stop_price=float(data.get('stop_price', 0)) if data.get('stop_price') else None
                )
                
                if order_id:
                    return jsonify({'success': True, 'order_id': order_id})
                else:
                    return jsonify({'error': '下单失败'}), 400
                    
            except Exception as e:
                self.logger.error(f"下单失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cancel_order', methods=['POST'])
        def cancel_order():
            """撤单API"""
            try:
                if not self.trading_engine:
                    return jsonify({'error': '交易引擎未启动'}), 400
                
                data = request.get_json()
                order_id = data['order_id']
                
                success = self.trading_engine.cancel_order(order_id)
                
                if success:
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': '撤单失败'}), 400
                    
            except Exception as e:
                self.logger.error(f"撤单失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/strategies')
        def get_strategies():
            """获取策略信息"""
            try:
                if self.strategy_manager:
                    strategies = self.strategy_manager.get_all_strategies_status()
                    return jsonify(strategies)
                return jsonify({})
            except Exception as e:
                self.logger.error(f"获取策略失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/strategy/control', methods=['POST'])
        def control_strategy():
            """策略控制API"""
            try:
                if not self.strategy_manager:
                    return jsonify({'error': '策略管理器未启动'}), 400
                
                data = request.get_json()
                strategy_id = data['strategy_id']
                action = data['action']
                
                success = False
                if action == 'start':
                    success = self.strategy_manager.start_strategy(strategy_id)
                elif action == 'stop':
                    success = self.strategy_manager.stop_strategy(strategy_id)
                elif action == 'pause':
                    success = self.strategy_manager.pause_strategy(strategy_id)
                elif action == 'resume':
                    success = self.strategy_manager.resume_strategy(strategy_id)
                
                if success:
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': f'策略{action}失败'}), 400
                    
            except Exception as e:
                self.logger.error(f"策略控制失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk/summary')
        def get_risk_summary():
            """获取风险摘要"""
            try:
                if self.risk_manager:
                    summary = self.risk_manager.get_risk_summary()
                    return jsonify(summary)
                return jsonify({})
            except Exception as e:
                self.logger.error(f"获取风险摘要失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_events(self):
        """设置SocketIO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """客户端连接"""
            self.logger.info(f"客户端连接: {request.sid}")
            emit('connected', {'status': 'success'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开"""
            self.logger.info(f"客户端断开: {request.sid}")
        
        @self.socketio.on('subscribe_market_data')
        def handle_subscribe_market_data(data):
            """订阅市场数据"""
            symbols = data.get('symbols', [])
            session['subscribed_symbols'] = symbols
            self.logger.info(f"客户端订阅市场数据: {symbols}")
    
    def _start_data_push_thread(self):
        """启动数据推送线程"""
        def push_data():
            while True:
                try:
                    # 推送市场数据
                    self._push_market_data()
                    
                    # 推送系统状态
                    self._push_system_status()
                    
                    # 推送交易统计
                    self._push_trading_stats()
                    
                    time.sleep(1)  # 每秒推送一次
                    
                except Exception as e:
                    self.logger.error(f"数据推送异常: {e}")
                    time.sleep(5)
        
        push_thread = threading.Thread(target=push_data, daemon=True)
        push_thread.start()
    
    def _push_market_data(self):
        """推送市场数据"""
        try:
            if self.trading_engine and self.trading_engine.market_data:
                market_data = {}
                
                # 获取主要股票的价格数据
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
                
                for symbol in symbols:
                    price = self.trading_engine.market_data.get_current_price(symbol)
                    order_book = self.trading_engine.market_data.get_order_book(symbol)
                    
                    if price:
                        market_data[symbol] = {
                            'price': price,
                            'timestamp': datetime.now().isoformat(),
                            'order_book': order_book
                        }
                
                if market_data:
                    self.socketio.emit('market_data_update', market_data)
                    
        except Exception as e:
            self.logger.error(f"推送市场数据失败: {e}")
    
    def _push_system_status(self):
        """推送系统状态"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'trading_engine_running': bool(self.trading_engine),
                'strategy_manager_running': bool(self.strategy_manager),
                'risk_manager_running': bool(self.risk_manager)
            }
            
            self.socketio.emit('system_status_update', status)
            
        except Exception as e:
            self.logger.error(f"推送系统状态失败: {e}")
    
    def _push_trading_stats(self):
        """推送交易统计"""
        try:
            if self.trading_engine:
                stats = self.trading_engine.get_trading_stats()
                self.socketio.emit('trading_stats_update', stats)
                
        except Exception as e:
            self.logger.error(f"推送交易统计失败: {e}")
    
    def create_templates(self):
        """创建HTML模板文件"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        # 创建主仪表板模板
        dashboard_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易系统 - 仪表板</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running { background-color: #28a745; }
        .status-stopped { background-color: #dc3545; }
        .card-header { background-color: #f8f9fa; }
        .metric-value { font-size: 1.5rem; font-weight: bold; }
        .metric-label { color: #6c757d; font-size: 0.9rem; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🚀 量化交易系统</span>
            <div class="d-flex">
                <span class="text-light me-3">
                    <span class="status-indicator" id="system-status"></span>
                    系统状态
                </span>
                <span class="text-light" id="current-time"></span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- 系统概览 -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">交易统计</div>
                    <div class="card-body text-center">
                        <div class="metric-value" id="total-trades">0</div>
                        <div class="metric-label">总交易数</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">总盈亏</div>
                    <div class="card-body text-center">
                        <div class="metric-value" id="total-pnl">$0.00</div>
                        <div class="metric-label">累计盈亏</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">胜率</div>
                    <div class="card-body text-center">
                        <div class="metric-value" id="win-rate">0%</div>
                        <div class="metric-label">交易胜率</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">活跃订单</div>
                    <div class="card-body text-center">
                        <div class="metric-value" id="active-orders">0</div>
                        <div class="metric-label">待执行订单</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- 市场数据 -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">市场数据</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>代码</th>
                                        <th>价格</th>
                                        <th>买价</th>
                                        <th>卖价</th>
                                        <th>更新时间</th>
                                    </tr>
                                </thead>
                                <tbody id="market-data-table">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 持仓信息 -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">持仓信息</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>代码</th>
                                        <th>数量</th>
                                        <th>均价</th>
                                        <th>市值</th>
                                        <th>盈亏</th>
                                    </tr>
                                </thead>
                                <tbody id="positions-table">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <!-- 订单管理 -->
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">订单管理</h5>
                    </div>
                    <div class="card-body">
                        <!-- 下单表单 -->
                        <form id="order-form" class="row g-3 mb-3">
                            <div class="col-md-2">
                                <select class="form-select" id="symbol" required>
                                    <option value="">选择代码</option>
                                    <option value="AAPL">AAPL</option>
                                    <option value="GOOGL">GOOGL</option>
                                    <option value="MSFT">MSFT</option>
                                    <option value="TSLA">TSLA</option>
                                    <option value="AMZN">AMZN</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <select class="form-select" id="side" required>
                                    <option value="buy">买入</option>
                                    <option value="sell">卖出</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <select class="form-select" id="order-type" required>
                                    <option value="market">市价</option>
                                    <option value="limit">限价</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <input type="number" class="form-control" id="quantity" placeholder="数量" required>
                            </div>
                            <div class="col-md-2">
                                <input type="number" class="form-control" id="price" placeholder="价格" step="0.01">
                            </div>
                            <div class="col-md-2">
                                <button type="submit" class="btn btn-primary">下单</button>
                            </div>
                        </form>

                        <!-- 活跃订单列表 -->
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>订单ID</th>
                                        <th>代码</th>
                                        <th>方向</th>
                                        <th>类型</th>
                                        <th>数量</th>
                                        <th>价格</th>
                                        <th>状态</th>
                                        <th>操作</th>
                                    </tr>
                                </thead>
                                <tbody id="orders-table">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 策略状态 -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">策略状态</h5>
                    </div>
                    <div class="card-body">
                        <div id="strategies-list">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket连接
        const socket = io();
        
        // 更新当前时间
        function updateTime() {
            document.getElementById('current-time').textContent = new Date().toLocaleString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // 处理WebSocket事件
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('system-status').className = 'status-indicator status-running';
        });

        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('system-status').className = 'status-indicator status-stopped';
        });

        socket.on('market_data_update', function(data) {
            updateMarketDataTable(data);
        });

        socket.on('trading_stats_update', function(data) {
            updateTradingStats(data);
        });

        // 更新市场数据表格
        function updateMarketDataTable(data) {
            const tbody = document.getElementById('market-data-table');
            tbody.innerHTML = '';
            
            for (const [symbol, info] of Object.entries(data)) {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${symbol}</td>
                    <td>$${info.price.toFixed(2)}</td>
                    <td>$${info.order_book.bid ? info.order_book.bid.toFixed(2) : '-'}</td>
                    <td>$${info.order_book.ask ? info.order_book.ask.toFixed(2) : '-'}</td>
                    <td>${new Date(info.timestamp).toLocaleTimeString()}</td>
                `;
            }
        }

        // 更新交易统计
        function updateTradingStats(data) {
            document.getElementById('total-trades').textContent = data.total_trades || 0;
            document.getElementById('total-pnl').textContent = `$${(data.total_pnl || 0).toFixed(2)}`;
            document.getElementById('win-rate').textContent = `${((data.win_rate || 0) * 100).toFixed(1)}%`;
            document.getElementById('active-orders').textContent = data.active_orders || 0;
        }

        // 下单表单提交
        document.getElementById('order-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const orderData = {
                symbol: document.getElementById('symbol').value,
                side: document.getElementById('side').value,
                order_type: document.getElementById('order-type').value,
                quantity: document.getElementById('quantity').value,
                price: document.getElementById('price').value || null
            };

            fetch('/api/place_order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('下单成功！订单ID: ' + data.order_id);
                    loadOrders();
                } else {
                    alert('下单失败: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('下单失败: ' + error);
            });
        });

        // 加载数据
        function loadPositions() {
            fetch('/api/positions')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('positions-table');
                    tbody.innerHTML = '';
                    
                    for (const [symbol, position] of Object.entries(data)) {
                        if (position.quantity !== 0) {
                            const row = tbody.insertRow();
                            const pnlClass = position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
                            row.innerHTML = `
                                <td>${symbol}</td>
                                <td>${position.quantity}</td>
                                <td>$${position.avg_price.toFixed(2)}</td>
                                <td>$${position.market_value.toFixed(2)}</td>
                                <td class="${pnlClass}">$${position.unrealized_pnl.toFixed(2)}</td>
                            `;
                        }
                    }
                });
        }

        function loadOrders() {
            fetch('/api/orders')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('orders-table');
                    tbody.innerHTML = '';
                    
                    data.forEach(order => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${order.order_id.substring(0, 8)}...</td>
                            <td>${order.symbol}</td>
                            <td>${order.side}</td>
                            <td>${order.order_type}</td>
                            <td>${order.quantity}</td>
                            <td>$${order.price ? order.price.toFixed(2) : 'Market'}</td>
                            <td>${order.status}</td>
                            <td>
                                <button class="btn btn-sm btn-danger" onclick="cancelOrder('${order.order_id}')">撤单</button>
                            </td>
                        `;
                    });
                });
        }

        function cancelOrder(orderId) {
            if (confirm('确定要撤销这个订单吗？')) {
                fetch('/api/cancel_order', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({order_id: orderId})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('撤单成功！');
                        loadOrders();
                    } else {
                        alert('撤单失败: ' + data.error);
                    }
                });
            }
        }

        // 定期刷新数据
        setInterval(loadPositions, 5000);
        setInterval(loadOrders, 3000);
        
        // 初始加载
        loadPositions();
        loadOrders();
    </script>
</body>
</html>
        '''
        
        with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        self.logger.info("HTML模板文件已创建")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """启动Web服务器"""
        try:
            # 创建模板文件
            self.create_templates()
            
            self.logger.info(f"启动Web仪表板: http://{host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"启动Web服务器失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建Web仪表板
    dashboard = WebDashboard()
    
    try:
        # 启动Web服务器
        dashboard.run(debug=True)
        
    except Exception as e:
        print(f"启动失败: {e}")
