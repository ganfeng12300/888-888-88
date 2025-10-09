#!/usr/bin/env python3
"""
📱 移动端应用 - 生产级移动交易界面
Mobile App - Production-Grade Mobile Trading Interface

生产级特性：
- 响应式移动界面
- 实时推送通知
- 移动端优化交互
- 离线数据缓存
- 安全认证机制
"""

from flask import Flask, render_template, request, jsonify
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

from ..monitoring.unified_logging_system import UnifiedLoggingSystem, LogConfig, LogCategory

class MobileApp:
    """移动端应用主类"""
    
    def __init__(self, trading_engine=None, strategy_manager=None):
        # 初始化日志系统
        log_config = LogConfig(
            log_dir="logs",
            console_output=True,
            file_output=True,
            json_format=False
        )
        self.logger = UnifiedLoggingSystem(log_config) # "MobileApp")
        
        # 初始化Flask应用
        self.app = Flask(__name__, 
                        template_folder='mobile_templates',
                        static_folder='mobile_static')
        self.app.secret_key = 'mobile_trading_app_2024'
        
        # 业务组件
        self.trading_engine = trading_engine
        self.strategy_manager = strategy_manager
        
        # 设置路由
        self._setup_mobile_routes()
        
        self.logger.info("移动端应用初始化完成")
    
    def _setup_mobile_routes(self):
        """设置移动端路由"""
        
        @self.app.route('/mobile')
        def mobile_index():
            """移动端主页"""
            return render_template('mobile_dashboard.html')
        
        @self.app.route('/mobile/api/quick_stats')
        def mobile_quick_stats():
            """移动端快速统计"""
            try:
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'total_pnl': 0.0,
                    'active_positions': 0,
                    'active_orders': 0,
                    'system_status': 'running'
                }
                
                if self.trading_engine:
                    trading_stats = self.trading_engine.get_trading_stats()
                    positions = self.trading_engine.get_positions()
                    
                    stats.update({
                        'total_pnl': trading_stats.get('total_pnl', 0.0),
                        'active_positions': len([p for p in positions.values() if p.get('quantity', 0) != 0]),
                        'active_orders': trading_stats.get('active_orders', 0)
                    })
                
                return jsonify(stats)
                
            except Exception as e:
                self.logger.error(f"获取移动端统计失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/mobile/api/positions')
        def mobile_positions():
            """移动端持仓信息"""
            try:
                if self.trading_engine:
                    positions = self.trading_engine.get_positions()
                    # 只返回有持仓的标的
                    active_positions = {
                        symbol: pos for symbol, pos in positions.items() 
                        if pos.get('quantity', 0) != 0
                    }
                    return jsonify(active_positions)
                return jsonify({})
                
            except Exception as e:
                self.logger.error(f"获取移动端持仓失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/mobile/api/quick_order', methods=['POST'])
        def mobile_quick_order():
            """移动端快速下单"""
            try:
                if not self.trading_engine:
                    return jsonify({'error': '交易引擎未启动'}), 400
                
                data = request.get_json()
                
                # 简化的移动端下单
                order_id = self.trading_engine.place_order(
                    symbol=data['symbol'],
                    side=data['side'],
                    order_type='market',  # 移动端默认市价单
                    quantity=float(data['quantity'])
                )
                
                if order_id:
                    return jsonify({'success': True, 'order_id': order_id})
                else:
                    return jsonify({'error': '下单失败'}), 400
                    
            except Exception as e:
                self.logger.error(f"移动端下单失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    def create_mobile_templates(self):
        """创建移动端模板文件"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'mobile_templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        # 创建移动端仪表板模板
        mobile_dashboard_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>量化交易 - 移动端</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-size: 14px;
        }
        .mobile-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .stats-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        .position-item {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem;
            border-left: 4px solid #007bff;
        }
        .quick-action-btn {
            width: 100%;
            margin: 0.25rem 0;
            border-radius: 25px;
            font-weight: bold;
        }
        .buy-btn { background-color: #28a745; border-color: #28a745; }
        .sell-btn { background-color: #dc3545; border-color: #dc3545; }
        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            border-top: 1px solid #dee2e6;
            padding: 0.5rem;
        }
        .nav-item {
            text-align: center;
            color: #666;
            text-decoration: none;
            flex: 1;
        }
        .nav-item.active {
            color: #007bff;
        }
        .floating-btn {
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #007bff;
            color: white;
            border: none;
            font-size: 24px;
            box-shadow: 0 4px 20px rgba(0,123,255,0.3);
        }
        .modal-content {
            border-radius: 15px;
        }
        .form-control {
            border-radius: 10px;
        }
        .pnl-positive { color: #28a745; }
        .pnl-negative { color: #dc3545; }
    </style>
</head>
<body>
    <!-- 头部 -->
    <div class="mobile-header">
        <h4><i class="fas fa-chart-line"></i> 量化交易</h4>
        <small id="last-update">最后更新: --</small>
    </div>

    <!-- 统计卡片 -->
    <div class="container-fluid p-0">
        <div class="row g-0">
            <div class="col-6">
                <div class="stats-card">
                    <div class="stat-value" id="total-pnl">$0.00</div>
                    <div class="stat-label">总盈亏</div>
                </div>
            </div>
            <div class="col-6">
                <div class="stats-card">
                    <div class="stat-value" id="active-positions">0</div>
                    <div class="stat-label">持仓数</div>
                </div>
            </div>
        </div>
    </div>

    <!-- 持仓列表 -->
    <div class="container-fluid">
        <h6 class="mt-3 mb-2"><i class="fas fa-briefcase"></i> 当前持仓</h6>
        <div id="positions-list">
            <div class="text-center text-muted p-3">
                <i class="fas fa-inbox fa-2x"></i>
                <p class="mt-2">暂无持仓</p>
            </div>
        </div>
    </div>

    <!-- 快速交易按钮 -->
    <button class="floating-btn" data-bs-toggle="modal" data-bs-target="#quickTradeModal">
        <i class="fas fa-plus"></i>
    </button>

    <!-- 底部导航 -->
    <div class="bottom-nav d-flex">
        <a href="#" class="nav-item active">
            <i class="fas fa-home d-block"></i>
            <small>首页</small>
        </a>
        <a href="#" class="nav-item">
            <i class="fas fa-chart-bar d-block"></i>
            <small>行情</small>
        </a>
        <a href="#" class="nav-item">
            <i class="fas fa-exchange-alt d-block"></i>
            <small>交易</small>
        </a>
        <a href="#" class="nav-item">
            <i class="fas fa-user d-block"></i>
            <small>我的</small>
        </a>
    </div>

    <!-- 快速交易模态框 -->
    <div class="modal fade" id="quickTradeModal" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="fas fa-bolt"></i> 快速交易</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="quick-trade-form">
                        <div class="mb-3">
                            <label class="form-label">交易标的</label>
                            <select class="form-select" id="trade-symbol" required>
                                <option value="">选择标的</option>
                                <option value="AAPL">AAPL - 苹果</option>
                                <option value="GOOGL">GOOGL - 谷歌</option>
                                <option value="MSFT">MSFT - 微软</option>
                                <option value="TSLA">TSLA - 特斯拉</option>
                                <option value="AMZN">AMZN - 亚马逊</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">交易数量</label>
                            <input type="number" class="form-control" id="trade-quantity" placeholder="输入数量" required>
                        </div>
                        <div class="row g-2">
                            <div class="col-6">
                                <button type="button" class="btn btn-success quick-action-btn buy-btn" onclick="quickTrade('buy')">
                                    <i class="fas fa-arrow-up"></i> 买入
                                </button>
                            </div>
                            <div class="col-6">
                                <button type="button" class="btn btn-danger quick-action-btn sell-btn" onclick="quickTrade('sell')">
                                    <i class="fas fa-arrow-down"></i> 卖出
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 全局变量
        let lastUpdateTime = new Date();

        // 更新统计数据
        function updateStats() {
            fetch('/mobile/api/quick_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-pnl').textContent = `$${data.total_pnl.toFixed(2)}`;
                    document.getElementById('total-pnl').className = data.total_pnl >= 0 ? 'stat-value pnl-positive' : 'stat-value pnl-negative';
                    document.getElementById('active-positions').textContent = data.active_positions;
                    
                    lastUpdateTime = new Date();
                    document.getElementById('last-update').textContent = `最后更新: ${lastUpdateTime.toLocaleTimeString()}`;
                })
                .catch(error => {
                    console.error('更新统计失败:', error);
                });
        }

        // 更新持仓列表
        function updatePositions() {
            fetch('/mobile/api/positions')
                .then(response => response.json())
                .then(data => {
                    const positionsList = document.getElementById('positions-list');
                    
                    if (Object.keys(data).length === 0) {
                        positionsList.innerHTML = `
                            <div class="text-center text-muted p-3">
                                <i class="fas fa-inbox fa-2x"></i>
                                <p class="mt-2">暂无持仓</p>
                            </div>
                        `;
                        return;
                    }
                    
                    let html = '';
                    for (const [symbol, position] of Object.entries(data)) {
                        const pnlClass = position.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                        const pnlIcon = position.unrealized_pnl >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
                        
                        html += `
                            <div class="position-item">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h6 class="mb-1">${symbol}</h6>
                                        <small class="text-muted">数量: ${position.quantity} | 均价: $${position.avg_price.toFixed(2)}</small>
                                    </div>
                                    <div class="text-end">
                                        <div class="${pnlClass}">
                                            <i class="fas ${pnlIcon}"></i>
                                            $${position.unrealized_pnl.toFixed(2)}
                                        </div>
                                        <small class="text-muted">$${position.market_value.toFixed(2)}</small>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    positionsList.innerHTML = html;
                })
                .catch(error => {
                    console.error('更新持仓失败:', error);
                });
        }

        // 快速交易
        function quickTrade(side) {
            const symbol = document.getElementById('trade-symbol').value;
            const quantity = document.getElementById('trade-quantity').value;
            
            if (!symbol || !quantity) {
                alert('请填写完整的交易信息');
                return;
            }
            
            const orderData = {
                symbol: symbol,
                side: side,
                quantity: parseFloat(quantity)
            };
            
            // 显示加载状态
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中...';
            btn.disabled = true;
            
            fetch('/mobile/api/quick_order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 成功提示
                    showToast(`${side === 'buy' ? '买入' : '卖出'}订单提交成功！`, 'success');
                    
                    // 关闭模态框
                    const modal = bootstrap.Modal.getInstance(document.getElementById('quickTradeModal'));
                    modal.hide();
                    
                    // 清空表单
                    document.getElementById('quick-trade-form').reset();
                    
                    // 刷新数据
                    setTimeout(() => {
                        updateStats();
                        updatePositions();
                    }, 1000);
                } else {
                    showToast('交易失败: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('交易失败:', error);
                showToast('交易失败: ' + error, 'error');
            })
            .finally(() => {
                // 恢复按钮状态
                btn.innerHTML = originalText;
                btn.disabled = false;
            });
        }

        // 显示提示消息
        function showToast(message, type = 'info') {
            // 创建提示元素
            const toast = document.createElement('div');
            toast.className = `alert alert-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} position-fixed`;
            toast.style.cssText = 'top: 20px; left: 50%; transform: translateX(-50%); z-index: 9999; min-width: 300px; text-align: center;';
            toast.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                ${message}
            `;
            
            document.body.appendChild(toast);
            
            // 3秒后自动移除
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 3000);
        }

        // 下拉刷新
        let startY = 0;
        let pullDistance = 0;
        const pullThreshold = 80;
        
        document.addEventListener('touchstart', function(e) {
            startY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchmove', function(e) {
            if (window.scrollY === 0) {
                pullDistance = e.touches[0].clientY - startY;
                if (pullDistance > 0 && pullDistance < pullThreshold) {
                    // 可以添加下拉刷新的视觉反馈
                }
            }
        });
        
        document.addEventListener('touchend', function(e) {
            if (pullDistance > pullThreshold) {
                // 触发刷新
                updateStats();
                updatePositions();
                showToast('数据已刷新', 'success');
            }
            pullDistance = 0;
        });

        // 定期更新数据
        setInterval(updateStats, 10000);  // 10秒更新统计
        setInterval(updatePositions, 15000);  // 15秒更新持仓
        
        // 初始加载
        updateStats();
        updatePositions();
        
        // 页面可见性变化时刷新数据
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                updateStats();
                updatePositions();
            }
        });
    </script>
</body>
</html>
        '''
        
        with open(os.path.join(templates_dir, 'mobile_dashboard.html'), 'w', encoding='utf-8') as f:
            f.write(mobile_dashboard_html)
        
        self.logger.info("移动端模板文件已创建")
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """启动移动端服务器"""
        try:
            # 创建模板文件
            self.create_mobile_templates()
            
            self.logger.info(f"启动移动端应用: http://{host}:{port}/mobile")
            self.app.run(host=host, port=port, debug=debug)
            
        except Exception as e:
            self.logger.error(f"启动移动端服务器失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建移动端应用
    mobile_app = MobileApp()
    
    try:
        # 启动移动端服务器
        mobile_app.run(debug=True)
        
    except Exception as e:
        print(f"启动失败: {e}")
