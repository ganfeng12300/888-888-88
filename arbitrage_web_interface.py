#!/usr/bin/env python3
"""
🌐 专业套利系统Web界面 - 收益拉满版
Professional Arbitrage System Web Interface - Maximum Profit Edition

功能：
- 🚀 套利系统控制面板
- 📊 实时收益监控
- 💰 复利增长可视化
- 🔄 多交易所状态监控
- 🎯 套利机会实时展示
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import asyncio
import threading
import json
from datetime import datetime
import time

from arbitrage_system_core import arbitrage_system

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arbitrage_system_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class ArbitrageWebInterface:
    """套利系统Web界面"""
    
    def __init__(self):
        self.system = arbitrage_system
        self.is_broadcasting = False
        
    def start_data_broadcast(self):
        """启动数据广播"""
        if not self.is_broadcasting:
            self.is_broadcasting = True
            threading.Thread(target=self._broadcast_loop, daemon=True).start()
    
    def _broadcast_loop(self):
        """数据广播循环"""
        while self.is_broadcasting:
            try:
                # 广播系统状态
                status = self.system.get_system_status()
                socketio.emit('system_status', status)
                
                # 广播套利机会
                opportunities = self._format_opportunities()
                socketio.emit('arbitrage_opportunities', opportunities)
                
                # 广播复利增长数据
                growth_data = self._get_compound_growth_data()
                socketio.emit('compound_growth', growth_data)
                
                time.sleep(1)  # 1秒更新一次
                
            except Exception as e:
                print(f"数据广播错误: {e}")
                time.sleep(5)
    
    def _format_opportunities(self):
        """格式化套利机会数据"""
        opportunities = []
        for opp in self.system.opportunities[:10]:  # 显示前10个机会
            opportunities.append({
                'type': opp.type.value,
                'symbol': opp.symbol,
                'exchange_a': opp.exchange_a,
                'exchange_b': opp.exchange_b,
                'price_a': opp.price_a,
                'price_b': opp.price_b,
                'spread_percentage': opp.spread_percentage * 100,
                'expected_profit': opp.expected_profit * 100,
                'signal_strength': opp.signal_strength.value,
                'timestamp': opp.timestamp.isoformat()
            })
        return opportunities
    
    def _get_compound_growth_data(self):
        """获取复利增长数据"""
        growth_data = self.system.stats.get('compound_growth', [])
        return growth_data[-100:]  # 返回最近100个数据点

# 创建Web界面实例
web_interface = ArbitrageWebInterface()

@app.route('/')
def index():
    """主页"""
    return render_template('arbitrage_dashboard.html')

@app.route('/api/system/status')
def get_system_status():
    """获取系统状态"""
    return jsonify(arbitrage_system.get_system_status())

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """启动套利系统"""
    try:
        # 在新线程中启动异步系统
        def run_system():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(arbitrage_system.start_arbitrage_engine())
        
        threading.Thread(target=run_system, daemon=True).start()
        return jsonify({'success': True, 'message': '套利系统启动成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动失败: {e}'})

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """停止套利系统"""
    try:
        arbitrage_system.stop_system()
        return jsonify({'success': True, 'message': '套利系统已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止失败: {e}'})

@app.route('/api/opportunities')
def get_opportunities():
    """获取套利机会"""
    return jsonify(web_interface._format_opportunities())

@app.route('/api/compound/projection')
def get_compound_projection():
    """获取复利预测"""
    days = request.args.get('days', 365, type=int)
    current_capital = arbitrage_system.current_capital
    daily_rate = arbitrage_system.daily_target_rate
    
    projection = []
    capital = current_capital
    
    for day in range(1, days + 1):
        capital *= (1 + daily_rate)
        if day % 30 == 0 or day in [1, 7, 14, 90, 180, 365]:
            projection.append({
                'day': day,
                'capital': round(capital, 2),
                'profit': round(capital - current_capital, 2),
                'growth_rate': round((capital - current_capital) / current_capital * 100, 2)
            })
    
    return jsonify(projection)

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print('客户端已连接')
    web_interface.start_data_broadcast()
    
    # 发送初始数据
    emit('system_status', arbitrage_system.get_system_status())
    emit('arbitrage_opportunities', web_interface._format_opportunities())

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print('客户端已断开连接')

if __name__ == '__main__':
    # 启动Web界面
    print("🌐 启动专业套利系统Web界面...")
    print("📊 访问地址: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
