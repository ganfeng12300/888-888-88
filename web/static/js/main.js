// AI量化交易系统 - 前端JavaScript主文件
// 实现实时数据更新、图表渲染、WebSocket通信等功能

class TradingSystemUI {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.updateInterval = 1000; // 1秒更新间隔
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        console.log('🚀 AI量化交易系统UI初始化...');
        
        // 初始化WebSocket连接
        this.initWebSocket();
        
        // 初始化图表
        this.initCharts();
        
        // 启动定时更新
        this.startPeriodicUpdates();
        
        // 绑定事件监听器
        this.bindEventListeners();
        
        console.log('✅ UI初始化完成');
    }
    
    initWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('🔗 WebSocket连接成功');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('disconnect', () => {
                console.log('❌ WebSocket连接断开');
                this.isConnected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('real_time_update', (data) => {
                this.handleRealTimeUpdate(data);
            });
            
            this.socket.on('error', (error) => {
                console.error('WebSocket错误:', error);
            });
            
        } catch (error) {
            console.error('WebSocket初始化失败:', error);
        }
    }
    
    initCharts() {
        // 初始化收益曲线图表
        const ctx = document.getElementById('pnl-chart');
        if (ctx) {
            this.charts.pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '累计收益率',
                        data: [],
                        borderColor: '#ffd700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#ffffff'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#888888'
                            },
                            grid: {
                                color: '#333333'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#888888',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: {
                                color: '#333333'
                            }
                        }
                    }
                }
            });
            
            // 生成模拟数据
            this.generateMockChartData();
        }
    }
    
    generateMockChartData() {
        const labels = [];
        const data = [];
        let currentValue = 0;
        
        for (let i = 0; i < 50; i++) {
            const time = new Date(Date.now() - (49 - i) * 60000);
            labels.push(time.toLocaleTimeString('zh-CN', { 
                hour: '2-digit', 
                minute: '2-digit' 
            }));
            
            currentValue += (Math.random() - 0.4) * 0.5;
            data.push(parseFloat(currentValue.toFixed(2)));
        }
        
        this.charts.pnlChart.data.labels = labels;
        this.charts.pnlChart.data.datasets[0].data = data;
        this.charts.pnlChart.update('none');
    }
    
    startPeriodicUpdates() {
        // 更新系统时间
        setInterval(() => {
            this.updateSystemTime();
        }, 1000);
        
        // 定期请求数据更新
        setInterval(() => {
            this.requestDataUpdate();
        }, this.updateInterval);
        
        // 定期更新模拟数据
        setInterval(() => {
            this.updateMockData();
        }, 5000);
    }
    
    updateSystemTime() {
        const now = new Date();
        const timeString = now.toLocaleString('zh-CN', {
            timeZone: 'Asia/Shanghai',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const timeElement = document.getElementById('system-time');
        if (timeElement) {
            timeElement.textContent = timeString;
        }
    }
    
    requestDataUpdate() {
        if (this.isConnected && this.socket) {
            this.socket.emit('request_update');
        } else {
            // 如果WebSocket未连接，使用HTTP API
            this.fetchDataViaHTTP();
        }
    }
    
    async fetchDataViaHTTP() {
        try {
            // 获取系统状态
            const systemStatus = await fetch('/api/system_status').then(r => r.json());
            this.updateSystemStatus(systemStatus);
            
            // 获取硬件指标
            const hardwareMetrics = await fetch('/api/hardware_metrics').then(r => r.json());
            this.updateHardwareMetrics(hardwareMetrics);
            
            // 获取持仓信息
            const positions = await fetch('/api/positions').then(r => r.json());
            this.updatePositions(positions);
            
            // 获取交易记录
            const trades = await fetch('/api/recent_trades').then(r => r.json());
            this.updateTradesList(trades);
            
        } catch (error) {
            console.error('HTTP数据获取失败:', error);
        }
    }
    
    handleRealTimeUpdate(data) {
        console.log('📊 收到实时数据更新:', data);
        
        if (data.hardware_metrics) {
            this.updateHardwareMetrics(data.hardware_metrics);
        }
        
        if (data.ai_training_progress) {
            this.updateAITrainingProgress(data.ai_training_progress);
        }
        
        if (data.trading_performance) {
            this.updateTradingPerformance(data.trading_performance);
        }
        
        if (data.market_data) {
            this.updateMarketData(data.market_data);
        }
    }
    
    updateSystemStatus(status) {
        // 更新运行时间
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement && status.uptime) {
            uptimeElement.textContent = status.uptime;
        }
        
        // 更新总收益率
        const totalPnlElement = document.getElementById('total-pnl');
        if (totalPnlElement && status.total_pnl) {
            totalPnlElement.textContent = status.total_pnl;
        }
        
        // 更新日收益率
        const dailyPnlElement = document.getElementById('daily-pnl');
        if (dailyPnlElement && status.daily_pnl) {
            dailyPnlElement.textContent = status.daily_pnl;
        }
        
        // 更新最大回撤
        const maxDrawdownElement = document.getElementById('max-drawdown');
        if (maxDrawdownElement && status.max_drawdown) {
            maxDrawdownElement.textContent = status.max_drawdown;
        }
        
        // 更新胜率
        const winRateElement = document.getElementById('win-rate');
        if (winRateElement && status.win_rate) {
            winRateElement.textContent = status.win_rate;
        }
        
        // 更新交易数量
        const totalTradesElement = document.getElementById('total-trades');
        if (totalTradesElement && status.total_trades) {
            totalTradesElement.textContent = status.total_trades;
        }
        
        // 更新AI平均等级
        const aiLevelElement = document.getElementById('ai-average-level');
        if (aiLevelElement && status.ai_average_level) {
            aiLevelElement.textContent = status.ai_average_level;
        }
        
        // 更新系统健康状态
        const systemHealthElement = document.getElementById('system-health');
        if (systemHealthElement && status.system_health) {
            systemHealthElement.textContent = status.system_health;
        }
    }
    
    updateHardwareMetrics(metrics) {
        if (!metrics) return;
        
        // 更新CPU使用率
        if (metrics.cpu) {
            const cpuUsageBar = document.getElementById('cpu-usage');
            if (cpuUsageBar) {
                cpuUsageBar.style.width = `${metrics.cpu.usage}%`;
                cpuUsageBar.parentElement.nextElementSibling.textContent = 
                    `${metrics.cpu.usage}% | ${metrics.cpu.temperature}°C`;
            }
        }
        
        // 更新GPU使用率
        if (metrics.gpu) {
            const gpuUsageBar = document.getElementById('gpu-usage');
            if (gpuUsageBar) {
                gpuUsageBar.style.width = `${metrics.gpu.usage}%`;
                const memoryPercent = ((metrics.gpu.memory_used / metrics.gpu.memory_total) * 100).toFixed(1);
                gpuUsageBar.parentElement.nextElementSibling.textContent = 
                    `${metrics.gpu.usage}% | ${metrics.gpu.temperature}°C | ${metrics.gpu.memory_used.toFixed(1)}GB/${(metrics.gpu.memory_total/1024).toFixed(1)}GB`;
            }
        }
        
        // 更新内存使用率
        if (metrics.memory) {
            const memoryUsageBar = document.getElementById('memory-usage');
            if (memoryUsageBar) {
                memoryUsageBar.style.width = `${metrics.memory.usage_percent}%`;
                memoryUsageBar.parentElement.nextElementSibling.textContent = 
                    `${metrics.memory.used.toFixed(1)}GB/${metrics.memory.total.toFixed(1)}GB (${metrics.memory.usage_percent.toFixed(1)}%)`;
            }
        }
    }
    
    updateAITrainingProgress(progressData) {
        const aiModelsContainer = document.getElementById('ai-models');
        if (!aiModelsContainer || !progressData) return;
        
        aiModelsContainer.innerHTML = '';
        
        Object.entries(progressData).forEach(([modelName, data]) => {
            const modelElement = document.createElement('div');
            modelElement.className = 'ai-model';
            
            modelElement.innerHTML = `
                <div class="ai-model-header">
                    <span class="ai-model-name">${modelName}</span>
                    <span class="ai-level">Lv.${data.level}</span>
                </div>
                <div class="ai-accuracy">准确率: ${data.accuracy}%</div>
                <div class="ai-progress">
                    <div class="ai-progress-fill" style="width: ${data.training_progress}%"></div>
                </div>
            `;
            
            aiModelsContainer.appendChild(modelElement);
        });
    }
    
    updatePositions(positions) {
        const positionsTable = document.querySelector('#positions-table tbody');
        if (!positionsTable || !positions) return;
        
        positionsTable.innerHTML = '';
        
        positions.forEach(position => {
            const row = document.createElement('tr');
            const pnlClass = position.pnl.startsWith('+') ? 'pnl-positive' : 'pnl-negative';
            const sideClass = position.side === 'LONG' ? 'position-long' : 'position-short';
            
            row.innerHTML = `
                <td>${position.symbol}</td>
                <td class="${sideClass}">${position.side}</td>
                <td>${position.size}</td>
                <td>$${position.entry_price.toLocaleString()}</td>
                <td>$${position.current_price.toLocaleString()}</td>
                <td class="${pnlClass}">${position.pnl} (${position.pnl_percent})</td>
                <td>${position.leverage}</td>
            `;
            
            positionsTable.appendChild(row);
        });
    }
    
    updateTradesList(trades) {
        const tradesContainer = document.getElementById('trades-list');
        if (!tradesContainer || !trades) return;
        
        tradesContainer.innerHTML = '';
        
        trades.slice(0, 10).forEach(trade => {
            const tradeElement = document.createElement('div');
            tradeElement.className = 'trade-item';
            
            const pnlClass = trade.pnl > 0 ? 'pnl-positive' : 'pnl-negative';
            const pnlSign = trade.pnl > 0 ? '+' : '';
            
            tradeElement.innerHTML = `
                <div class="trade-info">
                    <div class="trade-symbol">${trade.symbol}</div>
                    <div class="trade-details">${trade.time} | ${trade.side} | ${trade.strategy}</div>
                </div>
                <div class="trade-pnl ${pnlClass}">
                    ${pnlSign}$${Math.abs(trade.pnl).toFixed(2)}
                </div>
            `;
            
            tradesContainer.appendChild(tradeElement);
        });
    }
    
    updateMarketData(marketData) {
        if (!marketData) return;
        
        // 更新BTC价格
        const btcPriceElement = document.getElementById('btc-price');
        if (btcPriceElement && marketData.btc_price) {
            btcPriceElement.textContent = `$${marketData.btc_price.toLocaleString()}`;
        }
        
        // 更新ETH价格
        const ethPriceElement = document.getElementById('eth-price');
        if (ethPriceElement && marketData.eth_price) {
            ethPriceElement.textContent = `$${marketData.eth_price.toLocaleString()}`;
        }
    }
    
    updateMockData() {
        // 更新收益曲线图表
        if (this.charts.pnlChart) {
            const chart = this.charts.pnlChart;
            const now = new Date();
            const timeLabel = now.toLocaleTimeString('zh-CN', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            // 添加新数据点
            chart.data.labels.push(timeLabel);
            const lastValue = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1] || 0;
            const newValue = lastValue + (Math.random() - 0.4) * 0.3;
            chart.data.datasets[0].data.push(parseFloat(newValue.toFixed(2)));
            
            // 保持数据点数量在合理范围内
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
        
        // 添加新的系统日志
        this.addSystemLog();
    }
    
    addSystemLog() {
        const logsContainer = document.getElementById('logs-container');
        if (!logsContainer) return;
        
        const logMessages = [
            { type: 'info', message: 'AI模型训练完成，准确率提升至78.5%' },
            { type: 'success', message: 'BTCUSDT多单平仓，盈利+$400' },
            { type: 'warning', message: 'GPU温度达到75°C，启动智能降频' },
            { type: 'info', message: '网格策略触发，开始建仓' },
            { type: 'success', message: '套利机会检测到，执行交易' },
            { type: 'info', message: 'AI等级提升：强化学习AI升至Lv.46' }
        ];
        
        const randomLog = logMessages[Math.floor(Math.random() * logMessages.length)];
        const now = new Date();
        const timeString = now.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${randomLog.type}`;
        logEntry.innerHTML = `
            <span class="timestamp">${timeString}</span>
            <span class="message">${randomLog.message}</span>
        `;
        
        // 添加到顶部
        logsContainer.insertBefore(logEntry, logsContainer.firstChild);
        
        // 保持日志数量在合理范围内
        const logEntries = logsContainer.querySelectorAll('.log-entry');
        if (logEntries.length > 20) {
            logsContainer.removeChild(logEntries[logEntries.length - 1]);
        }
    }
    
    updateConnectionStatus(connected) {
        // 可以在这里添加连接状态指示器
        console.log(connected ? '🟢 系统在线' : '🔴 系统离线');
    }
    
    bindEventListeners() {
        // 可以在这里添加用户交互事件监听器
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('页面隐藏，暂停更新');
            } else {
                console.log('页面显示，恢复更新');
                this.requestDataUpdate();
            }
        });
    }
    
    updateTradingPerformance(performance) {
        // 更新交易绩效数据
        if (performance && performance.current_performance) {
            const perf = performance.current_performance;
            
            // 更新各种绩效指标
            if (perf.total_pnl !== undefined) {
                const element = document.getElementById('total-pnl');
                if (element) {
                    element.textContent = `${perf.total_pnl > 0 ? '+' : ''}${(perf.total_pnl * 100).toFixed(1)}%`;
                }
            }
            
            if (perf.win_rate !== undefined) {
                const element = document.getElementById('win-rate');
                if (element) {
                    element.textContent = `${(perf.win_rate * 100).toFixed(1)}%`;
                }
            }
        }
    }
}

// 页面加载完成后初始化UI
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 页面加载完成，初始化交易系统UI...');
    window.tradingUI = new TradingSystemUI();
});

// 全局错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
});

// 未处理的Promise拒绝
window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise拒绝:', event.reason);
});

