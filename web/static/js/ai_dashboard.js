/**
 * 🦊 猎狐AI量化交易系统 - AI仪表板
 * 实时AI状态监控，豪华黑金界面交互
 * 专为史诗级AI量化交易设计
 */

class AILevelSystem {
    constructor() {
        this.levels = [
            { name: '青铜', min: 1, max: 20, color: '#CD7F32', multiplier: 2 },
            { name: '白银', min: 21, max: 40, color: '#C0C0C0', multiplier: 5 },
            { name: '黄金', min: 41, max: 60, color: '#FFD700', multiplier: 8 },
            { name: '铂金', min: 61, max: 80, color: '#E5E4E2', multiplier: 12 },
            { name: '钻石', min: 81, max: 95, color: '#B9F2FF', multiplier: 16 },
            { name: '史诗', min: 96, max: 100, color: '#FF6B35', multiplier: 20 }
        ];
    }

    getLevelInfo(level) {
        for (const tier of this.levels) {
            if (level >= tier.min && level <= tier.max) {
                return tier;
            }
        }
        return this.levels[0];
    }

    calculateProgress(level) {
        const tier = this.getLevelInfo(level);
        const progress = ((level - tier.min) / (tier.max - tier.min)) * 100;
        return Math.max(0, Math.min(100, progress));
    }
}

class AIModelMonitor {
    constructor(modelId, modelName, modelType) {
        this.modelId = modelId;
        this.modelName = modelName;
        this.modelType = modelType;
        this.status = 'initializing';
        this.accuracy = 0;
        this.confidence = 0;
        this.latency = 0;
        this.predictions = 0;
        this.errors = 0;
        this.level = 1;
        this.experience = 0;
        this.performance = 0;
        
        this.element = null;
        this.charts = {};
        
        this.createUI();
        this.startMonitoring();
    }

    createUI() {
        const aiModelsContainer = document.getElementById('ai-models-container');
        if (!aiModelsContainer) return;

        this.element = document.createElement('div');
        this.element.className = 'ai-model';
        this.element.id = `ai-model-${this.modelId}`;
        
        this.element.innerHTML = `
            <div class="ai-model-header">
                <div class="ai-model-name">${this.modelName}</div>
                <div class="ai-status ${this.status}" id="status-${this.modelId}">${this.status.toUpperCase()}</div>
            </div>
            
            <div class="ai-level-info">
                <div class="ai-level-badge" id="level-badge-${this.modelId}">
                    <span class="level-tier">青铜</span>
                    <span class="level-number">Lv.${this.level}</span>
                </div>
                <div class="ai-experience-bar">
                    <div class="experience-fill" id="exp-fill-${this.modelId}" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="ai-metrics">
                <div class="ai-metric">
                    <div class="ai-metric-label">准确率</div>
                    <div class="ai-metric-value" id="accuracy-${this.modelId}">0%</div>
                </div>
                <div class="ai-metric">
                    <div class="ai-metric-label">置信度</div>
                    <div class="ai-metric-value" id="confidence-${this.modelId}">0%</div>
                </div>
                <div class="ai-metric">
                    <div class="ai-metric-label">延迟</div>
                    <div class="ai-metric-value" id="latency-${this.modelId}">0ms</div>
                </div>
                <div class="ai-metric">
                    <div class="ai-metric-label">预测数</div>
                    <div class="ai-metric-value" id="predictions-${this.modelId}">0</div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progress-${this.modelId}" style="width: 0%"></div>
            </div>
            
            <div class="ai-performance-chart" id="chart-${this.modelId}"></div>
        `;
        
        aiModelsContainer.appendChild(this.element);
        
        // 添加点击事件
        this.element.addEventListener('click', () => this.showDetailModal());
    }

    updateStatus(data) {
        this.status = data.status || this.status;
        this.accuracy = data.accuracy || this.accuracy;
        this.confidence = data.confidence || this.confidence;
        this.latency = data.latency || this.latency;
        this.predictions = data.predictions || this.predictions;
        this.errors = data.errors || this.errors;
        this.level = data.level || this.level;
        this.experience = data.experience || this.experience;
        this.performance = data.performance || this.performance;

        this.updateUI();
    }

    updateUI() {
        if (!this.element) return;

        // 更新状态
        const statusElement = document.getElementById(`status-${this.modelId}`);
        if (statusElement) {
            statusElement.textContent = this.status.toUpperCase();
            statusElement.className = `ai-status ${this.status}`;
        }

        // 更新等级信息
        const levelSystem = new AILevelSystem();
        const levelInfo = levelSystem.getLevelInfo(this.level);
        const levelProgress = levelSystem.calculateProgress(this.level);

        const levelBadge = document.getElementById(`level-badge-${this.modelId}`);
        if (levelBadge) {
            levelBadge.innerHTML = `
                <span class="level-tier" style="color: ${levelInfo.color}">${levelInfo.name}</span>
                <span class="level-number">Lv.${this.level}</span>
            `;
        }

        const expFill = document.getElementById(`exp-fill-${this.modelId}`);
        if (expFill) {
            expFill.style.width = `${levelProgress}%`;
            expFill.style.background = `linear-gradient(90deg, ${levelInfo.color}, #FFD700)`;
        }

        // 更新指标
        this.updateMetric('accuracy', `${(this.accuracy * 100).toFixed(1)}%`);
        this.updateMetric('confidence', `${(this.confidence * 100).toFixed(1)}%`);
        this.updateMetric('latency', `${this.latency.toFixed(0)}ms`);
        this.updateMetric('predictions', this.predictions.toString());

        // 更新进度条
        const progressFill = document.getElementById(`progress-${this.modelId}`);
        if (progressFill) {
            const progress = Math.min(100, this.performance * 100);
            progressFill.style.width = `${progress}%`;
            
            // 根据性能设置颜色
            if (progress >= 80) {
                progressFill.style.background = 'linear-gradient(90deg, #00FF00, #00FFFF)';
            } else if (progress >= 60) {
                progressFill.style.background = 'linear-gradient(90deg, #FFD700, #00FFFF)';
            } else {
                progressFill.style.background = 'linear-gradient(90deg, #FF6B35, #FFD700)';
            }
        }

        // 更新性能图表
        this.updatePerformanceChart();
    }

    updateMetric(metricName, value) {
        const element = document.getElementById(`${metricName}-${this.modelId}`);
        if (element) {
            element.textContent = value;
            
            // 添加闪烁效果表示更新
            element.style.animation = 'metricUpdate 0.5s ease-in-out';
            setTimeout(() => {
                element.style.animation = '';
            }, 500);
        }
    }

    updatePerformanceChart() {
        // 简化的性能图表更新
        const chartElement = document.getElementById(`chart-${this.modelId}`);
        if (!chartElement) return;

        // 创建简单的性能条
        const performanceLevel = Math.min(100, this.performance * 100);
        chartElement.innerHTML = `
            <div class="mini-chart">
                <div class="chart-bar" style="height: ${performanceLevel}%; background: linear-gradient(180deg, #00FF00, #FFD700)"></div>
            </div>
        `;
    }

    showDetailModal() {
        const modal = document.getElementById('ai-detail-modal');
        if (!modal) return;

        const modalContent = modal.querySelector('.modal-content');
        modalContent.innerHTML = `
            <div class="modal-header">
                <h2>${this.modelName} 详细信息</h2>
                <span class="modal-close">&times;</span>
            </div>
            <div class="modal-body">
                <div class="detail-section">
                    <h3>基本信息</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>模型ID:</label>
                            <span>${this.modelId}</span>
                        </div>
                        <div class="detail-item">
                            <label>模型类型:</label>
                            <span>${this.modelType}</span>
                        </div>
                        <div class="detail-item">
                            <label>当前状态:</label>
                            <span class="status-${this.status}">${this.status.toUpperCase()}</span>
                        </div>
                        <div class="detail-item">
                            <label>等级:</label>
                            <span>Lv.${this.level}</span>
                        </div>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h3>性能指标</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>准确率:</label>
                            <span>${(this.accuracy * 100).toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <label>置信度:</label>
                            <span>${(this.confidence * 100).toFixed(2)}%</span>
                        </div>
                        <div class="detail-item">
                            <label>平均延迟:</label>
                            <span>${this.latency.toFixed(2)}ms</span>
                        </div>
                        <div class="detail-item">
                            <label>预测次数:</label>
                            <span>${this.predictions}</span>
                        </div>
                        <div class="detail-item">
                            <label>错误次数:</label>
                            <span>${this.errors}</span>
                        </div>
                        <div class="detail-item">
                            <label>成功率:</label>
                            <span>${((this.predictions - this.errors) / Math.max(this.predictions, 1) * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h3>历史表现</h3>
                    <div id="detail-chart-${this.modelId}" class="detail-chart"></div>
                </div>
            </div>
        `;

        modal.style.display = 'block';
        
        // 绑定关闭事件
        const closeBtn = modal.querySelector('.modal-close');
        closeBtn.onclick = () => modal.style.display = 'none';
        
        window.onclick = (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        };
    }

    startMonitoring() {
        // 模拟数据更新
        setInterval(() => {
            this.simulateDataUpdate();
        }, 2000);
    }

    simulateDataUpdate() {
        // 模拟AI模型数据更新
        const statusOptions = ['ready', 'training', 'predicting'];
        const randomStatus = statusOptions[Math.floor(Math.random() * statusOptions.length)];
        
        this.updateStatus({
            status: randomStatus,
            accuracy: Math.min(1, this.accuracy + (Math.random() - 0.5) * 0.01),
            confidence: Math.min(1, this.confidence + (Math.random() - 0.5) * 0.02),
            latency: Math.max(10, this.latency + (Math.random() - 0.5) * 5),
            predictions: this.predictions + Math.floor(Math.random() * 3),
            errors: this.errors + (Math.random() < 0.1 ? 1 : 0),
            performance: Math.min(1, this.performance + (Math.random() - 0.5) * 0.01)
        });
    }
}

class TradingDashboard {
    constructor() {
        this.aiModels = new Map();
        this.isConnected = false;
        this.websocket = null;
        
        this.init();
    }

    init() {
        this.createAIModels();
        this.setupWebSocket();
        this.setupEventListeners();
        this.startSystemMonitoring();
        
        console.log('🦊 猎狐AI仪表板初始化完成');
    }

    createAIModels() {
        const aiModelsData = [
            { id: 'meta_commander', name: '元学习AI指挥官', type: 'Meta Learning' },
            { id: 'rl_trader', name: '强化学习交易员', type: 'Reinforcement Learning' },
            { id: 'lstm_prophet', name: '时序预测先知', type: 'Time Series' },
            { id: 'ensemble_advisor', name: '集成学习智囊团', type: 'Ensemble Learning' },
            { id: 'transfer_adapter', name: '迁移学习适配器', type: 'Transfer Learning' },
            { id: 'expert_guardian', name: '专家系统守护者', type: 'Expert System' },
            { id: 'sentiment_scout', name: '情感分析侦察兵', type: 'Sentiment Analysis' },
            { id: 'factor_miner', name: '量化因子挖掘机', type: 'Factor Mining' }
        ];

        aiModelsData.forEach(modelData => {
            const aiModel = new AIModelMonitor(modelData.id, modelData.name, modelData.type);
            this.aiModels.set(modelData.id, aiModel);
        });
    }

    setupWebSocket() {
        // 设置WebSocket连接用于实时数据
        try {
            this.websocket = new WebSocket(`ws://${window.location.host}/ws`);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus(true);
                console.log('🔗 WebSocket连接已建立');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ WebSocket消息解析失败:', error);
                }
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus(false);
                console.log('🔌 WebSocket连接已断开');
                
                // 尝试重连
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket错误:', error);
            };
            
        } catch (error) {
            console.error('❌ WebSocket初始化失败:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'ai_status_update':
                if (this.aiModels.has(data.model_id)) {
                    this.aiModels.get(data.model_id).updateStatus(data.data);
                }
                break;
                
            case 'system_stats':
                this.updateSystemStats(data.data);
                break;
                
            case 'trading_signal':
                this.updateTradingSignals(data.data);
                break;
                
            case 'portfolio_update':
                this.updatePortfolio(data.data);
                break;
                
            default:
                console.log('📨 未知消息类型:', data.type);
        }
    }

    setupEventListeners() {
        // 设置各种事件监听器
        
        // 图表控制按钮
        document.querySelectorAll('.chart-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const timeframe = e.target.dataset.timeframe;
                this.updateChart(timeframe);
            });
        });
        
        // 系统控制按钮
        const emergencyStopBtn = document.getElementById('emergency-stop');
        if (emergencyStopBtn) {
            emergencyStopBtn.addEventListener('click', () => {
                this.emergencyStop();
            });
        }
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 's':
                        e.preventDefault();
                        this.emergencyStop();
                        break;
                }
            }
        });
    }

    startSystemMonitoring() {
        // 定期更新系统状态
        setInterval(() => {
            this.updateSystemMetrics();
        }, 1000);
        
        // 定期更新交易数据
        setInterval(() => {
            this.updateTradingData();
        }, 5000);
    }

    updateSystemMetrics() {
        // 更新CPU使用率
        const cpuElement = document.getElementById('cpu-usage');
        if (cpuElement) {
            const cpuUsage = Math.random() * 100;
            cpuElement.textContent = `${cpuUsage.toFixed(1)}%`;
            cpuElement.className = cpuUsage > 80 ? 'status-value negative' : 'status-value';
        }
        
        // 更新内存使用率
        const memoryElement = document.getElementById('memory-usage');
        if (memoryElement) {
            const memoryUsage = Math.random() * 100;
            memoryElement.textContent = `${memoryUsage.toFixed(1)}%`;
            memoryElement.className = memoryUsage > 90 ? 'status-value negative' : 'status-value';
        }
        
        // 更新GPU温度
        const gpuTempElement = document.getElementById('gpu-temp');
        if (gpuTempElement) {
            const gpuTemp = 65 + Math.random() * 20;
            gpuTempElement.textContent = `${gpuTemp.toFixed(0)}°C`;
            gpuTempElement.className = gpuTemp > 80 ? 'status-value negative' : 'status-value';
        }
    }

    updateTradingData() {
        // 更新总收益
        const totalPnlElement = document.getElementById('total-pnl');
        if (totalPnlElement) {
            const pnl = (Math.random() - 0.4) * 10000;
            totalPnlElement.textContent = `$${pnl.toFixed(2)}`;
            totalPnlElement.className = pnl >= 0 ? 'status-value positive' : 'status-value negative';
        }
        
        // 更新今日收益
        const dailyPnlElement = document.getElementById('daily-pnl');
        if (dailyPnlElement) {
            const dailyPnl = (Math.random() - 0.3) * 1000;
            dailyPnlElement.textContent = `$${dailyPnl.toFixed(2)}`;
            dailyPnlElement.className = dailyPnl >= 0 ? 'status-value positive' : 'status-value negative';
        }
        
        // 更新AI等级
        const aiLevelElement = document.getElementById('ai-level');
        if (aiLevelElement) {
            const level = Math.floor(Math.random() * 100) + 1;
            const levelSystem = new AILevelSystem();
            const levelInfo = levelSystem.getLevelInfo(level);
            aiLevelElement.innerHTML = `<span style="color: ${levelInfo.color}">${levelInfo.name} Lv.${level}</span>`;
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? '已连接' : '断开连接';
            statusElement.className = connected ? 'status-value positive' : 'status-value negative';
        }
    }

    updateChart(timeframe) {
        console.log(`📊 更新图表时间框架: ${timeframe}`);
        // 这里可以集成真实的图表库如Chart.js或TradingView
    }

    updateSystemStats(data) {
        // 更新系统统计信息
        console.log('📊 系统统计更新:', data);
    }

    updateTradingSignals(data) {
        // 更新交易信号
        const signalsContainer = document.getElementById('trading-signals');
        if (signalsContainer && data.signals) {
            let signalsHTML = '';
            data.signals.forEach(signal => {
                const signalClass = signal.action === 'BUY' ? 'buy' : 
                                  signal.action === 'SELL' ? 'sell' : 'hold';
                signalsHTML += `
                    <div class="signal-item">
                        <span class="signal-name">${signal.symbol}</span>
                        <span class="signal-value ${signalClass}">${signal.action}</span>
                    </div>
                `;
            });
            signalsContainer.innerHTML = signalsHTML;
        }
    }

    updatePortfolio(data) {
        // 更新投资组合
        const portfolioContainer = document.getElementById('portfolio-positions');
        if (portfolioContainer && data.positions) {
            let positionsHTML = '';
            data.positions.forEach(position => {
                const pnlClass = position.pnl >= 0 ? 'positive' : 'negative';
                positionsHTML += `
                    <div class="position-item">
                        <span class="position-symbol">${position.symbol}</span>
                        <span class="position-amount">${position.amount}</span>
                        <span class="position-pnl ${pnlClass}">$${position.pnl.toFixed(2)}</span>
                    </div>
                `;
            });
            portfolioContainer.innerHTML = positionsHTML;
        }
    }

    refreshData() {
        console.log('🔄 刷新数据');
        // 刷新所有数据
        this.aiModels.forEach(model => {
            model.simulateDataUpdate();
        });
    }

    emergencyStop() {
        if (confirm('⚠️ 确定要执行紧急停止吗？这将停止所有交易活动。')) {
            console.log('🛑 执行紧急停止');
            
            // 发送紧急停止信号
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'emergency_stop',
                    timestamp: new Date().toISOString()
                }));
            }
            
            // 更新UI状态
            const emergencyBtn = document.getElementById('emergency-stop');
            if (emergencyBtn) {
                emergencyBtn.textContent = '系统已停止';
                emergencyBtn.style.background = '#FF0040';
                emergencyBtn.disabled = true;
            }
        }
    }
}

// 页面加载完成后初始化仪表板
document.addEventListener('DOMContentLoaded', () => {
    // 添加必要的CSS动画
    const style = document.createElement('style');
    style.textContent = `
        @keyframes metricUpdate {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); color: #00FFFF; }
            100% { transform: scale(1); }
        }
        
        .mini-chart {
            height: 30px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            margin-top: 10px;
        }
        
        .chart-bar {
            width: 20px;
            background: linear-gradient(180deg, #00FF00, #FFD700);
            border-radius: 2px;
            transition: height 0.3s ease;
        }
        
        .ai-level-badge {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .level-tier {
            font-size: 12px;
            font-weight: 700;
        }
        
        .level-number {
            font-size: 12px;
            color: #FFD700;
        }
        
        .ai-experience-bar {
            height: 4px;
            background: #0A0A0A;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .experience-fill {
            height: 100%;
            background: linear-gradient(90deg, #FFD700, #00FFFF);
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
        }
        
        .modal-content {
            background: linear-gradient(135deg, #1E1E1E 0%, #0A0A0A 100%);
            margin: 5% auto;
            padding: 0;
            border: 2px solid #FFD700;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            color: #FFD700;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #FFD700;
        }
        
        .modal-close {
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: #FFD700;
        }
        
        .modal-close:hover {
            color: #00FFFF;
        }
        
        .modal-body {
            padding: 20px;
        }
        
        .detail-section {
            margin-bottom: 30px;
        }
        
        .detail-section h3 {
            color: #FFD700;
            margin-bottom: 15px;
            text-shadow: 0 0 10px #FFD700;
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .detail-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(255, 215, 0, 0.1);
            border-radius: 4px;
            border: 1px solid rgba(255, 215, 0, 0.3);
        }
        
        .detail-item label {
            color: #C0C0C0;
            font-weight: 700;
        }
        
        .detail-chart {
            height: 200px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #FFD700;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #C0C0C0;
        }
    `;
    document.head.appendChild(style);
    
    // 创建模态框
    const modal = document.createElement('div');
    modal.id = 'ai-detail-modal';
    modal.className = 'modal';
    modal.innerHTML = '<div class="modal-content"></div>';
    document.body.appendChild(modal);
    
    // 初始化仪表板
    window.tradingDashboard = new TradingDashboard();
    
    console.log('🦊 猎狐AI仪表板已启动');
});

