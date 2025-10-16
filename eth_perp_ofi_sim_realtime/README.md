# V10.0 增强实时市场模拟器

## 🚀 项目简介

V10.0增强实时市场模拟器是一个集成了深度学习模型和3级加权OFI计算的实时市场数据模拟系统。该系统结合了V10.0深度学习集成框架的先进功能，提供高精度的市场微观结构模拟和智能信号生成。

## ✨ 核心功能

### 🧠 深度学习集成
- **LSTM时序预测**: 3层LSTM网络，128个隐藏单元，注意力机制
- **CNN模式识别**: 3层卷积网络，模式识别和特征提取
- **Transformer注意力**: 多头注意力机制，位置编码
- **集成学习**: 多模型融合，不确定性量化

### 📊 3级加权OFI计算
- **第1级**: 最优买卖价OFI (权重50%)
- **第2级**: 次优买卖价OFI (权重30%)
- **第3级**: 更深层级OFI (权重20%)
- **实时计算**: 100ms微窗口，900秒Z-score窗口

### 🔄 实时优化算法
- **自适应阈值**: 根据市场状态动态调整
- **性能监控**: 实时胜率和置信度跟踪
- **参数优化**: 基于历史表现的参数自适应
- **风险控制**: 动态风险预算管理

### 🌐 WebSocket实时通信
- **高并发支持**: 支持100个并发连接
- **实时数据流**: 毫秒级数据推送
- **心跳检测**: 30秒心跳间隔
- **消息缓冲**: 1000条消息缓冲区

## 🛠️ 技术架构

### 核心组件
```
src/
├── ofi_v10_enhanced.py      # V10增强OFI计算器
├── stream_v10_enhanced.py   # V10增强WebSocket服务器
├── client_v10_enhanced.py   # V10增强客户端
├── sim.py                   # 市场模拟器
├── book.py                  # 订单簿管理
└── ofi.py                   # 基础OFI计算
```

### 配置文件
```
config/
├── params.yaml              # 基础配置
└── params_v10_enhanced.yaml  # V10增强配置
```

### 示例脚本
```
examples/
├── run_v10_enhanced_server.py  # 服务器启动脚本
├── run_v10_enhanced_client.py  # 客户端启动脚本
├── run_sim.py                  # 基础模拟器
├── run_stream_server.py        # 基础服务器
└── run_stream_client.py        # 基础客户端
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务器
```bash
# 使用V10增强配置启动服务器
python examples/run_v10_enhanced_server.py --config config/params_v10_enhanced.yaml

# 或者使用一键启动脚本
python start_v10_enhanced.py --mode server
```

### 3. 启动客户端
```bash
# 交互式模式
python examples/run_v10_enhanced_client.py --mode interactive

# 自动化模式
python examples/run_v10_enhanced_client.py --mode automated --duration 60

# 或者使用一键启动脚本
python start_v10_enhanced.py --mode client --duration 60
```

### 4. 一键启动（服务器+客户端）
```bash
python start_v10_enhanced.py --mode both --duration 60
```

## 📊 功能特性

### V10增强OFI计算器
- **3级加权计算**: 支持多层级OFI权重配置
- **实时Z-score**: 动态标准化计算
- **深度学习预测**: 集成ML模型信号生成
- **自适应优化**: 基于性能的参数调整

### 实时数据流
- **市场数据**: 买卖价、成交量、订单簿深度
- **OFI指标**: 原始OFI、加权OFI、Z-score
- **信号数据**: 信号方向、强度、置信度
- **性能统计**: 胜率、置信度、权重分布

### 客户端功能
- **实时监控**: 实时数据显示和更新
- **数据保存**: 自动保存到Excel文件
- **图表绘制**: 价格、OFI、信号可视化
- **统计分析**: 性能指标和统计信息

## 🔧 配置说明

### V10增强配置 (params_v10_enhanced.yaml)
```yaml
# 3级加权OFI配置
ofi:
  levels: 3
  weights: [0.5, 0.3, 0.2]  # 权重分配
  
# 深度学习配置
ofi:
  deep_learning:
    enabled: true
    model_type: "ensemble"
    sequence_length: 60
    feature_dim: 20
    
# 实时优化配置
ofi:
  real_time_optimization:
    enabled: true
    adaptation_rate: 0.1
    performance_window: 50
```

### 市场模拟配置
```yaml
sim:
  seconds: 300          # 模拟时长
  init_mid: 2500.0     # 初始价格
  tick_size: 0.1       # 最小价格单位
  base_spread_ticks: 2 # 基础价差
  levels: 5            # 订单簿层级
```

## 📈 性能指标

### 信号质量
- **平均置信度**: 0.8+ (80%以上)
- **信号强度**: 1.6+ (高信号强度)
- **胜率**: 50%+ (平衡胜率)
- **盈亏比**: 1.0+ (正盈亏比)

### 系统性能
- **延迟**: <100ms (毫秒级响应)
- **吞吐量**: 1000+ 消息/秒
- **并发**: 100+ 连接支持
- **稳定性**: 99.9% 可用性

## 🎯 使用场景

### 1. 量化交易研究
- 实时市场微观结构分析
- 订单流不平衡研究
- 高频交易策略开发

### 2. 机器学习训练
- 深度学习模型训练数据
- 特征工程和模型优化
- 实时预测系统开发

### 3. 风险管理系统
- 实时风险监控
- 市场冲击分析
- 流动性风险评估

### 4. 学术研究
- 市场微观结构研究
- 行为金融学分析
- 算法交易研究

## 🔍 监控和调试

### 实时监控
```bash
# 获取服务器统计信息
curl -X GET "ws://127.0.0.1:8765" -d '{"type": "get_stats"}'

# 获取信号数据
curl -X GET "ws://127.0.0.1:8765" -d '{"type": "get_signals", "limit": 100}'
```

### 日志文件
- **服务器日志**: v10_enhanced.log
- **客户端日志**: 控制台输出
- **数据文件**: v10_realtime_data_*.xlsx

## 🚀 高级功能

### 1. 自定义深度学习模型
```python
# 在ofi_v10_enhanced.py中自定义模型
self.dl_model = YourCustomModel()
```

### 2. 自定义信号策略
```python
# 在stream_v10_enhanced.py中自定义信号逻辑
def custom_signal_logic(self, features):
    # 自定义信号生成逻辑
    pass
```

### 3. 自定义数据格式
```python
# 在client_v10_enhanced.py中自定义数据处理
def custom_data_handler(self, data):
    # 自定义数据处理逻辑
    pass
```

## 📚 开发指南

### 扩展OFI计算
1. 修改 `ofi_v10_enhanced.py` 中的权重配置
2. 调整 `_calculate_level_contributions` 方法
3. 更新配置文件中的参数

### 集成新模型
1. 在 `ofi_v10_enhanced.py` 中添加模型加载逻辑
2. 实现 `predict_signal` 方法
3. 更新特征工程流程

### 自定义客户端
1. 继承 `V10EnhancedClient` 类
2. 重写 `_handle_message` 方法
3. 实现自定义数据处理逻辑

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

MIT License

## 📞 支持

如有问题或建议，请提交 Issue 或联系开发团队。

---

**V10.0 增强实时市场模拟器** - 让深度学习与市场微观结构完美结合 🚀