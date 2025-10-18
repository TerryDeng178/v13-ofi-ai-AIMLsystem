# V13 OFI+AI 量化交易系统

> **从真实开始，每一步都验证**

## 🎯 项目简介

V13是一个基于Order Flow Imbalance (OFI)的量化交易系统，逐步集成AI模型优化。

**核心原则**:
- ✅ **真实优先**: 每个功能都用真实币安数据验证
- ✅ **简单开始**: 先做OFI核心，再扩展AI
- ✅ **逐步迭代**: 每个阶段都有可见效果
- ✅ **全局思维**: 所有组件协同工作

## 📁 项目结构

```
v13_ofi_ai_system/
├── README.md                    # 项目说明（本文件）
├── requirements.txt             # Python依赖
├── config/
│   └── binance_config.yaml     # 币安API配置
├── src/                        # 核心源代码
│   ├── binance_websocket_client.py    # WebSocket数据接入
│   ├── real_ofi_calculator.py         # 真实OFI计算
│   ├── binance_testnet_trader.py      # 测试网交易
│   ├── simple_ofi_strategy.py         # 简单OFI策略
│   ├── simple_ai_model.py             # 简单AI模型（阶段3）
│   └── ai_enhanced_ofi_strategy.py    # AI增强策略（阶段3）
├── tests/                      # 测试代码
│   └── test_ofi_signal_validity.py    # OFI信号验证
├── data/                       # 数据存储
│   ├── ofi_history.csv         # OFI历史数据
│   └── trades.csv              # 交易记录
├── examples/                   # 示例代码
│   └── run_live_trading.py     # 实盘运行示例
└── docs/                       # 文档
    └── V13_DEVELOPMENT_GUIDE.md # 开发指导

```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置币安API

编辑 `config/binance_config.yaml`:
```yaml
binance:
  testnet:
    api_key: "YOUR_TESTNET_API_KEY"
    api_secret: "YOUR_TESTNET_API_SECRET"
  mainnet:
    api_key: "YOUR_MAINNET_API_KEY"
    api_secret: "YOUR_MAINNET_API_SECRET"
```

### 3. 运行WebSocket数据接收（阶段1.1）

```bash
python src/binance_websocket_client.py
```

### 4. 运行OFI计算（阶段1.2）

```bash
python src/real_ofi_calculator.py
```

## 📊 开发阶段

- [x] **阶段0**: 项目准备（已完成）
- [ ] **阶段1**: 真实OFI核心（3-5天）
  - [ ] 1.1 币安WebSocket数据接入
  - [ ] 1.2 真实OFI计算
  - [ ] 1.3 OFI信号验证
- [ ] **阶段2**: 简单但真实的交易（2-3天）
  - [ ] 2.1 币安测试网交易接入
  - [ ] 2.2 简单OFI交易策略
  - [ ] 2.3 真实交易测试
- [ ] **阶段3**: 逐步加入AI（5-7天）
  - [ ] 3.1 数据收集与特征工程
  - [ ] 3.2 简单AI模型训练
  - [ ] 3.3 AI增强交易策略
- [ ] **阶段4**: 深度学习优化（可选，7-10天）

## 🎯 当前目标

**阶段1.1**: 实现币安WebSocket客户端，能实时接收订单簿数据

**成功指标**:
- ✅ 能连续接收1小时以上的订单簿数据
- ✅ 数据完整性 >95%
- ✅ 延迟 <500ms

## 📝 开发日志

### 2025-01-17
- ✅ 创建V13项目结构
- ⏳ 开始阶段1.1: WebSocket数据接入

## 📚 参考文档

- [V13开发指导](docs/V13_DEVELOPMENT_GUIDE.md) - 完整的开发路线图
- [V12经验总结](../🌟V12_TECHNICAL_FRAMEWORK_DEVELOPMENT_PLAN.md) - V12项目的经验教训

## 🎯 Quality Gates (质量门控)

### 生产环境质量标准

**硬线指标 (必须达标)**:
- ✅ **P(|Z|>2) ≤ 8%** - Z-score尾部控制
- ✅ **median(|Z|) ≤ 1.0** - Z-score中位数控制
- ✅ **数据完整性 = 100%** - 解析错误=0，队列丢弃率=0%
- ✅ **数据一致性 = 100%** - 逐笔守恒错误=0
- ✅ **ID健康 = 100%** - 重复ID=0，倒序ID=0

**优化指标 (持续改进)**:
- 🎯 **P(|Z|>3) ≤ 2%** - 当前4.65%，目标≤2%
- 🎯 **P95(|Z|) ≤ 3.0** - 当前2.71，持续优化
- 🎯 **延迟优化** - 分析模式≤5s，实时模式≤1s

### 配置档位

- **分析模式**: `config/profiles/analysis.env` (WATERMARK_MS=2000)
- **实时模式**: `config/profiles/realtime.env` (WATERMARK_MS=500)

## ⚠️ 重要原则

**绝对禁止**:
- ❌ 使用Mock数据（除单元测试）
- ❌ 构建"假"组件
- ❌ 跳过真实数据验证
- ❌ 在没有真实基础前过度优化

**每日检查**:
- [ ] 今天的代码用了真实数据吗？
- [ ] 今天的功能经过验证了吗？
- [ ] 今天的进展能展示效果吗？

## 📞 联系与支持

如有问题，请查看 `docs/V13_DEVELOPMENT_GUIDE.md` 获取详细指导。

---

**版本**: V13.0.0  
**状态**: 开发中  
**最后更新**: 2025-01-17

