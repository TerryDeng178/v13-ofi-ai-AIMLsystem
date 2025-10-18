# CVD系统快速参考指南

## 🚀 **5分钟快速开始**

### 1. 运行测试
```bash
cd v13_ofi_ai_system/examples
python run_realtime_cvd.py --symbol ETHUSDT --duration 300
```

### 2. 分析结果
```bash
python analysis_cvd.py --data ../data/cvd_ethusdt_*.parquet
```

## ⚙️ **核心配置 (Step 1.6 生产版)**

```bash
# 最优配置 - 直接复制使用
CVD_Z_MODE=delta
HALF_LIFE_TRADES=300
WINSOR_LIMIT=8.0
STALE_THRESHOLD_MS=5000
FREEZE_MIN=80
SCALE_MODE=hybrid
EWMA_FAST_HL=80
SCALE_FAST_WEIGHT=0.35
SCALE_SLOW_WEIGHT=0.65
MAD_WINDOW_TRADES=300
MAD_SCALE_FACTOR=1.4826
MAD_MULTIPLIER=1.45
WATERMARK_MS=2000
```

## 📊 **质量指标**

| 指标 | 目标 | 当前 | 状态 |
|------|------|------|------|
| P(|Z|>2) | ≤8% | 5.73% | ✅ |
| P(|Z|>3) | ≤2% | 4.65% | 🎯 |
| median(|Z|) | ≤1.0 | 0.0013 | ✅ |
| 数据完整性 | 100% | 100% | ✅ |

## 🔧 **常用命令**

### 配置切换
```bash
# 分析模式
source ../config/profiles/analysis.env

# 实时模式  
source ../config/profiles/realtime.env
```

### 数据清理
```bash
# 清理旧数据
rm -rf ../data/cvd_*
rm -rf ../figs_cvd_*
```

### 监控检查
```bash
# 检查最新数据质量
python analysis_cvd.py --data $(ls -t ../data/cvd_*/cvd_*.parquet | head -1)
```

## 🚨 **故障快速修复**

### Z-score不达标
```bash
# 提高地板
MAD_MULTIPLIER=1.50

# 调整权重
SCALE_FAST_WEIGHT=0.30
```

### 延迟过高
```bash
# 降低水位线
WATERMARK_MS=500
```

### 数据解析错误
```bash
# 检查网络连接
ping fstream.binancefuture.com
```

## 📁 **关键文件**

- **核心引擎**: `src/real_cvd_calculator.py`
- **数据采集**: `examples/run_realtime_cvd.py`
- **数据分析**: `examples/analysis_cvd.py`
- **生产配置**: `config/profiles/analysis.env`
- **详细文档**: `docs/CVD_SYSTEM_README.md`

## 🎯 **下一步**

1. **生产部署**: 使用`config/profiles/analysis.env`
2. **监控设置**: 参考`docs/monitoring/dashboard_config.md`
3. **性能优化**: 查看`docs/roadmap/P1.2_optimization_plan.md`

---
**快速支持**: 查看详细文档 `docs/CVD_SYSTEM_README.md`
