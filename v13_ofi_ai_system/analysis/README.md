# OFI+CVD 信号分析工具使用说明

## 📋 文档信息
- **模块路径**: `v13_ofi_ai_system/analysis/`
- **版本**: v2.0-prod
- **创建时间**: 2025-10-21
- **最后更新**: 2025-10-22
- **任务来源**: Task 1.3.2 - 创建OFI+CVD信号分析工具

## 🎯 模块概述

本模块是V13 OFI+CVD+AI系统的核心信号分析工具，基于Task 1.3.1收集的分区化数据，提供OFI、CVD、Fusion、背离四类信号的离线质量评估与对比分析。

### 核心功能
- ✅ **信号质量评估**: AUC、PR-AUC、IC、F1、Brier、ECE等指标计算
- ✅ **切片稳健性分析**: 按活跃度/时段/波动/交易对切片分析
- ✅ **阈值扫描优化**: 网格搜索最佳阈值（|z|∈[0.5,3.0], step=0.1）
- ✅ **校准分析**: Platt/Isotonic校准，提升概率预测准确性
- ✅ **事件分析**: 背离/枢轴/异常事件的后向收益分析
- ✅ **可视化输出**: ROC/PR曲线、单调性图、校准图等

## 📦 模块结构

```
analysis/
├── __init__.py                 # 模块初始化
├── ofi_cvd_signal_eval.py      # 主分析逻辑与CLI (1173行)
├── utils_labels.py             # 标签构造与切片工具 (362行)
├── plots.py                    # 可视化图表生成 (373行)
└── README.md                   # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保Python环境
python --version  # 需要Python 3.8+

# 安装依赖
pip install pandas numpy scikit-learn matplotlib seaborn

# 进入项目目录
cd v13_ofi_ai_system
```

### 2. 基本使用

```bash
# 运行完整分析
python -m analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT,BTCUSDT \
  --date-from 2025-10-18 --date-to 2025-10-21 \
  --horizons 60,180,300,900 \
  --fusion "w_ofi=0.6,w_cvd=0.4,gate=0.0" \
  --labels mid \
  --calibration platt \
  --out artifacts/analysis/ofi_cvd \
  --run-tag 20251022_eval
```

### 3. 输出结果

分析完成后，将在`artifacts/analysis/ofi_cvd/`目录下生成：

```
artifacts/analysis/ofi_cvd/
├── run_tag.txt                     # 配置指纹
├── summary/
│   ├── metrics_overview.csv        # 总表（各信号×各窗口）
│   ├── slices_active_quiet.csv    # 活跃度切片
│   ├── slices_tod.csv             # 时段切片
│   ├── slices_vol.csv             # 波动切片
│   ├── merge_time_diff_ms.csv     # 合并时间差分布
│   ├── platt_samples.csv          # Platt校准样本量
│   └── slice_auc_active_vs_quiet.csv # 切片AUC对比
├── charts/                         # 可视化图表
│   ├── roc_curves.png             # ROC曲线
│   ├── pr_curves.png              # PR曲线
│   ├── monotonicity.png           # 单调性分析
│   ├── calibration.png            # 校准分析
│   └── events_analysis.png        # 事件分析
└── reports/
    └── report_20251022.json       # 机器可读摘要
```

## 🔧 核心模块详解

### 1. ofi_cvd_signal_eval.py - 主分析逻辑

**功能**: 信号分析的核心引擎，包含数据加载、指标计算、切片分析、阈值扫描等。

**关键类**:
- `OFICVDSignalEvaluator`: 主分析器类
- `LabelConstructor`: 标签构造器
- `MetricsCalculator`: 指标计算器
- `SliceAnalyzer`: 切片分析器

**核心方法**:
```python
# 数据加载
def load_data(self) -> Dict[str, pd.DataFrame]

# 标签构造
def construct_labels(self, prices_df: pd.DataFrame) -> pd.DataFrame

# 指标计算
def calculate_metrics(self, signals_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict

# 切片分析
def analyze_slices(self, signals_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict

# 阈值扫描
def scan_thresholds(self, signal: np.ndarray, label: np.ndarray) -> Dict
```

**CLI参数**:
```bash
--data-root          # 数据根目录
--symbols            # 交易对列表
--date-from/--date-to # 日期范围
--horizons           # 前瞻窗口（秒）
--fusion             # Fusion权重配置
--labels             # 标签类型（mid/micro/trade）
--calibration        # 校准方法（platt/isotonic）
--out                # 输出目录
--run-tag            # 运行标签
```

### 2. utils_labels.py - 标签构造与切片工具

**功能**: 负责前瞻标签构造、数据切片、时间对齐等核心功能。

**关键类**:
- `LabelConstructor`: 标签构造器
- `SliceAnalyzer`: 切片分析器
- `TimeAligner`: 时间对齐器

**核心方法**:
```python
# 构造前瞻标签
def construct_labels(self, prices_df: pd.DataFrame) -> pd.DataFrame

# 时间对齐
def align_timestamps(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame

# 切片分析
def analyze_slices(self, df: pd.DataFrame) -> Dict

# 数据质量检查
def validate_data_quality(self, df: pd.DataFrame) -> Dict
```

**支持的标签类型**:
- `mid`: 中间价标签（推荐）
- `micro`: 微观价格标签
- `trade`: 成交价标签

**切片维度**:
- `regime`: 活跃度切片（Active/Quiet）
- `tod`: 时段切片（Tokyo/London/NY）
- `vol`: 波动切片（low/mid/high）

### 3. plots.py - 可视化图表生成

**功能**: 生成各种分析图表，包括ROC/PR曲线、单调性分析、校准图等。

**关键类**:
- `PlotGenerator`: 图表生成器
- `ROCAnalyzer`: ROC分析器
- `CalibrationAnalyzer`: 校准分析器

**核心方法**:
```python
# 生成ROC曲线
def plot_roc_curves(self, metrics_data: Dict) -> None

# 生成PR曲线
def plot_pr_curves(self, metrics_data: Dict) -> None

# 生成单调性分析
def plot_monotonicity(self, metrics_data: Dict) -> None

# 生成校准分析
def plot_calibration(self, metrics_data: Dict) -> None

# 生成事件分析
def plot_events_analysis(self, events_data: Dict) -> None
```

**输出图表**:
- `roc_curves.png`: ROC曲线对比
- `pr_curves.png`: PR曲线对比
- `monotonicity.png`: 单调性分析
- `calibration.png`: 校准分析
- `events_analysis.png`: 事件分析

## 📊 核心指标说明

### 分类指标
- **AUC**: 受试者工作特征曲线下面积，衡量分类性能
- **PR-AUC**: 精确率-召回率曲线下面积，适用于不平衡数据
- **F1**: 精确率和召回率的调和平均
- **IC**: 信息系数，信号与收益的秩相关

### 校准指标
- **Brier**: 概率预测的均方误差
- **ECE**: 期望校准误差，衡量概率校准质量

### 排序指标
- **Kendall τ**: 肯德尔秩相关系数，衡量单调性
- **Top-K命中率**: 前K%信号的命中率

## ⚙️ 配置参数

### Fusion配置
```python
fusion_config = {
    "w_ofi": 0.6,        # OFI权重
    "w_cvd": 0.4,        # CVD权重
    "gate": 0.0          # 门控阈值（默认关闭）
}
```

### 校准配置
```python
calibration_config = {
    "method": "platt",    # 校准方法（platt/isotonic）
    "train_window": 7200, # 训练窗口（秒）
    "test_window": 1800   # 测试窗口（秒）
}
```

### 阈值扫描配置
```python
threshold_config = {
    "min_threshold": 0.5,  # 最小阈值
    "max_threshold": 3.0,  # 最大阈值
    "step": 0.1,           # 步长
    "target_metric": "pr_auc"  # 目标指标
}
```

## 🔍 使用场景

### 场景1: 信号质量评估
```bash
# 评估ETHUSDT信号质量
python -m analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT \
  --date-from 2025-10-18 --date-to 2025-10-21 \
  --horizons 60,180,300 \
  --fusion "w_ofi=0.6,w_cvd=0.4,gate=0.0" \
  --out artifacts/analysis/eth_eval
```

### 场景2: 参数优化
```bash
# 扫描Fusion权重
python -m analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT \
  --fusion "w_ofi=0.7,w_cvd=0.3,gate=0.0" \
  --out artifacts/analysis/weight_scan
```

### 场景3: 切片分析
```bash
# 分析不同时段表现
python -m analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT,BTCUSDT \
  --slices "tod=Tokyo,London,NY" \
  --out artifacts/analysis/slice_analysis
```

## 📈 输出解读

### metrics_overview.csv
各信号×各窗口的性能指标总表：
```csv
signal_type,horizon,auc,pr_auc,ic,f1,brier,ece
OFI,60,0.560,0.520,0.03,0.45,0.25,0.08
CVD,60,0.440,0.420,0.01,0.38,0.28,0.12
Fusion,60,0.606,0.580,0.05,0.52,0.22,0.06
```

### report_YYYYMMDD.json
机器可读的摘要报告：
```json
{
  "config_fingerprint": "v2.0-prod-sha1hash",
  "cvd_direction": "flipped",
  "best_thresholds": {
    "ofi": 1.8,
    "cvd": 1.6,
    "fusion": {"w_ofi":0.6,"w_cvd":0.4,"gate":0.0}
  },
  "windows": {
    "60s":{"AUC":0.60,"IC":0.03},
    "300s":{"AUC":0.62,"IC":0.04}
  },
  "stability": {"active_vs_quiet_delta_auc":0.07},
  "calibration": {"ece":0.08},
  "divergence": {"winrate_5m":0.57,"p5_tail":-0.35e-3}
}
```

## 🚨 阻断条件

### 质量门控
- 任一核心窗口 Fusion AUC < 0.58
- 全部窗口 ECE > 0.10
- 样本匹配率 < 80%（merge_asof）

### 例外条件（切片放量Plan B）
若 Active/London/Tokyo 任一切片 AUC ≥ 0.60 且 ECE ≤ 0.10，可仅在该切片放量。

## 🔧 故障排除

### 常见问题

#### 1. 数据加载失败
```bash
# 检查数据目录结构
ls -la data/ofi_cvd/date=2025-10-22/symbol=ETHUSDT/

# 检查文件权限
chmod -R 755 data/ofi_cvd/
```

#### 2. 内存不足
```bash
# 减少数据量
--date-from 2025-10-22 --date-to 2025-10-22

# 减少窗口数
--horizons 60,180
```

#### 3. 标签构造失败
```bash
# 检查价格数据质量
python -c "
import pandas as pd
df = pd.read_parquet('data/ofi_cvd/date=2025-10-22/symbol=ETHUSDT/kind=prices/part-0.parquet')
print(df.head())
print(df['price'].describe())
"
```

### 调试模式
```bash
# 启用详细日志
python -m analysis.ofi_cvd_signal_eval \
  --data-root data/ofi_cvd \
  --symbols ETHUSDT \
  --debug \
  --out artifacts/analysis/debug
```

## 📚 技术细节

### 时间对齐算法
```python
# 基于时间戳的asof对齐
def align_timestamps(df, horizon):
    df['ts_ms_fwd'] = df['ts_ms'] + horizon * 1000
    merged = pd.merge_asof(
        df, df[['ts_ms', 'price']],
        left_on='ts_ms', right_on='ts_ms_fwd',
        direction='forward', tolerance=5000
    )
    return merged
```

### 信号合并算法
```python
# 使用merge_asof进行近似匹配
def merge_signals_with_labels(signals_df, labels_df):
    merged = pd.merge_asof(
        signals_df.sort_values('ts_ms'),
        labels_df.sort_values('ts_ms'),
        on='ts_ms', direction='nearest', tolerance=1000
    )
    return merged
```

### 校准算法
```python
# Platt校准
def platt_calibration(scores, labels):
    from sklearn.calibration import CalibratedClassifierCV
    calibrator = CalibratedClassifierCV(method='sigmoid')
    calibrator.fit(scores.reshape(-1, 1), labels)
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
```

## 🔗 相关文档

- **任务卡**: [Task_1.3.2_创建OFI+CVD信号分析工具.md](../TASKS/Stage1_真实OFI+CVD核心/Task_1.3.2_创建OFI+CVD信号分析工具.md)
- **问题记录**: [POSTMORTEM_Task_1.3.2.md](../TASKS/Stage1_真实OFI+CVD核心/POSTMORTEM_Task_1.3.2.md)
- **配置文档**: [system.yaml](../config/system.yaml)
- **数据收集**: [Task_1.3.1_收集历史OFI+CVD数据.md](../TASKS/Stage1_真实OFI+CVD核心/Task_1.3.1_收集历史OFI+CVD数据.md)

## 📞 支持与反馈

- **项目**: V13 OFI+CVD+AI System
- **模块**: analysis
- **版本**: v2.0-prod
- **维护者**: V13 OFI+CVD+AI System Team

---

**最后更新**: 2025-10-22  
**文档版本**: v2.0  
**状态**: ✅ 稳定（已通过完整测试 + 灰度部署验证）
