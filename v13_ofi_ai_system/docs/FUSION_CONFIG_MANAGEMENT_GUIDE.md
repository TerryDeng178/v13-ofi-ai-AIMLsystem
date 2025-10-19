# 融合指标配置管理指南

## 📋 概述

本文档介绍融合指标配置的统一管理方案，包括配置迁移、环境覆盖、热更新等功能。

## 🎯 配置管理架构

### 配置层次结构
```
config/
├── system.yaml                    # 主配置文件（融合指标配置）
├── environments/                  # 环境特定配置
│   ├── development.yaml          # 开发环境覆盖
│   ├── testing.yaml              # 测试环境覆盖
│   └── production.yaml           # 生产环境覆盖
└── calibration/                   # 校准配置
    └── divergence_score_calibration.json
```

### 配置优先级
1. **环境变量** (最高优先级)
2. **环境特定配置** (`environments/*.yaml`)
3. **系统配置** (`system.yaml`)
4. **默认值** (最低优先级)

## 🔧 配置参数说明

### 权重配置
```yaml
fusion_metrics:
  weights:
    w_ofi: 0.6        # OFI权重
    w_cvd: 0.4        # CVD权重
    # 自动归一化：确保权重和为1.0
```

### 信号阈值
```yaml
fusion_metrics:
  thresholds:
    fuse_buy: 1.5           # 买入阈值
    fuse_strong_buy: 2.5    # 强买入阈值
    fuse_sell: -1.5         # 卖出阈值
    fuse_strong_sell: -2.5  # 强卖出阈值
```

### 一致性阈值
```yaml
fusion_metrics:
  consistency:
    min_consistency: 0.3        # 最小一致性要求
    strong_min_consistency: 0.7 # 强信号一致性要求
```

### 数据处理参数
```yaml
fusion_metrics:
  data_processing:
    z_clip: 5.0           # Z值裁剪阈值
    max_lag: 0.300        # 最大时间差(秒)
    warmup_samples: 30    # 暖启动样本数
```

### 去噪参数
```yaml
fusion_metrics:
  denoising:
    hysteresis_exit: 1.2  # 迟滞退出阈值
    cooldown_secs: 1.0    # 冷却时间(秒)
    min_duration: 2       # 最小持续次数
```

## 🚀 使用方法

### 1. 基本使用
```python
from src.utils.config_loader import ConfigLoader
from src.ofi_cvd_fusion import OFI_CVD_Fusion

# 创建配置加载器
config_loader = ConfigLoader()

# 从配置创建融合指标实例
fusion = OFI_CVD_Fusion(config_loader=config_loader)

# 使用融合指标
result = fusion.update(ts=time.time(), z_ofi=2.0, z_cvd=1.5, lag_sec=0.1)
```

### 2. 环境变量覆盖
```bash
# 使用环境变量覆盖配置
export V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_BUY=3.0
export V13__FUSION_METRICS__THRESHOLDS__FUSE_STRONG_SELL=-3.0

# 重新加载配置
config_loader.load(reload=True)
```

### 3. 配置热更新
```python
from src.fusion_config_hot_update import create_fusion_hot_updater

# 创建热更新器
hot_updater = create_fusion_hot_updater(
    config_loader=config_loader,
    fusion_instance=fusion
)

# 添加更新回调
def on_config_update(new_config):
    print(f"配置已更新: {new_config}")

hot_updater.add_update_callback(on_config_update)

# 开始监控
hot_updater.start_watching()
```

## 🔄 配置迁移

### 迁移现有配置
```bash
# 执行配置迁移
python scripts/migrate_fusion_config.py --action migrate

# 干跑模式（不实际修改文件）
python scripts/migrate_fusion_config.py --action migrate --dry-run

# 比较配置差异
python scripts/migrate_fusion_config.py --action compare

# 验证配置
python scripts/migrate_fusion_config.py --action validate
```

### 回滚配置
```bash
# 回滚到备份配置
python scripts/migrate_fusion_config.py --action rollback --backup-file config_backup/fusion_config_backup_1234567890.yaml
```

## 🌍 环境特定配置

### 开发环境配置
```yaml
# config/environments/development.yaml
fusion_metrics:
  thresholds:
    fuse_strong_buy: 2.0    # 开发环境使用较低阈值
    fuse_strong_sell: -2.0
```

### 生产环境配置
```yaml
# config/environments/production.yaml
fusion_metrics:
  thresholds:
    fuse_strong_buy: 3.0    # 生产环境使用更严格阈值
    fuse_strong_sell: -3.0
  consistency:
    min_consistency: 0.4    # 生产环境要求更高一致性
    strong_min_consistency: 0.8
```

## 📊 监控和告警

### Prometheus指标
```yaml
fusion_metrics:
  monitoring:
    prometheus:
      port: 8002
      path: "/metrics"
      scrape_interval: "5s"
    
    alerts:
      consistency_threshold: 0.2  # 一致性告警阈值
      lag_threshold: 0.5          # 滞后告警阈值
```

### Grafana仪表盘
```yaml
fusion_metrics:
  monitoring:
    grafana:
      dashboard_uid: "fusion-metrics"
      refresh_interval: "5s"
```

## 🔧 配置验证

### 自动验证
- 权重和必须为1.0
- 阈值逻辑正确（强买入 > 买入，强卖出 < 卖出）
- 一致性阈值在0-1范围内
- 强信号一致性 > 最小一致性

### 手动验证
```python
# 验证配置
python scripts/migrate_fusion_config.py --action validate
```

## 📝 最佳实践

### 1. 配置管理
- 使用环境变量进行临时覆盖
- 使用环境特定配置文件进行持久化覆盖
- 定期备份配置文件
- 使用版本控制管理配置变更

### 2. 热更新
- 在生产环境谨慎使用热更新
- 设置适当的更新间隔（避免频繁更新）
- 监控配置更新统计信息
- 准备回滚方案

### 3. 监控告警
- 设置合理的告警阈值
- 监控配置更新成功率
- 监控融合指标性能
- 定期检查配置一致性

## 🚨 故障排除

### 常见问题

#### 1. 配置加载失败
```python
# 检查配置文件是否存在
config_file = Path("config/system.yaml")
if not config_file.exists():
    print("配置文件不存在")

# 检查YAML语法
import yaml
with open(config_file, 'r') as f:
    yaml.safe_load(f)  # 会抛出异常如果语法错误
```

#### 2. 权重归一化问题
```python
# 检查权重和
total_weight = fusion.cfg.w_ofi + fusion.cfg.w_cvd
if abs(total_weight - 1.0) > 1e-6:
    print(f"权重和不为1.0: {total_weight}")
```

#### 3. 热更新不工作
```python
# 检查文件监控状态
stats = hot_updater.get_update_stats()
print(f"更新统计: {stats}")

# 检查监控路径
for path in hot_updater.watch_paths:
    if not Path(path).exists():
        print(f"监控路径不存在: {path}")
```

## 📚 相关文档

- [系统配置指南](../SYSTEM_CONFIG_GUIDE.md)
- [背离检测配置](../divergence_tuning.md)
- [生产部署指南](../PRODUCTION_DEPLOYMENT_GUIDE.md)

## 🔗 相关文件

- `config/system.yaml` - 主配置文件
- `config/environments/*.yaml` - 环境特定配置
- `src/fusion_config_hot_update.py` - 配置热更新模块
- `scripts/migrate_fusion_config.py` - 配置迁移工具
- `examples/fusion_config_example.py` - 使用示例

---

**文档版本**: v1.0  
**创建日期**: 2025-10-20  
**最后更新**: 2025-10-20
