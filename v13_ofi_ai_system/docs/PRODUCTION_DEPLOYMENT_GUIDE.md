# 背离检测模块生产环境部署指南

## 🎯 部署概述

**模块**: 背离检测/评分校准  
**部署策略**: 灰度上线（Canary Deployment）  
**上线范围**: 信号只读 + 渐进放量  
**目标**: 安全、渐进地推进到生产环境  

## ✅ 为什么可以上线

### 验收标准全部通过
- ✅ **准确率达标**: acc@10 ≥ 55%, acc@20 ≥ 55%
- ✅ **单调性显著**: Spearman ρ > 0 且 p < 0.05
- ✅ **指标闭合**: Prometheus/Grafana 三类图卡闭合 ±10%
- ✅ **热更新有效**: 配置热更新机制正常工作

### 最优参数清晰
- **全局最佳**: swing_L=13, z_hi=2.0, z_mid=0.8
- **分桶最佳**: 已导出 best_by_bucket.yaml
- **校准映射**: divergence_score_calibration.json 完整

### 性能余量充足
- **计算延迟**: P95 ≈ 3.06ms（目标 < 10ms）
- **事件延迟**: 目标 P95 < 2.5ms
- **管线延迟**: 目标 P95 < 100ms

## 📦 要发布的工件

### 参数与校准
```
runs/real_test/best_global.yaml              # 默认参数
runs/real_test/best_by_bucket.yaml           # 分桶覆盖
config/calibration/divergence_score_calibration.json  # 校准映射
```

### 系统配置
```
config/environments/production.yaml          # 生产环境配置
config/system.yaml                           # 主配置文件
```

### 观测与告警
```
runs/metrics_test/prometheus_divergence.yml  # Prometheus配置
runs/metrics_test/divergence_metrics_exporter.py  # 指标导出器
runs/metrics_test/dashboards/divergence_overview.json  # Grafana仪表盘
runs/metrics_test/alerting_rules/divergence_alerts.yaml  # 告警规则
runs/metrics_test/alerting_rules/production_slo_alerts.yaml  # SLO告警
```

## 🚀 上线步骤

### 阶段1: 预检 (Pre-flight)
```bash
# 1. 检查所有配置文件
python scripts/deploy_production.py --action check

# 2. 验证指标对齐
python scripts/metrics_alignment.py --out runs/metrics_test

# 3. 测试配置热更新
python scripts/config_hot_update.py --test

# 4. 检查回滚脚本
python scripts/rollback_production.py --list-backups
```

**预期结果**: 所有检查通过，无错误

### 阶段2: 灰度部署 (Canary)
```bash
# 1. 执行灰度部署
python scripts/deploy_production.py --action deploy

# 2. 验证部署状态
curl http://localhost:8003/metrics

# 3. 检查Grafana仪表盘
# 访问: http://localhost:3000/d/divergence-monitoring-prod
```

**预期结果**: 
- 指标导出器运行正常
- Prometheus开始收集指标
- Grafana仪表盘显示数据
- 背离检测模块运行（只读模式）

### 阶段3: 观察期 (Observation)
**持续时间**: 30-60分钟  
**监控重点**:
- 事件延迟 P95 < 2.5ms
- 管线延迟 P95 < 100ms
- 背离模块计算 P95 < 10ms
- 指标闭合误差 < 10%
- 错误率 < 1%
- 可用性 > 99.9%

### 阶段4: 放量 (Scale-up)
**条件**: 观察期指标正常  
**操作**:
- 逐步增加目标品种（BTCUSDT → ETHUSDT → 其他）
- 监控系统负载和性能
- 根据实际情况调整参数

## 📊 SLO监控指标

### 延迟指标
| 指标 | 阈值 | 告警阈值 | 回滚阈值 |
|------|------|----------|----------|
| 事件延迟P95 | < 2.5ms | > 2ms | > 5ms |
| 管线延迟P95 | < 100ms | > 50ms | > 200ms |
| 背离计算P95 | < 10ms | > 5ms | > 20ms |

### 准确性指标
| 指标 | 阈值 | 告警阈值 | 回滚阈值 |
|------|------|----------|----------|
| 指标闭合误差 | < 10% | > 5% | > 15% |
| 错误率 | < 1% | > 0.5% | > 2% |
| 可用性 | > 99.9% | < 99.5% | < 99% |

### 业务指标
| 指标 | 阈值 | 告警阈值 | 回滚阈值 |
|------|------|----------|----------|
| 事件数/小时 | 10-100 | < 5 或 > 200 | < 1 或 > 500 |
| 分数中位数 | > 50 | < 40 | < 30 |
| 命中率 | > 55% | < 50% | < 45% |

## 🚨 告警规则

### 关键告警 (Critical) - 立即回滚
- 背离检测延迟 > 20ms 持续10分钟
- 系统可用性 < 99% 持续5分钟
- 错误率 > 2% 持续5分钟
- 指标闭合误差 > 15% 持续10分钟

### 警告告警 (Warning) - 密切监控
- 背离检测延迟 > 10ms 持续5分钟
- 系统负载过高 (CPU > 80% 持续10分钟)
- 内存使用过高 (> 80% 持续10分钟)
- 队列积压过多 (> 1000 持续5分钟)

## 🔄 回滚计划

### 自动回滚触发条件
- 背离检测延迟 > 20ms 持续10分钟
- 系统可用性 < 99% 持续5分钟
- 错误率 > 2% 持续5分钟
- 关键指标异常持续15分钟

### 回滚操作
```bash
# 1. 紧急回滚（立即停止）
python scripts/rollback_production.py --type emergency --reason "slo_violation"

# 2. 完整回滚（恢复到指定备份）
python scripts/rollback_production.py --type full --backup-timestamp 1734672000 --reason "performance_issue"

# 3. 列出可用备份
python scripts/rollback_production.py --list-backups
```

### 回滚验证
- 系统恢复到稳定状态
- 关键指标恢复正常
- 无错误日志
- 监控仪表盘正常

## 📋 部署检查清单

### 预部署检查
- [ ] 所有配置文件存在且格式正确
- [ ] 依赖包已安装
- [ ] 验收标准验证通过
- [ ] 监控系统准备就绪

### 部署过程
- [ ] 预检通过
- [ ] 灰度部署成功
- [ ] 服务健康检查通过
- [ ] 监控指标正常

### 部署后验证
- [ ] 功能验证通过
- [ ] 性能验证通过
- [ ] 监控验证通过
- [ ] 告警规则生效

## 🎯 成功标准

### 技术指标
- [ ] 所有SLO指标达标
- [ ] 无关键告警
- [ ] 系统稳定运行30分钟以上
- [ ] 性能指标符合预期

### 业务指标
- [ ] 背离检测准确率 ≥ 55%
- [ ] 信号生成稳定
- [ ] 监控数据完整
- [ ] 用户体验良好

## 📞 应急联系

### 技术负责人
- **主负责人**: V13 Team Lead
- **备用负责人**: V13 Tech Lead
- **联系方式**: [待填写]

### 运维团队
- **监控团队**: [待填写]
- **基础设施团队**: [待填写]
- **联系方式**: [待填写]

## 🔧 故障排除

### 常见问题
1. **指标导出器无法启动**
   - 检查端口8003是否被占用
   - 验证Prometheus配置格式
   - 查看错误日志

2. **Grafana仪表盘无数据**
   - 检查Prometheus数据源配置
   - 验证指标查询语句
   - 确认时间范围设置

3. **背离检测模块异常**
   - 检查配置文件格式
   - 验证参数文件路径
   - 查看模块日志

4. **热更新不工作**
   - 检查文件监控权限
   - 验证配置文件格式
   - 查看热更新日志

### 调试命令
```bash
# 检查服务状态
curl http://localhost:8003/metrics

# 查看配置
python scripts/config_hot_update.py --test

# 检查回滚状态
python scripts/rollback_production.py --list-backups

# 验证指标对齐
python scripts/metrics_alignment.py --out runs/metrics_test
```

## 📝 部署记录模板

### 部署信息
- **部署时间**: [填写]
- **部署版本**: v13.0
- **部署人员**: [填写]
- **部署环境**: 生产环境

### 部署结果
- **部署状态**: [成功/失败]
- **关键指标**: [填写]
- **问题记录**: [填写]
- **后续行动**: [填写]

---

## 🎉 总结

本次部署采用灰度上线策略，确保安全、渐进地推进背离检测模块到生产环境。所有验收标准已通过，配置和监控体系完整，回滚机制健全。建议严格按照检查清单执行，密切监控关键指标，确保部署成功。

**重要提醒**: 本次上线仅限配置与监控层，不涉及交易执行流。后续按V13路线完成阶段1.2/1.3后再评估开单。

---
*文档版本: v13.0*  
*创建日期: 2025-10-20*  
*最后更新: 2025-10-20*
