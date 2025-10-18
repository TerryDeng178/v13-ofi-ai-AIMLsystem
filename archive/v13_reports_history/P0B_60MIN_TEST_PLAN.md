# P0-B 60分钟正式验收测试计划

## 📋 测试概述
- **测试级别**: Silver（60分钟正式验收）
- **主测试Symbol**: ETHUSDT
- **压力测试Symbol**: BTCUSDT（10-15分钟）
- **测试目标**: 验证P0-B"四绿灯"标准

---

## 🎯 四绿灯验收标准

### 1️⃣ ID健康（核心）
- ✅ `agg_dup_count` = 0
- ✅ `agg_backward_count` ≤ 0.5%（理想为0）
- ✅ `late_event_dropped` ≈ 0

### 2️⃣ 到达节奏
- ✅ `p99_interarrival` ≤ 5s
- ✅ `gaps_over_10s` = 0（≥10秒缺口数）

### 3️⃣ 一致性（守恒）
- ✅ **逐笔守恒错误** = 0（最硬指标）
- ✅ **首尾守恒误差** ≈ 0（容差<1e-6）

### 4️⃣ 水位线健康
- ✅ `buffer_size_p95` < 100
- ✅ `buffer_size_max` < 200（信息项）
- ✅ `late_event_dropped` ≈ 0（再次强调）

---

## 📅 测试执行计划

### Phase 1: ETHUSDT 60分钟正式验收
```bash
Symbol: ETHUSDT
Duration: 3600s (60分钟)
Output: v13_ofi_ai_system/data/cvd_p0b_60min_ethusdt
```

**预期数据量**: ~3000-5000条记录（基于快测291条/5分钟）

**监控重点**:
1. 前10分钟：观察启动稳定性
2. 30分钟节点：中期数据质量检查
3. 最后10分钟：观察长时间运行稳定性

### Phase 2: BTCUSDT 10-15分钟压力测试
```bash
Symbol: BTCUSDT
Duration: 600-900s (10-15分钟)
Output: v13_ofi_ai_system/data/cvd_p0b_stress_btcusdt
```

**压力特性**: BTCUSDT交易频率约为ETHUSDT的3-5倍

**监控重点**:
1. `buffer_size_p95` / `buffer_size_max` - 是否仍在健康范围
2. `agg_dup_count` / `agg_backward_count` - ID健康是否为绿
3. `queue_dropped_rate` - 是否出现背压丢弃
4. CPU/内存 - 资源消耗是否可控

---

## ✅ 验收清单

### ETHUSDT 60分钟测试
- [ ] 测试运行完成（≥3600s）
- [ ] 数据采集完成（≥3000条）
- [ ] 四绿灯验证通过
- [ ] 分析报告生成

### BTCUSDT 压力测试
- [ ] 测试运行完成（10-15分钟）
- [ ] 高负载场景验证
- [ ] ID健康保持绿灯
- [ ] buffer_size稳定性确认

### 文档与提交
- [ ] 更新P0B_60MIN_TEST_REPORT.md
- [ ] 更新P0B_PHASE_SUMMARY.md
- [ ] Git提交（三件套）
- [ ] 打标签v13_p0b_pass

---

## 📊 预期结果

### 四绿灯全部通过 ✅
- **ID健康**: agg_dup=0, agg_backward=0, late_dropped≈0
- **到达节奏**: p99_interarrival≤5s, gaps_over_10s=0
- **一致性**: 逐笔守恒0错误, 首尾守恒误差≈0
- **水位线健康**: buffer_size_p95<100, late_dropped≈0

### 通过标准
- ETHUSDT 60分钟：四绿灯全部通过
- BTCUSDT 10-15分钟：ID健康绿灯 + buffer稳定

---

## 🚨 异常预案

### 如果出现问题
1. **ID健康未通过** → P0-B修复失败，需重新分析
2. **守恒错误>0** → P0-B核心目标未达成，需回退
3. **buffer_size_p95>100** → 水位线参数需调优（P1任务）
4. **late_event_dropped>5** → 背压处理需优化（P1任务）

### 问题记录
*（如有问题，在此记录）*

---

**计划创建时间**: 2025-10-18 04:55  
**预计完成时间**: 2025-10-18 06:10  
**执行状态**: 🟢 准备启动

