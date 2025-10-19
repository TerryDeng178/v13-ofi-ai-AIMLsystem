# 仪表盘启动指南

## 🚀 快速启动

### 方法1：使用简化启动脚本（推荐）

```bash
# 在 v13_ofi_ai_system 目录下运行
start_dashboard_simple.bat
```

### 方法2：手动启动

1. **启动指标服务器**：
   ```bash
   cd v13_ofi_ai_system/grafana
   python simple_metrics_server.py 8000
   ```

2. **启动监控服务**：
   ```bash
   cd v13_ofi_ai_system
   docker-compose up -d
   ```

## 📊 访问地址

启动成功后，可以通过以下地址访问：

- **Grafana仪表盘**: http://localhost:3000
  - 用户名: `admin`
  - 密码: `admin`

- **Prometheus**: http://localhost:9090

- **指标端点**: http://localhost:8000/metrics

## 📋 已导入的仪表盘

1. **Strategy Mode Overview** - 策略模式概览
   - 当前模式状态
   - 切换历史和原因
   - 市场触发因子
   - 模式时长趋势

2. **Strategy Performance** - 策略性能监控
   - 参数更新耗时（P50/P95/P99）
   - 更新失败统计
   - 性能趋势分析

3. **Strategy Alerts** - 策略告警监控
   - 实时告警状态
   - 告警趋势和历史
   - 心跳监控

## 🔧 故障排查

### 问题1：指标服务器启动失败
**解决方案**：
- 检查端口8000是否被占用
- 使用 `python grafana/simple_metrics_server.py 8000` 手动启动
- 检查Python环境和依赖

### 问题2：Docker服务启动失败
**解决方案**：
- 确保Docker Desktop已启动
- 检查端口9090和3000是否被占用
- 运行 `docker-compose logs` 查看错误日志

### 问题3：仪表盘显示"No Data"
**解决方案**：
- 检查Prometheus是否正常采集指标
- 验证数据源配置
- 确认时间范围设置

## 📈 验证步骤

1. **检查指标服务器**：
   ```bash
   curl http://localhost:8000/health
   # 应该返回: {"status": "healthy"}
   ```

2. **检查Prometheus**：
   ```bash
   curl http://localhost:9090/api/v1/query?query=up
   # 应该返回JSON格式的查询结果
   ```

3. **检查Grafana**：
   - 访问 http://localhost:3000
   - 使用 admin/admin 登录
   - 查看仪表盘列表

## 🎯 使用建议

1. **首次使用**：
   - 建议先运行简化启动脚本
   - 等待所有服务启动完成
   - 验证所有访问地址正常

2. **日常使用**：
   - 可以单独启动指标服务器进行测试
   - 使用Docker Compose管理监控服务
   - 定期检查服务状态

3. **开发调试**：
   - 修改指标生成器模拟不同场景
   - 调整仪表盘配置
   - 添加新的告警规则

---

**文档版本**: V1.0  
**最后更新**: 2025-10-19  
**维护者**: V13 Team
