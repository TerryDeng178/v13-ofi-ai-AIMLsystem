# Grafana手动配置指南

## 🚀 快速配置步骤

### 1. 添加Prometheus数据源

1. 在Grafana中，点击左侧菜单的 **Configuration** (齿轮图标)
2. 选择 **Data sources**
3. 点击 **Add data source**
4. 选择 **Prometheus**
5. 配置以下设置：
   - **Name**: `Prometheus`
   - **URL**: `http://localhost:9090`
   - **Access**: `Server (default)`
6. 点击 **Save & Test**

### 2. 导入仪表盘

#### 导入Strategy Mode Overview仪表盘

1. 点击左侧菜单的 **"+"** 图标
2. 选择 **Import**
3. 点击 **Upload JSON file**
4. 选择文件：`v13_ofi_ai_system/grafana/dashboards/strategy_mode_overview.json`
5. 点击 **Load**
6. 配置以下设置：
   - **Name**: `Strategy Mode Overview`
   - **Folder**: `General` (或创建新文件夹)
   - **Prometheus**: 选择刚才创建的Prometheus数据源
7. 点击 **Import**

#### 导入Strategy Performance仪表盘

1. 重复上述步骤1-3
2. 选择文件：`v13_ofi_ai_system/grafana/dashboards/strategy_performance.json`
3. 配置名称：`Strategy Performance`
4. 点击 **Import**

#### 导入Strategy Alerts仪表盘

1. 重复上述步骤1-3
2. 选择文件：`v13_ofi_ai_system/grafana/dashboards/strategy_alerts.json`
3. 配置名称：`Strategy Alerts`
4. 点击 **Import**

### 3. 验证配置

#### 检查数据源
- 确保Prometheus数据源状态为绿色 ✅
- 测试查询：`up`

#### 检查仪表盘
- 访问 **Dashboards** 查看导入的仪表盘
- 点击仪表盘名称进入查看
- 确认时间范围设置为 **Last 6 hours**
- 确认时区设置为 **Asia/Hong_Kong**

#### 检查指标数据
- 在仪表盘中查看是否有数据
- 如果没有数据，检查指标服务器是否运行：
  ```bash
  curl http://localhost:8000/metrics
  ```

## 🔧 故障排查

### 问题1：仪表盘显示"No Data"
**解决方案**：
1. 检查Prometheus数据源是否正常
2. 验证指标服务器是否运行：`http://localhost:8000/metrics`
3. 检查时间范围设置

### 问题2：Prometheus连接失败
**解决方案**：
1. 确保Prometheus服务正在运行：`http://localhost:9090`
2. 检查Docker容器状态：`docker ps`
3. 重启Docker服务：`docker-compose restart`

### 问题3：指标数据为空
**解决方案**：
1. 启动指标服务器：
   ```bash
   cd v13_ofi_ai_system/grafana
   python simple_metrics_server.py 8000
   ```
2. 验证指标端点：`http://localhost:8000/health`

## 📊 预期结果

配置完成后，您应该看到：

1. **3个仪表盘**：
   - Strategy Mode Overview
   - Strategy Performance  
   - Strategy Alerts

2. **数据源**：
   - Prometheus (状态：绿色)

3. **指标数据**：
   - 策略模式相关指标正常显示
   - 时间序列图表有数据

## 🎯 下一步

配置完成后，您可以：
1. 查看策略模式切换监控
2. 分析性能指标
3. 设置告警规则
4. 自定义仪表盘

---

**需要帮助？** 如果遇到问题，请检查：
- Docker服务是否运行
- 指标服务器是否启动
- 网络连接是否正常
