# Grafana手动配置指南

## 📋 前置条件检查

在开始配置之前，请确保以下服务正在运行：

### 1. 检查Docker服务状态
```bash
# 检查Docker是否运行
docker --version

# 检查Docker Compose状态
docker compose ps
```

### 2. 检查监控服务状态
```bash
# 检查Prometheus (端口9090)
curl http://localhost:9090/-/healthy

# 检查Grafana (端口3000)
curl http://localhost:3000/api/health

# 检查指标服务器 (端口8000)
curl http://localhost:8000/health
```

### 3. 启动完整监控栈（如果未运行）
```bash
# 进入项目目录
cd v13_ofi_ai_system

# 启动监控服务
docker compose up -d

# 启动指标服务器
cd grafana
python simple_metrics_server.py 8000
```

## 🚀 详细配置步骤

### 步骤1：访问Grafana

1. **打开浏览器访问Grafana**
   - 地址：http://localhost:3000
   - 用户名：`admin`
   - 密码：从`.env`文件中的`GF_ADMIN_PASSWORD`获取（默认：`admin`）

2. **首次登录设置**
   - 如果是首次访问，Grafana会要求设置新密码
   - 建议使用强密码并记录保存

### 步骤2：添加Prometheus数据源

#### 2.1 进入数据源配置
1. 在Grafana左侧菜单中，点击 **Configuration** (齿轮图标)
2. 选择 **Data sources**
3. 点击 **Add data source** 按钮

#### 2.2 选择Prometheus数据源
1. 在数据源列表中找到 **Prometheus**
2. 点击 **Select** 按钮

#### 2.3 配置Prometheus连接
在配置页面填写以下信息：

**基本设置：**
- **Name**: `Prometheus` (建议保持默认)
- **URL**: `http://localhost:9090` (确保端口正确)
- **Access**: `Server (default)` (推荐选择)

**高级设置（可选）：**
- **HTTP Method**: `GET` (默认)
- **Timeout**: `30s` (默认)
- **Basic Auth**: 关闭 (Prometheus通常不需要认证)

#### 2.4 测试连接
1. 点击 **Save & Test** 按钮
2. 等待测试结果：
   - ✅ **绿色提示**：数据源配置成功
   - ❌ **红色提示**：检查Prometheus服务是否运行

#### 2.5 验证数据源
1. 点击 **Explore** 按钮进入查询界面
2. 输入测试查询：`up`
3. 点击 **Run query** 查看结果
4. 如果看到数据，说明连接成功

### 步骤3：导入仪表盘

#### 3.1 导入Strategy Mode Overview仪表盘

**准备文件：**
- 文件路径：`v13_ofi_ai_system/grafana/dashboards/strategy_mode_overview.json`
- 确保文件存在且可访问

**导入步骤：**
1. 在Grafana左侧菜单中，点击 **"+"** 图标
2. 选择 **Import**
3. 点击 **Upload JSON file** 按钮
4. 浏览并选择文件：`strategy_mode_overview.json`
5. 点击 **Load** 按钮

**配置设置：**
- **Name**: `Strategy Mode Overview` (建议保持默认)
- **Folder**: `General` (或创建新文件夹如 `V13 Strategy`)
- **Prometheus**: 选择刚才创建的Prometheus数据源
- **UID**: 自动生成（建议保持默认）

**完成导入：**
1. 点击 **Import** 按钮
2. 等待导入完成
3. 点击 **View Dashboard** 查看仪表盘

#### 3.2 导入Strategy Performance仪表盘

**重复导入流程：**
1. 再次点击 **"+"** → **Import**
2. 选择文件：`strategy_performance.json`
3. 配置名称：`Strategy Performance`
4. 选择相同的Prometheus数据源
5. 点击 **Import**

#### 3.3 导入Strategy Alerts仪表盘

**重复导入流程：**
1. 再次点击 **"+"** → **Import**
2. 选择文件：`strategy_alerts.json`
3. 配置名称：`Strategy Alerts`
4. 选择相同的Prometheus数据源
5. 点击 **Import**

### 步骤4：配置仪表盘设置

#### 4.1 设置时区
1. 进入任意仪表盘
2. 点击右上角的 **时间选择器**
3. 点击 **设置图标** (齿轮)
4. 选择时区：**Asia/Hong_Kong**
5. 点击 **Apply**

#### 4.2 设置默认时间范围
1. 在时间选择器中
2. 选择 **Last 6 hours** (推荐)
3. 点击 **Apply**

#### 4.3 配置变量（如果需要）
1. 进入仪表盘设置
2. 点击 **Variables** 标签
3. 检查 `$env` 和 `$symbol` 变量是否正确配置
4. 保存设置

### 步骤5：验证配置

#### 5.1 检查数据源状态
1. 返回 **Configuration** → **Data sources**
2. 确认Prometheus数据源状态为绿色 ✅
3. 点击 **Test** 按钮验证连接

#### 5.2 检查仪表盘列表
1. 点击左侧菜单 **Dashboards**
2. 确认看到3个仪表盘：
   - Strategy Mode Overview
   - Strategy Performance
   - Strategy Alerts

#### 5.3 验证仪表盘功能
**进入Strategy Mode Overview仪表盘：**
1. 点击仪表盘名称进入
2. 检查右上角时间范围：**Last 6 hours**
3. 检查时区设置：**Asia/Hong_Kong**
4. 查看面板是否有数据：
   - Current Mode 应显示 0 或 1
   - Last Switch Ago 应显示时间
   - Switches Today 应显示数字

**进入Strategy Performance仪表盘：**
1. 检查性能指标是否显示
2. 查看参数更新耗时图表
3. 确认直方图有数据

**进入Strategy Alerts仪表盘：**
1. 检查告警状态面板
2. 查看心跳监控状态
3. 确认告警趋势图表

#### 5.4 检查指标数据
如果仪表盘显示"No Data"：

1. **检查指标服务器状态：**
   ```bash
   # 检查指标端点
   curl http://localhost:8000/metrics
   
   # 检查健康状态
   curl http://localhost:8000/health
   ```

2. **检查Prometheus抓取状态：**
   - 访问 http://localhost:9090/targets
   - 确认 `strategy-mode-manager` 状态为 UP
   - 检查是否有抓取错误

3. **重启指标服务器（如果需要）：**
   ```bash
   # 停止现有服务器
   # 按 Ctrl+C 停止当前进程
   
   # 重新启动
   cd v13_ofi_ai_system/grafana
   python simple_metrics_server.py 8000
   ```

## 🔧 故障排查

### 问题1：仪表盘显示"No Data"

**症状：** 仪表盘面板显示"No Data"或空白

**诊断步骤：**
1. **检查时间范围：**
   - 确认时间范围设置为最近6小时
   - 尝试设置为"Last 1 hour"

2. **检查数据源：**
   - 进入 Configuration → Data sources
   - 点击Prometheus数据源的"Test"按钮
   - 确认状态为绿色

3. **检查指标服务器：**
   ```bash
   # 检查指标端点
   curl http://localhost:8000/metrics | head -20
   
   # 检查健康状态
   curl http://localhost:8000/health
   ```

**解决方案：**
- 如果指标服务器未运行，启动它：
  ```bash
  cd v13_ofi_ai_system/grafana
  python simple_metrics_server.py 8000
  ```

### 问题2：Prometheus连接失败

**症状：** 数据源测试失败，显示连接错误

**诊断步骤：**
1. **检查Prometheus服务：**
   ```bash
   # 检查容器状态
   docker ps | grep prometheus
   
   # 检查服务健康
   curl http://localhost:9090/-/healthy
   ```

2. **检查端口占用：**
   ```bash
   # 检查9090端口
   netstat -an | grep 9090
   ```

**解决方案：**
1. **重启Prometheus服务：**
   ```bash
   docker compose restart prometheus
   ```

2. **检查Docker Compose配置：**
   ```bash
   # 查看服务日志
   docker compose logs prometheus
   ```

### 问题3：指标数据为空

**症状：** 指标服务器运行但Prometheus抓取不到数据

**诊断步骤：**
1. **检查Prometheus目标状态：**
   - 访问 http://localhost:9090/targets
   - 查看 `strategy-mode-manager` 状态
   - 检查是否有抓取错误

2. **检查网络连接：**
   ```bash
   # 从Prometheus容器内测试
   docker exec v13-prometheus wget -qO- http://host.docker.internal:8000/metrics
   ```

**解决方案：**
1. **重启监控栈：**
   ```bash
   docker compose down
   docker compose up -d
   ```

2. **检查防火墙设置：**
   - 确保8000端口可访问
   - 检查Windows防火墙设置

### 问题4：Grafana无法访问

**症状：** 浏览器无法访问 http://localhost:3000

**诊断步骤：**
1. **检查Grafana容器：**
   ```bash
   docker ps | grep grafana
   docker compose logs grafana
   ```

2. **检查端口占用：**
   ```bash
   netstat -an | grep 3000
   ```

**解决方案：**
1. **重启Grafana：**
   ```bash
   docker compose restart grafana
   ```

2. **检查Docker Desktop：**
   - 确保Docker Desktop正在运行
   - 重启Docker Desktop（如果需要）

### 问题5：仪表盘导入失败

**症状：** 导入JSON文件时出现错误

**诊断步骤：**
1. **检查文件路径：**
   - 确认JSON文件存在
   - 检查文件权限

2. **检查文件格式：**
   - 验证JSON文件格式正确
   - 检查文件大小是否合理

**解决方案：**
1. **重新下载文件：**
   - 从项目仓库重新获取JSON文件
   - 确保文件完整

2. **手动复制内容：**
   - 打开JSON文件
   - 复制内容到Grafana的"Import via panel json"选项

### 问题6：时区显示错误

**症状：** 时间显示不正确

**解决方案：**
1. **设置Grafana时区：**
   - 进入任意仪表盘
   - 点击时间选择器 → 设置
   - 选择 Asia/Hong_Kong

2. **设置浏览器时区：**
   - 确保系统时区设置正确
   - 重启浏览器

## 📊 预期结果

配置完成后，您应该看到：

### ✅ 成功指标
1. **3个仪表盘正常显示**：
   - Strategy Mode Overview - 显示模式切换监控
   - Strategy Performance - 显示性能指标
   - Strategy Alerts - 显示告警状态

2. **数据源状态正常**：
   - Prometheus数据源状态：绿色 ✅
   - 连接测试通过

3. **指标数据正常显示**：
   - Current Mode: 显示 0 或 1
   - Last Switch Ago: 显示时间间隔
   - Switches Today: 显示切换次数
   - 性能图表有数据

4. **时区和时间设置正确**：
   - 时区：Asia/Hong_Kong
   - 时间范围：Last 6 hours

### 📈 数据验证
**在Strategy Mode Overview仪表盘中：**
- Current Mode面板应显示当前模式状态
- Last Switch Ago应显示距离上次切换的时间
- Switches Today应显示今日切换次数
- 时间序列图表应显示模式切换趋势

**在Strategy Performance仪表盘中：**
- P50/P95/P99性能指标应显示数值
- 参数更新耗时图表应有数据
- 直方图应显示性能分布

**在Strategy Alerts仪表盘中：**
- 心跳状态应显示绿色（正常）
- 告警趋势图表应有数据
- 告警阈值面板应显示当前值

## 🎯 下一步操作

配置完成后，您可以：

### 1. 监控策略模式
- 观察模式切换频率和原因
- 分析市场触发因子
- 监控价差和波动率

### 2. 性能分析
- 查看参数更新性能
- 分析失败率和趋势
- 优化系统性能

### 3. 告警管理
- 设置告警阈值
- 配置通知渠道
- 监控系统健康状态

### 4. 自定义配置
- 修改仪表盘布局
- 添加自定义面板
- 调整告警规则

## 🆘 获取帮助

如果遇到问题，请按以下顺序检查：

### 1. 基础检查
- ✅ Docker Desktop是否运行
- ✅ 所有容器是否启动
- ✅ 指标服务器是否运行
- ✅ 网络连接是否正常

### 2. 服务检查
```bash
# 检查所有服务状态
docker compose ps

# 检查服务日志
docker compose logs

# 验证指标数据
curl http://localhost:8000/metrics | head -10
```

### 3. 联系支持
- 查看项目文档：`docs/GRAFANA_DASHBOARD_GUIDE.md`
- 检查任务状态：`TASKS/README.md`
- 查看AI协作指南：`AI_COLLABORATION_GUIDE.md`

---

**配置完成！** 🎉 您现在可以开始使用完整的监控系统来监控策略模式切换和性能指标了。
