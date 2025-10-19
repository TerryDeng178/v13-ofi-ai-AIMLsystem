# Grafana 仪表盘导入指南

## 🎯 当前问题
您看到了"创建新仪表盘"页面，说明仪表盘没有正确导入。

## 🔧 解决方案

### 方法1：通过Grafana界面导入（推荐）

1. **在Grafana中，点击 "Import dashboard" 按钮**
2. **选择 "Upload JSON file"**
3. **依次导入以下3个文件**：
   - `grafana/dashboards/strategy_mode_overview.json`
   - `grafana/dashboards/strategy_performance.json`
   - `grafana/dashboards/strategy_alerts.json`

### 方法2：通过文件系统导入

1. **将仪表盘文件复制到Grafana的dashboards目录**
2. **重启Grafana服务**

## 📍 当前状态
- 您在"创建新仪表盘"页面
- 需要导入已准备好的仪表盘文件
- 导入后才能在仪表盘列表中看到它们

## 🚀 立即操作

**请点击 "Import dashboard" 按钮，然后选择上传JSON文件！**

导入完成后，您就能在仪表盘列表中看到3个仪表盘，然后点击进入查看数据。
