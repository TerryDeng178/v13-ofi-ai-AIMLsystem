#!/bin/bash
# 背离检测模块快速部署脚本
# 版本: v13.0
# 用途: 一键部署到生产环境

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查函数
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 $1 未找到，请先安装"
        exit 1
    fi
}

check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件 $1 不存在"
        exit 1
    fi
}

check_directory() {
    if [ ! -d "$1" ]; then
        log_error "目录 $1 不存在"
        exit 1
    fi
}

# 主函数
main() {
    log_info "开始背离检测模块快速部署..."
    
    # 1. 环境检查
    log_info "步骤1: 环境检查"
    check_command python
    check_command pip
    check_file "requirements.txt"
    check_file "config/environments/production.yaml"
    check_directory "runs/real_test"
    check_directory "runs/metrics_test"
    log_success "环境检查通过"
    
    # 2. 依赖安装
    log_info "步骤2: 安装依赖"
    pip install -r requirements.txt
    log_success "依赖安装完成"
    
    # 3. 预检查
    log_info "步骤3: 预检查"
    python scripts/deploy_production.py --action check
    log_success "预检查通过"
    
    # 4. 指标对齐验证
    log_info "步骤4: 指标对齐验证"
    python scripts/metrics_alignment.py --out runs/metrics_test
    log_success "指标对齐验证通过"
    
    # 5. 配置热更新测试
    log_info "步骤5: 配置热更新测试"
    python scripts/config_hot_update.py --test
    log_success "配置热更新测试通过"
    
    # 6. 部署确认
    log_warning "即将开始生产环境部署，请确认："
    echo "  - 所有检查已通过"
    echo "  - 配置文件已准备"
    echo "  - 监控系统已就绪"
    echo "  - 回滚方案已准备"
    echo ""
    read -p "是否继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "部署已取消"
        exit 0
    fi
    
    # 7. 执行部署
    log_info "步骤6: 执行生产环境部署"
    python scripts/deploy_production.py --action deploy
    log_success "部署完成"
    
    # 8. 部署后验证
    log_info "步骤7: 部署后验证"
    sleep 10  # 等待服务启动
    
    # 检查指标端点
    if curl -s http://localhost:8003/metrics > /dev/null; then
        log_success "指标端点正常"
    else
        log_warning "指标端点不可用，请检查服务状态"
    fi
    
    # 9. 完成
    log_success "背离检测模块部署完成！"
    echo ""
    echo "下一步操作："
    echo "  1. 监控系统状态30-60分钟"
    echo "  2. 检查Grafana仪表盘: http://localhost:3000/d/divergence-monitoring-prod"
    echo "  3. 观察关键指标是否达标"
    echo "  4. 如有问题，使用回滚脚本: python scripts/rollback_production.py --type emergency"
    echo ""
    echo "监控命令："
    echo "  - 查看指标: curl http://localhost:8003/metrics"
    echo "  - 检查配置: python scripts/config_hot_update.py --test"
    echo "  - 列出备份: python scripts/rollback_production.py --list-backups"
    echo ""
}

# 错误处理
trap 'log_error "部署过程中发生错误，请检查日志"; exit 1' ERR

# 执行主函数
main "$@"
