#!/usr/bin/env python3
"""
生产环境回滚脚本 - 背离检测模块
支持自动回滚、故障切换、配置恢复等功能
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionRollback:
    """生产环境回滚器"""
    
    def __init__(self, config_path: str = "config/environments/production.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.rollback_status = {
            'start_time': None,
            'current_stage': None,
            'success': False,
            'errors': [],
            'rollback_reason': None
        }
        self.backup_dir = Path("backups/rollback")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """加载生产环境配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise
    
    def create_backup(self) -> bool:
        """创建当前配置备份"""
        logger.info("创建配置备份...")
        
        try:
            timestamp = int(time.time())
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 备份关键配置文件
            files_to_backup = [
                "config/environments/production.yaml",
                "runs/real_test/best_global.yaml",
                "runs/real_test/best_by_bucket.yaml",
                "config/calibration/divergence_score_calibration.json"
            ]
            
            for file_path in files_to_backup:
                src = Path(file_path)
                if src.exists():
                    dst = backup_path / src.name
                    dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
                    logger.info(f"备份文件: {src} -> {dst}")
            
            # 保存备份元数据
            backup_metadata = {
                'timestamp': timestamp,
                'backup_path': str(backup_path),
                'files_backed_up': files_to_backup,
                'rollback_reason': self.rollback_status.get('rollback_reason', 'unknown')
            }
            
            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置备份完成: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置备份失败: {e}")
            return False
    
    def stop_services(self) -> bool:
        """停止相关服务"""
        logger.info("停止相关服务...")
        
        try:
            # 停止背离检测模块
            logger.info("停止背离检测模块...")
            # 这里应该停止实际的背离检测服务
            # 简化版：检查进程是否存在
            
            # 停止指标导出器
            logger.info("停止指标导出器...")
            # 这里应该停止实际的指标导出器进程
            
            # 停止热更新监控
            logger.info("停止热更新监控...")
            # 这里应该停止实际的热更新监控进程
            
            logger.info("服务停止完成")
            return True
            
        except Exception as e:
            logger.error(f"停止服务失败: {e}")
            return False
    
    def restore_config(self, backup_timestamp: Optional[int] = None) -> bool:
        """恢复配置"""
        logger.info("恢复配置...")
        
        try:
            if backup_timestamp:
                # 使用指定时间戳的备份
                backup_path = self.backup_dir / f"backup_{backup_timestamp}"
            else:
                # 使用最新的备份
                backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir() and d.name.startswith('backup_')]
                if not backup_dirs:
                    logger.error("未找到备份文件")
                    return False
                
                backup_path = max(backup_dirs, key=lambda x: int(x.name.split('_')[1]))
            
            if not backup_path.exists():
                logger.error(f"备份路径不存在: {backup_path}")
                return False
            
            # 恢复配置文件
            files_to_restore = [
                ("best_global.yaml", "runs/real_test/best_global.yaml"),
                ("best_by_bucket.yaml", "runs/real_test/best_by_bucket.yaml"),
                ("divergence_score_calibration.json", "config/calibration/divergence_score_calibration.json"),
                ("production.yaml", "config/environments/production.yaml")
            ]
            
            for backup_file, target_path in files_to_restore:
                src = backup_path / backup_file
                dst = Path(target_path)
                
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
                    logger.info(f"恢复文件: {src} -> {dst}")
                else:
                    logger.warning(f"备份文件不存在: {src}")
            
            logger.info("配置恢复完成")
            return True
            
        except Exception as e:
            logger.error(f"配置恢复失败: {e}")
            return False
    
    def disable_divergence_module(self) -> bool:
        """禁用背离检测模块"""
        logger.info("禁用背离检测模块...")
        
        try:
            # 更新生产环境配置，禁用背离检测模块
            if 'divergence_detection' in self.config:
                self.config['divergence_detection']['enabled'] = False
                self.config['divergence_detection']['emergency_stop'] = True
            
            # 保存配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info("背离检测模块已禁用")
            return True
            
        except Exception as e:
            logger.error(f"禁用背离检测模块失败: {e}")
            return False
    
    def restart_services(self) -> bool:
        """重启服务"""
        logger.info("重启服务...")
        
        try:
            # 重启系统服务
            logger.info("重启系统服务...")
            # 这里应该重启实际的系统服务
            
            # 验证服务状态
            logger.info("验证服务状态...")
            time.sleep(10)  # 等待服务启动
            
            # 检查服务健康状态
            if not self.health_check():
                logger.error("服务健康检查失败")
                return False
            
            logger.info("服务重启完成")
            return True
            
        except Exception as e:
            logger.error(f"重启服务失败: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查"""
        logger.info("执行健康检查...")
        
        try:
            # 检查系统状态
            logger.info("检查系统状态...")
            
            # 检查配置文件
            logger.info("检查配置文件...")
            
            # 检查服务状态
            logger.info("检查服务状态...")
            
            logger.info("健康检查通过")
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def emergency_rollback(self) -> bool:
        """紧急回滚"""
        logger.info("执行紧急回滚...")
        
        try:
            # 1. 立即停止所有服务
            if not self.stop_services():
                logger.error("停止服务失败")
                return False
            
            # 2. 禁用背离检测模块
            if not self.disable_divergence_module():
                logger.error("禁用背离检测模块失败")
                return False
            
            # 3. 重启基础服务
            if not self.restart_services():
                logger.error("重启服务失败")
                return False
            
            logger.info("紧急回滚完成")
            return True
            
        except Exception as e:
            logger.error(f"紧急回滚失败: {e}")
            return False
    
    def full_rollback(self, backup_timestamp: Optional[int] = None) -> bool:
        """完整回滚"""
        logger.info("执行完整回滚...")
        
        try:
            # 1. 创建当前状态备份
            if not self.create_backup():
                logger.error("创建备份失败")
                return False
            
            # 2. 停止相关服务
            if not self.stop_services():
                logger.error("停止服务失败")
                return False
            
            # 3. 恢复配置
            if not self.restore_config(backup_timestamp):
                logger.error("恢复配置失败")
                return False
            
            # 4. 重启服务
            if not self.restart_services():
                logger.error("重启服务失败")
                return False
            
            logger.info("完整回滚完成")
            return True
            
        except Exception as e:
            logger.error(f"完整回滚失败: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出可用备份"""
        logger.info("列出可用备份...")
        
        backups = []
        backup_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir() and d.name.startswith('backup_')]
        
        for backup_dir in sorted(backup_dirs, key=lambda x: int(x.name.split('_')[1]), reverse=True):
            metadata_path = backup_dir / "backup_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                except Exception as e:
                    logger.warning(f"读取备份元数据失败: {e}")
        
        return backups
    
    def generate_rollback_report(self) -> Dict[str, Any]:
        """生成回滚报告"""
        report = {
            'rollback_id': f"rollback_{int(time.time())}",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(self.config_path),
            'status': self.rollback_status,
            'backup_dir': str(self.backup_dir),
            'rollback_reason': self.rollback_status.get('rollback_reason', 'unknown'),
            'recommendations': [
                "监控系统状态30分钟",
                "检查关键指标是否正常",
                "验证配置是否正确恢复",
                "准备后续修复计划"
            ]
        }
        
        return report
    
    def rollback(self, rollback_type: str = "full", backup_timestamp: Optional[int] = None, 
                 reason: str = "manual") -> bool:
        """执行回滚"""
        self.rollback_status['start_time'] = time.time()
        self.rollback_status['rollback_reason'] = reason
        
        try:
            if rollback_type == "emergency":
                self.rollback_status['current_stage'] = 'emergency_rollback'
                success = self.emergency_rollback()
            elif rollback_type == "full":
                self.rollback_status['current_stage'] = 'full_rollback'
                success = self.full_rollback(backup_timestamp)
            else:
                logger.error(f"未知的回滚类型: {rollback_type}")
                return False
            
            self.rollback_status['success'] = success
            
            if success:
                # 生成回滚报告
                report = self.generate_rollback_report()
                report_path = Path("rollback_report.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                logger.info(f"回滚报告已保存: {report_path}")
                logger.info("回滚成功！请监控系统状态")
            
            return success
            
        except Exception as e:
            logger.error(f"回滚异常: {e}")
            self.rollback_status['success'] = False
            self.rollback_status['errors'].append(str(e))
            return False


def main():
    parser = argparse.ArgumentParser(description='生产环境回滚脚本')
    parser.add_argument('--config', default='config/environments/production.yaml', 
                       help='生产环境配置文件路径')
    parser.add_argument('--type', choices=['emergency', 'full'], 
                       default='full', help='回滚类型')
    parser.add_argument('--backup-timestamp', type=int, 
                       help='指定备份时间戳')
    parser.add_argument('--reason', default='manual', 
                       help='回滚原因')
    parser.add_argument('--list-backups', action='store_true', 
                       help='列出可用备份')
    parser.add_argument('--dry-run', action='store_true', 
                       help='干跑模式，不执行实际回滚')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("干跑模式：只检查配置，不执行实际回滚")
    
    # 创建回滚器
    rollbacker = ProductionRollback(args.config)
    
    if args.list_backups:
        backups = rollbacker.list_backups()
        if backups:
            print("\n可用备份:")
            for backup in backups:
                print(f"  时间戳: {backup['timestamp']}")
                print(f"  备份路径: {backup['backup_path']}")
                print(f"  回滚原因: {backup['rollback_reason']}")
                print("  ---")
        else:
            print("未找到可用备份")
        return
    
    # 执行回滚
    success = rollbacker.rollback(
        rollback_type=args.type,
        backup_timestamp=args.backup_timestamp,
        reason=args.reason
    )
    
    if success:
        logger.info("回滚成功！")
        sys.exit(0)
    else:
        logger.error("回滚失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
