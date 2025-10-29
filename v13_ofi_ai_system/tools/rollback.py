#!/usr/bin/env python3
"""
Core Algorithm 一键回滚脚本
基于最终修复验证报告的回滚预案
"""

import os
import sys
import yaml
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class CoreAlgorithmRollback:
    """Core Algorithm 回滚管理器"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.backup_dir = Path("config/backups")
        self.rollback_log = Path("logs/rollback.log")
        
    def create_backup(self):
        """创建当前配置备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 备份关键配置文件
        config_files = [
            "defaults.yaml",
            "system.yaml",
            "canary_deployment.yaml"
        ]
        
        for config_file in config_files:
            src = self.config_dir / config_file
            if src.exists():
                dst = backup_path / config_file
                shutil.copy2(src, dst)
                print(f"Backed up {config_file} to {dst}")
        
        return backup_path
    
    def rollback_to_previous_version(self):
        """回滚到上一个版本"""
        print("=== Core Algorithm 回滚开始 ===")
        
        # 1. 创建当前状态备份
        current_backup = self.create_backup()
        print(f"Current state backed up to: {current_backup}")
        
        # 2. 查找上一个稳定版本
        previous_version = self.find_previous_stable_version()
        if not previous_version:
            print("ERROR: No previous stable version found")
            return False
        
        print(f"Rolling back to: {previous_version}")
        
        # 3. 恢复配置文件
        if not self.restore_config_files(previous_version):
            print("ERROR: Failed to restore config files")
            return False
        
        # 4. 清空system.yaml覆盖
        self.clear_system_overrides()
        
        # 5. 触发组件热重载
        if not self.trigger_hot_reload():
            print("ERROR: Failed to trigger hot reload")
            return False
        
        # 6. 验证回滚结果
        if not self.verify_rollback():
            print("ERROR: Rollback verification failed")
            return False
        
        print("=== 回滚完成 ===")
        return True
    
    def find_previous_stable_version(self) -> str:
        """查找上一个稳定版本"""
        # 查找备份目录中的稳定版本
        if not self.backup_dir.exists():
            return None
        
        backups = sorted(self.backup_dir.iterdir(), key=os.path.getmtime, reverse=True)
        
        for backup in backups:
            if backup.is_dir() and backup.name.startswith("stable_"):
                return backup.name
        
        # 如果没有稳定版本，使用最新的备份
        if backups:
            return backups[0].name
        
        return None
    
    def restore_config_files(self, version: str) -> bool:
        """恢复配置文件"""
        try:
            version_path = self.backup_dir / version
            
            # 恢复defaults.yaml
            defaults_src = version_path / "defaults.yaml"
            defaults_dst = self.config_dir / "defaults.yaml"
            if defaults_src.exists():
                shutil.copy2(defaults_src, defaults_dst)
                print(f"Restored defaults.yaml from {version}")
            
            # 恢复system.yaml
            system_src = version_path / "system.yaml"
            system_dst = self.config_dir / "system.yaml"
            if system_src.exists():
                shutil.copy2(system_src, system_dst)
                print(f"Restored system.yaml from {version}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to restore config files: {e}")
            return False
    
    def clear_system_overrides(self):
        """清空system.yaml的覆盖配置"""
        system_yaml = self.config_dir / "system.yaml"
        
        if system_yaml.exists():
            # 创建最小化的system.yaml
            minimal_config = {
                "version": "rollback",
                "timestamp": datetime.now().isoformat(),
                "description": "Rollback configuration - minimal overrides"
            }
            
            with open(system_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(minimal_config, f, default_flow_style=False)
            
            print("Cleared system.yaml overrides")
    
    def trigger_hot_reload(self) -> bool:
        """触发组件热重载"""
        try:
            # 发送SIGUSR1信号触发热重载
            # 这里需要根据实际部署方式调整
            print("Triggering hot reload...")
            
            # 示例：通过API触发重载
            # subprocess.run(["curl", "-X", "POST", "http://localhost:8080/reload"])
            
            # 或者通过文件系统信号
            reload_signal = Path("config/reload.trigger")
            reload_signal.touch()
            
            print("Hot reload signal sent")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to trigger hot reload: {e}")
            return False
    
    def verify_rollback(self) -> bool:
        """验证回滚结果"""
        print("Verifying rollback...")
        
        # 等待系统稳定
        time.sleep(30)
        
        # 运行健康检查
        try:
            result = subprocess.run([
                "python", "tools/shadow_go_nogo.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("Rollback verification: PASS")
                return True
            else:
                print(f"Rollback verification: FAIL - {result.stderr}")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to verify rollback: {e}")
            return False
    
    def emergency_rollback(self):
        """紧急回滚 - 使用硬编码的安全参数"""
        print("=== 紧急回滚模式 ===")
        
        # 使用硬编码的安全参数
        emergency_config = {
            "fusion": {
                "thresholds": {
                    "fuse_buy": 1.0,
                    "fuse_sell": -1.0,
                    "fuse_strong_buy": 1.8,  # 回滚到原始值
                    "fuse_strong_sell": -1.8
                },
                "consistency": {
                    "min_consistency": 0.15,  # 回滚到原始值
                    "strong_min_consistency": 0.50  # 回滚到原始值
                }
            },
            "divergence": {
                "min_strength": 0.80,  # 回滚到原始值
                "min_separation_secs": 90,
                "count_conflict_only_when_fusion_ge": 1.0
            }
        }
        
        # 直接写入defaults.yaml
        defaults_yaml = self.config_dir / "defaults.yaml"
        with open(defaults_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(emergency_config, f, default_flow_style=False)
        
        print("Emergency rollback configuration applied")
        
        # 清空system.yaml
        self.clear_system_overrides()
        
        # 触发重载
        self.trigger_hot_reload()
        
        print("Emergency rollback completed")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python rollback.py [normal|emergency]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    rollback = CoreAlgorithmRollback()
    
    if mode == "normal":
        success = rollback.rollback_to_previous_version()
        if success:
            print("Normal rollback completed successfully")
            sys.exit(0)
        else:
            print("Normal rollback failed")
            sys.exit(1)
    
    elif mode == "emergency":
        rollback.emergency_rollback()
        print("Emergency rollback completed")
        sys.exit(0)
    
    else:
        print("Invalid mode. Use 'normal' or 'emergency'")
        sys.exit(1)

if __name__ == "__main__":
    main()
