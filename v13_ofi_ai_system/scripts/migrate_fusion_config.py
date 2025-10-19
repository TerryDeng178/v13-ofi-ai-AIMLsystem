#!/usr/bin/env python3
"""
融合指标配置迁移工具

将融合指标配置从代码内硬编码迁移到统一配置系统：
- 从现有代码中提取配置参数
- 生成统一的YAML配置文件
- 验证配置的完整性和正确性
- 提供配置对比和回滚功能

Author: V13 OFI+CVD AI Trading System
Date: 2025-10-20
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.ofi_cvd_fusion import OFICVDFusionConfig
from src.utils.config_loader import ConfigLoader


class FusionConfigMigrator:
    """融合指标配置迁移器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化迁移器
        
        Args:
            project_root: 项目根目录，如果为None则自动检测
        """
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent
        else:
            self.project_root = project_root
        
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "config_backup"
        
        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)
    
    def extract_current_config(self) -> Dict[str, Any]:
        """
        从当前代码中提取配置参数
        
        Returns:
            提取的配置字典
        """
        # 使用默认配置作为当前配置
        default_config = OFICVDFusionConfig()
        
        return {
            'weights': {
                'w_ofi': default_config.w_ofi,
                'w_cvd': default_config.w_cvd
            },
            'thresholds': {
                'fuse_buy': default_config.fuse_buy,
                'fuse_strong_buy': default_config.fuse_strong_buy,
                'fuse_sell': default_config.fuse_sell,
                'fuse_strong_sell': default_config.fuse_strong_sell
            },
            'consistency': {
                'min_consistency': default_config.min_consistency,
                'strong_min_consistency': default_config.strong_min_consistency
            },
            'data_processing': {
                'z_clip': default_config.z_clip,
                'max_lag': default_config.max_lag,
                'warmup_samples': default_config.min_warmup_samples
            },
            'denoising': {
                'hysteresis_exit': default_config.hysteresis_exit,
                'cooldown_secs': default_config.cooldown_secs,
                'min_duration': default_config.min_consecutive
            }
        }
    
    def load_system_config(self) -> Dict[str, Any]:
        """
        加载当前系统配置
        
        Returns:
            系统配置字典
        """
        try:
            config_loader = ConfigLoader(str(self.config_dir))
            config = config_loader.load()
            return config.get('fusion_metrics', {})
        except Exception as e:
            print(f"加载系统配置失败: {e}")
            return {}
    
    def compare_configs(self, current: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较当前配置和系统配置
        
        Args:
            current: 当前配置
            system: 系统配置
            
        Returns:
            比较结果
        """
        differences = []
        missing_in_system = []
        missing_in_current = []
        
        def compare_dict(current_dict: Dict, system_dict: Dict, path: str = ""):
            for key, value in current_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in system_dict:
                    missing_in_system.append(current_path)
                elif isinstance(value, dict) and isinstance(system_dict[key], dict):
                    compare_dict(value, system_dict[key], current_path)
                elif value != system_dict[key]:
                    differences.append({
                        'path': current_path,
                        'current': value,
                        'system': system_dict[key]
                    })
        
        compare_dict(current, system)
        
        # 检查系统配置中多出的项
        def find_extra_in_system(current_dict: Dict, system_dict: Dict, path: str = ""):
            for key, value in system_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in current_dict:
                    missing_in_current.append(current_path)
                elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                    find_extra_in_system(current_dict[key], value, current_path)
        
        find_extra_in_system(current, system)
        
        return {
            'differences': differences,
            'missing_in_system': missing_in_system,
            'missing_in_current': missing_in_current,
            'is_identical': len(differences) == 0 and len(missing_in_system) == 0 and len(missing_in_current) == 0
        }
    
    def backup_current_config(self) -> Path:
        """
        备份当前系统配置
        
        Returns:
            备份文件路径
        """
        import time
        timestamp = int(time.time())
        backup_file = self.backup_dir / f"fusion_config_backup_{timestamp}.yaml"
        
        try:
            system_config = self.load_system_config()
            with open(backup_file, 'w', encoding='utf-8') as f:
                yaml.dump(system_config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"当前配置已备份到: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"备份配置失败: {e}")
            raise
    
    def migrate_config(self, dry_run: bool = False) -> bool:
        """
        执行配置迁移
        
        Args:
            dry_run: 是否为干跑模式（不实际修改文件）
            
        Returns:
            迁移是否成功
        """
        try:
            print("开始融合指标配置迁移...")
            
            # 1. 提取当前配置
            print("1. 提取当前配置参数...")
            current_config = self.extract_current_config()
            
            # 2. 加载系统配置
            print("2. 加载系统配置...")
            system_config = self.load_system_config()
            
            # 3. 比较配置
            print("3. 比较配置差异...")
            comparison = self.compare_configs(current_config, system_config)
            
            if comparison['is_identical']:
                print("✅ 配置已同步，无需迁移")
                return True
            
            # 4. 显示差异
            print("\n配置差异分析:")
            if comparison['differences']:
                print("  - 参数值差异:")
                for diff in comparison['differences']:
                    print(f"    {diff['path']}: {diff['current']} -> {diff['system']}")
            
            if comparison['missing_in_system']:
                print("  - 系统配置中缺失的参数:")
                for missing in comparison['missing_in_system']:
                    print(f"    {missing}")
            
            if comparison['missing_in_current']:
                print("  - 当前配置中缺失的参数:")
                for missing in comparison['missing_in_current']:
                    print(f"    {missing}")
            
            if dry_run:
                print("\n🔍 干跑模式：未实际修改配置文件")
                return True
            
            # 5. 备份当前配置
            print("\n4. 备份当前配置...")
            backup_file = self.backup_current_config()
            
            # 6. 更新系统配置
            print("5. 更新系统配置...")
            system_yaml_file = self.config_dir / "system.yaml"
            
            if system_yaml_file.exists():
                with open(system_yaml_file, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
            else:
                full_config = {}
            
            # 更新融合指标配置
            full_config['fusion_metrics'] = current_config
            
            # 保存更新后的配置
            with open(system_yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 配置迁移完成，备份文件: {backup_file}")
            return True
            
        except Exception as e:
            print(f"❌ 配置迁移失败: {e}")
            return False
    
    def rollback_config(self, backup_file: Path) -> bool:
        """
        回滚配置
        
        Args:
            backup_file: 备份文件路径
            
        Returns:
            回滚是否成功
        """
        try:
            if not backup_file.exists():
                print(f"❌ 备份文件不存在: {backup_file}")
                return False
            
            # 加载备份配置
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_config = yaml.safe_load(f)
            
            # 加载当前系统配置
            system_yaml_file = self.config_dir / "system.yaml"
            with open(system_yaml_file, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
            
            # 恢复融合指标配置
            full_config['fusion_metrics'] = backup_config
            
            # 保存恢复后的配置
            with open(system_yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 配置回滚完成: {backup_file}")
            return True
            
        except Exception as e:
            print(f"❌ 配置回滚失败: {e}")
            return False
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        try:
            print("验证融合指标配置...")
            
            # 加载系统配置
            config_loader = ConfigLoader(str(self.config_dir))
            config = config_loader.load()
            
            # 创建融合指标实例进行验证
            fusion = OFI_CVD_Fusion(config_loader=config_loader)
            
            # 检查配置参数
            cfg = fusion.cfg
            
            # 验证权重
            if cfg.w_ofi < 0 or cfg.w_cvd < 0:
                print("❌ 权重不能为负数")
                return False
            
            if abs(cfg.w_ofi + cfg.w_cvd - 1.0) > 1e-6:
                print("❌ 权重和必须为1.0")
                return False
            
            # 验证阈值
            if cfg.fuse_strong_buy <= cfg.fuse_buy:
                print("❌ 强买入阈值必须大于买入阈值")
                return False
            
            if cfg.fuse_strong_sell >= cfg.fuse_sell:
                print("❌ 强卖出阈值必须小于卖出阈值")
                return False
            
            # 验证一致性阈值
            if not (0 <= cfg.min_consistency <= 1):
                print("❌ 最小一致性阈值必须在0-1之间")
                return False
            
            if not (0 <= cfg.strong_min_consistency <= 1):
                print("❌ 强信号一致性阈值必须在0-1之间")
                return False
            
            if cfg.strong_min_consistency <= cfg.min_consistency:
                print("❌ 强信号一致性阈值必须大于最小一致性阈值")
                return False
            
            print("✅ 配置验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 配置验证失败: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='融合指标配置迁移工具')
    parser.add_argument('--action', choices=['migrate', 'compare', 'validate', 'rollback'], 
                       default='migrate', help='执行操作')
    parser.add_argument('--dry-run', action='store_true', 
                       help='干跑模式，不实际修改文件')
    parser.add_argument('--backup-file', type=str, 
                       help='回滚时指定的备份文件路径')
    parser.add_argument('--project-root', type=str, 
                       help='项目根目录路径')
    
    args = parser.parse_args()
    
    # 创建迁移器
    project_root = Path(args.project_root) if args.project_root else None
    migrator = FusionConfigMigrator(project_root)
    
    if args.action == 'migrate':
        success = migrator.migrate_config(dry_run=args.dry_run)
        if success:
            print("✅ 迁移操作完成")
        else:
            print("❌ 迁移操作失败")
            exit(1)
    
    elif args.action == 'compare':
        current_config = migrator.extract_current_config()
        system_config = migrator.load_system_config()
        comparison = migrator.compare_configs(current_config, system_config)
        
        print("\n配置比较结果:")
        print(f"是否相同: {comparison['is_identical']}")
        print(f"差异数量: {len(comparison['differences'])}")
        print(f"系统配置缺失: {len(comparison['missing_in_system'])}")
        print(f"当前配置缺失: {len(comparison['missing_in_current'])}")
        
        if comparison['differences']:
            print("\n详细差异:")
            for diff in comparison['differences']:
                print(f"  {diff['path']}: {diff['current']} -> {diff['system']}")
    
    elif args.action == 'validate':
        success = migrator.validate_config()
        if not success:
            exit(1)
    
    elif args.action == 'rollback':
        if not args.backup_file:
            print("❌ 回滚操作需要指定备份文件路径")
            exit(1)
        
        backup_file = Path(args.backup_file)
        success = migrator.rollback_config(backup_file)
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
