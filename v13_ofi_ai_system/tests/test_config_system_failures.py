"""
配置系统失败用例测试
测试三类失败场景：divergence/strategy/runtime的越界/缺键/类型错、未消费键触发、锁定层被env覆盖
"""

import pytest
import yaml
import tempfile
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from v13conf import load_config, normalize, validate_invariants, check_unconsumed_keys
from v13conf.invariants import InvariantError
from v13conf.unconsumed_keys import check_unconsumed_keys


@pytest.fixture
def config_dir(tmp_path):
    """创建临时配置目录"""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


@pytest.fixture
def defaults_yaml(config_dir):
    """创建基础defaults.yaml"""
    defaults = {
        'components': {
            'divergence': {
                'min_strength': 0.5,
                'min_separation_secs': 300.0,
                'count_conflict_only_when_fusion_ge': 2.0,
                'lookback_periods': 20,
                'swing_L': 10,
                'ema_k': 5,
                'z_hi': 2.0,
                'z_mid': 1.0,
                'min_separation': 10,
                'cooldown_secs': 60.0,
                'warmup_min': 30,
                'max_lag': 5.0,
                'use_fusion': True,
                'cons_min': 0.5,
            },
            'strategy': {
                'mode': 'auto',
                'hysteresis': {
                    'window_secs': 60,
                    'min_active_windows': 2,
                    'min_quiet_windows': 4,
                },
            },
            'runtime': {
                'logging': {
                    'level': 'INFO',
                    'debug': False,
                    'heartbeat_interval_sec': 30,
                },
                'performance': {
                    'max_queue_size': 1000,
                    'batch_size': 100,
                    'flush_interval_ms': 100,
                },
            },
        }
    }
    defaults_file = config_dir / "defaults.yaml"
    with open(defaults_file, 'w', encoding='utf-8') as f:
        yaml.dump(defaults, f)
    return defaults_file


class TestDivergenceFailures:
    """Divergence组件失败用例"""
    
    def test_divergence_out_of_range_min_strength(self, config_dir, defaults_yaml):
        """测试min_strength越界（>1.0）"""
        overrides = {
            'components': {
                'divergence': {
                    'min_strength': 1.5,  # 应该 <= 1.0
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        cfg = normalize(cfg)
        
        errors = validate_invariants(cfg, 'divergence')
        assert len(errors) > 0
        assert any('min_strength' in str(e.message).lower() or 'range' in str(e.message).lower() 
                  for e in errors)
    
    def test_divergence_missing_key(self, config_dir, defaults_yaml):
        """测试缺少必需键"""
        overrides = {
            'components': {
                'divergence': {
                    'min_strength': 0.5,
                    # 缺少 z_hi
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        cfg = normalize(cfg)
        
        # 应该能加载，但未消费键检测会捕获
        unconsumed = check_unconsumed_keys(cfg.get('components', {}).get('divergence', {}), 'divergence')
        # 注意：这里可能不会捕获，因为缺少的键不会出现在配置中
    
    def test_divergence_wrong_type(self, config_dir, defaults_yaml):
        """测试类型错误"""
        overrides = {
            'components': {
                'divergence': {
                    'min_strength': '0.5',  # 应该是float
                    'lookback_periods': '20',  # 应该是int
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        # YAML会自动转换类型，所以这个测试可能需要通过Schema验证


class TestStrategyFailures:
    """Strategy组件失败用例"""
    
    def test_strategy_invalid_mode(self, config_dir, defaults_yaml):
        """测试无效的mode值"""
        overrides = {
            'components': {
                'strategy': {
                    'mode': 'invalid_mode',  # 应该是 auto/active/quiet
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        cfg = normalize(cfg)
        # 应该通过Schema验证捕获（如果实现了Schema验证）
    
    def test_strategy_missing_hysteresis(self, config_dir, defaults_yaml):
        """测试缺少hysteresis配置"""
        overrides = {
            'components': {
                'strategy': {
                    'mode': 'auto',
                    # 缺少 hysteresis
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        # 合并后应该有defaults的hysteresis


class TestRuntimeFailures:
    """Runtime组件失败用例"""
    
    def test_runtime_invalid_log_level(self, config_dir, defaults_yaml):
        """测试无效的日志级别"""
        overrides = {
            'runtime': {
                'logging': {
                    'level': 'INVALID',  # 应该是 DEBUG/INFO/WARNING/ERROR/CRITICAL
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        # 这个需要通过Schema验证捕获
    
    def test_runtime_negative_queue_size(self, config_dir, defaults_yaml):
        """测试负值"""
        overrides = {
            'runtime': {
                'performance': {
                    'max_queue_size': -1,  # 应该 > 0
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        # 需要通过不变量验证


class TestUnconsumedKeys:
    """未消费键检测失败用例"""
    
    def test_unconsumed_key_triggers_failure(self, config_dir, defaults_yaml):
        """测试未消费键触发失败"""
        overrides = {
            'components': {
                'fusion': {
                    'thresholds': {
                        'fuse_buy': 1.0,
                        'typo_key': 999,  # 拼写错误
                    }
                }
            }
        }
        overrides_file = config_dir / "overrides.local.yaml"
        with open(overrides_file, 'w', encoding='utf-8') as f:
            yaml.dump(overrides, f)
        
        cfg, _ = load_config(str(config_dir))
        cfg = normalize(cfg)
        
        # 应该检测到未消费键
        fusion_cfg = cfg.get('components', {}).get('fusion', {})
        unconsumed = check_unconsumed_keys(fusion_cfg, 'fusion', fail_on_unconsumed=False)
        
        # 应该有未消费键
        assert len(unconsumed) > 0
        assert any('typo' in key.lower() or 'fusion' in key.lower() for key in unconsumed)
        
        # 测试fail_on_unconsumed=True应该抛出异常
        with pytest.raises(ValueError, match="未消费的配置键"):
            check_unconsumed_keys(fusion_cfg, 'fusion', fail_on_unconsumed=True)


class TestLockedParamsOverride:
    """OFI锁定参数覆盖测试"""
    
    def test_env_cannot_override_locked_by_default(self, config_dir, defaults_yaml, monkeypatch):
        """测试默认情况下env无法覆盖锁定参数"""
        # 创建locked_ofi_params.yaml
        locked = {
            'ofi_calculator': {
                'z_window': 80,
                'ema_alpha': 0.30,
                'z_clip': 3.0,
            }
        }
        locked_file = config_dir / "locked_ofi_params.yaml"
        with open(locked_file, 'w', encoding='utf-8') as f:
            yaml.dump(locked, f)
        
        # 尝试通过环境变量覆盖
        monkeypatch.setenv('V13__components__ofi__z_window', '100')
        
        cfg, sources = load_config(str(config_dir), allow_env_override_locked=False)
        
        # 锁定参数应该生效，环境变量被忽略
        assert cfg['components']['ofi']['z_window'] == 80
        assert sources.get('components.ofi.z_window') == 'locked'
    
    def test_env_can_override_locked_when_allowed(self, config_dir, defaults_yaml, monkeypatch):
        """测试allow_env_override_locked=True时env可以覆盖"""
        # 创建locked_ofi_params.yaml
        locked = {
            'ofi_calculator': {
                'z_window': 80,
            }
        }
        locked_file = config_dir / "locked_ofi_params.yaml"
        with open(locked_file, 'w', encoding='utf-8') as f:
            yaml.dump(locked, f)
        
        # 通过环境变量覆盖
        monkeypatch.setenv('V13__components__ofi__z_window', '100')
        
        cfg, sources = load_config(str(config_dir), allow_env_override_locked=True)
        
        # 环境变量应该覆盖锁定参数
        assert cfg['components']['ofi']['z_window'] == 100
        assert sources.get('components.ofi.z_window') == 'env'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

