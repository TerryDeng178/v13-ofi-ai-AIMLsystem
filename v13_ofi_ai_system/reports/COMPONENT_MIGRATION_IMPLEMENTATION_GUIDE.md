# 组件迁移实施指南 - 快速完成剩余工作

## 当前状态

✅ **已完成**:
- 运行时加载工具 (`v13conf/runtime_loader.py`)
- OFI组件库式接口改造
- OFI服务式入口 (`bin/ofi.py`)
- 构建工具支持（conf_build.py 已支持5个组件）
- CI/CD 集成（6阶段验证）

⚠️ **待完成**:
- CVD、FUSION、DIVERGENCE、STRATEGY MODE 组件库式接口改造
- 剩余4个服务式入口（bin/cvd.py, bin/fusion.py, bin/divergence.py, bin/strategy.py）
- CoreAlgo 和 Paper Trading Simulator 切换到库式配置注入

---

## 快速实施步骤

### 步骤1: 修改组件 `__init__()` 方法（5-10分钟/组件）

**模板代码**（以CVD为例）：

```python
# src/real_cvd_calculator.py
from typing import Dict, Any

def __init__(self, symbol: str, cfg: CVDConfig = None, config_loader=None, 
             runtime_cfg: Dict[str, Any] = None):
    # 优先使用运行时配置字典（库式调用）
    if runtime_cfg is not None:
        cvd_cfg = runtime_cfg.get('cvd', {}) if isinstance(runtime_cfg, dict) else {}
        cfg = CVDConfig(
            z_window=cvd_cfg.get('z_window', 150),
            ema_alpha=cvd_cfg.get('ema_alpha', 0.2),
            use_tick_rule=cvd_cfg.get('use_tick_rule', True),
            warmup_min=cvd_cfg.get('warmup_min', 3),
            winsor_limit=cvd_cfg.get('winsor_limit', 2.0),
            mad_multiplier=cvd_cfg.get('mad_multiplier', 1.70),
            # ... 其他参数
        )
    elif config_loader:
        # 兼容旧接口
        cfg = self._load_from_config_loader(config_loader, symbol)
    elif cfg is None:
        cfg = CVDConfig()
    
    # ... 后续初始化逻辑保持不变
```

**需要修改的文件**:
1. `src/real_cvd_calculator.py` - 添加 `runtime_cfg` 参数
2. `src/ofi_cvd_fusion.py` - 添加 `runtime_cfg` 参数
3. `src/ofi_cvd_divergence.py` - 添加 `runtime_cfg` 参数
4. `src/utils/strategy_mode_manager.py` - 添加 `runtime_cfg` 参数

---

### 步骤2: 创建服务式入口（2分钟/组件）

**模板**（复制 `bin/ofi.py`，修改组件名）：

```python
# bin/cvd.py (其他组件同理)
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from v13conf.runtime_loader import load_component_runtime_config, print_component_effective_config

def main():
    parser = argparse.ArgumentParser(description='CVD计算器组件')
    parser.add_argument('--config', default=None)
    parser.add_argument('--dry-run-config', action='store_true')
    parser.add_argument('--compat-global-config', action='store_true')
    parser.add_argument('--print-effective', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    cfg = load_component_runtime_config(
        component='cvd',  # 修改这里
        pack_path=args.config,
        compat_global=args.compat_global_config
    )
    
    if args.print_effective:
        print_component_effective_config(cfg, 'cvd', verbose=args.verbose)
    
    if args.dry_run_config:
        print("[DRY-RUN] CVD组件配置验证通过")
        return 0
    
    print("[INFO] CVD组件初始化成功（库式调用模式）")
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

**需要创建的文件**:
- `bin/cvd.py`
- `bin/fusion.py`
- `bin/divergence.py`
- `bin/strategy.py`

---

### 步骤3: 更新 CoreAlgo 使用库式配置（可选，推荐）

**修改 `core/core_algo.py::_init_components()`**：

```python
def _init_components(self):
    """初始化成熟组件 - 使用库式运行时配置"""
    try:
        # 从运行时配置字典提取组件配置
        components_cfg = self.system_config.get('components', {})
        
        # 1. OFI计算器
        ofi_cfg = components_cfg.get('ofi', {})
        self.ofi_calc = RealOFICalculator(
            self.symbol,
            runtime_cfg={'ofi': ofi_cfg}  # 库式调用
        )
        
        # 2. CVD计算器
        cvd_cfg = components_cfg.get('cvd', {})
        self.cvd_calc = RealCVDCalculator(
            self.symbol,
            runtime_cfg={'cvd': cvd_cfg}  # 库式调用
        )
        
        # 3. FUSION
        fusion_cfg = components_cfg.get('fusion', {})
        self.fusion = OFI_CVD_Fusion(
            runtime_cfg={'fusion': fusion_cfg}  # 库式调用
        )
        
        # 4. DIVERGENCE
        divergence_cfg = components_cfg.get('divergence', {})
        self.divergence = DivergenceDetector(
            runtime_cfg={'divergence': divergence_cfg}  # 库式调用
        )
        
        # 5. STRATEGY MODE
        strategy_cfg = components_cfg.get('strategy', {})
        self.strategy_manager = StrategyModeManager(
            runtime_cfg={'strategy': strategy_cfg}  # 库式调用
        )
```

---

## 验收测试

```bash
# 1. 构建所有运行时包
python tools/conf_build.py all --base-dir config

# 2. 服务式入口验证
python bin/ofi.py --dry-run-config
python bin/cvd.py --dry-run-config
python bin/fusion.py --dry-run-config
python bin/divergence.py --dry-run-config
python bin/strategy.py --dry-run-config

# 3. CI验证（本地）
python tools/conf_build.py all --base-dir config --dry-run-config
```

---

## 注意事项

1. **保持向后兼容**: `config_loader` 参数仍保留，避免破坏现有代码
2. **配置路径**: 运行时配置字典的结构为 `{'components': {'ofi': {...}, 'cvd': {...}, ...}}`
3. **错误处理**: 如果 `runtime_cfg` 中缺少某个键，使用默认值（从 `defaults.yaml` 复制）
4. **测试覆盖**: 修改后运行现有单元测试，确保功能正常

---

## 预计时间

- 步骤1（组件改造）: 20-30分钟（4个组件 × 5-10分钟）
- 步骤2（服务式入口）: 10分钟（4个文件 × 2分钟）
- 步骤3（CoreAlgo更新）: 10-15分钟
- 验收测试: 10分钟

**总计**: 约1小时完成全部迁移工作。

