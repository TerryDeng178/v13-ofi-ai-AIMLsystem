
# -*- coding: utf-8 -*-
"""
UnifiedConfigLoader
一个简单稳健的"统一配置加载器"，支持：
- 分层文件：defaults.{yaml|json} -> system.yaml -> overrides.local.{yaml|json}（可选）
- 环境变量覆盖（前缀 V13__，多级用双下划线）
- 点号路径读取 config.get("a.b.c", default)
- 快速热加载 reload()
- 导出有效配置 dump_effective()

加载优先级：defaults.yaml (兜底) -> system.yaml (主配置) -> overrides.local.yaml (可选覆盖) -> env(V13__) (运行时覆盖)

注意：此实现为"轻量稳健版"，避免引入复杂依赖，仅用 PyYAML（或内置json）解析。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from pathlib import Path
import os
import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

__all__ = ["UnifiedConfigLoader", "load_config"]


def _deep_merge(a: dict, b: dict) -> dict:
    """浅/深字典合并：b 覆盖 a"""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_env_value(v: str):
    """把字符串环境变量解析成合适类型（int/float/bool/json/原始字符串）"""
    s = v.strip()
    low = s.lower()
    if low in {"true","false"}:
        return low == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    # JSON 对象/数组/数字/布尔
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


class UnifiedConfigLoader:
    """
    统一配置加载器：
      cfg = UnifiedConfigLoader("/path/to/config")
      v = cfg.get("fusion_metrics.thresholds.fuse_buy", 1.5)
    """
    def __init__(self, base_dir: Union[str, Path] = None, env_prefix: str = "V13"):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.env_prefix = env_prefix
        self._cfg: Dict[str, Any] = {}
        self.reload()

    # --- public api ---
    def get(self, path: str = None, default: Any = None):
        """
        获取配置值，支持旧路径到新路径的 Shim 映射
        
        Args:
            path: 配置路径（如 "fusion_metrics.thresholds.fuse_buy"）
            default: 默认值
        
        Returns:
            配置值，如果路径不存在返回 default
        """
        if not path:
            return self._cfg
        
        # 旧路径到新路径的映射（Shim）
        LEGACY_PATH_MAP = {
            "components.fusion.thresholds": "fusion_metrics.thresholds",
            "components.fusion.thresholds.fuse_buy": "fusion_metrics.thresholds.fuse_buy",
            "components.fusion.thresholds.fuse_strong_buy": "fusion_metrics.thresholds.fuse_strong_buy",
            "components.fusion.thresholds.fuse_sell": "fusion_metrics.thresholds.fuse_sell",
            "components.fusion.thresholds.fuse_strong_sell": "fusion_metrics.thresholds.fuse_strong_sell",
            "components.strategy.triggers.market": "strategy_mode.triggers.market",
            "components.strategy.triggers.market.min_trades_per_min": "strategy_mode.triggers.market.min_trades_per_min",
            "components.strategy.triggers.market.min_quote_updates_per_sec": "strategy_mode.triggers.market.min_quote_updates_per_sec",
        }
        
        # 检查是否为旧路径
        if path in LEGACY_PATH_MAP:
            new_path = LEGACY_PATH_MAP[path]
            import warnings
            warnings.warn(
                f"DEPRECATED: 配置路径 '{path}' 已废弃，请使用 '{new_path}'。"
                f"Shim 映射已自动重定向，但建议尽快迁移到新路径。",
                DeprecationWarning,
                stacklevel=2
            )
            path = new_path
        
        # 尝试新路径
        cur = self._cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return dict(self._cfg)
    
    def dump_effective(self, output_path: Union[str, Path] = None, format: str = "json") -> str:
        """
        导出最终生效的配置到文件
        
        Args:
            output_path: 输出文件路径，如果为 None 则返回 JSON 字符串
            format: 输出格式，"json" 或 "yaml"
        
        Returns:
            如果 output_path 为 None，返回 JSON 字符串；否则返回文件路径
        """
        if format.lower() == "yaml":
            if not _HAS_YAML:
                raise RuntimeError("YAML format requires PyYAML, but it's not installed")
            content = yaml.dump(self._cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            content = json.dumps(self._cfg, indent=2, ensure_ascii=False, sort_keys=False)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            return str(output_file)
        else:
            return content
    
    def export(self) -> Dict[str, Any]:
        """导出有效配置（别名方法，兼容性）"""
        return self.to_dict()

    def reload(self):
        """
        按优先级顺序加载配置：
        1. defaults.yaml (兜底配置)
        2. system.yaml (主配置，覆盖 defaults)
        3. overrides.local.yaml (可选本地覆盖，覆盖 system)
        4. 环境变量 V13__* (运行时覆盖，优先级最高)
        """
        cfg = {}
        
        # 1) defaults.yaml (兜底配置)
        defaults = self._load_file_first(["defaults.yaml","defaults.yml","defaults.json"])
        if defaults:
            cfg = _deep_merge(cfg, defaults)
        
        # 2) system.yaml (主配置，覆盖 defaults)
        system_config = self._load_file_first(["system.yaml","system.yml","system.json"])
        if system_config:
            cfg = _deep_merge(cfg, system_config)
        
        # 3) overrides.local.yaml (可选本地覆盖，覆盖 system)
        overrides = self._load_file_first(["overrides.local.yaml","overrides.local.yml","overrides.local.json"])
        if overrides:
            cfg = _deep_merge(cfg, overrides)
        
        # 4) 环境变量覆盖：V13__A__B__C=xxx -> {"A":{"B":{"C":xxx}}}
        # 环境变量优先级最高，覆盖所有文件配置
        envs = {k: v for k, v in os.environ.items() if k.startswith(self.env_prefix + "__")}
        for k, v in envs.items():
            key_path = k[len(self.env_prefix)+2:]  # 去掉前缀和双下划线
            parts = [p for p in key_path.split("__") if p]
            cur = cfg
            for i, p in enumerate(parts):
                if i == len(parts)-1:
                    cur[p] = _parse_env_value(v)
                else:
                    cur = cur.setdefault(p, {})
        
        self._cfg = cfg
        return self

    # --- helpers ---
    def _load_file_first(self, names):
        for name in names:
            p = (self.base_dir / name)
            if p.exists():
                try:
                    if p.suffix.lower() == ".json":
                        return json.loads(p.read_text(encoding="utf-8"))
                    if _HAS_YAML:
                        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception as e:
                    # 解析失败也不中断主流程
                    print(f"[UnifiedConfigLoader] WARN parse {p.name} failed: {e}")
        return {}

# 便捷函数（与旧接口兼容）
_singleton: Optional[UnifiedConfigLoader] = None

def load_config(base_dir: Union[str, Path] = None) -> UnifiedConfigLoader:
    global _singleton
    _singleton = UnifiedConfigLoader(base_dir=base_dir)
    return _singleton
