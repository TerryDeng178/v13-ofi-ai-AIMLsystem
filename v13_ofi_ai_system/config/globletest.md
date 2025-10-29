做一次“全局配置到位检查（GCC：Global Config Check）”。下面这套一次跑通即可的验收清单 + 快速自检脚本，能把 harvest、ofi、cvd、fusion、divergence、strategymode、core_algo 的统一配置落实到位，并与 V13 的“统一配置系统 / 动态模式切换 / 原子热更新 / 监控告警”设计对齐。

一次性验收清单（GCC-10）

单一真源
确认所有组件只通过 config/system.yaml（默认由 defaults.yaml 兜底，环境变量仅用于覆盖）取值；不允许组件直接读 env。优先级：ENV > system.yaml > defaults.yaml。

构造函数注入
每个组件的构造函数都接收 cfg 子树（如 OFI(cfg.ofi)），不得在组件内部再去解析全局或读取 env；此前你已经推进了这项改造，这一步是“验收”。

配置架构对齐
system.yaml 顶层建议含：system/binance/storage/logging/monitoring/harvest/ofi/cvd/fusion/divergence/strategymode/core_algo，键名与组件参数一一对应（无多义、无同名异义）。

动态模式 & 原子热更新
开启/验证“动态模式切换 + 原子热更新（RCU）”，修改少量门限（如 ofi.z_window、fusion.thresholds）应在不重启、不掉线的前提下生效，且不丢序列。

有效配置回显
所有组件启动时输出有效配置快照（已脱敏）与配置指纹（hash），并写入 metrics：config_version、reloads_total、cfg_source（defaults/system/env）。Grafana 面板能直接看到。

监控阈值绑定
metrics.json 与配置联动：延迟分位、接收率、序列一致性、策略阈值等指标的报警阈值要来自配置（非硬编码），以便灰度/回滚。

跨组件一致性约束

符号统一大小写 & 市场侧一致（例：ETHUSDT）。

全局时区统一（文档默认 HKT，你如采用 JST，请在 system.timezone 统一并让所有时间戳/窗口依赖该键）。

严格模式
配置加载器启用 strict=true：未知键报错、类型不匹配报错、缺省用 defaults.yaml 兜底但写日志。

回退路径与只读白名单
把允许热更的键（如阈值/窗口）与需要重启的键（如数据路径/流端点）分开，避免在运行中更改“破坏性键”。

冒烟跑通
以相同 system.yaml，依次 60–120 秒 dry-run：harvest→ofi→cvd→fusion→divergence→strategymode→core_algo，确保：

全部打印有效配置与指纹；

指标达标（接收率>1/s、p95 event 延迟<2.5s、resyncs=0 等，按档案阈值）；

修改 1–2 个门限可热更成功、无丢序列。

3 个“快速自检”命令

扫 env 直读（必须为 0 条）

grep -RInE "os\\.getenv|os\\.environ\\[|dotenv" src/ -n || true


验证构造注入（应只出现 cfg 子树）

grep -RInE "def __init__\\(.*cfg" src/{harvest,ofi,cvd,fusion,divergence,strategymode,core_algo} -n


检查未知配置键（对比 schema）

# 见下方脚本：输出 Unknown Keys / Type Errors 即 FAIL
python tools/validate_config.py

极简校验脚本（可直接放 tools/validate_config.py）
import sys, yaml, json, pathlib
SCHEMA = {
  "system": {"timezone": str, "env": str},
  "binance": {"ws_endpoint": str, "streams": dict},
  "storage": {"raw_dir": str, "parquet_dir": str},
  "logging": {"level": str, "dir": str},
  "monitoring": {"metrics_path": str, "enable": bool},
  "harvest": {"symbols": list, "depth": int, "interval_ms": int},
  "ofi": {"z_window": int, "z_clip": float, "weights": list},
  "cvd": {"z_window": int, "ema_alpha": float, "reset_period": (int, type(None))},
  "fusion": {"w_ofi": float, "w_cvd": float, "buy": float, "sell": float,
             "strong_buy": float, "strong_sell": float, "min_consistency": float},
  "divergence": {"lookback": int, "z_ofi_min": float, "z_cvd_min": float},
  "strategymode": {"active": bool, "min_trades_per_min": int},
  "core_algo": {"cooldown_secs": float, "position_max": float}
}
def _check_types(tree, schema, prefix=""):
    errs, unknown = [], []
    for k, v in tree.items():
        if k not in schema: unknown.append(prefix+k); continue
        expected = schema[k]
        if isinstance(expected, dict):
            if not isinstance(v, dict): errs.append(f"type:{prefix+k} expected dict"); continue
            e2,u2 = _check_types(v, expected, prefix+k+"."); errs+=e2; unknown+=u2
        else:
            if not isinstance(v, expected if isinstance(expected, tuple) else (expected,)):
                errs.append(f"type:{prefix+k} expected {expected}")
    return errs, unknown
cfg = yaml.safe_load(open("config/system.yaml"))
errs, unknown = _check_types(cfg, SCHEMA)
print(json.dumps({"type_errors": errs, "unknown_keys": unknown}, indent=2, ensure_ascii=False))
if errs or unknown: sys.exit(1)

GO / NO-GO 判定（一次跑通即可）

GO：

env 直读 = 0；

validate_config 无 unknown/type 错误；

7 个组件均打印“有效配置指纹”，Grafana 可见 config_version；

热更 1–2 个阈值成功，序列一致（resyncs=0）。

NO-GO：任一项不达标，回到第 1–3 步修复再验收。

以上流程与 V13 的“统一配置系统 / 环境变量覆盖 / 动态模式 / 原子热更新 / 监控告警 / 分阶段硬性阈值”完全一致，跑完即能确信“全局配置已真正落地”，且后续灰度和参数调优有据可依。