体检做完就该开刀了。下面给你一套**“一迭代拿下 GCC→GO 状态”的下一步方案**（按优先级可直接并行拆为 4 个 PR）。我把每一步的目标、改动点、验收标准和可直接喂给 CURSOR 的变更提示都写好了。

1) Fix-ENV：彻底消灭组件内直读 env（硬门槛）

目标：所有运行参数只来自统一配置（system.yaml 为主、defaults.yaml 兜底、V13__* 环境变量仅作覆盖），组件内部不得直接 os.getenv / os.environ。
改动点（最少化补丁）：

src/binance_trade_stream.py：第 48、402、712 行把 LOG_LEVEL / WS_URL / SYMBOL 的直读，改成从配置子树读取（如 cfg.logging.level / cfg.data_source.websocket.connection / cfg.data_source.default_symbol），保留 V13__ 覆盖链路（由加载器统一处理）。

src/port_manager.py：第 76、80 行不再直接 os.environ[...]，通过配置键（如 cfg.ports.<name>）获取；仅在加载器阶段映射 V13__PORT_*。
证据与定位：这 5 处直读已在体检里列出（你可对照文件和行号直接改）
验收：重新跑 python tools/gcc_check.py，“环境变量直读检查”=0 条（必须）。

CURSOR 提示词（直接贴给它执行）：

将 src/binance_trade_stream.py 中第 48/402/712 行对 LOG_LEVEL/WS_URL/SYMBOL 的环境变量直读替换为从注入的 cfg 子树读取；src/port_manager.py 第 76/80 行改为通过 cfg.ports 获取端口。要求：

不在组件里解析 env；2) 兼容已有调用（构造函数已接收 cfg）；3) 记录无法找到键时的告警并使用 defaults 兜底；4) 单元测试覆盖“无 env、仅 system.yaml、V13__ 覆盖”三种路径；5) 变更最小、接口向后兼容。

2) Fix-LOADER：增加 system.yaml 主配置加载（优先级第二）

目标：落实“defaults → system → overrides.local → env(V13__)”的优先级链。
改动点：config/unified_config_loader.py.reload() 内新增 system.yaml 读取并在 defaults 之上 merge、在 overrides.local 之前；保留现有 V13__ 前缀的环境变量覆盖。
现状：只加载了 defaults.yaml / overrides.local.yaml 与 env，缺失 system.yaml（体检已指出）。
验收：

supports_system_yaml: true，加载顺序符合报告建议；

输出一份 effective-config.json（运行时回显），确认 key 取值链正确。

再跑体检，“配置加载器检查”→ PASS。

CURSOR 提示词：

在 config/unified_config_loader.py 的 reload() 中加入对 config/system.yaml 的加载与深度合并，顺序为 defaults → system → overrides.local → env(V13__)。提供 dump_effective(path) 方法导出最终生效配置。新增单测覆盖四级优先级与冲突解析。

3) Fix-SCHEMA：更新校验 Schema，关掉“误报的 unknown_keys”

目标：让 tools/validate_config.py 的 Schema 与实际 system.yaml 对齐，避免把合法段落（如 strategy_mode, fusion_metrics, data_harvest, logging.level_by_mode 等）判为 unknown。
现状：unknown_keys=33（正常是 Schema 简化导致）；需要分区化/分层校验。
建议实现：

按大段落定义子模式：logging/monitoring/strategy_mode/fusion_metrics/divergence_detection/data_source/data_harvest/paths/...；

未覆盖段落暂设 AdditionalProperties=True + 记录“弱告警”；

增加“严格模式”开关（CI 用严格，开发机可宽松）。
验收：python tools/validate_config.py --format text 输出 unknown_keys=0（严格模式），或仅为“白名单外新增段”的弱告警（宽松模式）。

CURSOR 提示词：

扩展 tools/validate_config.py：为 system.yaml 的各大段定义子 Schema；实现 --strict/--lenient 模式；严格模式 unknown_keys=0，宽松模式允许未登记段但打印弱告警。补充 5 份样例配置的参数化测试。

4) Fix-INJECT：补齐 Divergence / StrategyMode 的构造注入

目标：所有组件构造函数都接收各自 cfg 子树；组件内部不得再解析全局配置/读取 env。
现状：Divergence 与 StrategyMode 两处未完成注入（体检清单已点名具体文件）。
改动点：

src/divergence_metrics.py、src/ofi_cvd_divergence.py：新增 __init__(self, cfg, ...)，内部用 self.cfg；

src/utils/strategy_mode_manager.py：同样注入并移除内部取全局配置的路径；

保证与 Fusion/Real OFI/CVD 一致的注入范式（你已有的组件多数已合规）。
验收：grep -R "def __init__" src | grep -E "cfg|config" 覆盖这三处；冒烟时通过。

CURSOR 提示词：

为 divergence_metrics.py、ofi_cvd_divergence.py、utils/strategy_mode_manager.py 增加 cfg 构造注入并移除内部对全局配置的解析/任何 env 读取；保证公共接口不变，补单测（默认 cfg、缺省兜底、无破坏）。

5) Smoke & Guardrail：启用“严格模式 + 冒烟”一键验证

目标：把体检项变成可重复、可度量的“出厂测试”。
落地脚本：

# 01 体检（必须 0 直读/0 unknown）
python tools/gcc_check.py
python tools/validate_config.py --format text --strict

# 02 有效配置回显（便于审计与对比）
python -c "from config.unified_config_loader import load; import json; print(json.dumps(load().export(),indent=2))" > reports/effective-config.json

# 03 组件冒烟（最小数据 / dry-run）
python tools/smoke_run.py --symbol ETHUSDT --duration 60s --no-network  # 仅走配置装配链


验收：

GCC 总体状态从 NO-GO → GO；

“单一真源”“构造函数注入”“配置架构对齐”三项变为 PASS，其余运行时项进入可执行验证清单。

6) PR 规划与里程碑（建议 4 个独立 PR）

PR#1 Config-EnvPurge：Fix-ENV（最小补丁 + 单测）；

PR#2 Config-LoaderChain：Fix-LOADER（支持 system.yaml + 回显）；

PR#3 Config-Schema：Fix-SCHEMA（分层校验 + 严格/宽松模式）；

PR#4 Config-Injection：Fix-INJECT（Divergence/StrategyMode 注入改造）。

每个 PR 合并前都跑第 5 步的一键验证脚本；合并后打 GCC-10 标签入库，更新任务板。

7) 任务板与系统路线对齐

在 任务索引里新增 “Stage0 · Global Config Remediation” 分区，并把上面 4 个 PR 对应为 4 个任务卡（完成后解锁 Stage1 的 1.2/1.3）。这与现有 V13 的阶段化路线完全吻合。

验收清单（合并前逐项勾）

 gcc_check.py 环境变量直读 = 0；unknown_keys = 0（strict）。

 加载顺序：defaults → system → overrides.local → env(V13__)；effective-config.json 可追溯。

 Divergence / StrategyMode 具备 cfg 构造注入；grep 检查通过。

 冒烟脚本 60s 通过，无异常日志、无直读 env。

 变更最小、对现有调用与测试零破坏。