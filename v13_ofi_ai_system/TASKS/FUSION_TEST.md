角色

你是一名资深量化 QA/DevOps 工程师，目标是完整测试与验收 FUSION 组件的功能正确性、时序鲁棒、性能与基础信号质量。

目标组件

目标文件：ofi_cvd_fusion.py（项目根或子目录内，需自动查找并导入 OFI_CVD_Fusion 类）

禁止修改生产代码，仅允许在 tests/ 与 scripts/ 下新增文件。

需要产出的制品

tests/test_fusion_unit.py —— 逻辑正确性单元测试（pytest）

scripts/run_fusion_synthetic_eval.py —— 合成数据批量评测脚本

scripts/run_fusion_offline_eval.py —— 离线数据评测脚本（若检测到本地采集数据）

results/fusion_metrics_summary.csv —— 关键指标表

results/fusion_test_report.md —— 摘要报告（可读给业务/研发）

依赖约束

仅用本地依赖；如缺失，请在脚本里优雅降级或提示安装：pytest, numpy, pandas。

不访问外网。

时间戳一律以秒为单位传入 FUSION（采集器里常用 ts_ms → /1000.0）。

FUSION 输入约定（用于测试脚本）

update(z_ofi: float, z_cvd: float, ts: float, lag_sec: float=0.0, price: float|None=None) -> dict

典型返回含：signal, score/fusion_score, consistency, reason_codes, stats，以及你在实现里已有的可观测字段。

任务 A：单元测试（pytest）

创建 tests/test_fusion_unit.py，覆盖以下场景并断言结果（必要时构造自定义 cfg）：

最小持续门槛

cfg：min_consecutive=2, 其他取默认；

输入两帧连续 BUY 倾向（z_ofi=z_cvd=+3），ts 逐帧 +0.1s，lag_sec=0；

断言：第1帧 signal==NEUTRAL 且理由包含 min_duration；第2帧给出 BUY 或更强。

一致性临界提升

设定 BUY 阈值 fuse_buy，喂“接近阈值但略低”的分数且 OFI/CVD 同向、强度比接近；

断言：出现 consistency_boost，允许首帧放行（若你的实现是放行）；否则断言被 min_duration 抑制（保守模式）。

冷却期

发出一次非中性信号后，在 cooldown_secs 内再次给出强信号；

断言：第二次被抑制为 NEUTRAL，理由含 cooldown。

单因子降级

令 lag_sec > max_lag，并设置 |z_ofi| > |z_cvd|；

断言：reason_codes 含 lag_exceeded 与 degraded_ofi_only；一致性被视作 1.0；可产生非中性信号。

迟滞退出

买入阈值上穿产生 BUY 后，轻微回落不足以触发 hysteresis_exit；

断言：保持买持有；当回落超过退出阈值，才回到 NEUTRAL。

热更新接口

调用 set_thresholds(w_ofi=0.8, w_cvd=0.6)；

断言：权重被归一化且更新成功；其他阈值可选测一项。

在每个用例末尾，校验 fusion.stats 中相关计数（如 downgrades、cooldown_blocks、min_duration_blocks）有合理增量。

任务 B：合成数据评测

创建 scripts/run_fusion_synthetic_eval.py，生成 3 组 3 分钟、100Hz（dt=0.01s）的序列，统计并输出指标：

场景 S1：同向强信号 + 小抖动
z_ofi = 2.8 + noise(0,0.2), z_cvd = 3.2 + noise(0,0.2), lag_sec=0

场景 S2：交替滞后 + 超时降级
z_ofi ~ N(3,0.5), z_cvd ~ N(2,0.5)，每隔 2s 注入 lag_sec = max_lag*1.5 的段落

场景 S3：对冲/反向
z_ofi ~ +N(2,0.5), z_cvd ~ -N(2,0.5)，一致性应接近 0

对三组分别运行 update()，聚合以下指标并写入 results/fusion_metrics_summary.csv（追加模式，含场景列）：

updates、non_neutral_rate、downgrades/updates、cooldown_blocks/updates、min_duration_blocks/updates、consistency_boost/updates

update_cost_ms 的 p50/p95/p99（用 time.perf_counter() 采样）

若 signal 有强弱层级：强信号占比、强信号→弱信号回退次数

同时生成 results/fusion_test_report.md 的合成数据章节，给出各指标结论与是否达标（门槛示例：p99(update_cost)<3ms，non_neutral_rate 在 2%~15% 之间可作为默认参考，按你脚本里 cfg 调整）。

任务 C：离线数据评测（可选自动启用）

创建 scripts/run_fusion_offline_eval.py，自动搜寻以下文件（不存在则跳过本任务）：

预览/宽表 CSV/Parquet：文件名或列名包含任一关键字

OFI：ofi_z

CVD：z_cvd 或 cvd_z

时间戳：ts_ms 或 second_ts

滞后：lag_ms_to_trade（若无则 lag_sec=0）

场景：scenario_2x2（可选）

价格/收益：mid, price, return_1s/5s/30s（可选）

逻辑：

对齐时间戳到秒（或使用已有对齐列）；

计算 lag_sec = lag_ms_to_trade/1000.0（无则置 0）；

顺序调用 fusion.update(...) 收集输出；

若存在未来收益列 return_{H}，计算以下三类信号质量指标：

Hit@H = mean(sign(fwd_ret_H) == side(signal))（仅非中性）

IC@H = spearman(fusion_score, fwd_ret_H) 与 ICIR = mean(IC)/std(IC)

Top-10% 分位与全样本平均收益对比（Lift）

指标写入 results/fusion_metrics_summary.csv，并在 results/fusion_test_report.md 追加“离线评测”章节（按场景分组给出表格）。

任务 D：报告生成规范

results/fusion_test_report.md 需包含：

版本信息（从 ofi_cvd_fusion.py 读取 __file__ 修改时间、OFI_CVD_Fusion.__dict__ 中关键 cfg 字段快照）

合成数据概览表 + 关键结论（✅/⚠️）

（若有）离线数据的 Hit/IC/Lift 对比（Fusion vs OFI vs CVD，若能从文件中读取单因子 Z 列则一起算）

性能统计（p50/p95/p99）

规则命中分布（reason_codes 计数 Top5）

“下一步建议”（若某项未达标，给出调参方向：min_consecutive/cooldown/fuse_*）

任务 E：执行与验收

在项目根创建 Makefile（可选）：

make test-unit → 运行 pytest

make eval-synth → 跑合成评测

make eval-offline → 跑离线评测（若数据存在）

make report → 汇总生成 MD

直接给我打印以下路径，便于下载/查阅：

results/fusion_metrics_summary.csv

results/fusion_test_report.md

验收标准（示例，可在报告中标注是否达标）：

p99(update_cost_ms) < 3 ms（本机环境下，可适度放宽）

合成 S1：non_neutral_rate 在 5%~20%，consistency_boost/updates > 0

合成 S2：downgrades/updates > 0，且一致性并未整体坍塌

离线评测（若有收益列）：Fusion 的 Hit@5s 与 IC@5s 不低于 单因子最优者（给出 Δ）

实现要点与小贴士

导入组件时需要动态定位 ofi_cvd_fusion.py，用 importlib.util.spec_from_file_location 方案以避免包路径问题。

性能统计用 time.perf_counter()，避免 time.time() 的系统时钟抖动。

若实现里 SignalType 是枚举，测试中用同一枚举；如难以导入，按返回字典里的字符串/整数代号比较。

生成噪声用 numpy.random.default_rng(42) 固定种子，保证复现性。

发现缺依赖时，仅在脚本启动时报清晰提示，不要自动安装（保持离线）。