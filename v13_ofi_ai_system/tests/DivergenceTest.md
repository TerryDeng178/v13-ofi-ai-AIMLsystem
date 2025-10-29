你是项目的测试执行工程师。请在当前仓库中为「Divergence 背离计算器」实现并运行一套自动化测试，按以下步骤操作并产出可追溯的报告与工件。

使用这个目录的真实6个交易对的数据：F:\ofi_cvd_framework\ofi_cvd_framework\v13_ofi_ai_system\deploy\data\ofi_cvd\date=2025-10-27 来完成以下测试

目标

依据以下“验收标准”对 ofi_cvd_divergence.py 中的背离计算器进行自动化验证，并输出：

tests/test_divergence_detector.py（pytest）

测试报告：reports/junit_divergence.xml（JUnit）与 reports/REPORT_DIVERGENCE_TEST.md（可读版）

关键性能指标（p50/p95/p99 单次 update 耗时）

验收标准（必须满足）

接口契约：非法输入（NaN/inf、非正 ts/price）不得产出事件且不污染状态。

功能正确性：能检测四类背离（看涨/看跌常规、看涨/看跌隐藏），事件结构包含 ts/type/score/channels/lookback/pivots/debug/warmup/stats。

稳定性：同一 (event_type, channel) 在 cooldown_secs 内抑制，且同一 (idx_a, idx_b) 仅触发一次（去重）。

融合一致性：仅在提供 fusion_score 时启用 Fusion 通道；更高 consistency 提高 Fusion 事件得分。

值域与健壮性：z_ofi/z_cvd 与 fusion_score 裁剪到 [-5,5]；NaN/inf 不进入枢轴/评分。

统计一致性：events_total == sum(events_by_type.values())；枢轴计数与分通道统计一致。

性能建议：1e5 条样本基准测试，update() p99 ≤ 1ms（若机器性能不足，至少 p99 ≤ 5ms 且平均 <1ms）。

实施细节

解析被测 API

打开 /mnt/data/ofi_cvd_divergence.py（或仓库内同名文件，优先以项目根目录文件为准）。

读取并确定导出的类/函数名称（通常为 DivergenceDetector, DivergenceConfig，以及 update(...) 的完整签名）。

在测试代码中封装 call_update(...)，根据实际签名传参（优先关键字）。如签名包含：ts, price, z_ofi, z_cvd, fusion_score=None, consistency=None, warmup=False, lag_sec=0.0，则照此调用；若参数名不同，请自动适配。

创建测试文件 tests/test_divergence_detector.py

使用 pytest。包含以下测试用例与工具：

make_detector(**overrides)：创建被测实例；将配置字段（如 swing_L, min_separation, cooldown_secs, warmup_min, max_lag, use_fusion, cons_min, *_weight, *_threshold）按 小 swing_L / 小分离 / 关闭暖启动 的便于出信号的设置覆盖（如 swing_L=2, min_separation=1, warmup_min=0, cooldown_secs=0.5）。若无法直接实例化 DivergenceConfig，则实例化后通过属性写入。

feed_series(...)：按序推送 (ts, price, z_ofi, z_cvd, [fusion_score], [consistency])，返回最后一次非空事件与 detector 的 stats。

输入校验：构造包含 NaN/inf 和负 ts/price 的样本，断言无事件输出且统计不异常。

四类背离判定（构造可控枢轴）：

通过三角波价格序列制造 LL/HL/HH/LH 的成对枢轴，指标序列分别设置为 HL/LH/LL/HH，以覆盖：

价格LL + 指标HL → bull_div

价格HH + 指标LH → bear_div

价格HL + 指标LL → hidden_bull

价格LH + 指标HH → hidden_bear

每类至少断言：event["type"] 正确、event["channel"] 为对应通道（OFI 或 CVD）、event["pivots"] 的 A/B 值与构造一致、event["score"] ≥ weak_threshold、event 含必需字段集。

冷却与去重：

设置 cooldown_secs=2，同一 (event_type, channel) 触发两次且间隔 <2s：第二次应被抑制（通过 stats["suppressed_by_reason"]["cooldown"] 或等效字段 +1 断言）。

同一 (idx_a, idx_b) 配对在固定窗口反复 update()：只允许首次产出事件，后续均无事件。

Fusion 一致性：

开启 use_fusion=True，提供 fusion_score 与两个不同 consistency（如 0.2 与 0.8），其余输入相同；断言后一者的 score 更高，且 event["channel"] == "fusion"。

值域裁剪与健壮性：

注入超界值（如 fusion_score=9、z_ofi=12、z_cvd=-11），断言不会“爆表”（事件得分不过度膨胀）且事件仍可正常判断；再注入 float("nan")/float("inf")，断言不产出事件。

统计一致性：跑完一段样本，断言 events_total == sum(events_by_type.values())，枢轴计数之和与总枢轴一致。

性能基准：

构造 100k 条随机但有限、可裁剪的样本（price>0），记录 time.perf_counter() 环绕 update() 的耗时分布，打印 p50/p95/p99；断言建议门槛（p99 ≤ 1ms；若 CI 环境慢，放宽到 ≤5ms，且平均 <1ms）。

测试中请固定随机种子，避免抖动；所有断言给出清晰错误消息。

依赖与运行

新建 requirements-test.txt（如不存在）并确保含：pytest, numpy（如你生成的测试需要）, pytest-benchmark（可选），statistics 使用标准库即可。

运行：

python -m pip install -r requirements-test.txt
pytest -q --maxfail=1 --disable-warnings --junitxml=reports/junit_divergence.xml


若 reports/ 目录不存在请先创建。

生成可读报告 reports/REPORT_DIVERGENCE_TEST.md

内容包含：测试环境（CPU/内存/OS/Python 版本）、被测文件的 git 简要信息（commit/改动摘要）、各用例通过/失败表、关键断言截图（文本形式即可）、性能统计（p50/p95/p99）、改进建议（若有）。

在报告末尾给出“签收判定”：是否满足验收标准；若不满足，逐条列出整改项。

输出清单

在终端最后回显绝对路径：

tests/test_divergence_detector.py

reports/junit_divergence.xml

reports/REPORT_DIVERGENCE_TEST.md

同时打印关键统计（事件总数、各类型事件计数、抑制计数、性能 p50/p95/p99）。

重要约束：

不得修改被测核心文件的业务逻辑；如需暴露最小接口用于测试（例如读取 stats），须以非侵入方式（反射或属性读取），或在测试内做兼容适配。

测试需可重复、不依赖外网，并在常见 CI 机器上稳定通过。

请开始执行，完成后把关键信息与报告路径打印出来。
