# CURSOR 提示词 — Task 1.2.1：创建 OFI 计算器基础类（纯计算）
**只做纯计算，不做任何 I/O。** 完成文件：`v13_ofi_ai_system/src/real_ofi_calculator.py`；通过测试：`tests/test_real_ofi_calculator.py`。

## 规则
- 禁止修改 `binance_websocket_client.py` 与 `utils/async_logging.py`。
- 仅实现 `update_with_snapshot`；`update_with_l2_delta` 保持 `NotImplementedError`。
- 不引入第三方库（numpy/pandas 等）；使用标准库。
- 提供 docstring、类型注解；`py_compile` 与 `pytest` 必须通过。

## 定义
- 权重 w（长度 K、非负、归一化为 1）：默认 `[0.4, 0.25, 0.2, 0.1, 0.05]`（按 K 裁剪/填充）。
- Δ 同档位：`Δb_k = bids[k].qty - prev_bids[k].qty`，`Δa_k = asks[k].qty - prev_asks[k].qty`。
- OFI：`ofi = Σ w_k * (Δb_k - Δa_k)`；返回 `k_components` 明细。
- z-ofi：滚动窗口 `z_window`，前 `max(5, z_window//5)` 为 warmup。
- EMA：`ema = α*ofi + (1-α)*ema`，首帧设为 ofi。
