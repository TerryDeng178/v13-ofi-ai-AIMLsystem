# -*- coding: utf-8 -*-
from v13_ofi_ai_system.src.real_ofi_calculator import RealOFICalculator, OFIConfig

def test_weights_valid():
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=5))
    assert len(calc.w) == 5
    assert abs(sum(calc.w) - 1.0) < 1e-6
    assert all(x >= 0 for x in calc.w)

def test_ofi_direction():
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=3))
    b1 = [(100.0, 5.0), (99.9, 3.0), (99.8, 2.0)]
    a1 = [(100.1, 4.0), (100.2, 3.5), (100.3, 2.5)]
    calc.update_with_snapshot(b1, a1)
    b2 = [(100.0, 6.0), (99.9, 3.0), (99.8, 2.0)]
    a2 = [(100.1, 3.6), (100.2, 3.5), (100.3, 2.5)]
    r = calc.update_with_snapshot(b2, a2)
    assert r["ofi"] > 0.0

def test_warmup_behavior():
    calc = RealOFICalculator("ETHUSDT", OFIConfig(levels=2, z_window=20))
    b = [(100.0, 1.0), (99.9, 1.0)]
    a = [(100.1, 1.0), (100.2, 1.0)]
    r = calc.update_with_snapshot(b, a)
    assert r["meta"]["warmup"] is True
    for _ in range(20):
        r = calc.update_with_snapshot(b, a)
    assert r["meta"]["warmup"] is False
    assert r["z_ofi"] is not None
