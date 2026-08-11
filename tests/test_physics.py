from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from emitax.synth.declaration_gen import generate_honest
from emitax.synth.tamper_gen import apply_scenario
from emitax.verify.physics_rules import fit_params, check_physics, rollup_physics

CFG = {
    "columns": {"gases": {"CO2": {"mass": "CO2 Mass (short tons)"},
                          "SO2": {"mass": "SO2 Mass (lbs)"},
                          "NOx": {"mass": "NOx Mass (lbs)"}}},
    "declaration": {"honest_noise_std": 0.0, "seed": 1},
    "physics": {"rel_tolerance": 0.10, "ratio_low_quantile": 0.05},
}
GASES = ["CO2", "SO2", "NOx"]


def _measurement():
    # CO2 ≈ 0.1 * heat_input بثبات (معامل انبعاث نظيف)
    hi = np.array([1000.0, 1200.0, 1100.0, 900.0, 1300.0, 800.0])
    return pd.DataFrame({
        "facility": ["A"] * 6, "unit": ["U1"] * 6,
        "ym": [f"2025-0{i+1}" for i in range(6)],
        "CO2_mass": hi * 0.1,
        "SO2_mass": hi * 0.01,
        "NOx_mass": hi * 0.02,
        "heat_input": hi,
    })


def test_fit_learns_factor_and_ratios():
    p = fit_params(_measurement(), CFG)
    assert abs(p["co2_hi_factor"] - 0.1049) < 1e-6
    assert p["ratios"]["SO2"]["center"] > 0 and p["ratios"]["NOx"]["center"] > 0


def test_physics_passes_honest():
    m = _measurement(); p = fit_params(m, CFG)
    h = generate_honest(m, CFG)
    r = rollup_physics(check_physics(h, p, CFG))
    assert r["physics_flag"].sum() == 0


def test_physics_catches_scale_down_via_co2_floor():
    m = _measurement(); p = fit_params(m, CFG)
    h = generate_honest(m, CFG)
    t = apply_scenario(h.copy(), h.index.tolist(), GASES, "scale_down",
                       {"factor": 0.6}, np.random.default_rng(0))
    r = rollup_physics(check_physics(t, p, CFG))
    assert r["physics_flag"].all()          # CO₂ تحت أرضية الوقود


def test_physics_catches_ratio_break():
    m = _measurement(); p = fit_params(m, CFG)
    h = generate_honest(m, CFG)
    # خفض SO2 فقط → يكسر نسبة SO2/CO2
    t = apply_scenario(h.copy(), h.index.tolist(), GASES, "ratio_break",
                       {"gas": "SO2", "factor": 0.3}, np.random.default_rng(0))
    r = rollup_physics(check_physics(t, p, CFG))
    assert r["physics_flag"].all()