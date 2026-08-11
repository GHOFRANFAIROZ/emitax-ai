from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from emitax.synth.declaration_gen import generate_honest
from emitax.synth.tamper_gen import apply_scenario
from emitax.verify.peer_iforest import build_peer_pool, check_peer, rollup_peer

CFG = {
    "columns": {"gases": {"CO2": {"mass": "CO2 Mass (short tons)"},
                          "SO2": {"mass": "SO2 Mass (lbs)"},
                          "NOx": {"mass": "NOx Mass (lbs)"}}},
    "declaration": {"honest_noise_std": 0.0, "seed": 1},
    "peer": {"low_quantile": 0.05, "min_peers": 2},
}
GASES = ["CO2", "SO2", "NOx"]


def _measurement():
    # وحدتان متشابهتان (قطاع) بكثافة CO2 ثابتة ~0.1
    hi = np.array([1000.0, 1200.0, 1100.0, 900.0, 1000.0, 1300.0])
    return pd.DataFrame({
        "facility": ["A"] * 6, "unit": ["U1", "U1", "U1", "U2", "U2", "U2"],
        "ym": ["2025-01", "2025-02", "2025-03"] * 2,
        "CO2_mass": hi * 0.1, "SO2_mass": hi * 0.01, "NOx_mass": hi * 0.02,
        "heat_input": hi,
    })


def test_pool_has_bounds():
    p = build_peer_pool(_measurement(), CFG)
    assert p["n_units"] == 2
    assert p["gases"]["CO2"]["low"] > 0


def test_honest_within_sector():
    m = _measurement(); p = build_peer_pool(m, CFG)
    h = generate_honest(m, CFG)
    r = rollup_peer(check_peer(h, p, CFG))
    assert r["peer_flag"].sum() == 0


def test_uniform_underreport_caught_by_peer():
    m = _measurement(); p = build_peer_pool(m, CFG)
    h = generate_honest(m, CFG)
    # تخفيض منتظم لكل صفوف U1 → كثافته تحت القطاع
    idx = h.index[h["unit"] == "U1"].tolist()
    t = apply_scenario(h.copy(), idx, GASES, "scale_down", {"factor": 0.6}, np.random.default_rng(0))
    r = rollup_peer(check_peer(t, p, CFG))
    caught = r[r["unit"] == "U1"]["peer_flag"].mean()
    assert caught == 1.0