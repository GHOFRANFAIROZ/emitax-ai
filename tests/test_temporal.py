from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from emitax.synth.declaration_gen import generate_honest
from emitax.verify.temporal_check import check_temporal, rollup_temporal

CFG = {
    "columns": {"gases": {"CO2": {"mass": "CO2 Mass (short tons)"},
                          "SO2": {"mass": "SO2 Mass (lbs)"},
                          "NOx": {"mass": "NOx Mass (lbs)"}}},
    "declaration": {"honest_noise_std": 0.0, "seed": 1},
    "temporal": {"mad_k": 3.5, "min_months": 3, "drift_slope_frac": 0.03},
}


def _measurement(n=8):
    hi = np.linspace(1000, 1300, n)
    return pd.DataFrame({
        "facility": ["A"] * n, "unit": ["U1"] * n,
        "ym": [f"2025-{i+1:02d}" for i in range(n)],
        "CO2_mass": hi * 0.1, "SO2_mass": hi * 0.01, "NOx_mass": hi * 0.02,
        "heat_input": hi,
    })


def test_honest_history_no_flags():
    h = generate_honest(_measurement(), CFG)
    r = rollup_temporal(check_temporal(h, CFG))
    assert (r["temporal_flag"] == False).all() if len(r) else True


def test_single_month_drop_is_flagged():
    m = _measurement(); h = generate_honest(m, CFG)
    h.loc[3, "decl_CO2"] *= 0.5          # هبوط مفاجئ بشهر واحد
    r = rollup_temporal(check_temporal(h, CFG))
    hit = r[(r["ym"] == "2025-04") & (r["temporal_flag"])]
    assert len(hit) == 1


def test_creeping_drift_is_flagged():
    m = _measurement(12); h = generate_honest(m, CFG)
    for j, ix in enumerate(h.index):
        for g in ["CO2", "SO2", "NOx"]:
            h.loc[ix, f"decl_{g}"] *= (1.0 - 0.05 * j)   # خفض زاحف
    r = rollup_temporal(check_temporal(h, CFG))
    assert r["fired_rules"].str.contains("creeping_drift").any()