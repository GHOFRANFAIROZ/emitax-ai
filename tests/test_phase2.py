from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from emitax.synth.declaration_gen import generate_honest
from emitax.synth.tamper_gen import apply_scenario, make_labeled_set
from emitax.verify.measurement_check import run_check, rollup_records
from emitax.eval.metrics import detection_metrics

CFG = {
    "columns": {"gases": {"CO2": {"mass": "CO2 Mass (short tons)"},
                          "SO2": {"mass": "SO2 Mass (lbs)"},
                          "NOx": {"mass": "NOx Mass (lbs)"}}},
    "declaration": {"honest_noise_std": 0.0, "seed": 1},
    "tamper": {"fraction": 0.5, "seed": 1},
    "check": {"rel_tolerance": 0.05},
}
GASES = ["CO2", "SO2", "NOx"]


def _measurement():
    return pd.DataFrame({
        "facility": ["A"] * 4, "unit": ["U1"] * 4,
        "ym": ["2025-01", "2025-02", "2025-03", "2025-04"],
        "CO2_mass": [100.0, 120.0, 110.0, 90.0],
        "SO2_mass": [10.0, 12.0, 11.0, 9.0],
        "NOx_mass": [20.0, 24.0, 22.0, 18.0],
        "heat_input": [1000.0, 1200.0, 1100.0, 900.0],
    })


def test_honest_equals_measurement_when_no_noise():
    m = _measurement()
    h = generate_honest(m, CFG)
    # بلا ضجيج → البيان = القياس، وكلّها غير مُلاعَبة
    assert np.allclose(h["decl_CO2"], m["CO2_mass"])
    assert (~h["is_tampered"]).all()


def test_honest_not_flagged_by_check():
    m = _measurement()
    h = generate_honest(m, CFG)
    records = rollup_records(run_check(h, m, CFG))
    assert records["record_flag"].sum() == 0        # لا إنذارات كاذبة على الصادق


def test_scale_down_is_caught():
    m = _measurement()
    h = generate_honest(m, CFG)
    t = apply_scenario(h.copy(), h.index.tolist(), GASES, "scale_down",
                       {"factor": 0.7}, np.random.default_rng(0))
    records = rollup_records(run_check(t, m, CFG))
    assert records["record_flag"].all()             # تخفيض 30% يُكشَف كلّه


def test_check_is_directional_over_report_not_flagged():
    m = _measurement()
    h = generate_honest(m, CFG)
    # مبالغة (أعلى من القياس) يجب ألّا تُعلَّم — نهتمّ بالتخفيض فقط
    over = apply_scenario(h.copy(), h.index.tolist(), GASES, "scale_down",
                          {"factor": 1.3}, np.random.default_rng(0))
    records = rollup_records(run_check(over, m, CFG))
    assert records["record_flag"].sum() == 0


def test_metrics_basic():
    labels = pd.Series([True, True, False, False])
    flags = pd.Series([True, False, False, False])
    mm = detection_metrics(labels, flags)
    assert mm["TP"] == 1 and mm["FN"] == 1 and mm["FP"] == 0 and mm["TN"] == 2
    assert mm["recall"] == pytest.approx(0.5)
    assert mm["fp_rate"] == pytest.approx(0.0)
