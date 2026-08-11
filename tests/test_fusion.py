from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from emitax.verify.fusion import combine_checks, fuse, decide

CFG = {"fusion": {"weights": {"measurement": 45, "physics": 30, "peer": 25, "temporal": 20},
                  "high_threshold": 85, "medium_threshold": 55}}


def _base():
    return pd.DataFrame({"facility": ["A"] * 3, "unit": ["U1"] * 3,
                         "ym": ["2025-01", "2025-02", "2025-03"]})


def test_clean_record_approved():
    base = _base()
    rollups = {
        "measurement": pd.DataFrame({"facility": ["A"], "unit": ["U1"], "ym": ["2025-01"],
                                     "record_flag": [False]}),
    }
    out = decide(rollups, base, CFG)
    assert (out["action"] == "approve").all()
    assert (out["trust_score"] == 100).all()


def test_measurement_flag_lowers_trust_and_requests_docs():
    base = _base()
    rollups = {"measurement": pd.DataFrame({"facility": ["A"], "unit": ["U1"], "ym": ["2025-01"],
                                            "record_flag": [True], "flagged_gases": ["CO2"]})}
    out = decide(rollups, base, CFG).set_index("ym")
    assert out.loc["2025-01", "trust_score"] == 55       # 100 - 45
    assert out.loc["2025-01", "action"] == "request_docs"


def test_multiple_checks_trigger_human_review():
    base = _base()
    key = {"facility": ["A"], "unit": ["U1"], "ym": ["2025-01"]}
    rollups = {
        "measurement": pd.DataFrame({**key, "record_flag": [True], "flagged_gases": ["CO2"]}),
        "physics":     pd.DataFrame({**key, "physics_flag": [True], "fired_rules": ["co2_emission_floor"]}),
        "peer":        pd.DataFrame({**key, "peer_flag": [True], "flagged_gases": ["CO2"]}),
    }
    out = decide(rollups, base, CFG).set_index("ym")
    # 100 - 45 - 30 - 25 = 0 → human_review
    assert out.loc["2025-01", "trust_score"] == 0
    assert out.loc["2025-01", "action"] == "human_review"
    assert out.loc["2025-01", "n_checks"] == 3


def test_reason_is_not_black_box():
    base = _base()
    rollups = {"peer": pd.DataFrame({"facility": ["A"], "unit": ["U1"], "ym": ["2025-01"],
                                     "peer_flag": [True], "flagged_gases": ["SO2"]})}
    out = decide(rollups, base, CFG).set_index("ym")
    assert "القطاع" in out.loc["2025-01", "reason"]      # سبب مقروء موجود