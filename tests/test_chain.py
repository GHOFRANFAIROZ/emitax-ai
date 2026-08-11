from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emitax.chain.record import (build_record, onchain_payload, offchain_payload,
                                 verify_record)

CFG = {"columns": {"gases": {"CO2": {"mass": "CO2 Mass (short tons)"},
                             "SO2": {"mass": "SO2 Mass (lbs)"},
                             "NOx": {"mass": "NOx Mass (lbs)"}}},
       "chain": {"schema_version": "emitax-ai/1.0"}}

DEC = {"facility": "A", "unit": "U1", "ym": "2025-01", "trust_score": 55,
       "action": "request_docs", "fired_checks": "measurement", "reason": "البيان أقلّ من القياس"}
DECL = {"decl_CO2": 100.0, "decl_SO2": 10.0, "decl_NOx": 20.0, "decl_heat_input": 1000.0}
MEAS = {"CO2_mass": 130.0, "SO2_mass": 12.0, "NOx_mass": 22.0}
VA = "2026-01-01T00:00:00+00:00"


def test_record_has_hash_and_verifies():
    rec = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    assert "record_hash" in rec and len(rec["record_hash"]) == 64
    assert verify_record(rec) is True


def test_hash_is_deterministic():
    a = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    b = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    assert a["record_hash"] == b["record_hash"]


def test_tampering_breaks_hash():
    rec = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    rec2 = dict(rec); rec2["declared"] = dict(rec2["declared"]); rec2["declared"]["CO2"] *= 0.5
    assert verify_record(rec2) is False          # أي تعديل يُكشف


def test_onchain_has_no_raw_values():
    rec = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    oc = onchain_payload(rec)
    assert "declared" not in oc and "measured_summary" not in oc   # لا قيَم خام على السلسلة
    assert "record_hash" in oc and "trust_score" in oc and "action" in oc


def test_offchain_is_full():
    rec = build_record(DEC, DECL, MEAS, CFG, verified_at=VA)
    oc = offchain_payload(rec)
    assert "declared" in oc and oc["declared"]["CO2"] == 100.0