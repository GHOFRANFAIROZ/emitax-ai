"""Analiz katmanı testleri — veri gerektirmez (sentetik karar tablosu)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from emitax.analysis import stats as S


def _sample():
    # 5 beyan: 2 onay, 1 açıklama, 2 insan denetimi
    return pd.DataFrame([
        {"facility": "F", "unit": "U1", "ym": "2025-01", "trust_score": 100,
         "action": "approve", "fired_checks": "", "n_checks": 0, "is_tampered": False},
        {"facility": "F", "unit": "U1", "ym": "2025-02", "trust_score": 90,
         "action": "approve", "fired_checks": "", "n_checks": 0, "is_tampered": False},
        {"facility": "F", "unit": "U1", "ym": "2025-03", "trust_score": 70,
         "action": "request_docs", "fired_checks": "physics", "n_checks": 1, "is_tampered": False},
        {"facility": "F", "unit": "U2", "ym": "2025-01", "trust_score": 25,
         "action": "human_review", "fired_checks": "measurement,peer", "n_checks": 2, "is_tampered": True},
        {"facility": "F", "unit": "U2", "ym": "2025-02", "trust_score": 0,
         "action": "human_review", "fired_checks": "measurement,physics,peer,temporal",
         "n_checks": 4, "is_tampered": True},
    ])


def test_decision_summary_counts():
    s = S.decision_summary(_sample())
    assert s["toplam_beyan"] == 5
    assert s["otomatik_onay"] == 2
    assert s["insan_denetimi"] == 2
    assert s["aciklama_talebi"] == 1
    # şüpheli oranı = onay dışı / toplam = 3/5 = %60
    assert s["supheli_orani"] == 60.0


def test_action_distribution_sums_to_total():
    d = S.action_distribution(_sample())
    assert int(d["adet"].sum()) == 5
    assert set(d["karar"]) == {"approve", "request_docs", "human_review"}


def test_check_fire_rates_exact_match():
    fr = S.check_fire_rates(_sample()).set_index("kontrol")["tetiklenme"].to_dict()
    # measurement 2 kez, physics 2 kez, peer 2 kez, temporal 1 kez
    assert fr["measurement"] == 2
    assert fr["physics"] == 2
    assert fr["peer"] == 2
    assert fr["temporal"] == 1


def test_trust_by_unit_riskiest_first():
    t = S.trust_by_unit(_sample())
    # en düşük ortalama güvenli birim ilk sırada (U2)
    assert t.iloc[0]["unit"] == "U2"


def test_confusion_perfect_on_sample():
    m = S.confusion_if_labeled(_sample())
    # 2 manipüle (U2) -> ikisi de human_review = şüpheli -> recall 1.0
    assert m["recall"] == 1.0
    assert m["TP"] == 2 and m["FN"] == 0


def test_confusion_none_without_label():
    df = _sample().drop(columns=["is_tampered"])
    assert S.confusion_if_labeled(df) is None
