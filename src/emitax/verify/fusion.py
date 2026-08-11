from __future__ import annotations
import pandas as pd

# =========================================================
#  الدمج ودرجة الثقة (٠–١٠٠) وتدفّق القرار
#  يوحّد مخرجات الفحوص الأربعة → درجة ثقة → قرار (اعتماد/توضيح/تدقيق بشري).
#  ليس صندوقاً أسود: كل خصم مرتبط بفحص وسبب مقروء.
#  السجل (الدرجة + الأسباب) هو الدليل الذي يُسجَّل على البلوكتشين — مستقلّ عن قرار الإنسان.
# =========================================================

KEYS = ["facility", "unit", "ym"]

# اسم الفحص → (عمود العلَم في rollup الخاص به، عمود سبب اختياري)
CHECK_SPEC = {
    "measurement": ("record_flag", "flagged_gases"),
    "physics":     ("physics_flag", "fired_rules"),
    "peer":        ("peer_flag", "flagged_gases"),
    "temporal":    ("temporal_flag", "fired_rules"),
}

REASON_AR = {
    "measurement": "البيان أقلّ من القياس",
    "physics": "غير معقول فيزيائياً",
    "peer": "أقلّ من القطاع",
    "temporal": "انحراف عن تاريخ الوحدة",
}


def combine_checks(rollups: dict[str, pd.DataFrame], base: pd.DataFrame) -> pd.DataFrame:
    """يضمّ rollups الفحوص إلى إطار أساس (كل سجلّات البيان) ويملأ الغائب False.

    rollups: {"measurement": df, "physics": df, "peer": df, "temporal": df} (أي مجموعة فرعية).
    base: إطار فيه على الأقلّ الأعمدة KEYS (عادةً البيان نفسه).
    """
    out = base[KEYS].drop_duplicates().copy()
    for name, df in rollups.items():
        flag_col = CHECK_SPEC[name][0]
        cols = KEYS + [c for c in [flag_col, CHECK_SPEC[name][1]] if c in df.columns]
        if df is None or df.empty or flag_col not in df.columns:
            out[flag_col] = False
            continue
        out = out.merge(df[cols], on=KEYS, how="left")
        out[flag_col] = out[flag_col].fillna(False).astype(bool)
    return out


def fuse(combined: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """يحسب درجة الثقة والقرار والسبب لكل سجل. دالة pure."""
    fcfg = cfg.get("fusion", {})
    weights = fcfg.get("weights", {"measurement": 45, "physics": 30, "peer": 25, "temporal": 20})
    hi = fcfg.get("high_threshold", 85)
    med = fcfg.get("medium_threshold", 55)

    rows = []
    for _, r in combined.iterrows():
        penalty = 0
        fired = []
        for name, (flag_col, _reason_col) in CHECK_SPEC.items():
            if bool(r.get(flag_col, False)):
                penalty += weights.get(name, 0)
                fired.append(name)
        trust = max(0, min(100, 100 - penalty))
        if trust >= hi:
            action = "approve"          # اعتماد آلي → بلوكتشين
        elif trust >= med:
            action = "request_docs"     # طلب توضيح/وثيقة (١٥ يوماً)
        else:
            action = "human_review"     # طابور تدقيق بشري
        reason = "؛ ".join(REASON_AR[c] for c in fired) if fired else "لا تعارض"
        rows.append({**{k: r[k] for k in KEYS},
                     "trust_score": trust, "action": action,
                     "fired_checks": ",".join(fired), "n_checks": len(fired),
                     "reason": reason})
    return pd.DataFrame(rows)


def decide(rollups: dict[str, pd.DataFrame], base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """اختصار: دمج + تسجيل درجة الثقة والقرار في خطوة واحدة."""
    return fuse(combine_checks(rollups, base), cfg)