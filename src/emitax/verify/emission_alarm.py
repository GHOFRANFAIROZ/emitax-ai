from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  إنذار ارتفاع الانبعاث (طلب İsmail)
#  اتّجاهي للأعلى — عكس كشف الغشّ. ينبّه: "الانبعاث مرتفع — افحص/غيّر البِلَع".
#  قاعدة بسيطة (ليست ML): كثافة أعلى بكثير من الوسيط الطبيعي (× مضاعِف)،
#  أو تجاوز حدّ قياس مطلق اختياري.
# =========================================================

def fit_alarm_bounds(reference, cfg):
    """يحسب حدّ التحذير/الحرج لكل غاز = مضاعِف × وسيط كثافة القطاع الطبيعي."""
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in reference.columns]
    a = cfg.get("alarm", {})
    warn_m = a.get("warn_multiplier", 1.3); crit_m = a.get("crit_multiplier", 1.6)
    legal = a.get("legal_limits", {}) or {}
    d = reference[reference.get("heat_input", 0) > 0]
    bounds = {}
    for g in gases:
        inten = (d[f"{g}_mass"] / d["heat_input"]).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(inten):
            continue
        med = float(inten.median())
        warn = min(med * warn_m, legal[g]) if g in legal and legal[g] else med * warn_m
        crit = min(med * crit_m, legal[g]) if g in legal and legal[g] else med * crit_m
        bounds[g] = {"median": med, "warn": warn, "crit": crit}
    return bounds


def check_emission_alarm(declarations, bounds, cfg):
    """يفحص كل بيان: هل كثافة الانبعاث فوق حدّ التحذير/الحرج؟
    يُرجع: facility, unit, ym, gas, intensity, level (warning/critical), message."""
    if "decl_heat_input" not in declarations.columns:
        return pd.DataFrame(columns=["facility", "unit", "ym", "gas", "level"])
    d = declarations.reset_index(drop=True)
    hi = d["decl_heat_input"].to_numpy(dtype=float)
    rows = []
    for g, b in bounds.items():
        gcol = f"decl_{g}"
        if gcol not in d.columns:
            continue
        inten = np.divide(d[gcol].to_numpy(dtype=float), hi,
                          out=np.full(len(d), np.nan), where=hi > 0)
        for i in range(len(d)):
            v = inten[i]
            if not np.isfinite(v):
                continue
            if v >= b["crit"]:
                level, msg = "critical", f"{g} emisyonu cok yuksek — bilesenleri kontrol edin/degistirin"
            elif v >= b["warn"]:
                level, msg = "warning", f"{g} emisyonu yuksek — kontrol onerilir"
            else:
                continue
            rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                         "gas": g, "intensity": float(v), "level": level, "message": msg})
    return pd.DataFrame(rows)


def rollup_alarm(alarm_long):
    """طيّ إلى مستوى السجل: أعلى مستوى إنذار + الغازات المعنيّة."""
    if alarm_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "alarm_level", "alarm_gases"])
    order = {"warning": 1, "critical": 2}

    def _agg(sub):
        lvl = max(sub["level"], key=lambda x: order.get(x, 0))
        return pd.Series({"alarm_level": lvl, "alarm_gases": ",".join(sorted(sub["gas"].unique()))})
    return (alarm_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())
