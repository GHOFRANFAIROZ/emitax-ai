from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  الفحص الفيزيائي (٢): معقولية البيان بذاته — لا يحتاج القياس
#  (أ) أرضية معامل الانبعاث: الوقود/الطاقة المُصرَّحة لا يمكن أن تنتج CO₂ أقلّ من حدّها.
#  (ب) نِسَب الغازات: خفض غاز واحد يكسر النسبة ويترك أثراً.
#  المعايير تُتعلَّم من القياس (data-driven) → مرنة مع أي داتا.
# =========================================================

def _ipcc_co2_factor(measurement: pd.DataFrame, cfg: dict) -> float:
    """معامل انبعاث CO₂ مرجعي من IPCC 2006 حسب نوع الوقود (يُطبَّق لا يُتعلَّم)."""
    ipcc = cfg.get("physics_ipcc", {})
    factors = ipcc.get("co2_factor_by_fuel", {})
    fuel = ""
    if "fuel_type" in measurement.columns and measurement["fuel_type"].notna().any():
        fuel = str(measurement["fuel_type"].dropna().iloc[0])
    for name, val in factors.items():
        if name.lower() in fuel.lower():
            return float(val)
    return float(ipcc.get("default_co2_factor", 0.1049))


def fit_params(measurement: pd.DataFrame, cfg: dict) -> dict:
    """معامل CO₂ من IPCC 2006 (مُطبَّق) + الحدود الدنيا لنِسَب الغازات (من الداتا)."""
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in measurement.columns]
    d = measurement[measurement.get("heat_input", 0) > 0].copy()

    co2_factor = _ipcc_co2_factor(measurement, cfg)   # IPCC 2006، ليس متعلَّماً

    ql = cfg.get("physics", {}).get("ratio_low_quantile", 0.05)
    ratios = {}
    if "CO2_mass" in d:
        base = d[d["CO2_mass"] > 0]
        for g in gases:
            if g == "CO2":
                continue
            r = base[f"{g}_mass"] / base["CO2_mass"]
            ratios[g] = {"center": float(r.median()), "low": float(r.quantile(ql))}
    return {"co2_hi_factor": co2_factor, "ratios": ratios}


def check_physics(declarations: pd.DataFrame, params: dict, cfg: dict) -> pd.DataFrame:
    """يطبّق الفحص الفيزيائي على البيان. يُرجع جدولاً طويلاً: rule, gas, value, expected, flag, reason."""
    tol = cfg.get("physics", {}).get("rel_tolerance", 0.10)
    rows = []
    d = declarations.reset_index(drop=True)

    # (أ) أرضية معامل الانبعاث لـ CO₂ من الطاقة المُصرَّحة
    if params.get("co2_hi_factor") and "decl_heat_input" in d.columns and "decl_CO2" in d.columns:
        expected = d["decl_heat_input"].to_numpy() * params["co2_hi_factor"]
        declared = d["decl_CO2"].to_numpy()
        rel = np.divide(declared - expected, expected,
                        out=np.zeros_like(expected), where=expected > 0)
        for i in range(len(d)):
            rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                         "rule": "co2_emission_floor", "gas": "CO2",
                         "value": declared[i], "expected": expected[i],
                         "flag": bool(rel[i] < -tol),
                         "reason": "CO₂ المُصرَّح أقلّ من حدّ الوقود" if rel[i] < -tol else ""})

    # (ب) نِسَب الغازات إلى CO₂
    for g, band in params.get("ratios", {}).items():
        gcol = f"decl_{g}"
        if gcol not in d.columns or "decl_CO2" not in d.columns:
            continue
        co2 = d["decl_CO2"].to_numpy()
        ratio = np.divide(d[gcol].to_numpy(), co2, out=np.full(len(d), np.nan), where=co2 > 0)
        for i in range(len(d)):
            low = band["low"]
            flagged = bool(np.isfinite(ratio[i]) and ratio[i] < low)
            rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                         "rule": f"ratio_{g}_to_CO2", "gas": g,
                         "value": ratio[i], "expected": low,
                         "flag": flagged,
                         "reason": f"نسبة {g}/CO₂ أقلّ من الحدّ" if flagged else ""})

    # (ج) عدم السالبية
    for g in [c[5:] for c in d.columns if c.startswith("decl_") and c != "decl_heat_input"]:
        vals = d[f"decl_{g}"].to_numpy()
        for i in range(len(d)):
            if vals[i] < 0:
                rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                             "rule": "nonneg", "gas": g, "value": vals[i], "expected": 0.0,
                             "flag": True, "reason": "قيمة سالبة"})

    return pd.DataFrame(rows)


def rollup_physics(physics_long: pd.DataFrame) -> pd.DataFrame:
    """طيّ نتائج الفحص الفيزيائي إلى مستوى السجل: مشبوه لو أي قاعدة عُلِّمت."""
    def _agg(sub: pd.DataFrame) -> pd.Series:
        fired = sub.loc[sub["flag"], "rule"].tolist()
        return pd.Series({"physics_flag": bool(len(fired) > 0),
                          "fired_rules": ",".join(sorted(set(fired)))})
    return (physics_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())