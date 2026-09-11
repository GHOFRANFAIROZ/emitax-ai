from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  Ana kontrol (1): beyanın ölçümle karşılaştırılması
#  gap = beyan − ölçülen ; toleransı aşan negatif fark = şüpheli azaltma
# =========================================================

def _measured_wide(measurement: pd.DataFrame, gases: list[str]) -> pd.DataFrame:
    keep = ["facility", "unit", "ym"] + [f"{g}_mass" for g in gases if f"{g}_mass" in measurement.columns]
    return measurement[keep].copy()


def run_check(declarations: pd.DataFrame, measurement: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Her beyanı ilgili ölçümle karşılaştırır ve uzun tablo döndürür (her gaz için bir satır):

    facility, unit, ym, gas, measured, declared, gap, rel_gap, flag
    flag=True olur, rel_gap < -tolerance olduğunda (ölçülenden belirgin şekilde az beyan). pure fonksiyon.
    """
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in measurement.columns]
    tol = cfg.get("check", {}).get("rel_tolerance", 0.05)
    m = _measured_wide(measurement, gases)
    j = declarations.merge(m, on=["facility", "unit", "ym"], how="inner")

    rows = []
    for g in gases:
        dcol, mcol = f"decl_{g}", f"{g}_mass"
        if dcol not in j.columns or mcol not in j.columns:
            continue
        measured = j[mcol].to_numpy(dtype=float)
        declared = j[dcol].to_numpy(dtype=float)
        gap = declared - measured
        rel = np.divide(gap, measured, out=np.zeros_like(gap), where=measured > 0)
        sub = pd.DataFrame({
            "facility": j["facility"], "unit": j["unit"], "ym": j["ym"], "gas": g,
            "measured": measured, "declared": declared, "gap": gap, "rel_gap": rel,
            "flag": rel < -tol,
        })
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def rollup_records(check_long: pd.DataFrame) -> pd.DataFrame:
    """Gaz sonuçlarını kayıt düzeyine indirger: herhangi bir gaz işaretliyse kayıt şüpheli.

    Döndürür: facility, unit, ym, record_flag, flagged_gases, worst_rel_gap.
    """
    def _agg(sub: pd.DataFrame) -> pd.Series:
        flagged = sub.loc[sub["flag"], "gas"].tolist()
        return pd.Series({
            "record_flag": bool(len(flagged) > 0),
            "flagged_gases": ",".join(flagged),
            "worst_rel_gap": sub["rel_gap"].min(),
        })

    return (
        check_long.groupby(["facility", "unit", "ym"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
