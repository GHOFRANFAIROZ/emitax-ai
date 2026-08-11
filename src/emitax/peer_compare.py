from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  فحص الأقران القطاعي (٣): مقارنة كثافة الانبعاث المُصرَّحة
#  بتوزيع القطاع (مصانع مماثلة). كثافة أقلّ بشكل شاذّ من القطاع = مشبوه.
#  مرجع القطاع يأتي من داتا مرجعية [7] (هنا: من القياس عبر كل الوحدات).
#  ميزته الفريدة: يمسك التخفيض المنتظم عبر كل أشهر الوحدة (اللي يفوت الفحص الزمني).
#  محايد لعدد المصانع — كلّما زادوا، متن المرجع أكثر.
# =========================================================

def build_peer_pool(reference: pd.DataFrame, cfg: dict) -> dict:
    """يبني توزيع كثافة القطاع لكل غاز من داتا مرجعية (قياس عبر كل الوحدات).

    كثافة = mass / heat_input. يُرجع لكل غاز: median + الحدّ الأدنى (مئين منخفض).
    """
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in reference.columns]
    lq = cfg.get("peer", {}).get("low_quantile", 0.05)
    d = reference[reference.get("heat_input", 0) > 0].copy()
    pool = {"n_units": int(d.groupby(["facility", "unit"]).ngroups), "gases": {}}
    for g in gases:
        inten = (d[f"{g}_mass"] / d["heat_input"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(inten):
            pool["gases"][g] = {"median": float(inten.median()),
                                "low": float(inten.quantile(lq))}
    return pool


def check_peer(declarations: pd.DataFrame, pool: dict, cfg: dict) -> pd.DataFrame:
    """يقارن كثافة البيان بحدّ القطاع الأدنى. جدول طويل: gas, intensity, sector_low, flag, reason."""
    if pool.get("n_units", 0) < cfg.get("peer", {}).get("min_peers", 2):
        return pd.DataFrame(columns=["facility", "unit", "ym", "gas", "rule", "flag"])
    if "decl_heat_input" not in declarations.columns:
        return pd.DataFrame(columns=["facility", "unit", "ym", "gas", "rule", "flag"])

    d = declarations.reset_index(drop=True)
    hi = d["decl_heat_input"].to_numpy(dtype=float)
    rows = []
    for g, band in pool["gases"].items():
        gcol = f"decl_{g}"
        if gcol not in d.columns:
            continue
        inten = np.divide(d[gcol].to_numpy(dtype=float), hi,
                          out=np.full(len(d), np.nan), where=hi > 0)
        for i in range(len(d)):
            flagged = bool(np.isfinite(inten[i]) and inten[i] < band["low"])
            rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                         "gas": g, "rule": "peer_low_intensity",
                         "intensity": inten[i], "sector_low": band["low"],
                         "flag": flagged,
                         "reason": f"كثافة {g} أقلّ من القطاع" if flagged else ""})
    return pd.DataFrame(rows)


def rollup_peer(peer_long: pd.DataFrame) -> pd.DataFrame:
    """طيّ إلى مستوى السجل: مشبوه لو أي غاز أقلّ من القطاع."""
    if peer_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "peer_flag", "flagged_gases"])

    def _agg(sub: pd.DataFrame) -> pd.Series:
        flagged = sub.loc[sub["flag"], "gas"].tolist()
        return pd.Series({"peer_flag": bool(len(flagged) > 0),
                          "flagged_gases": ",".join(sorted(set(flagged)))})
    return (peer_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())
