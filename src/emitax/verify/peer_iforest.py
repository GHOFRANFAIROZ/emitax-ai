from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# =========================================================
#  فحص الأقران (دعم مزدوج: الدوال القديمة لـ test_peer.py + Isolation Forest)
# =========================================================

# --- الدوال القديمة (لضمان نجاح اختبارات test_peer.py القديمة) ---
def build_peer_pool(reference: pd.DataFrame, cfg: dict) -> dict:
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
    if peer_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "peer_flag", "flagged_gases"])

    def _agg(sub: pd.DataFrame) -> pd.Series:
        flagged = sub.loc[sub["flag"], "gas"].tolist()
        return pd.Series({"peer_flag": bool(len(flagged) > 0),
                          "flagged_gases": ",".join(sorted(set(flagged)))})
    return (peer_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())


# --- دوال Isolation Forest الجديدة (لتطابق التقرير 5.2) ---

def _intensities(df: pd.DataFrame, gases: list[str], mass_suffix: str) -> pd.DataFrame:
    hi = df["heat_input"] if mass_suffix == "_mass" else df["decl_heat_input"]
    out = pd.DataFrame(index=df.index)
    for g in gases:
        col = f"{g}{mass_suffix}"
        if col in df.columns:
            out[g] = np.divide(df[col], hi, out=np.full(len(df), np.nan), where=hi > 0)
    return out


def fit_iforest(reference: pd.DataFrame, cfg: dict):
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in reference.columns]
    ic = cfg.get("peer_iforest", {})
    X = _intensities(reference[reference.get("heat_input", 0) > 0], gases, "_mass").dropna()
    model = IsolationForest(contamination=ic.get("contamination", 0.10),
                            random_state=ic.get("random_state", 42))
    model.fit(X.values)
    return {"model": model, "gases": list(X.columns),
            "medians": {g: float(X[g].median()) for g in X.columns}}


def check_peer_iforest(declarations: pd.DataFrame, fitted: dict, cfg: dict) -> pd.DataFrame:
    gases = fitted["gases"]
    if "decl_heat_input" not in declarations.columns or not gases:
        return pd.DataFrame(columns=["facility", "unit", "ym", "peer_flag"])
    d = declarations.reset_index(drop=True)
    X = _decl_intensity(d, gases)
    valid = X.notna().all(axis=1)
    pred = np.ones(len(d))
    if valid.any():
        pred[valid.values] = fitted["model"].predict(X[valid].values)
    rows = []
    for i in range(len(d)):
        outlier = pred[i] == -1
        below = any(np.isfinite(X[g].iloc[i]) and X[g].iloc[i] < fitted["medians"][g] for g in gases)
        flag = bool(outlier and below)
        rows.append({"facility": d["facility"][i], "unit": d["unit"][i], "ym": d["ym"][i],
                     "peer_flag": flag})
    return pd.DataFrame(rows)


def _decl_intensity(df: pd.DataFrame, gases: list[str]) -> pd.DataFrame:
    hi = df["decl_heat_input"].to_numpy(dtype=float)
    out = pd.DataFrame(index=df.index)
    for g in gases:
        col = f"decl_{g}"
        out[g] = np.divide(df[col].to_numpy(dtype=float), hi,
                           out=np.full(len(df), np.nan), where=hi > 0) if col in df.columns else np.nan
    return out


def rollup_peer_iforest(peer_long: pd.DataFrame) -> pd.DataFrame:
    if peer_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "peer_flag"])
    return peer_long.groupby(["facility", "unit", "ym"], dropna=False)["peer_flag"].any().reset_index()