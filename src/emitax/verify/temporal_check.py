from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  Zamansal kontrol (4): tesisin kendi geçmişiyle karşılaştırılması (self-referential)
#  Temel = emisyon yoğunluğu (emisyon / enerji), ayları boyunca.
#  Yakalar: bir ayda ani düşüş ve aşağı yönlü kademeli aylık kayma.
#  Tasarım notu: tüm aylara yayılmış düzenli azaltma burada görünmez (temel de onunla kayar)
#  → bunu dış kontroller yakalar (ölçüm/fizik/akran). Kontroller tamamlayıcıdır.
# =========================================================

def _intensity(df: pd.DataFrame, gas: str) -> np.ndarray:
    """yoğunluk = decl_<gas> / decl_heat_input (her ay için)."""
    e = df[f"decl_{gas}"].to_numpy(dtype=float)
    hi = df["decl_heat_input"].to_numpy(dtype=float)
    return np.divide(e, hi, out=np.full(len(df), np.nan), where=hi > 0)


def _robust(x: np.ndarray) -> tuple[float, float]:
    """ortanca + ortanca mutlak sapma (ölçekli MAD)."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, mad


def check_temporal(declarations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Her (tesis, birim) için: birimin ayları boyunca yoğunluğundan bir temel çizgi kurar ve işaretler:
      - low_outlier: bir ayın yoğunluğu < median - k*MAD (ani düşüş)
      - creeping_drift: yoğunlukta sürekli aşağı yönlü zamansal eğim
    Uzun tablo döndürür: facility, unit, ym, gas, rule, value, expected, flag, reason.
    """
    if "decl_heat_input" not in declarations.columns:
        return pd.DataFrame(columns=["facility", "unit", "ym", "gas", "rule", "flag"])
    gases = [c[5:] for c in declarations.columns
             if c.startswith("decl_") and c != "decl_heat_input"]
    k = cfg.get("temporal", {}).get("mad_k", 3.5)
    min_m = cfg.get("temporal", {}).get("min_months", 3)
    drift_frac = cfg.get("temporal", {}).get("drift_slope_frac", 0.03)

    rows = []
    for (fac, unit), sub in declarations.groupby(["facility", "unit"], dropna=False):
        sub = sub.sort_values("ym").reset_index(drop=True)
        if len(sub) < min_m:
            continue
        for g in gases:
            inten = _intensity(sub, g)
            med, mad = _robust(inten)
            if not np.isfinite(med) or mad == 0:
                mad = max(mad, 1e-9)
            low_th = med - k * mad
            # (a) bir ay için ani düşüş
            for i in range(len(sub)):
                if np.isfinite(inten[i]) and inten[i] < low_th:
                    rows.append({"facility": fac, "unit": unit, "ym": sub["ym"][i], "gas": g,
                                 "rule": "low_outlier", "value": inten[i], "expected": med,
                                 "flag": True, "reason": f"{g} yoğunluğu birimin geçmişinin altında"})
            # (b) kademeli kayma: belirgin negatif doğrusal eğim
            idx = np.arange(len(sub))
            ok = np.isfinite(inten)
            if ok.sum() >= min_m:
                slope = np.polyfit(idx[ok], inten[ok], 1)[0]
                if med > 0 and slope < -drift_frac * med:
                    # birim düzeyinde kayma sinyali olarak son ayı işaretleriz
                    rows.append({"facility": fac, "unit": unit, "ym": sub["ym"].iloc[-1], "gas": g,
                                 "rule": "creeping_drift", "value": float(slope), "expected": 0.0,
                                 "flag": True, "reason": f"{g} için kademeli düşüş eğilimi"})
    return pd.DataFrame(rows)


def rollup_temporal(temporal_long: pd.DataFrame) -> pd.DataFrame:
    """Kayıt düzeyine indirger: herhangi bir zamansal kural işaretlendiyse şüpheli."""
    if temporal_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "temporal_flag", "fired_rules"])

    def _agg(sub: pd.DataFrame) -> pd.Series:
        fired = sub.loc[sub["flag"], "rule"].tolist()
        return pd.Series({"temporal_flag": bool(len(fired) > 0),
                          "fired_rules": ",".join(sorted(set(fired)))})
    return (temporal_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())