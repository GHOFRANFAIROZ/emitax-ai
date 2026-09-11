from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  Sektörel akran kontrolü (3): beyan edilen emisyon yoğunluğunun
#  sektör dağılımıyla (benzer tesisler) karşılaştırılması. Sektörden anormal düşük yoğunluk = şüpheli.
#  Sektör referansı bir referans veriden [7] gelir (burada: tüm birimlerdeki ölçümden).
#  Benzersiz avantajı: birimin tüm aylarına yayılmış düzenli azaltmayı yakalar (zamansal kontrolün kaçırdığı).
#  Tesis sayısından bağımsız — sayı arttıkça referans güçlenir.
# =========================================================

def build_peer_pool(reference: pd.DataFrame, cfg: dict) -> dict:
    """Her gaz için sektör yoğunluğu dağılımını referans veriden kurar (tüm birimlerdeki ölçüm).

    yoğunluk = mass / heat_input. Her gaz için döndürür: median + alt sınır (düşük yüzdelik).
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
    """Beyan yoğunluğunu sektörün alt sınırıyla karşılaştırır. Uzun tablo: gas, intensity, sector_low, flag, reason."""
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
                         "reason": f"{g} yoğunluğu sektörün altında" if flagged else ""})
    return pd.DataFrame(rows)


def rollup_peer(peer_long: pd.DataFrame) -> pd.DataFrame:
    """Kayıt düzeyine indirger: herhangi bir gaz sektörün altındaysa şüpheli."""
    if peer_long.empty:
        return pd.DataFrame(columns=["facility", "unit", "ym", "peer_flag", "flagged_gases"])

    def _agg(sub: pd.DataFrame) -> pd.Series:
        flagged = sub.loc[sub["flag"], "gas"].tolist()
        return pd.Series({"peer_flag": bool(len(flagged) > 0),
                          "flagged_gases": ",".join(sorted(set(flagged)))})
    return (peer_long.groupby(["facility", "unit", "ym"], dropna=False)
            .apply(_agg, include_groups=False).reset_index())
