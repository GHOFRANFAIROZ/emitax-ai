from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  Etiketli manipülasyon üreteci (yönlü: gerçeğin altına azaltma)
#  Her senaryo belirli satırları değiştirir ve is_tampered=True + senaryo adıyla etiketler
# =========================================================

def _decl_col(gas: str) -> str:
    return f"decl_{gas}"


def apply_scenario(decl: pd.DataFrame, idx, gases: list[str], scenario: str,
                   params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """idx satırlarına ve gases gazlarına bir manipülasyon senaryosu uygular. Değiştirilmiş bir kopya döndürür."""
    d = decl.copy()
    for g in gases:
        col = _decl_col(g)
        if col not in d.columns:
            continue
        if scenario == "scale_down":                         # bir faktör < 1 ile çarpma
            d.loc[idx, col] = d.loc[idx, col] * params["factor"]
        elif scenario == "zero_out":                          # sıfırlama (çalışma sırasında gizleme)
            d.loc[idx, col] = 0.0
        elif scenario == "flatline":                          # ortancanın bir kesrine sabitleme
            d.loc[idx, col] = d[col].median() * params.get("value_frac", 0.5)
        elif scenario == "cap":                               # üst tavana kırpma
            ceil = d[col].quantile(params.get("q", 0.5))
            d.loc[idx, col] = d.loc[idx, col].clip(upper=ceil)
        elif scenario == "ratio_break":                       # yalnızca tek bir gazı azaltma (oranları bozar)
            only = params.get("gas", g)
            if g == only:
                d.loc[idx, col] = d.loc[idx, col] * params.get("factor", 0.6)
        else:
            raise ValueError(f"bilinmeyen senaryo: {scenario}")
    # etiketle
    d.loc[idx, "is_tampered"] = True
    d.loc[idx, "scenario"] = scenario
    d.loc[idx, "gases_tampered"] = ",".join(gases if scenario != "ratio_break" else [params.get("gas", gases[0])])
    return d


DEFAULT_SCENARIOS = [
    ("scale_down", {"factor": 0.70}),
    ("scale_down", {"factor": 0.85}),
    ("zero_out",   {}),
    ("flatline",   {"value_frac": 0.4}),
    ("ratio_break", {"gas": "SO2", "factor": 0.5}),
    ("cap",        {"q": 0.4}),
]


def make_labeled_set(honest: pd.DataFrame, cfg: dict, gases: list[str],
                     seed: int | None = None) -> pd.DataFrame:
    """Etiketli bir değerlendirme kümesi oluşturur: kayıtların bir kısmını rastgele senaryolarla manipüle eder, kalanı dürüst kalır.

    Her manipüle kayıt is_tampered=True + scenario taşır. tohumlu → tekrarlanabilir.
    """
    tcfg = cfg.get("tamper", {})
    frac = tcfg.get("fraction", 0.35)
    rng = np.random.default_rng(tcfg.get("seed", 7) if seed is None else seed)

    d = honest.copy().reset_index(drop=True)
    n = len(d)
    n_tamper = max(1, int(round(frac * n)))
    victims = rng.choice(n, size=n_tamper, replace=False)

    for row in victims:
        scenario, params = DEFAULT_SCENARIOS[rng.integers(len(DEFAULT_SCENARIOS))]
        # Tüm gaz senaryoları için: gazların hepsine uygula; ratio_break tek bir gaza
        tgt_gases = gases
        d = apply_scenario(d, [row], tgt_gases, scenario, params, rng)
    return d
