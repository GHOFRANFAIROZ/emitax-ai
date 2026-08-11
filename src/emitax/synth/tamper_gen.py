from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  مولّد التلاعب المُوسوم (اتّجاهي: تخفيض تحت الحقيقة)
#  كل سيناريو يُعدّل صفوفاً محدّدة ويعلّمها is_tampered=True + اسم السيناريو
# =========================================================

def _decl_col(gas: str) -> str:
    return f"decl_{gas}"


def apply_scenario(decl: pd.DataFrame, idx, gases: list[str], scenario: str,
                   params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """يطبّق سيناريو تلاعب على الصفوف idx والغازات gases. يُرجع نسخة معدّلة."""
    d = decl.copy()
    for g in gases:
        col = _decl_col(g)
        if col not in d.columns:
            continue
        if scenario == "scale_down":                         # ضرب × عامل < 1
            d.loc[idx, col] = d.loc[idx, col] * params["factor"]
        elif scenario == "zero_out":                          # تصفير (إخفاء أثناء التشغيل)
            d.loc[idx, col] = 0.0
        elif scenario == "flatline":                          # تثبيت على كسر من الوسيط
            d.loc[idx, col] = d[col].median() * params.get("value_frac", 0.5)
        elif scenario == "cap":                               # قصّ سقف أعلى
            ceil = d[col].quantile(params.get("q", 0.5))
            d.loc[idx, col] = d.loc[idx, col].clip(upper=ceil)
        elif scenario == "ratio_break":                       # خفض غاز واحد فقط (يكسر النِسَب)
            only = params.get("gas", g)
            if g == only:
                d.loc[idx, col] = d.loc[idx, col] * params.get("factor", 0.6)
        else:
            raise ValueError(f"سيناريو غير معروف: {scenario}")
    # وسم
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
    """يبني مجموعة تقييم مُوسومة: يُلاعب نسبة من السجلّات بسيناريوهات عشوائية، والباقي يبقى صادقاً.

    كل سجل مُلاعَب يحمل is_tampered=True + scenario. مبذَّر → قابل لإعادة الإنتاج.
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
        # لسيناريوهات كل الغازات: طبّق على الغازات كلّها؛ ratio_break على غاز واحد
        tgt_gases = gases
        d = apply_scenario(d, [row], tgt_gases, scenario, params, rng)
    return d
