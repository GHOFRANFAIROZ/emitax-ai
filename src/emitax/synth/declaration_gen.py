from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  مولّد البيان الشهري الصادق
#  البيان الصادق ≈ القياس + ضجيج إبلاغ صغير (تقريب/تقريب رقمي طبيعي)
# =========================================================

def gas_list(cfg: dict, monthly: pd.DataFrame) -> list[str]:
    return [g for g in cfg["columns"]["gases"] if f"{g}_mass" in monthly.columns]


def generate_honest(monthly: pd.DataFrame, cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """يبني بياناً شهرياً صادقاً لكل (مصنع، وحدة، شهر) من القياس.

    decl_<gas> = measured_<gas> * (1 + ضجيج صغير)  ثم قصّ عند صفر.
    يضيف decl_heat_input (للفحص الفيزيائي لاحقاً) و is_tampered=False.
    مبذَّر (seed) → قابل لإعادة الإنتاج.
    """
    gases = gas_list(cfg, monthly)
    std = cfg.get("declaration", {}).get("honest_noise_std", 0.01)
    base_seed = cfg.get("declaration", {}).get("seed", 42)
    rng = np.random.default_rng(base_seed if seed is None else seed)

    out = monthly[["facility", "unit", "ym"]].copy().reset_index(drop=True)
    for g in gases:
        noise = rng.normal(0.0, std, len(monthly))
        out[f"decl_{g}"] = (monthly[f"{g}_mass"].to_numpy() * (1.0 + noise)).clip(min=0.0)
    if "heat_input" in monthly.columns:
        out["decl_heat_input"] = monthly["heat_input"].to_numpy()
    out["is_tampered"] = False
    out["scenario"] = "honest"
    out["gases_tampered"] = ""
    return out
