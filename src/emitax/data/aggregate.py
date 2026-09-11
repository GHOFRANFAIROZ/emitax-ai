from __future__ import annotations
import pandas as pd

# =========================================================
#  Aylık toplulaştırma: beyanın kendisiyle karşılaştırıldığı nesnel ölçüm (Kontrol 1)
# =========================================================

def aggregate_monthly(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Saatliği → aylığa çevir, her (tesis, birim) için.

    Gaz ve enerji kütlelerini ay boyunca toplar (çalışma saatleri NaN=yok sayılarak toplanır).
    Sonuç = tesisin beyanıyla karşılaştıracağımız gerçek aylık «ölçüm».
    pure fonksiyon.
    """
    gases = list(cfg["columns"]["gases"])
    d = df.copy()
    d["ym"] = d["ts"].dt.to_period(cfg["aggregate"]["period"]).astype(str)

    agg: dict[str, str] = {}
    for g in gases:
        col = f"{g}_mass"
        if col in d.columns:
            agg[col] = "sum"
    if "heat_input" in d.columns:
        agg["heat_input"] = "sum"
    if "gross_load" in d.columns:
        agg["gross_load"] = "mean"
    if "op_time" in d.columns:
        agg["op_time"] = "sum"

    out = (
        d.groupby(["facility", "unit", "ym"], dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"op_time": "operating_hours"})
    )
    return out


def measurement_available(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Her (tesis, birim) için: gerçek gaz ölçümü var mı (hepsi NaN değil)? = SEÖS/CEMS var mı.

    Hangi kontrollerin geçerli olduğunu belirler: ölçüm varsa → Kontrol 1 ana; yoksa → Kontrol 2/3/4'e dayanırız.
    """
    gases = list(cfg["columns"]["gases"])
    mass_cols = [f"{g}_mass" for g in gases if f"{g}_mass" in df.columns]

    def _avail(sub: pd.DataFrame) -> bool:
        return bool(sub[mass_cols].notna().any().any())

    return (
        df.groupby(["facility", "unit"])[mass_cols]
        .apply(lambda s: bool(s.notna().any().any()))
        .rename("cems_available")
        .reset_index()
    )
