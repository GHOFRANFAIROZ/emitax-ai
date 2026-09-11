from __future__ import annotations
import pandas as pd

# =========================================================
#  pure fonksiyonlar — diskten okumaz, test edilmesi kolay
# =========================================================

def strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarındaki boşlukları temizler."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_rename_map(cfg: dict) -> dict[str, str]:
    """Harita: CEMS'teki ham ad → mantıksal ad. Ayarlardan kurulur (esnek)."""
    c = cfg["columns"]
    m = {
        c["facility"]: "facility",
        c["facility_id"]: "facility_id",
        c["unit"]: "unit",
        c["date"]: "date",
        c["hour"]: "hour",
        c["operating_time"]: "op_time",
        c["fuel_type"]: "fuel_type",
        c["context"]["heat_input"]: "heat_input",
        c["context"]["gross_load"]: "gross_load",
    }
    for gas, cols in c["gases"].items():
        m[cols["mass"]] = f"{gas}_mass"
        if cols.get("rate"):
            m[cols["rate"]] = f"{gas}_rate"
    return m


def select_and_rename(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Yalnızca bilinen sütunları tutar ve yeniden adlandırır; fazla sütunları yok sayar.
    Farklı dosyalar arasında sütun sayısı veya sırası değişse de çalışır (boyut/şema esnekliği)."""
    m = build_rename_map(cfg)
    present = {k: v for k, v in m.items() if k in df.columns}
    missing = [k for k in m if k not in df.columns]
    return df[list(present)].rename(columns=present), missing


def coerce_numeric(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Sayısal sütunları çevirir; sayısal olmayan her değer → çökmek yerine NaN."""
    df = df.copy()
    gases = cfg["columns"]["gases"]
    num = ["op_time", "heat_input", "gross_load"]
    num += [f"{g}_mass" for g in gases]
    num += [f"{g}_rate" for g in gases if gases[g].get("rate")]
    for col in num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Date + Hour'dan bir zaman sütunu oluşturur."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
    return df


# =========================================================
#  Diskten okuyan tek yer (I/O)
# =========================================================

def load_cems(path: str, cfg: dict) -> pd.DataFrame:
    """Tek bir CEMS dosyasını okur (büyük dosyalar için parça parça okumayı destekler) → temiz mantıksal DataFrame.

    Boyutla esnek: data.chunksize ayarlarsanız dosyayı belleği taşırmadan parça parça okur.
    """
    chunksize = cfg["data"].get("chunksize")
    reader = pd.read_csv(path, chunksize=chunksize) if chunksize else [pd.read_csv(path)]
    frames = []
    for raw in reader:
        raw = strip_cols(raw)
        tidy, _missing = select_and_rename(raw, cfg)
        frames.append(tidy)
    df = pd.concat(frames, ignore_index=True)
    df = coerce_numeric(df, cfg)
    df = build_timestamp(df)
    if cfg["columns"].get("drop_all_nan", True):
        df = df.dropna(axis=1, how="all")   # Tamamen boş sütunları siler (Steam Load gibi)
    return df
