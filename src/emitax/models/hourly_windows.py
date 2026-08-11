from __future__ import annotations
import numpy as np
import pandas as pd

# =========================================================
#  تجهيز النوافذ الساعية للـ LSTM-Autoencoder
#  لكل وحدة: ميزات الانبعاث الساعية (مُعايَرة) → نوافذ منزلقة لا تعبر الفجوات.
# =========================================================

def unit_feature_matrix(hourly: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """يبني ميزات الانبعاث الساعية (كتل الغازات + Heat Input) على ساعات التشغيل."""
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in hourly.columns]
    feats = [f"{g}_mass" for g in gases] + (["heat_input"] if "heat_input" in hourly.columns else [])
    d = hourly.copy()
    if "op_time" in d.columns:
        d = d[d["op_time"] > 0.99]                 # التشغيل الطبيعي فقط
    d = d.dropna(subset=feats)
    return d, feats


def standardize(train: np.ndarray, *others: np.ndarray):
    """تعيير z-score على التدريب فقط (منع تسرّب). يُرجع (scaled_train, (mean,std), scaled_others...)."""
    mean = train.mean(axis=0); std = train.std(axis=0) + 1e-8
    out = [(train - mean) / std] + [((o - mean) / std) for o in others]
    return (*out, (mean, std))


def make_windows(d: pd.DataFrame, feats: list[str], seq_len: int = 24, stride: int = 6) -> np.ndarray:
    """نوافذ منزلقة لكل وحدة، ضمن مقاطع زمنية متّصلة (لا تعبر فجوات > ساعتين)."""
    windows = []
    for _, sub in d.sort_values(["facility", "unit", "ts"]).groupby(["facility", "unit"]):
        ts = pd.to_datetime(sub["ts"]).to_numpy()
        X = sub[feats].to_numpy(dtype=float)
        # قصّ المقاطع عند الفجوات الزمنية
        gaps = np.where(np.diff(ts).astype("timedelta64[h]").astype(int) > 2)[0] + 1
        for seg in np.split(np.arange(len(X)), gaps):
            if len(seg) < seq_len:
                continue
            Xi = X[seg]
            for i in range(0, len(Xi) - seq_len + 1, stride):
                windows.append(Xi[i:i + seq_len])
    return np.asarray(windows, dtype=np.float32)
