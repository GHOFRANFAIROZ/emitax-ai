from __future__ import annotations
import json
import numpy as np
import pandas as pd

# =========================================================
#  استدلال الفحص الزمني بالـ LSTM-AE عبر ONNX Runtime (تقرير 5.1)
#  يحمّل نموذج ONNX + معايير التعيير + العتبة (bundle)، يحسب خطأ إعادة البناء
#  للنوافذ الساعية، ويرفع علَماً زمنياً لكل (وحدة، شهر) يتجاوز العتبة.
# =========================================================

def load_bundle(onnx_path: str, meta_path: str):
    """يحمّل جلسة ONNX + (mean,std,threshold,seq_len,feats) من ملف meta JSON."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["mean"] = np.array(meta["mean"], dtype=np.float32)
    meta["std"] = np.array(meta["std"], dtype=np.float32)
    return sess, meta


def _errors(sess, windows: np.ndarray) -> np.ndarray:
    """خطأ إعادة البناء لكل نافذة عبر ONNX Runtime."""
    inp = sess.get_inputs()[0].name
    recon = sess.run(None, {inp: windows.astype(np.float32)})[0]
    return ((recon - windows) ** 2).mean(axis=(1, 2))


def detect_month_anomalies(hourly: pd.DataFrame, sess, meta: dict, cfg: dict) -> pd.DataFrame:
    """لكل (مصنع، وحدة، شهر): يبني نوافذ ساعية، يحسب خطأ إعادة البناء، ويعلّم الشهر
    إن تجاوز أقصى خطأ فيه العتبة. يُرجع: facility, unit, ym, temporal_score, temporal_flag."""
    from emitax.models.hourly_windows import unit_feature_matrix, make_windows
    d, feats = unit_feature_matrix(hourly, cfg)
    d = d.copy()
    d["ym"] = pd.to_datetime(d["ts"]).dt.to_period("M").astype(str)
    T = meta["seq_len"]; thr = meta["threshold"]
    rows = []
    for (fac, unit, ym), sub in d.groupby(["facility", "unit", "ym"]):
        if len(sub) < T:
            continue
        W = make_windows(sub.assign(facility=fac, unit=unit), feats, seq_len=T, stride=max(1, T // 4))
        if len(W) == 0:
            continue
        W = (W - meta["mean"]) / meta["std"]
        err = _errors(sess, W)
        score = float(err.max())
        rows.append({"facility": fac, "unit": unit, "ym": ym,
                     "temporal_score": score, "temporal_flag": bool(score > thr)})
    return pd.DataFrame(rows)


def rollup_temporal_lstm(df: pd.DataFrame) -> pd.DataFrame:
    """يوحّد المخرج مع بقية الفحوص (عمود temporal_flag لدمج fusion)."""
    cols = ["facility", "unit", "ym", "temporal_flag"]
    return df[cols] if not df.empty else pd.DataFrame(columns=cols)
