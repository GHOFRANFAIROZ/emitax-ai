from __future__ import annotations
import pandas as pd

# =========================================================
#  مقاييس الكشف (على مستوى السجل)
# =========================================================

def detection_metrics(labels: pd.Series, flags: pd.Series) -> dict:
    """labels/flags منطقية بنفس الطول. يُرجع Recall, FP-rate, Precision, F1, TP/FP/FN/TN."""
    labels = labels.astype(bool).to_numpy()
    flags = flags.astype(bool).to_numpy()
    tp = int((labels & flags).sum())
    fp = int((~labels & flags).sum())
    fn = int((labels & ~flags).sum())
    tn = int((~labels & ~flags).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(recall=recall, fp_rate=fp_rate, precision=precision, f1=f1,
                TP=tp, FP=fp, FN=fn, TN=tn)


def by_scenario(mixed: pd.DataFrame, record_flags: pd.DataFrame) -> pd.DataFrame:
    """Recall لكل سيناريو (على السجلّات المُلاعَبة فقط) + خطّ FP من الصادقة."""
    j = mixed.merge(record_flags, on=["facility", "unit", "ym"], how="left")
    out = []
    for scen, sub in j.groupby("scenario"):
        if scen == "honest":
            fp = detection_metrics(sub["is_tampered"], sub["record_flag"])
            out.append({"scenario": "honest", "n": len(sub),
                        "recall": None, "fp_rate": fp["fp_rate"]})
        else:
            caught = sub["record_flag"].astype(bool).sum()
            out.append({"scenario": scen, "n": len(sub),
                        "recall": caught / len(sub), "fp_rate": None})
    return pd.DataFrame(out)
