from __future__ import annotations
import pandas as pd

# =========================================================
#  Analiz katmanı (istatistik): sistemin doğrulama çıktısından
#  özet istatistikler hesaplar. Yeni bir tespit yapmaz — mevcut
#  kararları (trust_decisions) okunur özetlere ve sayımlara çevirir.
#  Tüm fonksiyonlar pure: DataFrame girer, DataFrame/dict çıkar (I/O yok).
# =========================================================

ACTIONS = ["approve", "request_docs", "human_review"]
CHECKS = ["measurement", "physics", "peer", "temporal"]


def decision_summary(decisions: pd.DataFrame) -> dict:
    """Genel özet: kaç beyan, ortalama güven skoru, karar dağılımı.

    decisions: fuse() çıktısı (facility, unit, ym, trust_score, action, fired_checks, n_checks).
    dict döndürür (rapor/başlık için).
    """
    n = len(decisions)
    out = {
        "toplam_beyan": int(n),
        "ortalama_guven": round(float(decisions["trust_score"].mean()), 1) if n else 0.0,
        "medyan_guven": round(float(decisions["trust_score"].median()), 1) if n else 0.0,
        "otomatik_onay": int((decisions["action"] == "approve").sum()),
        "aciklama_talebi": int((decisions["action"] == "request_docs").sum()),
        "insan_denetimi": int((decisions["action"] == "human_review").sum()),
    }
    out["supheli_orani"] = round(
        100.0 * (n - out["otomatik_onay"]) / n, 1) if n else 0.0
    return out


def action_distribution(decisions: pd.DataFrame) -> pd.DataFrame:
    """Karar dağılımı tablosu: her karar türünden kaç adet + yüzde."""
    n = len(decisions)
    counts = decisions["action"].value_counts()
    rows = []
    for a in ACTIONS:
        c = int(counts.get(a, 0))
        rows.append({"karar": a, "adet": c,
                     "yuzde": round(100.0 * c / n, 1) if n else 0.0})
    return pd.DataFrame(rows)


def check_fire_rates(decisions: pd.DataFrame) -> pd.DataFrame:
    """Her kontrol kaç beyanda bayrak kaldırdı (fired_checks sütunundan sayılır)."""
    n = len(decisions)
    fired = decisions["fired_checks"].fillna("").astype(str)
    rows = []
    for chk in CHECKS:
        # bir kontrol adı virgülle ayrılmış listede tam eşleşmeli
        c = int(fired.apply(lambda s: chk in s.split(",") if s else False).sum())
        rows.append({"kontrol": chk, "tetiklenme": c,
                     "oran_yuzde": round(100.0 * c / n, 1) if n else 0.0})
    return pd.DataFrame(rows)


def trust_by_unit(decisions: pd.DataFrame) -> pd.DataFrame:
    """Birim başına ortalama güven skoru + beyan sayısı (hangi tesis daha riskli?)."""
    g = (decisions.groupby(["facility", "unit"], dropna=False)
         .agg(beyan_sayisi=("trust_score", "size"),
              ortalama_guven=("trust_score", "mean"),
              min_guven=("trust_score", "min"))
         .reset_index())
    g["ortalama_guven"] = g["ortalama_guven"].round(1)
    return g.sort_values("ortalama_guven")


def monthly_trust(decisions: pd.DataFrame) -> pd.DataFrame:
    """Ay bazında ortalama güven skoru (zaman içindeki eğilim)."""
    g = (decisions.groupby("ym", dropna=False)
         .agg(ortalama_guven=("trust_score", "mean"),
              beyan_sayisi=("trust_score", "size"))
         .reset_index()
         .sort_values("ym"))
    g["ortalama_guven"] = g["ortalama_guven"].round(1)
    return g


def confusion_if_labeled(labeled: pd.DataFrame) -> dict | None:
    """Etiket (is_tampered) varsa performans metrikleri döndürür; yoksa None.

    labeled: decisions + is_tampered sütunu (değerlendirme kümesinde bulunur).
    'şüpheli' = otomatik onay dışındaki her karar.
    """
    if "is_tampered" not in labeled.columns:
        return None
    y = labeled["is_tampered"].astype(bool)
    suspect = labeled["action"] != "approve"
    tp = int((y & suspect).sum()); fp = int((~y & suspect).sum())
    fn = int((y & ~suspect).sum()); tn = int((~y & ~suspect).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": round(recall, 3), "precision": round(precision, 3),
            "fp_rate": round(fp_rate, 3), "f1": round(f1, 3)}
