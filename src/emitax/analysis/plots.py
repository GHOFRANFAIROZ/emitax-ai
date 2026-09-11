from __future__ import annotations
from pathlib import Path
import pandas as pd

# =========================================================
#  Analiz katmanı (görseller): istatistikleri jüri için Türkçe
#  etiketli grafiklere çevirir. matplotlib "Agg" arka ucu kullanılır
#  (pencere açmaz, doğrudan PNG kaydeder — sunucuda da çalışır).
# =========================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import stats as S

# jüri sunumu için sade, tutarlı renkler
_COLORS = {"approve": "#2e7d32", "request_docs": "#f9a825", "human_review": "#c62828"}
_ACTION_TR = {"approve": "Otomatik onay",
              "request_docs": "Açıklama talebi",
              "human_review": "İnsan denetimi"}
_CHECK_TR = {"measurement": "Ölçüm (K1)", "physics": "Fizik (K2)",
             "peer": "Akran (K3)", "temporal": "Zamansal (K4)"}


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    fig.tight_layout()
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_action_distribution(decisions: pd.DataFrame, outdir: Path) -> Path:
    """Karar dağılımı — pasta grafiği (otomatik onay / açıklama / insan denetimi)."""
    dist = S.action_distribution(decisions)
    dist = dist[dist["adet"] > 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(dist["adet"],
           labels=[_ACTION_TR[a] for a in dist["karar"]],
           colors=[_COLORS[a] for a in dist["karar"]],
           autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title("Karar Dağılımı", fontsize=14, weight="bold")
    return _save(fig, outdir, "01_karar_dagilimi.png")


def plot_trust_histogram(decisions: pd.DataFrame, outdir: Path) -> Path:
    """Güven skoru dağılımı — histogram + karar eşikleri (55 / 85)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(decisions["trust_score"], bins=20, range=(0, 100),
            color="#1565c0", edgecolor="white")
    ax.axvline(55, color="#f9a825", linestyle="--", linewidth=2, label="Eşik 55")
    ax.axvline(85, color="#2e7d32", linestyle="--", linewidth=2, label="Eşik 85")
    ax.set_xlabel("Güven skoru (0–100)")
    ax.set_ylabel("Beyan sayısı")
    ax.set_title("Güven Skoru Dağılımı", fontsize=14, weight="bold")
    ax.legend()
    return _save(fig, outdir, "02_guven_skoru_dagilimi.png")


def plot_check_fire_rates(decisions: pd.DataFrame, outdir: Path) -> Path:
    """Her kontrolün tetiklenme oranı — yatay çubuk grafiği."""
    fr = S.check_fire_rates(decisions)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([_CHECK_TR[c] for c in fr["kontrol"]], fr["tetiklenme"],
            color="#6a1b9a")
    for i, v in enumerate(fr["tetiklenme"]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=11)
    ax.set_xlabel("Bayrak kaldırılan beyan sayısı")
    ax.set_title("Kontrol Başına Tetiklenme", fontsize=14, weight="bold")
    return _save(fig, outdir, "03_kontrol_tetiklenme.png")


def plot_trust_by_unit(decisions: pd.DataFrame, outdir: Path) -> Path:
    """Birim başına ortalama güven skoru — çubuk grafiği (riskli birimler)."""
    t = S.trust_by_unit(decisions)
    labels = [f"{f}/{u}" for f, u in zip(t["facility"], t["unit"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c62828" if v < 55 else "#f9a825" if v < 85 else "#2e7d32"
              for v in t["ortalama_guven"]]
    ax.bar(labels, t["ortalama_guven"], color=colors)
    ax.axhline(85, color="#2e7d32", linestyle="--", linewidth=1)
    ax.axhline(55, color="#f9a825", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Ortalama güven skoru")
    ax.set_title("Birim Başına Ortalama Güven", fontsize=14, weight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, outdir, "04_birim_guven.png")


def plot_monthly_trust(decisions: pd.DataFrame, outdir: Path) -> Path:
    """Ay bazında ortalama güven skoru — çizgi grafiği (zaman eğilimi)."""
    m = S.monthly_trust(decisions)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(m["ym"].astype(str), m["ortalama_guven"],
            marker="o", color="#1565c0", linewidth=2)
    ax.axhline(85, color="#2e7d32", linestyle="--", linewidth=1, label="Onay eşiği")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Ay")
    ax.set_ylabel("Ortalama güven skoru")
    ax.set_title("Aylık Güven Skoru Eğilimi", fontsize=14, weight="bold")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, outdir, "05_aylik_guven_egilimi.png")


def plot_confusion(labeled: pd.DataFrame, outdir: Path) -> Path | None:
    """Etiketliyse: performans (TP/FP/FN/TN) — 2x2 karışıklık matrisi. Yoksa None."""
    m = S.confusion_if_labeled(labeled)
    if m is None:
        return None
    grid = [[m["TP"], m["FN"]], [m["FP"], m["TN"]]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Şüpheli dedi", "Temiz dedi"])
    ax.set_yticklabels(["Gerçekte manipüle", "Gerçekte dürüst"])
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{grid[i][j]}",
                    ha="center", va="center", fontsize=13, weight="bold")
    ax.set_title(f"Performans  (Recall={m['recall']}, FP-oranı={m['fp_rate']})",
                 fontsize=13, weight="bold")
    return _save(fig, outdir, "06_performans_matrisi.png")
