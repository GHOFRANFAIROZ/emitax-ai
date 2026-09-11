"""
Aşama 1 (ek) — EDA, görselleri docs/eda/ içine kaydeder.

    python run_eda.py

Çizer: her gaz için aylık toplamlar (mevsimsellik), gazların Heat Input'a oranı (zaman içinde),
ve korelasyon matrisi. Hepsi kaydedilen görsellerdir (etkileşimli pencere yok).
"""
from __future__ import annotations
import sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emitax.config import load_config
from emitax.data.loader import load_cems
from emitax.data.aggregate import aggregate_monthly


def main() -> None:
    cfg = load_config("configs/default.yaml")
    files = sorted(glob.glob(cfg["data"]["raw_glob"]))
    if not files:
        print("data/raw/ içinde dosya yok."); return
    df = pd.concat([load_cems(f, cfg) for f in files], ignore_index=True)
    gases = [g for g in cfg["columns"]["gases"] if f"{g}_mass" in df.columns]
    outdir = Path("docs/eda"); outdir.mkdir(parents=True, exist_ok=True)

    # 1) mevsimsellik: her birim ve her gaz için aylık toplamlar
    monthly = aggregate_monthly(df, cfg)
    for g in gases:
        col = f"{g}_mass"
        if col not in monthly.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for (fac, unit), sub in monthly.groupby(["facility", "unit"]):
            ax.plot(sub["ym"], sub[col], marker="o", label=f"{fac}-{unit}")
        ax.set_title(f"{g} — Aylık toplam (mevsimsellik)"); ax.set_xlabel("Ay"); ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(outdir / f"monthly_{g}.png", dpi=110); plt.close(fig)

    # 2) gazların Heat Input'a oranı (çalışma saatleri)
    if "heat_input" in df.columns:
        d = df[(df.get("op_time", 1) > 0.99) & (df["heat_input"] > 0)].copy()
        fig, ax = plt.subplots(figsize=(10, 4))
        for g in gases:
            if f"{g}_mass" in d.columns:
                ax.plot(d["ts"], d[f"{g}_mass"] / d["heat_input"], lw=0.4, label=f"{g}/HI")
        ax.set_title("Gazların ısı girdisine oranı (zaman serisi)"); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / "ratios_over_time.png", dpi=110); plt.close(fig)

    # 3) korelasyon matrisi
    numcols = [c for c in ["heat_input", "gross_load"] + [f"{g}_mass" for g in gases] if c in df.columns]
    corr = df[numcols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numcols))); ax.set_xticklabels(numcols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(numcols))); ax.set_yticklabels(numcols, fontsize=7)
    for i in range(len(numcols)):
        for j in range(len(numcols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Korelasyon")
    fig.tight_layout(); fig.savefig(outdir / "correlation.png", dpi=110); plt.close(fig)

    print(f"[EDA] Görseller {outdir}/ klasörüne kaydedildi (monthly_*.png, ratios_over_time.png, correlation.png)")


if __name__ == "__main__":
    main()
