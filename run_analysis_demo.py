"""
Aşama 9 (ek) — Analiz katmanı: sistemin doğrulama sonuçlarından
Türkçe etiketli istatistikler ve grafikler üretir (jüri sunumu için).

Girdi:  data/processed/trust_decisions.csv   (run_fusion_demo.py çıktısı)
Çıktı:  docs/analysis/*.png  +  data/processed/analysis_summary.csv

Not: 'is_tampered' sütunu varsa performans matrisi de çizilir; yoksa
o grafik atlanır (gerçek/canlı veride etiket olmayabilir).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
from emitax.analysis import stats as S
from emitax.analysis import plots as P

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "analysis"


def main():
    src = PROC / "trust_decisions.csv"
    if not src.exists():
        print(f"[Hata] {src} bulunamadı. Önce: python run_fusion_demo.py")
        return
    dec = pd.read_csv(src)

    # etiket varsa değerlendirme kümesinden ekle (opsiyonel)
    labeled = dec.copy()
    ev = PROC / "check_records.csv"
    if "is_tampered" not in dec.columns and ev.exists():
        try:
            lab = pd.read_csv(ev)[["facility", "unit", "ym", "is_tampered"]]
            labeled = dec.merge(lab, on=["facility", "unit", "ym"], how="left")
        except Exception:
            pass

    # ---- 1) metin özet ----
    summ = S.decision_summary(dec)
    print("[Genel özet]")
    for k, v in summ.items():
        print(f"   {k:20s} = {v}")

    print("\n[Karar dağılımı]")
    print(S.action_distribution(dec).to_string(index=False))

    print("\n[Kontrol tetiklenme oranları]")
    print(S.check_fire_rates(dec).to_string(index=False))

    print("\n[Birim başına güven]")
    print(S.trust_by_unit(dec).to_string(index=False))

    perf = S.confusion_if_labeled(labeled)
    if perf:
        print("\n[Performans (etiketli)]")
        for k, v in perf.items():
            print(f"   {k:10s} = {v}")

    # ---- 2) grafikler ----
    paths = [
        P.plot_action_distribution(dec, OUT),
        P.plot_trust_histogram(dec, OUT),
        P.plot_check_fire_rates(dec, OUT),
        P.plot_trust_by_unit(dec, OUT),
        P.plot_monthly_trust(dec, OUT),
    ]
    conf = P.plot_confusion(labeled, OUT)
    if conf:
        paths.append(conf)

    # ---- 3) özet tabloyu kaydet ----
    S.action_distribution(dec).to_csv(PROC / "analysis_summary.csv", index=False)

    print(f"\n[Kaydedildi] {len(paths)} grafik -> {OUT}/")
    for p in paths:
        print(f"   - {p.name}")
    print(f"[Kaydedildi] {PROC/'analysis_summary.csv'}")
    print("\nAnaliz katmanı tamamlandı.")


if __name__ == "__main__":
    main()
