"""
Aşama 1 — Veri hazırlığı (CEMS dosyasını data/raw/ içine koyduktan sonra çalıştırın).

    python run_prepare.py

Tüm CEMS dosyalarını okur, temizler, aylık toplar, öncesi/sonrası tanılama yazar ve kaydeder:
    data/processed/hourly_clean.parquet     (temiz saatlik)
    data/processed/monthly_measurement.csv  (aylık ölçüm = Kontrol 1 referansı)
    data/processed/cems_availability.csv    (hangi birimde ölçüm var)
"""
from __future__ import annotations
import sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
from emitax.config import load_config
from emitax.data.loader import load_cems
from emitax.data.aggregate import aggregate_monthly, measurement_available


def main() -> None:
    cfg = load_config("configs/default.yaml")
    files = sorted(glob.glob(cfg["data"]["raw_glob"]))
    if not files:
        print("data/raw/ içinde dosya yok — önce oraya bir CEMS dosyası koyun.")
        return

    # --- yükleme (tek veya birden çok dosyayla çalışır) ---
    df = pd.concat([load_cems(f, cfg) for f in files], ignore_index=True)
    print(f"[Yükleme] dosya: {len(files)} | satır: {len(df):,} | tutulan sütunlar: {list(df.columns)}\n")

    units = df.groupby(["facility", "unit"]).size()
    print("[Birimler] birim başına satır:")
    print(units.to_string(), "\n")

    if "op_time" in df.columns:
        op = (df["op_time"] > 0.99).sum()
        print(f"[Çalışma] tam çalışma saati: {op:,} / {len(df):,}  (duruş: {(df['op_time']==0).sum():,})\n")

    # --- hızlı mantık kontrolü: çalışma saatlerinde CO2/HeatInput oranı ---
    if {"CO2_mass", "heat_input"}.issubset(df.columns):
        d = df[(df.get("op_time", 1) > 0.99) & (df["heat_input"] > 0)]
        ratio = (d["CO2_mass"] / d["heat_input"]).median()
        c = cfg["sanity"]
        ok = abs(ratio - c["co2_per_heat_input_center"]) <= c["co2_per_heat_input_tol"]
        print(f"[Fiziksel sağlamlık] CO2/HeatInput ortancası = {ratio:.4f} "
              f"(beklenen ~{c['co2_per_heat_input_center']}) -> {'uygun' if ok else 'kontrol et!'}\n")

    # --- aylık toplulaştırma (Kontrol 1 referansı) ---
    monthly = aggregate_monthly(df, cfg)
    print(f"[Aylık toplulaştırma] satır: {len(monthly)} (birim × ay). örnek:")
    print(monthly.head(6).to_string(index=False), "\n")

    avail = measurement_available(df, cfg)
    print("[Ölçüm mevcudiyeti] birim başına:")
    print(avail.to_string(index=False), "\n")

    # --- kaydet ---
    outdir = Path(cfg["data"]["processed_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(outdir / "hourly_clean.parquet", index=False)
        hourly_path = outdir / "hourly_clean.parquet"
    except Exception:
        df.to_csv(outdir / "hourly_clean.csv", index=False)
        hourly_path = outdir / "hourly_clean.csv"
    monthly.to_csv(outdir / "monthly_measurement.csv", index=False)
    avail.to_csv(outdir / "cems_availability.csv", index=False)
    print(f"[Kaydedildi] {hourly_path}\n[Kaydedildi] {outdir/'monthly_measurement.csv'}\n[Kaydedildi] {outdir/'cems_availability.csv'}")
    print("\nAşama 1 tamamlandı. Sonraki: EDA (run_eda.py) ardından beyan üreteci + Kontrol 1.")


if __name__ == "__main__":
    main()
