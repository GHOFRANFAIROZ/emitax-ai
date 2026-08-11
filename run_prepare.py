"""
المرحلة ١ — تجهيز الداتا (شغّليها بعد وضع ملف CEMS في data/raw/).

    python run_prepare.py

يقرأ كل ملفّات CEMS، ينظّف، يجمّع شهرياً، يطبع تشخيصاً قبل/بعد، ويحفظ:
    data/processed/hourly_clean.parquet     (الساعي النظيف)
    data/processed/monthly_measurement.csv  (القياس الشهري = مرجع الفحص ١)
    data/processed/cems_availability.csv    (أي وحدة عندها قياس)
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
        print("لا ملفّات في data/raw/ — ضعي ملف CEMS هناك أولاً.")
        return

    # --- تحميل (يعمل مع ملف أو عدّة ملفّات) ---
    df = pd.concat([load_cems(f, cfg) for f in files], ignore_index=True)
    print(f"[تحميل] ملفّات: {len(files)} | صفوف: {len(df):,} | أعمدة محفوظة: {list(df.columns)}\n")

    units = df.groupby(["facility", "unit"]).size()
    print("[الوحدات] صفوف لكل وحدة:")
    print(units.to_string(), "\n")

    if "op_time" in df.columns:
        op = (df["op_time"] > 0.99).sum()
        print(f"[التشغيل] ساعات تشغيل كاملة: {op:,} / {len(df):,}  (توقّف: {(df['op_time']==0).sum():,})\n")

    # --- فحص عقلاني سريع: نسبة CO2/HeatInput على ساعات التشغيل ---
    if {"CO2_mass", "heat_input"}.issubset(df.columns):
        d = df[(df.get("op_time", 1) > 0.99) & (df["heat_input"] > 0)]
        ratio = (d["CO2_mass"] / d["heat_input"]).median()
        c = cfg["sanity"]
        ok = abs(ratio - c["co2_per_heat_input_center"]) <= c["co2_per_heat_input_tol"]
        print(f"[سلامة فيزيائية] وسيط CO2/HeatInput = {ratio:.4f} "
              f"(متوقّع ~{c['co2_per_heat_input_center']}) -> {'موافق' if ok else 'راجعي!'}\n")

    # --- التجميع الشهري (مرجع الفحص ١) ---
    monthly = aggregate_monthly(df, cfg)
    print(f"[تجميع شهري] صفوف: {len(monthly)} (وحدة × شهر). عيّنة:")
    print(monthly.head(6).to_string(index=False), "\n")

    avail = measurement_available(df, cfg)
    print("[توفّر القياس] لكل وحدة:")
    print(avail.to_string(index=False), "\n")

    # --- حفظ ---
    outdir = Path(cfg["data"]["processed_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(outdir / "hourly_clean.parquet", index=False)
        hourly_path = outdir / "hourly_clean.parquet"
    except Exception:
        df.to_csv(outdir / "hourly_clean.csv", index=False)
        hourly_path = outdir / "hourly_clean.csv"
    monthly.to_csv(outdir / "monthly_measurement.csv", index=False)
    avail.to_csv(outdir / "cems_availability.csv", index=False)
    print(f"[حفظ] {hourly_path}\n[حفظ] {outdir/'monthly_measurement.csv'}\n[حفظ] {outdir/'cems_availability.csv'}")
    print("\nالمرحلة ١ تمّت. التالي: EDA (run_eda.py) ثم مولّد البيان + الفحص ١.")


if __name__ == "__main__":
    main()
