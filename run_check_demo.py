"""
المرحلة ٢+٣ — عرض الفحص الرئيسي (البيان مقابل القياس) وأوّل أرقام Recall/FP.

    python run_check_demo.py

يقرأ data/processed/monthly_measurement.csv (ناتج المرحلة ١)، يولّد بياناً صادقاً،
يحقن تلاعباً مُوسوماً، يشغّل الفحص، ويطبع/يحفظ النتائج والمقاييس.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set, apply_scenario
from emitax.verify.measurement_check import run_check, rollup_records
from emitax.eval.metrics import detection_metrics, by_scenario
import numpy as np


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("شغّلي run_prepare.py أولاً (لا يوجد monthly_measurement.csv)."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)
    print(f"[قياس] {len(measurement)} صف (وحدة×شهر) | غازات: {gases}\n")

    # 1) بيان صادق
    honest = generate_honest(measurement, cfg)
    # 2) حقن تلاعب مُوسوم
    mixed = make_labeled_set(honest, cfg, gases)
    n_t = int(mixed["is_tampered"].sum())
    print(f"[توليد] بيانات: {len(mixed)} | مُلاعَب: {n_t} | صادق: {len(mixed)-n_t}\n")

    # 3) الفحص الرئيسي
    check_long = run_check(mixed, measurement, cfg)
    records = rollup_records(check_long)
    labeled = mixed.merge(records, on=["facility", "unit", "ym"], how="left")

    # 4) المقاييس (على مستوى السجل)
    overall = detection_metrics(labeled["is_tampered"], labeled["record_flag"].fillna(False))
    print("[مقاييس عامة]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {overall[k]:.3f}")
    print(f"   TP={overall['TP']} FP={overall['FP']} FN={overall['FN']} TN={overall['TN']}\n")

    print("[حسب السيناريو]")
    print(by_scenario(mixed, records).to_string(index=False), "\n")

    # 5) مسح الحساسية: كم يكشف الفحص من التخفيض حسب شدّته
    print("[مسح شدّة التخفيض scale_down] (كل السجلّات، كل الغازات)")
    for factor in [0.95, 0.90, 0.85, 0.70, 0.50]:
        rng = np.random.default_rng(0)
        tampered = apply_scenario(honest.copy(), honest.index.tolist(), gases,
                                  "scale_down", {"factor": factor}, rng)
        cl = run_check(tampered, measurement, cfg)
        rec = rollup_records(cl)
        r = detection_metrics(pd.Series([True]*len(rec)), rec["record_flag"])
        print(f"   factor={factor:.2f}  (تخفيض {int((1-factor)*100)}%)  Recall={r['recall']:.2f}")

    outdir = Path(cfg["data"]["processed_dir"])
    check_long.to_csv(outdir / "check_results_long.csv", index=False)
    records.to_csv(outdir / "check_records.csv", index=False)
    print(f"\n[حفظ] {outdir/'check_results_long.csv'} , {outdir/'check_records.csv'}")
    print("المرحلة ٢+٣ تمّت. التالي: الفحص الفيزيائي (م٤) ثم الزمني والأقران.")


if __name__ == "__main__":
    main()
