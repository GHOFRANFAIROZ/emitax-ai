"""
المرحلة ٤ — عرض الفحص الفيزيائي (يمسك التلاعب بلا الحاجة للقياس).

    python run_physics_demo.py

يتعلّم المعايير من القياس، يولّد بياناً صادقاً + مُلاعَباً، ويشغّل الفحص الفيزيائي وحده.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set, apply_scenario
from emitax.verify.physics_rules import fit_params, check_physics, rollup_physics
from emitax.eval.metrics import detection_metrics
import numpy as np


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("شغّلي run_prepare.py أولاً."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    params = fit_params(measurement, cfg)
    print(f"[معايير مُتعلَّمة] معامل CO₂/HeatInput = {params['co2_hi_factor']:.4f}")
    for g, b in params["ratios"].items():
        print(f"   نسبة {g}/CO₂: وسيط={b['center']:.4f}  حدّ أدنى={b['low']:.4f}")
    print()

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)

    phys = rollup_physics(check_physics(mixed, params, cfg))
    labeled = mixed.merge(phys, on=["facility", "unit", "ym"], how="left").fillna({"physics_flag": False})

    mm = detection_metrics(labeled["is_tampered"], labeled["physics_flag"])
    print("[الفحص الفيزيائي وحده]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    # يكشف بلا قياس: طبّق تخفيضاً على الكلّ وشغّل الفيزياء فقط
    print("[مسح scale_down عبر الفيزياء وحدها]")
    for factor in [0.95, 0.90, 0.70, 0.50]:
        t = apply_scenario(honest.copy(), honest.index.tolist(), gases,
                           "scale_down", {"factor": factor}, np.random.default_rng(0))
        pr = rollup_physics(check_physics(t, params, cfg))
        r = detection_metrics(pd.Series([True] * len(pr)), pr["physics_flag"])
        print(f"   factor={factor:.2f}  Recall={r['recall']:.2f}")
    print("\nملاحظة: الفيزياء تمسك التخفيض عبر أرضية معامل الانبعاث، وكسر النِسَب عبر حدود النِسَب.")


if __name__ == "__main__":
    main()