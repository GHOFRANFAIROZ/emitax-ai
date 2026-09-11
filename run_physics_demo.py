"""
Aşama 4 — Fiziksel kontrolün gösterimi (ölçüme ihtiyaç duymadan manipülasyonu yakalar).

    python run_physics_demo.py

Parametreleri ölçümden öğrenir, dürüst + manipüle beyan üretir ve yalnızca fiziksel kontrolü çalıştırır.
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
        print("önce run_prepare.py çalıştırın."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    params = fit_params(measurement, cfg)
    print(f"[Öğrenilen parametreler] CO₂/HeatInput katsayısı = {params['co2_hi_factor']:.4f}")
    for g, b in params["ratios"].items():
        print(f"   {g}/CO₂ oranı: ortanca={b['center']:.4f}  alt sınır={b['low']:.4f}")
    print()

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)

    phys = rollup_physics(check_physics(mixed, params, cfg))
    labeled = mixed.merge(phys, on=["facility", "unit", "ym"], how="left").fillna({"physics_flag": False})

    mm = detection_metrics(labeled["is_tampered"], labeled["physics_flag"])
    print("[Yalnızca fiziksel kontrol]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    # Ölçümsüz yakalar: hepsine bir azaltma uygula ve yalnızca fiziği çalıştır
    print("[scale_down taraması yalnızca fizik ile]")
    for factor in [0.95, 0.90, 0.70, 0.50]:
        t = apply_scenario(honest.copy(), honest.index.tolist(), gases,
                           "scale_down", {"factor": factor}, np.random.default_rng(0))
        pr = rollup_physics(check_physics(t, params, cfg))
        r = detection_metrics(pd.Series([True] * len(pr)), pr["physics_flag"])
        print(f"   factor={factor:.2f}  Recall={r['recall']:.2f}")
    print("\nNot: fizik, azaltmayı emisyon katsayısı tabanıyla, oran bozulmasını da oran sınırlarıyla yakalar.")


if __name__ == "__main__":
    main()