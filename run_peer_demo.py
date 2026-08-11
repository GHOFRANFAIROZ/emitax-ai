"""
المرحلة ٦ — عرض فحص الأقران القطاعي.

    python run_peer_demo.py

يبني مرجع القطاع من القياس، يولّد بياناً صادقاً + مُلاعَباً، ويشغّل فحص الأقران وحده.
كما يوضّح ميزته: يمسك التخفيض المنتظم عبر كل الأشهر (الذي يفوت الفحص الزمني).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set, apply_scenario
from emitax.verify.peer_iforest import build_peer_pool, check_peer, rollup_peer
from emitax.verify.temporal_check import check_temporal, rollup_temporal
from emitax.eval.metrics import detection_metrics


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("شغّلي run_prepare.py أولاً."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    pool = build_peer_pool(measurement, cfg)
    print(f"[مرجع القطاع] وحدات: {pool['n_units']}")
    for g, b in pool["gases"].items():
        print(f"   {g}: وسيط={b['median']:.4f}  حدّ أدنى={b['low']:.4f}")
    print()

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)
    pr = rollup_peer(check_peer(mixed, pool, cfg))
    labeled = mixed.merge(pr, on=["facility", "unit", "ym"], how="left")
    labeled["peer_flag"] = labeled["peer_flag"].fillna(False)
    mm = detection_metrics(labeled["is_tampered"], labeled["peer_flag"])
    print("[فحص الأقران وحده — تلاعب متفرّق]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    # الميزة الفريدة: تخفيض منتظم عبر كل أشهر الوحدة
    uni = honest.copy()
    u = uni["unit"].unique()[0]
    idx = uni.index[uni["unit"] == u].tolist()
    uni = apply_scenario(uni, idx, gases, "scale_down", {"factor": 0.7}, np.random.default_rng(0))
    urecs = uni[uni["unit"] == u][["facility", "unit", "ym"]]

    # ملاحظة: rollup لا يُرجع إلّا الصفوف المُعلَّمة → نضمّها لكل صفوف الوحدة ونملأ الباقي False
    pm = urecs.merge(rollup_peer(check_peer(uni, pool, cfg)), on=["facility", "unit", "ym"], how="left")
    peer_caught = pm["peer_flag"].fillna(False).mean()
    tm = urecs.merge(rollup_temporal(check_temporal(uni, cfg)), on=["facility", "unit", "ym"], how="left")
    temp_caught = tm["temporal_flag"].fillna(False).mean()
    print(f"[تخفيض منتظم 30% على كل أشهر {u}]")
    print(f"   الأقران يكشف: {peer_caught:.2f}   |   الزمني يكشف: {temp_caught:.2f}")
    print("   → الأقران يمسك التخفيض المنتظم لأنّه مرجع خارجي (القطاع)، والزمني ذاتي المرجع فيفوته.")


if __name__ == "__main__":
    main()