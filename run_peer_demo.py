"""
Aşama 6 — Sektörel akran kontrolünün gösterimi.

    python run_peer_demo.py

Ölçümden sektör referansını kurar, dürüst + manipüle beyan üretir ve yalnızca akran kontrolünü çalıştırır.
Ayrıca avantajını gösterir: tüm aylara yayılmış düzenli azaltmayı yakalar (zamansal kontrolün kaçırdığı).
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
        print("önce run_prepare.py çalıştırın."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    pool = build_peer_pool(measurement, cfg)
    print(f"[Sektör referansı] birim: {pool['n_units']}")
    for g, b in pool["gases"].items():
        print(f"   {g}: ortanca={b['median']:.4f}  alt sınır={b['low']:.4f}")
    print()

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)
    pr = rollup_peer(check_peer(mixed, pool, cfg))
    labeled = mixed.merge(pr, on=["facility", "unit", "ym"], how="left")
    labeled["peer_flag"] = labeled["peer_flag"].fillna(False)
    mm = detection_metrics(labeled["is_tampered"], labeled["peer_flag"])
    print("[Yalnızca akran kontrolü — dağınık manipülasyon]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    # Benzersiz avantaj: birimin tüm aylarına yayılmış düzenli azaltma
    uni = honest.copy()
    u = uni["unit"].unique()[0]
    idx = uni.index[uni["unit"] == u].tolist()
    uni = apply_scenario(uni, idx, gases, "scale_down", {"factor": 0.7}, np.random.default_rng(0))
    urecs = uni[uni["unit"] == u][["facility", "unit", "ym"]]

    # Not: rollup yalnızca işaretli satırları döndürür → bunları birimin tüm satırlarına birleştirir ve kalanı False doldururuz
    pm = urecs.merge(rollup_peer(check_peer(uni, pool, cfg)), on=["facility", "unit", "ym"], how="left")
    peer_caught = pm["peer_flag"].fillna(False).mean()
    tm = urecs.merge(rollup_temporal(check_temporal(uni, cfg)), on=["facility", "unit", "ym"], how="left")
    temp_caught = tm["temporal_flag"].fillna(False).mean()
    print(f"[{u} biriminin tüm aylarında düzenli %30 azaltma]")
    print(f"   akran yakalıyor: {peer_caught:.2f}   |   zamansal yakalıyor: {temp_caught:.2f}")
    print("   → akran, dış referans (sektör) olduğu için düzenli azaltmayı yakalar; zamansal ise öz-referanslı olduğundan kaçırır.")


if __name__ == "__main__":
    main()