"""
Aşama 5 — Zamansal kontrolün gösterimi (tesisin kendi geçmişiyle karşılaştırılması).

    python run_temporal_demo.py

Dürüst beyan kurar, (a) etiketli rastgele manipülasyon, (b) kademeli kayma enjekte eder,
yalnızca zamansal kontrolü çalıştırır ve metrikleri yazar.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set
from emitax.verify.temporal_check import check_temporal, rollup_temporal
from emitax.eval.metrics import detection_metrics


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("önce run_prepare.py çalıştırın."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)
    honest = generate_honest(measurement, cfg)

    # (a) her ay için manipülasyon (ani düşüş)
    mixed = make_labeled_set(honest, cfg, gases)
    tl = rollup_temporal(check_temporal(mixed, cfg))
    labeled = mixed.merge(tl, on=["facility", "unit", "ym"], how="left")
    labeled["temporal_flag"] = labeled["temporal_flag"].fillna(False)
    mm = detection_metrics(labeled["is_tampered"], labeled["temporal_flag"])
    print("[Yalnızca zamansal kontrol — dağınık aylık manipülasyon]")
    for kk in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {kk:10} = {mm[kk]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    # (b) kademeli kayma: bir birim için ay ay tedrici azaltma
    drift = honest.copy()
    u = drift["unit"].unique()[0]
    mask = drift["unit"] == u
    months = drift.loc[mask].sort_values("ym").index
    for j, ix in enumerate(months):
        f = 1.0 - 0.04 * j          # her ay ek %4 azaltma
        for g in gases:
            drift.loc[ix, f"decl_{g}"] *= f
    dl = rollup_temporal(check_temporal(drift, cfg))
    fired = dl[dl["temporal_flag"]]["fired_rules"].str.contains("creeping_drift").any()
    print(f"[Kademeli kayma] birim {u}: creeping_drift işaretlendi mi? -> {'evet' if fired else 'hayır'}")


if __name__ == "__main__":
    main()