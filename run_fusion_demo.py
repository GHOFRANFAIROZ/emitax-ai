"""
Aşama 7 — Kapanış gösterimi: dört kontrol, tek bir güven skoru ve kararda birleştirilir.

    python run_fusion_demo.py

Dört kontrolü manipüle edilmiş beyan üzerinde çalıştırır, 0–100 güven skorunda birleştirir,
ve yazar: birleşik sistem metrikleri, üç kararın dağılımı ve gerekçeleriyle örnek kayıtlar.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set
from emitax.verify.measurement_check import run_check, rollup_records
from emitax.verify.physics_rules import fit_params, check_physics, rollup_physics
from emitax.verify.temporal_check import check_temporal, rollup_temporal
from emitax.verify.peer_iforest import fit_iforest, check_peer_iforest, rollup_peer_iforest
from emitax.verify.fusion import decide
from emitax.eval.metrics import detection_metrics


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("önce run_prepare.py çalıştırın."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)

    # dört kontrol
    rollups = {
        "measurement": rollup_records(run_check(mixed, measurement, cfg)),
        "physics":     rollup_physics(check_physics(mixed, fit_params(measurement, cfg), cfg)),
        "temporal":    rollup_temporal(check_temporal(mixed, cfg)),
        "peer":        rollup_peer_iforest(check_peer_iforest(mixed, fit_iforest(measurement, cfg), cfg)),
    }
    decisions = decide(rollups, mixed, cfg)
    labeled = mixed.merge(decisions, on=["facility", "unit", "ym"], how="left")

    # Birleşik sistem: "şüpheli" = otomatik onay dışındaki her karar
    labeled["suspect"] = labeled["action"] != "approve"
    mm = detection_metrics(labeled["is_tampered"], labeled["suspect"])
    print("[Birleşik sistem (onay dışı her karar = şüpheli)]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print(f"   TP={mm['TP']} FP={mm['FP']} FN={mm['FN']} TN={mm['TN']}\n")

    print("[Karar dağılımı]")
    print(labeled.groupby(["is_tampered", "action"]).size().to_string(), "\n")

    print("[Örnek kayıtlar: güven skoru, karar ve gerekçe]")
    cols = ["unit", "ym", "is_tampered", "trust_score", "action", "fired_checks", "reason"]
    print(labeled.sort_values("trust_score")[cols].head(10).to_string(index=False))

    outdir = Path(cfg["data"]["processed_dir"])
    # is_tampered dahil kaydet (analiz katmanı performans matrisi için)
    save_cols = ["facility","unit","ym","trust_score","action","fired_checks","n_checks","reason","is_tampered"]
    save_cols = [c for c in save_cols if c in labeled.columns]
    labeled[save_cols].to_csv(outdir / "trust_decisions.csv", index=False)
    print(f"\n[Kaydedildi] {outdir/'trust_decisions.csv'}")
    print("Aşama 7 tamamlandı. Sonraki: blokzincire bağlama (K8).")


if __name__ == "__main__":
    main()