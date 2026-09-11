"""
LSTM ile nihai birleştirme — dört kontrolü (biri ONNX üzerinden LSTM) + birleştirmeyi çalıştırır.

    python run_fusion_lstm_demo.py

Gerektirir: data/processed/lstm_ae.onnx + lstm_meta.json (Colab eğitiminden),
        ve monthly_measurement.csv + hourly_clean.parquet (run_prepare.py'den).
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
from emitax.verify.peer_iforest import fit_iforest, check_peer_iforest, rollup_peer_iforest
from emitax.verify.lstm_temporal import load_bundle, detect_month_anomalies, rollup_temporal_lstm
from emitax.verify.fusion import decide
from emitax.eval.metrics import detection_metrics


def main() -> None:
    cfg = load_config("configs/default.yaml")
    proc = Path(cfg["data"]["processed_dir"])
    onnx, meta = proc / "lstm_ae.onnx", proc / "lstm_meta.json"
    if not onnx.exists() or not meta.exists():
        print("önce Colab'da LSTM eğitin ve lstm_ae.onnx + lstm_meta.json dosyalarını data/processed/ içine koyun."); return
    measurement = pd.read_csv(proc / "monthly_measurement.csv")
    hourly = pd.read_parquet(proc / "hourly_clean.parquet")
    gases = gas_list(cfg, measurement)

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)

    sess, m = load_bundle(str(onnx), str(meta))
    rollups = {
        "measurement": rollup_records(run_check(mixed, measurement, cfg)),
        "physics":     rollup_physics(check_physics(mixed, fit_params(measurement, cfg), cfg)),
        "peer":        rollup_peer_iforest(check_peer_iforest(mixed, fit_iforest(measurement, cfg), cfg)),
        "temporal":    rollup_temporal_lstm(detect_month_anomalies(hourly, sess, m, cfg)),  # ONNX üzerinden LSTM
    }
    decisions = decide(rollups, mixed, cfg)
    labeled = mixed.merge(decisions, on=["facility", "unit", "ym"], how="left")
    labeled["suspect"] = labeled["action"] != "approve"
    mm = detection_metrics(labeled["is_tampered"], labeled["suspect"])
    print("[Nihai birleştirme — 4 kontrol, biri LSTM(ONNX)]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print("\n[Karar dağılımı]")
    print(labeled.groupby(["is_tampered", "action"]).size().to_string())


if __name__ == "__main__":
    main()
