"""
الدمج النهائي مع الـ LSTM — يشغّل الفحوص الأربعة (ومنها الـ LSTM عبر ONNX) + الدمج.

    python run_fusion_lstm_demo.py

يتطلّب: data/processed/lstm_ae.onnx + lstm_meta.json (من تدريب Colab)،
        و monthly_measurement.csv + hourly_clean.parquet (من run_prepare.py).
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
        print("درّبي الـ LSTM على Colab أولاً وحطّي lstm_ae.onnx + lstm_meta.json في data/processed/."); return
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
        "temporal":    rollup_temporal_lstm(detect_month_anomalies(hourly, sess, m, cfg)),  # LSTM عبر ONNX
    }
    decisions = decide(rollups, mixed, cfg)
    labeled = mixed.merge(decisions, on=["facility", "unit", "ym"], how="left")
    labeled["suspect"] = labeled["action"] != "approve"
    mm = detection_metrics(labeled["is_tampered"], labeled["suspect"])
    print("[الدمج النهائي — 4 فحوص منها LSTM(ONNX)]")
    for k in ["recall", "fp_rate", "precision", "f1"]:
        print(f"   {k:10} = {mm[k]:.3f}")
    print("\n[توزيع القرارات]")
    print(labeled.groupby(["is_tampered", "action"]).size().to_string())


if __name__ == "__main__":
    main()
