"""
EMITAX — LSTM Autoencoder Yerel Eğitim Betiği
(EMITAX_lstm_colab.py dosyasının yerel sürümü)

Colab betiğiyle aynı eğitim mantığını kullanır; yalnızca yerel dosya yollarıyla
çalışır ve !pip / CUDA / git clone adımlarına ihtiyaç duymaz.

Modeli saatlik emisyon verileri üzerinde eğitir, anomali eşiğini hesaplar
ve aşağıdaki dosyaları üretir:

    data/processed/lstm_ae.onnx
    data/processed/lstm_meta.json

Çalıştırma:
    python train_lstm_local.py
"""

from __future__ import annotations

import sys
import os
import json
import glob
from pathlib import Path

# emitax paketinin src/ klasöründen içe aktarılabilmesini sağlar.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from emitax.config import load_config
from emitax.data.loader import load_cems
from emitax.models.hourly_windows import unit_feature_matrix, make_windows
from emitax.models.lstm_ae import (
    build_model,
    train,
    reconstruction_error,
    export_onnx,
)


def main() -> None:
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))

    # [3] Saatlik CEMS verilerini yükle.
    # data/raw/ klasöründeki tüm CSV dosyaları birlikte kullanılır.
    raw_files = glob.glob(str(ROOT / "data" / "raw" / "*.csv"))

    if not raw_files:
        raise SystemExit(
            "data/raw/ klasöründe CSV dosyası bulunamadı. "
            "Önce CEMS veri dosyasını ekleyin."
        )

    hourly = pd.concat(
        [load_cems(f, cfg) for f in raw_files],
        ignore_index=True,
    )

    print("Saatlik veri satiri:", len(hourly))

    # [4] Zaman pencerelerini oluştur, eğitim/doğrulama verisini ayır
    # ve yalnızca eğitim verisine göre normalizasyon parametrelerini hesapla.
    d, feats = unit_feature_matrix(hourly, cfg)

    W = make_windows(
        d,
        feats,
        seq_len=cfg["lstm"]["seq_len"],
        stride=cfg["lstm"]["stride"],
    )

    k = int(len(W) * 0.8)
    Xtr, Xva = W[:k], W[k:]

    mean = Xtr.reshape(-1, len(feats)).mean(0)
    std = Xtr.reshape(-1, len(feats)).std(0) + 1e-8

    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    print("Egitim/dogrulama pencereleri:", Xtr.shape, Xva.shape)

    # [5] LSTM Autoencoder modelini oluştur ve eğit.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", dev)

    model = build_model(
        len(feats),
        cfg["lstm"]["seq_len"],
        cfg["lstm"]["latent"],
        cfg["lstm"]["hidden"],
    )

    hist = train(
        model,
        Xtr,
        epochs=cfg["lstm"]["epochs"],
        lr=1e-3,
        batch=128,
        device=dev,
    )

    print("Son epoch loss:", round(hist[-1], 5))

    # [6] Normal eğitim verisinin reconstruction error dağılımından
    # anomali eşiğini hesapla.
    err_tr = reconstruction_error(model, Xtr, device=dev)

    threshold = float(
        np.percentile(
            err_tr,
            cfg["lstm"]["threshold_pct"],
        )
    )

    err_va = reconstruction_error(model, Xva, device=dev)

    print(
        f"Threshold (p{cfg['lstm']['threshold_pct']})={threshold:.4f} | "
        f"validation > threshold={(err_va > threshold).mean():.3f}"
    )

    # [7] Eğitilmiş modeli ONNX formatında ve model meta verileriyle kaydet.
    out_dir = ROOT / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)

    export_onnx(
        model.cpu(),
        cfg["lstm"]["seq_len"],
        len(feats),
        str(out_dir / "lstm_ae.onnx"),
        device="cpu",
    )

    json.dump(
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "threshold": threshold,
            "seq_len": cfg["lstm"]["seq_len"],
            "feats": feats,
        },
        open(
            out_dir / "lstm_meta.json",
            "w",
            encoding="utf-8",
        ),
    )

    print("Kaydedildi: lstm_ae.onnx + lstm_meta.json ->", out_dir)


if __name__ == "__main__":
    main()