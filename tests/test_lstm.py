from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")   # يتخطّى محلياً إن لم يوجد torch، ويعمل على Colab
from emitax.models.lstm_ae import build_model, train, reconstruction_error
from emitax.models.hourly_windows import make_windows
import pandas as pd


def test_model_shape_roundtrip():
    m = build_model(n_features=4, seq_len=24, latent=16, hidden=32)
    x = torch.randn(5, 24, 4)
    y = m(x)
    assert tuple(y.shape) == (5, 24, 4)          # المخرج = المدخل شكلاً


def test_train_reduces_loss_and_error_shape():
    X = np.random.randn(64, 24, 4).astype("float32")
    m = build_model(4, 24, 16, 32)
    hist = train(m, X, epochs=3, batch=32)
    assert hist[-1] <= hist[0] * 1.5             # لا ينفجر
    err = reconstruction_error(m, X)
    assert err.shape == (64,) and np.isfinite(err).all()


def test_windows_respect_gaps():
    ts = pd.date_range("2025-01-01", periods=30, freq="h").to_list()
    ts = ts[:15] + [t + pd.Timedelta(hours=10) for t in ts[15:]]   # فجوة بالمنتصف
    d = pd.DataFrame({"facility": ["A"] * 30, "unit": ["U1"] * 30, "ts": ts,
                      "CO2_mass": np.arange(30.0)})
    W = make_windows(d, ["CO2_mass"], seq_len=10, stride=5)
    assert W.shape[1:] == (10, 1)                # لا تعبر النوافذ الفجوة
