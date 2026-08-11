# EMITAX — LSTM-Autoencoder تدريب على Colab (GPU)
# انسخي كل كتلة #%% في خلية Colab منفصلة وشغّليها بالترتيب.
# النتيجة: نموذج مدرَّب + معايير التعيير + العتبة + ملف ONNX (للتطبيق).

#%% [1] GPU + المكتبات
import torch
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
!pip -q install onnx onnxscript onnxruntime pyarrow

#%% [2] جلب الكود من GitHub (ريبو Emitax الخاص) + Drive
# !git clone https://<TOKEN>@github.com/<user>/emitax-ai.git
# from google.colab import drive; drive.mount('/content/drive')
import sys; sys.path.insert(0, "emitax-ai/src")

#%% [3] تحميل الداتا الساعية (كل مصانع الفحم في data/raw/ → hourly_clean)
import pandas as pd, numpy as np
from emitax.config import load_config
from emitax.data.loader import load_cems
import glob
cfg = load_config("emitax-ai/configs/default.yaml")
hourly = pd.concat([load_cems(f, cfg) for f in glob.glob("emitax-ai/data/raw/*.csv")], ignore_index=True)
print("hourly rows:", len(hourly))

#%% [4] نوافذ + تقسيم زمني + تعيير (fit على التدريب فقط)
from emitax.models.hourly_windows import unit_feature_matrix, make_windows
d, feats = unit_feature_matrix(hourly, cfg)
W = make_windows(d, feats, seq_len=cfg["lstm"]["seq_len"], stride=cfg["lstm"]["stride"])
k = int(len(W) * 0.8); Xtr, Xva = W[:k], W[k:]
mean = Xtr.reshape(-1, len(feats)).mean(0); std = Xtr.reshape(-1, len(feats)).std(0) + 1e-8
Xtr = (Xtr - mean) / std; Xva = (Xva - mean) / std
print("train/val windows:", Xtr.shape, Xva.shape)

#%% [5] بناء + تدريب
from emitax.models.lstm_ae import build_model, train, reconstruction_error, export_onnx
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(len(feats), cfg["lstm"]["seq_len"], cfg["lstm"]["latent"], cfg["lstm"]["hidden"])
hist = train(model, Xtr, epochs=cfg["lstm"]["epochs"], lr=1e-3, batch=128, device=dev)
print("son epoch loss:", round(hist[-1], 5))

#%% [6] العتبة من خطأ التدريب الطبيعي (مئين)
err_tr = reconstruction_error(model, Xtr, device=dev)
threshold = float(np.percentile(err_tr, cfg["lstm"]["threshold_pct"]))
err_va = reconstruction_error(model, Xva, device=dev)
print(f"threshold(p{cfg['lstm']['threshold_pct']})={threshold:.4f} | val>thr={(err_va>threshold).mean():.3f}")

#%% [7] حفظ: ONNX + meta (mean/std/threshold/seq_len/feats)
import json, os
os.makedirs("emitax-ai/data/processed", exist_ok=True)
export_onnx(model.cpu(), cfg["lstm"]["seq_len"], len(feats), "emitax-ai/data/processed/lstm_ae.onnx", device="cpu")
json.dump({"mean": mean.tolist(), "std": std.tolist(), "threshold": threshold,
           "seq_len": cfg["lstm"]["seq_len"], "feats": feats},
          open("emitax-ai/data/processed/lstm_meta.json", "w"))
print("saved: lstm_ae.onnx + lstm_meta.json  (انسخيهما إلى data/processed/ محلياً)")
