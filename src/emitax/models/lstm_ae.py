from __future__ import annotations
import numpy as np

# =========================================================
#  LSTM-Autoencoder (PyTorch) — الفحص الزمني بالتعلّم العميق (تقرير 5.1)
#  يتدرّب على النوافذ الساعية الطبيعية لكل وحدة؛ خطأ إعادة البناء العالي = شذوذ زمني.
#  نفس فلسفة محرّك STN، منقولة إلى PyTorch + قابلة للتصدير ONNX.
#  torch يُستورَد داخل الدوال (ثقيل) — يُشغَّل على Colab (GPU).
# =========================================================

def build_model(n_features: int, seq_len: int, latent: int = 64, hidden: int = 128):
    import torch
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self, F, T, latent, hidden):
            super().__init__()
            self.T = T
            self.enc1 = nn.LSTM(F, hidden, batch_first=True)
            self.enc2 = nn.LSTM(hidden, latent, batch_first=True)
            self.dec1 = nn.LSTM(latent, latent, batch_first=True)
            self.dec2 = nn.LSTM(latent, hidden, batch_first=True)
            self.out = nn.Linear(hidden, F)

        def forward(self, x):                      # x: (B, T, F)
            h, _ = self.enc1(x)
            _, (hn, _) = self.enc2(h)              # hn: (1, B, latent)
            z = hn[-1]                             # (B, latent) = التمثيل الكامن
            z = z.unsqueeze(1).repeat(1, self.T, 1)  # RepeatVector
            d, _ = self.dec1(z)
            d, _ = self.dec2(d)
            return self.out(d)                     # (B, T, F)

    return LSTMAutoencoder(n_features, seq_len, latent, hidden)


def train(model, X, epochs: int = 30, lr: float = 1e-3, batch: int = 64, device: str = "cpu"):
    """يدرّب النموذج على إعادة بناء X (نوافذ طبيعية). يُرجع سجلّ الخسارة."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        history.append(tot / len(ds))
    return history


def reconstruction_error(model, X, device: str = "cpu") -> np.ndarray:
    """خطأ إعادة البناء لكل نافذة (متوسّط MSE عبر الزمن والميزات)."""
    import torch
    model = model.to(device).eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        recon = model(xb)
        err = ((recon - xb) ** 2).mean(dim=(1, 2))
    return err.cpu().numpy()


def export_onnx(model, seq_len: int, n_features: int, path: str, device: str = "cpu"):
    """يصدّر النموذج إلى ONNX (تشغيل خفيف داخل التطبيق — تقرير 5.1)."""
    import torch
    model = model.to(device).eval()
    dummy = torch.randn(1, seq_len, n_features, device=device)
    torch.onnx.export(model, dummy, path,
                      input_names=["window"], output_names=["reconstruction"],
                      dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
                      opset_version=18)
    return path
